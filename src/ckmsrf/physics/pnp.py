r"""Plug-and-play ADMM data updates for complex MIMO pilot observations.

The measurement model is ``Y = H @ P + N``.  With a scaled ADMM dual
variable ``U``, the data-consistency update is

.. math::

    Z^{k+1} = \arg\min_Z
        \frac{\|Y-ZP\|_F^2}{2\sigma_n^2}
        + \frac{\rho_k}{2}\|Z-(X^k-U^k)\|_F^2.

``pnp_data_step`` evaluates this update through the existing RC proximal
solver.  ``adaptive_pnp_data_step`` optionally chooses ``rho`` independently
for every batch item by matching the residual energy to the known noise
variance while penalising lag-one residual correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .rc import rc_proximal_update


def _batch_positive(
    value: torch.Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Return a positive batch parameter with shape ``[B,1,1]``."""

    result = torch.as_tensor(value, device=device, dtype=dtype)
    if result.numel() == 1:
        result = result.reshape(()).expand(batch_size)
    else:
        try:
            result = result.reshape(batch_size, -1)[:, 0]
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must be scalar or contain one value per batch item"
            ) from exc
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return result.view(batch_size, 1, 1)


def pnp_data_step(
    prior_mean: torch.Tensor,
    cache: dict[str, torch.Tensor],
    rho: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate the exact PnP-ADMM ``z`` update.

    Args:
        prior_mean: Complex ``X-U`` tensor with shape ``[B,Nr,Nt]``.
        cache: Result of :func:`ckmsrf.physics.rc.build_physics_cache`.
        rho: Positive scalar or one positive value per batch item.

    Returns:
        Complex data-consistent ``Z`` with the same shape as ``prior_mean``.
    """

    if prior_mean.ndim == 2:
        prior_mean = prior_mean.unsqueeze(0)
    if prior_mean.ndim != 3 or not prior_mean.is_complex():
        raise ValueError("prior_mean must be complex [B,Nr,Nt]")
    rho_batched = _batch_positive(
        rho,
        prior_mean.shape[0],
        device=prior_mean.device,
        dtype=prior_mean.real.dtype,
        name="rho",
    )
    # rc_proximal_update uses precision M + I / prior_var, hence
    # prior_var=1/rho is exactly the scaled-ADMM quadratic penalty.
    return rc_proximal_update(prior_mean, cache, prior_var=rho_batched.reciprocal())


# A name that mirrors the variable used in the ADMM equations.
pnp_z_step = pnp_data_step


def residual_statistics(
    channel: torch.Tensor,
    cache: dict[str, torch.Tensor],
    *,
    whiteness_weight: float = 0.25,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute batchwise measurement-residual energy and whiteness scores.

    Whiteness is measured as the mean magnitude of the normalised lag-one
    correlation along the pilot dimension.  The total score is

    ``abs(log(residual_energy / noise_variance)) + weight * whiteness``.

    A smaller score therefore favours residuals that look like the specified
    observation noise instead of merely forcing their norm to zero.
    """

    if whiteness_weight < 0:
        raise ValueError("whiteness_weight must be non-negative")
    if channel.ndim != 3 or not channel.is_complex():
        raise ValueError("channel must be complex [B,Nr,Nt]")
    residual = torch.bmm(channel, cache["pilot"]) - cache["observation"]
    energy = residual.abs().square().mean(dim=(-2, -1))
    target = cache["sigma2"].reshape(channel.shape[0], -1)[:, 0]
    energy_mismatch = torch.abs(torch.log((energy + eps) / (target + eps)))

    if residual.shape[-1] < 2:
        whiteness = torch.zeros_like(energy)
    else:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        left = centered[..., :-1]
        right = centered[..., 1:]
        numerator = (left.conj() * right).sum(dim=-1).abs()
        denominator = torch.sqrt(
            left.abs().square().sum(dim=-1) * right.abs().square().sum(dim=-1)
        )
        # Average receiver-wise correlations so that unrelated rows do not
        # cancel one another in the complex sum.
        whiteness = (numerator / denominator.clamp_min(eps)).mean(dim=-1)

    score = energy_mismatch + float(whiteness_weight) * whiteness
    return {
        "residual_energy": energy,
        "noise_energy": target,
        "energy_mismatch": energy_mismatch,
        "residual_whiteness": whiteness,
        "rho_score": score,
    }


def adaptive_pnp_data_step(
    x: torch.Tensor,
    scaled_dual: torch.Tensor,
    cache: dict[str, torch.Tensor],
    rho: torch.Tensor | float,
    *,
    candidate_factors: Sequence[float] = (0.5, 1.0, 2.0),
    whiteness_weight: float = 0.25,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Select a batchwise ``rho`` and perform the corresponding exact z-step.

    The scaled dual represents the unscaled multiplier divided by ``rho``.
    Every candidate therefore uses ``U_candidate = U * rho / rho_candidate``;
    this rescaling is required to preserve the same unscaled multiplier when
    the penalty changes.

    Returns:
        ``(z, selected_rho, rescaled_dual, diagnostics)`` where ``selected_rho``
        has shape ``[B,1,1]``.
    """

    if x.ndim != 3 or not x.is_complex():
        raise ValueError("x must be complex [B,Nr,Nt]")
    if scaled_dual.shape != x.shape or not scaled_dual.is_complex():
        raise ValueError("scaled_dual must be a complex tensor matching x")
    if not candidate_factors:
        raise ValueError("candidate_factors must not be empty")
    if rho_min <= 0 or rho_max < rho_min:
        raise ValueError("rho bounds must satisfy 0 < rho_min <= rho_max")

    factors = tuple(float(factor) for factor in candidate_factors)
    if any(not torch.isfinite(torch.tensor(factor)) or factor <= 0 for factor in factors):
        raise ValueError("candidate_factors must contain finite positive values")
    current_rho = _batch_positive(
        rho,
        x.shape[0],
        device=x.device,
        dtype=x.real.dtype,
        name="rho",
    )

    z_candidates = []
    dual_candidates = []
    rho_candidates = []
    statistic_candidates: dict[str, list[torch.Tensor]] = {}
    for factor in factors:
        candidate_rho = (current_rho * factor).clamp(min=rho_min, max=rho_max)
        candidate_dual = scaled_dual * (current_rho / candidate_rho)
        candidate_z = pnp_data_step(x - candidate_dual, cache, candidate_rho)
        statistics = residual_statistics(
            candidate_z,
            cache,
            whiteness_weight=whiteness_weight,
        )
        z_candidates.append(candidate_z)
        dual_candidates.append(candidate_dual)
        rho_candidates.append(candidate_rho.reshape(x.shape[0]))
        for key, value in statistics.items():
            statistic_candidates.setdefault(key, []).append(value)

    z_stack = torch.stack(z_candidates, dim=0)
    dual_stack = torch.stack(dual_candidates, dim=0)
    rho_stack = torch.stack(rho_candidates, dim=0)
    score_stack = torch.stack(statistic_candidates["rho_score"], dim=0)
    choice = score_stack.argmin(dim=0)
    batch_index = torch.arange(x.shape[0], device=x.device)

    selected_z = z_stack[choice, batch_index]
    selected_dual = dual_stack[choice, batch_index]
    selected_rho = rho_stack[choice, batch_index].view(x.shape[0], 1, 1)
    diagnostics: dict[str, torch.Tensor] = {
        "rho_choice": choice,
        "rho": selected_rho.reshape(x.shape[0]),
    }
    for key, values in statistic_candidates.items():
        stacked = torch.stack(values, dim=0)
        diagnostics[key] = stacked[choice, batch_index]
    return selected_z, selected_rho, selected_dual, diagnostics


__all__ = [
    "adaptive_pnp_data_step",
    "pnp_data_step",
    "pnp_z_step",
    "residual_statistics",
]
