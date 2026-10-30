"""Pilot-based/FIG data-consistency guidance for SRF sampling."""

from __future__ import annotations

import torch


def apply_measurement(channel: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.bmm(channel, cache["pilot"])


def apply_range_projection(channel: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.bmm(channel, cache["projection"])


def apply_null_projection(channel: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    return channel - apply_range_projection(channel, cache)


def measurement_interpolant(
    observation: torch.Tensor,
    noise_anchor: torch.Tensor,
    cache: dict[str, torch.Tensor],
    sigma_t: torch.Tensor,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    sigma_t = sigma_t.reshape(-1, 1, 1)
    return (1.0 - sigma_t) * observation + weight * sigma_t * apply_measurement(noise_anchor, cache)


fig_measurement_interpolant = measurement_interpolant


def data_consistency_gradient(
    channel: torch.Tensor,
    target_observation: torch.Tensor,
    cache: dict[str, torch.Tensor],
    *,
    normalize_by_operator: bool = True,
) -> torch.Tensor:
    residual = apply_measurement(channel, cache) - target_observation
    gradient = torch.bmm(residual, cache["p_h"])
    if normalize_by_operator:
        gradient = gradient / cache["fig_lipschitz"]
    return gradient


fig_data_consistency_gradient = data_consistency_gradient


def pilot_guided_update(
    channel: torch.Tensor,
    target_observation: torch.Tensor,
    cache: dict[str, torch.Tensor],
    *,
    step_size: float | torch.Tensor = 0.15,
    steps: int = 1,
    noise_gating: float = 1.0,
    max_step: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run normalized PBG gradient steps and return diagnostics."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    step = torch.as_tensor(step_size, device=channel.device, dtype=channel.real.dtype)
    step = step.reshape(-1, 1, 1) if step.numel() > 1 else step
    gate = 1.0 / (1.0 + noise_gating * cache["sigma2"])
    effective_step = (step * gate * cache["pilot_ratio"].clamp(max=1.0)).clamp_max(max_step)
    grad_norms = []
    updated = channel
    for _ in range(steps):
        gradient = data_consistency_gradient(updated, target_observation, cache)
        updated = updated - effective_step * gradient
        grad_norms.append(torch.linalg.vector_norm(gradient.flatten(1), dim=1).mean())
    residual = apply_measurement(updated, cache) - target_observation
    zero = torch.zeros((), device=channel.device, dtype=channel.real.dtype)
    return updated, {
        "pbg_residual": torch.linalg.vector_norm(residual.flatten(1), dim=1).mean(),
        "pbg_gradient_norm": torch.stack(grad_norms).mean() if grad_norms else zero,
        "pbg_step": effective_step.mean(),
    }


pbg_update = pilot_guided_update


def fig_plus_projection(
    guided_channel: torch.Tensor,
    prior_mean: torch.Tensor,
    noise_anchor: torch.Tensor,
    cache: dict[str, torch.Tensor],
    sigma_t: torch.Tensor,
    *,
    mix: float = 0.2,
) -> torch.Tensor:
    """Blend observable PBG content with the prior's null-space content."""
    sigma_t = sigma_t.reshape(-1, 1, 1)
    interpolated_prior = (1.0 - sigma_t) * prior_mean + sigma_t * noise_anchor
    projected = apply_null_projection(interpolated_prior, cache) + apply_range_projection(
        guided_channel, cache
    )
    return (1.0 - mix) * guided_channel + mix * projected

