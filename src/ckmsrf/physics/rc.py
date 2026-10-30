"""Closed-form rectified-channel (RC) proximal update."""

from __future__ import annotations

from typing import Any

import torch

from .observation import _ensure_batched_pilot


def _batch_variance(value: Any, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() == 1:
        value = value.reshape(()).expand(batch)
    return value.reshape(batch, -1)[:, :1].view(batch, 1, 1)


def build_physics_cache(
    observation: torch.Tensor,
    pilot: torch.Tensor,
    noise_scale: torch.Tensor | float | None = None,
    *,
    noise_var: torch.Tensor | float | None = None,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Precompute RC, projection and guidance terms for one observation batch."""
    if observation.ndim == 2:
        observation = observation.unsqueeze(0)
    if observation.ndim != 3 or not observation.is_complex():
        raise ValueError("observation must be complex [B,Nr,Np]")
    batch = observation.shape[0]
    pilot = _ensure_batched_pilot(
        pilot.to(device=observation.device, dtype=observation.dtype), batch
    )
    real_dtype = observation.real.dtype
    if noise_var is None:
        noise_scale = 1.0 if noise_scale is None else noise_scale
        sigma2 = _batch_variance(noise_scale, batch, observation.device, real_dtype).square()
    else:
        sigma2 = _batch_variance(noise_var, batch, observation.device, real_dtype)
    sigma2 = sigma2.clamp_min(eps)
    p_h = pilot.mH
    p_p_h = torch.bmm(pilot, p_h)
    y_p_h = torch.bmm(observation, p_h)
    p_pinv = torch.linalg.pinv(pilot)
    projection = torch.bmm(pilot, p_pinv)
    nt, npilots = pilot.shape[-2:]
    identity = torch.eye(nt, device=pilot.device, dtype=pilot.dtype).expand(batch, nt, nt)
    lipschitz = torch.linalg.eigvalsh(p_p_h).real.amax(dim=-1).clamp_min(eps).view(batch, 1, 1)
    return {
        "sigma2": sigma2,
        "p_h": p_h,
        "p_p_h": p_p_h,
        "p_pinv": p_pinv,
        "projection": projection,
        "proj": projection,
        "fig_lipschitz": lipschitz,
        "pilot_ratio": torch.full(
            (batch, 1, 1), npilots / nt, device=pilot.device, dtype=real_dtype
        ),
        "y_p_h": y_p_h,
        "identity": identity,
        "i_nt": identity,
        "I_Nt": identity,
        "M": p_p_h / sigma2,
        "R": y_p_h / sigma2,
        "observation": observation,
        "y_cplx": observation,
        "pilot": pilot,
        "p_cplx": pilot,
    }


build_rcflow_physics_cache = build_physics_cache


def rc_prior_variance_from_time(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Original SRF/RC bounded variance schedule."""
    t = t.reshape(-1)
    t2 = t.clamp_min(eps).square()
    one_minus_t2 = (1.0 - t).clamp_min(eps).square()
    return (t2 / (t2 + one_minus_t2)).view(-1, 1, 1)


def rc_proximal_update(
    prior_mean: torch.Tensor,
    cache: dict[str, torch.Tensor],
    prior_var: torch.Tensor | float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Solve the complex Gaussian posterior mean in closed form."""
    if prior_mean.ndim == 2:
        prior_mean = prior_mean.unsqueeze(0)
    prior_var = _batch_variance(
        prior_var,
        prior_mean.shape[0],
        prior_mean.device,
        prior_mean.real.dtype,
    ).clamp_min(eps)
    precision = cache["M"] + cache["I_Nt"] / prior_var
    rhs = cache["R"] + prior_mean / prior_var
    return torch.linalg.solve(precision.mH, rhs.mH).mH


original_rc_prox_update = rc_proximal_update
