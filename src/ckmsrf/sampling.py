"""Conditional SRF sampling with optional RC/PBG measurement correction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .losses import gm_to_mean, u_to_x0_gm
from .losses.gmflow import extract_condition
from .physics import (
    build_physics_cache,
    complex_to_twochannel,
    fig_plus_projection,
    pilot_guided_update,
    rc_prior_variance_from_time,
    rc_proximal_update,
    twochannel_to_complex,
)


def _first(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = batch.get(name)
        if value is not None:
            return value
    return None


def _as_complex(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    value = value.to(device)
    if value.is_complex():
        return value
    return twochannel_to_complex(value.float())


def _randn_complex(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    kwargs = {"device": device, "dtype": real_dtype, "generator": generator}
    return torch.complex(torch.randn(shape, **kwargs), torch.randn(shape, **kwargs)) / 2.0**0.5


@torch.no_grad()
def sample_ckmsrf(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    steps: int = 20,
    outer_loops: int = 1,
    lambda_power: float = 1.0,
    beta_power: float = 1.0,
    use_rc: bool = True,
    pbg_steps: int = 0,
    pbg_step_size: float = 0.15,
    fig_plus_mix: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate a complex channel from CKM conditions and pilot observations.

    The sampler retains the original additive SRF path.  A conditional GM
    prior predicts ``H_0`` at every time point; when ``P`` and ``Y`` are
    present, the closed-form RC posterior is applied before the anchored
    rectification step.  PBG/FIG+ are optional refinements rather than hidden
    changes to the learned prior.
    """

    if steps < 1 or outer_loops < 1:
        raise ValueError("steps and outer_loops must both be positive")
    if lambda_power <= 0.0 or beta_power <= 0.0:
        raise ValueError("lambda_power and beta_power must be positive")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    pilot_value = _first(batch, "P", "pilot", "p")
    observation_value = _first(batch, "Y", "observation", "y")
    truth_value = _first(batch, "H", "h_complex", "h")

    pilot = None if pilot_value is None else _as_complex(pilot_value, device=device)
    observation = (
        None if observation_value is None else _as_complex(observation_value, device=device)
    )
    if observation is not None and observation.ndim == 2:
        observation = observation.unsqueeze(0)
    if pilot is not None and pilot.ndim == 2:
        pilot = pilot.unsqueeze(0)

    if observation is not None and pilot is not None:
        batch_size, nr = observation.shape[:2]
        nt = pilot.shape[-2]
        complex_dtype = observation.dtype
    elif truth_value is not None:
        truth = _as_complex(truth_value, device=device)
        if truth.ndim == 2:
            truth = truth.unsqueeze(0)
        batch_size, nr, nt = truth.shape
        complex_dtype = truth.dtype
    else:
        raise KeyError("Sampling requires either H or both P and Y to infer dimensions")

    condition = extract_condition(batch)
    noise_var = _first(batch, "noise_var", "observation_noise_var")
    cache = None
    if use_rc:
        if observation is None or pilot is None:
            raise KeyError("use_rc=True requires complex P and Y in the batch")
        cache = build_physics_cache(
            observation,
            pilot,
            noise_var=None if noise_var is None else noise_var.to(device),
        )

    h_t = _randn_complex(
        (batch_size, nr, nt),
        device=device,
        dtype=complex_dtype,
        generator=generator,
    )
    pbg_residuals: list[torch.Tensor] = []

    for _ in range(outer_loops):
        noise_anchor = h_t.detach().clone()
        for index in range(steps):
            fraction = index / steps
            next_fraction = (index + 1) / steps
            t_current_value = max((1.0 - fraction) ** lambda_power, 1.0e-5)
            t_next_value = max((1.0 - next_fraction) ** beta_power, 0.0)
            t_current = torch.full(
                (batch_size,),
                t_current_value,
                device=device,
                dtype=h_t.real.dtype,
            )
            h_t_real = complex_to_twochannel(h_t)
            gm_u = model(h_t_real, t_current, condition=condition)
            gm_x0 = u_to_x0_gm(gm_u, h_t_real, t_current)
            prior_mean = twochannel_to_complex(gm_to_mean(gm_x0))

            posterior_mean = prior_mean
            if cache is not None:
                posterior_mean = rc_proximal_update(
                    prior_mean,
                    cache,
                    rc_prior_variance_from_time(t_current),
                )

            t_next = torch.full(
                (batch_size, 1, 1),
                t_next_value,
                device=device,
                dtype=h_t.real.dtype,
            )
            h_next = t_next * noise_anchor + (1.0 - t_next) * posterior_mean

            if cache is not None and pbg_steps > 0 and t_next_value > 0.0:
                h_guided, diagnostics = pilot_guided_update(
                    h_next,
                    observation,
                    cache,
                    step_size=pbg_step_size,
                    steps=pbg_steps,
                )
                pbg_residuals.append(diagnostics["pbg_residual"])
                if fig_plus_mix > 0.0:
                    h_guided = fig_plus_projection(
                        h_guided,
                        posterior_mean,
                        noise_anchor,
                        cache,
                        t_next.reshape(batch_size),
                        mix=fig_plus_mix,
                    )
                h_next = h_guided
            h_t = h_next

    diagnostics: dict[str, float] = {
        "steps": float(steps),
        "outer_loops": float(outer_loops),
    }
    if observation is not None and pilot is not None:
        residual = torch.bmm(h_t, pilot) - observation
        diagnostics["measurement_residual"] = float(
            torch.linalg.vector_norm(residual.flatten(1), dim=1).mean().cpu()
        )
    if pbg_residuals:
        diagnostics["pbg_residual"] = float(torch.stack(pbg_residuals).mean().cpu())
    return h_t, diagnostics


__all__ = ["sample_ckmsrf"]

