"""Split-Gibbs PnP-DM sampling for complex MIMO pilot observations.

This module adapts Wu et al., *Principled Probabilistic Imaging using
Diffusion Models as Plug-and-Play Priors* (NeurIPS 2024), to the channel
observation model ``Y = H @ P + N``.  It is intentionally separate from the
deterministic PnP-ADMM implementation in :mod:`ckmsrf.sampling_cm`.

The learned prior acts on the real/imaginary two-channel representation.  A
PnP-DM coupling scale ``rho`` therefore denotes the standard deviation of
each real component.  In complex notation the corresponding circular complex
variance is ``2 * rho**2``; keeping this factor explicit avoids silently
mixing PnP-DM's coupling scale with the complex measurement-noise variance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import torch
from torch import nn

from .losses.gmflow import extract_condition
from .physics.rc import build_physics_cache
from .sampling_cm import (
    _as_complex,
    _batch_positive,
    _call_prior_denoiser,
    _first,
    _model_device,
    _model_sigma_min,
    _output_parameterization,
    _residual_norm,
    _synchronize_if_cuda,
)


def _component_gaussian(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return complex noise with unit variance in each real component."""

    kwargs = {
        "device": device,
        "dtype": dtype,
        "generator": generator,
    }
    return torch.complex(torch.randn(shape, **kwargs), torch.randn(shape, **kwargs))


def _circular_complex_standard_normal(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return circular complex ``CN(0, I)`` noise."""

    return _component_gaussian(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    ) / math.sqrt(2.0)


def sample_linear_gaussian_likelihood(
    current: torch.Tensor,
    cache: dict[str, torch.Tensor],
    coupling_std: torch.Tensor | float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Draw the exact linear-Gaussian PnP-DM likelihood conditional.

    With real-component coupling standard deviation ``rho``, this samples

    .. math::

        Z\mid H,Y \sim \mathcal{CN}(m, Q^{-1}),\qquad
        Q=P P^H/\sigma_n^2 + I/(2\rho^2).

    The posterior mean and covariance draw use one Cholesky factor.  The
    normal equations are algebraically positive definite, but extremely
    small measurement variances can make a rank-deficient pilot Gram matrix
    numerically indefinite in ``complex64``.  The common path remains in the
    model dtype; only that failure path rebuilds the normal equations in
    ``complex128`` from the original pilot/observation tensors.
    """

    if current.ndim == 2:
        current = current.unsqueeze(0)
    if current.ndim != 3 or not current.is_complex():
        raise ValueError("current must be complex [B,Nr,Nt]")

    batch_size = current.shape[0]
    rho = _batch_positive(
        coupling_std,
        batch_size,
        device=current.device,
        dtype=current.real.dtype,
        name="coupling_std",
    )
    complex_variance = 2.0 * rho.square()
    precision = cache["M"] + cache["I_Nt"] / complex_variance
    rhs = cache["R"] + current / complex_variance
    # Numerical roundoff can make an analytically Hermitian matrix differ by
    # a few ulps from its conjugate transpose, which Cholesky need not accept.
    precision = 0.5 * (precision + precision.mH)
    try:
        precision_cholesky = torch.linalg.cholesky(precision)
        mean = torch.cholesky_solve(rhs.mH, precision_cholesky).mH
    except torch.linalg.LinAlgError as float32_error:
        # Do not up-cast cache["M"] here: its low-eigenvalue information may
        # already have been lost while forming P P^H / sigma^2 in complex64.
        # Rebuild both the Gram matrix and the right-hand side in complex128.
        work_complex_dtype = torch.complex128
        work_real_dtype = torch.float64
        pilot = cache["pilot"].to(dtype=work_complex_dtype)
        observation = cache["observation"].to(dtype=work_complex_dtype)
        sigma2 = cache["sigma2"].to(dtype=work_real_dtype)
        coupling_variance = complex_variance.to(dtype=work_real_dtype)
        identity = torch.eye(
            pilot.shape[-2], device=current.device, dtype=work_complex_dtype
        ).expand(batch_size, -1, -1)
        gram = torch.bmm(pilot, pilot.mH) / sigma2
        precision = 0.5 * (gram + gram.mH) + identity / coupling_variance
        rhs = (
            torch.bmm(observation, pilot.mH) / sigma2
            + current.to(dtype=work_complex_dtype) / coupling_variance
        )
        try:
            precision_cholesky = torch.linalg.cholesky(precision)
            mean = torch.cholesky_solve(rhs.mH, precision_cholesky).mH
        except torch.linalg.LinAlgError as double_error:
            finite_inputs = all(
                bool(torch.isfinite(value).all())
                for value in (pilot, observation, sigma2, coupling_variance)
            )
            detail = (
                "all inputs were finite" if finite_inputs else "non-finite inputs were detected"
            )
            raise FloatingPointError(
                "PnP-DM likelihood precision remained non-positive-definite "
                f"after a complex128 rebuild ({detail}). Check the pilot, "
                "observation, and noise variance for this batch."
            ) from double_error

    white = _circular_complex_standard_normal(
        current.shape,
        device=current.device,
        dtype=precision_cholesky.real.dtype,
        generator=generator,
    )
    # Treat every receive-antenna row as a column while solving, then convert
    # it back with the Hermitian transpose.  Plain transpose would conjugate
    # the target covariance for a genuinely complex pilot matrix and sample
    # Q^{-T} instead of Q^{-1}.
    white_columns = white.mH
    noise_columns = torch.linalg.solve_triangular(
        precision_cholesky.mH,
        white_columns,
        upper=True,
    )
    return (mean + noise_columns.mH).to(dtype=current.dtype)


def _edm_sigma_grid(
    *,
    sigma_min: float,
    sigma_max: float,
    steps: int,
    schedule_power: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``steps`` positive descending Karras/EDM noise levels."""

    if steps < 2:
        raise ValueError("steps must be at least 2")
    ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
    inv_power = 1.0 / float(schedule_power)
    maximum = torch.as_tensor(sigma_max, device=device, dtype=dtype).pow(inv_power)
    minimum = torch.as_tensor(sigma_min, device=device, dtype=dtype).pow(inv_power)
    grid = (maximum + ramp * (minimum - maximum)).pow(schedule_power)
    grid[0] = float(sigma_max)
    grid[-1] = float(sigma_min)
    return grid


def _reverse_levels(grid: torch.Tensor, coupling_std: float) -> torch.Tensor:
    """Select the first grid level at or below ``coupling_std`` and append zero."""

    start = torch.as_tensor(coupling_std, device=grid.device, dtype=grid.dtype)
    tolerance = torch.finfo(grid.dtype).eps * max(1.0, abs(float(coupling_std))) * 8.0
    tail = grid[grid <= start + tolerance]
    if tail.numel() == 0:
        raise ValueError("coupling_std is below the minimum diffusion grid level")
    return torch.cat((tail, torch.zeros_like(tail[:1])))


def reverse_diffusion_prior(
    model: nn.Module,
    noisy_channel: torch.Tensor,
    coupling_std: float,
    condition: Mapping[str, Any] | None,
    *,
    sigma_min: float,
    sigma_max: float,
    prior_steps: int,
    schedule_power: float = 7.0,
    solver: str = "sde",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Sample the Gaussian-denoising posterior by reverse VE diffusion.

    ``prior_steps`` defines the number of positive levels in the *global* EDM
    grid.  Following Algorithm 3 of PnP-DM, a prior step starts from the first
    grid level at or below the current coupling scale and finishes at the clean
    boundary zero.  Later outer iterations therefore generally use fewer NFEs.
    """

    if prior_steps < 2:
        raise ValueError("prior_steps must be at least 2")
    if schedule_power <= 0.0:
        raise ValueError("schedule_power must be positive")
    solver = str(solver).lower().replace("-", "_")
    if solver == "ode":
        solver = "pf_ode"
    if solver not in {"sde", "pf_ode"}:
        raise ValueError("solver must be sde or pf_ode")
    if _output_parameterization(model) != "clean":
        raise ValueError("PnP-DM requires a direct clean-signal diffusion denoiser")
    if not sigma_min <= coupling_std <= sigma_max:
        raise ValueError("coupling_std must lie in [sigma_min, sigma_max]")

    grid = _edm_sigma_grid(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        steps=prior_steps,
        schedule_power=schedule_power,
        device=noisy_channel.device,
        dtype=noisy_channel.real.dtype,
    )
    levels = _reverse_levels(grid, coupling_std)
    state = noisy_channel
    batch_size = state.shape[0]

    for index in range(max(0, levels.numel() - 1)):
        sigma = levels[index]
        sigma_next = levels[index + 1]
        sigma_batch = sigma.expand(batch_size)
        denoised = _call_prior_denoiser(model, state, sigma_batch, condition)
        multiplier = 2.0 if solver == "sde" else 1.0
        derivative = multiplier * (state - denoised) / sigma
        state = state + (sigma_next - sigma) * derivative

        # Algorithm 3 of Wu et al. omits stochastic diffusion only on the
        # final positive-noise -> clean-boundary transition.
        if solver == "sde" and index != levels.numel() - 2:
            diffusion_std = torch.sqrt(
                (2.0 * sigma * (sigma - sigma_next)).clamp_min(0.0)
            )
            state = state + diffusion_std * _component_gaussian(
                state.shape,
                device=state.device,
                dtype=state.real.dtype,
                generator=generator,
            )

    return state, max(0, levels.numel() - 1), levels.detach()


def _selected_estimate(
    state: torch.Tensor,
    posterior_sum: torch.Tensor | None,
    posterior_count: int,
    estimate_mode: str,
) -> torch.Tensor:
    if estimate_mode == "posterior_mean" and posterior_sum is not None and posterior_count > 0:
        return posterior_sum / float(posterior_count)
    return state


@torch.no_grad()
def sample_ckm_diffusion_pnpdm(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    iterations: int = 100,
    rho_start: float = 1.0,
    rho_min: float = 0.05,
    rho_decay: float = 0.9,
    sigma_min: float = 2.0e-3,
    sigma_max: float = 2.0,
    prior_steps: int = 100,
    schedule_power: float = 7.0,
    prior_solver: str = "sde",
    initialization: str = "ls",
    estimate_mode: str = "posterior_mean",
    burn_in: int = 40,
    thin: int = 3,
    generator: torch.Generator | None = None,
    snapshot_iterations: Sequence[int] = (),
    record_snapshot_timing: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run CKM-conditional PnP-DM for ``Y = H @ P + N``.

    The outer loop is a Split Gibbs chain.  Each iteration draws the exact
    linear-Gaussian likelihood conditional and then runs a partial reverse
    diffusion from the first grid level at or below ``rho_k`` to zero.
    ``posterior_mean`` returns the
    running mean of thinned post-burn-in states; snapshots before the first
    retained state transparently fall back to the current chain state and
    report a sample count of zero.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("Require 0 < sigma_min < sigma_max")
    if not 0.0 < rho_min <= rho_start:
        raise ValueError("Require 0 < rho_min <= rho_start")
    if not 0.0 < rho_decay <= 1.0:
        raise ValueError("rho_decay must lie in (0,1]")
    if prior_steps < 2:
        raise ValueError("prior_steps must be at least 2")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if thin < 1:
        raise ValueError("thin must be positive")
    estimate_mode = str(estimate_mode).lower().replace("-", "_")
    if estimate_mode == "mean":
        estimate_mode = "posterior_mean"
    if estimate_mode not in {"last", "posterior_mean"}:
        raise ValueError("estimate_mode must be last or posterior_mean")
    if estimate_mode == "posterior_mean" and burn_in + thin > iterations:
        raise ValueError(
            "posterior_mean requires at least one retained state: "
            "burn_in + thin <= iterations"
        )
    retained_capacity = max(0, (iterations - burn_in) // thin)
    first_retained_iteration = burn_in + thin
    if (
        estimate_mode == "posterior_mean"
        and retained_capacity > 1
        and rho_decay < 1.0
        and rho_start * (rho_decay ** (first_retained_iteration - 1))
        > rho_min * (1.0 + 1.0e-12)
    ):
        raise ValueError(
            "posterior_mean would mix states from different rho values; "
            "increase burn_in until the first retained state reaches rho_min"
        )
    initialization = str(initialization).lower()
    if initialization not in {"ls", "zero"}:
        raise ValueError("initialization must be ls or zero")

    requested_snapshots = tuple(sorted({int(value) for value in snapshot_iterations}))
    if any(value < 1 or value > iterations for value in requested_snapshots):
        raise ValueError("snapshot_iterations must lie in [1, iterations]")
    if record_snapshot_timing and not requested_snapshots:
        raise ValueError("record_snapshot_timing requires snapshot_iterations")

    device = _model_device(model, batch)
    effective_sigma_min = max(float(sigma_min), _model_sigma_min(model, sigma_min))
    if rho_min < effective_sigma_min or rho_start > sigma_max:
        raise ValueError(
            "PnP-DM coupling schedule must stay inside the denoiser range: "
            f"[{effective_sigma_min}, {sigma_max}]"
        )
    if _output_parameterization(model) != "clean":
        raise ValueError("PnP-DM requires a diffusion checkpoint with clean output")

    pilot_value = _first(batch, "P", "pilot", "p")
    observation_value = _first(batch, "Y", "observation", "y")
    if pilot_value is None or observation_value is None:
        raise KeyError("PnP-DM sampling requires P/pilot and Y/observation")
    pilot = _as_complex(pilot_value, device=device)
    observation = _as_complex(observation_value, device=device)
    if pilot.ndim == 2:
        pilot = pilot.unsqueeze(0)
    if observation.ndim == 2:
        observation = observation.unsqueeze(0)
    if pilot.ndim != 3 or observation.ndim != 3:
        raise ValueError("P and Y must be batched matrices")
    if pilot.shape[0] == 1 and observation.shape[0] > 1:
        pilot = pilot.expand(observation.shape[0], -1, -1)
    if pilot.shape[0] != observation.shape[0]:
        raise ValueError("P and Y batch sizes are inconsistent")
    if observation.shape[-1] != pilot.shape[-1]:
        raise ValueError("Y and P pilot dimensions are inconsistent")

    noise_var = _first(batch, "noise_var", "observation_noise_var")
    if noise_var is None:
        raise KeyError(
            "Principled PnP-DM likelihood sampling requires the complex "
            "measurement-noise variance"
        )
    if torch.is_tensor(noise_var):
        noise_var = noise_var.to(device=device)
    cache = build_physics_cache(observation, pilot, noise_var=noise_var)
    batch_size = observation.shape[0]
    expected_shape = (batch_size, observation.shape[-2], pilot.shape[-2])

    initial_value = _first(
        batch,
        "initial",
        "H_init",
        "initial_estimate",
        "ls_estimate",
        "h_init",
    )
    if initial_value is not None:
        state = _as_complex(initial_value, device=device)
        if state.ndim == 2:
            state = state.unsqueeze(0)
    elif initialization == "ls":
        state = torch.bmm(observation, cache["p_pinv"])
    else:
        state = torch.zeros(expected_shape, device=device, dtype=observation.dtype)
    if state.shape != expected_shape:
        raise ValueError(f"Initial channel must have shape {expected_shape}, got {tuple(state.shape)}")

    condition = extract_condition(batch)
    initial_norm = _residual_norm(state, cache)
    posterior_sum: torch.Tensor | None = None
    posterior_count = 0
    cumulative_nfe = 0
    rho_history: list[float] = []
    prior_start_sigma_history: list[torch.Tensor] = []
    prior_nfe_history: list[int] = []
    iteration_estimates: dict[int, torch.Tensor] = {}
    iteration_seconds: dict[int, float] = {}
    iteration_nfe: dict[int, int] = {}
    iteration_sample_count: dict[int, int] = {}
    iteration_estimate_mode: dict[int, str] = {}

    _synchronize_if_cuda(device)
    sampling_start = time.perf_counter()
    for iteration in range(1, iterations + 1):
        coupling_std = max(rho_start * (rho_decay ** (iteration - 1)), rho_min)
        likelihood_sample = sample_linear_gaussian_likelihood(
            state,
            cache,
            coupling_std,
            generator=generator,
        )
        state, current_nfe, current_levels = reverse_diffusion_prior(
            model,
            likelihood_sample,
            coupling_std,
            condition,
            sigma_min=effective_sigma_min,
            sigma_max=sigma_max,
            prior_steps=prior_steps,
            schedule_power=schedule_power,
            solver=prior_solver,
            generator=generator,
        )
        cumulative_nfe += int(current_nfe)
        rho_history.append(float(coupling_std))
        # Keep this on-device during the timed region. Converting a CUDA scalar
        # to Python here would force one synchronization per outer iteration.
        prior_start_sigma_history.append(current_levels[0].detach())
        prior_nfe_history.append(int(current_nfe))

        if iteration > burn_in and (iteration - burn_in) % thin == 0:
            posterior_sum = state.clone() if posterior_sum is None else posterior_sum + state
            posterior_count += 1

        if iteration in requested_snapshots:
            iteration_estimates[iteration] = _selected_estimate(
                state,
                posterior_sum,
                posterior_count,
                estimate_mode,
            )
            iteration_nfe[iteration] = cumulative_nfe
            iteration_sample_count[iteration] = posterior_count
            iteration_estimate_mode[iteration] = (
                "posterior_mean"
                if estimate_mode == "posterior_mean" and posterior_count > 0
                else "last"
            )
            if record_snapshot_timing:
                _synchronize_if_cuda(device)
                iteration_seconds[iteration] = time.perf_counter() - sampling_start

    _synchronize_if_cuda(device)
    sampling_seconds = time.perf_counter() - sampling_start
    estimate = iteration_estimates.get(
        iterations,
        _selected_estimate(state, posterior_sum, posterior_count, estimate_mode),
    )
    final_norm = _residual_norm(estimate, cache)
    diagnostics: dict[str, Any] = {
        "iterations": int(iterations),
        "data_consistency_evaluations": int(iterations),
        "likelihood_random_draws": int(iterations),
        "denoiser_nfe": int(cumulative_nfe),
        "denoiser_nfe_per_iteration": None,
        "prior_nfe_history": torch.tensor(prior_nfe_history, dtype=torch.int64),
        "rho_history": torch.tensor(rho_history, dtype=torch.float64),
        "prior_start_sigma_history": torch.stack(prior_start_sigma_history)
        .to(dtype=torch.float64)
        .cpu(),
        "prior_steps": int(prior_steps),
        "prior_solver": str(prior_solver).lower().replace("-", "_"),
        "schedule_power": float(schedule_power),
        "rho_start": float(rho_start),
        "rho_min": float(rho_min),
        "rho_decay": float(rho_decay),
        "initialization": initialization,
        "estimate_mode": estimate_mode,
        "burn_in": int(burn_in),
        "thin": int(thin),
        "first_retained_iteration": int(first_retained_iteration),
        "posterior_sample_count": int(posterior_count),
        "sampling_seconds": float(sampling_seconds),
        "measurement_residual_initial": float(initial_norm.mean().cpu()),
        "measurement_residual": float(final_norm.mean().cpu()),
        "iteration_estimates": iteration_estimates,
        "iteration_seconds": iteration_seconds,
        "iteration_denoiser_nfe": iteration_nfe,
        "iteration_sample_count": iteration_sample_count,
        "iteration_estimate_mode": iteration_estimate_mode,
    }
    return estimate, diagnostics


__all__ = [
    "reverse_diffusion_prior",
    "sample_ckm_diffusion_pnpdm",
    "sample_linear_gaussian_likelihood",
]
