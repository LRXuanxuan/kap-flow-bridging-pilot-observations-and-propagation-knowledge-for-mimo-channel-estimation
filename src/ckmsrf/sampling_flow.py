r"""PnP-Flow and FlowADMM solvers for complex MIMO pilot recovery.

Both solvers require a straight latent-to-data Flow Matching prior trained on
``x_t=(1-t)x_latent+t*x_data``.  The older additive/VE velocity prior is
deliberately rejected because its path orientation and denoiser formula are
different.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import torch
from torch import nn

from .losses.gmflow import extract_condition
from .physics.observation import complex_to_twochannel, twochannel_to_complex
from .physics.pnp import pnp_data_step
from .physics.rc import build_physics_cache


def _first(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = batch.get(name)
        if value is not None:
            return value
    return None


def _model_device(model: nn.Module, batch: Mapping[str, Any]) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    for name in ("Y", "observation", "y", "P", "pilot", "p"):
        value = batch.get(name)
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _as_complex(value: Any, *, device: torch.device) -> torch.Tensor:
    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    result = result.to(device)
    if result.is_complex():
        return result
    return twochannel_to_complex(result.float())


def _output_parameterization(model: nn.Module) -> str:
    for candidate in (model, getattr(model, "module", None)):
        if candidate is None:
            continue
        value = getattr(candidate, "output_parameterization", None)
        if value is not None:
            return str(value).lower()
    return "unknown"


def _validate_straight_flow_model(model: nn.Module) -> None:
    parameterization = _output_parameterization(model)
    if parameterization != "straight_velocity":
        raise ValueError(
            "PnP-Flow/FlowADMM require a straight-flow checkpoint with "
            "output_parameterization='straight_velocity'; got "
            f"{parameterization!r}. The additive/VE Flow checkpoint is not "
            "semantically compatible."
        )


def _prepare_problem(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    initialization: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    Mapping[str, Any] | None,
    str,
]:
    device = _model_device(model, batch)
    pilot_value = _first(batch, "P", "pilot", "p")
    observation_value = _first(batch, "Y", "observation", "y")
    if pilot_value is None or observation_value is None:
        raise KeyError("Flow solver sampling requires P/pilot and Y/observation")
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
    if torch.is_tensor(noise_var):
        noise_var = noise_var.to(device=device)
    cache = build_physics_cache(observation, pilot, noise_var=noise_var)

    initial_value = _first(
        batch,
        "initial",
        "H_init",
        "initial_estimate",
        "ls_estimate",
        "h_init",
    )
    if initial_value is not None:
        initial = _as_complex(initial_value, device=device)
        if initial.ndim == 2:
            initial = initial.unsqueeze(0)
        initialization_used = "provided"
    else:
        initialization_used = str(initialization).lower().replace("-", "_")
        if initialization_used in {"ls", "least_squares", "pinv"}:
            initial = torch.bmm(observation, cache["p_pinv"])
            initialization_used = "ls"
        elif initialization_used in {"adjoint", "matched_filter"}:
            initial = cache["y_p_h"].clone()
            initialization_used = "adjoint"
        elif initialization_used == "zero":
            initial = torch.zeros(
                observation.shape[0],
                observation.shape[-2],
                pilot.shape[-2],
                device=device,
                dtype=observation.dtype,
            )
        else:
            raise ValueError("initialization must be ls, adjoint, or zero")

    expected_shape = (
        observation.shape[0],
        observation.shape[-2],
        pilot.shape[-2],
    )
    if initial.shape != expected_shape:
        raise ValueError(
            f"Initial channel must have shape {expected_shape}, got {tuple(initial.shape)}"
        )
    return (
        pilot,
        observation,
        cache,
        initial,
        extract_condition(batch),
        initialization_used,
    )


def _straight_velocity(
    model: nn.Module,
    state: torch.Tensor,
    time_batch: torch.Tensor,
    condition: Mapping[str, Any] | None,
) -> torch.Tensor:
    state_twochannel = complex_to_twochannel(state)
    if condition is None:
        prediction = model(state_twochannel, time_batch)
    else:
        prediction = model(state_twochannel, time_batch, condition=condition)
    if isinstance(prediction, Mapping):
        for key in ("velocity", "prediction", "output"):
            if key in prediction:
                prediction = prediction[key]
                break
        else:
            raise TypeError("Straight-flow output mapping has no velocity tensor")
    if isinstance(prediction, (tuple, list)):
        if not prediction:
            raise TypeError("Straight-flow model returned an empty sequence")
        prediction = prediction[0]
    if not torch.is_tensor(prediction):
        raise TypeError("Straight-flow model must return a tensor")
    velocity = prediction if prediction.is_complex() else twochannel_to_complex(prediction)
    if velocity.shape != state.shape:
        raise ValueError(
            f"Flow velocity shape {tuple(velocity.shape)} does not match "
            f"state shape {tuple(state.shape)}"
        )
    if not torch.isfinite(velocity.real).all() or not torch.isfinite(velocity.imag).all():
        raise FloatingPointError("Straight-flow model produced non-finite values")
    return velocity


def _average_renoise_denoise(
    model: nn.Module,
    reference: torch.Tensor,
    time_batch: torch.Tensor,
    condition: Mapping[str, Any] | None,
    *,
    latent_samples: int,
    latent_std: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if latent_samples < 1:
        raise ValueError("latent_samples must be positive")
    if latent_std <= 0.0:
        raise ValueError("latent_std must be positive")
    reference_twochannel = complex_to_twochannel(reference)
    expand = (-1,) + (1,) * (reference_twochannel.ndim - 1)
    time_expanded = time_batch.view(expand)
    estimate_sum = torch.zeros_like(reference)
    for _ in range(latent_samples):
        latent = torch.randn(
            reference_twochannel.shape,
            device=reference_twochannel.device,
            dtype=reference_twochannel.dtype,
            generator=generator,
        )
        path_state_twochannel = (
            time_expanded * reference_twochannel
            + (1.0 - time_expanded) * float(latent_std) * latent
        )
        path_state = twochannel_to_complex(path_state_twochannel)
        velocity = _straight_velocity(model, path_state, time_batch, condition)
        estimate_sum = estimate_sum + path_state + (
            1.0 - time_batch
        ).view(-1, 1, 1) * velocity
    return estimate_sum / float(latent_samples)


def _residual_norm(channel: torch.Tensor, cache: Mapping[str, torch.Tensor]) -> torch.Tensor:
    residual = torch.bmm(channel, cache["pilot"]) - cache["observation"]
    return torch.linalg.vector_norm(residual.flatten(1), dim=1)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_flowadmm_sample_plan(
    iterations: int,
    *,
    schedule: str = "constant",
    k_avg: int = 5,
    k_avg_early: int = 1,
    k_avg_mid: int = 2,
    k_avg_late: int = 5,
    switch_frac: float = 0.5,
    switch_frac_2: float = 0.8,
) -> tuple[int, ...]:
    """Build the constant or official three-phase ``N_k`` MC plan."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    schedule = str(schedule).lower().replace("-", "_")
    if schedule == "constant":
        if k_avg < 1:
            raise ValueError("k_avg must be positive")
        return (int(k_avg),) * iterations
    if schedule != "three_phase":
        raise ValueError("schedule must be constant or three_phase")
    values = (int(k_avg_early), int(k_avg_mid), int(k_avg_late))
    if min(values) < 1:
        raise ValueError("three-phase sample counts must be positive")
    if not 0.0 <= switch_frac <= switch_frac_2 <= 1.0:
        raise ValueError("require 0 <= switch_frac <= switch_frac_2 <= 1")
    first = int(torch.floor(torch.tensor(switch_frac * iterations)).item())
    second = int(torch.floor(torch.tensor(switch_frac_2 * iterations)).item())
    return tuple(
        values[0] if index < first else values[1] if index < second else values[2]
        for index in range(iterations)
    )


@torch.no_grad()
def sample_pnp_flow(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    iterations: int = 20,
    step_scale: float = 1.0,
    alpha: float = 1.0,
    time_min: float = 0.0,
    time_max: float = 1.0,
    latent_samples: int = 5,
    latent_std: float = 1.0,
    initialization: str = "ls",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    r"""Run the full PnP-Flow FBS/reprojection trajectory.

    The data step uses ``F(H)=||HP-Y||^2/(2*sigma_y^2)`` and
    ``gamma_k=step_scale*(1-t_k)^alpha/L_F``.  Explicit division by the MIMO
    operator Lipschitz constant is the scale-normalized counterpart of the
    paper's unit-norm imaging operators.
    """

    _validate_straight_flow_model(model)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if step_scale <= 0.0:
        raise ValueError("step_scale must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    if not 0.0 <= time_min < time_max <= 1.0:
        raise ValueError("Require 0 <= time_min < time_max <= 1")
    pilot, observation, cache, x, condition, initialization_used = _prepare_problem(
        model, batch, initialization=initialization
    )
    del pilot, observation
    batch_size = x.shape[0]
    device = x.device
    initial_residual = _residual_norm(x, cache)
    time_history = []
    gamma_history = []
    residual_history = []

    _synchronize_if_cuda(device)
    sampling_start = time.perf_counter()
    for iteration in range(iterations):
        fraction = iteration / float(iterations)
        time_value = time_min + fraction * (time_max - time_min)
        time_batch = torch.full(
            (batch_size,),
            time_value,
            device=device,
            dtype=x.real.dtype,
        )
        decay = (1.0 - time_batch).pow(float(alpha)).view(batch_size, 1, 1)
        residual = torch.bmm(x, cache["pilot"]) - cache["observation"]
        raw_gradient = torch.bmm(residual, cache["p_h"]) / cache["sigma2"]
        fidelity_lipschitz = cache["fig_lipschitz"] / cache["sigma2"]
        gamma = float(step_scale) * decay / fidelity_lipschitz
        data_updated = x - gamma * raw_gradient
        x = _average_renoise_denoise(
            model,
            data_updated,
            time_batch,
            condition,
            latent_samples=latent_samples,
            latent_std=latent_std,
            generator=generator,
        )
        time_history.append(time_batch.detach().cpu())
        gamma_history.append(gamma.reshape(batch_size).detach().cpu())
        residual_history.append(_residual_norm(x, cache).detach().cpu())
    _synchronize_if_cuda(device)
    sampling_seconds = time.perf_counter() - sampling_start

    final_residual = _residual_norm(x, cache)
    diagnostics: dict[str, Any] = {
        "solver": "pnp-flow",
        "iterations": int(iterations),
        "data_consistency_evaluations": int(iterations),
        "denoiser_nfe": int(iterations * latent_samples),
        "denoiser_nfe_per_iteration": int(latent_samples),
        "latent_samples_per_iteration": int(latent_samples),
        "latent_std": float(latent_std),
        "step_scale": float(step_scale),
        "alpha": float(alpha),
        "time_min": float(time_min),
        "time_max": float(time_max),
        "initialization": initialization_used,
        "prior_output_parameterization": _output_parameterization(model),
        "measurement_residual_initial": float(initial_residual.mean().cpu()),
        "measurement_residual": float(final_residual.mean().cpu()),
        "time_history": torch.stack(time_history),
        "gamma_history": torch.stack(gamma_history),
        "measurement_residual_history": torch.stack(residual_history),
        "sampling_seconds": float(sampling_seconds),
    }
    return x, diagnostics


@torch.no_grad()
def sample_flow_admm(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    iterations: int = 20,
    tau: float = 1.0,
    time_min: float = 0.1,
    time_max: float = 0.95,
    gamma: float = 1.0,
    k_avg: int = 5,
    k_avg_schedule: str = "constant",
    k_avg_early: int = 1,
    k_avg_mid: int = 2,
    k_avg_late: int = 5,
    switch_frac: float = 0.5,
    switch_frac_2: float = 0.8,
    latent_std: float = 1.0,
    initialization: str = "ls",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    r"""Run FlowADMM with the deterministic-expectation operator estimated by MC."""

    _validate_straight_flow_model(model)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 <= time_min < time_max <= 1.0:
        raise ValueError("Require 0 <= time_min < time_max <= 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    sample_plan = build_flowadmm_sample_plan(
        iterations,
        schedule=k_avg_schedule,
        k_avg=k_avg,
        k_avg_early=k_avg_early,
        k_avg_mid=k_avg_mid,
        k_avg_late=k_avg_late,
        switch_frac=switch_frac,
        switch_frac_2=switch_frac_2,
    )
    (
        pilot,
        observation,
        cache,
        initial,
        condition,
        initialization_used,
    ) = _prepare_problem(
        model, batch, initialization=initialization
    )
    del pilot, observation
    batch_size = initial.shape[0]
    device = initial.device
    x = initial
    z = initial.clone()
    scaled_dual = torch.zeros_like(initial)
    initial_residual = _residual_norm(z, cache)
    time_history = []
    primal_history = []
    dual_history = []
    residual_history = []

    _synchronize_if_cuda(device)
    sampling_start = time.perf_counter()
    for iteration, latent_samples in enumerate(sample_plan):
        previous_z = z
        fraction = ((iteration + 1) / float(iterations)) ** float(gamma)
        time_value = time_min + fraction * (time_max - time_min)
        time_batch = torch.full(
            (batch_size,),
            time_value,
            device=device,
            dtype=x.real.dtype,
        )

        # prox_{tau F}(z-u) equals the existing PnP data step with rho=1/tau.
        x = pnp_data_step(z - scaled_dual, cache, rho=1.0 / float(tau))
        z = _average_renoise_denoise(
            model,
            x + scaled_dual,
            time_batch,
            condition,
            latent_samples=latent_samples,
            latent_std=latent_std,
            generator=generator,
        )
        scaled_dual = scaled_dual + x - z

        primal = torch.linalg.vector_norm((x - z).flatten(1), dim=1)
        dual = torch.linalg.vector_norm((z - previous_z).flatten(1), dim=1) / float(tau)
        time_history.append(time_batch.detach().cpu())
        primal_history.append(primal.detach().cpu())
        dual_history.append(dual.detach().cpu())
        residual_history.append(_residual_norm(z, cache).detach().cpu())
    _synchronize_if_cuda(device)
    sampling_seconds = time.perf_counter() - sampling_start

    final_residual = _residual_norm(z, cache)
    diagnostics: dict[str, Any] = {
        "solver": "flow-admm",
        "iterations": int(iterations),
        "data_consistency_evaluations": int(iterations),
        "denoiser_nfe": int(sum(sample_plan)),
        "denoiser_nfe_per_iteration": None,
        "latent_samples_per_iteration": list(sample_plan),
        "latent_std": float(latent_std),
        "tau": float(tau),
        "time_min": float(time_min),
        "time_max": float(time_max),
        "gamma": float(gamma),
        "k_avg_schedule": str(k_avg_schedule),
        "initialization": initialization_used,
        "prior_output_parameterization": _output_parameterization(model),
        "measurement_residual_initial": float(initial_residual.mean().cpu()),
        "measurement_residual": float(final_residual.mean().cpu()),
        "primal_residual": float(primal_history[-1].mean()),
        "dual_residual": float(dual_history[-1].mean()),
        "time_history": torch.stack(time_history),
        "primal_history": torch.stack(primal_history),
        "dual_history": torch.stack(dual_history),
        "measurement_residual_history": torch.stack(residual_history),
        "sampling_seconds": float(sampling_seconds),
    }
    return z, diagnostics


__all__ = [
    "build_flowadmm_sample_plan",
    "sample_flow_admm",
    "sample_pnp_flow",
]
