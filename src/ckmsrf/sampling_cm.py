"""Direct learned-prior sampling with strict pilot-constrained PnP-ADMM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import torch
from torch import nn

from .losses.gmflow import extract_condition
from .physics.observation import complex_to_twochannel, twochannel_to_complex
from .physics.pnp import (
    adaptive_pnp_data_step,
    pnp_data_step,
    residual_statistics,
)
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
    value = value if torch.is_tensor(value) else torch.as_tensor(value)
    value = value.to(device)
    if value.is_complex():
        return value
    return twochannel_to_complex(value.float())


def _batch_positive(
    value: torch.Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
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


def _cm_prediction_to_complex(prediction: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(prediction, Mapping):
        for key in ("x0", "prediction", "denoised", "sample", "output"):
            if key in prediction:
                prediction = prediction[key]
                break
        else:
            raise TypeError(
                "Prior mapping output must contain x0, prediction, denoised, sample, or output"
            )
    if isinstance(prediction, (tuple, list)):
        if not prediction:
            raise TypeError("Prior returned an empty sequence")
        prediction = prediction[0]
    if not torch.is_tensor(prediction):
        raise TypeError("Prior must return a tensor or a mapping containing one")
    if prediction.is_complex():
        result = prediction
    else:
        result = twochannel_to_complex(prediction)
    if result.shape != reference.shape:
        raise ValueError(
            f"Prior output shape {tuple(result.shape)} does not match {tuple(reference.shape)}"
        )
    if not torch.isfinite(result.real).all() or not torch.isfinite(result.imag).all():
        raise FloatingPointError("Prior produced non-finite channel values")
    return result


def _output_parameterization(model: nn.Module) -> str:
    candidates = (model, getattr(model, "module", None))
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "output_parameterization", None)
        if value is not None:
            return str(value).lower()
    return "clean"


def _model_sigma_min(model: nn.Module, default: float) -> float:
    """Read ``sigma_min`` through common runtime wrappers such as DDP."""

    candidates = (model, getattr(model, "module", None))
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "sigma_min", None)
        if value is not None:
            return float(value)
    return float(default)


def _call_prior_denoiser(
    model: nn.Module,
    channel: torch.Tensor,
    sigma: torch.Tensor,
    condition: Mapping[str, Any] | None,
) -> torch.Tensor:
    model_input = complex_to_twochannel(channel)
    if condition is None:
        prediction = model(model_input, sigma)
    else:
        prediction = model(model_input, sigma, condition=condition)
    prediction_complex = _cm_prediction_to_complex(prediction, channel)
    parameterization = _output_parameterization(model)
    if parameterization == "clean":
        return prediction_complex
    if parameterization == "velocity":
        sigma_batched = sigma.to(
            device=channel.device, dtype=channel.real.dtype
        ).reshape(channel.shape[0], 1, 1)
        return channel - sigma_batched * prediction_complex
    raise ValueError(f"Unsupported prior output parameterization: {parameterization}")


def _residual_norm(channel: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    residual = torch.bmm(channel, cache["pilot"]) - cache["observation"]
    return torch.linalg.vector_norm(residual.flatten(1), dim=1)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample_ckmcm_pnp(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    iterations: int = 4,
    lambda_reg: torch.Tensor | float = 0.1,
    rho: torch.Tensor | float = 1.0,
    adaptive_rho: bool = False,
    sigma_min: float = 1e-3,
    sigma_max: float = 1.0,
    momentum: float = 0.0,
    denoiser_relaxation: float = 1.0,
    dual_damping: float = 1.0,
    final_data_step: bool = False,
    rho_candidate_factors: Sequence[float] = (0.5, 1.0, 2.0),
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    whiteness_weight: float = 0.25,
    snapshot_iterations: Sequence[int] = (),
    record_snapshot_timing: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate a complex MIMO channel with direct-prior PnP-ADMM.

    Each iteration follows the explicit order ``z=data``, ``x=prior``,
    ``u=dual`` and does not inject random noise.  The CM scale follows the
    proximal interpretation ``sigma=sqrt(lambda_reg/rho)`` and is clipped to
    ``[sigma_min, sigma_max]``.  ``denoiser_relaxation`` and
    ``dual_damping`` make the PnP iteration robust when a learned CM is not a
    strictly non-expansive proximal map; both default to one to preserve the
    unrelaxed ADMM update.  ``final_data_step`` optionally reports one last
    exact physics proximal update rather than the unconstrained CM output.
    ``snapshot_iterations`` returns intermediate estimates from the same
    deterministic trajectory, allowing an iteration sweep to cost only the
    largest requested iteration count.  A final data step is applied to a
    snapshot without changing the state used by later iterations.

    The batch must contain a complex pilot matrix under ``P``/``pilot`` and a
    complex observation under ``Y``/``observation``.  An optional
    ``initial``/``H_init`` value overrides the default least-squares start.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min <= sigma_max")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0,1)")
    if not 0.0 < denoiser_relaxation <= 1.0:
        raise ValueError("denoiser_relaxation must lie in (0,1]")
    if not 0.0 <= dual_damping <= 1.0:
        raise ValueError("dual_damping must lie in [0,1]")
    requested_snapshots = tuple(sorted({int(value) for value in snapshot_iterations}))
    if any(value < 1 or value > iterations for value in requested_snapshots):
        raise ValueError("snapshot_iterations must lie in [1, iterations]")
    if record_snapshot_timing and not requested_snapshots:
        raise ValueError("record_snapshot_timing requires snapshot_iterations")

    device = _model_device(model, batch)
    # A ConditionalConsistencyUNet enforces its own exact identity boundary.
    # Never send a scale below that boundary even if the sampler's generic
    # lower clamp was configured more aggressively.
    model_sigma_min = _model_sigma_min(model, sigma_min)
    effective_sigma_min = max(float(sigma_min), model_sigma_min)
    if sigma_max < effective_sigma_min:
        raise ValueError(
            f"sigma_max={sigma_max} is below the model sigma_min={model_sigma_min}"
        )
    pilot_value = _first(batch, "P", "pilot", "p")
    observation_value = _first(batch, "Y", "observation", "y")
    if pilot_value is None or observation_value is None:
        raise KeyError("PnP-ADMM sampling requires P/pilot and Y/observation")
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
    batch_size = observation.shape[0]
    current_rho = _batch_positive(
        rho,
        batch_size,
        device=device,
        dtype=observation.real.dtype,
        name="rho",
    )
    lambda_batched = _batch_positive(
        lambda_reg,
        batch_size,
        device=device,
        dtype=observation.real.dtype,
        name="lambda_reg",
    )

    initial_value = _first(
        batch,
        "initial",
        "H_init",
        "initial_estimate",
        "ls_estimate",
        "h_init",
    )
    if initial_value is None:
        x = torch.bmm(observation, cache["p_pinv"])
    else:
        x = _as_complex(initial_value, device=device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
    expected_shape = (batch_size, observation.shape[-2], pilot.shape[-2])
    if x.shape != expected_shape:
        raise ValueError(f"Initial channel must have shape {expected_shape}, got {tuple(x.shape)}")
    scaled_dual = torch.zeros_like(x)
    condition = extract_condition(batch)

    initial_norm = _residual_norm(x, cache)
    rho_history = []
    sigma_history = []
    primal_history = []
    dual_history = []
    energy_history = []
    whiteness_history = []
    iteration_estimates: dict[int, torch.Tensor] = {}
    iteration_seconds: dict[int, float] = {}

    _synchronize_if_cuda(device)
    sampling_start = time.perf_counter()

    for iteration in range(1, iterations + 1):
        previous_x = x
        if adaptive_rho:
            z, current_rho, scaled_dual, _ = adaptive_pnp_data_step(
                x,
                scaled_dual,
                cache,
                current_rho,
                candidate_factors=rho_candidate_factors,
                whiteness_weight=whiteness_weight,
                rho_min=rho_min,
                rho_max=rho_max,
            )
        else:
            z = pnp_data_step(x - scaled_dual, cache, current_rho)

        sigma = torch.sqrt(lambda_batched / current_rho).clamp(
            min=effective_sigma_min,
            max=sigma_max,
        )
        denoised_x = _call_prior_denoiser(
            model,
            z + scaled_dual,
            sigma.reshape(batch_size),
            condition,
        )
        denoised_x = denoised_x + float(momentum) * (denoised_x - previous_x)
        x = z + float(denoiser_relaxation) * (denoised_x - z)
        scaled_dual = float(dual_damping) * (scaled_dual + z - x)

        primal = torch.linalg.vector_norm((z - x).flatten(1), dim=1)
        dual = current_rho.reshape(batch_size) * torch.linalg.vector_norm(
            (x - previous_x).flatten(1), dim=1
        )
        statistics = residual_statistics(
            x,
            cache,
            whiteness_weight=whiteness_weight,
        )
        rho_history.append(current_rho.reshape(batch_size).detach().cpu())
        sigma_history.append(sigma.reshape(batch_size).detach().cpu())
        primal_history.append(primal.detach().cpu())
        dual_history.append(dual.detach().cpu())
        energy_history.append(statistics["residual_energy"].detach().cpu())
        whiteness_history.append(statistics["residual_whiteness"].detach().cpu())

        if iteration in requested_snapshots:
            snapshot = x
            if final_data_step:
                snapshot = pnp_data_step(x - scaled_dual, cache, current_rho)
            iteration_estimates[iteration] = snapshot
            if record_snapshot_timing:
                _synchronize_if_cuda(device)
                iteration_seconds[iteration] = time.perf_counter() - sampling_start

    estimate = iteration_estimates.get(iterations, x)
    if final_data_step and iterations not in iteration_estimates:
        estimate = pnp_data_step(x - scaled_dual, cache, current_rho)
    _synchronize_if_cuda(device)
    sampling_seconds = time.perf_counter() - sampling_start
    final_norm = _residual_norm(estimate, cache)
    diagnostics: dict[str, Any] = {
        "iterations": float(iterations),
        "denoiser_nfe": int(iterations),
        "denoiser_nfe_per_iteration": 1,
        "prior_output_parameterization": _output_parameterization(model),
        "adaptive_rho": float(adaptive_rho),
        "denoiser_relaxation": float(denoiser_relaxation),
        "dual_damping": float(dual_damping),
        "final_data_step": float(final_data_step),
        "measurement_residual_initial": float(initial_norm.mean().cpu()),
        "measurement_residual": float(final_norm.mean().cpu()),
        "sampling_seconds": float(sampling_seconds),
        "primal_residual": float(primal_history[-1].mean()),
        "dual_residual": float(dual_history[-1].mean()),
        "rho": float(rho_history[-1].mean()),
        "sigma": float(sigma_history[-1].mean()),
        "residual_energy": float(energy_history[-1].mean()),
        "residual_whiteness": float(whiteness_history[-1].mean()),
        "rho_history": torch.stack(rho_history),
        "sigma_history": torch.stack(sigma_history),
        "primal_history": torch.stack(primal_history),
        "dual_history": torch.stack(dual_history),
        "residual_energy_history": torch.stack(energy_history),
        "residual_whiteness_history": torch.stack(whiteness_history),
    }
    if requested_snapshots:
        diagnostics["iteration_estimates"] = iteration_estimates
        diagnostics["iteration_denoiser_nfe"] = {
            iteration: iteration for iteration in requested_snapshots
        }
    if record_snapshot_timing:
        diagnostics["iteration_seconds"] = iteration_seconds
    return estimate, diagnostics


__all__ = ["sample_ckmcm_pnp"]
