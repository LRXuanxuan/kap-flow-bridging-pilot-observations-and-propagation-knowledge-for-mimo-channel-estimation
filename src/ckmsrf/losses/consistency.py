"""Supervised cross-scale consistency objective for complex MIMO channels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from ckmsrf.physics import complex_to_twochannel, data_consistency_loss, twochannel_to_complex

from .gmflow import extract_condition


def _first(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch and batch[key] is not None:
            return batch[key]
    return None


def pseudo_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 3.0e-2,
) -> torch.Tensor:
    """Elementwise pseudo-Huber loss with a finite derivative near zero."""

    delta_tensor = prediction.new_tensor(delta)
    difference = prediction - target
    return (torch.sqrt(difference.square() + delta_tensor.square()) - delta_tensor).mean()


class ConsistencyObjective(nn.Module):
    """Train a conditional CM using endpoint and cross-scale supervision.

    A shared standard-normal perturbation constructs
    ``x_sigma = x0 + sigma * epsilon`` at two noise scales.  ``sigma`` is the
    per-real-channel corruption standard deviation; snapshot time is read only
    through the CKM ``condition`` mapping.

    ``teacher_model`` is optional.  Without it, the lower-noise student output
    is stop-gradient for the consistency target while still receiving its own
    endpoint-supervision gradient.  With a teacher, its lower-noise prediction
    supplies the target under ``torch.no_grad()``.

    A custom physics loss may be passed at construction.  It must accept
    ``(predicted_complex_channel, batch)`` and return a scalar.  Otherwise the
    repository's pilot residual is used whenever ``pilot/P`` and
    ``observation/Y`` are present.
    """

    def __init__(
        self,
        *,
        sigma_min: float = 2.0e-3,
        sigma_max: float = 1.0,
        rho: float = 7.0,
        num_scales: int = 40,
        consistency_weight: float = 1.0,
        endpoint_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        data_consistency_weight: float = 0.0,
        pseudo_huber_delta: float = 3.0e-2,
        physics_loss_fn: Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor] | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if sigma_min <= 0.0 or sigma_max < sigma_min:
            raise ValueError("Require 0 < sigma_min <= sigma_max")
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        if num_scales < 2:
            raise ValueError("num_scales must be >= 2")
        if pseudo_huber_delta <= 0.0:
            raise ValueError("pseudo_huber_delta must be positive")
        weights = (
            consistency_weight,
            endpoint_weight,
            reconstruction_weight,
            data_consistency_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Loss weights must be non-negative")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.num_scales = int(num_scales)
        self.consistency_weight = float(consistency_weight)
        self.endpoint_weight = float(endpoint_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.data_consistency_weight = float(data_consistency_weight)
        self.pseudo_huber_delta = float(pseudo_huber_delta)
        self.physics_loss_fn = physics_loss_fn
        self.eps = float(eps)

    @staticmethod
    def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
        try:
            parameter = next(model.parameters())
        except StopIteration:
            return torch.device("cpu"), torch.float32
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        return parameter.device, dtype

    @staticmethod
    def _channel(
        batch: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = _first(batch, "h_complex", "H", "channel", "h")
        if value is None:
            raise KeyError("Batch requires h_complex/H or a two-channel h tensor")
        value = torch.as_tensor(value, device=device)
        if value.is_complex():
            if value.ndim == 2:
                value = value.unsqueeze(0)
            value = complex_to_twochannel(value)
        elif value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[1] != 2:
            raise ValueError(f"Expected channel [B,2,Nr,Nt], got {tuple(value.shape)}")
        return value.to(dtype=dtype)

    def sample_sigma_pair(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample adjacent levels from a Karras/EDM noise discretization."""

        ramp = torch.linspace(0.0, 1.0, self.num_scales, device=device, dtype=dtype)
        inv_rho = 1.0 / self.rho
        minimum = self.sigma_min**inv_rho
        maximum = self.sigma_max**inv_rho
        levels = (minimum + ramp * (maximum - minimum)).pow(self.rho)
        # ``pow`` can round the first float32 level to slightly below
        # ``sigma_min`` (for example, 0.0019999996 instead of 0.002).  The
        # model intentionally rejects sub-minimum noise scales, so pin both
        # analytical schedule endpoints exactly after constructing the grid.
        levels[0] = torch.as_tensor(self.sigma_min, device=device, dtype=dtype)
        levels[-1] = torch.as_tensor(self.sigma_max, device=device, dtype=dtype)
        index = torch.randint(0, self.num_scales - 1, (batch_size,), device=device)
        low = levels[index]
        high = levels[index + 1]
        return high, low

    def _condition_drop_mask(self, model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
        probability = float(getattr(model, "cond_dropout", 0.0)) if model.training else 0.0
        if probability <= 0.0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < probability

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        *,
        teacher_model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        device, dtype = self._model_device_dtype(model)
        x0 = self._channel(batch, device=device, dtype=dtype)
        batch_size = x0.shape[0]
        sigma_high, sigma_low = self.sample_sigma_pair(
            batch_size,
            device=device,
            dtype=dtype,
        )
        noise = torch.randn_like(x0)
        expand = (-1,) + (1,) * (x0.ndim - 1)
        x_high = x0 + sigma_high.view(expand) * noise
        x_low = x0 + sigma_low.view(expand) * noise

        condition = extract_condition(batch)
        cond_drop_mask = self._condition_drop_mask(model, batch_size, device)
        prediction_high = model(
            x_high,
            sigma_high,
            condition=condition,
            cond_drop_mask=cond_drop_mask,
        )
        if prediction_high.shape != x0.shape:
            raise ValueError(
                f"Consistency model returned {tuple(prediction_high.shape)}; expected {tuple(x0.shape)}"
            )

        if teacher_model is None:
            prediction_low = model(
                x_low,
                sigma_low,
                condition=condition,
                cond_drop_mask=cond_drop_mask,
            )
            consistency_target = prediction_low.detach()
            endpoint_loss = 0.5 * (
                pseudo_huber_loss(
                    prediction_high,
                    x0,
                    delta=self.pseudo_huber_delta,
                )
                + pseudo_huber_loss(
                    prediction_low,
                    x0,
                    delta=self.pseudo_huber_delta,
                )
            )
        else:
            with torch.no_grad():
                consistency_target = teacher_model(
                    x_low,
                    sigma_low,
                    condition=condition,
                    cond_drop_mask=cond_drop_mask,
                )
            endpoint_loss = pseudo_huber_loss(
                prediction_high,
                x0,
                delta=self.pseudo_huber_delta,
            )

        consistency_loss = pseudo_huber_loss(
            prediction_high,
            consistency_target,
            delta=self.pseudo_huber_delta,
        )
        squared_error = (prediction_high - x0).square().flatten(1).sum(dim=1)
        signal_energy = x0.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        reconstruction_nmse = (squared_error / signal_energy).mean()

        predicted_complex = twochannel_to_complex(prediction_high)
        if self.physics_loss_fn is not None:
            dc = self.physics_loss_fn(predicted_complex, batch)
            if not torch.is_tensor(dc):
                dc = torch.as_tensor(dc, device=device, dtype=dtype)
            dc = dc.to(device=device, dtype=dtype)
            if dc.ndim != 0:
                dc = dc.mean()
        else:
            pilot = _first(batch, "pilot", "P", "p")
            observation = _first(batch, "observation", "Y", "y")
            observation_noise_var = _first(batch, "noise_var", "observation_noise_var")
            dc = torch.zeros((), device=device, dtype=dtype)
            if pilot is not None and observation is not None:
                pilot = torch.as_tensor(pilot, device=device)
                observation = torch.as_tensor(observation, device=device)
                normalized_noise_var = (
                    None
                    if observation_noise_var is None
                    else torch.as_tensor(observation_noise_var, device=device)
                )
                dc = data_consistency_loss(
                    predicted_complex,
                    observation,
                    pilot,
                    noise_var=normalized_noise_var,
                    eps=self.eps,
                ).to(dtype=dtype)
            elif self.data_consistency_weight != 0.0:
                raise KeyError("data_consistency_weight > 0 requires pilot/P and observation/Y")

        loss = (
            self.consistency_weight * consistency_loss
            + self.endpoint_weight * endpoint_loss
            + self.reconstruction_weight * reconstruction_nmse
            + self.data_consistency_weight * dc
        )
        return {
            "loss": loss,
            "consistency_loss": consistency_loss.detach(),
            "endpoint_loss": endpoint_loss.detach(),
            "reconstruction_nmse": reconstruction_nmse.detach(),
            "data_consistency": dc.detach(),
            "sigma_high": sigma_high.mean().detach(),
            "sigma_low": sigma_low.mean().detach(),
        }


ConsistencyMatchingLoss = ConsistencyObjective

__all__ = ["ConsistencyMatchingLoss", "ConsistencyObjective", "pseudo_huber_loss"]
