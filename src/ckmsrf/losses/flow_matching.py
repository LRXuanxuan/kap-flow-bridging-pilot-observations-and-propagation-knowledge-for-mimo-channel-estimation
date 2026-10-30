"""Conditional velocity matching for an additive channel-noise path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .gmflow import extract_condition


def _first(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = batch.get(name)
        if value is not None:
            return value
    return None


class FlowMatchingObjective(nn.Module):
    """Regress the velocity of ``x_sigma=x0+sigma*epsilon``.

    The target velocity with respect to the continuous path coordinate
    ``sigma`` is exactly ``epsilon``.  The corresponding one-step denoiser is
    ``x0_hat=x_sigma-sigma*v_theta``.  Sampling ``sigma`` from the same
    log-normal distribution as the EDM baseline isolates target
    parameterization (clean prediction versus velocity prediction) without
    changing the CKM backbone or the training-scale distribution.
    """

    def __init__(
        self,
        *,
        sigma_min: float = 2.0e-3,
        sigma_max: float = 2.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if sigma_min <= 0.0 or sigma_max <= sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max")
        if p_std <= 0.0:
            raise ValueError("p_std must be positive")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.eps = float(eps)

    @staticmethod
    def _model_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(model.parameters(), None)
        if parameter is None:
            return torch.device("cpu"), torch.float32
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        return parameter.device, dtype

    @staticmethod
    def _channel(
        batch: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        value = _first(batch, "h", "H_twochannel", "channel_twochannel")
        if value is None:
            complex_value = _first(batch, "H", "h_complex", "channel")
            if complex_value is None:
                raise KeyError("Batch requires h or H")
            from ckmsrf.physics.observation import complex_to_twochannel

            value = complex_to_twochannel(complex_value.to(device))
        else:
            value = value.to(device)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4 or value.shape[1] != 2:
            raise ValueError(f"Expected channel [B,2,Nr,Nt], got {tuple(value.shape)}")
        return value.to(dtype=dtype)

    def sample_sigma(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device, dtype=dtype)
        sigma = torch.exp(normal * self.p_std + self.p_mean)
        lower = self.sigma_min * (1.0 + 1.0e-4)
        return sigma.clamp(min=lower, max=self.sigma_max)

    @staticmethod
    def _condition_drop_mask(
        model: nn.Module, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        probability = float(getattr(model, "cond_dropout", 0.0)) if model.training else 0.0
        if probability <= 0.0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < probability

    def forward(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        device, dtype = self._model_device_dtype(model)
        x0 = self._channel(batch, device=device, dtype=dtype)
        batch_size = x0.shape[0]
        sigma = self.sample_sigma(batch_size, device=device, dtype=dtype)
        expand = (-1,) + (1,) * (x0.ndim - 1)
        target_velocity = torch.randn_like(x0)
        path_state = x0 + sigma.view(expand) * target_velocity
        predicted_velocity = model(
            path_state,
            sigma,
            condition=extract_condition(batch),
            cond_drop_mask=self._condition_drop_mask(model, batch_size, device),
        )
        if predicted_velocity.shape != x0.shape:
            raise ValueError(
                f"Flow velocity has shape {tuple(predicted_velocity.shape)}; "
                f"expected {tuple(x0.shape)}"
            )

        per_sample_velocity_mse = (
            (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1)
        )
        flow_matching_loss = per_sample_velocity_mse.mean()
        clean_prediction = (
            path_state - sigma.view(expand) * predicted_velocity
        )
        squared_error = (clean_prediction - x0).square().flatten(1).sum(dim=1)
        signal_energy = x0.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        reconstruction_nmse = (squared_error / signal_energy).mean()
        return {
            "loss": flow_matching_loss,
            "flow_matching_loss": flow_matching_loss.detach(),
            "velocity_mse": per_sample_velocity_mse.mean().detach(),
            "reconstruction_nmse": reconstruction_nmse.detach(),
            "sigma_mean": sigma.mean().detach(),
        }


__all__ = ["FlowMatchingObjective"]
