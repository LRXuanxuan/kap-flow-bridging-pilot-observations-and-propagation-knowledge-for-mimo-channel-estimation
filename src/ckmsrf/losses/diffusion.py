"""Noise-conditioned diffusion denoising objective for complex MIMO channels.

This objective deliberately reuses the same conditional U-Net and continuous
noise coordinate as the consistency model.  It therefore isolates the
training objective (independent denoising versus cross-scale consistency)
while keeping model capacity, CKM conditioning and the downstream PnP solver
fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ckmsrf.physics import complex_to_twochannel

from .gmflow import extract_condition


def _first(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = batch.get(key)
        if value is not None:
            return value
    return None


class DiffusionDenoisingObjective(nn.Module):
    """Train an EDM/VE-style clean-signal denoiser.

    Samples follow ``x_sigma = x0 + sigma * epsilon``.  The network predicts
    ``x0`` directly and is optimized independently at every noise level; no
    cross-scale consistency target is used.  The default EDM weighting keeps
    small and large noise levels on comparable scales.
    """

    def __init__(
        self,
        *,
        sigma_min: float = 2.0e-3,
        sigma_max: float = 2.0,
        sigma_data: float = 1.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        weighting: str = "edm",
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if sigma_min <= 0.0 or sigma_max <= sigma_min:
            raise ValueError("Require 0 < sigma_min < sigma_max")
        if sigma_data <= 0.0 or p_std <= 0.0:
            raise ValueError("sigma_data and p_std must be positive")
        weighting = str(weighting).lower()
        if weighting not in {"edm", "uniform", "none"}:
            raise ValueError("weighting must be edm, uniform, or none")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.weighting = weighting
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

    def sample_sigma(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device, dtype=dtype)
        sigma = torch.exp(normal * self.p_std + self.p_mean)
        # Avoid sampling the exact identity boundary: at sigma_min the CM-style
        # preconditioner intentionally has no learnable residual contribution.
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

    def forward(self, model: nn.Module, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        device, dtype = self._model_device_dtype(model)
        x0 = self._channel(batch, device=device, dtype=dtype)
        batch_size = x0.shape[0]
        sigma = self.sample_sigma(batch_size, device=device, dtype=dtype)
        expand = (-1,) + (1,) * (x0.ndim - 1)
        noisy = x0 + sigma.view(expand) * torch.randn_like(x0)
        prediction = model(
            noisy,
            sigma,
            condition=extract_condition(batch),
            cond_drop_mask=self._condition_drop_mask(model, batch_size, device),
        )
        if prediction.shape != x0.shape:
            raise ValueError(
                f"Diffusion denoiser returned {tuple(prediction.shape)}; "
                f"expected {tuple(x0.shape)}"
            )

        per_sample_mse = (prediction - x0).square().flatten(1).mean(dim=1)
        if self.weighting == "edm":
            sigma_sq = sigma.square()
            data_sq = sigma.new_tensor(self.sigma_data).square()
            weight = (sigma_sq + data_sq) / (sigma_sq * data_sq).clamp_min(self.eps)
        else:
            weight = torch.ones_like(sigma)
        diffusion_loss = (weight * per_sample_mse).mean()
        squared_error = (prediction - x0).square().flatten(1).sum(dim=1)
        signal_energy = x0.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        reconstruction_nmse = (squared_error / signal_energy).mean()
        return {
            "loss": diffusion_loss,
            "diffusion_loss": diffusion_loss.detach(),
            "denoising_mse": per_sample_mse.mean().detach(),
            "reconstruction_nmse": reconstruction_nmse.detach(),
            "sigma_mean": sigma.mean().detach(),
        }


__all__ = ["DiffusionDenoisingObjective"]
