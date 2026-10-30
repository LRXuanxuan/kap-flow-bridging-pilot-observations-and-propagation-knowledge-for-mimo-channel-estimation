"""Conditional Gaussian-mixture flow objective with optional physics terms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ckmsrf.physics import complex_to_twochannel, data_consistency_loss, twochannel_to_complex

from .gm import gm_nll_loss, gm_to_mean, reverse_transition_additive, u_to_x0_gm


_CONDITION_KEYS = (
    "scalar",
    "ckm_scalar",
    "context_scalar",
    "state",
    "position",
    "velocity",
    "orientation",
    "tx_beam",
    "tx_beam_probability",
    "rx_beam",
    "rx_beam_probability",
    "adp",
    "ckm_adp",
    "time",
    "snapshot",
    "snapshot_index",
    "frequency",
    "subcarrier",
    "subcarrier_index",
)


def _first(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch and batch[key] is not None:
            return batch[key]
    return None


def extract_condition(batch: Mapping[str, Any]) -> Mapping[str, Any] | None:
    nested = _first(batch, "condition", "context", "ckm")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    for key in _CONDITION_KEYS:
        if key in batch and batch[key] is not None:
            result[key] = batch[key]
    return result or None


class GMFlowObjective(nn.Module):
    """Likelihood training for the additive SRF path ``x_t=x_0+t*u``.

    Call as ``metrics = objective(model, batch)``.  ``batch`` may provide a
    complex channel under ``h_complex``/``H`` or a two-channel tensor under
    ``h``.  Physics fine-tuning additionally requires ``pilot``/``P`` and
    ``observation``/``Y``.
    """

    def __init__(
        self,
        *,
        snr_min_db: float = -10.0,
        snr_max_db: float = 20.0,
        snr_step_db: float = 5.0,
        time_mu: float = -0.4,
        time_sigma: float = 1.0,
        transition_ratio: float = 0.5,
        prior_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        data_consistency_weight: float = 0.0,
        gm_nll_weight: float | None = None,
        physics_weight: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not 0.0 < transition_ratio <= 1.0:
            raise ValueError("transition_ratio must lie in (0,1]")
        snr_values = torch.arange(snr_min_db, snr_max_db + 0.5 * snr_step_db, snr_step_db)
        if snr_values.numel() == 0:
            raise ValueError("Empty SNR range")
        self.register_buffer("snr_values_db", snr_values, persistent=False)
        self.time_mu = float(time_mu)
        self.time_sigma = float(time_sigma)
        self.transition_ratio = float(transition_ratio)
        self.prior_weight = float(prior_weight if gm_nll_weight is None else gm_nll_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.data_consistency_weight = float(
            data_consistency_weight if physics_weight is None else physics_weight
        )
        self.eps = float(eps)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        normal = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        return normal.sigmoid().clamp_min(1e-4)

    def _channel(self, batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        value = _first(batch, "h_complex", "H", "channel", "h")
        if value is None:
            raise KeyError("Batch requires h_complex/H or two-channel h")
        value = value.to(device)
        if value.is_complex():
            if value.ndim == 2:
                value = value.unsqueeze(0)
            complex_channel = value
            real_channel = complex_to_twochannel(value)
        else:
            real_channel = value
            if real_channel.ndim == 3:
                real_channel = real_channel.unsqueeze(0)
            complex_channel = twochannel_to_complex(real_channel)
        return real_channel.float(), complex_channel

    def forward(self, model: nn.Module, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        x0, _ = self._channel(batch, device)
        batch_size = x0.shape[0]
        snr_index = torch.randint(self.snr_values_db.numel(), (batch_size,), device=device)
        snr_db = self.snr_values_db.to(device=device)[snr_index]
        noise_var = torch.pow(torch.tensor(10.0, device=device), -snr_db / 10.0)
        noise = torch.randn_like(x0) * torch.sqrt(noise_var / 2.0).view(-1, 1, 1, 1)

        t_high = self.sample_time(batch_size, device)
        t_low = t_high * (1.0 - self.transition_ratio)
        x_high = x0 + t_high.view(-1, 1, 1, 1) * noise
        x_low = x0 + t_low.view(-1, 1, 1, 1) * noise
        condition = extract_condition(batch)
        gm_u = model(x_high, t_high, condition=condition)
        gm_low = reverse_transition_additive(gm_u, x_high, t_low, t_high)
        prior_nll = gm_nll_loss(gm_low, x_low)

        gm_x0 = u_to_x0_gm(gm_u, x_high, t_high)
        x0_est = gm_to_mean(gm_x0)
        numerator = (x0_est - x0).square().flatten(1).sum(dim=1)
        denominator = x0.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        reconstruction_nmse = (numerator / denominator).mean()
        velocity_mse = (gm_to_mean(gm_u) - noise).square().flatten(1).sum(dim=1).mean()

        pilot = _first(batch, "pilot", "P", "p")
        observation = _first(batch, "observation", "Y", "y")
        observation_noise_var = _first(batch, "noise_var", "observation_noise_var")
        dc = torch.zeros((), device=device, dtype=x0.dtype)
        if pilot is not None and observation is not None:
            pilot = pilot.to(device)
            observation = observation.to(device)
            x0_est_complex = twochannel_to_complex(x0_est)
            normalized_noise_var = (
                None
                if observation_noise_var is None
                else torch.as_tensor(observation_noise_var, device=device)
            )
            dc = data_consistency_loss(
                x0_est_complex,
                observation,
                pilot,
                noise_var=normalized_noise_var,
                eps=self.eps,
            )
        elif self.data_consistency_weight != 0.0:
            raise KeyError("data_consistency_weight > 0 requires pilot/P and observation/Y")

        loss = (
            self.prior_weight * prior_nll
            + self.reconstruction_weight * reconstruction_nmse
            + self.data_consistency_weight * dc
        )
        return {
            "loss": loss,
            "prior_nll": prior_nll.detach(),
            "nll": prior_nll.detach(),
            "reconstruction_nmse": reconstruction_nmse.detach(),
            "data_consistency": dc.detach(),
            "velocity_mse": velocity_mse.detach(),
            "unweighted_mse": velocity_mse.detach(),
            "flow_snr_db": snr_db.mean().detach(),
            "time_high": t_high.mean().detach(),
        }


GMFlowMatchingLoss = GMFlowObjective
