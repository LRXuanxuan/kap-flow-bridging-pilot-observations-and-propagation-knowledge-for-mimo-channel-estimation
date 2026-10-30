"""Conditional Flow Matching on a straight Gaussian-to-channel path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ckmsrf.condition_corruption import (
    StructuredConditionCorruptor,
    canonicalize_condition_batch,
)

from .gmflow import extract_condition


def _first(batch: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = batch.get(name)
        if value is not None:
            return value
    return None


class StraightFlowMatchingObjective(nn.Module):
    r"""Regress ``x_data-x_latent`` along a straight conditional flow.

    The training pair is independently coupled by default:

    ``x_latent ~ N(0, latent_std^2 I)``,
    ``x_t=(1-t)x_latent+t*x_data``, and
    ``v_target=x_data-x_latent``.

    This is the path assumed by PnP-Flow and FlowADMM.  It must not be
    replaced by the additive/VE objective ``x_data+sigma*epsilon``.
    """

    def __init__(
        self,
        *,
        time_min: float = 0.0,
        time_max: float = 1.0,
        latent_std: float = 1.0,
        coupling: str = "independent",
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if not 0.0 <= time_min < time_max <= 1.0:
            raise ValueError("Require 0 <= time_min < time_max <= 1")
        if latent_std <= 0.0:
            raise ValueError("latent_std must be positive")
        coupling = str(coupling).lower().replace("-", "_")
        if coupling not in {"independent", "minibatch_ot"}:
            raise ValueError("coupling must be independent or minibatch_ot")
        self.time_min = float(time_min)
        self.time_max = float(time_max)
        self.latent_std = float(latent_std)
        self.coupling = coupling
        self.eps = float(eps)

    @staticmethod
    def _minibatch_ot_permutation(
        x_latent: torch.Tensor, x_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact uniform mini-batch OT pairing via Hungarian assignment.

        Only the Gaussian latent samples are permuted.  Each channel sample
        therefore stays aligned with its own CKM condition.  The assignment
        is discrete and intentionally computed outside autograd.
        """

        batch_size = x_data.shape[0]
        if batch_size == 1:
            index = torch.zeros(1, device=x_data.device, dtype=torch.long)
            cost = (x_latent - x_data).square().flatten(1).mean(dim=1)
            return index, cost.mean()
        latent_flat = x_latent.detach().float().flatten(1)
        data_flat = x_data.detach().float().flatten(1)
        cost_tensor = torch.cdist(latent_flat, data_flat).square()
        if not bool(torch.isfinite(cost_tensor).all()):
            raise FloatingPointError("Mini-batch OT cost contains non-finite values")
        cost = cost_tensor.cpu().tolist()

        # O(B^3) Hungarian algorithm for a square, uniform assignment.  This
        # avoids adding POT/SciPy to the original server environment.
        u = [0.0] * (batch_size + 1)
        v = [0.0] * (batch_size + 1)
        p = [0] * (batch_size + 1)
        way = [0] * (batch_size + 1)
        for row in range(1, batch_size + 1):
            p[0] = row
            column_0 = 0
            min_values = [float("inf")] * (batch_size + 1)
            used = [False] * (batch_size + 1)
            while True:
                used[column_0] = True
                row_0 = p[column_0]
                delta = float("inf")
                column_1 = 0
                for column in range(1, batch_size + 1):
                    if used[column]:
                        continue
                    current = cost[row_0 - 1][column - 1] - u[row_0] - v[column]
                    if current < min_values[column]:
                        min_values[column] = current
                        way[column] = column_0
                    if min_values[column] < delta:
                        delta = min_values[column]
                        column_1 = column
                for column in range(batch_size + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        min_values[column] -= delta
                column_0 = column_1
                if p[column_0] == 0:
                    break
            while True:
                column_1 = way[column_0]
                p[column_0] = p[column_1]
                column_0 = column_1
                if column_0 == 0:
                    break

        # p[data_column] gives the selected latent row.
        data_to_latent = torch.tensor(
            [p[column] - 1 for column in range(1, batch_size + 1)],
            device=x_data.device,
            dtype=torch.long,
        )
        data_index = torch.arange(batch_size, device=x_data.device)
        mean_cost = cost_tensor.to(x_data.device)[data_to_latent, data_index].mean()
        return data_to_latent, mean_cost

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

    @staticmethod
    def _condition_drop_mask(
        model: nn.Module, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        probability = float(getattr(model, "cond_dropout", 0.0)) if model.training else 0.0
        if probability <= 0.0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < probability

    def sample_time(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        uniform = torch.rand(batch_size, device=device, dtype=dtype)
        return self.time_min + (self.time_max - self.time_min) * uniform

    def forward(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        device, dtype = self._model_device_dtype(model)
        x_data = self._channel(batch, device=device, dtype=dtype)
        batch_size = x_data.shape[0]
        time = self.sample_time(batch_size, device=device, dtype=dtype)
        expand = (-1,) + (1,) * (x_data.ndim - 1)
        x_latent = self.latent_std * torch.randn_like(x_data)
        if self.coupling == "minibatch_ot":
            permutation, coupling_cost = self._minibatch_ot_permutation(
                x_latent, x_data
            )
            x_latent = x_latent[permutation]
        else:
            coupling_cost = (
                (x_latent.detach() - x_data.detach()).square().flatten(1).mean()
            )
        path_state = (
            (1.0 - time).view(expand) * x_latent
            + time.view(expand) * x_data
        )
        target_velocity = x_data - x_latent
        predicted_velocity = model(
            path_state,
            time,
            condition=extract_condition(batch),
            cond_drop_mask=self._condition_drop_mask(model, batch_size, device),
        )
        if predicted_velocity.shape != x_data.shape:
            raise ValueError(
                f"Straight-flow velocity has shape {tuple(predicted_velocity.shape)}; "
                f"expected {tuple(x_data.shape)}"
            )

        per_sample_velocity_mse = (
            (predicted_velocity - target_velocity).square().flatten(1).mean(dim=1)
        )
        flow_matching_loss = per_sample_velocity_mse.mean()
        clean_prediction = (
            path_state + (1.0 - time).view(expand) * predicted_velocity
        )
        squared_error = (clean_prediction - x_data).square().flatten(1).sum(dim=1)
        signal_energy = x_data.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        reconstruction_nmse = (squared_error / signal_energy).mean()
        return {
            "loss": flow_matching_loss,
            "straight_flow_matching_loss": flow_matching_loss.detach(),
            "velocity_mse": per_sample_velocity_mse.mean().detach(),
            "reconstruction_nmse": reconstruction_nmse.detach(),
            "time_mean": time.mean().detach(),
            "coupling_cost": coupling_cost.detach(),
        }


class RobustStraightFlowMatchingObjective(StraightFlowMatchingObjective):
    """Straight Flow Matching with training-only structured CKM augmentation.

    Validation remains clean by default, so the selected checkpoint answers
    whether robustness training damages the nominal-condition case.  The
    FlowADMM solver is deliberately outside this objective and remains
    identical for the regular and robust priors.
    """

    def __init__(
        self,
        *,
        corruption: Mapping[str, Any] | None = None,
        adp_shape: tuple[int, int, int] | list[int] = (16, 16, 12),
        apply_in_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.corruptor = StructuredConditionCorruptor.from_config(
            corruption, adp_shape=adp_shape
        )
        self.apply_in_eval = bool(apply_in_eval)

    def forward(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        should_corrupt = self.corruptor.enabled and (
            model.training or self.apply_in_eval
        )
        if not should_corrupt:
            return super().forward(model, batch)
        condition = self.corruptor(extract_condition(batch))
        augmented = canonicalize_condition_batch(batch, condition)
        metrics = super().forward(model, augmented)
        branch_mask = (
            condition.get("branch_mask") if isinstance(condition, Mapping) else None
        )
        if torch.is_tensor(branch_mask):
            metrics["condition_keep_rate"] = branch_mask.mean().to(
                device=metrics["loss"].device
            )
        return metrics


__all__ = [
    "RobustStraightFlowMatchingObjective",
    "StraightFlowMatchingObjective",
]
