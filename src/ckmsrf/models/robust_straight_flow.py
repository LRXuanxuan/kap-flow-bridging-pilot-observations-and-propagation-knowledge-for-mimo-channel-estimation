"""Mask-aware straight-flow prior for CKM-mismatch robustness studies.

This module leaves the straight latent-to-data velocity field and every
FlowADMM update unchanged.  It only enriches the conditional prior with an
explicit modality-availability input, so an all-zero feature can be
distinguished from a deliberately missing CKM modality.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .condition import CKMConditionEncoder, _first
from .straight_flow import ConditionalStraightFlowUNet


class MaskAwareCKMConditionEncoder(CKMConditionEncoder):
    """CKM encoder augmented by a per-sample branch availability mask.

    ``branch_mask`` is always ordered as
    ``[scalar, state, beam, adp, time_frequency]``.  The inherited branch
    encoders and fusion MLP preserve their parameter names and shapes; the
    only new trainable component is a zero-initialized presence projection.
    This makes a non-strict common-initialization warm start scientifically
    transparent while strict checkpoint loads still reject model families.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("use_presence_mask", None)
        super().__init__(*args, **kwargs)
        self.branch_count = sum(
            (
                self.use_scalar,
                self.use_state,
                self.use_beam,
                self.use_adp,
                self.use_time_frequency,
            )
        )
        self.presence_projection = nn.Linear(
            self.branch_count, self.embed_dim, bias=False
        )
        nn.init.zeros_(self.presence_projection.weight)

    @staticmethod
    def _external_presence(
        condition: Mapping[str, Any],
        index: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = _first(condition, "branch_mask", "modality_mask")
        if mask is None:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)
        mask = torch.as_tensor(mask, device=device, dtype=dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.reshape(mask.shape[0], -1)
        if mask.shape != (batch_size, 5):
            raise ValueError(
                "branch_mask must be [B,5] in order "
                "[scalar,state,beam,adp,time_frequency]"
            )
        return mask[:, index : index + 1].clamp(0.0, 1.0)

    def forward(
        self,
        condition: Mapping[str, Any] | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        condition = {} if condition is None else condition
        if not isinstance(condition, Mapping):
            raise TypeError("condition must be a mapping of continuous CKM tensors")
        branches: list[torch.Tensor] = []
        presences: list[torch.Tensor] = []

        if self.use_scalar:
            value, present = self._present_or_zero(
                _first(condition, "scalar", "ckm_scalar", "context_scalar"),
                self.scalar_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            present = present * self._external_presence(
                condition, 0, batch_size=batch_size, device=device, dtype=dtype
            )
            branches.append(self.scalar_encoder(value) * present)
            presences.append(present)

        if self.use_state:
            value, present = self._present_or_zero(
                self._state_value(condition),
                self.state_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            present = present * self._external_presence(
                condition, 1, batch_size=batch_size, device=device, dtype=dtype
            )
            branches.append(self.state_encoder(value) * present)
            presences.append(present)

        if self.use_beam:
            tx, tx_present = self._present_or_zero(
                _first(condition, "tx_beam", "tx_beam_probability", "beam_tx"),
                self.tx_beam_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            rx, rx_present = self._present_or_zero(
                _first(condition, "rx_beam", "rx_beam_probability", "beam_rx"),
                self.rx_beam_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            present = torch.maximum(tx_present, rx_present) * self._external_presence(
                condition, 2, batch_size=batch_size, device=device, dtype=dtype
            )
            branches.append(self.beam_encoder(torch.cat((tx, rx), dim=-1)) * present)
            presences.append(present)

        if self.use_adp:
            value, present = self._adp_value(
                _first(condition, "adp", "ckm_adp", "context_adp"),
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            present = present * self._external_presence(
                condition, 3, batch_size=batch_size, device=device, dtype=dtype
            )
            branches.append(self.adp_encoder(value) * present)
            presences.append(present)

        if self.use_time_frequency:
            time, time_present = self._present_or_zero(
                _first(condition, "time", "snapshot", "snapshot_index", "time_coordinate"),
                1,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            frequency, frequency_present = self._present_or_zero(
                _first(
                    condition,
                    "frequency",
                    "subcarrier",
                    "subcarrier_index",
                    "frequency_coordinate",
                ),
                1,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            present = torch.maximum(time_present, frequency_present) * self._external_presence(
                condition, 4, batch_size=batch_size, device=device, dtype=dtype
            )
            coordinates = torch.cat((time, frequency), dim=-1)
            branches.append(self.tf_encoder(self.tf_features(coordinates)) * present)
            presences.append(present)

        embedding = self.fusion(torch.cat(branches, dim=-1))
        return embedding + self.presence_projection(torch.cat(presences, dim=-1))


class RobustConditionalStraightFlowUNet(ConditionalStraightFlowUNet):
    """Straight-flow velocity U-Net with a mask-aware CKM encoder.

    Its solver-facing output is exactly the same ``straight_velocity`` as the
    regular prior.  The persistent guard makes strict loads of ordinary and
    robust checkpoints fail in both directions instead of silently mixing
    model families.
    """

    output_parameterization = "straight_velocity"

    def __init__(
        self,
        *args: Any,
        condition: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        condition_kwargs = dict(condition or {})
        condition_embed_dim = int(condition_kwargs.pop("embed_dim", 256))
        cond_dropout = float(
            condition_kwargs.pop(
                "dropout_prob", condition_kwargs.pop("cond_dropout", 0.1)
            )
        )
        force_unconditional = bool(condition_kwargs.pop("force_unconditional", False))
        condition_kwargs.pop("enabled", None)
        condition_kwargs.pop("use_presence_mask", None)
        encoder = MaskAwareCKMConditionEncoder(
            embed_dim=condition_embed_dim,
            **condition_kwargs,
        )
        super().__init__(
            *args,
            condition_encoder=encoder,
            condition_embed_dim=condition_embed_dim,
            cond_dropout=cond_dropout,
            **kwargs,
        )
        self.force_unconditional = force_unconditional
        self.register_buffer(
            "_robust_straight_flow_family_guard",
            torch.tensor([7, 14, 2], dtype=torch.int64),
        )


__all__ = [
    "MaskAwareCKMConditionEncoder",
    "RobustConditionalStraightFlowUNet",
]
