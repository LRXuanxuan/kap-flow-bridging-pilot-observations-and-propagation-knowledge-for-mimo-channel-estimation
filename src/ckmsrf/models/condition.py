"""Encoders for continuous environment/CKM side information.

The encoder intentionally keeps every input branch independent.  This makes it
possible to ablate scalar CKM, motion state, beam priors and ADP without
changing the channel U-Net.  Missing branches are represented by zeros rather
than learned "missing" constants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _as_float_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        value = value.to(device=device)
    else:
        value = torch.as_tensor(value, device=device)
    if value.is_complex():
        value = value.abs()
    return torch.nan_to_num(value.to(dtype=dtype))


def _fit_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    """Pad/truncate metadata defensively while keeping a fixed model schema."""
    x = x.reshape(x.shape[0], -1)
    if x.shape[1] == size:
        return x
    if x.shape[1] > size:
        return x[:, :size]
    return torch.nn.functional.pad(x, (0, size - x.shape[1]))


class FourierPairEmbedding(nn.Module):
    """Fixed Fourier features for normalized snapshot/frequency coordinates."""

    def __init__(self, num_bands: int = 8, max_frequency: float = 32.0):
        super().__init__()
        if num_bands < 1:
            raise ValueError("num_bands must be positive")
        bands = torch.logspace(0.0, math.log10(max_frequency), num_bands)
        self.register_buffer("bands", bands, persistent=False)
        self.output_dim = 4 * num_bands + 2

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = _fit_last_dim(coordinates, 2)
        angles = math.pi * coordinates.unsqueeze(-1) * self.bands.to(dtype=coordinates.dtype)
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
        return torch.cat((coordinates, fourier), dim=-1)


class _VectorBranch(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
        )


class CKMConditionEncoder(nn.Module):
    """Encode continuous CKM and mobility metadata into one condition vector.

    Expected condition keys (aliases in parentheses):

    ``scalar`` (``ckm_scalar``), ``state`` or position/velocity/orientation,
    ``tx_beam`` and ``rx_beam``, ``adp``, and normalized ``time``/``frequency``
    coordinates.  The ADP accepts flattened ``[B, prod(adp_shape)]``, channels
    last ``[B, H, W, C]`` or channels first ``[B, C, H, W]`` tensors.
    """

    def __init__(
        self,
        *,
        scalar_dim: int = 6,
        state_dim: int = 8,
        tx_beam_dim: int = 64,
        rx_beam_dim: int = 16,
        adp_shape: tuple[int, int, int] | list[int] = (16, 16, 12),
        embed_dim: int = 256,
        branch_dim: int | None = None,
        fourier_bands: int = 8,
        use_scalar: bool = True,
        use_state: bool = True,
        use_beam: bool = True,
        use_adp: bool = True,
        use_time_frequency: bool = True,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.state_dim = int(state_dim)
        self.tx_beam_dim = int(tx_beam_dim)
        self.rx_beam_dim = int(rx_beam_dim)
        self.adp_shape = tuple(int(v) for v in adp_shape)
        if len(self.adp_shape) != 3:
            raise ValueError("adp_shape must be (height, width, channels)")
        self.embed_dim = int(embed_dim)
        self.branch_dim = int(branch_dim or max(32, embed_dim // 2))
        self.use_scalar = bool(use_scalar)
        self.use_state = bool(use_state)
        self.use_beam = bool(use_beam)
        self.use_adp = bool(use_adp)
        self.use_time_frequency = bool(use_time_frequency)

        branch_count = 0
        if self.use_scalar:
            self.scalar_encoder = _VectorBranch(self.scalar_dim, self.branch_dim)
            branch_count += 1
        if self.use_state:
            self.state_encoder = _VectorBranch(self.state_dim, self.branch_dim)
            branch_count += 1
        if self.use_beam:
            self.beam_encoder = _VectorBranch(self.tx_beam_dim + self.rx_beam_dim, self.branch_dim)
            branch_count += 1
        if self.use_adp:
            adp_channels = self.adp_shape[2]
            hidden = max(16, self.branch_dim // 2)
            self.adp_encoder = nn.Sequential(
                nn.Conv2d(adp_channels, hidden, 3, padding=1),
                nn.GroupNorm(math.gcd(hidden, 8), hidden),
                nn.SiLU(),
                nn.Conv2d(hidden, self.branch_dim, 3, stride=2, padding=1),
                nn.GroupNorm(math.gcd(self.branch_dim, 8), self.branch_dim),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            branch_count += 1
        if self.use_time_frequency:
            self.tf_features = FourierPairEmbedding(fourier_bands)
            self.tf_encoder = _VectorBranch(self.tf_features.output_dim, self.branch_dim)
            branch_count += 1
        if branch_count == 0:
            raise ValueError("At least one condition branch must be enabled")

        self.fusion = nn.Sequential(
            nn.Linear(branch_count * self.branch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    @staticmethod
    def _present_or_zero(
        value: Any,
        dim: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value is None:
            return (
                torch.zeros(batch_size, dim, device=device, dtype=dtype),
                torch.zeros(batch_size, 1, device=device, dtype=dtype),
            )
        value = _as_float_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size).unsqueeze(-1)
        elif value.ndim == 1:
            value = value.unsqueeze(0) if batch_size == 1 else value.unsqueeze(-1)
        value = _fit_last_dim(value, dim)
        if value.shape[0] != batch_size:
            raise ValueError(f"Condition batch {value.shape[0]} does not match channel batch {batch_size}")
        return value, torch.ones(batch_size, 1, device=device, dtype=dtype)

    def _state_value(self, condition: Mapping[str, Any]) -> Any:
        state = _first(condition, "state", "mobility_state")
        if state is not None:
            return state
        pieces = [
            _first(condition, "position", "position_obs", "rx_position"),
            _first(condition, "velocity", "velocity_obs", "rx_velocity"),
            _first(condition, "orientation", "orientation_sincos", "heading"),
        ]
        pieces = [piece for piece in pieces if piece is not None]
        if not pieces:
            return None
        reference = pieces[0]
        device = reference.device if torch.is_tensor(reference) else torch.device("cpu")
        tensors = []
        for piece in pieces:
            tensor = piece if torch.is_tensor(piece) else torch.as_tensor(piece, device=device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            tensors.append(tensor.reshape(tensor.shape[0], -1))
        return torch.cat(tensors, dim=-1)

    def _adp_value(
        self,
        value: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width, channels = self.adp_shape
        if value is None:
            return (
                torch.zeros(batch_size, channels, height, width, device=device, dtype=dtype),
                torch.zeros(batch_size, 1, device=device, dtype=dtype),
            )
        value = _as_float_tensor(value, device=device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        elif value.ndim == 3 and batch_size == 1:
            value = value.unsqueeze(0)
        if value.ndim == 2:
            expected = height * width * channels
            if value.shape[1] != expected:
                raise ValueError(f"Flattened ADP has {value.shape[1]} values; expected {expected}")
            value = value.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        elif value.ndim == 4 and value.shape[-1] == channels:
            value = value.permute(0, 3, 1, 2)
        elif value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(
                "ADP must be [B, H*W*C], [B, H, W, C], or [B, C, H, W]"
            )
        if value.shape[0] != batch_size:
            raise ValueError(f"ADP batch {value.shape[0]} does not match channel batch {batch_size}")
        return value.contiguous(), torch.ones(batch_size, 1, device=device, dtype=dtype)

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
        branches = []

        if self.use_scalar:
            x, present = self._present_or_zero(
                _first(condition, "scalar", "ckm_scalar", "context_scalar"),
                self.scalar_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            branches.append(self.scalar_encoder(x) * present)

        if self.use_state:
            x, present = self._present_or_zero(
                self._state_value(condition),
                self.state_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            branches.append(self.state_encoder(x) * present)

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
            present = torch.maximum(tx_present, rx_present)
            branches.append(self.beam_encoder(torch.cat((tx, rx), dim=-1)) * present)

        if self.use_adp:
            adp, present = self._adp_value(
                _first(condition, "adp", "ckm_adp", "context_adp"),
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            branches.append(self.adp_encoder(adp) * present)

        if self.use_time_frequency:
            time, time_present = self._present_or_zero(
                _first(condition, "time", "snapshot", "snapshot_index", "time_coordinate"),
                1,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            frequency, frequency_present = self._present_or_zero(
                _first(condition, "frequency", "subcarrier", "subcarrier_index", "frequency_coordinate"),
                1,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            coordinates = torch.cat((time, frequency), dim=-1)
            present = torch.maximum(time_present, frequency_present)
            branches.append(self.tf_encoder(self.tf_features(coordinates)) * present)

        return self.fusion(torch.cat(branches, dim=-1))
