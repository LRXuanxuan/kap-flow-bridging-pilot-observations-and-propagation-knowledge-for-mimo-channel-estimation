"""Structured, reproducible corruption for imperfect CKM conditions.

The channel, pilots, and observations never pass through this module.  It
only perturbs the *normalized* CKM condition presented to a learned prior.
This keeps a clean/corrupt comparison attributable to environmental-prior
quality instead of changing the inverse problem itself.

``state`` follows the dataset schema
``[observed_position(3), velocity(3), sin(yaw), cos(yaw)]``.  The component
noise parameters below deliberately make position and velocity mismatch
separable; ``state_noise_std`` is retained only as a compatibility fallback
for the earlier joint-state Robust-CKMSRF experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


BRANCH_ORDER = ("scalar", "state", "beam", "adp", "time_frequency")
"""Column order used by the ``branch_mask`` condition tensor."""

CONDITION_ID_FIELDS = (
    "scene_id",
    "trajectory_id",
    "snapshot_idx",
    "subcarrier_idx",
    "sample_seed",
)
"""Stable sample identity fields required by fixed corruption at evaluation."""

# Keep this list aligned with ``losses.gmflow._CONDITION_KEYS``.  A dataset
# supplies both nested ``condition`` and flat aliases.  Once a nested condition
# has been corrupted, the aliases must be removed or ``extract_condition``
# would overwrite it with the clean values on the next model call.
TOP_LEVEL_CONDITION_KEYS = (
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


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _batch_size(condition: Mapping[str, Any]) -> int:
    for value in condition.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Cannot infer a batch size from an empty CKM condition")


def _randn_like(
    value: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    return torch.randn(
        value.shape,
        device=value.device,
        dtype=value.dtype,
        generator=generator,
    )


def _apply_sample_mask(value: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    shape = (keep.shape[0],) + (1,) * (value.ndim - 1)
    return value * keep.reshape(shape).to(device=value.device, dtype=value.dtype)


def _as_probability(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    value = value.clamp_min(0.0)
    return value / value.sum(dim=-1, keepdim=True).clamp_min(eps)


def _adp_to_bchw(
    value: torch.Tensor, adp_shape: tuple[int, int, int]
) -> tuple[torch.Tensor, str]:
    height, width, channels = adp_shape
    if value.ndim == 2:
        expected = height * width * channels
        if value.shape[1] != expected:
            raise ValueError(
                f"Flattened ADP has {value.shape[1]} values; expected {expected}"
            )
        return value.reshape(-1, height, width, channels).permute(0, 3, 1, 2), "flat"
    if value.ndim == 4 and value.shape[-1] == channels:
        return value.permute(0, 3, 1, 2), "bhwc"
    if value.ndim == 4 and value.shape[1] == channels:
        return value, "bchw"
    raise ValueError("ADP must be [B,H*W*C], [B,H,W,C], or [B,C,H,W]")


def _restore_adp(value: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "flat":
        return value.permute(0, 2, 3, 1).reshape(value.shape[0], -1)
    if layout == "bhwc":
        return value.permute(0, 2, 3, 1)
    return value


class StructuredConditionCorruptor:
    """Apply modality-aware CKM perturbations without mutating the input.

    All magnitudes are in the data loader's normalized feature space.  A
    branch mask records whole-modality dropouts, allowing a mask-aware prior
    to distinguish unavailable CKM from a valid all-zero feature value.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        scalar_noise_std: float = 0.0,
        state_noise_std: float = 0.0,
        position_noise_std: float | None = None,
        velocity_noise_std: float | None = None,
        orientation_noise_std: float | None = None,
        beam_noise_std: float = 0.0,
        beam_uniform_mix: float = 0.0,
        beam_max_shift: int = 0,
        adp_noise_std: float = 0.0,
        adp_gain_std: float = 0.0,
        adp_blur_kernel: int = 1,
        adp_max_shift: int = 0,
        scalar_dropout: float = 0.0,
        state_dropout: float = 0.0,
        beam_dropout: float = 0.0,
        adp_dropout: float = 0.0,
        time_frequency_dropout: float = 0.0,
        adp_shape: tuple[int, int, int] | list[int] = (16, 16, 12),
        **unknown: Any,
    ) -> None:
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unsupported condition-corruption settings: {names}")
        self.enabled = bool(enabled)
        self.scalar_noise_std = float(scalar_noise_std)
        self.state_noise_std = float(state_noise_std)
        self.position_noise_std = (
            None if position_noise_std is None else float(position_noise_std)
        )
        self.velocity_noise_std = (
            None if velocity_noise_std is None else float(velocity_noise_std)
        )
        self.orientation_noise_std = (
            None if orientation_noise_std is None else float(orientation_noise_std)
        )
        self.beam_noise_std = float(beam_noise_std)
        self.beam_uniform_mix = float(beam_uniform_mix)
        self.beam_max_shift = int(beam_max_shift)
        self.adp_noise_std = float(adp_noise_std)
        self.adp_gain_std = float(adp_gain_std)
        self.adp_blur_kernel = int(adp_blur_kernel)
        self.adp_max_shift = int(adp_max_shift)
        self.dropout = {
            "scalar": float(scalar_dropout),
            "state": float(state_dropout),
            "beam": float(beam_dropout),
            "adp": float(adp_dropout),
            "time_frequency": float(time_frequency_dropout),
        }
        self.adp_shape = tuple(int(value) for value in adp_shape)
        if len(self.adp_shape) != 3 or any(value <= 0 for value in self.adp_shape):
            raise ValueError("adp_shape must contain three positive values")
        for name, probability in self.dropout.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name}_dropout must lie in [0, 1]")
        for name in (
            "scalar_noise_std",
            "state_noise_std",
            "beam_noise_std",
            "adp_noise_std",
            "adp_gain_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "position_noise_std",
            "velocity_noise_std",
            "orientation_noise_std",
        ):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.beam_uniform_mix <= 1.0:
            raise ValueError("beam_uniform_mix must lie in [0, 1]")
        if self.beam_max_shift < 0 or self.adp_max_shift < 0:
            raise ValueError("shift magnitudes must be non-negative")
        if self.adp_blur_kernel < 1 or self.adp_blur_kernel % 2 == 0:
            raise ValueError("adp_blur_kernel must be a positive odd integer")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
        *,
        adp_shape: tuple[int, int, int] | list[int] = (16, 16, 12),
    ) -> "StructuredConditionCorruptor":
        values = dict(config or {})
        # This is an objective/evaluator policy rather than a mathematical
        # perturbation parameter.
        values.pop("apply_in_eval", None)
        values.setdefault("adp_shape", adp_shape)
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        """Return the resolved, serializable profile used for an evaluation."""

        return {
            "enabled": self.enabled,
            "scalar_noise_std": self.scalar_noise_std,
            "state_noise_std": self.state_noise_std,
            "position_noise_std": self.position_noise_std,
            "velocity_noise_std": self.velocity_noise_std,
            "orientation_noise_std": self.orientation_noise_std,
            "beam_noise_std": self.beam_noise_std,
            "beam_uniform_mix": self.beam_uniform_mix,
            "beam_max_shift": self.beam_max_shift,
            "adp_noise_std": self.adp_noise_std,
            "adp_gain_std": self.adp_gain_std,
            "adp_blur_kernel": self.adp_blur_kernel,
            "adp_max_shift": self.adp_max_shift,
            "scalar_dropout": self.dropout["scalar"],
            "state_dropout": self.dropout["state"],
            "beam_dropout": self.dropout["beam"],
            "adp_dropout": self.dropout["adp"],
            "time_frequency_dropout": self.dropout["time_frequency"],
            "adp_shape": list(self.adp_shape),
        }

    def _keep_masks(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        return {
            name: (
                torch.rand((batch_size,), device=device, generator=generator)
                >= probability
            ).float()
            for name, probability in self.dropout.items()
        }

    @staticmethod
    def _shift_rows(
        value: torch.Tensor,
        maximum: int,
        *,
        dims: tuple[int, ...],
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if maximum == 0:
            return value
        shifted = []
        for row in value:
            shifts = torch.randint(
                -maximum,
                maximum + 1,
                (len(dims),),
                device=value.device,
                generator=generator,
            ).tolist()
            shifted.append(torch.roll(row, tuple(int(v) for v in shifts), dims=dims))
        return torch.stack(shifted)

    def _component_noise_stds(self) -> tuple[float, float, float]:
        """Resolve new split-state settings and the legacy joint fallback."""

        split = (
            self.position_noise_std,
            self.velocity_noise_std,
            self.orientation_noise_std,
        )
        if all(value is None for value in split):
            return (self.state_noise_std, self.state_noise_std, self.state_noise_std)
        return tuple(0.0 if value is None else value for value in split)  # type: ignore[return-value]

    def _corrupt_state(
        self,
        value: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        result = value.clone()
        position_std, velocity_std, orientation_std = self._component_noise_stds()
        if position_std and result.shape[-1] > 0:
            width = min(3, result.shape[-1])
            result[..., :width] += position_std * _randn_like(
                result[..., :width], generator
            )
        if velocity_std and result.shape[-1] > 3:
            width = min(6, result.shape[-1])
            result[..., 3:width] += velocity_std * _randn_like(
                result[..., 3:width], generator
            )
        if orientation_std and result.shape[-1] > 6:
            result[..., 6:] += orientation_std * _randn_like(
                result[..., 6:], generator
            )
        return result

    def __call__(
        self,
        condition: Mapping[str, Any] | None,
        *,
        generator: torch.Generator | None = None,
    ) -> Mapping[str, Any] | None:
        if condition is None or not self.enabled:
            return condition
        if not isinstance(condition, Mapping):
            raise TypeError("condition must be a mapping")
        result = dict(condition)
        batch_size = _batch_size(condition)
        reference = next(
            value
            for value in condition.values()
            if torch.is_tensor(value) and value.ndim > 0
        )
        masks = self._keep_masks(
            batch_size, device=reference.device, generator=generator
        )

        scalar = _first(condition, "scalar", "ckm_scalar", "context_scalar")
        if torch.is_tensor(scalar):
            corrupted = scalar.clone()
            if self.scalar_noise_std:
                corrupted = corrupted + self.scalar_noise_std * _randn_like(
                    corrupted, generator
                )
            key = next(
                key
                for key in ("scalar", "ckm_scalar", "context_scalar")
                if key in condition
            )
            result[key] = _apply_sample_mask(corrupted, masks["scalar"])

        state = _first(condition, "state", "mobility_state")
        if torch.is_tensor(state):
            corrupted = self._corrupt_state(state, generator=generator)
            corrupted = _apply_sample_mask(corrupted, masks["state"])
            key = next(key for key in ("state", "mobility_state") if key in condition)
            result[key] = corrupted
            # Preserve consistency for auxiliary fields that might be consumed
            # by a future condition encoder instead of ``state``.
            if torch.is_tensor(condition.get("observed_position")):
                result["observed_position"] = corrupted[..., :3]
            if torch.is_tensor(condition.get("velocity")) and corrupted.shape[-1] > 3:
                result["velocity"] = corrupted[..., 3:6]
        else:
            observed_position = _first(
                condition, "observed_position", "position", "position_obs"
            )
            if torch.is_tensor(observed_position):
                corrupted = observed_position.clone()
                position_std, _, _ = self._component_noise_stds()
                if position_std:
                    corrupted = corrupted + position_std * _randn_like(corrupted, generator)
                key = next(
                    key
                    for key in ("observed_position", "position", "position_obs")
                    if key in condition
                )
                result[key] = _apply_sample_mask(corrupted, masks["state"])
            velocity = _first(condition, "velocity", "velocity_obs")
            if torch.is_tensor(velocity):
                corrupted = velocity.clone()
                _, velocity_std, _ = self._component_noise_stds()
                if velocity_std:
                    corrupted = corrupted + velocity_std * _randn_like(corrupted, generator)
                key = next(key for key in ("velocity", "velocity_obs") if key in condition)
                result[key] = _apply_sample_mask(corrupted, masks["state"])

        for aliases in (
            ("tx_beam", "tx_beam_probability", "beam_tx"),
            ("rx_beam", "rx_beam_probability", "beam_rx"),
        ):
            beam = _first(condition, *aliases)
            if not torch.is_tensor(beam):
                continue
            corrupted = beam.clone()
            if self.beam_noise_std:
                corrupted = corrupted + self.beam_noise_std * _randn_like(
                    corrupted, generator
                )
            corrupted = _as_probability(corrupted)
            if self.beam_uniform_mix:
                uniform = torch.full_like(corrupted, 1.0 / corrupted.shape[-1])
                corrupted = (
                    (1.0 - self.beam_uniform_mix) * corrupted
                    + self.beam_uniform_mix * uniform
                )
            corrupted = self._shift_rows(
                corrupted,
                self.beam_max_shift,
                dims=(-1,),
                generator=generator,
            )
            key = next(key for key in aliases if key in condition)
            result[key] = _apply_sample_mask(corrupted, masks["beam"])

        adp = _first(condition, "adp", "ckm_adp", "context_adp")
        if torch.is_tensor(adp):
            image, layout = _adp_to_bchw(adp.clone(), self.adp_shape)
            if self.adp_blur_kernel > 1:
                pad = self.adp_blur_kernel // 2
                image = F.avg_pool2d(
                    image,
                    kernel_size=self.adp_blur_kernel,
                    stride=1,
                    padding=pad,
                )
            image = self._shift_rows(
                image,
                self.adp_max_shift,
                dims=(-2, -1),
                generator=generator,
            )
            if self.adp_gain_std:
                gain = torch.exp(
                    self.adp_gain_std
                    * torch.randn(
                        (batch_size, 1, 1, 1),
                        device=image.device,
                        dtype=image.dtype,
                        generator=generator,
                    )
                )
                image = image * gain
            if self.adp_noise_std:
                image = image + self.adp_noise_std * _randn_like(image, generator)
            image = _apply_sample_mask(image, masks["adp"])
            key = next(key for key in ("adp", "ckm_adp", "context_adp") if key in condition)
            result[key] = _restore_adp(image, layout)

        for aliases in (
            ("time", "snapshot", "snapshot_index", "time_coordinate"),
            ("frequency", "subcarrier", "subcarrier_index", "frequency_coordinate"),
        ):
            value = _first(condition, *aliases)
            if torch.is_tensor(value):
                key = next(key for key in aliases if key in condition)
                result[key] = _apply_sample_mask(value.clone(), masks["time_frequency"])

        result["branch_mask"] = torch.stack(
            [masks[name] for name in BRANCH_ORDER], dim=-1
        )
        return result


def canonicalize_condition_batch(
    batch: Mapping[str, Any], condition: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return ``batch`` with a canonical nested condition and no stale aliases."""

    result = dict(batch)
    for key in TOP_LEVEL_CONDITION_KEYS:
        result.pop(key, None)
    result.pop("context", None)
    result.pop("ckm", None)
    result["condition"] = None if condition is None else dict(condition)
    return result


def _sample_identifier(batch: Mapping[str, Any], index: int) -> tuple[int, ...]:
    values = []
    for field in CONDITION_ID_FIELDS:
        value = batch.get(field)
        if not torch.is_tensor(value):
            raise KeyError(
                f"Stable CKM corruption requires tensor field {field!r}"
            )
        values.append(int(value[index].detach().cpu().reshape(-1)[0]))
    return tuple(values)


def _seed_from_identifier(identifier: tuple[int, ...], base_seed: int) -> int:
    """Generate a portable 63-bit seed from the corruption seed and sample ID."""

    state = int(base_seed) & 0x7FFF_FFFF_FFFF_FFFF
    for value in identifier:
        state ^= int(value) & 0x7FFF_FFFF_FFFF_FFFF
        state = (
            state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
        ) & 0x7FFF_FFFF_FFFF_FFFF
    return state


def _slice_condition(condition: Mapping[str, Any], index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in condition.items():
        result[key] = value[index : index + 1] if torch.is_tensor(value) else value
    return result


def _stack_conditions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot stack an empty list of conditions")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("Corrupted condition rows have inconsistent keys")
    result: dict[str, Any] = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = torch.cat(values, dim=0) if all(
            torch.is_tensor(value) for value in values
        ) else values[0]
    return result


def stable_corrupt_batch(
    batch: Mapping[str, Any],
    corruptor: StructuredConditionCorruptor,
    *,
    condition: Mapping[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Apply a fixed mismatch independently to every sample in ``batch``.

    The result is invariant to SNR, batch ordering/size, worker allocation and
    DDP partitioning as long as the five ``CONDITION_ID_FIELDS`` are retained.
    """

    if condition is None:
        raise KeyError("Condition corruption requested for a batch without condition")
    batch_size = _batch_size(condition)
    corrupted_rows: list[Mapping[str, Any]] = []
    for index in range(batch_size):
        row = _slice_condition(condition, index)
        reference = next(
            value for value in row.values() if torch.is_tensor(value) and value.ndim > 0
        )
        generator_device = (
            reference.device if reference.device.type == "cuda" else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(_seed_from_identifier(_sample_identifier(batch, index), seed))
        corrupted = corruptor(row, generator=generator)
        if not isinstance(corrupted, Mapping):
            raise TypeError("Corruptor must return a condition mapping")
        corrupted_rows.append(corrupted)
    return canonicalize_condition_batch(batch, _stack_conditions(corrupted_rows))


def profile_fingerprint(profile: Mapping[str, Any]) -> str:
    """Stable SHA-256 identifier recorded beside every mismatch result."""

    encoded = json.dumps(dict(profile), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "BRANCH_ORDER",
    "CONDITION_ID_FIELDS",
    "TOP_LEVEL_CONDITION_KEYS",
    "StructuredConditionCorruptor",
    "canonicalize_condition_batch",
    "profile_fingerprint",
    "stable_corrupt_batch",
]
