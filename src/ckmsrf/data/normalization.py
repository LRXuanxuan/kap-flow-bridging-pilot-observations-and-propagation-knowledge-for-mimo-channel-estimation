"""Train-only streaming normalization statistics for CKM-SRF data."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .shard_index import DatasetIndex, ShardRecord, _field, load_mat_shard


_FEATURE_PATHS = {
    "scalar": ("/context/scalar_feature", "context", "scalarFeature"),
    "adp": ("/context/angular_delay_power", "context", "angularDelayPower"),
    "position": ("/state/position_xyz", "state", "positionXYZ"),
    "velocity": ("/state/velocity_xyz", "state", "velocityXYZ"),
    "orientation": ("/state/ue_orientation_deg", "state", "ueOrientationDeg"),
    "observed_position": (
        "/context/observed_position_xyz",
        "context",
        "observedPositionXYZ",
    ),
}
_DEFAULT_FEATURES = (
    "scalar",
    "adp",
    "position",
    "velocity",
    "observed_position",
)


@dataclass
class _VectorMoments:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected [observations, features], got {values.shape}.")
        if values.shape[0] == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite values encountered while computing statistics.")
        batch_count = int(values.shape[0])
        batch_mean = np.mean(values, axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def finalize(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot finalize empty streaming moments.")
        variance = self.m2 / max(self.count, 1)
        return self.mean, np.maximum(np.sqrt(variance), epsilon)


@dataclass
class NormalizationStats:
    """Serializable train-split statistics.

    ``h_rms`` is ``sqrt(E |H|^2)`` over all selected training coefficients.
    Feature statistics are per feature dimension over trajectory snapshots.
    """

    h_rms: float
    feature_mean: dict[str, list[float]] = field(default_factory=dict)
    feature_std: dict[str, list[float]] = field(default_factory=dict)
    split: str = "train"
    num_trajectories: int = 0
    num_channel_coefficients: int = 0
    epsilon: float = 1e-6
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.h_rms = float(self.h_rms)
        if not np.isfinite(self.h_rms) or self.h_rms <= 0.0:
            raise ValueError(f"h_rms must be positive and finite, got {self.h_rms}.")

    def normalize_h(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(h) / self.h_rms, dtype=np.complex64)

    def denormalize_h(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(h) * self.h_rms, dtype=np.complex64)

    def normalize_feature(self, name: str, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if name not in self.feature_mean:
            return value
        mean = np.asarray(self.feature_mean[name], dtype=np.float32)
        std = np.asarray(self.feature_std[name], dtype=np.float32)
        if value.shape[-1] != mean.size:
            raise ValueError(
                f"Feature {name!r} has width {value.shape[-1]}, expected {mean.size}."
            )
        return np.asarray((value - mean) / std, dtype=np.float32)

    def denormalize_feature(self, name: str, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if name not in self.feature_mean:
            return value
        mean = np.asarray(self.feature_mean[name], dtype=np.float32)
        std = np.asarray(self.feature_std[name], dtype=np.float32)
        return np.asarray(value * std + mean, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "h_rms": self.h_rms,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "split": self.split,
            "num_trajectories": self.num_trajectories,
            "num_channel_coefficients": self.num_channel_coefficients,
            "epsilon": self.epsilon,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def _h5_complex_chunk(
    handle: h5py.File,
    record: ShardRecord,
    start: int,
    stop: int,
) -> np.ndarray:
    selection: Any
    if record.storage_order == "matlab_reversed":
        selection = (slice(start, stop), slice(None), slice(None), slice(None), slice(None))
    else:
        selection = (slice(None), slice(None), slice(None), slice(None), slice(start, stop))
    if "/H_real" in handle:
        real = np.asarray(handle["/H_real"][selection], dtype=np.float32)
        imag = np.asarray(handle["/H_imag"][selection], dtype=np.float32)
        return np.asarray(real + 1j * imag, dtype=np.complex64)
    return np.asarray(handle["/H"][selection], dtype=np.complex64)


def _logical_feature_chunk(
    dataset: h5py.Dataset,
    record: ShardRecord,
    start: int,
    stop: int,
) -> np.ndarray:
    if record.storage_order == "matlab_reversed":
        value = np.asarray(dataset[start:stop, ...])
        if value.ndim == 2:  # [sample, snapshot] scalar feature
            value = value[..., None]
        return value.reshape(-1, value.shape[-1])
    value = np.asarray(dataset[..., start:stop])
    if value.ndim == 2:  # [snapshot, sample]
        value = value[None, ...]
    axes = (value.ndim - 1, value.ndim - 2, *range(value.ndim - 2))
    logical = np.transpose(value, axes=axes)
    return logical.reshape(-1, int(np.prod(value.shape[:-2])))


def _mat_array(shard: Any, group: str, field_name: str) -> np.ndarray:
    value = _field(_field(shard, group), field_name)
    while isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        value = value.reshape(-1)[0]
    return np.asarray(value)


def _mat_feature_chunk(
    shard: Any, group: str, field_name: str, start: int, stop: int
) -> np.ndarray:
    value = _mat_array(shard, group, field_name)[..., start:stop]
    if value.ndim == 2:
        value = value[None, ...]
    logical = np.transpose(value, axes=(value.ndim - 1, value.ndim - 2, *range(value.ndim - 2)))
    return logical.reshape(-1, int(np.prod(value.shape[:-2])))


def compute_normalization(
    index: DatasetIndex,
    *,
    features: Iterable[str] = _DEFAULT_FEATURES,
    chunk_trajectories: int = 8,
    max_trajectories: int | None = None,
    epsilon: float = 1e-6,
    allow_non_train: bool = False,
) -> NormalizationStats:
    """Stream over shards without loading the full dataset into memory.

    By default this function refuses validation/test indices. This guard makes
    accidental evaluation-statistic leakage an explicit opt-in operation.
    """

    if index.split != "train" and not allow_non_train:
        raise ValueError(
            f"Normalization must be computed from train only, got {index.split!r}."
        )
    if chunk_trajectories < 1:
        raise ValueError("chunk_trajectories must be at least one.")
    if max_trajectories is not None and max_trajectories < 1:
        raise ValueError("max_trajectories must be positive when provided.")
    selected_features = tuple(features)
    unknown = set(selected_features) - set(_FEATURE_PATHS)
    if unknown:
        raise KeyError(f"Unknown normalization features: {sorted(unknown)}")

    moments = {name: _VectorMoments() for name in selected_features}
    channel_energy = 0.0
    channel_count = 0
    trajectories_seen = 0

    for record in index.records:
        if max_trajectories is not None and trajectories_seen >= max_trajectories:
            break
        take = record.num_trajectories
        if max_trajectories is not None:
            take = min(take, max_trajectories - trajectories_seen)
        path = record.resolve(index.root)

        if record.backend == "plain_h5":
            with h5py.File(path, "r") as handle:
                for start in range(0, take, chunk_trajectories):
                    stop = min(start + chunk_trajectories, take)
                    h = _h5_complex_chunk(handle, record, start, stop)
                    channel_energy += float(
                        np.sum(
                            h.real.astype(np.float64) ** 2
                            + h.imag.astype(np.float64) ** 2,
                            dtype=np.float64,
                        )
                    )
                    channel_count += int(h.size)
                    for name in selected_features:
                        h5_path = _FEATURE_PATHS[name][0]
                        if h5_path not in handle:
                            continue
                        moments[name].update(
                            _logical_feature_chunk(handle[h5_path], record, start, stop)
                        )
        else:
            shard = load_mat_shard(path)
            h_full = np.asarray(_field(shard, "H_true"))
            for start in range(0, take, chunk_trajectories):
                stop = min(start + chunk_trajectories, take)
                h = np.asarray(h_full[..., start:stop], dtype=np.complex64)
                channel_energy += float(
                    np.sum(
                        h.real.astype(np.float64) ** 2
                        + h.imag.astype(np.float64) ** 2,
                        dtype=np.float64,
                    )
                )
                channel_count += int(h.size)
                for name in selected_features:
                    _, group, field_name = _FEATURE_PATHS[name]
                    moments[name].update(
                        _mat_feature_chunk(shard, group, field_name, start, stop)
                    )
        trajectories_seen += take

    if channel_count == 0:
        raise ValueError("No channel coefficients were read while computing statistics.")
    h_rms = float(np.sqrt(channel_energy / channel_count))
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    for name, accumulator in moments.items():
        if accumulator.count == 0:
            continue
        mean, std = accumulator.finalize(epsilon)
        means[name] = mean.tolist()
        stds[name] = std.tolist()
    return NormalizationStats(
        # Physical mmWave channels can legitimately have RMS well below
        # 1e-6.  ``epsilon`` is a feature-standard-deviation floor and must
        # not silently rescale the channel by orders of magnitude.
        h_rms=max(h_rms, float(np.finfo(np.float64).tiny)),
        feature_mean=means,
        feature_std=stds,
        split=index.split,
        num_trajectories=trajectories_seen,
        num_channel_coefficients=channel_count,
        epsilon=epsilon,
    )


__all__ = ["NormalizationStats", "compute_normalization"]
