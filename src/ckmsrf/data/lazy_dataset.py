"""Lazy, trajectory-indexed PyTorch dataset for CKM-SRF shards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .observation import (
    ObservationConfig,
    complex_to_channels,
    derive_seed,
    synthesize_observation,
)
from .shard_index import (
    DatasetIndex,
    ShardRecord,
    _field,
    build_shard_index,
    canonical_backend,
    load_mat_shard,
)


def _unwrap(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        value = value.reshape(-1)[0]
    return value


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(_unwrap(value))


class _H5HandleCache:
    """Per-process LRU of read-only HDF5 handles.

    No live h5py object is serialized.  This is essential for Windows
    DataLoader workers (spawn) and also prevents handles inherited across a
    Unix fork from being reused unsafely.
    """

    def __init__(self, max_open_files: int) -> None:
        if max_open_files < 1:
            raise ValueError("max_open_files must be at least one.")
        self.max_open_files = int(max_open_files)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def get(self, path: str | Path) -> h5py.File:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        key = str(Path(path).resolve())
        handle = self._handles.pop(key, None)
        if handle is None:
            handle = h5py.File(key, "r")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except (OSError, ValueError):
                pass
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(int(state["max_open_files"]))


class _MatObjectCache:
    """Small per-process cache for the shard-eager raw MATLAB fallback."""

    def __init__(self, max_open_files: int) -> None:
        self.max_items = max(1, int(max_open_files))
        self._pid = os.getpid()
        self._values: OrderedDict[str, Any] = OrderedDict()

    def get_shard(self, path: str | Path) -> Any:
        if self._pid != os.getpid():
            self._values.clear()
            self._pid = os.getpid()
        key = str(Path(path).resolve())
        value = self._values.pop(key, None)
        if value is None:
            value = load_mat_shard(key)
        self._values[key] = value
        while len(self._values) > self.max_items:
            self._values.popitem(last=False)
        return value

    def get_scene(self, path: str | Path) -> Any:
        if self._pid != os.getpid():
            self._values.clear()
            self._pid = os.getpid()
        key = "scene::" + str(Path(path).resolve())
        value = self._values.pop(key, None)
        if value is None:
            try:
                import hdf5storage  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "Reading raw MATLAB scene files requires hdf5storage."
                ) from exc
            loaded = hdf5storage.loadmat(str(path), variable_names=["scenePackage"])
            value = loaded["scenePackage"]
        self._values[key] = value
        while len(self._values) > self.max_items:
            self._values.popitem(last=False)
        return value

    def close(self) -> None:
        self._values.clear()

    def __getstate__(self) -> dict[str, Any]:
        return {"max_open_files": self.max_items}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(int(state["max_open_files"]))


def _read_complex_slice(
    handle: h5py.File, base: str, selection: Any
) -> np.ndarray:
    real_name, imag_name = f"{base}_real", f"{base}_imag"
    if real_name in handle:
        real = np.asarray(handle[real_name][selection])
        imag = np.asarray(handle[imag_name][selection])
        return np.asarray(real + 1j * imag, dtype=np.complex64)
    if base in handle:
        value = np.asarray(handle[base][selection])
        if np.iscomplexobj(value):
            return np.asarray(value, dtype=np.complex64)
        if value.dtype.fields and {"real", "imag"} <= set(value.dtype.fields):
            return np.asarray(value["real"] + 1j * value["imag"], dtype=np.complex64)
    raise KeyError(f"Complex dataset {base!r} was not found in {handle.filename}.")


def _read_feature(
    handle: h5py.File,
    name: str,
    *,
    sample: int,
    snapshot: int,
    storage_order: str,
) -> np.ndarray:
    dataset = handle[name]
    if storage_order == "matlab_reversed":
        return np.asarray(dataset[sample, snapshot, ...]).reshape(-1)
    return np.asarray(dataset[..., snapshot, sample]).reshape(-1)


def _read_sample_scalar(
    dataset: h5py.Dataset, sample: int, *, storage_order: str
) -> Any:
    if dataset.ndim == 0:
        return dataset[()]
    if storage_order == "matlab_reversed":
        value = dataset[sample, ...]
    else:
        value = dataset[..., sample]
    return np.asarray(value).reshape(-1)[0]


def _read_time(handle: h5py.File, snapshot: int, storage_order: str) -> float:
    if "/state/time_sec" not in handle:
        return float(snapshot)
    dataset = handle["/state/time_sec"]
    if dataset.ndim == 1:
        return float(dataset[snapshot])
    if storage_order == "matlab_reversed":
        return float(np.asarray(dataset[..., snapshot]).reshape(-1)[0])
    return float(np.asarray(dataset[snapshot, ...]).reshape(-1)[0])


@dataclass(frozen=True)
class _Address:
    trajectory_global: int
    record: ShardRecord
    trajectory_local: int
    subcarrier: int
    snapshot: int


class CKMSRFDataset(Dataset[dict[str, Any]]):
    """Return one 2-D channel slice plus its continuous environment context.

    Sampling modes:

    * ``random_tf``: one deterministic pseudo-random (snapshot, subcarrier)
      per trajectory and epoch;
    * ``center``: the lower center snapshot/subcarrier;
    * ``all_tf``: every trajectory/snapshot/subcarrier combination;
    * ``fixed`` / ``fixed_eval``: configured pairs for deterministic validation.

    The dataset is safe to pickle because no live file handle is part of its
    serialized state. Call :meth:`set_epoch` before recreating the DataLoader
    workers each epoch when ``random_tf`` is used.
    """

    def __init__(
        self,
        index: DatasetIndex | str | Path,
        *,
        split: str | None = None,
        backend: str = "auto",
        scene_ids: Sequence[int] | None = None,
        sampling: str = "random_tf",
        fixed_pairs: Sequence[tuple[int, int]] | None = None,
        normalization: Any | str | Path | None = None,
        include_observation: bool = False,
        observation: ObservationConfig | Mapping[str, Any] | None = None,
        adp_shape: Sequence[int] = (16, 16, 12),
        seed: int = 0,
        max_open_files: int = 2,
    ) -> None:
        if isinstance(index, DatasetIndex):
            self.index = index
        else:
            source = Path(index).expanduser()
            if source.is_file() and source.suffix.lower() == ".json":
                self.index = DatasetIndex.load(source)
            else:
                if split is None:
                    raise ValueError("split is required when index is a dataset root.")
                self.index = build_shard_index(
                    source, split, backend=canonical_backend(backend), scene_ids=scene_ids
                )

        aliases = {"fixed_eval": "fixed", "fixed_tf": "fixed"}
        self.sampling = aliases.get(sampling.lower(), sampling.lower())
        if self.sampling not in {"random_tf", "center", "all_tf", "fixed"}:
            raise ValueError(
                "sampling must be random_tf, center, all_tf, fixed, or fixed_eval."
            )
        self.fixed_pairs = (
            tuple((int(k), int(t)) for k, t in fixed_pairs)
            if fixed_pairs is not None
            else None
        )
        if self.sampling == "fixed" and not self.fixed_pairs:
            first = self.index.records[0]
            self.fixed_pairs = (
                ((first.num_subcarriers - 1) // 2, (first.num_snapshots - 1) // 2),
            )
        self._validate_fixed_pairs()

        if isinstance(normalization, (str, Path)):
            from .normalization import NormalizationStats

            self.normalization = NormalizationStats.load(normalization)
        else:
            self.normalization = normalization
        self.include_observation = bool(include_observation)
        self.adp_shape = tuple(int(value) for value in adp_shape)
        if len(self.adp_shape) != 3 or any(value <= 0 for value in self.adp_shape):
            raise ValueError("adp_shape must contain three positive dimensions.")
        if observation is None:
            self.observation = ObservationConfig(enabled=self.include_observation)
        elif isinstance(observation, Mapping):
            self.observation = ObservationConfig(**dict(observation))
        else:
            self.observation = observation
        self.include_observation = self.include_observation and self.observation.enabled
        if self.include_observation:
            missing = set(self.index.scene_ids) - set(self.index.scene_files)
            if missing:
                raise ValueError(
                    f"Online observations require scene pilot files; missing {sorted(missing)}."
                )

        self.seed = int(seed)
        self.epoch = 0
        self.max_open_files = int(max_open_files)
        self._h5_cache = _H5HandleCache(self.max_open_files)
        # Raw MAT shards are hundreds of MB; retaining one is the safe default.
        self._mat_cache = _MatObjectCache(min(self.max_open_files, 1))
        self._expanded_cumulative = self._make_expanded_cumulative()

    @classmethod
    def from_root(cls, root: str | Path, split: str, **kwargs: Any) -> "CKMSRFDataset":
        return cls(root, split=split, **kwargs)

    def _validate_fixed_pairs(self) -> None:
        if not self.fixed_pairs:
            return
        for record in self.index.records:
            for subcarrier, snapshot in self.fixed_pairs:
                if not (0 <= subcarrier < record.num_subcarriers):
                    raise ValueError(f"Fixed subcarrier {subcarrier} is invalid for {record.path}.")
                if not (0 <= snapshot < record.num_snapshots):
                    raise ValueError(f"Fixed snapshot {snapshot} is invalid for {record.path}.")

    def _make_expanded_cumulative(self) -> list[int]:
        cumulative: list[int] = []
        total = 0
        for record in self.index.records:
            multiplier = 1
            if self.sampling == "all_tf":
                multiplier = record.num_subcarriers * record.num_snapshots
            elif self.sampling == "fixed":
                multiplier = len(self.fixed_pairs or ())
            total += record.num_trajectories * multiplier
            cumulative.append(total)
        return cumulative

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._expanded_cumulative[-1]

    def _address(self, item: int) -> _Address:
        size = len(self)
        if item < 0:
            item += size
        if not 0 <= item < size:
            raise IndexError(f"Dataset index {item} is outside [0, {size}).")

        record_idx = int(np.searchsorted(self._expanded_cumulative, item, side="right"))
        start = 0 if record_idx == 0 else self._expanded_cumulative[record_idx - 1]
        offset = item - start
        record = self.index.records[record_idx]

        if self.sampling == "all_tf":
            per_trajectory = record.num_snapshots * record.num_subcarriers
            local_trajectory, remainder = divmod(offset, per_trajectory)
            snapshot, subcarrier = divmod(remainder, record.num_subcarriers)
        elif self.sampling == "fixed":
            pairs = self.fixed_pairs or ()
            local_trajectory, pair_index = divmod(offset, len(pairs))
            subcarrier, snapshot = pairs[pair_index]
        else:
            local_trajectory = offset
            if self.sampling == "center":
                subcarrier = (record.num_subcarriers - 1) // 2
                snapshot = (record.num_snapshots - 1) // 2
            else:
                random_seed = derive_seed(
                    self.seed, self.epoch, record.scene_id, record.shard_id, local_trajectory
                )
                subcarrier = int(random_seed % record.num_subcarriers)
                snapshot = int((random_seed >> 32) % record.num_snapshots)

        trajectory_start = sum(
            r.num_trajectories for r in self.index.records[:record_idx]
        )
        return _Address(
            trajectory_global=trajectory_start + local_trajectory,
            record=record,
            trajectory_local=local_trajectory,
            subcarrier=subcarrier,
            snapshot=snapshot,
        )

    def _plain_sample(self, address: _Address) -> dict[str, Any]:
        record = address.record
        n, k, t = address.trajectory_local, address.subcarrier, address.snapshot
        handle = self._h5_cache.get(record.resolve(self.index.root))
        if record.storage_order == "matlab_reversed":
            h = _read_complex_slice(handle, "/H", (n, t, k, slice(None), slice(None))).T
            channel_scale = float(handle["/channel_scale_lin"][n, t, k])
        else:
            h = _read_complex_slice(handle, "/H", (slice(None), slice(None), k, t, n))
            channel_scale = float(handle["/channel_scale_lin"][k, t, n])

        def feature(name: str) -> np.ndarray:
            return _read_feature(
                handle, name, sample=n, snapshot=t, storage_order=record.storage_order
            ).astype(np.float32, copy=False)

        sample = {
            "H": np.asarray(h, dtype=np.complex64),
            "channel_scale_lin": channel_scale,
            "position": feature("/state/position_xyz"),
            "velocity": feature("/state/velocity_xyz"),
            "orientation": feature("/state/ue_orientation_deg"),
            "observed_position": feature("/context/observed_position_xyz"),
            "scalar": feature("/context/scalar_feature"),
            "adp": feature("/context/angular_delay_power"),
            "tx_beam": feature("/context/tx_beam_probability"),
            "rx_beam": feature("/context/rx_beam_probability"),
            "time_sec": _read_time(handle, t, record.storage_order),
        }
        sample["grid_indices"] = (
            feature("/context/grid_indices").astype(np.int64, copy=False)
            if "/context/grid_indices" in handle
            else np.zeros(4, dtype=np.int64)
        )
        sample["grid_weights"] = (
            feature("/context/grid_weights")
            if "/context/grid_weights" in handle
            else np.zeros(4, dtype=np.float32)
        )
        optional_snapshot = {"block_id": "/state/block_id"}
        for key, path in optional_snapshot.items():
            if path in handle:
                if record.storage_order == "matlab_reversed":
                    sample[key] = int(handle[path][n, t])
                else:
                    sample[key] = int(handle[path][t, n])
            else:
                sample[key] = 0
        for key, path, default in (
            ("pilot_id_raw", "/observation/pilot_id", 1),
            ("snr_db", "/observation/snr_db", 0.0),
            ("cfo_hz", "/observation/cfo_hz", 0.0),
            ("noise_var", "/observation/noise_var", 0.0),
            ("sample_seed", "/sample_seed", 0),
        ):
            sample[key] = (
                _read_sample_scalar(handle[path], n, storage_order=record.storage_order)
                if path in handle
                else default
            )
        return sample

    def _mat_sample(self, address: _Address) -> dict[str, Any]:
        record = address.record
        n, k, t = address.trajectory_local, address.subcarrier, address.snapshot
        shard = self._mat_cache.get_shard(record.resolve(self.index.root))

        def nested(group: str, name: str) -> np.ndarray:
            value = _as_array(_field(_field(shard, group), name))
            return np.asarray(value[..., t, n]).reshape(-1).astype(np.float32)

        h_all = _as_array(_field(shard, "H_true"))
        sample: dict[str, Any] = {
            "H": np.asarray(h_all[:, :, k, t, n], dtype=np.complex64),
            "channel_scale_lin": float(
                _as_array(_field(shard, "channelScaleLin"))[k, t, n]
            ),
            "position": nested("state", "positionXYZ"),
            "velocity": nested("state", "velocityXYZ"),
            "orientation": nested("state", "ueOrientationDeg"),
            "observed_position": nested("context", "observedPositionXYZ"),
            "scalar": nested("context", "scalarFeature"),
            "adp": nested("context", "angularDelayPower"),
            "tx_beam": nested("context", "txBeamProbability"),
            "rx_beam": nested("context", "rxBeamProbability"),
        }
        state = _field(shard, "state")
        time_axis = _as_array(_field(state, "timeSec")).reshape(-1)
        sample["time_sec"] = float(time_axis[t])
        block = _as_array(_field(state, "blockId"))
        sample["block_id"] = int(block[t, n])
        context = _field(shard, "context")
        try:
            sample["grid_indices"] = np.asarray(
                _as_array(_field(context, "gridIndices"))[:, t, n], dtype=np.int64
            )
            sample["grid_weights"] = np.asarray(
                _as_array(_field(context, "gridWeights"))[:, t, n], dtype=np.float32
            )
        except KeyError:
            sample["grid_indices"] = np.zeros(4, dtype=np.int64)
            sample["grid_weights"] = np.zeros(4, dtype=np.float32)
        observation = _field(shard, "observation")
        for key, field, default in (
            ("pilot_id_raw", "pilotId", 1),
            ("snr_db", "snrDb", 0.0),
            ("cfo_hz", "cfoHz", 0.0),
            ("noise_var", "noiseVar", 0.0),
        ):
            try:
                sample[key] = _as_array(_field(observation, field)).reshape(-1)[n]
            except KeyError:
                sample[key] = default
        try:
            sample["sample_seed"] = _as_array(_field(shard, "sampleSeed")).reshape(-1)[n]
        except KeyError:
            sample["sample_seed"] = 0
        return sample

    def _pilot(self, scene_id: int, pilot_id_raw: int, nt: int) -> tuple[np.ndarray, int]:
        scene_path = self.index.resolve_scene_file(scene_id)
        if scene_path.suffix.lower() in {".h5", ".hdf5"}:
            handle = self._h5_cache.get(scene_path)
            if "/pilot/P_real" in handle:
                dataset = handle["/pilot/P_real"]
                number = dataset.shape[0] if dataset.shape[-1] == nt else dataset.shape[-1]
                # MATLAB ids are 1-based. Accept an explicit zero id defensively.
                pilot_index = pilot_id_raw - 1 if pilot_id_raw >= 1 else pilot_id_raw
                pilot_index %= int(number)
                if dataset.shape[-1] == nt:  # reversed [bank, pilot, tx]
                    p = _read_complex_slice(
                        handle, "/pilot/P", (pilot_index, slice(None), slice(None))
                    ).T
                else:  # semantic [tx, pilot, bank]
                    p = _read_complex_slice(
                        handle, "/pilot/P", (slice(None), slice(None), pilot_index)
                    )
            else:
                dataset = handle["/pilot/P"]
                number = dataset.shape[0] if dataset.shape[-1] == nt else dataset.shape[-1]
                pilot_index = (pilot_id_raw - 1 if pilot_id_raw >= 1 else pilot_id_raw) % int(number)
                if dataset.shape[-1] == nt:
                    p = np.asarray(dataset[pilot_index]).T
                else:
                    p = np.asarray(dataset[..., pilot_index])
            return np.asarray(p, dtype=np.complex64), int(pilot_index)

        package = self._mat_cache.get_scene(scene_path)
        pilot_bank = _field(_field(package, "pilotBank"), "P")
        bank = _as_array(pilot_bank)
        pilot_index = (pilot_id_raw - 1 if pilot_id_raw >= 1 else pilot_id_raw) % bank.shape[-1]
        return np.asarray(bank[..., pilot_index], dtype=np.complex64), int(pilot_index)

    def _normalize(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.normalization is None:
            return sample
        sample["H"] = self.normalization.normalize_h(sample["H"])
        feature_map = {
            "scalar": "scalar",
            "adp": "adp",
            "position": "position",
            "velocity": "velocity",
            "orientation": "orientation",
            "observed_position": "observed_position",
        }
        for sample_key, stats_key in feature_map.items():
            sample[sample_key] = self.normalization.normalize_feature(
                stats_key, sample[sample_key]
            )
        sample["noise_var"] = float(sample["noise_var"]) / (
            float(self.normalization.h_rms) ** 2
        )
        return sample

    @staticmethod
    def _float_tensor(value: Any) -> torch.Tensor:
        array = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(array))

    def __getitem__(self, item: int) -> dict[str, Any]:
        address = self._address(item)
        raw = (
            self._plain_sample(address)
            if address.record.backend == "plain_h5"
            else self._mat_sample(address)
        )
        # Orientation must remain in physical degrees for the periodic
        # sin/cos state representation; z-scoring an angle before trigonometry
        # would destroy its meaning.
        orientation_deg_raw = float(np.asarray(raw["orientation"]).reshape(-1)[0])
        raw = self._normalize(raw)
        # The generator linearizes ADP with MATLAB/Fortran order (AoD is the
        # fastest axis). Convert the vector to C order so the model's standard
        # reshape [AoD, AoA, delay] preserves those physical axes.
        adp = np.asarray(raw["adp"], dtype=np.float32).reshape(-1)
        if adp.size == int(np.prod(self.adp_shape)):
            raw["adp"] = np.reshape(adp, self.adp_shape, order="F").reshape(-1)
        h = np.asarray(raw["H"], dtype=np.complex64)
        k, t = address.subcarrier, address.snapshot
        record = address.record
        tf = np.asarray(
            [
                2.0 * k / max(record.num_subcarriers - 1, 1) - 1.0,
                2.0 * t / max(record.num_snapshots - 1, 1) - 1.0,
            ],
            dtype=np.float32,
        )
        yaw_rad = np.deg2rad(orientation_deg_raw)
        state = np.concatenate(
            (
                np.asarray(raw["observed_position"], dtype=np.float32).reshape(-1)[:3],
                np.asarray(raw["velocity"], dtype=np.float32).reshape(-1)[:3],
                np.asarray([np.sin(yaw_rad), np.cos(yaw_rad)], dtype=np.float32),
            )
        )
        if state.size != 8:
            raise ValueError(
                "The state condition requires observed_position[3], velocity[3], "
                f"and orientation sin/cos[2], got width {state.size}."
            )
        condition = {
            "scalar": self._float_tensor(raw["scalar"]),
            "adp": self._float_tensor(raw["adp"]),
            "tx_beam": self._float_tensor(raw["tx_beam"]),
            "rx_beam": self._float_tensor(raw["rx_beam"]),
            "position": self._float_tensor(raw["position"]),
            "velocity": self._float_tensor(raw["velocity"]),
            "orientation": self._float_tensor(raw["orientation"]),
            "observed_position": self._float_tensor(raw["observed_position"]),
            "state": self._float_tensor(state),
            "tf": self._float_tensor(tf),
            "frequency": torch.tensor(float(tf[0]), dtype=torch.float32),
            "time": torch.tensor(float(tf[1]), dtype=torch.float32),
            "time_sec": torch.tensor(float(raw["time_sec"]), dtype=torch.float32),
            "grid_indices": torch.from_numpy(
                np.ascontiguousarray(np.asarray(raw["grid_indices"], dtype=np.int64))
            ),
            "grid_weights": self._float_tensor(raw["grid_weights"]),
        }
        output: dict[str, Any] = {
            "H": torch.from_numpy(np.ascontiguousarray(h)),
            "h": torch.from_numpy(complex_to_channels(h)),
            "condition": condition,
            "scene_id": torch.tensor(record.scene_id, dtype=torch.int64),
            "trajectory_id": torch.tensor(address.trajectory_global, dtype=torch.int64),
            "trajectory_in_shard": torch.tensor(
                address.trajectory_local, dtype=torch.int64
            ),
            "shard_id": torch.tensor(record.shard_id, dtype=torch.int64),
            "snapshot_idx": torch.tensor(t, dtype=torch.int64),
            "subcarrier_idx": torch.tensor(k, dtype=torch.int64),
            "pilot_id": torch.tensor(max(int(raw["pilot_id_raw"]) - 1, 0), dtype=torch.int64),
            "pilot_id_raw": torch.tensor(int(raw["pilot_id_raw"]), dtype=torch.int64),
            "snr_db": torch.tensor(float(raw["snr_db"]), dtype=torch.float32),
            "noise_var": torch.tensor(float(raw["noise_var"]), dtype=torch.float32),
            "cfo_hz": torch.tensor(float(raw["cfo_hz"]), dtype=torch.float32),
            "sample_seed": torch.tensor(int(raw["sample_seed"]), dtype=torch.int64),
            "channel_scale_lin": torch.tensor(
                float(raw["channel_scale_lin"]), dtype=torch.float32
            ),
            "block_id": torch.tensor(int(raw["block_id"]), dtype=torch.int64),
        }
        # Flat aliases ease experiment scripts while `condition` remains the
        # canonical, collate-friendly model input.
        output.update(
            {
                "ckm_scalar": condition["scalar"],
                "ckm_adp": condition["adp"],
                "tx_beam_probability": condition["tx_beam"],
                "rx_beam_probability": condition["rx_beam"],
                "position_xyz": condition["position"],
                "velocity_xyz": condition["velocity"],
                "orientation_deg": condition["orientation"],
                "observed_position_xyz": condition["observed_position"],
                "tf": condition["tf"],
                "state": condition["state"],
                "frequency": condition["frequency"],
                "time": condition["time"],
            }
        )

        if self.include_observation:
            p, pilot_index = self._pilot(
                record.scene_id, int(raw["pilot_id_raw"]), record.num_tx
            )
            noise_mode = self.observation.noise_mode
            selected_noise = float(raw["noise_var"]) if noise_mode == "stored" else None
            selected_snr = float(raw["snr_db"]) if noise_mode == "snr" else None
            if noise_mode == "none":
                selected_noise = 0.0
            noise_seed = (
                derive_seed(
                    self.observation.base_seed,
                    int(raw["sample_seed"]),
                    record.scene_id,
                    k,
                    t,
                    self.epoch if self.sampling == "random_tf" else 0,
                )
                if self.observation.deterministic
                else None
            )
            y, effective_noise = synthesize_observation(
                h, p, noise_var=selected_noise, snr_db=selected_snr, seed=noise_seed
            )
            output.update(
                {
                    "P": torch.from_numpy(np.ascontiguousarray(p)),
                    "p": torch.from_numpy(complex_to_channels(p)),
                    "Y": torch.from_numpy(np.ascontiguousarray(y)),
                    "y": torch.from_numpy(complex_to_channels(y)),
                    "pilot_id": torch.tensor(pilot_index, dtype=torch.int64),
                    "noise_var": torch.tensor(effective_noise, dtype=torch.float32),
                    # Per-slice online measurements intentionally exclude ICI;
                    # use apply_pilot_with_cfo with a full H snapshot when ICI
                    # is part of an experiment.
                    "cfo_applied": torch.tensor(False),
                }
            )
        return output

    def close(self) -> None:
        self._h5_cache.close()
        self._mat_cache.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_h5_cache"] = _H5HandleCache(self.max_open_files)
        state["_mat_cache"] = _MatObjectCache(min(self.max_open_files, 1))
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError, ValueError):
            pass


def seed_worker(worker_id: int) -> None:
    """Top-level, Windows-picklable DataLoader worker seeding function."""

    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


__all__ = ["CKMSRFDataset", "ObservationConfig", "seed_worker"]
