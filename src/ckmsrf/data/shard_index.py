"""Portable shard discovery and trajectory indexing.

The generated CKM-SRF data are split by scene and data split, for example::

    scene_01_train_shard_001.h5
    scene_01_ckm.h5

Only relative file names are persisted.  A saved index can therefore be moved
with the data and reopened with a different ``root`` on another machine.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


_SHARD_RE = re.compile(
    r"^scene_(?P<scene>\d+)_(?P<split>train|val|test)_shard_"
    r"(?P<shard>\d+)\.(?P<ext>h5|hdf5|mat)$",
    flags=re.IGNORECASE,
)
_SCENE_RE = re.compile(
    r"^scene_(?P<scene>\d+)_ckm\.(?P<ext>h5|hdf5|mat)$",
    flags=re.IGNORECASE,
)


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode_attr(value.reshape(-1)[0])
    return str(value)


def canonical_backend(backend: str) -> str:
    """Map configuration-friendly aliases to the two supported backends."""

    value = backend.lower().strip()
    aliases = {
        "h5": "plain_h5",
        "hdf5": "plain_h5",
        "plain_hdf5": "plain_h5",
        "mat": "mat_shard_stream",
        "matlab": "mat_shard_stream",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "plain_h5", "mat_shard_stream"}:
        raise ValueError(
            f"Unsupported backend {backend!r}; expected plain_h5, "
            "mat_shard_stream, or auto."
        )
    return value


@dataclass(frozen=True)
class ShardRecord:
    """Metadata needed to address trajectories without opening a shard."""

    path: str
    scene_id: int
    split: str
    shard_id: int
    num_trajectories: int
    num_subcarriers: int
    num_snapshots: int
    num_rx: int
    num_tx: int
    backend: str = "plain_h5"
    storage_order: str = "matlab_reversed"

    def resolve(self, root: str | Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else Path(root) / path


@dataclass
class DatasetIndex:
    """An index over trajectories in one train/validation/test split."""

    root: str
    split: str
    records: list[ShardRecord]
    scene_files: dict[int, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.root = str(Path(self.root).expanduser().resolve())
        self.split = self.split.lower()
        self.records = list(self.records)
        self.scene_files = {int(k): str(v) for k, v in self.scene_files.items()}
        if not self.records:
            raise ValueError(f"No shard records were supplied for split {self.split!r}.")
        if any(record.split.lower() != self.split for record in self.records):
            raise ValueError("All records in a DatasetIndex must have the same split.")
        self.records.sort(key=lambda r: (r.scene_id, r.shard_id, r.path))
        cumulative: list[int] = []
        total = 0
        for record in self.records:
            if record.num_trajectories <= 0:
                raise ValueError(f"Shard {record.path} contains no trajectories.")
            total += int(record.num_trajectories)
            cumulative.append(total)
        self._cumulative = cumulative

    @property
    def num_trajectories(self) -> int:
        return self._cumulative[-1]

    @property
    def scene_ids(self) -> tuple[int, ...]:
        return tuple(sorted({record.scene_id for record in self.records}))

    def locate(self, trajectory_index: int) -> tuple[ShardRecord, int]:
        """Return ``(record, local_trajectory_index)`` for a global index."""

        size = self.num_trajectories
        if trajectory_index < 0:
            trajectory_index += size
        if not 0 <= trajectory_index < size:
            raise IndexError(
                f"Trajectory index {trajectory_index} is outside [0, {size})."
            )
        record_index = bisect_right(self._cumulative, trajectory_index)
        start = 0 if record_index == 0 else self._cumulative[record_index - 1]
        return self.records[record_index], trajectory_index - start

    def resolve_scene_file(self, scene_id: int) -> Path:
        try:
            value = self.scene_files[int(scene_id)]
        except KeyError as exc:
            raise KeyError(f"No scene CKM file was indexed for scene {scene_id}.") from exc
        path = Path(value)
        return path if path.is_absolute() else Path(self.root) / path

    def save(self, path: str | Path) -> Path:
        """Save JSON using only paths relative to ``root`` where possible."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "split": self.split,
            "records": [asdict(record) for record in self.records],
            "scene_files": {str(k): v for k, v in self.scene_files.items()},
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(
        cls, path: str | Path, *, root: str | Path | None = None
    ) -> "DatasetIndex":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls(
            root=str(Path(root).expanduser() if root is not None else source.parent),
            split=payload["split"],
            records=[ShardRecord(**record) for record in payload["records"]],
            scene_files={int(k): v for k, v in payload["scene_files"].items()},
            schema_version=int(payload.get("schema_version", 1)),
        )


def _h5_dataset(handle: h5py.File, *names: str) -> h5py.Dataset:
    for name in names:
        if name in handle:
            return handle[name]
    raise KeyError(f"None of the datasets {names!r} exists in {handle.filename}.")


def _infer_h5_record(path: Path, match: re.Match[str]) -> ShardRecord:
    with h5py.File(path, "r") as handle:
        h_dataset = _h5_dataset(handle, "/H_real", "/H")
        shape = tuple(int(v) for v in h_dataset.shape)
        if len(shape) != 5:
            raise ValueError(f"{path}: expected a rank-5 H array, got {shape}.")

        explicit_order = _decode_attr(handle.attrs.get("storage_order", "")).lower()
        if explicit_order in {"semantic", "python", "rx_first"}:
            storage_order = "semantic"
        elif explicit_order in {"matlab_reversed", "reversed", "matlab"}:
            storage_order = "matlab_reversed"
        else:
            pilot = _h5_dataset(handle, "/observation/pilot_id")
            num_observations = int(np.prod(pilot.shape))
            if shape[0] == num_observations and shape[-1] != num_observations:
                storage_order = "matlab_reversed"
            elif shape[-1] == num_observations:
                storage_order = "semantic"
            else:
                # MATLAB's high-level HDF5 exporter reverses array dimensions.
                storage_order = "matlab_reversed"

        if storage_order == "matlab_reversed":
            n, snapshots, subcarriers, nt, nr = shape
        else:
            nr, nt, subcarriers, snapshots, n = shape
        # MATLAB's h5writeatt stores scalar attributes as single-element arrays,
        # so accept both 0-D and 1-element attribute shapes here.
        raw_scene = handle.attrs.get("scene_id", match.group("scene"))
        scene_id = int(np.asarray(raw_scene).reshape(-1)[0])

    return ShardRecord(
        path="",  # filled after it is made relative to the dataset root
        scene_id=scene_id,
        split=match.group("split").lower(),
        shard_id=int(match.group("shard")),
        num_trajectories=n,
        num_subcarriers=subcarriers,
        num_snapshots=snapshots,
        num_rx=nr,
        num_tx=nt,
        backend="plain_h5",
        storage_order=storage_order,
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
        result = value[name]
        # hdf5storage preserves each MATLAB scalar-struct wrapper as one or
        # more leading singleton axes.  They are not semantic data axes (for
        # example H_true otherwise becomes [1,Nr,Nt,Nsc,T,N]).  Remove exactly
        # the wrapper axes while preserving legitimate singleton feature axes
        # such as orientation[1,T,N] and pilot_id[1,N].
        if value.ndim > 0 and all(size == 1 for size in value.shape):
            result = result[(0,) * value.ndim]
        while isinstance(result, np.ndarray) and result.size == 1 and result.dtype == object:
            result = result.reshape(-1)[0]
        return result
    if hasattr(value, name):
        return getattr(value, name)
    raise KeyError(f"MATLAB object has no field {name!r}.")


def load_mat_shard(path: str | Path) -> Any:
    """Load one raw MATLAB shard.

    This fallback is intentionally shard-eager: raw v7.3 MATLAB structs are
    reference graphs and cannot be sliced efficiently with plain h5py.  The
    dataset keeps at most a small LRU of such shards in each worker.
    """

    try:
        import hdf5storage  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "mat_shard_stream requires the optional dependency hdf5storage. "
            "Install the project with `pip install -e .[mat]`."
        ) from exc
    loaded = hdf5storage.loadmat(str(path), variable_names=["shard"])
    if "shard" not in loaded:
        raise KeyError(f"{path} does not contain a MATLAB variable named 'shard'.")
    return loaded["shard"]


def _infer_mat_record(path: Path, match: re.Match[str]) -> ShardRecord:
    # A MATLAB v7.3 scalar struct stores each field as an HDF5 reference.  Its
    # target shape can usually be inspected without materializing the ~700 MB
    # H array. Fall back to hdf5storage only for unusual MATLAB encodings.
    raw_shape: tuple[int, ...] | None = None
    try:
        with h5py.File(path, "r") as handle:
            node: h5py.Dataset | h5py.Group = handle["/shard/H_true"]
            while isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype):
                references = np.asarray(node[()]).reshape(-1)
                if references.size == 0:
                    break
                node = handle[references[0]]
            if isinstance(node, h5py.Dataset):
                raw_shape = tuple(int(v) for v in node.shape)
            elif isinstance(node, h5py.Group):
                for candidate in ("value", "real"):
                    if candidate in node and isinstance(node[candidate], h5py.Dataset):
                        raw_shape = tuple(int(v) for v in node[candidate].shape)
                        break
    except (KeyError, OSError, TypeError, ValueError):
        raw_shape = None

    if raw_shape is not None and len(raw_shape) == 5:
        # MATLAB numeric datasets appear reversed to h5py.
        n, snapshots, subcarriers, nt, nr = raw_shape
        scene_id = int(match.group("scene"))
    else:
        shard = load_mat_shard(path)
        h = np.asarray(_field(shard, "H_true"))
        if h.ndim != 5:
            raise ValueError(f"{path}: expected H_true rank 5, got {h.shape}.")
        nr, nt, subcarriers, snapshots, n = (int(v) for v in h.shape)
        try:
            metadata = _field(shard, "metadata")
            scene_id = int(np.asarray(_field(metadata, "sceneId")).reshape(-1)[0])
        except (KeyError, TypeError, ValueError):
            scene_id = int(match.group("scene"))
    return ShardRecord(
        path="",
        scene_id=scene_id,
        split=match.group("split").lower(),
        shard_id=int(match.group("shard")),
        num_trajectories=n,
        num_subcarriers=subcarriers,
        num_snapshots=snapshots,
        num_rx=nr,
        num_tx=nt,
        backend="mat_shard_stream",
        storage_order="semantic",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_shard_index(
    root: str | Path,
    split: str,
    *,
    backend: str = "auto",
    scene_ids: Iterable[int] | None = None,
    require_scene_files: bool = True,
) -> DatasetIndex:
    """Discover scene/shard files and inspect only their lightweight metadata.

    ``backend='auto'`` prefers exported plain HDF5 whenever both HDF5 and raw
    MATLAB files exist.  ``scene_ids=(1, 2, 3)`` can be used to enforce the
    three-scene experiment explicitly.
    """

    data_root = Path(root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, or test.")
    requested_backend = canonical_backend(backend)
    selected_scenes = None if scene_ids is None else {int(v) for v in scene_ids}

    candidates: list[tuple[Path, re.Match[str]]] = []
    for path in data_root.rglob("scene_*_shard_*"):
        match = _SHARD_RE.match(path.name)
        if match is None or match.group("split").lower() != split:
            continue
        scene_id = int(match.group("scene"))
        if selected_scenes is not None and scene_id not in selected_scenes:
            continue
        candidates.append((path, match))

    if requested_backend == "auto":
        has_h5 = any(m.group("ext").lower() in {"h5", "hdf5"} for _, m in candidates)
        active_backend = "plain_h5" if has_h5 else "mat_shard_stream"
    else:
        active_backend = requested_backend
    allowed_ext = {"h5", "hdf5"} if active_backend == "plain_h5" else {"mat"}
    candidates = [item for item in candidates if item[1].group("ext").lower() in allowed_ext]
    if not candidates:
        raise FileNotFoundError(
            f"No {active_backend} files for split {split!r} were found under {data_root}."
        )

    records: list[ShardRecord] = []
    seen: set[tuple[int, int]] = set()
    for path, match in sorted(candidates, key=lambda item: item[0].as_posix()):
        record = (
            _infer_h5_record(path, match)
            if active_backend == "plain_h5"
            else _infer_mat_record(path, match)
        )
        key = (record.scene_id, record.shard_id)
        if key in seen:
            raise ValueError(f"Duplicate scene/shard pair {key} under {data_root}.")
        seen.add(key)
        values = asdict(record)
        values["path"] = _relative(path, data_root)
        records.append(ShardRecord(**values))

    discovered_scenes = {record.scene_id for record in records}
    if selected_scenes is not None and discovered_scenes != selected_scenes:
        missing = sorted(selected_scenes - discovered_scenes)
        raise FileNotFoundError(f"No {split} shards found for requested scenes {missing}.")

    scene_files: dict[int, str] = {}
    for path in data_root.rglob("scene_*_ckm.*"):
        match = _SCENE_RE.match(path.name)
        if match is None or match.group("ext").lower() not in allowed_ext:
            continue
        scene_id = int(match.group("scene"))
        if scene_id in discovered_scenes:
            if scene_id in scene_files:
                raise ValueError(f"Multiple CKM files found for scene {scene_id}.")
            scene_files[scene_id] = _relative(path, data_root)
    if require_scene_files:
        missing = sorted(discovered_scenes - set(scene_files))
        if missing:
            raise FileNotFoundError(f"Missing scene CKM files for scenes {missing}.")

    return DatasetIndex(
        root=str(data_root), split=split, records=records, scene_files=scene_files
    )


__all__ = [
    "DatasetIndex",
    "ShardRecord",
    "build_shard_index",
    "canonical_backend",
    "load_mat_shard",
    "_field",
]
