#!/usr/bin/env python3
"""Prepare CKM-SRF HDF5 output for the KAP-Flow data pipeline.

KAP-Flow consumes the MATLAB generator output directly as plain HDF5
(``data.backend: plain_h5``), so no ``.mat`` conversion is required.  This
script closes the remaining practical gaps between ``export_*_to_hdf5.m`` and
the loader in ``src/ckmsrf/data/``:

1. **Root attributes.** MATLAB's ``h5writeatt`` stores ``scene_id`` as a
   one-element array, while ``ckmsrf.data.shard_index`` expects a Python
   scalar.  The exported files also omit ``storage_order``; the loader falls
   back to a shape heuristic that is ambiguous when the number of samples
   equals ``N_r``.  This script rewrites ``scene_id`` and records
   ``storage_order`` explicitly.
2. **Validation.** Every shard is checked against the datasets the training
   and evaluation loaders actually read.
3. **Portable split indices.** ``indices/{split}.json`` files are built with
   ``ckmsrf.data.build_shard_index``; record paths are stored relative to the
   data root.
4. **Train-only normalization.** ``normalization.json`` is computed with the
   same streaming code as training (``ckmsrf.data.compute_normalization``).

Run from anywhere; the repository ``src/`` directory is added to
``sys.path`` automatically::

    export CKMSRF_DATA_ROOT=/path/to/ckm_srf_64x16_h5
    python data_generation/python/prepare_kapflow_data.py \
        --data-root "$CKMSRF_DATA_ROOT"

PowerShell::

    $env:CKMSRF_DATA_ROOT = "<data root>"
    python data_generation/python/prepare_kapflow_data.py --data-root $env:CKMSRF_DATA_ROOT

Cross-scene folds use the same data root with a different source-scene set and
normalization file name.  For ``holdout_01`` (train on scenes 02/03, test on
scene 01), which must match
``main/configs/threefold/holdout_01_ckm.yaml``::

    python data_generation/python/prepare_kapflow_data.py \
        --data-root "$CKMSRF_DATA_ROOT" --scenes 2 3 \
        --normalization-file normalization_cross_scene_01.json

The script is idempotent: attributes are rewritten only when needed, an
existing normalization file is reused unless ``--refresh-normalization`` is
given, and ``--check-only`` reports whether any preparation is required
without writing anything.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if (_SRC / "ckmsrf").is_dir():
    sys.path.insert(0, str(_SRC))

try:
    import h5py
except ImportError:  # pragma: no cover - depends on the user's environment
    h5py = None  # type: ignore[assignment]

SCENE_RE = re.compile(r"^scene_(?P<scene>\d+)_ckm\.(?P<ext>h5|hdf5)$", re.IGNORECASE)
SHARD_RE = re.compile(
    r"^scene_(?P<scene>\d+)_(?P<split>train|val|test)_shard_"
    r"(?P<shard>\d+)\.(?P<ext>h5|hdf5)$",
    re.IGNORECASE,
)

# Datasets the KAP-Flow plain_h5 loader reads from a shard (see
# src/ckmsrf/data/lazy_dataset.py: _plain_sample, _read_time, _read_feature).
REQUIRED_SHARD_DATASETS = (
    "channel_scale_lin",
    "state/position_xyz",
    "state/velocity_xyz",
    "state/ue_orientation_deg",
    "state/time_sec",
    "state/block_id",
    "context/observed_position_xyz",
    "context/scalar_feature",
    "context/angular_delay_power",
    "context/tx_beam_probability",
    "context/rx_beam_probability",
    "context/grid_indices",
    "context/grid_weights",
    "observation/pilot_id",
    "observation/snr_db",
    "observation/cfo_hz",
    "observation/noise_var",
    "sample_seed",
)

# Datasets required in a scene file. Only /pilot/P is read by the KAP-Flow
# loader (to build online observations); the rest is the offline CKM context.
REQUIRED_SCENE_COMPLEX = ("pilot/P",)
REQUIRED_SCENE_DATASETS = (
    "codebook/Ftx",
    "codebook/Wrx",
    "ckm/grid_xy",
    "ckm/scalar_feature",
    "ckm/angular_delay_power",
    "ckm/tx_beam_probability",
    "ckm/rx_beam_probability",
    "split/assignment",
)

STORAGE_ORDER_ALIASES = {
    "semantic": "semantic",
    "python": "semantic",
    "rx_first": "semantic",
    "matlab_reversed": "matlab_reversed",
    "reversed": "matlab_reversed",
    "matlab": "matlab_reversed",
}
SCALAR_FEATURE_WIDTH = 6  # LoS, pathloss mean/std, mean delay, RMS delay, count
PAPER_ADP_WIDTH = 16 * 16 * 12


class Report:
    """Tiny collector so that --check-only and the fixing mode share output."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.pending: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"[ERROR] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN]  {message}")

    def fix(self, message: str) -> None:
        print(f"[FIX]   {message}")

    def pending_change(self, message: str) -> None:
        self.pending.append(message)
        print(f"[TODO]  {message}")

    def info(self, message: str) -> None:
        print(message)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _complex_dataset(handle: "h5py.File", base: str) -> str | None:
    """Return the actual HDF5 path holding a (possibly split) complex array."""

    if f"{base}_real" in handle:
        return f"{base}_real"
    if base in handle:
        return base
    return None


def _requires_scalar_rewrite(value: Any) -> bool:
    """True if a root attribute would break ``int(handle.attrs[...])``."""

    if value is None:
        return True
    ndim = getattr(value, "ndim", 0)
    return int(ndim) != 0


def _infer_storage_order(handle: "h5py.File", shape: tuple[int, ...]) -> tuple[str, str]:
    """Return ``(storage_order, explanation)`` for a rank-5 /H dataset.

    Mirrors the fallback in ``ckmsrf.data.shard_index._infer_h5_record`` and
    additionally flags the ambiguous ``N == N_r`` case.
    """

    raw = _decode(handle.attrs.get("storage_order", "")).strip().lower()
    if raw in STORAGE_ORDER_ALIASES:
        return STORAGE_ORDER_ALIASES[raw], "root attribute"

    pilot = handle.get("observation/pilot_id")
    if pilot is None:
        return "matlab_reversed", "MATLAB default (no /observation/pilot_id)"
    try:
        num_observations = math.prod(int(v) for v in pilot.shape)
    except (TypeError, ValueError):
        return "matlab_reversed", "MATLAB default (unreadable pilot shape)"
    first, last = shape[0], shape[-1]
    if first == num_observations and last != num_observations:
        return "matlab_reversed", "shape heuristic"
    if last == num_observations and first != num_observations:
        return "semantic", "shape heuristic"
    if first == num_observations == last:
        return "matlab_reversed", "ambiguous N == N_r; MATLAB default assumed"
    return "matlab_reversed", "MATLAB default (non-identifiable shape)"


def _logical_dims(shape: tuple[int, ...], storage_order: str) -> dict[str, int]:
    if storage_order == "matlab_reversed":
        n, snapshots, subcarriers, nt, nr = shape
    else:
        nr, nt, subcarriers, snapshots, n = shape
    return {
        "num_trajectories": int(n),
        "num_snapshots": int(snapshots),
        "num_subcarriers": int(subcarriers),
        "num_tx": int(nt),
        "num_rx": int(nr),
    }


def _width(handle: "h5py.File", path: str, storage_order: str) -> int | None:
    """Logical feature width of a per-snapshot dataset."""

    dataset = handle.get(path)
    if dataset is None or dataset.ndim < 1:
        return None
    return int(dataset.shape[-1] if storage_order == "matlab_reversed" else dataset.shape[0])


def inspect_and_patch_shard(
    path: Path, report: Report, *, check_only: bool
) -> dict[str, Any] | None:
    """Validate one shard, return its metadata, and optionally patch attrs."""

    match = SHARD_RE.match(path.name)
    if match is None:
        return None
    filename_scene = int(match.group("scene"))

    with h5py.File(path, "r") as handle:
        h_path = _complex_dataset(handle, "H")
        if h_path is None:
            report.error(
                f"{path}: missing /H_real (or /H); keep cfg.output.storeTrueChannel = true"
            )
            return None
        h_shape = tuple(int(v) for v in handle[h_path].shape)
        if len(h_shape) != 5:
            report.error(f"{path}: expected rank-5 /H, got {h_shape}")
            return None

        for name in REQUIRED_SHARD_DATASETS:
            if _complex_dataset(handle, name) is None:
                report.error(f"{path}: missing required dataset /{name}")

        storage_order, explanation = _infer_storage_order(handle, h_shape)
        dims = _logical_dims(h_shape, storage_order)
        scene_id_attr = handle.attrs.get("scene_id")
        needs_scene_id = _requires_scalar_rewrite(scene_id_attr)
        needs_storage_order = (
            _decode(handle.attrs.get("storage_order", "")).strip().lower()
            not in STORAGE_ORDER_ALIASES
        )

        report.info(
            f"[shard] scene {filename_scene:02d} {match.group('split')} "
            f"shard {int(match.group('shard')):03d}: raw /H {h_shape} -> "
            f"(N_r,N_t,N_f,N_s,N)=({dims['num_rx']},{dims['num_tx']},"
            f"{dims['num_subcarriers']},{dims['num_snapshots']},"
            f"{dims['num_trajectories']}); storage_order={storage_order} "
            f"({explanation})"
        )

        nt, nr = dims["num_tx"], dims["num_rx"]
        checks = {
            "context/scalar_feature": (SCALAR_FEATURE_WIDTH, "CKM scalar feature"),
            "context/tx_beam_probability": (nt, "tx beam feature"),
            "context/rx_beam_probability": (nr, "rx beam feature"),
            "context/observed_position_xyz": (3, "observed position"),
            "state/position_xyz": (3, "position"),
            "state/velocity_xyz": (3, "velocity"),
            "state/ue_orientation_deg": (1, "orientation"),
        }
        for name, (expected, label) in checks.items():
            width = _width(handle, name, storage_order)
            if width is not None and width != expected:
                report.error(
                    f"{path}: {label} width {width}, expected {expected} (/{(name)})"
                )
        adp_width = _width(handle, "context/angular_delay_power", storage_order)
        if adp_width is not None and adp_width != PAPER_ADP_WIDTH:
            report.warn(
                f"{path}: ADP width {adp_width}; the paper configuration uses "
                f"16x16x12={PAPER_ADP_WIDTH}"
            )

        meta: dict[str, Any] = {
            "path": path,
            "scene": filename_scene,
            "split": match.group("split").lower(),
            "shard": int(match.group("shard")),
            "storage_order": storage_order,
            "dims": dims,
            "needs_scene_id": needs_scene_id,
            "needs_storage_order": needs_storage_order,
        }

    if check_only:
        if needs_scene_id:
            report.pending_change(f"{path}: rewrite scene_id root attribute as a scalar")
        if needs_storage_order:
            report.pending_change(f"{path}: record storage_order={storage_order}")
        return meta

    if needs_scene_id or needs_storage_order:
        with h5py.File(path, "r+") as handle:
            if needs_scene_id:
                handle.attrs["scene_id"] = int(filename_scene)
                report.fix(f"{path}: wrote scalar scene_id={filename_scene}")
            if needs_storage_order:
                handle.attrs["storage_order"] = storage_order
                report.fix(f"{path}: wrote storage_order={storage_order}")
    return meta


def inspect_and_patch_scene(
    path: Path, report: Report, *, check_only: bool
) -> dict[str, Any] | None:
    match = SCENE_RE.match(path.name)
    if match is None:
        return None
    filename_scene = int(match.group("scene"))
    with h5py.File(path, "r") as handle:
        for name in REQUIRED_SCENE_COMPLEX:
            if _complex_dataset(handle, name) is None:
                report.error(f"{path}: missing required dataset /{name}")
        for name in REQUIRED_SCENE_DATASETS:
            if _complex_dataset(handle, name) is None:
                report.error(f"{path}: missing required dataset /{name}")

        pilot_path = _complex_dataset(handle, "pilot/P")
        pilot_shape = tuple(int(v) for v in handle[pilot_path].shape) if pilot_path else None
        summary = f"[scene] scene {filename_scene:02d}: raw /pilot/P {pilot_shape}"
        grid = handle.get("ckm/grid_xy")
        if grid is not None:
            summary += (
                f"; CKM grid {tuple(int(v) for v in grid.shape)}"
                " (raw MATLAB order -> (2, num_nodes))"
            )
        report.info(summary)

        scene_id_attr = handle.attrs.get("scene_id")
        needs_scene_id = _requires_scalar_rewrite(scene_id_attr)
        meta: dict[str, Any] = {
            "path": path,
            "scene": filename_scene,
            "pilot_shape": pilot_shape,
            "needs_scene_id": needs_scene_id,
        }

    if check_only:
        if needs_scene_id:
            report.pending_change(f"{path}: rewrite scene_id root attribute as a scalar")
        return meta
    if needs_scene_id:
        with h5py.File(path, "r+") as handle:
            handle.attrs["scene_id"] = int(filename_scene)
            report.fix(f"{path}: wrote scalar scene_id={filename_scene}")
    return meta


def _discover(root: Path) -> tuple[list[Path], list[Path]]:
    shards: list[Path] = []
    scenes: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
            continue
        if SHARD_RE.match(path.name):
            shards.append(path)
        elif SCENE_RE.match(path.name):
            scenes.append(path)
    return shards, scenes


def _prepare_indices(
    root: Path, splits: list[str], scenes: list[int], report: Report
) -> dict[str, Any]:
    from ckmsrf.data import build_shard_index

    indices: dict[str, Any] = {}
    index_dir = root / "indices"
    for split in splits:
        try:
            index = build_shard_index(
                root, split, backend="plain_h5", scene_ids=tuple(scenes)
            )
        except FileNotFoundError as exc:
            report.warn(f"split {split!r}: skipped ({exc})")
            continue
        destination = index_dir / f"{split}.json"
        index.save(destination)
        indices[split] = index
        report.info(
            f"[index] {split}: {len(index.records)} shard(s), "
            f"{index.num_trajectories} trajectory/record(s) -> {destination}"
        )
    if indices:
        report.info(
            "[note]  index records store paths relative to the data root; load them "
            "with DatasetIndex.load(path, root=<data root>). Training/evaluation "
            "resolve the root automatically."
        )
    return indices


def _prepare_normalization(
    root: Path, indices: dict[str, Any], args: argparse.Namespace, report: Report
) -> None:
    from ckmsrf.data import NormalizationStats, compute_normalization

    train_index = indices.get("train")
    if train_index is None:
        report.warn("no train index was built; normalization statistics were not computed")
        return
    destination = root / args.normalization_file
    stats = None
    if destination.is_file() and not args.refresh_normalization:
        stats = NormalizationStats.load(destination)
        if (
            args.max_trajectories is None
            and stats.num_trajectories != train_index.num_trajectories
        ):
            report.warn(
                f"{destination} covers {stats.num_trajectories} trajectories but the "
                f"train split has {train_index.num_trajectories}; recomputing"
            )
            stats = None
    if stats is not None:
        report.info(
            f"[norm]  reused {destination}: h_rms={stats.h_rms:.8g}, "
            f"trajectories={stats.num_trajectories}, split={stats.split}"
        )
    else:
        stats = compute_normalization(
            train_index,
            chunk_trajectories=args.chunk_trajectories,
            max_trajectories=args.max_trajectories,
        )
        stats.save(destination)
        report.info(
            f"[norm]  wrote {destination}: h_rms={stats.h_rms:.8g}, "
            f"trajectories={stats.num_trajectories}, "
            f"coefficients={stats.num_channel_coefficients}"
        )
    for name in ("scalar", "adp", "position", "velocity", "observed_position"):
        width = len(stats.feature_mean.get(name, []))
        if width:
            report.info(f"        feature {name}: width {width}")
    adp_width = len(stats.feature_mean.get("adp", []))
    if adp_width and adp_width != PAPER_ADP_WIDTH:
        report.warn(
            f"ADP width {adp_width} differs from the paper configuration "
            f"(16x16x12={PAPER_ADP_WIDTH}); check model.condition.adp_shape "
            "in the experiment YAML"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate generator HDF5 output, patch MATLAB root attributes, build "
            "portable split indices and train-only normalization statistics."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing scene_*_ckm.h5 and scene_*_shard_*.h5",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Scene ids to index (default: 1 2 3, matching the paper)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--normalization-file",
        default="normalization.json",
        help="Statistics file name below the data root (default: normalization.json)",
    )
    parser.add_argument("--chunk-trajectories", type=int, default=8)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit statistics to the first K train trajectories",
    )
    parser.add_argument(
        "--refresh-normalization",
        action="store_true",
        help="Recompute even if the statistics file already exists",
    )
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate only; exit 1 if preparation would change files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] data root not found: {root}")
        return 1
    if h5py is None:
        print("[ERROR] h5py is required; install the repository requirements first.")
        return 1

    report = Report()
    shards, scenes = _discover(root)
    if not shards:
        print(f"[ERROR] no scene_*_shard_*.h5 files found under {root}")
        return 1
    if not scenes:
        report.error(
            "no scene_*_ckm.h5 file found; export the scene packages with "
            "export_all_to_hdf5.m"
        )

    report.info(f"Data root: {root}")
    report.info(f"Found {len(scenes)} scene file(s) and {len(shards)} shard(s)")

    scene_meta: dict[int, dict[str, Any]] = {}
    for path in scenes:
        meta = inspect_and_patch_scene(path, report, check_only=args.check_only)
        if meta is not None:
            scene_meta[meta["scene"]] = meta
    shard_meta: list[dict[str, Any]] = []
    for path in shards:
        meta = inspect_and_patch_shard(path, report, check_only=args.check_only)
        if meta is not None:
            shard_meta.append(meta)

    found: dict[tuple[int, str], int] = {}
    for meta in shard_meta:
        key = (meta["scene"], meta["split"])
        found[key] = found.get(key, 0) + 1
    for scene in args.scenes:
        for split in args.splits:
            if found.get((scene, split), 0) == 0:
                report.warn(f"scene {scene:02d} split {split}: no shard file found")

    if report.errors:
        print("\nPreparation failed: fix the errors above and re-run.")
        return 1
    if args.check_only:
        if report.pending:
            print(
                "\nCheck-only: files need preparation before use by the KAP-Flow "
                "loader (run again without --check-only)."
            )
            return 1
        print(
            "\nCheck-only: exported HDF5 files already satisfy the KAP-Flow "
            "plain_h5 contract."
        )
        return 0

    indices = _prepare_indices(root, args.splits, args.scenes, report)
    if not args.skip_normalization:
        _prepare_normalization(root, indices, args, report)

    print("\nNext steps (from the repository root):")
    print('  export CKMSRF_DATA_ROOT="%s"' % root)
    print("  python -m ckmsrf.train \\")
    print("      --config configs/experiments/ckm_straight_flow_prior.yaml \\")
    print('      --data-root "$CKMSRF_DATA_ROOT"')
    print("PowerShell:")
    print('  $env:CKMSRF_DATA_ROOT = "%s"' % root)
    print(
        "  python -m ckmsrf.train --config "
        "configs/experiments/ckm_straight_flow_prior.yaml "
        "--data-root $env:CKMSRF_DATA_ROOT"
    )
    print(
        "\nThe training/evaluation loader can also build its tagged index cache "
        "and recompute missing statistics on its own; the files written here "
        "make that preparation explicit and portable."
    )
    if report.warnings:
        print(f"\nDone with {len(report.warnings)} warning(s); see the log above.")
    else:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
