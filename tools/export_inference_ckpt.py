#!/usr/bin/env python3
"""Export an inference-only KAP-Flow checkpoint.

Why this exists
---------------
The evaluation entry point (``src/ckmsrf/evaluate.py``) restores a checkpoint as
follows (``_load_model_weights``):

1. ``model.load_state_dict(payload["model"], strict=True)``;
2. if ``payload["ema"]["shadow"]`` is present it is copied over the model, so
   **the EMA weights are what the paper numbers use** (``--use-model-weights``
   opts out of step 2).

The training slots (``optimizer`` Adam state, ``scheduler``, ``scaler``,
``rng_state``) are only needed for an exact training resume and account for
roughly half of the 61.6 MB file.  This tool keeps everything evaluation needs
and drops the training-only slots.

Kept by default
---------------
``format_version``, ``model``, ``ema`` (decay / update_after_step / num_updates /
shadow), ``config``, ``normalization``, ``training_state``, ``epoch``,
``global_step``, ``best_metrics``, ``extra``.

Dropped by default
------------------
``optimizer``, ``scheduler``, ``scaler``, ``rng_state``.  Pass ``--keep-rng`` to
retain ``rng_state`` (not needed for evaluation; only useful for RNG audits).

A deterministic provenance marker ``extra["inference_export"]`` records the
source SHA256 and the dropped keys, so an exported file cannot be mistaken for a
resumable training checkpoint.  The marker contains no timestamps: the same
input always produces the same output bytes for a fixed tool version.

Verification (on by default)
----------------------------
* round-trip: reload the exported file and assert ``torch.equal`` for every
  tensor of ``model`` and of ``ema.shadow``, and equality of the JSON-ish
  metadata slots;
* real loader: import ``ckmsrf`` from ``--loader-src`` (default: ``<repo>/src``),
  build the model with the saved config, run ``_load_model_weights`` with the
  default ``prefer_ema=True`` and check that the loaded parameters match
  ``ema.shadow`` tensor-by-tensor, then repeat with ``prefer_ema=False`` to
  check the raw ``model`` slot is intact.

Usage
-----
    python tools/export_inference_ckpt.py best_nmse.pt -o main_straight_ckm.pt
    python tools/export_inference_ckpt.py best_nmse.pt -o out.pt --force --keep-rng
    python tools/export_inference_ckpt.py best_nmse.pt --json-report report.json

The release assets were produced with

    python tools/export_inference_ckpt.py <source .pt> -o <release_asset_name>.pt --force

from the authors' training checkpoints; see ``checkpoints/MANIFEST.json`` for the
per-asset SHA256 of both the original and the exported file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

TOOL_VERSION = 1

#: Slots required by evaluation (plus cheap provenance/metadata slots).
KEEP_KEYS = (
    "format_version",
    "model",
    "ema",
    "training_state",
    "epoch",
    "global_step",
    "best_metrics",
    "config",
    "normalization",
    "extra",
)

#: Slots that only matter for an exact training resume.
DROPPABLE_KEYS = ("optimizer", "scheduler", "scaler", "rng_state")

#: The only slot without which evaluation cannot run.
REQUIRED_KEYS = ("model",)

MARKER_KEY = "inference_export"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Streaming SHA256 of a file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Load a checkpoint on CPU, supporting both modern and legacy torch."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping, got {type(payload)!r}")
    return payload


def build_inference_payload(
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    keep_rng: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Build the reduced payload and return it together with the dropped keys."""

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Source checkpoint is missing required slot(s): {missing}")

    exported: dict[str, Any] = {}
    for key in KEEP_KEYS:
        if key in payload:
            exported[key] = payload[key]

    dropped: list[str] = []
    for key in DROPPABLE_KEYS:
        if key not in payload:
            continue
        if key == "rng_state" and keep_rng:
            exported["rng_state"] = payload["rng_state"]
        else:
            dropped.append(key)

    extra = dict(payload.get("extra") or {})
    extra[MARKER_KEY] = {
        "tool": "tools/export_inference_ckpt.py",
        "tool_version": TOOL_VERSION,
        "source_sha256": source_sha256,
        "kept_keys": [key for key in exported if key != "extra"],
        "dropped_keys": dropped,
        "note": (
            "Inference-only export: training-only slots were removed, so training "
            "cannot be resumed exactly from this file. The evaluation entry point "
            "loads it unchanged."
        ),
    }
    exported["extra"] = extra
    return exported, dropped


def save_atomic(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Serialise to memory, then atomically replace the destination.

    The bytes are produced with ``torch.save`` into a ``BytesIO`` on purpose:
    writing to a *file* embeds the file stem as the zip archive prefix, which
    would make the output depend on the temporary file name (and thus on the
    process id).  Serialising to memory keeps the archive prefix fixed at
    ``archive/`` so the same input always yields the same output bytes.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _assert_tensor_map_equal(
    label: str,
    original: Mapping[str, torch.Tensor],
    exported: Mapping[str, torch.Tensor],
) -> int:
    if set(original) != set(exported):
        missing = sorted(set(original) - set(exported))[:5]
        unexpected = sorted(set(exported) - set(original))[:5]
        raise AssertionError(
            f"{label}: key mismatch, missing={missing}, unexpected={unexpected}"
        )
    for name in original:
        left = original[name]
        right = exported[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise AssertionError(f"{label}/{name}: not a tensor")
        if not torch.equal(left, right):
            raise AssertionError(f"{label}/{name}: tensors differ (torch.equal=False)")
    return len(original)


def _assert_metadata_equal(
    original: Mapping[str, Any], exported: Mapping[str, Any]
) -> list[str]:
    checked: list[str] = []
    for key in (
        "format_version",
        "config",
        "normalization",
        "training_state",
        "epoch",
        "global_step",
        "best_metrics",
    ):
        if key not in original:
            continue
        if exported.get(key) != original.get(key):
            raise AssertionError(f"metadata slot {key!r} changed during export")
        checked.append(key)
    return checked


def verify_round_trip(
    source_payload: Mapping[str, Any], exported_path: str | Path
) -> dict[str, Any]:
    """Reload the exported file and compare it with the source payload."""

    exported = load_checkpoint(exported_path)
    model_tensors = _assert_tensor_map_equal(
        "model", source_payload["model"], exported["model"]
    )
    ema_source = source_payload.get("ema")
    ema_exported = exported.get("ema")
    if isinstance(ema_source, Mapping) and "shadow" in ema_source:
        if not isinstance(ema_exported, Mapping) or "shadow" not in ema_exported:
            raise AssertionError("ema.shadow disappeared during export")
        shadow_tensors = _assert_tensor_map_equal(
            "ema.shadow", ema_source["shadow"], ema_exported["shadow"]
        )
        for key in ("decay", "update_after_step", "num_updates"):
            if key in ema_source and ema_source[key] != ema_exported.get(key):
                raise AssertionError(f"ema.{key} changed during export")
    else:
        shadow_tensors = 0
    metadata = _assert_metadata_equal(source_payload, exported)
    return {
        "model_tensors": model_tensors,
        "ema_shadow_tensors": shadow_tensors,
        "metadata_slots": metadata,
        "torch_equal": True,
    }


def verify_with_loader(exported_path: str | Path, src_dir: str | Path) -> dict[str, Any]:
    """Load the exported file through the real evaluation entry point."""

    src_path = Path(src_dir).expanduser().resolve()
    if not (src_path / "ckmsrf").is_dir():
        raise FileNotFoundError(f"{src_path} does not contain the ckmsrf package")
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from ckmsrf.evaluate import (  # noqa: PLC0415 - optional heavy dependency
        _load_checkpoint,
        _load_model_weights,
        _normalization,
    )
    from ckmsrf.factory import build_model  # noqa: PLC0415

    payload = _load_checkpoint(Path(exported_path))
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise AssertionError("exported checkpoint has no config slot")

    def _loaded_parameters(prefer_ema: bool) -> tuple[int, str]:
        model = build_model(config)
        source = _load_model_weights(model, payload, prefer_ema=prefer_ema)
        state = model.state_dict()
        target = (
            payload["ema"]["shadow"]
            if prefer_ema and isinstance(payload.get("ema"), Mapping)
            else payload["model"]
        )
        compared = 0
        for name, expected in target.items():
            if name not in state:
                continue
            actual = state[name].detach().to(device="cpu")
            expected = expected.detach().to(device="cpu").to(dtype=actual.dtype)
            if not torch.equal(actual, expected):
                raise AssertionError(
                    f"loader mismatch for {name!r} (prefer_ema={prefer_ema})"
                )
            compared += 1
        parameters = sum(p.numel() for p in model.parameters())
        return parameters, compared, source

    ema_parameters, ema_tensors, ema_source = _loaded_parameters(True)
    _, model_tensors, model_source = _loaded_parameters(False)
    stats = _normalization(config, payload)
    if stats is None:
        raise AssertionError("normalization statistics were not resolvable")
    return {
        "weights_source_default": ema_source,
        "ema_parameters": ema_parameters,
        "ema_tensors_checked": ema_tensors,
        "model_tensors_checked": model_tensors,
        "raw_weights_source": model_source,
        "normalization_loaded": True,
        "loader_ok": True,
    }


def _default_loader_src() -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "src"
    return candidate if (candidate / "ckmsrf").is_dir() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a KAP-Flow training checkpoint (best_nmse.pt) to an "
            "inference-only checkpoint that the evaluation entry point loads "
            "unchanged."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="source training checkpoint (.pt)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="destination .pt (default: <input stem>.inference.pt next to the input)",
    )
    parser.add_argument(
        "--keep-rng",
        action="store_true",
        help="keep rng_state (not needed for evaluation)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the torch.equal round-trip check (not recommended)",
    )
    parser.add_argument(
        "--loader-src",
        type=Path,
        default=None,
        help="path to the repository src/ directory for real-loader verification "
        "(default: <repo>/src if present; pass 'none' to skip)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="write the result as JSON to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(f"{source.stem}.inference.pt")
    )
    if destination == source:
        raise ValueError("refusing to overwrite the source checkpoint in place")
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} exists; pass --force to overwrite")

    source_sha256 = sha256_file(source)
    payload = load_checkpoint(source)
    exported, dropped = build_inference_payload(
        payload, source_sha256=source_sha256, keep_rng=bool(args.keep_rng)
    )
    del payload  # release the source tensors before the round-trip load

    save_atomic(exported, destination)
    exported_sha256 = sha256_file(destination)

    report: dict[str, Any] = {
        "input": str(source),
        "output": str(destination),
        "sha256_original": source_sha256,
        "sha256_inference": exported_sha256,
        "size_original_bytes": source.stat().st_size,
        "size_inference_bytes": destination.stat().st_size,
        "dropped_keys": dropped,
        "kept_keys": [key for key in exported if key != "extra"],
        "tool_version": TOOL_VERSION,
    }
    if not args.no_verify:
        source_payload = load_checkpoint(source)
        report["round_trip"] = verify_round_trip(source_payload, destination)
        del source_payload

    loader_src: Path | None
    if args.loader_src is not None and str(args.loader_src).lower() == "none":
        loader_src = None
    elif args.loader_src is not None:
        loader_src = args.loader_src
    else:
        loader_src = _default_loader_src()
    if loader_src is not None:
        report["loader"] = verify_with_loader(destination, loader_src)
    else:
        report["loader"] = {"loader_ok": False, "reason": "src/ not found next to tools/"}

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
