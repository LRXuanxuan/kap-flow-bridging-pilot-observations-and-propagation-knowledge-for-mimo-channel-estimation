"""Fit train-only channel moments for LMMSE/Gaussian-MMSE baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .baselines import ChannelStatisticsAccumulator
from .config import add_config_arguments, load_config_from_args, to_serializable
from .engine import dataloader_kwargs, seed_everything
from .factory import (
    build_dataset,
    load_or_build_index,
    load_or_compute_normalization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit train-only Gaussian channel statistics for classical baselines."
    )
    add_config_arguments(parser)
    parser.add_argument("--protocol-checkpoint", type=Path,
                        help="Fit train-only moments using this prior's saved data config/normalization")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--refresh-normalization", action="store_true")
    return parser


def _default_output(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["run_root"]) / "classical_baselines" / "channel_stats.pt"


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.passes < 1:
        raise ValueError("--passes must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    payload = None
    if args.protocol_checkpoint is not None:
        if args.refresh_normalization:
            raise ValueError("Do not refresh normalization when using --protocol-checkpoint")
        from .evaluate import _load_checkpoint, _config_for_evaluation
        payload = _load_checkpoint(args.protocol_checkpoint)
        config = _config_for_evaluation(args, payload)
    else:
        config = load_config_from_args(args)
    seed = int(config.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    index = load_or_build_index(config, "train", refresh=args.refresh_index)
    if payload is not None:
        from .evaluate import _normalization
        normalization = _normalization(config, payload)
    else:
        normalization = load_or_compute_normalization(
            config, index, refresh=args.refresh_normalization
        )
    dataset = build_dataset(
        config,
        "train",
        index=index,
        normalization=normalization,
        include_observation=False,
    )
    view = dataset
    if args.max_samples is not None and args.max_samples < len(dataset):
        view = Subset(dataset, range(args.max_samples))

    accumulator = ChannelStatisticsAccumulator()
    total = len(view) * args.passes
    progress = tqdm(total=total, desc="fit channel statistics")
    for pass_index in range(args.passes):
        dataset.set_epoch(pass_index)
        # Recreate workers so random_tf observes the new deterministic epoch.
        kwargs = dataloader_kwargs(config, training=True, seed=seed + pass_index)
        kwargs["shuffle"] = False
        loader = DataLoader(view, drop_last=False, **kwargs)
        for batch in loader:
            accumulator.update(batch["H"], batch["scene_id"])
            progress.update(int(batch["H"].shape[0]))
    progress.close()

    groups = accumulator.finalize(shrinkage=args.shrinkage, jitter=args.jitter)
    destination = (args.output or _default_output(config)).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "estimator": "per_receiver_circular_complex_gaussian",
        "protocol_checkpoint": str(args.protocol_checkpoint) if args.protocol_checkpoint else None,
        "split": "train",
        "sampling": str(config.get("data", {}).get("sampling", "random_tf")),
        "passes": int(args.passes),
        "shrinkage": float(args.shrinkage),
        "jitter": float(args.jitter),
        "normalization": None if normalization is None else normalization.to_dict(),
        "groups": groups,
        "config": to_serializable(config),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(destination)

    summary = {
        "artifact": str(destination),
        "estimator": artifact["estimator"],
        "split": artifact["split"],
        "sampling": artifact["sampling"],
        "passes": artifact["passes"],
        "shrinkage": artifact["shrinkage"],
        "jitter": artifact["jitter"],
        "groups": {
            name: {
                "count": int(value["count"]),
                "mean_shape": list(value["mean"].shape),
                "covariance_shape": list(value["covariance"].shape),
            }
            for name, value in groups.items()
        },
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Channel statistics: {destination}")


if __name__ == "__main__":
    main()
