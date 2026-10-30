"""Deterministic LS and Gaussian-statistical baseline evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .baselines import gaussian_lmmse, least_squares, select_statistics
from .config import add_config_arguments, load_config_from_args, to_serializable
from .data import NormalizationStats
from .engine import create_runtime, dataloader_kwargs, seed_everything
from .factory import build_dataset, load_or_build_index, normalization_path
from .physics import generate_observation
from .subspace_metrics import SubspaceRecorder, problem_fingerprints


_METHODS = ("ls", "ml", "lmmse", "gaussian-mmse", "gaussian-map")
_GAUSSIAN_METHODS = {"lmmse", "gaussian-mmse", "gaussian-map"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LS/ML and empirical Gaussian LMMSE/MMSE/MAP baselines."
    )
    add_config_arguments(parser)
    parser.add_argument("--protocol-checkpoint", type=Path,
                        help="Use the learned-prior checkpoint data config and normalization for pairing")
    parser.add_argument("--subspace-metrics", action="store_true")
    parser.add_argument("--subspace-rank-rtol", type=float, default=1e-6)
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=_METHODS, default=["ls", "lmmse"]
    )
    parser.add_argument("--statistics-scope", choices=("global", "scene"), default="global")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--snr-db", nargs="+", type=float,
        default=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lmmse-jitter", type=float, default=1.0e-6)
    parser.add_argument("--refresh-index", action="store_true")
    return parser


def _load_statistics(path: Path | None, required: bool) -> Mapping[str, Any] | None:
    if path is None:
        if required:
            raise ValueError("Gaussian statistical methods require --statistics")
        return None
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, Mapping):
        raise TypeError("Statistics artifact root must be a mapping")
    return artifact


def _normalization(config: Mapping[str, Any]) -> NormalizationStats | None:
    mode = str(config.get("data", {}).get("normalization", {}).get("mode", "global_rms")).lower()
    if mode in {"none", "off", "false"}:
        return None
    path = normalization_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalization file: {path}")
    return NormalizationStats.load(path)


def _check_normalization(
    current: NormalizationStats | None, artifact: Mapping[str, Any] | None
) -> None:
    if artifact is None:
        return
    saved = artifact.get("normalization")
    if current is None and saved is None:
        return
    if current is None or not isinstance(saved, Mapping):
        raise ValueError("Dataset and statistics normalization modes differ")
    saved_rms = float(saved.get("h_rms", float("nan")))
    if not torch.isclose(
        torch.tensor(current.h_rms), torch.tensor(saved_rms), rtol=1.0e-6, atol=0.0
    ):
        raise ValueError(
            f"Statistics h_rms={saved_rms} differs from dataset h_rms={current.h_rms}"
        )


def _to_db(value: float) -> float:
    return 10.0 * torch.log10(torch.tensor(max(value, 1.0e-15))).item()


def _estimate(
    method: str,
    observation: torch.Tensor,
    pilot: torch.Tensor,
    noise_var: torch.Tensor,
    scene_ids: torch.Tensor,
    statistics: Mapping[str, Any] | None,
    *,
    scope: str,
    jitter: float,
) -> torch.Tensor:
    if method in {"ls", "ml"}:
        return least_squares(observation, pilot)
    if statistics is None:
        raise ValueError(f"{method} requires statistics")
    mean, covariance = select_statistics(statistics, scene_ids, scope=scope)
    return gaussian_lmmse(
        observation,
        pilot,
        noise_var,
        mean,
        covariance,
        jitter=jitter,
    )


def _rows_from_sums(
    method: str,
    snr_db: float,
    sums: Mapping[int, float],
    residual_sums: Mapping[int, float],
    counts: Mapping[int, int],
    elapsed: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scene_nmse = []
    for scene in sorted(counts):
        linear = sums[scene] / max(counts[scene], 1)
        scene_nmse.append(linear)
        rows.append({
            "method": method,
            "snr_db": float(snr_db),
            "scene_id": scene,
            "num_samples": counts[scene],
            "nmse_linear": linear,
            "nmse_db": _to_db(linear),
            "normalized_measurement_residual": residual_sums[scene] / max(counts[scene], 1),
            "seconds": elapsed,
        })
    total_count = sum(counts.values())
    rows.extend([
        {
            "method": method,
            "snr_db": float(snr_db),
            "scene_id": "macro",
            "num_samples": total_count,
            "nmse_linear": sum(scene_nmse) / max(len(scene_nmse), 1),
            "nmse_db": _to_db(sum(scene_nmse) / max(len(scene_nmse), 1)),
            "normalized_measurement_residual": None,
            "seconds": elapsed,
        },
        {
            "method": method,
            "snr_db": float(snr_db),
            "scene_id": "overall",
            "num_samples": total_count,
            "nmse_linear": sum(sums.values()) / max(total_count, 1),
            "nmse_db": _to_db(sum(sums.values()) / max(total_count, 1)),
            "normalized_measurement_residual": sum(residual_sums.values()) / max(total_count, 1),
            "seconds": elapsed,
        },
    ])
    return rows


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    methods = list(dict.fromkeys(args.methods))
    protocol_payload = None
    if args.protocol_checkpoint is not None:
        from .evaluate import _load_checkpoint, _config_for_evaluation
        protocol_payload = _load_checkpoint(args.protocol_checkpoint)
        config = _config_for_evaluation(args, protocol_payload)
    else:
        config = load_config_from_args(args)
    seed = int(config.get("training", {}).get("validation_seed", 9102))
    seed_everything(seed)
    statistics_path = None if args.statistics is None else args.statistics.expanduser().resolve()
    statistics = _load_statistics(
        statistics_path, any(method in _GAUSSIAN_METHODS for method in methods)
    )
    if protocol_payload is not None:
        from .evaluate import _normalization as checkpoint_normalization
        normalization = checkpoint_normalization(config, protocol_payload)
        if statistics is not None:
            from .factory import parse_scene_ids
            saved_scenes = statistics.get("config", {}).get("data", {}).get("train_scenes")
            current_scenes = config.get("data", {}).get("train_scenes")
            if statistics.get("split") != "train" or saved_scenes is None or current_scenes is None:
                raise ValueError("Paired LMMSE requires recorded train-only statistics and training scenes")
            if set(parse_scene_ids(saved_scenes)) != set(parse_scene_ids(current_scenes)):
                raise ValueError("LMMSE statistics training scenes differ from the reference prior")
    else:
        normalization = _normalization(config)
    _check_normalization(normalization, statistics)
    runtime = create_runtime(config)
    if args.subspace_metrics and int(getattr(runtime, "num_processes", 1)) != 1:
        raise ValueError("--subspace-metrics requires single-process classical evaluation")
    index = load_or_build_index(config, args.split, refresh=args.refresh_index)
    dataset = build_dataset(
        config, args.split, index=index, normalization=normalization,
        include_observation=True,
    )
    view = dataset
    if args.max_samples is not None and args.max_samples < len(dataset):
        view = Subset(dataset, range(max(0, args.max_samples)))
    loader = DataLoader(
        view, drop_last=False,
        **dataloader_kwargs(config, training=False, seed=seed),
    )
    loader = runtime.prepare(loader)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(config["paths"]["run_root"]) / "classical_baselines" / f"{args.split}_{stamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    if bool(getattr(runtime, "is_main_process", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
    runtime.wait_for_everyone()
    all_rows: list[dict[str, Any]] = []
    recorders = {}
    if args.subspace_metrics:
        for method in methods:
            (output_dir / method).mkdir(parents=True, exist_ok=True)
            recorders[method] = SubspaceRecorder(output_dir / method, rtol=args.subspace_rank_rtol)

    for snr_index, snr_db in enumerate(args.snr_db):
        method_sums = {method: {} for method in methods}
        method_residuals = {method: {} for method in methods}
        method_counts = {method: {} for method in methods}
        method_seconds = {method: 0.0 for method in methods}
        progress = tqdm(
            loader, desc=f"SNR {snr_db:g} dB",
            disable=not bool(getattr(runtime, "is_local_main_process", True)),
        )
        for batch_index, batch in enumerate(progress):
            truth = batch["H"].to(runtime.device)
            pilot = batch["P"].to(runtime.device)
            generator_device = runtime.device if runtime.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(seed + 100000 * snr_index + batch_index)
            observation, noise_var = generate_observation(
                truth,
                pilot,
                snr_db=torch.full(
                    (truth.shape[0],), float(snr_db), device=runtime.device,
                    dtype=truth.real.dtype,
                ),
                generator=generator,
            )
            scene_ids = batch["scene_id"].to(runtime.device)
            fingerprints = (problem_fingerprints(truth, pilot, observation, noise_var)
                            if recorders else None)
            for method in methods:
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                start = time.perf_counter()
                estimate = _estimate(
                    method, observation, pilot, noise_var, scene_ids, statistics,
                    scope=args.statistics_scope, jitter=args.lmmse_jitter,
                )
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                method_seconds[method] += time.perf_counter() - start
                if method in recorders:
                    recorders[method].add(estimate, truth, pilot, batch,
                        snr_db=snr_db, iterations=0, fingerprints=fingerprints)
                numerator = (estimate - truth).abs().square().flatten(1).sum(dim=1)
                denominator = truth.abs().square().flatten(1).sum(dim=1).clamp_min(1.0e-12)
                nmse = numerator / denominator
                residual = (torch.bmm(estimate, pilot) - observation).abs().square()
                residual = residual.flatten(1).mean(dim=1) / noise_var.clamp_min(1.0e-12)
                gathered = runtime.gather_for_metrics((nmse, residual, scene_ids))
                if bool(getattr(runtime, "is_main_process", True)):
                    gathered_nmse, gathered_residual, gathered_scene = gathered
                    for value, residual_value, scene in zip(
                        gathered_nmse.detach().cpu().tolist(),
                        gathered_residual.detach().cpu().tolist(),
                        gathered_scene.detach().cpu().tolist(),
                    ):
                        scene = int(scene)
                        method_sums[method][scene] = method_sums[method].get(scene, 0.0) + float(value)
                        method_residuals[method][scene] = method_residuals[method].get(scene, 0.0) + float(residual_value)
                        method_counts[method][scene] = method_counts[method].get(scene, 0) + 1
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break

        runtime.wait_for_everyone()
        if bool(getattr(runtime, "is_main_process", True)):
            for method in methods:
                rows = _rows_from_sums(
                    method, snr_db, method_sums[method], method_residuals[method],
                    method_counts[method], method_seconds[method],
                )
                all_rows.extend(rows)
                overall = next(row for row in rows if row["scene_id"] == "overall")
                print(f"{method} SNR {snr_db:g} dB: overall NMSE={overall['nmse_db']:.3f} dB")

    if bool(getattr(runtime, "is_main_process", True)):
        combined = output_dir / "metrics_all.csv"
        with combined.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        for method in methods:
            if method in recorders:
                recorders[method].finish()
            method_dir = output_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            method_rows = [{key: value for key, value in row.items() if key != "method"}
                           for row in all_rows if row["method"] == method]
            with (method_dir / "metrics.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
                writer.writeheader()
                writer.writerows(method_rows)
            metadata = {
                "method": method,
                "equivalence": (
                    "ML equals LS under circular AWGN" if method == "ml" else
                    "For the fitted Gaussian prior, LMMSE = MMSE = MAP"
                    if method in _GAUSSIAN_METHODS else None
                ),
                "statistics": None if statistics_path is None else str(statistics_path),
                "statistics_scope": args.statistics_scope,
                "protocol_checkpoint": str(args.protocol_checkpoint) if args.protocol_checkpoint else None,
                "subspace_metrics": bool(args.subspace_metrics),
                "subspace_rank_rtol": args.subspace_rank_rtol,
                "normalization": None if normalization is None else normalization.to_dict(),
                "lmmse_jitter": args.lmmse_jitter,
                "split": args.split,
                "config": to_serializable(config),
                "rows": method_rows,
            }
            (method_dir / "results.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(f"Classical baseline results: {output_dir}")
    runtime.end_training()


if __name__ == "__main__":
    main()
