"""Unified, portable training entry point for all CKM-SRF experiments."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import add_config_arguments, load_config_from_args, to_serializable
from .data import NormalizationStats
from .engine import (
    TrainingEngine,
    capture_rng_state,
    create_runtime,
    dataloader_kwargs,
    restore_rng_state,
    seed_everything,
)
from .factory import (
    build_dataset,
    build_model,
    build_objective,
    build_optimizer,
    build_scheduler,
    load_initial_weights,
    load_or_build_index,
    load_or_compute_normalization,
    normalization_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an unconditional or CKM-conditioned GMFlow/SRF prior."
    )
    add_config_arguments(parser)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--refresh-normalization", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _limit_dataset(dataset: Dataset[Any], maximum: int | None) -> Dataset[Any]:
    if maximum is None or int(maximum) >= len(dataset):
        return dataset
    return Subset(dataset, range(max(0, int(maximum))))


def _run_dir_from_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        if path.parent.name == "checkpoints":
            return path.parent.parent
        return path.parent
    if (path / "checkpoints" / "last.pt").is_file():
        return path
    if (path / "last.pt").is_file() and path.name == "checkpoints":
        return path.parent
    raise FileNotFoundError(f"Cannot infer a run directory from {path}")


def _resolve_run_and_resume(
    config: dict[str, Any], runtime: Any, run_id: str | None
) -> tuple[Path, str | Path | None]:
    root = Path(config["paths"]["run_root"])
    experiment = str(config.get("experiment", {}).get("name", "ckmsrf"))
    experiment_root = root / experiment
    resume = config.get("training", {}).get("resume")
    resume_text = "" if resume is None else str(resume).strip()

    if resume_text.lower() == "auto":
        candidates = sorted(
            experiment_root.glob("*/checkpoints/last.pt"),
            key=lambda value: value.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            resume = candidates[0]
            return candidates[0].parent.parent, resume
        resume = None
    elif resume_text and resume_text.lower() not in {"none", "false"}:
        source = Path(resume_text).expanduser()
        if source.is_dir() and (source / "checkpoints" / "last.pt").is_file():
            source = source / "checkpoints" / "last.pt"
        elif source.is_dir() and (source / "last.pt").is_file():
            source = source / "last.pt"
        return _run_dir_from_checkpoint(source), source

    requested = run_id or config.get("training", {}).get("run_id")
    if requested:
        return experiment_root / str(requested), None

    # Distributed workers share a small pointer written by the main process.
    # The pointer is only used during startup; every checkpoint remains inside
    # its self-contained run directory.
    pointer = experiment_root / ".active_run"
    if bool(getattr(runtime, "is_main_process", True)):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed = int(config.get("training", {}).get("seed", 3407))
        candidate = f"{stamp}_seed{seed}"
        suffix = 1
        while (experiment_root / candidate).exists():
            candidate = f"{stamp}_seed{seed}_{suffix:02d}"
            suffix += 1
        experiment_root.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_suffix(".tmp")
        temporary.write_text(candidate, encoding="utf-8")
        os.replace(temporary, pointer)
    runtime.wait_for_everyone()
    selected = pointer.read_text(encoding="utf-8").strip()
    return experiment_root / selected, None


def _training_semantics(config: Mapping[str, Any]) -> tuple[str, str]:
    model_type = str(config.get("model", {}).get("type", "gmflow")).lower()
    model_aliases = {
        "cm": "consistency",
        "velocity_flow": "flow_matching",
        "straight_flow": "straight_flow_matching",
    }
    model_type = model_aliases.get(model_type, model_type)
    loss_type = str(config.get("loss", {}).get("type", "gmflow")).lower()
    loss_aliases = {
        "fm": "flow_matching",
        "velocity_matching": "flow_matching",
        "straight_velocity_matching": "straight_flow_matching",
        "pnp_flow_matching": "straight_flow_matching",
        "denoising_diffusion": "diffusion",
        "edm": "diffusion",
    }
    return model_type, loss_aliases.get(loss_type, loss_type)


def _check_resume_semantics(
    config: Mapping[str, Any], checkpoint: str | Path
) -> None:
    """Prevent same-shaped heads from resuming under a different flow path."""

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        return
    saved_semantics = _training_semantics(payload["config"])
    current_semantics = _training_semantics(config)
    if saved_semantics != current_semantics:
        raise ValueError(
            "Resume checkpoint has incompatible training semantics: "
            f"saved model/loss={saved_semantics}, requested={current_semantics}. "
            "Use a matching config. --resume must never cross additive-Flow "
            "and straight-Flow paths."
        )


def _write_environment(run_dir: Path, runtime: Any) -> None:
    if not bool(getattr(runtime, "is_main_process", True)):
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    destination = run_dir / "environment.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


def _normalization_for_all_processes(
    config: Mapping[str, Any],
    train_index: Any,
    runtime: Any,
    *,
    refresh: bool,
) -> NormalizationStats | None:
    mode = str(
        config.get("data", {}).get("normalization", {}).get("mode", "global_rms")
    ).lower()
    if mode in {"none", "off", "false"}:
        return None
    if bool(getattr(runtime, "is_main_process", True)):
        load_or_compute_normalization(config, train_index, refresh=refresh)
    runtime.wait_for_everyone()
    return NormalizationStats.load(normalization_path(config))


def _reduce_metrics(
    runtime: Any, metrics: Mapping[str, torch.Tensor | float], batch_size: int
) -> dict[str, float]:
    names = sorted(metrics)
    values = []
    device = runtime.device
    for name in names:
        value = metrics[name]
        tensor = value.detach() if torch.is_tensor(value) else torch.tensor(value, device=device)
        values.append(tensor.float().reshape(()))
    row = torch.stack(values + [torch.tensor(float(batch_size), device=device)]).view(1, -1)
    gathered = runtime.gather_for_metrics(row).reshape(-1, row.shape[-1])
    weights = gathered[:, -1]
    denominator = weights.sum().clamp_min(1.0)
    return {
        name: float((gathered[:, index] * weights).sum().div(denominator).cpu())
        for index, name in enumerate(names)
    }


def _scalar_metrics(output: Mapping[str, Any]) -> dict[str, torch.Tensor | float]:
    """Keep scalar diagnostics so objectives can expose method-specific metrics."""

    result: dict[str, torch.Tensor | float] = {}
    for name, value in output.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                result[name] = value
        elif isinstance(value, (int, float)):
            result[name] = float(value)
    return result


@torch.no_grad()
def validate(
    engine: TrainingEngine,
    objective: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model = engine.unwrap_model()
    previous_training = engine.model.training
    rng_state = capture_rng_state()
    validation_seed = int(engine.config.get("training", {}).get("validation_seed", 9102))
    validation_seed += int(getattr(engine.runtime, "process_index", 0))
    seed_everything(validation_seed)
    sums: dict[str, float] = {}
    total = 0
    try:
        engine.model.eval()
        with engine.ema.average_parameters(model):
            for batch_index, batch in enumerate(loader):
                with engine.autocast():
                    output = objective(engine.model, batch)
                batch_size = int(batch["h"].shape[0])
                local = _scalar_metrics(output)
                reduced = _reduce_metrics(engine.runtime, local, batch_size)
                for name, value in reduced.items():
                    sums[name] = sums.get(name, 0.0) + value * batch_size
                total += batch_size
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
    finally:
        restore_rng_state(rng_state)
        engine.model.train(previous_training)
    # Every loop iteration already averages ranks. Weighting by this rank's
    # batch size is exact for equal distributed batch sizes and deterministic
    # validation; gather_for_metrics also removes duplicated tail samples.
    return {name: value / max(total, 1) for name, value in sums.items()}


def _training_total_steps(config: Mapping[str, Any], loader_length: int, processes: int) -> int:
    training = config.get("training", {})
    maximum = training.get("max_steps")
    if maximum is not None:
        return max(1, int(maximum))
    accumulation = int(config.get("hardware", {}).get("grad_accum_steps", 1))
    per_epoch = math.ceil(loader_length / max(accumulation * processes, 1))
    return max(1, per_epoch * int(training.get("epochs", 1)))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config_from_args(args)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if args.dry_run:
        training = config.setdefault("training", {})
        training["max_steps"] = 1
        training["max_train_samples"] = max(
            int(config.get("hardware", {}).get("batch_size", 1)), 1
        )
        training["max_val_samples"] = max(
            int(config.get("hardware", {}).get("eval_batch_size", 1)), 1
        )
        training["max_val_batches"] = 1
        normalization_cfg = config.setdefault("data", {}).setdefault(
            "normalization", {}
        )
        normalization_cfg["stats_file"] = ".ckmsrf_cache/normalization_debug.json"
        normalization_cfg["max_trajectories"] = 2

    seed = int(config.get("training", {}).get("seed", 3407))
    seed_everything(seed)
    runtime = create_runtime(config)
    run_dir, resume = _resolve_run_and_resume(config, runtime, args.run_id)
    if resume is not None:
        _check_resume_semantics(config, resume)
    config["training"]["resume"] = None if resume is None else str(resume)
    config["paths"]["run_dir"] = run_dir
    _write_environment(run_dir, runtime)

    # Build/cache the index once, then reopen it on every process using the
    # current machine's data_root.
    if bool(getattr(runtime, "is_main_process", True)):
        load_or_build_index(config, "train", refresh=args.refresh_index)
        load_or_build_index(config, "val", refresh=args.refresh_index)
    runtime.wait_for_everyone()
    train_index = load_or_build_index(config, "train")
    val_index = load_or_build_index(config, "val")
    normalization = _normalization_for_all_processes(
        config,
        train_index,
        runtime,
        refresh=args.refresh_normalization,
    )

    include_observation = bool(config.get("data", {}).get("include_observation", False))
    train_dataset = build_dataset(
        config,
        "train",
        index=train_index,
        normalization=normalization,
        include_observation=include_observation,
    )
    val_dataset = build_dataset(
        config,
        "val",
        index=val_index,
        normalization=normalization,
        include_observation=include_observation,
    )
    training_cfg = config.get("training", {})
    train_view = _limit_dataset(train_dataset, training_cfg.get("max_train_samples"))
    val_view = _limit_dataset(val_dataset, training_cfg.get("max_val_samples"))
    train_kwargs = dataloader_kwargs(config, training=True, seed=seed)
    val_kwargs = dataloader_kwargs(config, training=False, seed=seed + 1)
    train_loader = DataLoader(train_view, drop_last=True, **train_kwargs)
    val_loader = DataLoader(val_view, drop_last=False, **val_kwargs)

    model = build_model(config)
    if bool(config.get("hardware", {}).get("gradient_checkpointing", False)):
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing(True)
        elif bool(getattr(runtime, "is_main_process", True)):
            print("Warning: this model does not expose gradient checkpointing")
    init_checkpoint = args.init_checkpoint or training_cfg.get("init_checkpoint")
    if init_checkpoint and resume is None:
        missing, skipped = load_initial_weights(model, init_checkpoint, strict=False)
        if bool(getattr(runtime, "is_main_process", True)):
            print(
                f"Initialized from {init_checkpoint}; "
                f"missing={len(missing)}, skipped/unexpected={len(skipped)}"
            )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    objective = build_objective(config)
    optimizer = build_optimizer(config, model)
    total_steps = _training_total_steps(
        config,
        len(train_loader),
        int(getattr(runtime, "num_processes", 1)),
    )
    scheduler = build_scheduler(config, optimizer, total_steps=total_steps)

    with TrainingEngine(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        normalization=normalization,
        run_dir=run_dir,
        runtime=runtime,
    ) as engine:
        train_loader, val_loader = engine.prepare_dataloaders(train_loader, val_loader)
        if resume is not None:
            engine.resume(resume, strict=True)
        if engine.is_main_process:
            print(f"Run directory: {run_dir}")
            print(f"Model parameters: {parameter_count:,}")
            print(
                f"Training trajectories: {len(train_view):,}; "
                f"validation samples: {len(val_view):,}"
            )
            if normalization is not None:
                print(f"Training-only H RMS: {normalization.h_rms:.6g}")

        epochs = int(training_cfg.get("epochs", 1))
        max_steps = training_cfg.get("max_steps")
        log_every = int(training_cfg.get("log_every_steps", 20))
        validate_every = int(training_cfg.get("validate_every_epochs", 1))
        save_every = int(training_cfg.get("save_every_epochs", 1))
        max_val_batches = training_cfg.get("max_val_batches")
        stop = False

        for epoch in range(engine.state.epoch, epochs):
            train_dataset.set_epoch(epoch)
            if hasattr(train_loader, "set_epoch"):
                train_loader.set_epoch(epoch)
            engine.model.train()
            progress = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{epochs}",
                disable=not bool(getattr(runtime, "is_local_main_process", True)),
            )
            for batch in progress:
                with engine.accumulate():
                    with engine.autocast():
                        metrics = objective(engine.model, batch)
                    stepped = engine.backward_and_step(metrics["loss"])
                if not stepped:
                    continue
                if engine.state.global_step % log_every == 0:
                    train_metrics = _reduce_metrics(
                        engine.runtime,
                        _scalar_metrics(metrics),
                        int(batch["h"].shape[0]),
                    )
                    train_metrics["learning_rate"] = float(
                        engine.optimizer.param_groups[0]["lr"]
                    )
                    engine.logger.log(
                        train_metrics,
                        step=engine.state.global_step,
                        split="train",
                        epoch=epoch + 1,
                    )
                    progress.set_postfix(loss=f"{train_metrics['loss']:.4g}")
                if max_steps is not None and engine.state.global_step >= int(max_steps):
                    stop = True
                    break

            engine.finish_epoch(epoch + 1)
            if (epoch + 1) % validate_every == 0 or stop or args.dry_run:
                val_metrics = validate(
                    engine,
                    objective,
                    val_loader,
                    epoch=epoch + 1,
                    max_batches=None if max_val_batches is None else int(max_val_batches),
                )
                engine.logger.log(
                    val_metrics,
                    step=engine.state.global_step,
                    split="val",
                    epoch=epoch + 1,
                )
                if "prior_nll" in val_metrics:
                    engine.update_best(
                        "val_prior_nll",
                        val_metrics["prior_nll"],
                        checkpoint_name="best_nll.pt",
                    )
                if "consistency_loss" in val_metrics:
                    engine.update_best(
                        "val_consistency_loss",
                        val_metrics["consistency_loss"],
                        checkpoint_name="best_consistency.pt",
                    )
                if "reconstruction_nmse" in val_metrics:
                    engine.update_best(
                        "val_reconstruction_nmse",
                        val_metrics["reconstruction_nmse"],
                        checkpoint_name="best_nmse.pt",
                    )
                if engine.is_main_process:
                    summary = " ".join(
                        f"{name}={val_metrics[name]:.6g}"
                        for name in (
                            "prior_nll",
                            "diffusion_loss",
                            "denoising_mse",
                            "flow_matching_loss",
                            "straight_flow_matching_loss",
                            "velocity_mse",
                            "consistency_loss",
                            "endpoint_loss",
                            "reconstruction_nmse",
                        )
                        if name in val_metrics
                    )
                    print(f"val epoch={epoch + 1} {summary}")
            if (epoch + 1) % save_every == 0 or stop:
                engine.save("last.pt")
                if bool(training_cfg.get("keep_epoch_checkpoints", False)):
                    engine.save(f"epoch_{epoch + 1:04d}.pt")
            if args.dry_run or stop:
                break

        engine.save("last.pt")
        engine.logger.flush()
        if engine.is_main_process:
            print(
                f"Training finished at step {engine.state.global_step}; "
                f"best={json.dumps(to_serializable(engine.state.best_metrics), ensure_ascii=False)}"
            )


if __name__ == "__main__":
    main()
