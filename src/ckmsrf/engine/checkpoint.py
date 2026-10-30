"""Atomic, device-portable training checkpoints."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ckmsrf.config import to_serializable

from .reproducibility import capture_rng_state, restore_rng_state


@dataclass
class TrainingState:
    """Small mutable state whose epoch denotes the next epoch to run."""

    epoch: int = 0
    global_step: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, state: Mapping[str, Any] | None) -> "TrainingState":
        state = state or {}
        return cls(
            epoch=int(state.get("epoch", 0)),
            global_step=int(state.get("global_step", state.get("step", 0))),
            best_metrics={
                str(key): float(value)
                for key, value in state.get("best_metrics", {}).items()
            },
        )

    def is_improved(self, name: str, value: float, mode: str = "min") -> bool:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        current = self.best_metrics.get(name)
        improved = current is None or (
            value < current if mode == "min" else value > current
        )
        if improved:
            self.best_metrics[name] = float(value)
        return improved


def _unwrap_model(model: nn.Module, runtime: Any | None) -> nn.Module:
    if runtime is not None and hasattr(runtime, "unwrap_model"):
        return runtime.unwrap_model(model)
    return model.module if hasattr(model, "module") else model


def _state_or_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "state_dict"):
        return value.state_dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move loaded optimizer tensors after a CPU-mapped portable load."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class CheckpointManager:
    FORMAT_VERSION = 1

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str | os.PathLike[str]) -> Path:
        path = Path(name)
        if not path.is_absolute():
            path = self.directory / path
        if not path.suffix:
            path = path.with_suffix(".pt")
        return path

    def resolve_resume(self, resume: str | os.PathLike[str] | None) -> Path | None:
        if resume is None or str(resume).strip().lower() in {"", "none", "false"}:
            return None
        if str(resume).strip().lower() == "auto":
            candidate = self.path_for("last.pt")
            return candidate if candidate.is_file() else None
        candidate = Path(resume)
        if candidate.is_dir():
            candidate = candidate / "last.pt"
        elif not candidate.is_absolute():
            local_candidate = self.directory / candidate
            if local_candidate.exists():
                candidate = local_candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
        return candidate.resolve()

    def save(
        self,
        name: str,
        *,
        model: nn.Module,
        ema: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
        state: TrainingState,
        config: Mapping[str, Any],
        normalization: Any,
        runtime: Any | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        if runtime is not None and hasattr(runtime, "wait_for_everyone"):
            runtime.wait_for_everyone()
        if runtime is not None and not getattr(runtime, "is_main_process", True):
            return self.path_for(name)
        target = self.path_for(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        unwrapped = _unwrap_model(model, runtime)
        model_state = (
            runtime.get_state_dict(model)
            if runtime is not None and hasattr(runtime, "get_state_dict")
            else unwrapped.state_dict()
        )
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model": model_state,
            "ema": _state_or_value(ema),
            "optimizer": optimizer.state_dict(),
            "scheduler": _state_or_value(scheduler),
            "scaler": _state_or_value(scaler),
            "training_state": asdict(state),
            # Kept at top level too for convenient external inspection.
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metrics": dict(state.best_metrics),
            "rng_state": capture_rng_state(),
            "config": to_serializable(config),
            "normalization": to_serializable(_state_or_value(normalization)),
            "extra": to_serializable(dict(extra or {})),
        }
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return target

    def load(
        self,
        resume: str | os.PathLike[str] | None,
        *,
        model: nn.Module,
        ema: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        runtime: Any | None = None,
        strict: bool = True,
        restore_rng: bool = True,
        optimizer_device: torch.device | None = None,
    ) -> tuple[TrainingState, dict[str, Any]]:
        source = self.resolve_resume(resume)
        if source is None:
            return TrainingState(), {}
        try:
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            checkpoint = torch.load(source, map_location="cpu")

        version = int(checkpoint.get("format_version", 0))
        if version > self.FORMAT_VERSION:
            raise RuntimeError(
                f"Checkpoint format {version} is newer than supported "
                f"format {self.FORMAT_VERSION}"
            )
        unwrapped = _unwrap_model(model, runtime)
        unwrapped.load_state_dict(checkpoint["model"], strict=strict)
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if optimizer_device is not None:
                move_optimizer_state(optimizer, optimizer_device)
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if restore_rng:
            restore_rng_state(checkpoint.get("rng_state"))

        raw_state = checkpoint.get("training_state") or {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "best_metrics": checkpoint.get("best_metrics", {}),
        }
        return TrainingState.from_mapping(raw_state), checkpoint


__all__ = ["CheckpointManager", "TrainingState", "move_optimizer_state"]
