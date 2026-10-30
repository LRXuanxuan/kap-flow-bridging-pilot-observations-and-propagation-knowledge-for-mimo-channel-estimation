"""One-writer JSONL and optional TensorBoard experiment logging."""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch

from ckmsrf.config import save_resolved_config


def _scalar(value: Any) -> float | int | str | bool | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be logged as metrics")
        value = value.detach().cpu().item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


class ExperimentLogger:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        enable_jsonl: bool = True,
        enable_tensorboard: bool = True,
        is_main_process: bool = True,
        flush_every: int = 1,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.is_main_process = bool(is_main_process)
        self.flush_every = max(1, int(flush_every))
        self._writes = 0
        self._jsonl = None
        self._writer = None
        if not self.is_main_process:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if enable_jsonl:
            self._jsonl = (self.run_dir / "metrics.jsonl").open(
                "a", encoding="utf-8"
            )
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ImportError:
                warnings.warn(
                    "TensorBoard is unavailable; continuing with JSONL logging",
                    stacklevel=2,
                )

    def write_config(self, config: Mapping[str, Any]) -> Path | None:
        if not self.is_main_process:
            return None
        return save_resolved_config(config, self.run_dir / "resolved_config.yaml")

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        split: str | None = None,
        epoch: int | None = None,
    ) -> None:
        if not self.is_main_process:
            return
        clean = {str(key): _scalar(value) for key, value in metrics.items()}
        record: dict[str, Any] = {
            "time_unix": time.time(),
            "step": int(step),
            **clean,
        }
        if split is not None:
            record["split"] = split
        if epoch is not None:
            record["epoch"] = int(epoch)
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._writes += 1
            if self._writes % self.flush_every == 0:
                self._jsonl.flush()
        if self._writer is not None:
            prefix = f"{split}/" if split else ""
            for key, value in clean.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._writer.add_scalar(prefix + key, value, int(step))

    def flush(self) -> None:
        if self._jsonl is not None:
            self._jsonl.flush()
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.flush()
            self._jsonl.close()
            self._jsonl = None
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["ExperimentLogger"]
