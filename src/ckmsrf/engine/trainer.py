"""Reusable state, optimizer, EMA, logging and checkpoint orchestration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .checkpoint import CheckpointManager, TrainingState
from .ema import EMA
from .logging import ExperimentLogger
from .reproducibility import seed_everything
from .runtime import create_runtime, default_run_dir


class TrainingEngine:
    """Own cross-cutting training mechanics while leaving forward/loss to callers.

    Typical use::

        with engine.accumulate():
            with engine.autocast():
                loss = objective(engine.model, batch)
            stepped = engine.backward_and_step(loss)
    """

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        normalization: Any = None,
        run_dir: str | Path | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime or create_runtime(config)
        self.run_dir = Path(run_dir) if run_dir is not None else default_run_dir(config)
        self.normalization = normalization
        seed_everything(int(config.get("training", {}).get("seed", 3407)))

        if scheduler is None:
            self.model, self.optimizer = self.runtime.prepare(model, optimizer)
            self.scheduler = None
        else:
            self.model, self.optimizer, self.scheduler = self.runtime.prepare(
                model, optimizer, scheduler
            )
        self.scaler = getattr(self.runtime, "scaler", None)
        unwrapped = self.unwrap_model()
        training = config.get("training", {})
        self.ema = EMA(unwrapped, decay=float(training.get("ema_decay", 0.999)))
        self.state = TrainingState()
        self.checkpoints = CheckpointManager(self.run_dir / "checkpoints")
        logging_cfg = config.get("logging", {})
        self.logger = ExperimentLogger(
            self.run_dir,
            enable_jsonl=bool(logging_cfg.get("jsonl", True)),
            enable_tensorboard=bool(logging_cfg.get("tensorboard", True)),
            is_main_process=bool(getattr(self.runtime, "is_main_process", True)),
        )
        self.logger.write_config(config)
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def device(self) -> torch.device:
        return self.runtime.device

    @property
    def is_main_process(self) -> bool:
        return bool(getattr(self.runtime, "is_main_process", True))

    def unwrap_model(self) -> nn.Module:
        return self.runtime.unwrap_model(self.model)

    def prepare_dataloaders(self, *loaders: Any) -> Any:
        return self.runtime.prepare(*loaders)

    def accumulate(self) -> AbstractContextManager[Any]:
        return self.runtime.accumulate(self.model)

    def autocast(self) -> AbstractContextManager[Any]:
        return self.runtime.autocast()

    def backward_and_step(self, loss: torch.Tensor) -> bool:
        """Backpropagate one micro-batch and return whether optimizer stepped."""

        self.runtime.backward(loss)
        if not bool(getattr(self.runtime, "sync_gradients", True)):
            return False

        max_norm = self.config.get("training", {}).get("max_grad_norm")
        if max_norm is not None and float(max_norm) > 0:
            if (
                self.scaler is not None
                and self.runtime.__class__.__name__ == "LocalRuntime"
                and hasattr(self.scaler, "unscale_")
            ):
                self.scaler.unscale_(self.optimizer)
            self.runtime.clip_grad_norm_(self.model.parameters(), float(max_norm))

        if self.scaler is not None and self.runtime.__class__.__name__ == "LocalRuntime":
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if (
            self.scheduler is not None
            and self.config.get("scheduler", {}).get("step_interval", "step") == "step"
        ):
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.unwrap_model())
        self.state.global_step += 1
        return True

    def finish_epoch(self, next_epoch: int) -> None:
        self.state.epoch = int(next_epoch)
        if (
            self.scheduler is not None
            and self.config.get("scheduler", {}).get("step_interval", "step") == "epoch"
        ):
            self.scheduler.step()

    def save(self, name: str = "last.pt", **extra: Any) -> Path:
        return self.checkpoints.save(
            name,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.state,
            config=self.config,
            normalization=self.normalization,
            runtime=self.runtime,
            extra=extra,
        )

    def resume(
        self,
        resume: str | Path | None = None,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        source = resume if resume is not None else self.config.get("training", {}).get("resume")
        self.state, payload = self.checkpoints.load(
            source,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            runtime=self.runtime,
            strict=strict,
            restore_rng=True,
            optimizer_device=self.device,
        )
        return payload

    def update_best(
        self,
        metric: str,
        value: float,
        *,
        mode: str = "min",
        checkpoint_name: str | None = None,
    ) -> bool:
        improved = self.state.is_improved(metric, float(value), mode=mode)
        if improved and checkpoint_name:
            self.save(checkpoint_name, selected_metric=metric, selected_value=value)
        return improved

    def close(self) -> None:
        self.logger.close()
        self.runtime.wait_for_everyone()
        self.runtime.end_training()

    def __enter__(self) -> "TrainingEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["TrainingEngine"]
