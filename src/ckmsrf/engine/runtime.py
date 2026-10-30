"""Accelerate-backed runtime with a dependency-light local fallback."""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn

from ckmsrf.config import RuntimeSpec, resolve_device_precision

from .reproducibility import make_generator, seed_worker


class LocalRuntime:
    """Small subset of ``accelerate.Accelerator`` used by this project."""

    is_main_process = True
    is_local_main_process = True
    num_processes = 1
    process_index = 0

    def __init__(self, spec: RuntimeSpec, gradient_accumulation_steps: int = 1) -> None:
        self.spec = spec
        self.device = spec.device
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.sync_gradients = True
        self._micro_step = 0
        self.scaler = None
        if spec.precision == "fp16" and spec.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def prepare(self, *objects: Any) -> Any:
        prepared: list[Any] = []
        for item in objects:
            prepared.append(item.to(self.device) if isinstance(item, nn.Module) else item)
        return prepared[0] if len(prepared) == 1 else tuple(prepared)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @contextlib.contextmanager
    def accumulate(self, model: nn.Module) -> Iterator[None]:
        self._micro_step += 1
        self.sync_gradients = (
            self._micro_step % self.gradient_accumulation_steps == 0
        )
        no_sync = getattr(model, "no_sync", None)
        context = no_sync() if no_sync is not None and not self.sync_gradients else contextlib.nullcontext()
        with context:
            yield

    @contextlib.contextmanager
    def autocast(self) -> Iterator[None]:
        if self.spec.amp_enabled:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.spec.autocast_dtype,
                enabled=True,
            ):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        scaled_loss = loss / self.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def clip_grad_norm_(self, parameters: Any, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def gather_for_metrics(self, value: Any) -> Any:
        return value

    def wait_for_everyone(self) -> None:
        return None

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)

    def end_training(self) -> None:
        return None


def create_runtime(config: Mapping[str, Any]) -> Any:
    """Create an Accelerator when requested, otherwise a local runtime."""

    hardware = config.get("hardware", {})
    spec = resolve_device_precision(config)
    accumulation = int(hardware.get("grad_accum_steps", 1))
    if bool(hardware.get("use_accelerate", True)):
        try:
            from accelerate import Accelerator

            return Accelerator(
                gradient_accumulation_steps=accumulation,
                mixed_precision=spec.mixed_precision,
                cpu=spec.device.type == "cpu",
            )
        except ImportError:
            warnings.warn(
                "accelerate is not installed; falling back to one-process runtime",
                stacklevel=2,
            )
    return LocalRuntime(spec, gradient_accumulation_steps=accumulation)


def dataloader_kwargs(
    config: Mapping[str, Any],
    *,
    training: bool,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return portable DataLoader arguments, including a spawn-safe worker seed."""

    hardware = config.get("hardware", {})
    spec = resolve_device_precision(config)
    workers = int(hardware.get("num_workers", 0))
    pin_setting = hardware.get("pin_memory", "auto")
    pin_memory = spec.device.type == "cuda" if pin_setting == "auto" else bool(pin_setting)
    persistent_setting = hardware.get("persistent_workers", "auto")
    # CKMSRFDataset.set_epoch stores a normal integer. Spawned worker copies
    # only observe a new epoch when workers are recreated; therefore automatic
    # persistence is safe for fixed evaluation but deliberately off for train.
    persistent = (
        workers > 0 and not training
        if persistent_setting == "auto"
        else bool(persistent_setting)
    )
    persistent = persistent and workers > 0
    batch_key = "batch_size" if training else "eval_batch_size"
    base_seed = int(config.get("training", {}).get("seed", 3407) if seed is None else seed)
    return {
        "batch_size": int(hardware.get(batch_key, hardware.get("batch_size", 1))),
        "shuffle": bool(training),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
        "worker_init_fn": seed_worker,
        "generator": make_generator(base_seed),
    }


def default_run_dir(config: Mapping[str, Any]) -> Path:
    root = Path(config["paths"]["run_root"])
    name = str(config.get("experiment", {}).get("name", "ckmsrf"))
    return root / name


__all__ = [
    "LocalRuntime",
    "create_runtime",
    "dataloader_kwargs",
    "default_run_dir",
]
