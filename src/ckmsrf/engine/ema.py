"""Exponential moving average with checkpointable state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn


class EMA:
    """Track a floating-point EMA of model parameters and buffers."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        *,
        update_after_step: int = 0,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] | None = None
        self._copy_initial(model)

    @torch.no_grad()
    def _copy_initial(self, model: nn.Module) -> None:
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if torch.is_floating_point(tensor) or torch.is_complex(tensor)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        self.num_updates += 1
        decay = 0.0 if self.num_updates <= self.update_after_step else self.decay
        one_minus_decay = 1.0 - decay
        for name, average in self.shadow.items():
            if name not in current:
                raise KeyError(f"EMA parameter disappeared from model: {name}")
            value = current[name].detach().to(device=average.device)
            average.lerp_(value, one_minus_decay)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "num_updates": self.num_updates,
            "shadow": {key: value.clone() for key, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.update_after_step = int(state.get("update_after_step", 0))
        self.num_updates = int(state.get("num_updates", 0))
        loaded = state.get("shadow", {})
        missing = set(self.shadow) - set(loaded)
        unexpected = set(loaded) - set(self.shadow)
        if missing or unexpected:
            raise RuntimeError(
                f"EMA key mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        # Checkpoints are intentionally loaded on CPU.  Keep each existing EMA
        # tensor on its model-side device to avoid a full GPU->CPU copy per step
        # after resume.
        self.shadow = {
            key: loaded[key]
            .detach()
            .to(device=self.shadow[key].device, dtype=self.shadow[key].dtype)
            .clone()
            for key in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, average in self.shadow.items():
            if name in state:
                state[name].copy_(average.to(device=state[name].device))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        if self._backup is not None:
            raise RuntimeError("EMA.store() called twice without restore()")
        state = model.state_dict()
        self._backup = {
            name: state[name].detach().clone()
            for name in self.shadow
            if name in state
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            raise RuntimeError("EMA.restore() called before store()")
        state = model.state_dict()
        for name, value in self._backup.items():
            state[name].copy_(value.to(device=state[name].device))
        self._backup = None

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)


__all__ = ["EMA"]
