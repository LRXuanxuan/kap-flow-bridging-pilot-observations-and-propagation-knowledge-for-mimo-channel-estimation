"""LS and empirical circular-Gaussian channel estimators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _batched(value: torch.Tensor, ndim: int, name: str) -> tuple[torch.Tensor, bool]:
    squeeze = value.ndim == ndim - 1
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim - 1} or {ndim} dimensions")
    return value, squeeze


def least_squares(observation: torch.Tensor, pilot: torch.Tensor) -> torch.Tensor:
    """Minimum-norm LS estimate for ``Y = H P + N``.

    The implementation supports underdetermined pilots and returns
    ``Y @ pinv(P)``.  Under circular AWGN this is also the ML estimator.
    """

    observation, squeeze = _batched(observation, 3, "observation")
    pilot, _ = _batched(pilot, 3, "pilot")
    if pilot.shape[0] == 1 and observation.shape[0] > 1:
        pilot = pilot.expand(observation.shape[0], -1, -1)
    if pilot.shape[0] != observation.shape[0]:
        raise ValueError("Pilot and observation batch sizes differ")
    if observation.shape[-1] != pilot.shape[-1]:
        raise ValueError("Observation and pilot lengths differ")
    estimate = torch.bmm(observation, torch.linalg.pinv(pilot))
    return estimate.squeeze(0) if squeeze else estimate


def gaussian_lmmse(
    observation: torch.Tensor,
    pilot: torch.Tensor,
    noise_var: torch.Tensor | float,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    jitter: float = 1.0e-6,
) -> torch.Tensor:
    """Per-receiver empirical Gaussian LMMSE estimate.

    ``mean`` has shape ``[B,Nr,Nt]`` or ``[Nr,Nt]`` and ``covariance`` has
    shape ``[B,Nr,Nt,Nt]`` or ``[Nr,Nt,Nt]``.  For a Gaussian prior this
    estimator is simultaneously LMMSE, MMSE and MAP (posterior mean equals
    posterior mode).
    """

    observation, squeeze = _batched(observation, 3, "observation")
    pilot, _ = _batched(pilot, 3, "pilot")
    mean, _ = _batched(mean, 3, "mean")
    covariance, _ = _batched(covariance, 4, "covariance")
    batch = observation.shape[0]
    tensors = {"pilot": pilot, "mean": mean, "covariance": covariance}
    for name, value in tensors.items():
        if value.shape[0] == 1 and batch > 1:
            tensors[name] = value.expand(batch, *value.shape[1:])
        elif value.shape[0] != batch:
            raise ValueError(f"{name} batch size does not match observation")
    pilot = tensors["pilot"].to(device=observation.device, dtype=observation.dtype)
    mean = tensors["mean"].to(device=observation.device, dtype=observation.dtype)
    covariance = tensors["covariance"].to(
        device=observation.device, dtype=observation.dtype
    )
    if mean.shape[:2] != observation.shape[:2]:
        raise ValueError("Mean and observation receiver dimensions differ")
    if covariance.shape[:2] != mean.shape[:2] or covariance.shape[-2:] != (
        pilot.shape[-2],
        pilot.shape[-2],
    ):
        raise ValueError("Covariance dimensions are inconsistent with mean/pilot")

    variance = torch.as_tensor(
        noise_var, device=observation.device, dtype=observation.real.dtype
    )
    if variance.numel() == 1:
        variance = variance.reshape(1).expand(batch)
    else:
        variance = variance.reshape(batch, -1)[:, 0]
    if torch.any(variance < 0) or not torch.isfinite(variance).all():
        raise ValueError("noise_var must be finite and non-negative")

    # Column-vector form: y = A h + n with A=P^T.  P is not conjugated in the
    # forward physical operator H@P, hence this ordinary transpose is exact.
    operator = pilot.transpose(-2, -1).unsqueeze(1)  # [B,1,Np,Nt]
    operator_h = operator.mH
    system = operator @ covariance @ operator_h
    eye = torch.eye(
        system.shape[-1], device=system.device, dtype=system.dtype
    ).view(1, 1, system.shape[-1], system.shape[-1])
    scale = covariance.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
    stabilizer = float(jitter) * scale.clamp_min(1.0e-12)
    system = system + (variance[:, None] + stabilizer).view(batch, -1, 1, 1) * eye
    prior_column = mean.unsqueeze(-1)
    residual = observation.unsqueeze(-1) - operator @ prior_column
    solved = torch.linalg.solve(system, residual)
    correction = covariance @ operator_h @ solved
    estimate = (prior_column + correction).squeeze(-1)
    return estimate.squeeze(0) if squeeze else estimate


class ChannelStatisticsAccumulator:
    """Streaming first/second moments grouped globally and by scene."""

    def __init__(self) -> None:
        self._groups: dict[str, dict[str, Any]] = {}

    def _update_group(self, name: str, channel: torch.Tensor) -> None:
        values = channel.detach().to(device="cpu", dtype=torch.complex128)
        if values.ndim != 3:
            raise ValueError("channel must be complex [B,Nr,Nt]")
        current = self._groups.get(name)
        if current is None:
            current = {
                "count": 0,
                "sum": torch.zeros(values.shape[1:], dtype=torch.complex128),
                "outer": torch.zeros(
                    values.shape[1], values.shape[2], values.shape[2],
                    dtype=torch.complex128,
                ),
            }
            self._groups[name] = current
        current["count"] += int(values.shape[0])
        current["sum"] += values.sum(dim=0)
        current["outer"] += torch.einsum(
            "brt,bru->rtu", values, values.conj()
        )

    def update(self, channel: torch.Tensor, scene_ids: torch.Tensor) -> None:
        channel = torch.as_tensor(channel)
        if not channel.is_complex() or channel.ndim != 3:
            raise ValueError("channel must be complex [B,Nr,Nt]")
        scene_ids = torch.as_tensor(scene_ids).reshape(-1).cpu()
        if scene_ids.numel() != channel.shape[0]:
            raise ValueError("scene_ids must contain one id per channel")
        self._update_group("global", channel)
        for scene in torch.unique(scene_ids).tolist():
            mask = scene_ids == int(scene)
            self._update_group(f"scene_{int(scene)}", channel[mask.to(channel.device)])

    def finalize(
        self, *, shrinkage: float = 0.05, jitter: float = 1.0e-6
    ) -> dict[str, dict[str, torch.Tensor | int]]:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0,1]")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative")
        result: dict[str, dict[str, torch.Tensor | int]] = {}
        for name, raw in self._groups.items():
            count = int(raw["count"])
            if count < 2:
                raise ValueError(f"At least two samples are required for {name}")
            mean = raw["sum"] / count
            covariance = raw["outer"] / count - torch.einsum(
                "rt,ru->rtu", mean, mean.conj()
            )
            covariance = 0.5 * (covariance + covariance.mH)
            diagonal_power = covariance.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
            eye = torch.eye(covariance.shape[-1], dtype=covariance.dtype).unsqueeze(0)
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage * diagonal_power[:, None, None] * eye
                + jitter * diagonal_power.clamp_min(1.0e-12)[:, None, None] * eye
            )
            result[name] = {
                "count": count,
                "mean": mean.to(torch.complex64),
                "covariance": covariance.to(torch.complex64),
            }
        return result


def select_statistics(
    artifact: Mapping[str, Any],
    scene_ids: torch.Tensor,
    *,
    scope: str = "global",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select global or scene-wise mean/covariance for a batch."""

    groups = artifact.get("groups")
    if not isinstance(groups, Mapping) or "global" not in groups:
        raise KeyError("Statistics artifact has no global group")
    scope = str(scope).lower()
    if scope not in {"global", "scene"}:
        raise ValueError("scope must be global or scene")
    scene_ids = torch.as_tensor(scene_ids).reshape(-1).cpu()
    if scope == "global":
        group = groups["global"]
        return group["mean"].unsqueeze(0), group["covariance"].unsqueeze(0)

    means = []
    covariances = []
    for scene in scene_ids.tolist():
        key = f"scene_{int(scene)}"
        if key not in groups:
            raise KeyError(f"Statistics artifact has no {key} group")
        means.append(groups[key]["mean"])
        covariances.append(groups[key]["covariance"])
    return torch.stack(means), torch.stack(covariances)


__all__ = [
    "ChannelStatisticsAccumulator",
    "gaussian_lmmse",
    "least_squares",
    "select_statistics",
]
