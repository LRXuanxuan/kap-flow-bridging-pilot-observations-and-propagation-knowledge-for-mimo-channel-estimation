"""Gaussian-mixture algebra used by both training and SRF inference."""

from __future__ import annotations

import math

import torch


GM = dict[str, torch.Tensor]


def _broadcast_logstd(gm: GM) -> torch.Tensor:
    means, logstds = gm["means"], gm["logstds"]
    try:
        return torch.broadcast_to(logstds, means.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"logstds {tuple(logstds.shape)} cannot broadcast to means {tuple(means.shape)}"
        ) from exc


def validate_gm(gm: GM) -> None:
    required = {"means", "logstds", "logweights"}
    missing = required.difference(gm)
    if missing:
        raise KeyError(f"Missing GM fields: {sorted(missing)}")
    if gm["means"].ndim != 5:
        raise ValueError("means must be [B,K,C,H,W]")
    if gm["logweights"].shape[:2] != gm["means"].shape[:2]:
        raise ValueError("means and logweights disagree on B or K")
    _broadcast_logstd(gm)


def normalize_gm_logweights(logits: torch.Tensor) -> torch.Tensor:
    return logits.log_softmax(dim=1)


def gm_to_mean(gm: GM) -> torch.Tensor:
    validate_gm(gm)
    return (gm["logweights"].exp() * gm["means"]).sum(dim=1)


def gm_second_moment(gm: GM, eps: float = 1e-8) -> torch.Tensor:
    variance = (2.0 * _broadcast_logstd(gm)).exp().clamp_min(eps)
    return (gm["logweights"].exp() * (gm["means"].square() + variance)).sum(dim=1)


def gm_effective_variance(gm: GM, eps: float = 1e-8) -> torch.Tensor:
    mean = gm_to_mean(gm)
    return (gm_second_moment(gm, eps) - mean.square()).clamp_min(eps)


def gm_global_stats(gm: GM) -> dict[str, torch.Tensor]:
    weights = gm["logweights"].exp()
    return {
        "entropy_mean": -(weights.clamp_min(1e-12) * gm["logweights"]).sum(dim=1).mean(),
        "max_weight_mean": weights.max(dim=1).values.mean(),
        "logstd_mean": gm["logstds"].mean(),
    }


def sample_gm(gm: GM, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one component independently at every spatial location."""
    validate_gm(gm)
    means = gm["means"]
    batch, components, channels, height, width = means.shape
    probabilities = gm["logweights"][:, :, 0].permute(0, 2, 3, 1).reshape(-1, components).exp()
    indices = torch.multinomial(probabilities, 1, replacement=True, generator=generator)
    indices = indices.view(batch, height, width)
    gather = indices[:, None, None].expand(-1, 1, channels, -1, -1)
    chosen_mean = means.gather(1, gather).squeeze(1)
    chosen_logstd = _broadcast_logstd(gm).gather(1, gather).squeeze(1)
    noise = torch.randn(
        chosen_mean.shape,
        device=chosen_mean.device,
        dtype=chosen_mean.dtype,
        generator=generator,
    )
    return chosen_mean + chosen_logstd.exp() * noise


def _time4(time: torch.Tensor, batch: int, eps: float = 0.0) -> torch.Tensor:
    time = time.to(dtype=torch.float32) if not time.is_floating_point() else time
    if time.numel() == 1:
        time = time.expand(batch)
    time = time.reshape(batch, -1)[:, 0]
    return time.clamp_min(eps).view(batch, 1, 1, 1)


def u_to_x0_gm(gm_u: GM, x_t: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> GM:
    sigma4 = _time4(sigma, x_t.shape[0], eps).to(device=x_t.device, dtype=x_t.dtype)
    return {
        "means": x_t.unsqueeze(1) - sigma4.unsqueeze(1) * gm_u["means"],
        "logstds": gm_u["logstds"] + sigma4.log().unsqueeze(1),
        "logweights": gm_u["logweights"],
    }


def reverse_transition_additive(
    gm_u: GM,
    x_t_high: torch.Tensor,
    t_low: torch.Tensor,
    t_high: torch.Tensor,
    eps: float = 1e-6,
) -> GM:
    high = _time4(t_high, x_t_high.shape[0]).to(x_t_high)
    low = _time4(t_low, x_t_high.shape[0]).to(x_t_high)
    delta = (high - low).clamp_min(eps)
    return {
        "means": x_t_high.unsqueeze(1) - delta.unsqueeze(1) * gm_u["means"],
        "logstds": gm_u["logstds"] + delta.log().unsqueeze(1),
        "logweights": gm_u["logweights"],
    }


def gm_logprob(gm: GM, target: torch.Tensor) -> torch.Tensor:
    """Return per-pixel log likelihood with the real/imag channels coupled."""
    validate_gm(gm)
    target = target.unsqueeze(1)
    logstd = _broadcast_logstd(gm)
    component_logprob = -0.5 * ((target - gm["means"]) * (-logstd).exp()).square()
    component_logprob = component_logprob - logstd - 0.5 * math.log(2.0 * math.pi)
    component_logprob = component_logprob.sum(dim=2, keepdim=True) + gm["logweights"]
    return torch.logsumexp(component_logprob, dim=1)


def gm_nll_loss(gm: GM, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    nll = -gm_logprob(gm, target)
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def gm_tau2_from_x0_gm(gm_x0: GM, eps: float = 1e-8) -> torch.Tensor:
    return gm_effective_variance(gm_x0, eps).flatten(1).mean(dim=1).clamp_min(eps)

