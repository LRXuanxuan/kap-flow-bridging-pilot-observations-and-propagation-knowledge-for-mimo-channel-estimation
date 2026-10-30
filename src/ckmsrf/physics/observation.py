"""Complex MIMO pilot observation operator shared by training and inference."""

from __future__ import annotations

from typing import Any

import torch


def twochannel_to_complex(value: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,2,Nr,Nt]`` (or ``[2,Nr,Nt]``) to a complex tensor."""
    squeeze = value.ndim == 3
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError(f"Expected [B,2,Nr,Nt], got {tuple(value.shape)}")
    result = torch.view_as_complex(value.permute(0, 2, 3, 1).contiguous())
    return result.squeeze(0) if squeeze else result


def complex_to_twochannel(value: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,Nr,Nt]`` complex channels to ``[B,2,Nr,Nt]``."""
    if not value.is_complex():
        raise TypeError("complex_to_twochannel expects a complex tensor")
    squeeze = value.ndim == 2
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"Expected [B,Nr,Nt], got {tuple(value.shape)}")
    result = torch.view_as_real(value).permute(0, 3, 1, 2).contiguous()
    return result.squeeze(0) if squeeze else result


def _ensure_batched_pilot(pilot: torch.Tensor, batch_size: int) -> torch.Tensor:
    if pilot.ndim == 2:
        pilot = pilot.unsqueeze(0).expand(batch_size, -1, -1)
    if pilot.ndim != 3 or pilot.shape[0] != batch_size:
        raise ValueError(f"Pilot must be [Nt,Np] or [B,Nt,Np], got {tuple(pilot.shape)}")
    return pilot


def apply_measurement(channel: torch.Tensor, pilot: torch.Tensor) -> torch.Tensor:
    """Apply the physical pilot operator ``A(H)=H@P``."""
    squeeze = channel.ndim == 2
    if squeeze:
        channel = channel.unsqueeze(0)
    if channel.ndim != 3:
        raise ValueError(f"Channel must be [Nr,Nt] or [B,Nr,Nt], got {tuple(channel.shape)}")
    pilot = _ensure_batched_pilot(pilot.to(device=channel.device, dtype=channel.dtype), channel.shape[0])
    if channel.shape[-1] != pilot.shape[-2]:
        raise ValueError("Channel Nt and pilot Nt are inconsistent")
    result = torch.bmm(channel, pilot)
    return result.squeeze(0) if squeeze else result


def apply_adjoint(residual: torch.Tensor, pilot: torch.Tensor) -> torch.Tensor:
    """Apply ``A^H(R)=R@P^H``."""
    squeeze = residual.ndim == 2
    if squeeze:
        residual = residual.unsqueeze(0)
    pilot = _ensure_batched_pilot(
        pilot.to(device=residual.device, dtype=residual.dtype), residual.shape[0]
    )
    result = torch.bmm(residual, pilot.mH)
    return result.squeeze(0) if squeeze else result


def _batch_scalar(
    value: Any,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() == 1:
        value = value.expand(batch_size)
    value = value.reshape(batch_size, -1)[:, 0]
    return value


def generate_observation(
    channel: torch.Tensor,
    pilot: torch.Tensor,
    *,
    noise_var: torch.Tensor | float | None = None,
    snr_db: torch.Tensor | float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``Y=HP+N`` and return the complex-noise variance.

    ``noise_var`` is authoritative when available in the dataset.  If only SNR
    is given, variance is derived from the per-sample clean observation power.
    """
    clean = apply_measurement(channel, pilot)
    squeeze = clean.ndim == 2
    if squeeze:
        clean = clean.unsqueeze(0)
    real_dtype = clean.real.dtype
    if noise_var is None:
        if snr_db is None:
            raise ValueError("Either noise_var or snr_db must be supplied")
        snr = _batch_scalar(snr_db, clean.shape[0], device=clean.device, dtype=real_dtype)
        signal_power = clean.abs().square().mean(dim=(-2, -1))
        variance = signal_power / torch.pow(torch.tensor(10.0, device=clean.device), snr / 10.0)
    else:
        variance = _batch_scalar(noise_var, clean.shape[0], device=clean.device, dtype=real_dtype)
    variance = variance.clamp_min(torch.finfo(real_dtype).tiny)
    kwargs = {"device": clean.device, "dtype": real_dtype, "generator": generator}
    noise = torch.complex(torch.randn(clean.shape, **kwargs), torch.randn(clean.shape, **kwargs))
    noise = noise * torch.sqrt(variance / 2.0).view(-1, 1, 1)
    observed = clean + noise
    return (observed.squeeze(0), variance.squeeze(0)) if squeeze else (observed, variance)


def data_consistency_loss(
    channel: torch.Tensor,
    observation: torch.Tensor,
    pilot: torch.Tensor,
    *,
    noise_var: torch.Tensor | float | None = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Noise-normalized pilot residual used by physics-aware fine-tuning."""
    residual = apply_measurement(channel, pilot) - observation
    if residual.ndim == 2:
        residual = residual.unsqueeze(0)
    loss = residual.abs().square().mean(dim=(-2, -1))
    if noise_var is not None:
        variance = _batch_scalar(
            noise_var,
            residual.shape[0],
            device=residual.device,
            dtype=residual.real.dtype,
        )
        loss = loss / variance.clamp_min(eps)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")

