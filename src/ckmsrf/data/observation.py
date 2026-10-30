"""Finite-pilot measurement operators and deterministic online AWGN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ObservationConfig:
    """Controls online measurements returned by :class:`CKMSRFDataset`.

    ``noise_mode='stored'`` uses the shard's stored noise variance.  It is the
    closest reproduction of the generator. ``noise_mode='snr'`` recomputes
    the variance from the selected 2-D slice and requested SNR.  ``none``
    returns a noiseless measurement.
    """

    enabled: bool = True
    noise_mode: str = "stored"
    deterministic: bool = True
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_mode not in {"stored", "snr", "none"}:
            raise ValueError("noise_mode must be stored, snr, or none.")


def complex_to_channels(value: np.ndarray) -> np.ndarray:
    """Convert a complex array to contiguous ``[2, ...]`` float32 channels."""

    value = np.asarray(value)
    return np.stack((value.real, value.imag), axis=0).astype(np.float32, copy=False)


def channels_to_complex(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] != 2:
        raise ValueError(f"Expected leading real/imag dimension 2, got {value.shape}.")
    return np.asarray(value[0] + 1j * value[1], dtype=np.complex64)


def derive_seed(*values: int) -> int:
    """Stable 64-bit seed mixer independent of Python's randomized hash."""

    state = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        state ^= int(value) & mask
        state = (state + 0x9E3779B97F4A7C15) & mask
        state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
        state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & mask
        state ^= state >> 31
    return state & mask


def apply_pilot(h: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply ``Y0 = H @ P`` to one channel slice."""

    h = np.asarray(h, dtype=np.complex64)
    p = np.asarray(p, dtype=np.complex64)
    if h.ndim != 2 or p.ndim != 2 or h.shape[1] != p.shape[0]:
        raise ValueError(f"Incompatible H/P shapes: H={h.shape}, P={p.shape}.")
    return np.asarray(h @ p, dtype=np.complex64)


def cfo_ici_matrix(num_subcarriers: int, normalized_cfo: float) -> np.ndarray:
    """Return the OFDM ICI matrix used by the MATLAB data generator."""

    n = np.arange(num_subcarriers, dtype=np.float64)
    k = np.arange(num_subcarriers, dtype=np.float64)[:, None, None]
    m = np.arange(num_subcarriers, dtype=np.float64)[None, :, None]
    phase = normalized_cfo + m - k
    matrix = np.mean(
        np.exp(1j * 2.0 * np.pi * phase * n[None, None, :] / num_subcarriers),
        axis=-1,
    )
    return np.asarray(matrix, dtype=np.complex64)


def apply_pilot_with_cfo(
    h_all_subcarriers: np.ndarray,
    p: np.ndarray,
    *,
    normalized_cfo: float,
    snapshot_phase: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Apply pilots and exact inter-carrier interference to a full snapshot.

    Args:
        h_all_subcarriers: Complex ``[Nr, Nt, Nsc]`` array.
        p: Complex ``[Nt, Np]`` pilot matrix.

    Returns:
        Complex ``[Nr, Np, Nsc]`` observation.
    """

    h_all = np.asarray(h_all_subcarriers, dtype=np.complex64)
    p = np.asarray(p, dtype=np.complex64)
    if h_all.ndim != 3 or h_all.shape[1] != p.shape[0]:
        raise ValueError(f"Incompatible full H/P shapes: {h_all.shape}, {p.shape}.")
    projected = np.einsum("rtk,tp->rpk", h_all, p, optimize=True)
    cfo = cfo_ici_matrix(h_all.shape[-1], normalized_cfo)
    observed = np.einsum("km,rpm->rpk", cfo, projected, optimize=True)
    return np.asarray(snapshot_phase * observed, dtype=np.complex64)


def add_complex_awgn(
    signal: np.ndarray, noise_var: float, *, seed: int | None = None
) -> np.ndarray:
    """Add circular complex Gaussian noise with ``E|n|^2=noise_var``."""

    signal = np.asarray(signal, dtype=np.complex64)
    noise_var = float(noise_var)
    if not np.isfinite(noise_var) or noise_var < 0.0:
        raise ValueError(f"noise_var must be finite and non-negative, got {noise_var}.")
    if noise_var == 0.0:
        return signal.copy()
    rng = np.random.default_rng(seed)
    scale = np.sqrt(noise_var / 2.0)
    noise = scale * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return np.asarray(signal + noise, dtype=np.complex64)


def synthesize_observation(
    h: np.ndarray,
    p: np.ndarray,
    *,
    noise_var: float | None = None,
    snr_db: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Generate ``Y = H P + N`` and return its effective noise variance."""

    clean = apply_pilot(h, p)
    if noise_var is None:
        if snr_db is None:
            noise_var = 0.0
        else:
            signal_power = float(np.mean(np.abs(clean) ** 2, dtype=np.float64))
            noise_var = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_var = max(float(noise_var), float(np.finfo(np.float32).tiny)) if noise_var else 0.0
    return add_complex_awgn(clean, noise_var, seed=seed), noise_var


__all__ = [
    "ObservationConfig",
    "add_complex_awgn",
    "apply_pilot",
    "apply_pilot_with_cfo",
    "cfo_ici_matrix",
    "channels_to_complex",
    "complex_to_channels",
    "derive_seed",
    "synthesize_observation",
]
