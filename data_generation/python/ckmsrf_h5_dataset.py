"""Minimal NumPy loader for HDF5 files exported by the MATLAB project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _matlab_array(dataset: h5py.Dataset) -> np.ndarray:
    """Restore the dimension order used by MATLAB's high-level HDF5 API."""
    value = np.asarray(dataset)
    if value.ndim <= 1:
        return value
    return np.transpose(value, axes=tuple(range(value.ndim - 1, -1, -1)))


def _complex(handle: h5py.File, base: str) -> np.ndarray | None:
    real_name = f"{base}_real"
    imag_name = f"{base}_imag"
    if real_name not in handle:
        return None
    return _matlab_array(handle[real_name]) + 1j * _matlab_array(
        handle[imag_name]
    )


@dataclass
class CkmsrfSample:
    h_true: np.ndarray | None
    y_pilot: np.ndarray | None
    position_xyz: np.ndarray
    velocity_xyz: np.ndarray
    observed_position_xyz: np.ndarray
    scalar_feature: np.ndarray
    angular_delay_power: np.ndarray
    pilot_id: int
    snr_db: float
    cfo_hz: float


class CkmsrfShard:
    """Eager shard reader. Suitable for a MATLAB shard of moderate size."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.h_true = _complex(f, "/H")
            self.y_pilot = _complex(f, "/Y")
            self.position_xyz = _matlab_array(f["/state/position_xyz"])
            self.velocity_xyz = _matlab_array(f["/state/velocity_xyz"])
            self.observed_position_xyz = _matlab_array(
                f["/context/observed_position_xyz"]
            )
            self.scalar_feature = _matlab_array(f["/context/scalar_feature"])
            self.angular_delay_power = _matlab_array(
                f["/context/angular_delay_power"]
            )
            self.pilot_id = _matlab_array(
                f["/observation/pilot_id"]
            ).reshape(-1)
            self.snr_db = _matlab_array(
                f["/observation/snr_db"]
            ).reshape(-1)
            self.cfo_hz = _matlab_array(
                f["/observation/cfo_hz"]
            ).reshape(-1)
            self.attrs: dict[str, Any] = dict(f.attrs)

        self.num_samples = int(self.position_xyz.shape[-1])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> CkmsrfSample:
        return CkmsrfSample(
            h_true=None if self.h_true is None else self.h_true[..., index],
            y_pilot=None if self.y_pilot is None else self.y_pilot[..., index],
            position_xyz=self.position_xyz[..., index],
            velocity_xyz=self.velocity_xyz[..., index],
            observed_position_xyz=self.observed_position_xyz[..., index],
            scalar_feature=self.scalar_feature[..., index],
            angular_delay_power=self.angular_delay_power[..., index],
            pilot_id=int(self.pilot_id[index]),
            snr_db=float(self.snr_db[index]),
            cfo_hz=float(self.cfo_hz[index]),
        )


def load_scene(path: str | Path) -> dict[str, np.ndarray]:
    """Load the pilot bank and CKM arrays exported from a scene file."""
    with h5py.File(path, "r") as f:
        return {
            "pilot_bank": _complex(f, "/pilot/P"),
            "Ftx": _complex(f, "/codebook/Ftx"),
            "Wrx": _complex(f, "/codebook/Wrx"),
            "grid_xy": _matlab_array(f["/ckm/grid_xy"]),
            "ckm_scalar": _matlab_array(f["/ckm/scalar_feature"]),
            "ckm_adp": _matlab_array(f["/ckm/angular_delay_power"]),
            "split_assignment": _matlab_array(f["/split/assignment"]),
        }
