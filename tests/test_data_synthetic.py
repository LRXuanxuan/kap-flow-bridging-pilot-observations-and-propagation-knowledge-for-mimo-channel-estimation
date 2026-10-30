from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ckmsrf.data import (
    CKMSRFDataset,
    DatasetIndex,
    ObservationConfig,
    build_shard_index,
    compute_normalization,
)
from ckmsrf.data.shard_index import _field


def _write_matlab_order(handle: h5py.File, name: str, value: np.ndarray) -> None:
    value = np.asarray(value)
    if value.ndim > 1:
        value = np.transpose(value, axes=tuple(range(value.ndim - 1, -1, -1)))
    parent = str(Path(name).parent).replace("\\", "/")
    if parent not in {".", "/"}:
        handle.require_group(parent)
    handle.create_dataset(name, data=value)


def _write_complex(handle: h5py.File, base: str, value: np.ndarray) -> None:
    _write_matlab_order(handle, base + "_real", np.asarray(value).real.astype(np.float32))
    _write_matlab_order(handle, base + "_imag", np.asarray(value).imag.astype(np.float32))


def _make_scene(root: Path, scene_id: int, nt: int, npilot: int, banks: int) -> None:
    path = root / f"scene_{scene_id:02d}_ckm.h5"
    p = np.empty((nt, npilot, banks), dtype=np.complex64)
    for bank in range(banks):
        phase = np.exp(1j * 2 * np.pi * (bank + 1) * np.arange(nt) / nt)
        p[:, :, bank] = phase[:, None] / np.sqrt(nt)
    with h5py.File(path, "w") as handle:
        _write_complex(handle, "/pilot/P", p)
        handle.attrs["scene_id"] = scene_id
        handle.attrs["storage_order"] = "matlab_reversed"


def _make_shard(
    root: Path,
    scene_id: int,
    split: str,
    *,
    nr: int = 2,
    nt: int = 3,
    nsc: int = 4,
    snapshots: int = 2,
    trajectories: int = 2,
) -> np.ndarray:
    path = root / f"scene_{scene_id:02d}_{split}_shard_001.h5"
    offset = 1000 * scene_id + (100 if split == "val" else 0)
    values = np.arange(nr * nt * nsc * snapshots * trajectories, dtype=np.float32)
    h = values.reshape(nr, nt, nsc, snapshots, trajectories) + offset
    h = np.asarray(h + 1j * (0.1 * h + 1.0), dtype=np.complex64)
    scale = np.sqrt(np.sum(np.abs(h) ** 2, axis=(0, 1))).astype(np.float32)

    position = np.empty((3, snapshots, trajectories), dtype=np.float32)
    velocity = np.empty_like(position)
    observed = np.empty_like(position)
    for n in range(trajectories):
        for t in range(snapshots):
            position[:, t, n] = [scene_id + n, t, 1.5]
            velocity[:, t, n] = [1.0 + n, 2.0 + t, 0.0]
            observed[:, t, n] = position[:, t, n] + [0.25, -0.25, 0.0]
    orientation = np.zeros((1, snapshots, trajectories), dtype=np.float32)
    orientation[0, :, :] = np.asarray([[0.0, 90.0], [180.0, 270.0]])
    scalar = np.arange(6 * snapshots * trajectories, dtype=np.float32).reshape(
        6, snapshots, trajectories
    )
    adp = np.arange(12 * snapshots * trajectories, dtype=np.float32).reshape(
        12, snapshots, trajectories
    )
    tx = np.full((nt, snapshots, trajectories), 1.0 / nt, dtype=np.float32)
    rx = np.full((nr, snapshots, trajectories), 1.0 / nr, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        _write_complex(handle, "/H", h)
        _write_matlab_order(handle, "/channel_scale_lin", scale)
        _write_matlab_order(handle, "/state/position_xyz", position)
        _write_matlab_order(handle, "/state/velocity_xyz", velocity)
        _write_matlab_order(handle, "/state/ue_orientation_deg", orientation)
        _write_matlab_order(
            handle, "/state/time_sec", np.asarray([[0.0], [0.1]], dtype=np.float32)
        )
        _write_matlab_order(
            handle,
            "/state/block_id",
            np.arange(snapshots * trajectories, dtype=np.uint16).reshape(
                snapshots, trajectories
            ),
        )
        _write_matlab_order(handle, "/context/observed_position_xyz", observed)
        _write_matlab_order(handle, "/context/scalar_feature", scalar)
        _write_matlab_order(handle, "/context/angular_delay_power", adp)
        _write_matlab_order(handle, "/context/tx_beam_probability", tx)
        _write_matlab_order(handle, "/context/rx_beam_probability", rx)
        _write_matlab_order(
            handle, "/observation/pilot_id", np.asarray([[1, 2]], dtype=np.uint16)
        )
        _write_matlab_order(
            handle, "/observation/snr_db", np.asarray([[10.0, 20.0]], dtype=np.float32)
        )
        _write_matlab_order(
            handle, "/observation/cfo_hz", np.zeros((1, trajectories), dtype=np.float32)
        )
        _write_matlab_order(
            handle,
            "/observation/noise_var",
            np.asarray([[0.25, 0.125]], dtype=np.float32),
        )
        _write_matlab_order(
            handle,
            "/sample_seed",
            np.asarray([[11, 22]], dtype=np.uint32),
        )
        handle.attrs["schema_version"] = "1.0"
        handle.attrs["dimension_order_H"] = "rx,tx,subcarrier,snapshot,sample"
        handle.attrs["storage_order"] = "matlab_reversed"
        handle.attrs["scene_id"] = scene_id
        handle.attrs["split_name"] = split
    return h


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> tuple[Path, dict[tuple[int, str], np.ndarray]]:
    expected: dict[tuple[int, str], np.ndarray] = {}
    for scene_id in (1, 2, 3):
        _make_scene(tmp_path, scene_id, nt=3, npilot=2, banks=2)
        expected[(scene_id, "train")] = _make_shard(tmp_path, scene_id, "train")
    expected[(1, "val")] = _make_shard(tmp_path, 1, "val")
    return tmp_path, expected


def test_index_is_relative_and_portable(synthetic_root: tuple[Path, dict]) -> None:
    root, _ = synthetic_root
    index = build_shard_index(root, "train", backend="hdf5", scene_ids=(1, 2, 3))
    assert index.num_trajectories == 6
    assert index.scene_ids == (1, 2, 3)
    assert all(not Path(record.path).is_absolute() for record in index.records)
    path = index.save(root / "train_index.json")
    restored = DatasetIndex.load(path, root=root)
    assert restored.num_trajectories == index.num_trajectories
    assert restored.locate(2)[0].scene_id == 2


def test_lazy_fixed_sample_condition_and_online_observation(
    synthetic_root: tuple[Path, dict[tuple[int, str], np.ndarray]],
) -> None:
    root, expected = synthetic_root
    index = build_shard_index(root, "train", backend="plain_h5", scene_ids=(1, 2, 3))
    dataset = CKMSRFDataset(
        index,
        sampling="fixed_eval",
        fixed_pairs=[(1, 0)],
        include_observation=True,
        observation=ObservationConfig(noise_mode="stored", base_seed=7),
        adp_shape=(2, 3, 2),
    )
    sample = dataset[0]
    np.testing.assert_allclose(sample["H"].numpy(), expected[(1, "train")][:, :, 1, 0, 0])
    assert sample["h"].shape == (2, 2, 3)
    assert sample["condition"]["state"].shape == (8,)
    assert sample["condition"]["scalar"].shape == (6,)
    assert sample["condition"]["adp"].shape == (12,)
    raw_adp = np.arange(12 * 2 * 2, dtype=np.float32).reshape(12, 2, 2)[:, 0, 0]
    expected_adp = np.reshape(raw_adp, (2, 3, 2), order="F").reshape(-1)
    np.testing.assert_array_equal(sample["condition"]["adp"].numpy(), expected_adp)
    assert sample["P"].shape == (3, 2)
    assert sample["Y"].shape == (2, 2)
    assert torch.equal(sample["Y"], dataset[0]["Y"])
    assert sample["frequency"].item() == pytest.approx(-1.0 / 3.0)
    assert sample["time"].item() == pytest.approx(-1.0)

    # Access first so that live h5 handles exist, then verify they are excluded
    # from serialization and reopened lazily after a Windows-style spawn.
    restored = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(restored[0]["H"], sample["H"])
    assert torch.equal(restored[0]["Y"], sample["Y"])


def test_sampling_lengths_and_epoch_noise(synthetic_root: tuple[Path, dict]) -> None:
    root, _ = synthetic_root
    index = build_shard_index(root, "train", backend="plain_h5", scene_ids=(1, 2, 3))
    all_tf = CKMSRFDataset(index, sampling="all_tf")
    assert len(all_tf) == 6 * 4 * 2
    assert (all_tf[1]["snapshot_idx"].item(), all_tf[1]["subcarrier_idx"].item()) == (0, 1)

    random_tf = CKMSRFDataset(
        index,
        sampling="random_tf",
        include_observation=True,
        observation=ObservationConfig(noise_mode="stored", base_seed=99),
    )
    first = random_tf[0]
    random_tf.set_epoch(1)
    second = random_tf[0]
    # Even if the tiny synthetic grid hashes to the same t/f pair, the online
    # training noise is epoch-dependent.
    assert not torch.equal(first["Y"], second["Y"])


def test_streaming_train_only_normalization(
    synthetic_root: tuple[Path, dict[tuple[int, str], np.ndarray]],
) -> None:
    root, expected = synthetic_root
    train_index = build_shard_index(root, "train", backend="plain_h5", scene_ids=(1, 2, 3))
    stats = compute_normalization(train_index, chunk_trajectories=1)
    all_h = np.concatenate([expected[(scene, "train")].reshape(-1) for scene in (1, 2, 3)])
    expected_rms = np.sqrt(np.mean(np.abs(all_h.astype(np.complex128)) ** 2))
    assert stats.h_rms == pytest.approx(expected_rms)
    assert stats.num_trajectories == 6
    assert len(stats.feature_mean["scalar"]) == 6
    assert len(stats.feature_mean["adp"]) == 12

    normalized = CKMSRFDataset(train_index, sampling="center", normalization=stats)[0]
    assert normalized["H"].abs().square().mean().isfinite()

    val_index = build_shard_index(root, "val", backend="plain_h5", scene_ids=(1,))
    with pytest.raises(ValueError, match="train only"):
        compute_normalization(val_index)


def test_mat_struct_wrapper_axes_are_not_data_axes() -> None:
    dtype = np.dtype([("orientation", np.float32, (1, 8, 2))])
    wrapper = np.zeros((1,), dtype=dtype)
    value = _field(wrapper, "orientation")
    assert value.shape == (1, 8, 2)


def test_channel_rms_is_not_clamped_to_feature_epsilon(
    synthetic_root: tuple[Path, dict],
) -> None:
    root, _ = synthetic_root
    path = root / "scene_01_train_shard_001.h5"
    with h5py.File(path, "r+") as handle:
        handle["/H_real"][...] *= 1.0e-10
        handle["/H_imag"][...] *= 1.0e-10
    index = build_shard_index(root, "train", backend="plain_h5", scene_ids=(1,))
    stats = compute_normalization(index, max_trajectories=1)
    assert 0.0 < stats.h_rms < stats.epsilon
