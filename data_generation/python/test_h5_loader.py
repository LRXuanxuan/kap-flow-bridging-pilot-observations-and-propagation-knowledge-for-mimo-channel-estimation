"""Smoke test for MATLAB-to-Python dimension restoration."""

from __future__ import annotations

import argparse

import numpy as np

from ckmsrf_h5_dataset import CkmsrfShard, load_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard_h5")
    parser.add_argument("scene_h5")
    args = parser.parse_args()

    shard = CkmsrfShard(args.shard_h5)
    scene = load_scene(args.scene_h5)

    assert shard.h_true is not None
    assert shard.y_pilot is not None
    assert shard.h_true.shape == (16, 64, 4, 3, 6)
    assert shard.y_pilot.shape == (16, 19, 4, 3, 6)
    assert shard.position_xyz.shape == (3, 3, 6)
    assert shard.scalar_feature.shape == (6, 3, 6)
    assert len(shard) == 6
    assert shard[0].h_true.shape == (16, 64, 4, 3)
    assert np.isfinite(shard.h_true).all()
    assert np.isfinite(shard.y_pilot).all()

    assert scene["pilot_bank"].shape == (64, 19, 4)
    assert scene["Ftx"].shape == (64, 64)
    assert scene["Wrx"].shape == (16, 16)
    assert scene["grid_xy"].shape[1] == 2
    print("Python HDF5 loader tests passed.")


if __name__ == "__main__":
    main()
