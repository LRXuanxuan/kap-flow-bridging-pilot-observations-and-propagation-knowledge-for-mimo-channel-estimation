from pathlib import Path

from ckmsrf_h5_dataset import CkmsrfShard, load_scene


root = Path(__file__).resolve().parents[1]
h5_dir = root / "output_demo_h5"

scene = load_scene(h5_dir / "scene_01_ckm.h5")
shard = CkmsrfShard(h5_dir / "scene_01_train_shard_001.h5")
sample = shard[0]

print("pilot bank:", scene["pilot_bank"].shape)
print("number of samples:", len(shard))
print("H sample:", None if sample.h_true is None else sample.h_true.shape)
print("Y sample:", None if sample.y_pilot is None else sample.y_pilot.shape)
print("CKM scalar:", sample.scalar_feature.shape)
print("CKM ADP:", sample.angular_delay_power.shape)
