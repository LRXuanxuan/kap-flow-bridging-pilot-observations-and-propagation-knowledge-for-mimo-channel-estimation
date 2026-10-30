from __future__ import annotations

from pathlib import Path

import torch

from ckmsrf.config import load_config
from ckmsrf.factory import build_model, build_objective, load_initial_weights
from ckmsrf.losses import ConsistencyObjective, GMFlowObjective
from ckmsrf.models import ConditionalConsistencyUNet, ConditionalGMUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_config_still_builds_gmflow() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/ckmsrf_prior.yaml",
        project_root=PROJECT_ROOT,
        environ={},
    )
    assert isinstance(build_model(config), ConditionalGMUNet)
    assert isinstance(build_objective(config), GMFlowObjective)


def test_consistency_config_builds_cm_and_objective() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/ckmcm_prior.yaml",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides={
            "model": {
                "dim": 8,
                "dim_mults": [1, 2],
                "num_res_blocks": 1,
                "groups": 4,
                "condition": {"embed_dim": 32},
            }
        },
    )
    model = build_model(config)
    objective = build_objective(config)
    assert isinstance(model, ConditionalConsistencyUNet)
    assert isinstance(objective, ConsistencyObjective)
    assert model.sigma_min == config["consistency"]["sigma_min"]
    assert objective.num_scales == config["consistency"]["num_scales"]


def test_gmflow_checkpoint_warm_starts_shared_cm_layers(tmp_path: Path) -> None:
    shared_override = {
        "model": {
            "dim": 8,
            "dim_mults": [1, 2],
            "num_res_blocks": 1,
            "groups": 4,
            "condition": {"embed_dim": 32},
        }
    }
    gm_config = load_config(
        PROJECT_ROOT / "configs/experiments/ckmsrf_prior.yaml",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides=shared_override,
    )
    cm_config = load_config(
        PROJECT_ROOT / "configs/experiments/ckmcm_prior.yaml",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides=shared_override,
    )
    gm_model = build_model(gm_config)
    cm_model = build_model(cm_config)
    checkpoint = tmp_path / "gmflow.pt"
    torch.save({"model": gm_model.state_dict()}, checkpoint)

    missing, skipped = load_initial_weights(cm_model, checkpoint, strict=False)

    torch.testing.assert_close(cm_model.init_conv.weight, gm_model.init_conv.weight)
    assert "final_conv.weight" in missing
    assert any(name.startswith("final_conv") for name in skipped)
