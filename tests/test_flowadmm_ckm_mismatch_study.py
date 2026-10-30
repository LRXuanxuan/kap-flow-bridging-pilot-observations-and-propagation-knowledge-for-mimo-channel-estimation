from __future__ import annotations

import pytest

import json
from pathlib import Path

pytest.importorskip('tools.run_flowadmm_ckm_mismatch_study', reason='optional study tool is not shipped in this release')
from tools.run_flowadmm_ckm_mismatch_study import (
    _protocol_differences,
    _with_runtime_backend,
    main,
)


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_dry_run_plans_a_paired_four_cell_study(tmp_path: Path) -> None:
    checkpoint = tmp_path / "standard.pt"
    checkpoint.write_bytes(b"not read by dry run")
    output_root = tmp_path / "study"
    main(
        [
            "--run-tag",
            "paired",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--standard-checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )
    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    assert manifest["selection_policy"] == "standard-shared"
    assert manifest["matrix"]["rows"] == ["standard_prior", "robust_prior"]
    assert manifest["matrix"]["columns"] == ["clean_ckm", "fixed_mismatched_ckm"]
    standard = manifest["records"]["standard"]
    robust = manifest["records"]["robust"]
    assert standard["stages"]["sweep"]["status"] == "planned"
    assert robust["stages"]["sweep"]["status"] == "shared_standard_selection"
    for record in (standard, robust):
        cells = record["stages"]["test"]["cells"]
        assert set(cells) == {"clean", "mismatch"}
        clean = cells["clean"]["command"]
        mismatch = cells["mismatch"]["command"]
        assert _argument(clean, "--condition-mode") == "clean"
        assert _argument(mismatch, "--condition-mode") == "mismatch"
        assert _argument(clean, "--condition-corruption-profile") == _argument(
            mismatch, "--condition-corruption-profile"
        )
        for flag in (
            "--pnp-iterations",
            "--flow-admm-tau",
            "--flow-admm-time-min",
            "--flow-admm-time-max",
            "--flow-admm-gamma",
            "--flow-admm-k-avg",
        ):
            assert _argument(clean, flag) == _argument(mismatch, flag)
        assert "training.seed=3407" in clean
        assert "training.validation_seed=9102" in clean


def test_checkpoint_protocol_canonicalizes_auto_only_for_current_backend() -> None:
    base = {
        "data": {"backend": "plain_h5", "test_scenes": ["scene_01"]},
        "condition_schema": {"scalar_dim": 6},
        "stored_normalization": {"h_rms": 1.0},
    }
    automatic = {
        **base,
        "data": {**base["data"], "backend": "auto"},
    }
    assert _protocol_differences(
        _with_runtime_backend(base, "plain_h5"),
        _with_runtime_backend(automatic, "plain_h5"),
    ) == []

    incompatible = {
        **automatic,
        "data": {**automatic["data"], "test_scenes": ["scene_02"]},
    }
    assert _protocol_differences(
        _with_runtime_backend(base, "plain_h5"),
        _with_runtime_backend(incompatible, "plain_h5"),
    ) == ["data.test_scenes (['scene_01'] != ['scene_02'])"]
