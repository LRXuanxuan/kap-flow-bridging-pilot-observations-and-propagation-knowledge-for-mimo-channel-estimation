from __future__ import annotations

import pytest

import json
from pathlib import Path

from ckmsrf.config import load_config
from ckmsrf.factory import build_model
pytest.importorskip('tools.run_straight_flow_study', reason='optional study tool is not shipped in this release')
from tools.run_straight_flow_study import (
    DEFAULT_CKM_CONFIG,
    DEFAULT_NO_CKM_CONFIG,
    PriorJob,
    build_cases,
    build_lmmse_statistics_command,
    build_lmmse_test_command,
    build_parser,
    build_sweep_command,
    build_test_command,
    main,
)


def test_no_ckm_straight_flow_config_keeps_capacity_matched_condition_interface() -> None:
    config = load_config(DEFAULT_NO_CKM_CONFIG)
    conditioned = load_config(DEFAULT_CKM_CONFIG)

    assert config["model"]["type"] == "straight_flow_matching"
    assert config["model"]["condition"]["enabled"] is True
    assert config["model"]["condition"]["force_unconditional"] is True
    assert config["loss"]["type"] == "straight_flow_matching"
    assert sum(parameter.numel() for parameter in build_model(config).parameters()) == sum(
        parameter.numel() for parameter in build_model(conditioned).parameters()
    )


def test_cross_scene_cases_cover_each_holdout_once() -> None:
    cases = build_cases("cross-scene", [1, 2, 3])

    assert [case.test_scenes for case in cases] == [(1,), (2,), (3,)]
    assert [case.train_scenes for case in cases] == [(2, 3), (1, 3), (1, 2)]
    assert all(case.validation_scenes == case.train_scenes for case in cases)
    assert len({case.normalization_file for case in cases}) == 3


def test_sweep_and_test_commands_recover_scientific_config_from_checkpoint(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--study",
            "in-domain",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    args.paired_test_seed = 9123
    case = build_cases("in-domain", [1, 2, 3])[0]
    job = PriorJob(
        case=case,
        condition="no-ckm",
        config=DEFAULT_NO_CKM_CONFIG,
        experiment_name="unit",
        run_id="unit",
        checkpoint=tmp_path / "best_nmse.pt",
    )

    sweep = build_sweep_command(args, job, tmp_path / "validation")
    test = build_test_command(
        args,
        job,
        "flow-admm",
        {
            "pnp_iterations": 8,
            "tau": 0.25,
            "time_min": 0.2,
            "time_max": 0.95,
            "gamma": 0.5,
            "k_avg": 1,
            "k_avg_schedule": "constant",
        },
        tmp_path / "test",
    )
    statistics = build_lmmse_statistics_command(
        args, case, tmp_path / "statistics.pt"
    )
    lmmse_test = build_lmmse_test_command(
        args, case, tmp_path / "statistics.pt", tmp_path / "classical"
    )

    assert "--config" not in sweep
    assert "--config" not in test
    assert sweep[sweep.index("--split") + 1] == "val"
    assert test[test.index("--split") + 1] == "test"
    assert test[test.index("--flow-admm-tau") + 1] == "0.25"
    assert statistics[statistics.index("--output") + 1] == str(tmp_path / "statistics.pt")
    assert "data.train_scenes=[scene_01, scene_02, scene_03]" in statistics
    assert lmmse_test[lmmse_test.index("--statistics-scope") + 1] == "global"
    assert "training.validation_seed=9123" in lmmse_test


def test_dry_run_creates_complete_in_domain_factor_plan(tmp_path: Path) -> None:
    output_root = tmp_path / "study"
    main(
        [
            "--study",
            "in-domain",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
            "--ckm-checkpoint",
            str(tmp_path / "existing_ckm.pt"),
            "--paired-test-seed",
            "9123",
            "--include-lmmse",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    assert manifest["study"] == "in-domain"
    assert manifest["paired_test_seed"] == 9123
    assert len(manifest["records"]) == 2
    ckm, no_ckm = manifest["records"]
    assert ckm["external_checkpoint"] is True
    assert ckm["stages"]["train"]["status"] == "external_checkpoint"
    assert no_ckm["stages"]["train"]["status"] == "planned"
    assert set(no_ckm["stages"]["test"]["solvers"]) == {"pnp-flow", "flow-admm"}
    for record in (ckm, no_ckm):
        for solver_record in record["stages"]["test"]["solvers"].values():
            command = solver_record["command"]
            assert "training.seed=3407" in command
            assert "training.validation_seed=9123" in command
    lmmse = manifest["classical_lmmse"]
    assert lmmse["enabled"] is True
    assert lmmse["records"][0]["statistics_scope"] == "global"
    assert lmmse["records"][0]["stages"]["fit_train_statistics"]["status"] == "planned"
    assert lmmse["records"][0]["stages"]["locked_test"]["status"] == "planned"


def test_dry_run_cross_scene_plans_requested_folds_only(tmp_path: Path) -> None:
    output_root = tmp_path / "cross"
    main(
        [
            "--study",
            "cross-scene",
            "--held-out-scenes",
            "1",
            "3",
            "--conditions",
            "ckm",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    assert [record["case"]["test_scenes"] for record in manifest["records"]] == [[1], [3]]
    assert all(record["stages"]["train"]["status"] == "planned" for record in manifest["records"])


def test_dry_run_cross_scene_lmmse_has_no_held_out_statistics(tmp_path: Path) -> None:
    output_root = tmp_path / "cross_lmmse"
    main(
        [
            "--study",
            "cross-scene",
            "--held-out-scenes",
            "2",
            "--conditions",
            "ckm",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
            "--include-lmmse",
            "--paired-test-seed",
            "9123",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    record = manifest["classical_lmmse"]["records"][0]
    assert record["case"]["train_scenes"] == [1, 3]
    assert record["case"]["test_scenes"] == [2]
    assert record["stages"]["fit_train_statistics"]["excluded_test_scenes"] == [2]
    command = record["stages"]["locked_test"]["command"]
    assert command[command.index("--statistics-scope") + 1] == "global"
    assert "data.train_scenes=[scene_01, scene_03]" in command
    assert "data.test_scenes=[scene_02]" in command
    assert "training.validation_seed=9123" in command


def test_lmmse_only_dry_run_skips_learned_validation_selection(tmp_path: Path) -> None:
    output_root = tmp_path / "lmmse_only"
    main(
        [
            "--study",
            "cross-scene",
            "--held-out-scenes",
            "3",
            "--conditions",
            "ckm",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
            "--include-lmmse",
            "--lmmse-only",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    assert manifest["lmmse_only"] is True
    assert manifest["records"][0]["stages"] == {}
    lmmse_stages = manifest["classical_lmmse"]["records"][0]["stages"]
    assert lmmse_stages["fit_train_statistics"]["status"] == "planned"
    assert lmmse_stages["locked_test"]["status"] == "planned"


def test_lmmse_only_resume_preserves_existing_learned_manifest_records(tmp_path: Path) -> None:
    output_root = tmp_path / "preserve_manifest"
    output_root.mkdir()
    prior_records = [{"condition": "ckm", "stages": {"sweep": {"status": "completed"}}}]
    (output_root / "protocol_manifest.json").write_text(
        json.dumps({"records": prior_records}), encoding="utf-8"
    )
    main(
        [
            "--study",
            "cross-scene",
            "--held-out-scenes",
            "1",
            "--run-tag",
            "unit",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-root",
            str(output_root),
            "--include-lmmse",
            "--lmmse-only",
            "--resume",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text("utf-8"))
    assert manifest["records"] == prior_records
