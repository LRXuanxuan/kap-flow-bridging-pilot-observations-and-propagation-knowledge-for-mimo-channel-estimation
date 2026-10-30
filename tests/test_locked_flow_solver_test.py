from __future__ import annotations

import pytest

import json
from pathlib import Path

pytest.importorskip('tools.run_locked_flow_solver_test', reason='optional study tool is not shipped in this release')
from tools.run_locked_flow_solver_test import main


def test_dry_run_locks_the_validation_selected_protocol(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_nmse.pt"
    checkpoint.write_bytes(b"not_loaded_during_dry_run")
    output_root = tmp_path / "locked_test"

    main(
        [
            "--config",
            "experiment.yaml",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "locked_test_manifest.json").read_text("utf-8"))
    assert manifest["split"] == "test"
    assert manifest["validation_selection_protocol"]["iteration_grid"] == "1:40"
    assert [record["status"] for record in manifest["records"]] == ["planned", "planned"]
    commands = {record["method"]: record["command"] for record in manifest["records"]}
    assert "--pnp-iterations" in commands["pnp-flow"]
    assert commands["pnp-flow"][commands["pnp-flow"].index("--pnp-iterations") + 1] == "40"
    assert "--flow-admm-tau" in commands["flow-admm"]
    assert commands["flow-admm"][commands["flow-admm"].index("--flow-admm-tau") + 1] == "0.25"
    assert commands["flow-admm"][commands["flow-admm"].index("--flow-admm-k-avg") + 1] == "1"


def test_dry_run_can_target_one_locked_solver(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_nmse.pt"
    checkpoint.write_bytes(b"checkpoint")
    output_root = tmp_path / "flow_admm_only"

    main(
        [
            "--config",
            "experiment.yaml",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--methods",
            "flow-admm",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "locked_test_manifest.json").read_text("utf-8"))
    assert [record["method"] for record in manifest["records"]] == ["flow-admm"]
