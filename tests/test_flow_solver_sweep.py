from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip('tools.sweep_flow_solvers', reason='optional study tool is not shipped in this release')
from tools.sweep_flow_solvers import (
    _read_candidate_metrics,
    build_candidates,
    build_parser,
    main,
)


def _args() -> object:
    return build_parser().parse_args(
        ["--config", "experiment.yaml", "--checkpoint", "best_nmse.pt"]
    )


def test_default_candidate_grid_has_independent_solver_products() -> None:
    candidates = build_candidates(_args())

    assert len(candidates) == 18
    assert sum(item.solver == "pnp-flow" for item in candidates) == 6
    assert sum(item.solver == "flow-admm" for item in candidates) == 12
    assert len({item.identifier for item in candidates}) == len(candidates)


def test_metric_aggregation_uses_linear_nmse_average(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    rows = [
        {
            "pnp_iterations": 1,
            "snr_db": 0.0,
            "scene_id": "overall",
            "nmse_linear": 0.1,
            "normalized_measurement_residual": 0.5,
            "sampling_seconds": 1.0,
            "denoiser_nfe": 1,
        },
        {
            "pnp_iterations": 1,
            "snr_db": 10.0,
            "scene_id": "overall",
            "nmse_linear": 0.01,
            "normalized_measurement_residual": 0.3,
            "sampling_seconds": 2.0,
            "denoiser_nfe": 1,
        },
        {
            "pnp_iterations": 2,
            "snr_db": 0.0,
            "scene_id": "overall",
            "nmse_linear": 0.02,
            "normalized_measurement_residual": 0.2,
            "sampling_seconds": 3.0,
            "denoiser_nfe": 2,
        },
    ]
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = _read_candidate_metrics(metrics_path)

    assert [row["pnp_iterations"] for row in summary] == [1, 2]
    assert summary[0]["mean_nmse_linear"] == pytest.approx(0.055)
    assert summary[0]["aggregate_nmse_db"] == pytest.approx(-12.59637, abs=1e-4)
    assert summary[0]["mean_normalized_measurement_residual"] == pytest.approx(0.4)
    assert summary[0]["total_sampler_seconds"] == pytest.approx(3.0)
    assert summary[1]["denoiser_nfe"] == pytest.approx(2.0)


def test_dry_run_writes_an_executable_plan(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_nmse.pt"
    checkpoint.write_bytes(b"not_loaded_during_dry_run")
    output_root = tmp_path / "sweep"

    main(
        [
            "--config",
            "experiment.yaml",
            "--checkpoint",
            str(checkpoint),
            "--output-root",
            str(output_root),
            "--solvers",
            "pnp-flow",
            "--iteration-grid",
            "1",
            "2",
            "--dry-run",
        ]
    )

    manifest = json.loads((output_root / "sweep_manifest.json").read_text("utf-8"))
    assert manifest["dry_run"] is True
    assert len(manifest["records"]) == 6
    assert all(item["status"] == "planned" for item in manifest["records"])
    assert all("--sampler" in item["command"] for item in manifest["records"])


def test_test_split_requires_explicit_opt_in(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_nmse.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="--allow-test-sweep"):
        main(
            [
                "--config",
                "experiment.yaml",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--dry-run",
            ]
        )
