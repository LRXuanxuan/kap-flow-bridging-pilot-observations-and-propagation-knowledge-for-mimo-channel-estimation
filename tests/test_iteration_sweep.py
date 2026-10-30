from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ckmsrf.evaluate import _parse_iteration_grid
pytest.importorskip('tools.compare_pnp_iteration_sweeps', reason='optional study tool is not shipped in this release')
from tools.compare_pnp_iteration_sweeps import main as compare_main


def _write_metrics(path: Path, nmse_offset_db: float) -> None:
    rows = []
    for iteration in (1, 2):
        for snr_db in (0.0, 10.0):
            nmse_db = -float(iteration) + nmse_offset_db
            rows.append(
                {
                    "pnp_iterations": iteration,
                    "sampling_seconds_cumulative": 0.1 * iteration,
                    "snr_db": snr_db,
                    "scene_id": "overall",
                    "num_samples": 6,
                    "nmse_linear": 10.0 ** (nmse_db / 10.0),
                    "nmse_db": nmse_db,
                    "normalized_measurement_residual": 1.0,
                    "seconds": 1.0,
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_results(
    path: Path,
    *,
    method: str,
    pnp_lambda: float = 0.1,
) -> None:
    model_type = {
        "cm": "consistency",
        "diffusion": "consistency",
        "flow": "flow_matching",
    }[method]
    loss_type = {
        "cm": "consistency",
        "diffusion": "diffusion",
        "flow": "flow_matching",
    }[method]
    payload = {
        "split": "test",
        "sampler": {
            # pnp-cm and pnp-admm are aliases for the same core trajectory and
            # are intentionally not part of the protocol equality check.
            "name": "pnp-admm" if method == "flow" else "pnp-cm",
            "pnp_iteration_grid": [1, 2],
            "trajectory_reuse": True,
            "pnp_rho": 1.0,
            "pnp_lambda": pnp_lambda,
            "pnp_adaptive_rho": False,
            "pnp_momentum": 0.0,
            "pnp_denoiser_relaxation": 1.0,
            "pnp_dual_damping": 1.0,
            "pnp_final_data_step": False,
            "pnp_rho_candidate_factors": [0.5, 1.0, 2.0],
            "pnp_rho_min": 1.0e-6,
            "pnp_rho_max": 1.0e6,
            "pnp_whiteness_weight": 0.25,
        },
        "evaluation": {
            "validation_seed": 9102,
            "max_samples": 12,
            "max_batches": None,
            "world_size": 1,
            "normalization": {"h_rms": 1.0, "split": "train"},
        },
        "prior": {
            "weights_source": "ema",
            "parameter_count": 1234,
        },
        "config": {
            "training": {"seed": 3407, "validation_seed": 9102},
            "data": {
                "backend": "plain_h5",
                "index_file": "indices/{split}.json",
                "validation_scenes": ["scene_01"],
                "test_scenes": ["scene_01"],
                "validation_sampling": "fixed",
                "test_sampling": "fixed",
                "normalization": {
                    "mode": "global_rms",
                    "stats_file": "normalization.json",
                },
            },
            "model": {
                "type": model_type,
                "in_channels": 2,
                "out_channels": 2,
                "dim": 8,
                "dim_mults": [1, 2],
                "groups": 4,
                "condition": {
                    "enabled": True,
                    "embed_dim": 32,
                    "use_state": True,
                },
            },
            "loss": {"type": loss_type},
            "consistency": {
                "sigma_min": 0.002,
                "sigma_max": 2.0,
                "sigma_data": 1.0,
            },
            "hardware": {"eval_batch_size": 4, "precision": "fp32"},
            "sampling": {
                "pnp_rho_candidate_factors": [0.5, 1.0, 2.0],
                "pnp_rho_min": 1.0e-6,
                "pnp_rho_max": 1.0e6,
                "pnp_whiteness_weight": 0.25,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_method_output(
    root: Path,
    method: str,
    *,
    nmse_offset_db: float,
    pnp_lambda: float = 0.1,
) -> Path:
    method_dir = root / method
    method_dir.mkdir(parents=True)
    metrics = method_dir / "metrics.csv"
    _write_metrics(metrics, nmse_offset_db)
    _write_results(
        method_dir / "results.json",
        method=method,
        pnp_lambda=pnp_lambda,
    )
    return metrics


def test_iteration_grid_accepts_ranges_lists_and_deduplicates() -> None:
    assert _parse_iteration_grid(["1:3", "3,5", "7:9:2"]) == (1, 2, 3, 5, 7, 9)


def test_compare_iteration_sweeps_writes_aligned_tables(tmp_path: Path) -> None:
    cm_path = tmp_path / "cm.csv"
    diffusion_path = tmp_path / "diffusion.csv"
    output_dir = tmp_path / "comparison"
    _write_metrics(cm_path, nmse_offset_db=0.0)
    _write_metrics(diffusion_path, nmse_offset_db=1.0)

    with pytest.warns(RuntimeWarning, match="Protocol verification is incomplete"):
        compare_main(
            [
                "--cm-metrics",
                str(cm_path),
                "--diffusion-metrics",
                str(diffusion_path),
                "--output-dir",
                str(output_dir),
            ]
        )

    with (output_dir / "comparison_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["pnp_iterations"]) for row in rows] == [1, 2]
    assert all(float(row["cm_gain_over_diffusion_db"]) > 0.0 for row in rows)
    assert (output_dir / "comparison_per_snr.csv").is_file()
    assert (output_dir / "comparison_best.csv").is_file()


def test_compare_three_sweeps_writes_flow_gains_nfe_and_best_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cm_path = _write_method_output(
        tmp_path, "cm", nmse_offset_db=0.0
    )
    diffusion_path = _write_method_output(
        tmp_path, "diffusion", nmse_offset_db=1.0
    )
    # A lower offset makes Flow the best method at each matched K.
    flow_path = _write_method_output(
        tmp_path, "flow", nmse_offset_db=-1.0
    )
    output_dir = tmp_path / "comparison"

    compare_main(
        [
            "--cm-metrics",
            str(cm_path),
            "--diffusion-metrics",
            str(diffusion_path),
            "--flow-metrics",
            str(flow_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    with (output_dir / "comparison_per_snr.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        per_snr = list(csv.DictReader(handle))
    assert per_snr
    assert all("flow_nmse_db" in row for row in per_snr)
    assert all(float(row["cm_gain_over_diffusion_db"]) > 0.0 for row in per_snr)
    assert all(float(row["cm_gain_over_flow_db"]) < 0.0 for row in per_snr)
    assert all(float(row["diffusion_gain_over_flow_db"]) < 0.0 for row in per_snr)
    assert [float(row["flow_denoiser_nfe"]) for row in per_snr] == [1.0, 1.0, 2.0, 2.0]
    assert all(row["flow_nfe_inferred"] == "True" for row in per_snr)

    with (output_dir / "comparison_best.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        best = {row["method"]: row for row in csv.DictReader(handle)}
    assert set(best) == {"cm", "diffusion", "flow"}
    assert all(int(row["best_pnp_iterations"]) == 2 for row in best.values())
    assert float(best["flow"]["best_aggregate_nmse_db"]) < float(
        best["cm"]["best_aggregate_nmse_db"]
    )
    assert (
        "Best PnP-ADMM + additive-Flow denoiser: K=2"
        in capsys.readouterr().out
    )


def test_protocol_mismatch_fails_unless_explicitly_allowed(tmp_path: Path) -> None:
    cm_path = _write_method_output(
        tmp_path, "cm", nmse_offset_db=0.0
    )
    diffusion_path = _write_method_output(
        tmp_path, "diffusion", nmse_offset_db=1.0
    )
    flow_path = _write_method_output(
        tmp_path,
        "flow",
        nmse_offset_db=-1.0,
        pnp_lambda=0.2,
    )

    common = [
        "--cm-metrics",
        str(cm_path),
        "--diffusion-metrics",
        str(diffusion_path),
        "--flow-metrics",
        str(flow_path),
    ]
    with pytest.raises(ValueError, match="pnp_lambda"):
        compare_main(common + ["--output-dir", str(tmp_path / "rejected")])

    accepted = tmp_path / "accepted"
    with pytest.warns(RuntimeWarning, match="Evaluation protocol mismatch"):
        compare_main(
            common
            + [
                "--allow-protocol-mismatch",
                "--output-dir",
                str(accepted),
            ]
        )
    assert (accepted / "comparison_summary.csv").is_file()
    assert (accepted / "comparison_best.csv").is_file()
