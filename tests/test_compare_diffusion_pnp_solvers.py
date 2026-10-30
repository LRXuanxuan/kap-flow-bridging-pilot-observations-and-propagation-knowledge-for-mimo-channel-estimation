from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip('tools.compare_diffusion_pnp_solvers', reason='optional study tool is not shipped in this release')
from tools.compare_diffusion_pnp_solvers import main as compare_main


def _write_metrics(
    path: Path,
    *,
    iterations: tuple[int, ...],
    nmse_offset_db: float,
    prior_steps: int | None,
) -> None:
    rows = []
    for iteration in iterations:
        for snr_db in (0.0, 10.0):
            nmse_db = nmse_offset_db - 0.01 * iteration
            row = {
                "pnp_iterations": iteration,
                "data_consistency_evaluations": iteration,
                "denoiser_nfe": (
                    "" if prior_steps is None else iteration * prior_steps // 2
                ),
                "posterior_sample_count": (
                    "" if prior_steps is None else max(1, (iteration - 40) // 3)
                ),
                "pnpdm_prior_steps": "" if prior_steps is None else prior_steps,
                "sampling_seconds_cumulative": 0.02 * iteration,
                "snr_db": snr_db,
                "scene_id": "overall",
                "num_samples": 6,
                "nmse_linear": 10.0 ** (nmse_db / 10.0),
                "nmse_db": nmse_db,
                "normalized_measurement_residual": 0.9,
                "seconds": 1.0,
            }
            rows.append(row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_results(
    path: Path,
    *,
    method: str,
    prior_steps: int | None,
    max_samples: int = 6,
    chain_seed: int = 20_009_102,
) -> None:
    payload = {
        "checkpoint": "/shared/diffusion.pt",
        "split": "test",
        "prior": {
            "method": "diffusion",
            "model_type": "consistency",
            "output_parameterization": "clean",
            "weights_source": "ema",
            "parameter_count": 1234,
        },
        "evaluation": {
            "validation_seed": 9102,
            "max_samples": max_samples,
            "max_batches": None,
            "eval_batch_size": 4,
            "precision": "fp32",
            "world_size": 1,
            "normalization": {"h_rms": 1.0},
        },
        "sampler": {
            "name": method,
            "pnp_iterations": 100 if method == "pnp-dm" else 4,
            "pnp_dm": (
                {
                    "prior_steps": prior_steps,
                    "prior_solver": "sde",
                    "estimate_mode": "posterior_mean",
                    "burn_in": 40,
                    "thin": 3,
                    "seed": chain_seed,
                }
                if method == "pnp-dm"
                else None
            ),
        },
        "config": {
            "paths": {"data_root": "/shared/data"},
            "training": {"seed": 3407, "validation_seed": 9102},
            "data": {
                "backend": "plain_h5",
                "index_file": "indices/{split}.json",
                "test_scenes": ["scene_01"],
                "normalization": {"mode": "global_rms"},
            },
            "model": {
                "type": "consistency",
                "condition": {"enabled": True, "force_unconditional": False},
            },
            "loss": {"type": "diffusion"},
            "consistency": {"sigma_min": 0.002, "sigma_max": 2.0},
            "diffusion": {"weighting": "edm"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _method_output(
    root: Path,
    name: str,
    *,
    method: str,
    iterations: tuple[int, ...],
    nmse_offset_db: float,
    prior_steps: int | None,
    max_samples: int = 6,
    chain_seed: int = 20_009_102,
) -> Path:
    directory = root / name
    directory.mkdir()
    metrics = directory / "metrics.csv"
    _write_metrics(
        metrics,
        iterations=iterations,
        nmse_offset_db=nmse_offset_db,
        prior_steps=prior_steps,
    )
    _write_results(
        directory / "results.json",
        method=method,
        prior_steps=prior_steps,
        max_samples=max_samples,
        chain_seed=chain_seed,
    )
    return metrics


def test_compare_diffusion_solvers_writes_nfe_and_best_tables(
    tmp_path: Path,
) -> None:
    admm = _method_output(
        tmp_path,
        "admm",
        method="pnp-cm",
        iterations=(2, 4),
        nmse_offset_db=0.0,
        prior_steps=None,
    )
    pnpdm20 = _method_output(
        tmp_path,
        "pnpdm20",
        method="pnp-dm",
        iterations=(43, 100),
        nmse_offset_db=-0.2,
        prior_steps=20,
    )
    pnpdm100 = _method_output(
        tmp_path,
        "pnpdm100",
        method="pnp-dm",
        iterations=(43, 100),
        nmse_offset_db=-0.5,
        prior_steps=100,
    )
    output = tmp_path / "comparison"

    compare_main(
        [
            "--pnp-admm-metrics",
            str(admm),
            "--pnpdm-metrics",
            str(pnpdm20),
            str(pnpdm100),
            "--output-dir",
            str(output),
        ]
    )

    with (output / "quality_compute_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        summary = list(csv.DictReader(handle))
    assert len(summary) == 6
    admm_rows = [row for row in summary if row["method"] == "pnp_admm_diffusion"]
    assert [int(row["denoiser_nfe"]) for row in admm_rows] == [2, 4]
    assert all(row["nfe_inferred"] == "True" for row in admm_rows)
    pnpdm_rows = [row for row in summary if row["method"] == "pnp_dm"]
    assert {int(row["pnpdm_prior_steps"]) for row in pnpdm_rows} == {20, 100}
    assert all(row["estimator"] == "posterior_mean" for row in pnpdm_rows)

    with (output / "best_by_method.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        best = {row["method"]: row for row in csv.DictReader(handle)}
    assert set(best) == {"pnp_admm_diffusion", "pnp_dm"}
    assert int(best["pnp_dm"]["pnpdm_prior_steps"]) == 100
    assert (output / "nfe_pareto_frontier.csv").is_file()


def test_compare_diffusion_solvers_rejects_protocol_mismatch(
    tmp_path: Path,
) -> None:
    admm = _method_output(
        tmp_path,
        "admm",
        method="pnp-cm",
        iterations=(4,),
        nmse_offset_db=0.0,
        prior_steps=None,
    )
    pnpdm = _method_output(
        tmp_path,
        "pnpdm",
        method="pnp-dm",
        iterations=(43,),
        nmse_offset_db=-0.2,
        prior_steps=20,
        max_samples=5,
    )

    with pytest.raises(ValueError, match="max_samples"):
        compare_main(
            [
                "--pnp-admm-metrics",
                str(admm),
                "--pnpdm-metrics",
                str(pnpdm),
                "--output-dir",
                str(tmp_path / "comparison"),
            ]
        )


def test_compare_diffusion_solvers_aggregates_chain_seeds(tmp_path: Path) -> None:
    admm = _method_output(
        tmp_path,
        "admm",
        method="pnp-cm",
        iterations=(4,),
        nmse_offset_db=0.0,
        prior_steps=None,
    )
    first = _method_output(
        tmp_path,
        "pnpdm_seed1",
        method="pnp-dm",
        iterations=(43,),
        nmse_offset_db=-0.2,
        prior_steps=20,
        chain_seed=101,
    )
    second = _method_output(
        tmp_path,
        "pnpdm_seed2",
        method="pnp-dm",
        iterations=(43,),
        nmse_offset_db=0.2,
        prior_steps=20,
        chain_seed=202,
    )
    output = tmp_path / "comparison"

    compare_main(
        [
            "--pnp-admm-metrics",
            str(admm),
            "--pnpdm-metrics",
            str(first),
            str(second),
            "--output-dir",
            str(output),
        ]
    )

    with (output / "quality_compute_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    pnpdm = next(row for row in rows if row["method"] == "pnp_dm")
    assert int(pnpdm["replicate_count"]) == 2
    assert pnpdm["algorithm_seeds"] == "101,202"
    assert float(pnpdm["aggregate_nmse_db_std_across_runs"]) > 0.0
    assert (output / "quality_compute_runs.csv").is_file()
