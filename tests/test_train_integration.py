from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ckmsrf.evaluate import main as evaluate_main
from ckmsrf.evaluate_classical import main as evaluate_classical_main
from ckmsrf.fit_channel_stats import main as fit_channel_stats_main
from ckmsrf.train import main
from test_data_synthetic import _make_scene, _make_shard


def _make_dataset(root: Path) -> None:
    root.mkdir()
    for scene_id in (1, 2, 3):
        _make_scene(root, scene_id, nt=3, npilot=2, banks=2)
        _make_shard(root, scene_id, "train")
        _make_shard(root, scene_id, "val")


def test_one_step_training_and_checkpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)

    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/baseline_no_ckm.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "pytest",
            "--max-steps",
            "1",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_components=1",
            "--set",
            "model.condition.enabled=false",
            "--set",
            "data.normalization.max_trajectories=1",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
            "--set",
            "training.log_every_steps=1",
        ]
    )

    run_dir = run_root / "baseline_no_ckm" / "pytest"
    checkpoint = run_dir / "checkpoints" / "last.pt"
    assert checkpoint.is_file()
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["optimizer"]
    assert payload["ema"]["shadow"]
    assert payload["normalization"]["split"] == "train"


def test_conditional_physics_training_step(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/ckmsrf_full.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "conditional",
            "--max-steps",
            "1",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_components=1",
            "--set",
            "model.condition.embed_dim=32",
            "--set",
            "model.condition.tx_beam_dim=3",
            "--set",
            "model.condition.rx_beam_dim=2",
            "--set",
            "model.condition.adp_shape=[2,3,2]",
            "--set",
            "hardware.gradient_checkpointing=true",
            "--set",
            "data.normalization.max_trajectories=1",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
        ]
    )
    payload = torch.load(
        run_root / "ckmsrf_full/conditional/checkpoints/last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["global_step"] == 1
    assert any(name.startswith("condition_encoder") for name in payload["model"])

    output_dir = tmp_path / "evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(run_root / "ckmsrf_full/conditional/checkpoints/last.pt"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--steps",
            "2",
            "--max-batches",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "results.json").is_file()


def test_robust_straight_flow_mismatch_evaluation_records_fixed_condition_protocol(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/ckm_straight_flow_robust.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "robust",
            "--max-steps",
            "1",
            "--set",
            "data.backend=plain_h5",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_res_blocks=1",
            "--set",
            "model.groups=4",
            "--set",
            "model.condition.embed_dim=32",
            "--set",
            "model.condition.tx_beam_dim=3",
            "--set",
            "model.condition.rx_beam_dim=2",
            "--set",
            "model.condition.adp_shape=[2,3,2]",
            "--set",
            "data.normalization.max_trajectories=1",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
            "--set",
            "training.log_every_steps=1",
        ]
    )
    checkpoint = run_root / "ckm_straight_flow_robust/robust/checkpoints/best_nmse.pt"
    assert checkpoint.is_file()
    output_dir = tmp_path / "mismatch_evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "flow-admm",
            "--pnp-iterations",
            "1",
            "--flow-admm-k-avg",
            "1",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--condition-mode",
            "mismatch",
            "--condition-corruption-profile",
            str(
                project_root
                / "configs/experiments/ckm_condition_mismatch_medium.yaml"
            ),
            "--output-dir",
            str(output_dir),
        ]
    )
    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    condition = payload["evaluation"]["condition"]
    assert condition["mode"] == "mismatch"
    assert condition["profile_fingerprint"]
    assert condition["snr_independent"] is True
    assert condition["batch_partition_independent"] is True
    assert condition["channel_pilot_observation_modified"] is False


def test_classical_fit_and_evaluation_cli(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    stats_path = run_root / "classical" / "stats.pt"
    common = [
        "--config",
        str(project_root / "configs/experiments/ckmcm_prior.yaml"),
        "--profile",
        str(project_root / "configs/hardware/cpu_debug.yaml"),
        "--data-root",
        str(data_root),
        "--run-root",
        str(run_root),
        "--set",
        "data.backend=plain_h5",
    ]
    fit_channel_stats_main(
        common
        + [
            "--output",
            str(stats_path),
            "--passes",
            "1",
            "--num-workers",
            "0",
        ]
    )
    assert stats_path.is_file()
    output_dir = run_root / "classical" / "evaluation"
    evaluate_classical_main(
        common
        + [
            "--statistics",
            str(stats_path),
            "--statistics-scope",
            "global",
            "--methods",
            "ls",
            "lmmse",
            "gaussian-mmse",
            "--split",
            "val",
            "--snr-db",
            "0",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert (output_dir / "metrics_all.csv").is_file()
    assert (output_dir / "ls" / "metrics.csv").is_file()
    assert (output_dir / "lmmse" / "results.json").is_file()


def test_one_step_diffusion_training(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/ckm_diffusion_prior.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "diffusion",
            "--max-steps",
            "1",
            "--set",
            "data.backend=plain_h5",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_res_blocks=1",
            "--set",
            "model.groups=4",
            "--set",
            "model.condition.embed_dim=32",
            "--set",
            "model.condition.tx_beam_dim=3",
            "--set",
            "model.condition.rx_beam_dim=2",
            "--set",
            "model.condition.adp_shape=[2,3,2]",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
            "--set",
            "training.log_every_steps=1",
        ]
    )
    checkpoint = run_root / "ckm_diffusion_prior/diffusion/checkpoints/best_nmse.pt"
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["loss"]["type"] == "diffusion"

    output_dir = tmp_path / "diffusion_iteration_sweep"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "pnp-cm",
            "--pnp-iteration-grid",
            "1:3",
            "--pnp-rho",
            "1.0",
            "--pnp-lambda",
            "0.1",
            "--no-adaptive-rho",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )
    with (output_dir / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overall = [row for row in rows if row["scene_id"] == "overall"]
    assert [int(row["pnp_iterations"]) for row in overall] == [1, 2, 3]
    assert all(float(row["sampling_seconds_cumulative"]) >= 0.0 for row in overall)
    metadata = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert metadata["sampler"]["pnp_iteration_grid"] == [1, 2, 3]
    assert metadata["sampler"]["trajectory_reuse"] is True

    pnpdm_output = tmp_path / "pnpdm_iteration_sweep"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "pnp-dm",
            "--pnp-iteration-grid",
            "1:2",
            "--pnpdm-rho-start",
            "0.2",
            "--pnpdm-rho-min",
            "0.2",
            "--pnpdm-rho-decay",
            "1.0",
            "--pnpdm-prior-steps",
            "4",
            "--pnpdm-prior-solver",
            "pf-ode",
            "--pnpdm-estimate-mode",
            "posterior-mean",
            "--pnpdm-burn-in",
            "0",
            "--pnpdm-thin",
            "1",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(pnpdm_output),
        ]
    )
    with (pnpdm_output / "metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        pnpdm_rows = list(csv.DictReader(handle))
    pnpdm_overall = [row for row in pnpdm_rows if row["scene_id"] == "overall"]
    assert [int(row["pnp_iterations"]) for row in pnpdm_overall] == [1, 2]
    assert [int(row["posterior_sample_count"]) for row in pnpdm_overall] == [1, 2]
    assert int(pnpdm_overall[1]["denoiser_nfe"]) > int(
        pnpdm_overall[0]["denoiser_nfe"]
    )
    pnpdm_metadata = json.loads(
        (pnpdm_output / "results.json").read_text(encoding="utf-8")
    )
    assert pnpdm_metadata["sampler"]["name"] == "pnp-dm"
    assert pnpdm_metadata["sampler"]["solver_family"] == "split_gibbs_pnp_dm"
    assert pnpdm_metadata["sampler"]["trajectory_reuse"] is True
    assert pnpdm_metadata["sampler"]["pnp_dm"]["prior_steps"] == 4


def test_one_step_flow_training_and_pnp_admm_iteration_sweep(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/ckm_flow_prior.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "flow",
            "--max-steps",
            "1",
            "--set",
            "data.backend=plain_h5",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_res_blocks=1",
            "--set",
            "model.groups=4",
            "--set",
            "model.condition.embed_dim=32",
            "--set",
            "model.condition.tx_beam_dim=3",
            "--set",
            "model.condition.rx_beam_dim=2",
            "--set",
            "model.condition.adp_shape=[2,3,2]",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
            "--set",
            "training.log_every_steps=1",
        ]
    )

    checkpoint = run_root / "ckm_flow_prior/flow/checkpoints/best_nmse.pt"
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["global_step"] == 1
    assert payload["config"]["model"]["type"] == "flow_matching"
    assert payload["config"]["loss"]["type"] == "flow_matching"

    output_dir = tmp_path / "flow_iteration_sweep"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "pnp-admm",
            "--pnp-iteration-grid",
            "1:3",
            "--pnp-rho",
            "1.0",
            "--pnp-lambda",
            "0.1",
            "--no-adaptive-rho",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )
    with (output_dir / "metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    overall = [row for row in rows if row["scene_id"] == "overall"]
    assert [int(row["pnp_iterations"]) for row in overall] == [1, 2, 3]
    assert [int(row["denoiser_nfe"]) for row in overall] == [1, 2, 3]
    assert all(float(row["sampling_seconds_cumulative"]) >= 0.0 for row in overall)

    metadata = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert metadata["sampler"]["name"] == "pnp-admm"
    assert metadata["sampler"]["pnp_iteration_grid"] == [1, 2, 3]
    assert metadata["sampler"]["trajectory_reuse"] is True
    assert metadata["sampler"]["denoiser_nfe_per_iteration"] == 1
    assert metadata["prior"]["method"] == "flow_matching"
    assert metadata["prior"]["output_parameterization"] == "velocity"
    assert metadata["prior"]["parameter_count"] > 0
    assert metadata["prior"]["weights_source"] == "ema"
    assert metadata["config"]["model"]["type"] == "flow_matching"


def test_one_step_straight_flow_training_and_independent_solver_sweeps(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    _make_dataset(data_root)
    project_root = Path(__file__).resolve().parents[1]
    main(
        [
            "--config",
            str(project_root / "configs/experiments/ckm_straight_flow_prior.yaml"),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--data-root",
            str(data_root),
            "--run-root",
            str(run_root),
            "--run-id",
            "straight",
            "--max-steps",
            "1",
            "--set",
            "data.backend=plain_h5",
            "--set",
            "model.dim=8",
            "--set",
            "model.dim_mults=[1,2]",
            "--set",
            "model.num_res_blocks=1",
            "--set",
            "model.groups=4",
            "--set",
            "model.condition.embed_dim=32",
            "--set",
            "model.condition.tx_beam_dim=3",
            "--set",
            "model.condition.rx_beam_dim=2",
            "--set",
            "model.condition.adp_shape=[2,3,2]",
            "--set",
            "training.epochs=1",
            "--set",
            "training.max_train_samples=2",
            "--set",
            "training.max_val_samples=1",
            "--set",
            "training.max_val_batches=1",
            "--set",
            "training.log_every_steps=1",
        ]
    )
    checkpoint = (
        run_root
        / "ckm_straight_flow_prior/straight/checkpoints/best_nmse.pt"
    )
    assert checkpoint.is_file()

    pnp_output = tmp_path / "pnp_flow_sweep"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "pnp-flow",
            "--pnp-iteration-grid",
            "1:2",
            "--pnp-flow-latent-samples",
            "1",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(pnp_output),
        ]
    )
    with (pnp_output / "metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    overall = [row for row in rows if row["scene_id"] == "overall"]
    assert [int(row["pnp_iterations"]) for row in overall] == [1, 2]
    assert [int(row["denoiser_nfe"]) for row in overall] == [1, 2]
    assert all(float(row["sampling_seconds"]) >= 0.0 for row in overall)
    pnp_metadata = json.loads(
        (pnp_output / "results.json").read_text(encoding="utf-8")
    )
    assert pnp_metadata["sampler"]["name"] == "pnp-flow"
    assert pnp_metadata["sampler"]["trajectory_reuse"] is False
    assert (
        pnp_metadata["sampler"]["timing_semantics"]
        == "independent_run_per_iteration_count"
    )
    assert pnp_metadata["sampler"]["solver_family"] == "pnp_flow_fbs"
    assert pnp_metadata["prior"]["output_parameterization"] == "straight_velocity"
    assert pnp_metadata["prior"]["flow_path"] == "straight_latent_to_data"
    assert pnp_metadata["prior"]["flow_coupling"] == "minibatch_ot"

    admm_output = tmp_path / "flow_admm"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--profile",
            str(project_root / "configs/hardware/cpu_debug.yaml"),
            "--split",
            "val",
            "--snr-db",
            "0",
            "--sampler",
            "flow-admm",
            "--pnp-iterations",
            "2",
            "--flow-admm-k-avg",
            "2",
            "--max-batches",
            "1",
            "--num-workers",
            "0",
            "--output-dir",
            str(admm_output),
        ]
    )
    admm_metadata = json.loads(
        (admm_output / "results.json").read_text(encoding="utf-8")
    )
    assert admm_metadata["sampler"]["name"] == "flow-admm"
    assert admm_metadata["sampler"]["solver_family"] == "flow_admm"
    admm_rows = admm_metadata["rows"]
    overall_admm = [row for row in admm_rows if row["scene_id"] == "overall"]
    assert overall_admm[0]["denoiser_nfe"] == 4
