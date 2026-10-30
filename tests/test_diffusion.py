from __future__ import annotations

from pathlib import Path

import torch

from ckmsrf.config import load_config
from ckmsrf.factory import build_model, build_objective
from ckmsrf.losses import DiffusionDenoisingObjective


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    return load_config(
        PROJECT_ROOT / "configs/experiments/ckm_diffusion_prior.yaml",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides={
            "model": {
                "dim": 8,
                "dim_mults": [1, 2],
                "num_res_blocks": 1,
                "groups": 4,
                "condition": {
                    "embed_dim": 32,
                    "use_scalar": False,
                    "use_state": True,
                    "use_beam": False,
                    "use_adp": False,
                },
            }
        },
    )


def test_diffusion_config_builds_independent_denoising_objective() -> None:
    config = _small_config()
    assert isinstance(build_objective(config), DiffusionDenoisingObjective)


def test_diffusion_objective_is_finite_and_backpropagates() -> None:
    config = _small_config()
    model = build_model(config)
    objective = build_objective(config)
    batch = {
        "h": torch.randn(2, 2, 4, 4),
        "condition": {"state": torch.randn(2, 8)},
    }
    output = objective(model, batch)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["reconstruction_nmse"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_capacity_matched_unconditional_model_ignores_all_conditions() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/ckm_diffusion_no_ckm.yaml",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides={
            "model": {
                "dim": 8,
                "dim_mults": [1, 2],
                "num_res_blocks": 1,
                "groups": 4,
                "condition": {
                    "embed_dim": 32,
                    "use_scalar": False,
                    "use_state": True,
                    "use_beam": False,
                    "use_adp": False,
                    "use_time_frequency": False,
                },
            }
        },
    )
    model = build_model(config).eval()
    x = torch.randn(2, 2, 4, 4)
    sigma = torch.full((2,), 0.3)
    first = model(x, sigma, condition={"state": torch.randn(2, 8)})
    second = model(x, sigma, condition={"state": torch.randn(2, 8) + 100.0})
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_pnpdm_configs_change_only_inference_solver() -> None:
    for diffusion_name, pnpdm_name, force_unconditional in (
        ("ckm_diffusion_prior.yaml", "ckm_pnpdm_prior.yaml", False),
        ("ckm_diffusion_no_ckm.yaml", "ckm_pnpdm_no_ckm.yaml", True),
    ):
        diffusion = load_config(
            PROJECT_ROOT / "configs/experiments" / diffusion_name,
            project_root=PROJECT_ROOT,
            environ={},
        )
        pnpdm = load_config(
            PROJECT_ROOT / "configs/experiments" / pnpdm_name,
            project_root=PROJECT_ROOT,
            environ={},
        )
        for key in ("model", "loss", "consistency", "diffusion"):
            assert pnpdm[key] == diffusion[key]
        assert pnpdm["sampling"]["solver"] == "pnp_dm"
        assert bool(
            pnpdm["model"]["condition"].get("force_unconditional", False)
        ) is force_unconditional
