from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch

from ckmsrf.config import load_config
from ckmsrf.factory import build_model, build_objective
from ckmsrf.losses import FlowMatchingObjective
from ckmsrf.models import ConditionalFlowMatchingUNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _override() -> dict:
    return {
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
    }


def _config(name: str) -> dict:
    return load_config(
        PROJECT_ROOT / f"configs/experiments/{name}",
        project_root=PROJECT_ROOT,
        environ={},
        cli_overrides=_override(),
    )


def test_flow_config_builds_velocity_model_and_objective() -> None:
    config = _config("ckm_flow_prior.yaml")
    model = build_model(config)
    objective = build_objective(config)
    assert isinstance(model, ConditionalFlowMatchingUNet)
    assert isinstance(objective, FlowMatchingObjective)
    assert model.output_parameterization == "velocity"


def test_flow_objective_is_finite_and_backpropagates() -> None:
    config = _config("ckm_flow_prior.yaml")
    model = build_model(config)
    objective = build_objective(config)
    batch = {
        "h": torch.randn(2, 2, 4, 4),
        "condition": {"state": torch.randn(2, 8)},
    }
    output = objective(model, batch)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["velocity_mse"])
    assert torch.isfinite(output["reconstruction_nmse"])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_flow_and_consistency_models_are_capacity_and_initialization_matched() -> None:
    flow_config = _config("ckm_flow_prior.yaml")
    cm_config = _config("ckmcm_prior.yaml")
    torch.manual_seed(3407)
    flow_model = build_model(flow_config)
    torch.manual_seed(3407)
    cm_model = build_model(cm_config)

    flow_state = flow_model.state_dict()
    cm_state = cm_model.state_dict()
    assert flow_state.keys() == cm_state.keys()
    assert sum(parameter.numel() for parameter in flow_model.parameters()) == sum(
        parameter.numel() for parameter in cm_model.parameters()
    )
    for name in flow_state:
        torch.testing.assert_close(flow_state[name], cm_state[name], rtol=0.0, atol=0.0)


def test_zero_initialized_flow_head_gives_identity_denoiser() -> None:
    model = build_model(_config("ckm_flow_prior.yaml")).eval()
    x = torch.randn(2, 2, 4, 4)
    sigma = torch.tensor([0.1, 0.4])
    estimate = model.denoise(
        x,
        sigma,
        condition={"state": torch.randn(2, 8)},
    )
    torch.testing.assert_close(estimate, x, rtol=0.0, atol=0.0)


def test_oracle_velocity_exactly_recovers_clean_channel() -> None:
    model = build_model(_config("ckm_flow_prior.yaml")).eval()
    clean = torch.randn(2, 2, 4, 4)
    sigma = torch.tensor([0.05, 1.25])
    oracle_velocity = torch.randn_like(clean)
    noisy = clean + sigma[:, None, None, None] * oracle_velocity

    def forward_oracle(
        self: ConditionalFlowMatchingUNet,
        x: torch.Tensor,
        sigma_value: torch.Tensor | float,
        condition: object = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del self, x, sigma_value, condition, kwargs
        return oracle_velocity

    model.forward = MethodType(forward_oracle, model)
    recovered = model.denoise(noisy, sigma, condition={"state": torch.randn(2, 8)})
    torch.testing.assert_close(recovered, clean, rtol=1.0e-6, atol=1.0e-6)
