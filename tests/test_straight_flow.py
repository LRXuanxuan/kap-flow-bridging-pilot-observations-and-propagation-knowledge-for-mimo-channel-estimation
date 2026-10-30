from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch

from ckmsrf.config import load_config
from ckmsrf.factory import build_model, build_objective
from ckmsrf.losses import StraightFlowMatchingObjective
from ckmsrf.models import ConditionalStraightFlowUNet, RobustConditionalStraightFlowUNet
from ckmsrf.train import _check_resume_semantics


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


def test_straight_flow_config_builds_distinct_model_and_objective() -> None:
    config = _config("ckm_straight_flow_prior.yaml")
    model = build_model(config)
    objective = build_objective(config)
    assert isinstance(model, ConditionalStraightFlowUNet)
    assert isinstance(objective, StraightFlowMatchingObjective)
    assert model.output_parameterization == "straight_velocity"
    assert objective.coupling == "minibatch_ot"


def test_robust_straight_flow_builds_mask_aware_velocity_prior() -> None:
    config = _config("ckm_straight_flow_robust.yaml")
    model = build_model(config)
    objective = build_objective(config)
    assert isinstance(model, RobustConditionalStraightFlowUNet)
    assert model.output_parameterization == "straight_velocity"
    assert type(objective).__name__ == "RobustStraightFlowMatchingObjective"


def test_mask_aware_encoder_matches_regular_prior_for_clean_all_one_mask() -> None:
    torch.manual_seed(3407)
    regular = build_model(_config("ckm_straight_flow_prior.yaml")).eval()
    torch.manual_seed(3407)
    robust = build_model(_config("ckm_straight_flow_robust.yaml")).eval()
    missing, unexpected = robust.load_state_dict(regular.state_dict(), strict=False)
    assert set(missing) == {
        "_robust_straight_flow_family_guard",
        "condition_encoder.presence_projection.weight",
    }
    assert not unexpected
    x = torch.randn(2, 2, 4, 4)
    time = torch.tensor([0.25, 0.75])
    condition = {
        "state": torch.randn(2, 8),
        "time": torch.tensor([-1.0, 1.0]),
        "frequency": torch.tensor([1.0, -1.0]),
    }
    regular_value = regular(x, time, condition=condition)
    robust_value = robust(
        x,
        time,
        condition={**condition, "branch_mask": torch.ones(2, 5)},
    )
    torch.testing.assert_close(regular_value, robust_value, rtol=0.0, atol=0.0)


def test_robust_objective_corrupts_only_during_training_by_default() -> None:
    model = build_model(_config("ckm_straight_flow_robust.yaml"))
    objective = build_objective(_config("ckm_straight_flow_robust.yaml"))
    batch = {
        "h": torch.randn(3, 2, 4, 4),
        "condition": {"state": torch.randn(3, 8)},
    }
    model.eval()
    clean_metrics = objective(model, batch)
    assert "condition_keep_rate" not in clean_metrics
    model.train()
    robust_metrics = objective(model, batch)
    assert "condition_keep_rate" in robust_metrics
    assert torch.isfinite(robust_metrics["loss"])


def test_strict_checkpoint_load_rejects_regular_and_robust_model_swap() -> None:
    regular = build_model(_config("ckm_straight_flow_prior.yaml"))
    robust = build_model(_config("ckm_straight_flow_robust.yaml"))
    try:
        regular.load_state_dict(robust.state_dict(), strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("regular prior accepted a robust checkpoint")
    try:
        robust.load_state_dict(regular.state_dict(), strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("robust prior accepted a regular checkpoint")


def test_straight_flow_objective_is_finite_and_backpropagates() -> None:
    model = build_model(_config("ckm_straight_flow_prior.yaml"))
    objective = build_objective(_config("ckm_straight_flow_prior.yaml"))
    batch = {
        "h": torch.randn(3, 2, 4, 4),
        "condition": {"state": torch.randn(3, 8)},
    }
    output = objective(model, batch)
    for key in (
        "loss",
        "straight_flow_matching_loss",
        "velocity_mse",
        "reconstruction_nmse",
        "coupling_cost",
    ):
        assert torch.isfinite(output[key])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_minibatch_ot_permutation_preserves_data_order_and_is_optimal() -> None:
    latent = torch.tensor([[[[10.0]]], [[[0.0]]]])
    data = torch.tensor([[[[0.1]]], [[[9.9]]]])
    permutation, cost = StraightFlowMatchingObjective._minibatch_ot_permutation(
        latent, data
    )
    assert permutation.tolist() == [1, 0]
    torch.testing.assert_close(cost, torch.tensor(0.01), atol=1.0e-5, rtol=0.0)


def test_straight_flow_is_capacity_and_initialization_matched_to_cm() -> None:
    torch.manual_seed(3407)
    straight = build_model(_config("ckm_straight_flow_prior.yaml"))
    torch.manual_seed(3407)
    cm = build_model(_config("ckmcm_prior.yaml"))
    assert straight.state_dict().keys() == cm.state_dict().keys()
    assert sum(parameter.numel() for parameter in straight.parameters()) == sum(
        parameter.numel() for parameter in cm.parameters()
    )
    for name, value in straight.state_dict().items():
        torch.testing.assert_close(value, cm.state_dict()[name], rtol=0.0, atol=0.0)


def test_oracle_straight_velocity_exactly_recovers_data_endpoint() -> None:
    model = build_model(_config("ckm_straight_flow_prior.yaml")).eval()
    latent = torch.randn(2, 2, 4, 4)
    data = torch.randn(2, 2, 4, 4)
    time = torch.tensor([0.2, 0.75])
    state = (1.0 - time[:, None, None, None]) * latent + time[
        :, None, None, None
    ] * data
    oracle_velocity = data - latent

    def forward_oracle(
        self: ConditionalStraightFlowUNet,
        x: torch.Tensor,
        t: torch.Tensor | float,
        condition: object = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del self, x, t, condition, kwargs
        return oracle_velocity

    model.forward = MethodType(forward_oracle, model)
    recovered = model.denoise(
        state, time, condition={"state": torch.randn(2, 8)}
    )
    torch.testing.assert_close(recovered, data, rtol=1.0e-6, atol=1.0e-6)


def test_resume_guard_rejects_additive_checkpoint_for_straight_flow(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "additive.pt"
    torch.save({"config": _config("ckm_flow_prior.yaml")}, checkpoint)
    try:
        _check_resume_semantics(
            _config("ckm_straight_flow_prior.yaml"), checkpoint
        )
    except ValueError as exc:
        assert "must never cross additive-Flow and straight-Flow" in str(exc)
    else:
        raise AssertionError("cross-path resume should be rejected")
