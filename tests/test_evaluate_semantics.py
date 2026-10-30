from __future__ import annotations

import pytest

from ckmsrf.evaluate import (
    _check_checkpoint_config_semantics,
    _prior_semantics,
)


def _config(model_type: str, loss_type: str) -> dict:
    return {
        "model": {"type": model_type},
        "loss": {"type": loss_type},
    }


def test_prior_semantics_distinguish_clean_and_velocity_outputs() -> None:
    assert _prior_semantics(_config("consistency", "consistency")) == (
        "consistency",
        "clean",
    )
    assert _prior_semantics(_config("consistency", "diffusion")) == (
        "diffusion",
        "clean",
    )
    assert _prior_semantics(_config("flow_matching", "flow_matching")) == (
        "flow_matching",
        "velocity",
    )
    assert _prior_semantics(
        _config("straight_flow_matching", "straight_flow_matching")
    ) == ("straight_flow_matching", "straight_velocity")


def test_checkpoint_guard_rejects_same_shape_but_different_semantics() -> None:
    payload = {"config": _config("flow_matching", "flow_matching")}
    with pytest.raises(ValueError, match="semantically incompatible"):
        _check_checkpoint_config_semantics(
            _config("consistency", "consistency"),
            payload,
            allow_mismatch=False,
        )

    straight_payload = {
        "config": _config("straight_flow_matching", "straight_flow_matching")
    }
    with pytest.raises(ValueError, match="semantically incompatible"):
        _check_checkpoint_config_semantics(
            _config("flow_matching", "flow_matching"),
            straight_payload,
            allow_mismatch=False,
        )


def test_checkpoint_guard_requires_explicit_override_for_method_relabeling() -> None:
    payload = {"config": _config("consistency", "diffusion")}
    with pytest.raises(ValueError, match="training method"):
        _check_checkpoint_config_semantics(
            _config("consistency", "consistency"),
            payload,
            allow_mismatch=False,
        )

    _check_checkpoint_config_semantics(
        _config("consistency", "consistency"),
        payload,
        allow_mismatch=True,
    )


def test_checkpoint_guard_rejects_ckm_unconditional_swap() -> None:
    conditioned = _config("consistency", "diffusion")
    conditioned["model"]["condition"] = {
        "enabled": True,
        "force_unconditional": False,
    }
    unconditional = _config("consistency", "diffusion")
    unconditional["model"]["condition"] = {
        "enabled": True,
        "force_unconditional": True,
    }

    with pytest.raises(ValueError, match="model configuration differs"):
        _check_checkpoint_config_semantics(
            conditioned,
            {"config": unconditional},
            allow_mismatch=False,
        )
