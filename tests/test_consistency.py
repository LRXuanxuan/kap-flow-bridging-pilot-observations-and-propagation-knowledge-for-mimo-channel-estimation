from __future__ import annotations

import pytest
import torch

from ckmsrf.losses.consistency import ConsistencyObjective
from ckmsrf.models.consistency import ConditionalConsistencyUNet
from ckmsrf.physics import apply_measurement


def _condition(batch: int) -> dict[str, torch.Tensor]:
    return {
        "scalar": torch.randn(batch, 6),
        "state": torch.randn(batch, 8),
        "tx_beam": torch.softmax(torch.randn(batch, 8), dim=-1),
        "rx_beam": torch.softmax(torch.randn(batch, 4), dim=-1),
        "adp": torch.rand(batch, 4, 4, 3),
        # This is physical snapshot time, not the CM noise scale.
        "time": torch.rand(batch, 1),
        "frequency": torch.rand(batch, 1),
    }


def _model() -> ConditionalConsistencyUNet:
    return ConditionalConsistencyUNet(
        dim=8,
        dim_mults=(1, 2),
        num_res_blocks=1,
        num_components=3,
        groups=4,
        sigma_min=0.01,
        sigma_data=0.5,
        condition={
            "scalar_dim": 6,
            "state_dim": 8,
            "tx_beam_dim": 8,
            "rx_beam_dim": 4,
            "adp_shape": (4, 4, 3),
            "embed_dim": 32,
            "branch_dim": 16,
            "dropout_prob": 0.0,
        },
    )


def test_consistency_model_shape_and_exact_sigma_min_identity() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    x = torch.randn(2, 2, 8, 16)
    condition = _condition(2)

    boundary = model(x, 0.01, condition=condition)
    assert boundary.shape == x.shape
    torch.testing.assert_close(boundary, x, rtol=0.0, atol=0.0)

    prediction = model(x, torch.tensor([0.1, 0.7]), context=condition)
    assert prediction.shape == x.shape
    assert torch.isfinite(prediction).all()

    with pytest.raises(ValueError, match="either condition or context"):
        model(x, 0.1, condition=condition, context=condition)
    with pytest.raises(ValueError, match="sigma_min"):
        model(x, 0.001, condition=condition)


def test_preconditioning_coefficients_and_model_gradient() -> None:
    torch.manual_seed(29)
    model = _model().train()
    sigma = torch.tensor([model.sigma_min, 0.5])
    c_skip, c_out, c_in, c_noise = model.preconditioning(sigma)
    torch.testing.assert_close(c_skip[0], torch.tensor(1.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(c_out[0], torch.tensor(0.0), rtol=0.0, atol=0.0)
    assert (c_in > 0).all() and torch.isfinite(c_noise).all()

    x = torch.randn(2, 2, 8, 16)
    prediction = model(x, sigma, condition=_condition(2))
    prediction[1].square().mean().backward()
    assert model.final_conv.weight.grad is not None
    assert torch.isfinite(model.final_conv.weight.grad).all()


def test_consistency_objective_with_physics_backpropagates() -> None:
    torch.manual_seed(31)
    model = _model().train()
    batch_size, nr, nt, num_pilots = 2, 8, 16, 5
    channel = torch.complex(
        torch.randn(batch_size, nr, nt),
        torch.randn(batch_size, nr, nt),
    ) / 2**0.5
    pilot = torch.complex(
        torch.randn(batch_size, nt, num_pilots),
        torch.randn(batch_size, nt, num_pilots),
    ) / nt**0.5
    observation = apply_measurement(channel, pilot)
    objective = ConsistencyObjective(
        sigma_min=model.sigma_min,
        sigma_max=0.8,
        consistency_weight=1.0,
        endpoint_weight=1.0,
        reconstruction_weight=0.1,
        data_consistency_weight=0.01,
    )
    metrics = objective(
        model,
        {
            "h_complex": channel,
            "pilot": pilot,
            "observation": observation,
            "noise_var": torch.full((batch_size,), 0.1),
            "condition": _condition(batch_size),
        },
    )

    expected = {
        "loss",
        "consistency_loss",
        "endpoint_loss",
        "reconstruction_nmse",
        "data_consistency",
        "sigma_high",
        "sigma_low",
    }
    assert expected <= metrics.keys()
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
    metrics["loss"].backward()
    assert model.final_conv.weight.grad is not None
    assert torch.isfinite(model.final_conv.weight.grad).all()


def test_karras_sigma_schedule_respects_exact_minimum() -> None:
    """The lowest sampled level must not fall below the model boundary."""

    torch.manual_seed(37)
    objective = ConsistencyObjective(
        sigma_min=0.002,
        sigma_max=2.0,
        rho=7.0,
        num_scales=40,
    )
    high, low = objective.sample_sigma_pair(
        4096,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert bool((low >= objective.sigma_min).all())
    assert bool((high >= objective.sigma_min).all())


def test_objective_requires_observations_only_when_physics_is_weighted() -> None:
    model = _model().train()
    objective = ConsistencyObjective(
        sigma_min=model.sigma_min,
        sigma_max=0.2,
        data_consistency_weight=1.0,
    )
    with pytest.raises(KeyError, match="requires pilot"):
        objective(
            model,
            {
                "h": torch.randn(2, 2, 8, 16),
                "condition": _condition(2),
            },
        )
