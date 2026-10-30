from __future__ import annotations

import torch

from ckmsrf.losses import GMFlowObjective
from ckmsrf.models import ConditionalGMUNet, GMUNet
from ckmsrf.physics import (
    apply_measurement,
    build_physics_cache,
    data_consistency_loss,
    pilot_guided_update,
    rc_proximal_update,
)


def _condition(batch: int) -> dict[str, torch.Tensor]:
    return {
        "scalar": torch.randn(batch, 6),
        "state": torch.randn(batch, 8),
        "tx_beam": torch.softmax(torch.randn(batch, 8), dim=-1),
        "rx_beam": torch.softmax(torch.randn(batch, 4), dim=-1),
        "adp": torch.rand(batch, 4, 4, 3),
        "time": torch.rand(batch, 1),
        "frequency": torch.rand(batch, 1),
    }


def _model(conditional: bool = True):
    cls = ConditionalGMUNet if conditional else GMUNet
    kwargs = dict(
        dim=8,
        dim_mults=(1, 2),
        num_res_blocks=1,
        num_components=3,
        groups=4,
    )
    if conditional:
        kwargs["condition"] = {
            "scalar_dim": 6,
            "state_dim": 8,
            "tx_beam_dim": 8,
            "rx_beam_dim": 4,
            "adp_shape": (4, 4, 3),
            "embed_dim": 32,
            "branch_dim": 16,
            "dropout_prob": 0.0,
        }
    return cls(**kwargs)


def test_conditional_gm_shapes_and_normalized_weights():
    model = _model().eval()
    channel = torch.randn(2, 2, 8, 16)
    gm = model(channel, torch.tensor([0.2, 0.8]), condition=_condition(2))
    assert gm["means"].shape == (2, 3, 2, 8, 16)
    assert gm["logweights"].shape == (2, 3, 1, 8, 16)
    assert gm["logstds"].shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(gm["logweights"].exp().sum(dim=1), torch.ones(2, 1, 8, 16))


def test_zero_condition_injection_and_legacy_state_dict_compatibility():
    torch.manual_seed(7)
    legacy = _model(conditional=False).eval()
    conditional = _model(conditional=True).eval()
    incompat = conditional.load_state_dict(legacy.state_dict(), strict=False)
    assert not incompat.unexpected_keys
    assert all(key.startswith(("condition_encoder.", "condition_projection.")) for key in incompat.missing_keys)

    x = torch.randn(2, 2, 8, 16)
    t = torch.tensor([0.3, 0.7])
    expected = legacy(x, t)
    actual = conditional(x, t, condition=_condition(2))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_gmflow_objective_with_physics_terms_backpropagates():
    torch.manual_seed(11)
    model = _model().train()
    batch, nr, nt, npilots = 2, 8, 16, 5
    h = torch.complex(torch.randn(batch, nr, nt), torch.randn(batch, nr, nt)) / 2**0.5
    p = torch.complex(torch.randn(batch, nt, npilots), torch.randn(batch, nt, npilots)) / nt**0.5
    y = apply_measurement(h, p)
    objective = GMFlowObjective(
        snr_min_db=0,
        snr_max_db=0,
        reconstruction_weight=0.1,
        data_consistency_weight=0.01,
    )
    result = objective(
        model,
        {
            "h_complex": h,
            "pilot": p,
            "observation": y,
            "noise_var": torch.full((batch,), 0.1),
            "condition": _condition(batch),
        },
    )
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.final_conv.weight.grad is not None
    assert torch.isfinite(model.final_conv.weight.grad).all()


def test_rc_and_pbg_reduce_measurement_residual():
    torch.manual_seed(19)
    batch, nr, nt, npilots = 2, 4, 8, 3
    h_true = torch.complex(torch.randn(batch, nr, nt), torch.randn(batch, nr, nt))
    pilot = torch.complex(torch.randn(batch, nt, npilots), torch.randn(batch, nt, npilots)) / nt**0.5
    observation = apply_measurement(h_true, pilot)
    initial = torch.zeros_like(h_true)
    cache = build_physics_cache(observation, pilot, noise_var=torch.full((batch,), 1e-3))

    rc = rc_proximal_update(initial, cache, prior_var=torch.full((batch,), 0.5))
    assert data_consistency_loss(rc, observation, pilot) < data_consistency_loss(initial, observation, pilot)

    guided, diagnostics = pilot_guided_update(
        initial,
        observation,
        cache,
        step_size=0.5,
        steps=3,
        noise_gating=0.0,
    )
    assert data_consistency_loss(guided, observation, pilot) < data_consistency_loss(initial, observation, pilot)
    assert diagnostics["pbg_gradient_norm"] > 0
