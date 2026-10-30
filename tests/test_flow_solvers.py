from __future__ import annotations

import torch
from torch import nn

from ckmsrf.sampling_flow import (
    build_flowadmm_sample_plan,
    sample_flow_admm,
    sample_pnp_flow,
)


class ZeroStraightFlow(nn.Module):
    output_parameterization = "straight_velocity"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: object = None,
    ) -> torch.Tensor:
        del t, condition
        return torch.zeros_like(x) + 0.0 * self.anchor


class AdditiveFlow(ZeroStraightFlow):
    output_parameterization = "velocity"


def _batch() -> dict:
    pilot = torch.eye(2, dtype=torch.complex64).unsqueeze(0)
    truth = torch.tensor(
        [[[1.0 + 0.2j, -0.3 + 0.1j], [0.4 - 0.5j, 0.8 + 0.0j]]],
        dtype=torch.complex64,
    )
    return {
        "P": pilot,
        "Y": truth.clone(),
        "noise_var": torch.tensor([0.1]),
    }


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_pnp_flow_schedule_nfe_and_fixed_seed_reproducibility() -> None:
    model = ZeroStraightFlow()
    first, diagnostics = sample_pnp_flow(
        model,
        _batch(),
        iterations=3,
        latent_samples=2,
        generator=_generator(17),
    )
    second, second_diagnostics = sample_pnp_flow(
        model,
        _batch(),
        iterations=3,
        latent_samples=2,
        generator=_generator(17),
    )
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        diagnostics["time_history"][:, 0], torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0])
    )
    assert diagnostics["denoiser_nfe"] == 6
    assert diagnostics["data_consistency_evaluations"] == 3
    assert second_diagnostics["denoiser_nfe_per_iteration"] == 2


def test_pnp_flow_different_total_iterations_have_different_time_grids() -> None:
    model = ZeroStraightFlow()
    _, two = sample_pnp_flow(
        model, _batch(), iterations=2, latent_samples=1, generator=_generator(3)
    )
    _, three = sample_pnp_flow(
        model, _batch(), iterations=3, latent_samples=1, generator=_generator(3)
    )
    torch.testing.assert_close(two["time_history"][:, 0], torch.tensor([0.0, 0.5]))
    assert not torch.equal(two["time_history"], three["time_history"][:2])


def test_flowadmm_constant_and_three_phase_plans() -> None:
    assert build_flowadmm_sample_plan(4, schedule="constant", k_avg=3) == (
        3,
        3,
        3,
        3,
    )
    assert build_flowadmm_sample_plan(
        10,
        schedule="three_phase",
        k_avg_early=1,
        k_avg_mid=2,
        k_avg_late=5,
        switch_frac=0.5,
        switch_frac_2=0.8,
    ) == (1, 1, 1, 1, 1, 2, 2, 2, 5, 5)


def test_flowadmm_schedule_nfe_and_fixed_seed_reproducibility() -> None:
    model = ZeroStraightFlow()
    first, diagnostics = sample_flow_admm(
        model,
        _batch(),
        iterations=3,
        tau=0.5,
        time_min=0.1,
        time_max=0.9,
        gamma=1.0,
        k_avg=2,
        generator=_generator(29),
    )
    second, _ = sample_flow_admm(
        model,
        _batch(),
        iterations=3,
        tau=0.5,
        time_min=0.1,
        time_max=0.9,
        gamma=1.0,
        k_avg=2,
        generator=_generator(29),
    )
    torch.testing.assert_close(first, second)
    expected_time = torch.tensor(
        [0.1 + 0.8 / 3.0, 0.1 + 1.6 / 3.0, 0.9]
    )
    torch.testing.assert_close(
        diagnostics["time_history"][:, 0], expected_time, atol=1.0e-6, rtol=0.0
    )
    assert diagnostics["denoiser_nfe"] == 6
    assert diagnostics["latent_samples_per_iteration"] == [2, 2, 2]
    assert diagnostics["data_consistency_evaluations"] == 3


def test_flow_solvers_reject_additive_velocity_checkpoint() -> None:
    try:
        sample_pnp_flow(AdditiveFlow(), _batch(), iterations=1)
    except ValueError as exc:
        assert "additive/VE" in str(exc)
    else:
        raise AssertionError("additive Flow checkpoint should be rejected")

