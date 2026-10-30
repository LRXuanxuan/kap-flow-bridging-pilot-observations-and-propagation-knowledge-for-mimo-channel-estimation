from __future__ import annotations

import torch
from torch import nn
import pytest

from ckmsrf.physics.observation import apply_measurement
from ckmsrf.physics.rc import build_physics_cache
from ckmsrf.sampling_pnpdm import (
    reverse_diffusion_prior,
    sample_ckm_diffusion_pnpdm,
    sample_linear_gaussian_likelihood,
)


class CountingIdentityDenoiser(nn.Module):
    output_parameterization = "clean"
    sigma_min = 0.002

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.calls = 0
        self.conditions = []

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition=None):
        del sigma
        self.calls += 1
        self.conditions.append(condition)
        return x + self.anchor * 0.0


class ZeroDenoiser(CountingIdentityDenoiser):
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition=None):
        del sigma
        self.calls += 1
        self.conditions.append(condition)
        return torch.zeros_like(x) + self.anchor * 0.0


def _identity_problem(batch: int = 2):
    torch.manual_seed(71)
    nr, nt = 2, 3
    truth = torch.complex(torch.randn(batch, nr, nt), torch.randn(batch, nr, nt))
    pilot = torch.eye(nt, dtype=torch.complex64).expand(batch, -1, -1).clone()
    observation = apply_measurement(truth, pilot)
    noise_var = torch.full((batch,), 0.04, dtype=torch.float32)
    condition = torch.randn(batch, 6)
    return truth, pilot, observation, noise_var, condition


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_likelihood_draw_uses_real_component_coupling_variance() -> None:
    batch_size = 8192
    current = torch.zeros(batch_size, 1, 1, dtype=torch.complex64)
    pilot = torch.zeros(1, 1, dtype=torch.complex64)
    observation = torch.zeros(batch_size, 1, 1, dtype=torch.complex64)
    cache = build_physics_cache(observation, pilot, noise_var=1.0)
    rho = 0.3

    draw = sample_linear_gaussian_likelihood(
        current,
        cache,
        rho,
        generator=_generator(9102),
    )

    assert abs(float(draw.real.mean())) < 0.02
    assert abs(float(draw.imag.mean())) < 0.02
    torch.testing.assert_close(draw.real.var(), torch.tensor(rho**2), rtol=0.10, atol=0.01)
    torch.testing.assert_close(draw.imag.var(), torch.tensor(rho**2), rtol=0.10, atol=0.01)


def test_likelihood_draw_matches_general_complex_covariance() -> None:
    batch_size = 32768
    current = torch.zeros(batch_size, 1, 2, dtype=torch.complex64)
    pilot = torch.tensor(
        [[1.0 + 0.0j, 0.3 + 0.7j], [0.2 - 0.6j, 1.1 + 0.1j]],
        dtype=torch.complex64,
    )
    observation = torch.zeros(batch_size, 1, 2, dtype=torch.complex64)
    cache = build_physics_cache(observation, pilot, noise_var=0.4)
    rho = 0.45

    draw = sample_linear_gaussian_likelihood(
        current,
        cache,
        rho,
        generator=_generator(9103),
    ).squeeze(1)
    empirical = draw.mH @ draw / float(batch_size)
    target = torch.linalg.inv(cache["M"][0] + cache["I_Nt"][0] / (2.0 * rho**2))

    torch.testing.assert_close(empirical, target, rtol=0.04, atol=0.004)


def test_likelihood_draw_handles_low_noise_rank_deficient_pilots() -> None:
    """The exact precision is positive definite even when FP32 loses that fact."""

    torch.manual_seed(9104)
    batch_size, nt, npilots = 3, 64, 16
    current = torch.zeros(batch_size, 1, nt, dtype=torch.complex64)
    pilot = torch.complex(
        torch.randn(batch_size, nt, npilots),
        torch.randn(batch_size, nt, npilots),
    )
    observation = torch.zeros(batch_size, 1, npilots, dtype=torch.complex64)
    cache = build_physics_cache(
        observation,
        pilot,
        noise_var=torch.full((batch_size,), 1e-12),
    )

    draw = sample_linear_gaussian_likelihood(
        current,
        cache,
        coupling_std=0.05,
        generator=_generator(9104),
    )

    assert draw.dtype == torch.complex64
    assert bool(torch.isfinite(draw.real).all())
    assert bool(torch.isfinite(draw.imag).all())


def test_reverse_sde_is_reproducible_and_reports_actual_nfe() -> None:
    model = ZeroDenoiser()
    noisy = torch.ones(2, 2, 3, dtype=torch.complex64)
    kwargs = {
        "coupling_std": 0.5,
        "condition": None,
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "prior_steps": 12,
        "schedule_power": 3.0,
        "solver": "sde",
    }

    first, first_nfe, levels = reverse_diffusion_prior(
        model,
        noisy,
        generator=_generator(123),
        **kwargs,
    )
    second, second_nfe, _ = reverse_diffusion_prior(
        ZeroDenoiser(),
        noisy,
        generator=_generator(123),
        **kwargs,
    )
    different, _, _ = reverse_diffusion_prior(
        ZeroDenoiser(),
        noisy,
        generator=_generator(124),
        **kwargs,
    )

    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, different)
    assert first_nfe == second_nfe == levels.numel() - 1
    assert model.calls == first_nfe
    assert float(levels[0]) <= 0.5
    torch.testing.assert_close(levels[-2], torch.tensor(0.01))
    torch.testing.assert_close(levels[-1], torch.tensor(0.0))


def test_pnpdm_iteration_snapshots_reuse_one_random_prefix() -> None:
    truth, pilot, observation, noise_var, condition = _identity_problem()
    batch = {
        "P": pilot,
        "Y": observation,
        "noise_var": noise_var,
        "ckm_scalar": condition,
    }
    model = CountingIdentityDenoiser()
    _, diagnostics = sample_ckm_diffusion_pnpdm(
        model,
        batch,
        iterations=4,
        rho_start=0.5,
        rho_min=0.1,
        rho_decay=0.5,
        sigma_min=0.002,
        sigma_max=1.0,
        prior_steps=8,
        schedule_power=3.0,
        prior_solver="pf_ode",
        initialization="zero",
        estimate_mode="last",
        burn_in=0,
        thin=1,
        generator=_generator(404),
        snapshot_iterations=(1, 2, 4),
        record_snapshot_timing=True,
    )

    assert diagnostics["data_consistency_evaluations"] == 4
    assert diagnostics["likelihood_random_draws"] == 4
    assert diagnostics["denoiser_nfe"] == sum(diagnostics["prior_nfe_history"])
    assert model.calls == diagnostics["denoiser_nfe"]
    assert set(diagnostics["iteration_estimates"]) == {1, 2, 4}
    assert set(diagnostics["iteration_denoiser_nfe"]) == {1, 2, 4}
    assert diagnostics["iteration_seconds"][1] <= diagnostics["iteration_seconds"][4]
    torch.testing.assert_close(
        diagnostics["rho_history"],
        torch.tensor([0.5, 0.25, 0.125, 0.1], dtype=torch.float64),
    )
    assert torch.all(
        diagnostics["prior_start_sigma_history"] <= diagnostics["rho_history"]
    )
    assert model.conditions and all(value is not None for value in model.conditions)

    for iteration, snapshot in diagnostics["iteration_estimates"].items():
        independent, _ = sample_ckm_diffusion_pnpdm(
            CountingIdentityDenoiser(),
            batch,
            iterations=iteration,
            rho_start=0.5,
            rho_min=0.1,
            rho_decay=0.5,
            sigma_min=0.002,
            sigma_max=1.0,
            prior_steps=8,
            schedule_power=3.0,
            prior_solver="pf_ode",
            initialization="zero",
            estimate_mode="last",
            burn_in=0,
            thin=1,
            generator=_generator(404),
        )
        torch.testing.assert_close(snapshot, independent)


def test_pnpdm_sde_snapshot_matches_independent_random_prefix() -> None:
    _, pilot, observation, noise_var, _ = _identity_problem(batch=1)
    batch = {"P": pilot, "Y": observation, "noise_var": noise_var}
    common = {
        "rho_start": 0.5,
        "rho_min": 0.1,
        "rho_decay": 0.7,
        "sigma_min": 0.002,
        "sigma_max": 1.0,
        "prior_steps": 6,
        "schedule_power": 3.0,
        "prior_solver": "sde",
        "initialization": "zero",
        "estimate_mode": "last",
        "burn_in": 0,
        "thin": 1,
    }
    _, long_diagnostics = sample_ckm_diffusion_pnpdm(
        ZeroDenoiser(),
        batch,
        iterations=3,
        generator=_generator(406),
        snapshot_iterations=(2,),
        **common,
    )
    independent, _ = sample_ckm_diffusion_pnpdm(
        ZeroDenoiser(),
        batch,
        iterations=2,
        generator=_generator(406),
        **common,
    )

    torch.testing.assert_close(long_diagnostics["iteration_estimates"][2], independent)


def test_posterior_mean_counts_only_post_burn_in_thinned_states() -> None:
    _, pilot, observation, noise_var, _ = _identity_problem(batch=1)
    _, diagnostics = sample_ckm_diffusion_pnpdm(
        CountingIdentityDenoiser(),
        {"P": pilot, "Y": observation, "noise_var": noise_var},
        iterations=5,
        rho_start=0.3,
        rho_min=0.1,
        rho_decay=0.5,
        sigma_min=0.002,
        sigma_max=1.0,
        prior_steps=4,
        prior_solver="pf_ode",
        estimate_mode="posterior_mean",
        burn_in=1,
        thin=2,
        generator=_generator(505),
        snapshot_iterations=(1, 3, 5),
    )

    assert diagnostics["posterior_sample_count"] == 2
    assert diagnostics["iteration_sample_count"] == {1: 0, 3: 1, 5: 2}
    assert diagnostics["iteration_estimate_mode"] == {
        1: "last",
        3: "posterior_mean",
        5: "posterior_mean",
    }


def test_posterior_mean_rejects_samples_across_annealing_levels() -> None:
    _, pilot, observation, noise_var, _ = _identity_problem(batch=1)
    with pytest.raises(ValueError, match="different rho values"):
        sample_ckm_diffusion_pnpdm(
            CountingIdentityDenoiser(),
            {"P": pilot, "Y": observation, "noise_var": noise_var},
            iterations=5,
            rho_start=0.5,
            rho_min=0.1,
            rho_decay=0.9,
            sigma_min=0.002,
            sigma_max=1.0,
            prior_steps=4,
            estimate_mode="posterior_mean",
            burn_in=0,
            thin=1,
        )
