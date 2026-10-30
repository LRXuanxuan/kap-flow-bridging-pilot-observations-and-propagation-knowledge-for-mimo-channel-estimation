from __future__ import annotations

import torch
from torch import nn

from ckmsrf.physics.observation import apply_measurement
from ckmsrf.physics.pnp import adaptive_pnp_data_step, pnp_data_step
from ckmsrf.physics.rc import build_physics_cache
from ckmsrf.sampling_cm import sample_ckmcm_pnp


class IdentityCM(nn.Module):
    """Deterministic proximal stand-in used to isolate the ADMM plumbing."""

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition=None,
    ) -> torch.Tensor:
        del sigma, condition
        return x


class OffsetCM(nn.Module):
    """Simple non-data-consistent denoiser for output-projection coverage."""

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition=None,
    ) -> torch.Tensor:
        del sigma, condition
        # ``sample_ckmcm_pnp`` calls a CM in real/imag two-channel form.
        # Add a deterministic complex offset without constructing a complex
        # scalar in a floating-point tensor.
        offset = torch.zeros_like(x)
        offset[:, 0] = 0.25
        offset[:, 1] = 0.15
        return x + offset


class ZeroVelocityFlow(nn.Module):
    """Velocity prior whose induced denoiser is exactly the identity."""

    output_parameterization = "velocity"

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition=None,
    ) -> torch.Tensor:
        del sigma, condition
        return torch.zeros_like(x)


class CountingVelocityFlow(ZeroVelocityFlow):
    """Identity-equivalent velocity prior that records network evaluations."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition=None,
    ) -> torch.Tensor:
        self.calls += 1
        return super().forward(x, sigma, condition=condition)


def _identity_problem(batch: int = 2):
    torch.manual_seed(41)
    nr, nt = 2, 3
    truth = torch.complex(torch.randn(batch, nr, nt), torch.randn(batch, nr, nt))
    pilot = torch.eye(nt, dtype=torch.complex64).expand(batch, -1, -1).clone()
    observation = apply_measurement(truth, pilot)
    noise_var = torch.tensor([0.01, 0.04], dtype=torch.float32)[:batch]
    return truth, pilot, observation, noise_var


def test_exact_pnp_z_step_is_finite_and_reduces_data_residual():
    truth, pilot, observation, noise_var = _identity_problem()
    initial = torch.zeros_like(truth)
    cache = build_physics_cache(observation, pilot, noise_var=noise_var)
    z = pnp_data_step(initial, cache, rho=torch.tensor([1.0, 2.0]))

    assert z.shape == truth.shape and z.is_complex()
    assert torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()
    initial_residual = (apply_measurement(initial, pilot) - observation).norm()
    updated_residual = (apply_measurement(z, pilot) - observation).norm()
    assert updated_residual < initial_residual


def test_adaptive_rho_is_batchwise_and_rescales_the_scaled_dual():
    truth, pilot, observation, noise_var = _identity_problem()
    x = torch.zeros_like(truth)
    dual = torch.full_like(truth, 0.1 + 0.2j)
    cache = build_physics_cache(observation, pilot, noise_var=noise_var)
    z, selected_rho, rescaled_dual, diagnostics = adaptive_pnp_data_step(
        x,
        dual,
        cache,
        rho=torch.tensor([1.0, 3.0]),
        candidate_factors=(0.5, 1.0, 2.0),
    )

    assert z.shape == x.shape
    assert selected_rho.shape == (2, 1, 1)
    assert rescaled_dual.shape == dual.shape
    assert diagnostics["rho"].shape == (2,)
    assert torch.isfinite(diagnostics["rho_score"]).all()
    expected_dual = dual * (
        torch.tensor([1.0, 3.0]).view(2, 1, 1) / selected_rho
    )
    torch.testing.assert_close(rescaled_dual, expected_dual)


def test_identity_cm_pnp_sampler_has_finite_diagnostics_and_improves_residual():
    truth, pilot, observation, noise_var = _identity_problem()
    initial = torch.zeros_like(truth)
    estimate, diagnostics = sample_ckmcm_pnp(
        IdentityCM(),
        {
            "P": pilot,
            "Y": observation,
            "noise_var": noise_var,
            "initial": initial,
        },
        iterations=3,
        lambda_reg=0.1,
        rho=1.0,
        adaptive_rho=True,
        sigma_min=1e-3,
        sigma_max=1.0,
    )

    assert estimate.shape == truth.shape and estimate.is_complex()
    assert torch.isfinite(estimate.real).all() and torch.isfinite(estimate.imag).all()
    assert diagnostics["rho_history"].shape == (3, 2)
    assert diagnostics["sigma_history"].shape == (3, 2)
    assert torch.isfinite(diagnostics["rho_history"]).all()
    assert torch.isfinite(diagnostics["sigma_history"]).all()
    assert diagnostics["measurement_residual"] < diagnostics["measurement_residual_initial"]


def test_final_data_step_restores_measurement_consistency_after_cm_update():
    truth, pilot, observation, noise_var = _identity_problem()
    batch = {"P": pilot, "Y": observation, "noise_var": noise_var}
    unconstrained, _ = sample_ckmcm_pnp(
        OffsetCM(),
        batch,
        iterations=1,
        lambda_reg=0.1,
        rho=1.0,
        final_data_step=False,
    )
    projected, diagnostics = sample_ckmcm_pnp(
        OffsetCM(),
        batch,
        iterations=1,
        lambda_reg=0.1,
        rho=1.0,
        final_data_step=True,
    )
    unconstrained_residual = (apply_measurement(unconstrained, pilot) - observation).norm()
    projected_residual = (apply_measurement(projected, pilot) - observation).norm()

    assert projected_residual < unconstrained_residual
    assert diagnostics["final_data_step"] == 1.0


def test_iteration_snapshots_match_independent_pnp_runs():
    truth, pilot, observation, noise_var = _identity_problem()
    batch = {"P": pilot, "Y": observation, "noise_var": noise_var}
    _, diagnostics = sample_ckmcm_pnp(
        OffsetCM(),
        batch,
        iterations=4,
        lambda_reg=0.1,
        rho=1.0,
        final_data_step=True,
        snapshot_iterations=(1, 2, 4),
        record_snapshot_timing=True,
    )

    assert set(diagnostics["iteration_estimates"]) == {1, 2, 4}
    assert set(diagnostics["iteration_seconds"]) == {1, 2, 4}
    assert diagnostics["iteration_seconds"][1] <= diagnostics["iteration_seconds"][4]
    for iteration, snapshot in diagnostics["iteration_estimates"].items():
        independent, _ = sample_ckmcm_pnp(
            OffsetCM(),
            batch,
            iterations=iteration,
            lambda_reg=0.1,
            rho=1.0,
            final_data_step=True,
        )
        torch.testing.assert_close(snapshot, independent)


def test_velocity_parameterization_matches_equivalent_clean_denoiser():
    truth, pilot, observation, noise_var = _identity_problem()
    batch = {"P": pilot, "Y": observation, "noise_var": noise_var}
    clean_estimate, _ = sample_ckmcm_pnp(
        IdentityCM(),
        batch,
        iterations=3,
        lambda_reg=0.1,
        rho=1.0,
    )
    flow_estimate, diagnostics = sample_ckmcm_pnp(
        ZeroVelocityFlow(),
        batch,
        iterations=3,
        lambda_reg=0.1,
        rho=1.0,
    )
    torch.testing.assert_close(flow_estimate, clean_estimate)
    assert diagnostics["prior_output_parameterization"] == "velocity"


def test_iteration_sweep_uses_one_velocity_nfe_per_iteration():
    truth, pilot, observation, noise_var = _identity_problem()
    del truth
    model = CountingVelocityFlow()
    _, diagnostics = sample_ckmcm_pnp(
        model,
        {"P": pilot, "Y": observation, "noise_var": noise_var},
        iterations=4,
        lambda_reg=0.1,
        rho=1.0,
        snapshot_iterations=(1, 2, 4),
    )

    assert model.calls == 4
    assert diagnostics["denoiser_nfe"] == 4
    assert diagnostics["denoiser_nfe_per_iteration"] == 1
    assert diagnostics["iteration_denoiser_nfe"] == {1: 1, 2: 2, 4: 4}
