from pathlib import Path
import sys

import pytest
import torch

from ckmsrf.baselines import gaussian_lmmse, least_squares
from ckmsrf.subspace_metrics import pilot_subspace_errors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
study = pytest.importorskip('run_solver_subspace_study', reason='optional study tool is not shipped in this release')


def test_lmmse_cross_covariance_recovers_unobserved_component():
    # Observe only h_1, but h_2 correlates with h_1 through a complex covariance.
    pilot = torch.tensor([[[1.], [0.]]], dtype=torch.complex128)
    observation = torch.tensor([[[2.]]], dtype=torch.complex128)
    covariance = torch.tensor([[[1., -.8j], [.8j, 1.]]], dtype=torch.complex128)
    mean = torch.zeros(1, 2, dtype=torch.complex128)
    estimate = gaussian_lmmse(observation, pilot, .25, mean, covariance, jitter=0.)
    truth = torch.tensor([[[1.6, 1.28j]]], dtype=torch.complex128)
    torch.testing.assert_close(estimate, truth)
    ls = least_squares(observation, pilot)
    assert ls[0, 0, 1] == 0
    metrics = pilot_subspace_errors(estimate, truth, pilot)
    assert metrics["e_null"].item() < 1e-25
    assert pilot_subspace_errors(ls, truth, pilot)["e_null"].item() > .3


def test_pairing_ignores_method_iterations_but_rejects_different_data(tmp_path):
    base = dict(snr_db=10, scene_id=1, trajectory_id=2, snapshot_idx=0,
                subcarrier_idx=1, sample_seed=42, problem_sha256="a",
                nmse_linear=.3, e_obs=.1, e_null=.2)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    study.write_csv(a, [{**base, "iterations": 40}])
    study.write_csv(b, [{**base, "iterations": 0}])
    study.paired(study.samples(a, 40), study.samples(b, 0))
    study.write_csv(b, [{**base, "iterations": 0, "problem_sha256": "b"}])
    with pytest.raises(ValueError, match="H/P/Y/noise"):
        study.paired(study.samples(a, 40), study.samples(b, 0))
    with pytest.raises(ValueError, match="iteration"):
        study.samples(a, 0)


def test_subspace_check_rejects_nonadditive_error(tmp_path):
    row = dict(snr_db=10, scene_id=1, trajectory_id=2, snapshot_idx=0,
               subcarrier_idx=1, sample_seed=42, problem_sha256="a", iterations=40,
               nmse_linear=.3, e_obs=.1, e_null=.3)
    path = tmp_path / "bad.csv"
    study.write_csv(path, [row])
    with pytest.raises(ValueError, match="decomposition"):
        study.samples(path, 40)


def test_statistics_normalization_mismatch_rejected():
    from ckmsrf.evaluate_classical import _check_normalization
    from ckmsrf.data import NormalizationStats
    with pytest.raises(ValueError, match="h_rms"):
        _check_normalization(NormalizationStats(h_rms=2.), {"normalization": {"h_rms": 3.}})
