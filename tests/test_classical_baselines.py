from __future__ import annotations

import torch

from ckmsrf.baselines import (
    ChannelStatisticsAccumulator,
    gaussian_lmmse,
    least_squares,
    select_statistics,
)


def test_ls_recovers_square_noiseless_channel() -> None:
    channel = torch.tensor(
        [[[1.0 + 2.0j, -0.5 + 0.25j], [0.2 - 0.1j, 2.0 + 0.0j]]]
    )
    pilot = torch.eye(2, dtype=torch.complex64).unsqueeze(0)
    observation = torch.bmm(channel, pilot)
    torch.testing.assert_close(least_squares(observation, pilot), channel)


def test_scalar_gaussian_lmmse_matches_closed_form() -> None:
    observation = torch.tensor([[[2.0 + 0.0j]]])
    pilot = torch.tensor([[[1.0 + 0.0j]]])
    mean = torch.zeros(1, 1, 1, dtype=torch.complex64)
    covariance = torch.tensor([[[[4.0 + 0.0j]]]])
    estimate = gaussian_lmmse(
        observation, pilot, 1.0, mean, covariance, jitter=0.0
    )
    # R/(R+sigma2) * y = 4/5 * 2.
    torch.testing.assert_close(estimate, torch.tensor([[[1.6 + 0.0j]]]))


def test_statistics_are_train_grouped_and_hermitian() -> None:
    channels = torch.tensor(
        [
            [[1.0 + 0.0j, 0.0 + 1.0j]],
            [[2.0 + 0.0j, 0.0 + 2.0j]],
            [[3.0 + 0.0j, 0.0 + 3.0j]],
            [[4.0 + 0.0j, 0.0 + 4.0j]],
        ]
    )
    scenes = torch.tensor([1, 1, 2, 2])
    accumulator = ChannelStatisticsAccumulator()
    accumulator.update(channels, scenes)
    groups = accumulator.finalize(shrinkage=0.1)
    artifact = {"groups": groups}
    assert groups["global"]["count"] == 4
    covariance = groups["global"]["covariance"]
    torch.testing.assert_close(covariance, covariance.mH)
    means, covariances = select_statistics(
        artifact, torch.tensor([2, 1]), scope="scene"
    )
    assert means.shape == (2, 1, 2)
    assert covariances.shape == (2, 1, 2, 2)
