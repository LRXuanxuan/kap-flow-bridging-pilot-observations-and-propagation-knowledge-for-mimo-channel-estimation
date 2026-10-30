from __future__ import annotations

import torch

from ckmsrf.condition_corruption import (
    BRANCH_ORDER,
    StructuredConditionCorruptor,
    canonicalize_condition_batch,
    stable_corrupt_batch,
)
from ckmsrf.losses.gmflow import extract_condition


def _condition(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "scalar": torch.zeros(batch_size, 6),
        "state": torch.zeros(batch_size, 8),
        "observed_position": torch.zeros(batch_size, 3),
        "velocity": torch.zeros(batch_size, 3),
        "tx_beam": torch.full((batch_size, 4), 0.25),
        "rx_beam": torch.full((batch_size, 3), 1.0 / 3.0),
        "adp": torch.ones(batch_size, 2 * 2 * 2),
        "time": torch.linspace(-1.0, 1.0, batch_size),
        "frequency": torch.linspace(1.0, -1.0, batch_size),
    }


def _batch(batch_size: int = 2) -> dict[str, object]:
    condition = _condition(batch_size)
    return {
        "condition": condition,
        # Deliberate clean aliases.  A corrupt batch must remove them.
        "state": condition["state"],
        "ckm_scalar": condition["scalar"],
        "time": condition["time"],
        "frequency": condition["frequency"],
        "scene_id": torch.arange(1, batch_size + 1),
        "trajectory_id": torch.arange(100, 100 + batch_size),
        "snapshot_idx": torch.arange(batch_size),
        "subcarrier_idx": torch.arange(10, 10 + batch_size),
        "sample_seed": torch.arange(1000, 1000 + batch_size),
        "snr_db": torch.full((batch_size,), 0.0),
        "H": torch.randn(batch_size, 2, 2),
        "P": torch.randn(batch_size, 2, 2),
        "Y": torch.randn(batch_size, 2, 2),
    }


def _slice_batch(batch: dict[str, object], index: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in batch.items():
        if key == "condition":
            assert isinstance(value, dict)
            result[key] = {
                name: tensor[index : index + 1] for name, tensor in value.items()
            }
        elif torch.is_tensor(value):
            result[key] = value[index : index + 1]
        else:
            result[key] = value
    return result


def _profile() -> dict[str, object]:
    return {
        "enabled": True,
        "scalar_noise_std": 0.05,
        "position_noise_std": 0.2,
        "velocity_noise_std": 0.3,
        "orientation_noise_std": 0.0,
        "beam_noise_std": 0.02,
        "beam_uniform_mix": 0.1,
        "beam_max_shift": 1,
        "adp_noise_std": 0.03,
        "adp_gain_std": 0.1,
        "adp_blur_kernel": 3,
        "adp_max_shift": 1,
        "scalar_dropout": 0.0,
        "state_dropout": 0.0,
        "beam_dropout": 0.0,
        "adp_dropout": 0.0,
        "time_frequency_dropout": 0.0,
    }


def test_component_state_noise_is_separable_and_non_mutating() -> None:
    condition = _condition(1)
    original = {key: value.clone() for key, value in condition.items()}
    corruptor = StructuredConditionCorruptor(
        enabled=True,
        position_noise_std=0.4,
        velocity_noise_std=0.0,
        orientation_noise_std=0.0,
        adp_shape=(2, 2, 2),
    )
    generator = torch.Generator().manual_seed(17)
    corrupted = corruptor(condition, generator=generator)
    assert isinstance(corrupted, dict)
    assert not torch.equal(corrupted["state"][:, :3], original["state"][:, :3])
    torch.testing.assert_close(corrupted["state"][:, 3:], original["state"][:, 3:])
    torch.testing.assert_close(corrupted["observed_position"], corrupted["state"][:, :3])
    torch.testing.assert_close(corrupted["velocity"], corrupted["state"][:, 3:6])
    for key, value in condition.items():
        torch.testing.assert_close(value, original[key])
    assert corrupted["branch_mask"].shape == (1, len(BRANCH_ORDER))


def test_beam_corruption_keeps_probability_normalization() -> None:
    condition = _condition(3)
    corruptor = StructuredConditionCorruptor(
        enabled=True,
        beam_noise_std=0.1,
        beam_uniform_mix=0.2,
        beam_max_shift=1,
        adp_shape=(2, 2, 2),
    )
    corrupted = corruptor(condition, generator=torch.Generator().manual_seed(8))
    assert isinstance(corrupted, dict)
    torch.testing.assert_close(
        corrupted["tx_beam"].sum(dim=-1), torch.ones(3), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        corrupted["rx_beam"].sum(dim=-1), torch.ones(3), atol=1.0e-6, rtol=0.0
    )


def test_stable_corruption_is_batch_and_snr_independent() -> None:
    batch = _batch(2)
    corruptor = StructuredConditionCorruptor.from_config(_profile(), adp_shape=(2, 2, 2))
    full = stable_corrupt_batch(
        batch, corruptor, condition=extract_condition(batch), seed=260211950
    )
    first = stable_corrupt_batch(
        _slice_batch(batch, 0),
        corruptor,
        condition=extract_condition(_slice_batch(batch, 0)),
        seed=260211950,
    )
    second_input = _slice_batch(batch, 1)
    second_input["snr_db"] = torch.tensor([30.0])
    second = stable_corrupt_batch(
        second_input,
        corruptor,
        condition=extract_condition(second_input),
        seed=260211950,
    )
    assert isinstance(full["condition"], dict)
    assert isinstance(first["condition"], dict)
    assert isinstance(second["condition"], dict)
    for key, value in full["condition"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value[:1], first["condition"][key])
            torch.testing.assert_close(value[1:], second["condition"][key])
    assert "state" not in full
    assert "ckm_scalar" not in full
    assert "time" not in full
    assert "frequency" not in full


def test_canonicalized_condition_cannot_be_overwritten_by_clean_aliases() -> None:
    batch = _batch(1)
    changed = _condition(1)
    changed["state"].fill_(7.0)
    canonical = canonicalize_condition_batch(batch, changed)
    extracted = extract_condition(canonical)
    assert extracted is not None
    torch.testing.assert_close(extracted["state"], torch.full((1, 8), 7.0))
