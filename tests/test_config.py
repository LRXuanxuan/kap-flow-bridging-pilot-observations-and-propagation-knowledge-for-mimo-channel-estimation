from pathlib import Path

import torch
import yaml

from ckmsrf.config import (
    load_config,
    parse_cli_overrides,
    resolve_device_precision,
    to_serializable,
)
from ckmsrf.engine import CheckpointManager, EMA, TrainingState


def _write_yaml(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_config_precedence_and_portable_paths(tmp_path: Path) -> None:
    experiment = _write_yaml(
        tmp_path / "experiment.yaml",
        {
            "paths": {"data_root": "experiment-data", "run_root": "exp-runs"},
            "hardware": {"batch_size": 2, "num_workers": 1},
            "training": {"epochs": 11},
        },
    )
    profile = _write_yaml(
        tmp_path / "hardware.yaml",
        {"hardware": {"batch_size": 3, "num_workers": 2}},
    )
    local = _write_yaml(
        tmp_path / "local.yaml",
        {
            "paths": {"data_root": "local-data", "run_root": "local-runs"},
            "hardware": {"batch_size": 4, "num_workers": 3},
        },
    )
    config = load_config(
        experiment,
        profile,
        local,
        cli_overrides={"hardware": {"batch_size": 6}},
        environ={
            "CKMSRF_BATCH_SIZE": "5",
            "CKMSRF_NUM_WORKERS": "4",
            "CKMSRF_RUN_ROOT": "environment-runs",
        },
        project_root=tmp_path,
    )

    assert config["hardware"]["batch_size"] == 6  # CLI > environment
    assert config["hardware"]["num_workers"] == 4  # environment > local
    assert config["paths"]["data_root"] == (tmp_path / "local-data").resolve()
    assert config["paths"]["run_root"] == (tmp_path / "environment-runs").resolve()
    assert config["training"]["epochs"] == 11
    assert isinstance(config["paths"]["data_root"], Path)
    assert isinstance(to_serializable(config)["paths"]["data_root"], str)


def test_parse_dotted_cli_overrides() -> None:
    parsed = parse_cli_overrides(
        [
            "hardware.batch_size=8",
            "hardware.gradient_checkpointing=true",
            "model.dim_mults=[1, 2, 4]",
        ]
    )
    assert parsed == {
        "hardware": {"batch_size": 8, "gradient_checkpointing": True},
        "model": {"dim_mults": [1, 2, 4]},
    }


def test_explicit_cpu_runtime_is_fp32() -> None:
    spec = resolve_device_precision(
        {"hardware": {"device": "cpu", "precision": "fp32"}}
    )
    assert spec.device == torch.device("cpu")
    assert spec.precision == "fp32"
    assert spec.mixed_precision == "no"
    assert not spec.amp_enabled


def test_checkpoint_roundtrip_restores_full_training_state(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ema = EMA(model, decay=0.9)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    state = TrainingState(epoch=4, global_step=17, best_metrics={"val_nll": 1.2})

    manager = CheckpointManager(tmp_path)
    path = manager.save(
        "last.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        state=state,
        config={"paths": {"data_root": tmp_path}},
        normalization={"channel_rms": 0.25},
    )
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loaded_state, payload = manager.load(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert loaded_state == state
    assert payload["normalization"] == {"channel_rms": 0.25}
    assert payload["scheduler"] is not None
    assert payload["rng_state"]["torch"] is not None
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])

