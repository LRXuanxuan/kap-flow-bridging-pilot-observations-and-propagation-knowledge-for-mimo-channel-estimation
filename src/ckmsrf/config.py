"""Configuration loading and portable runtime resolution.

The merge order is deliberately explicit::

    defaults < experiment < hardware profile < local YAML < environment < CLI

Scientific settings live in experiment files, while machine-dependent settings
live below ``hardware`` and ``paths``.  Resolved paths are always
``pathlib.Path`` objects and are relative to the project root unless an absolute
path is supplied.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "ckmsrf"},
    "paths": {
        "data_root": "data",
        "run_root": "runs",
    },
    "data": {
        "backend": "auto",
        "index_file": "indices/{split}.json",
        "sampling": "random_tf",
        "validation_sampling": "fixed",
        "test_sampling": "fixed",
        "nr": 16,
        "nt": 64,
        "n_subcarriers": 48,
        "n_snapshots": 8,
        "pilot_length": 19,
        "include_observation": False,
        "observation": {
            "enabled": True,
            "noise_mode": "stored",
            "deterministic": True,
            "base_seed": 9102,
        },
        "normalization": {
            "mode": "global_rms",
            "stats_file": "normalization.json",
        },
    },
    "model": {
        "type": "gmflow",
        "in_channels": 2,
        "dim": 32,
        "dim_mults": [1, 2, 4],
        "groups": 8,
        "num_components": 2,
        "condition": {
            "enabled": True,
            "embed_dim": 128,
            "scalar_dim": 6,
            "state_dim": 8,
            "tx_beam_dim": 64,
            "rx_beam_dim": 16,
            "adp_shape": [16, 16, 12],
            "dropout_prob": 0.1,
            "use_scalar": True,
            "use_state": True,
            "use_beam": True,
            "use_adp": True,
        },
    },
    "flow": {
        "time_low": 0.0,
        "time_high": 1.0,
        "transition_ratio": 0.5,
        "noise_std": 1.0,
    },
    "loss": {
        "type": "gmflow",
        "gm_nll_weight": 1.0,
        "reconstruction_weight": 0.0,
        "physics_weight": 0.0,
    },
    "consistency": {
        "sigma_min": 0.002,
        "sigma_max": 2.0,
        "sigma_data": 1.0,
        "rho": 7.0,
        "num_scales": 40,
        "endpoint_weight": 1.0,
        "consistency_weight": 1.0,
        "pseudo_huber_c": 0.03,
    },
    "straight_flow_matching": {
        "time_min": 0.0,
        "time_max": 1.0,
        "latent_std": 1.0,
        "coupling": "independent",
    },
    "sampling": {
        "solver": "srf_rc",
        "steps": 20,
        "outer_loops": 1,
        "lambda_power": 1.0,
        "beta_power": 1.0,
        "pbg_steps": 0,
        "pbg_step_size": 0.15,
        "fig_plus_mix": 0.0,
        "pnp_iterations": 4,
        "pnp_rho": 1.0,
        "pnp_lambda": 0.1,
        "pnp_adaptive_rho": False,
        "pnp_rho_candidate_factors": [0.5, 1.0, 2.0],
        "pnp_rho_min": 1.0e-6,
        "pnp_rho_max": 1.0e6,
        "pnp_whiteness_weight": 0.25,
        "pnp_momentum": 0.0,
        "pnpdm_rho_start": 1.0,
        "pnpdm_rho_min": 0.05,
        "pnpdm_rho_decay": 0.9,
        "pnpdm_prior_steps": 100,
        "pnpdm_schedule_power": 7.0,
        "pnpdm_prior_solver": "sde",
        "pnpdm_initialization": "ls",
        "pnpdm_estimate_mode": "posterior_mean",
        "pnpdm_burn_in": 40,
        "pnpdm_thin": 3,
        "flow_initialization": "ls",
        "flow_latent_std": 1.0,
        "pnp_flow_step_scale": 1.0,
        "pnp_flow_alpha": 1.0,
        "pnp_flow_time_min": 0.0,
        "pnp_flow_time_max": 1.0,
        "pnp_flow_latent_samples": 5,
        "flow_admm_tau": 1.0,
        "flow_admm_time_min": 0.1,
        "flow_admm_time_max": 0.95,
        "flow_admm_gamma": 1.0,
        "flow_admm_k_avg": 5,
        "flow_admm_k_avg_schedule": "constant",
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-4,
        "betas": [0.9, 0.99],
    },
    "scheduler": {
        "name": "cosine",
        "warmup_steps": 1000,
        "step_interval": "step",
    },
    "training": {
        "seed": 3407,
        "validation_seed": 9102,
        "epochs": 200,
        "max_steps": None,
        "resume": None,
        "ema_decay": 0.999,
        "max_grad_norm": 1.0,
        "log_every_steps": 20,
        "validate_every_epochs": 1,
        "save_every_epochs": 1,
    },
    "hardware": {
        "device": "auto",
        "precision": "auto",
        "use_accelerate": True,
        "batch_size": 4,
        "eval_batch_size": 4,
        "grad_accum_steps": 1,
        "num_workers": 0,
        "pin_memory": "auto",
        "persistent_workers": "auto",
        "gradient_checkpointing": False,
    },
    "logging": {
        "tensorboard": True,
        "jsonl": True,
    },
}


ENV_OVERRIDES: dict[str, str] = {
    "CKMSRF_DATA_ROOT": "paths.data_root",
    "CKMSRF_RUN_ROOT": "paths.run_root",
    "CKMSRF_DEVICE": "hardware.device",
    "CKMSRF_PRECISION": "hardware.precision",
    "CKMSRF_BATCH_SIZE": "hardware.batch_size",
    "CKMSRF_EVAL_BATCH_SIZE": "hardware.eval_batch_size",
    "CKMSRF_GRAD_ACCUM_STEPS": "hardware.grad_accum_steps",
    "CKMSRF_NUM_WORKERS": "hardware.num_workers",
    "CKMSRF_USE_ACCELERATE": "hardware.use_accelerate",
    "CKMSRF_RESUME": "training.resume",
    "CKMSRF_SEED": "training.seed",
}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return dict(loaded)


def _parse_scalar(value: str) -> Any:
    """Use YAML scalar semantics for CLI/environment values."""

    parsed = yaml.safe_load(value)
    return value if parsed is None and value.strip().lower() != "null" else parsed


def set_by_dotted_key(
    mapping: MutableMapping[str, Any], dotted_key: str, value: Any
) -> None:
    """Set ``a.b.c`` in a nested mapping, creating intermediate mappings."""

    if not dotted_key or any(not part for part in dotted_key.split(".")):
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            cursor[part] = {}
        elif not isinstance(existing, MutableMapping):
            raise TypeError(
                f"Cannot set {dotted_key!r}: {part!r} is not a mapping"
            )
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_cli_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeated ``--set dotted.key=value`` arguments."""

    overrides: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Override must have KEY=VALUE form: {item!r}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(overrides, key.strip(), _parse_scalar(raw_value.strip()))
    return overrides


def _environment_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_name, dotted_key in ENV_OVERRIDES.items():
        if env_name in environ and environ[env_name] != "":
            set_by_dotted_key(
                overrides, dotted_key, _parse_scalar(environ[env_name])
            )

    packed = environ.get("CKMSRF_OVERRIDES")
    if packed:
        parsed = yaml.safe_load(packed)
        if not isinstance(parsed, Mapping):
            raise TypeError("CKMSRF_OVERRIDES must decode to a mapping")
        overrides = _deep_merge(overrides, parsed)
    return overrides


def _portable_path(value: str | os.PathLike[str], base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _resolve_path_fields(config: dict[str, Any], project_root: Path) -> None:
    paths = config.setdefault("paths", {})
    if not isinstance(paths, MutableMapping):
        raise TypeError("config.paths must be a mapping")
    for key, value in list(paths.items()):
        if value is not None:
            paths[key] = _portable_path(value, project_root)


def _validate_config(config: Mapping[str, Any]) -> None:
    hardware = config.get("hardware", {})
    if int(hardware.get("batch_size", 0)) < 1:
        raise ValueError("hardware.batch_size must be >= 1")
    if int(hardware.get("eval_batch_size", 0)) < 1:
        raise ValueError("hardware.eval_batch_size must be >= 1")
    if int(hardware.get("grad_accum_steps", 0)) < 1:
        raise ValueError("hardware.grad_accum_steps must be >= 1")
    if int(hardware.get("num_workers", -1)) < 0:
        raise ValueError("hardware.num_workers must be >= 0")
    if str(hardware.get("precision", "auto")).lower() not in {
        "auto",
        "fp32",
        "float32",
        "fp16",
        "float16",
        "bf16",
        "bfloat16",
    }:
        raise ValueError("hardware.precision must be auto, fp32, fp16, or bf16")
    decay = float(config.get("training", {}).get("ema_decay", 0.0))
    if not 0.0 <= decay < 1.0:
        raise ValueError("training.ema_decay must lie in [0, 1)")
    model_type = str(config.get("model", {}).get("type", "gmflow")).lower()
    if model_type not in {
        "gmflow",
        "consistency",
        "cm",
        "flow_matching",
        "velocity_flow",
        "straight_flow_matching",
        "straight_flow",
    }:
        raise ValueError(
            "model.type must be gmflow, consistency, flow_matching, or "
            "straight_flow_matching"
        )
    consistency = config.get("consistency", {})
    sigma_min = float(consistency.get("sigma_min", 0.002))
    sigma_max = float(consistency.get("sigma_max", 2.0))
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("consistency requires 0 < sigma_min < sigma_max")
    if float(consistency.get("sigma_data", 1.0)) <= 0.0:
        raise ValueError("consistency.sigma_data must be positive")
    if int(consistency.get("num_scales", 40)) < 2:
        raise ValueError("consistency.num_scales must be >= 2")
    if float(consistency.get("rho", 7.0)) <= 0.0:
        raise ValueError("consistency.rho must be positive")
    flow_matching = config.get("flow_matching", {})
    if float(flow_matching.get("p_std", 1.2)) <= 0.0:
        raise ValueError("flow_matching.p_std must be positive")
    straight_flow = config.get("straight_flow_matching", {})
    straight_time_min = float(straight_flow.get("time_min", 0.0))
    straight_time_max = float(straight_flow.get("time_max", 1.0))
    if not 0.0 <= straight_time_min < straight_time_max <= 1.0:
        raise ValueError(
            "straight_flow_matching requires 0 <= time_min < time_max <= 1"
        )
    if float(straight_flow.get("latent_std", 1.0)) <= 0.0:
        raise ValueError("straight_flow_matching.latent_std must be positive")
    straight_coupling = str(straight_flow.get("coupling", "independent")).lower()
    if straight_coupling not in {"independent", "minibatch_ot", "minibatch-ot"}:
        raise ValueError(
            "straight_flow_matching.coupling must be independent or minibatch_ot"
        )
    solver = str(config.get("sampling", {}).get("solver", "srf_rc")).lower()
    if solver not in {
        "srf_rc",
        "pnp_cm",
        "pnp_admm",
        "pnp_dm",
        "pnp_flow",
        "flow_admm",
    }:
        raise ValueError(
            "sampling.solver must be srf_rc, pnp_cm, pnp_admm, pnp_dm, "
            "pnp_flow, or flow_admm"
        )
    sampling = config.get("sampling", {})
    if int(sampling.get("pnp_iterations", 4)) < 1:
        raise ValueError("sampling.pnp_iterations must be >= 1")
    if float(sampling.get("pnp_rho", 1.0)) <= 0.0:
        raise ValueError("sampling.pnp_rho must be positive")
    if float(sampling.get("pnp_lambda", 0.1)) <= 0.0:
        raise ValueError("sampling.pnp_lambda must be positive")
    pnpdm_rho_start = float(sampling.get("pnpdm_rho_start", 1.0))
    pnpdm_rho_min = float(sampling.get("pnpdm_rho_min", 0.05))
    if not 0.0 < pnpdm_rho_min <= pnpdm_rho_start:
        raise ValueError(
            "sampling PnP-DM coupling requires 0 < rho_min <= rho_start"
        )
    pnpdm_rho_decay = float(sampling.get("pnpdm_rho_decay", 0.9))
    if not 0.0 < pnpdm_rho_decay <= 1.0:
        raise ValueError("sampling.pnpdm_rho_decay must lie in (0,1]")
    if int(sampling.get("pnpdm_prior_steps", 100)) < 2:
        raise ValueError("sampling.pnpdm_prior_steps must be >= 2")
    if float(sampling.get("pnpdm_schedule_power", 7.0)) <= 0.0:
        raise ValueError("sampling.pnpdm_schedule_power must be positive")
    pnpdm_solver = str(sampling.get("pnpdm_prior_solver", "sde")).lower().replace(
        "-", "_"
    )
    if pnpdm_solver == "ode":
        pnpdm_solver = "pf_ode"
    if pnpdm_solver not in {"sde", "pf_ode"}:
        raise ValueError("sampling.pnpdm_prior_solver must be sde or pf_ode")
    if str(sampling.get("pnpdm_initialization", "ls")).lower() not in {"ls", "zero"}:
        raise ValueError("sampling.pnpdm_initialization must be ls or zero")
    pnpdm_estimate_mode = str(
        sampling.get("pnpdm_estimate_mode", "posterior_mean")
    ).lower().replace("-", "_")
    if pnpdm_estimate_mode == "mean":
        pnpdm_estimate_mode = "posterior_mean"
    if pnpdm_estimate_mode not in {"last", "posterior_mean"}:
        raise ValueError(
            "sampling.pnpdm_estimate_mode must be last or posterior_mean"
        )
    pnpdm_burn_in = int(sampling.get("pnpdm_burn_in", 40))
    if pnpdm_burn_in < 0:
        raise ValueError("sampling.pnpdm_burn_in must be non-negative")
    pnpdm_thin = int(sampling.get("pnpdm_thin", 3))
    if pnpdm_thin < 1:
        raise ValueError("sampling.pnpdm_thin must be positive")
    if solver == "pnp_dm":
        pnpdm_iterations = int(sampling.get("pnp_iterations", 4))
        if pnpdm_rho_min < sigma_min or pnpdm_rho_start > sigma_max:
            raise ValueError(
                "sampling PnP-DM coupling schedule must lie inside "
                "consistency.[sigma_min,sigma_max]"
            )
        if (
            pnpdm_estimate_mode == "posterior_mean"
            and pnpdm_burn_in + pnpdm_thin
            > pnpdm_iterations
        ):
            raise ValueError(
                "sampling PnP-DM posterior_mean requires "
                "burn_in + thin <= pnp_iterations"
            )
        retained_capacity = max(
            0, (pnpdm_iterations - pnpdm_burn_in) // pnpdm_thin
        )
        first_retained = pnpdm_burn_in + pnpdm_thin
        if (
            pnpdm_estimate_mode == "posterior_mean"
            and retained_capacity > 1
            and pnpdm_rho_decay < 1.0
            and pnpdm_rho_start * (pnpdm_rho_decay ** (first_retained - 1))
            > pnpdm_rho_min * (1.0 + 1.0e-12)
        ):
            raise ValueError(
                "sampling PnP-DM posterior_mean would mix changing rho values; "
                "increase burn_in until retained states start at rho_min"
            )
    if str(sampling.get("flow_initialization", "ls")).lower() not in {
        "ls",
        "adjoint",
        "zero",
    }:
        raise ValueError("sampling.flow_initialization must be ls, adjoint, or zero")
    if float(sampling.get("flow_latent_std", 1.0)) <= 0.0:
        raise ValueError("sampling.flow_latent_std must be positive")
    if float(sampling.get("pnp_flow_step_scale", 1.0)) <= 0.0:
        raise ValueError("sampling.pnp_flow_step_scale must be positive")
    pnp_flow_alpha = float(sampling.get("pnp_flow_alpha", 1.0))
    if not 0.0 < pnp_flow_alpha <= 1.0:
        raise ValueError("sampling.pnp_flow_alpha must lie in (0,1]")
    pnp_flow_time_min = float(sampling.get("pnp_flow_time_min", 0.0))
    pnp_flow_time_max = float(sampling.get("pnp_flow_time_max", 1.0))
    if not 0.0 <= pnp_flow_time_min < pnp_flow_time_max <= 1.0:
        raise ValueError(
            "sampling PnP-Flow times require 0 <= time_min < time_max <= 1"
        )
    if int(sampling.get("pnp_flow_latent_samples", 5)) < 1:
        raise ValueError("sampling.pnp_flow_latent_samples must be positive")
    if float(sampling.get("flow_admm_tau", 1.0)) <= 0.0:
        raise ValueError("sampling.flow_admm_tau must be positive")
    flow_admm_time_min = float(sampling.get("flow_admm_time_min", 0.1))
    flow_admm_time_max = float(sampling.get("flow_admm_time_max", 0.95))
    if not 0.0 <= flow_admm_time_min < flow_admm_time_max <= 1.0:
        raise ValueError(
            "sampling FlowADMM times require 0 <= time_min < time_max <= 1"
        )
    if float(sampling.get("flow_admm_gamma", 1.0)) <= 0.0:
        raise ValueError("sampling.flow_admm_gamma must be positive")
    if int(sampling.get("flow_admm_k_avg", 5)) < 1:
        raise ValueError("sampling.flow_admm_k_avg must be positive")
    flow_admm_schedule = str(
        sampling.get("flow_admm_k_avg_schedule", "constant")
    ).lower().replace("-", "_")
    if flow_admm_schedule not in {"constant", "three_phase"}:
        raise ValueError(
            "sampling.flow_admm_k_avg_schedule must be constant or three_phase"
        )


def load_config(
    experiment_path: str | os.PathLike[str] | None = None,
    hardware_path: str | os.PathLike[str] | None = None,
    local_path: str | os.PathLike[str] | None = None,
    cli_overrides: Mapping[str, Any] | Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load, merge, validate and resolve one experiment configuration.

    ``cli_overrides`` accepts either a nested mapping or repeated
    ``dotted.key=value`` strings.  Pass an explicit ``environ`` in tests; the
    default is ``os.environ``.
    """

    root = _portable_path(project_root or PROJECT_ROOT, Path.cwd())
    env = os.environ if environ is None else environ

    experiment = _portable_path(
        experiment_path or root / "configs/experiments/ckmsrf_prior.yaml", root
    )
    config = _deep_merge(DEFAULT_CONFIG, _read_yaml(experiment))

    hardware: Path | None = None
    if hardware_path is not None:
        hardware = _portable_path(hardware_path, root)
        config = _deep_merge(config, _read_yaml(hardware))

    local: Path | None = None
    local_source = local_path or env.get("CKMSRF_LOCAL_CONFIG")
    if local_source is not None:
        local = _portable_path(local_source, root)
        config = _deep_merge(config, _read_yaml(local))
    else:
        candidate = root / "configs/local.yaml"
        if candidate.is_file():
            local = candidate.resolve()
            config = _deep_merge(config, _read_yaml(local))

    config = _deep_merge(config, _environment_overrides(env))
    if cli_overrides:
        if isinstance(cli_overrides, Mapping):
            cli_mapping = dict(cli_overrides)
        else:
            cli_mapping = parse_cli_overrides(cli_overrides)
        config = _deep_merge(config, cli_mapping)

    _resolve_path_fields(config, root)
    config["_meta"] = {
        "project_root": root,
        "experiment_config": experiment,
        "hardware_config": hardware,
        "local_config": local,
        "merge_order": [
            "defaults",
            "experiment",
            "hardware",
            "local",
            "environment",
            "cli",
        ],
    }
    _validate_config(config)
    return config


def to_serializable(value: Any) -> Any:
    """Recursively convert paths and tuples to checkpoint/YAML-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def save_resolved_config(config: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            to_serializable(config), handle, sort_keys=False, allow_unicode=True
        )
    temporary.replace(destination)
    return destination


@dataclass(frozen=True)
class RuntimeSpec:
    """Resolved execution device and automatic mixed-precision policy."""

    device: Any
    precision: str
    mixed_precision: str
    autocast_dtype: Any | None
    amp_enabled: bool


def resolve_device_precision(config: Mapping[str, Any]) -> RuntimeSpec:
    """Resolve ``device=auto`` and ``precision=auto`` against local hardware."""

    import torch

    hardware = config.get("hardware", config)
    requested_device = str(hardware.get("device", "auto")).lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")

    precision = str(hardware.get("precision", "auto")).lower()
    aliases = {
        "float32": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    precision = aliases.get(precision, precision)
    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        elif device.type == "mps":
            precision = "fp16"
        else:
            precision = "fp32"

    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this CUDA device lacks support")
        if device.type == "mps":
            raise RuntimeError("BF16 is not supported by the MPS runtime")
        dtype = torch.bfloat16
        mixed_precision = "bf16"
    elif precision == "fp16":
        if device.type == "cpu":
            raise RuntimeError("FP16 training on CPU is unsupported; use fp32")
        dtype = torch.float16
        mixed_precision = "fp16"
    elif precision == "fp32":
        dtype = None
        mixed_precision = "no"
    else:  # protected by validation for full configs
        raise ValueError(f"Unknown precision: {precision}")

    return RuntimeSpec(
        device=device,
        precision=precision,
        mixed_precision=mixed_precision,
        autocast_dtype=dtype,
        amp_enabled=precision != "fp32",
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common portable-configuration arguments to a CLI parser."""

    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML")
    parser.add_argument("--profile", type=Path, default=None, help="Hardware YAML")
    parser.add_argument("--local-config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--resume", default=None, help="auto, none, or checkpoint path")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--precision", default=None, help="auto, fp32, fp16, or bf16")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted-key override",
    )
    return parser


def _namespace_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides = parse_cli_overrides(getattr(args, "set_values", []))
    direct = {
        "paths.data_root": getattr(args, "data_root", None),
        "paths.run_root": getattr(args, "run_root", None),
        "training.resume": getattr(args, "resume", None),
        "hardware.device": getattr(args, "device", None),
        "hardware.precision": getattr(args, "precision", None),
        "hardware.batch_size": getattr(args, "batch_size", None),
        "hardware.grad_accum_steps": getattr(args, "grad_accum_steps", None),
        "hardware.num_workers": getattr(args, "num_workers", None),
    }
    for dotted_key, value in direct.items():
        if value is not None:
            set_by_dotted_key(overrides, dotted_key, value)
    return overrides


def load_config_from_args(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    return load_config(
        experiment_path=getattr(args, "config", None),
        hardware_path=getattr(args, "profile", None),
        local_path=getattr(args, "local_config", None),
        cli_overrides=_namespace_overrides(args),
        environ=environ,
        project_root=project_root,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "ENV_OVERRIDES",
    "PROJECT_ROOT",
    "RuntimeSpec",
    "add_config_arguments",
    "load_config",
    "load_config_from_args",
    "parse_cli_overrides",
    "resolve_device_precision",
    "save_resolved_config",
    "set_by_dotted_key",
    "to_serializable",
]
