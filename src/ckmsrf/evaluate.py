"""Deterministic scene-wise CKM-SRF posterior evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .condition_corruption import (
    BRANCH_ORDER,
    CONDITION_ID_FIELDS,
    StructuredConditionCorruptor,
    profile_fingerprint,
    stable_corrupt_batch,
)
from .config import (
    add_config_arguments,
    load_config_from_args,
    parse_cli_overrides,
    to_serializable,
)
from .data import NormalizationStats
from .engine import create_runtime, dataloader_kwargs, seed_everything
from .factory import build_dataset, build_model, load_or_build_index, normalization_path
from .physics import generate_observation
from .sampling import sample_ckmsrf
from .sampling_cm import sample_ckmcm_pnp
from .sampling_flow import sample_flow_admm, sample_pnp_flow
from .sampling_pnpdm import sample_ckm_diffusion_pnpdm
from .losses.gmflow import extract_condition
from .subspace_metrics import SubspaceRecorder, problem_fingerprints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CKM-SRF on fixed test slices.")
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subspace-metrics", action="store_true",
                        help="Save paired per-sample pilot range/null errors (single process).")
    parser.add_argument("--subspace-rank-rtol", type=float, default=1e-6)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=[-10, -5, 0, 5, 10, 15, 20, 25, 30],
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--sampler",
        choices=(
            "auto",
            "srf-rc",
            "pnp-cm",
            "pnp-admm",
            "pnp-dm",
            "pnp-flow",
            "flow-admm",
        ),
        default="auto",
        help="Inference solver; auto selects from the checkpoint model type.",
    )
    parser.add_argument("--pnp-iterations", type=int, default=None)
    parser.add_argument(
        "--pnp-iteration-grid",
        nargs="+",
        default=None,
        metavar="K",
        help=(
            "Evaluate multiple PnP iteration counts in one trajectory. "
            "Accepts integers, comma-separated values, and inclusive ranges "
            "such as 1:40 or 1:40:2."
        ),
    )
    parser.add_argument("--pnp-rho", type=float, default=None)
    parser.add_argument("--pnp-lambda", type=float, default=None)
    parser.add_argument("--pnp-momentum", type=float, default=None)
    parser.add_argument("--pnpdm-rho-start", type=float, default=None)
    parser.add_argument("--pnpdm-rho-min", type=float, default=None)
    parser.add_argument("--pnpdm-rho-decay", type=float, default=None)
    parser.add_argument("--pnpdm-prior-steps", type=int, default=None)
    parser.add_argument("--pnpdm-schedule-power", type=float, default=None)
    parser.add_argument(
        "--pnpdm-prior-solver",
        choices=("sde", "pf-ode", "ode"),
        default=None,
    )
    parser.add_argument(
        "--pnpdm-initialization",
        choices=("ls", "zero"),
        default=None,
    )
    parser.add_argument(
        "--pnpdm-estimate-mode",
        choices=("last", "posterior-mean", "mean"),
        default=None,
    )
    parser.add_argument("--pnpdm-burn-in", type=int, default=None)
    parser.add_argument("--pnpdm-thin", type=int, default=None)
    parser.add_argument("--pnpdm-seed", type=int, default=None)
    parser.add_argument(
        "--flow-initialization",
        choices=("ls", "adjoint", "zero"),
        default=None,
        help="Initial channel for PnP-Flow/FlowADMM; LS is the fair MIMO default.",
    )
    parser.add_argument("--flow-latent-std", type=float, default=None)
    parser.add_argument("--flow-latent-seed", type=int, default=None)
    parser.add_argument("--pnp-flow-step-scale", type=float, default=None)
    parser.add_argument("--pnp-flow-alpha", type=float, default=None)
    parser.add_argument("--pnp-flow-time-min", type=float, default=None)
    parser.add_argument("--pnp-flow-time-max", type=float, default=None)
    parser.add_argument("--pnp-flow-latent-samples", type=int, default=None)
    parser.add_argument("--flow-admm-tau", type=float, default=None)
    parser.add_argument("--flow-admm-time-min", type=float, default=None)
    parser.add_argument("--flow-admm-time-max", type=float, default=None)
    parser.add_argument("--flow-admm-gamma", type=float, default=None)
    parser.add_argument("--flow-admm-k-avg", type=int, default=None)
    parser.add_argument(
        "--flow-admm-k-avg-schedule",
        choices=("constant", "three-phase"),
        default=None,
    )
    parser.add_argument(
        "--condition-mode",
        choices=("clean", "mismatch"),
        default="clean",
        help=(
            "CKM presented to the prior. 'mismatch' perturbs only the "
            "normalized condition; H, P and Y stay untouched."
        ),
    )
    parser.add_argument(
        "--condition-corruption-profile",
        type=Path,
        default=None,
        help=(
            "YAML profile containing condition_corruption or robust_condition. "
            "Required for --condition-mode mismatch unless the saved config "
            "contains one of these sections."
        ),
    )
    parser.add_argument(
        "--condition-corruption-seed",
        type=int,
        default=260211950,
        help=(
            "Base seed for fixed per-sample CKM mismatch. It is independent "
            "of SNR, batch size and DDP rank."
        ),
    )
    parser.add_argument(
        "--adaptive-rho",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--outer-loops", type=int, default=None)
    parser.add_argument("--pbg-steps", type=int, default=None)
    parser.add_argument("--pbg-step-size", type=float, default=None)
    parser.add_argument("--fig-plus-mix", type=float, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-model-weights", action="store_true")
    parser.add_argument(
        "--allow-checkpoint-config-mismatch",
        action="store_true",
        help=(
            "Allow an explicit config whose learned-prior semantics differ "
            "from the checkpoint metadata. This is unsafe unless the caller "
            "also handles the changed output interpretation."
        ),
    )
    parser.add_argument("--refresh-index", action="store_true")
    return parser


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint root must be a mapping")
    return payload


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _config_for_evaluation(args: argparse.Namespace, payload: Mapping[str, Any]) -> dict[str, Any]:
    current = load_config_from_args(args)
    saved = payload.get("config")
    if args.config is not None or not isinstance(saved, Mapping):
        config = current
        if not isinstance(saved, Mapping) and args.config is None:
            raise ValueError("A legacy checkpoint without config requires --config")
    else:
        config = copy.deepcopy(dict(saved))
        # Hardware is inherently machine-local. Paths are replaced when the
        # current local/env/CLI resolution points to an existing data root or
        # when the user explicitly supplied an override.
        config["hardware"] = copy.deepcopy(current["hardware"])
        explicit_data = args.data_root is not None or bool(os.getenv("CKMSRF_DATA_ROOT"))
        explicit_run = args.run_root is not None or bool(os.getenv("CKMSRF_RUN_ROOT"))
        if explicit_data or Path(current["paths"]["data_root"]).exists():
            config.setdefault("paths", {})["data_root"] = current["paths"]["data_root"]
        if explicit_run:
            config.setdefault("paths", {})["run_root"] = current["paths"]["run_root"]
        config["_meta"] = current.get("_meta", {})
        config = _deep_merge(config, parse_cli_overrides(args.set_values))

    # Direct common CLI flags are already reflected in ``current``. Apply the
    # ones that are meaningful when using a saved scientific configuration.
    if args.device is not None:
        config.setdefault("hardware", {})["device"] = args.device
    if args.precision is not None:
        config.setdefault("hardware", {})["precision"] = args.precision
    if args.batch_size is not None:
        config.setdefault("hardware", {})["eval_batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.setdefault("hardware", {})["num_workers"] = args.num_workers
    return config


def _prior_semantics(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return canonical training method and network-output interpretation."""

    model_type = str(config.get("model", {}).get("type", "gmflow")).lower()
    if model_type == "cm":
        model_type = "consistency"
    if model_type in {"straight_flow_matching", "straight_flow"}:
        output_parameterization = "straight_velocity"
    elif model_type in {"flow_matching", "velocity_flow"}:
        output_parameterization = "velocity"
    elif model_type == "consistency":
        output_parameterization = "clean"
    else:
        output_parameterization = "distribution"

    loss_type = str(config.get("loss", {}).get("type", "")).lower()
    if loss_type in {
        "straight_flow_matching",
        "straight_velocity_matching",
        "pnp_flow_matching",
    }:
        prior_method = "straight_flow_matching"
    elif loss_type in {"flow_matching", "velocity_matching", "fm"}:
        prior_method = "flow_matching"
    elif loss_type in {"diffusion", "denoising_diffusion", "edm"}:
        prior_method = "diffusion"
    elif model_type == "consistency":
        prior_method = "consistency"
    else:
        prior_method = "gmflow"
    return prior_method, output_parameterization


def _check_checkpoint_config_semantics(
    config: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    allow_mismatch: bool,
) -> None:
    """Prevent same-shaped direct heads from being interpreted incorrectly."""

    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        return
    current_method, current_output = _prior_semantics(config)
    saved_method, saved_output = _prior_semantics(saved)
    mismatches = []
    if current_output != saved_output:
        mismatches.append(
            f"output parameterization {saved_output!r} -> {current_output!r}"
        )
    if current_method != saved_method:
        mismatches.append(f"training method {saved_method!r} -> {current_method!r}")
    # CKM-conditioned and force-unconditional U-Nets intentionally have the
    # same parameter shapes. Strict state-dict loading cannot detect swapping
    # their configs, so guard the complete learned-prior fingerprint here.
    for label, key in (
        ("model configuration", "model"),
        ("consistency configuration", "consistency"),
        ("diffusion configuration", "diffusion"),
    ):
        saved_value = saved.get(key, {})
        current_value = config.get(key, {})
        if saved_value != current_value:
            mismatches.append(f"{label} differs")
    if mismatches and not allow_mismatch:
        details = "; ".join(mismatches)
        raise ValueError(
            "Evaluation config is semantically incompatible with the checkpoint: "
            f"{details}. Omit --config to use the saved scientific config, pass a "
            "matching config, or use --allow-checkpoint-config-mismatch only for "
            "an intentional expert ablation."
        )


def _load_model_weights(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    prefer_ema: bool,
) -> str:
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint has no model state dictionary")
    model.load_state_dict(state, strict=True)
    ema = payload.get("ema")
    if prefer_ema and isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping):
        current = model.state_dict()
        for name, value in ema["shadow"].items():
            if name in current:
                current[name].copy_(value.to(dtype=current[name].dtype))
        return "ema"
    return "model"


def _normalization(config: Mapping[str, Any], payload: Mapping[str, Any]) -> NormalizationStats | None:
    saved = payload.get("normalization")
    if isinstance(saved, Mapping) and saved:
        return NormalizationStats(**dict(saved))
    mode = str(
        config.get("data", {}).get("normalization", {}).get("mode", "global_rms")
    ).lower()
    if mode in {"none", "off", "false"}:
        return None
    path = normalization_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Normalization was not stored in the checkpoint and is missing: {path}"
        )
    return NormalizationStats.load(path)


def _to_db(value: float) -> float:
    return 10.0 * torch.log10(torch.tensor(max(value, 1.0e-15))).item()


def _parse_iteration_grid(values: list[str]) -> tuple[int, ...]:
    iterations: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                iterations.add(int(token))
                continue
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(
                    f"Invalid iteration range {token!r}; use START:STOP[:STEP]"
                )
            start, stop = int(parts[0]), int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError(
                    f"Invalid iteration range {token!r}; require STOP >= START and STEP > 0"
                )
            iterations.update(range(start, stop + 1, step))
    if not iterations or min(iterations) < 1:
        raise ValueError("PnP iteration counts must be positive integers")
    return tuple(sorted(iterations))


def _output_dir(args: argparse.Namespace, checkpoint: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return run_dir / "evaluations" / f"{args.split}_{stamp}"


def _read_condition_corruption_profile(path: Path) -> dict[str, Any]:
    """Load a portable corruption profile without changing scientific config."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Condition-corruption profile does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Condition-corruption profile must be a mapping: {resolved}")
    for key in ("condition_corruption", "robust_condition"):
        value = loaded.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return dict(loaded)


def _condition_corruption_setup(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[StructuredConditionCorruptor | None, dict[str, Any] | None, str | None]:
    """Resolve an optional profile and enforce explicit mismatch provenance."""

    if args.condition_corruption_profile is not None:
        profile = _read_condition_corruption_profile(args.condition_corruption_profile)
        source = str(args.condition_corruption_profile.expanduser().resolve())
    else:
        configured = config.get("condition_corruption")
        source = "config.condition_corruption"
        if not isinstance(configured, Mapping):
            configured = config.get("robust_condition")
            source = "config.robust_condition"
        profile = dict(configured) if isinstance(configured, Mapping) else None
    if profile is None:
        if args.condition_mode == "mismatch":
            raise ValueError(
                "--condition-mode mismatch requires --condition-corruption-profile "
                "or a condition_corruption/robust_condition section in the config"
            )
        return None, None, None
    if args.condition_mode == "mismatch":
        profile = {**profile, "enabled": True}
    adp_shape = tuple(
        config.get("model", {}).get("condition", {}).get("adp_shape", (16, 16, 12))
    )
    corruptor = StructuredConditionCorruptor.from_config(profile, adp_shape=adp_shape)
    if args.condition_mode == "mismatch" and not corruptor.enabled:
        raise ValueError("The resolved mismatch profile must have enabled=true")
    return corruptor, corruptor.as_dict(), source


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = _load_checkpoint(checkpoint)
    config = _config_for_evaluation(args, payload)
    _check_checkpoint_config_semantics(
        config,
        payload,
        allow_mismatch=bool(args.allow_checkpoint_config_mismatch),
    )
    condition_corruptor, condition_profile, condition_profile_source = (
        _condition_corruption_setup(args, config)
    )
    seed = int(config.get("training", {}).get("validation_seed", 9102))
    seed_everything(seed)
    runtime = create_runtime(config)
    if args.subspace_metrics and int(getattr(runtime, "num_processes", 1)) != 1:
        raise ValueError("--subspace-metrics currently requires single-process evaluation")

    index = load_or_build_index(config, args.split, refresh=args.refresh_index)
    stats = _normalization(config, payload)
    dataset = build_dataset(
        config,
        args.split,
        index=index,
        normalization=stats,
        include_observation=True,
    )
    view = dataset
    if args.max_samples is not None and args.max_samples < len(dataset):
        view = Subset(dataset, range(max(0, args.max_samples)))
    loader_kwargs = dataloader_kwargs(config, training=False, seed=seed)
    loader = DataLoader(view, drop_last=False, **loader_kwargs)

    model = build_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    weights_source = _load_model_weights(
        model, payload, prefer_ema=not args.use_model_weights
    )
    prior_method, configured_output_parameterization = _prior_semantics(config)
    output_parameterization = str(
        getattr(model, "output_parameterization", configured_output_parameterization)
    ).lower()
    model, loader = runtime.prepare(model, loader)
    model.eval()

    sampler_cfg = config.get("sampling", {})
    consistency_cfg = config.get("consistency", {})
    model_type = str(config.get("model", {}).get("type", "gmflow")).lower()
    direct_prior_types = {"consistency", "cm", "flow_matching", "velocity_flow"}
    straight_flow_types = {"straight_flow_matching", "straight_flow"}
    denoiser_admm_samplers = {"pnp-cm", "pnp-admm"}
    pnpdm_samplers = {"pnp-dm"}
    flow_solver_samplers = {"pnp-flow", "flow-admm"}
    iterative_samplers = (
        denoiser_admm_samplers | pnpdm_samplers | flow_solver_samplers
    )
    sampler = str(args.sampler)
    if sampler == "auto":
        configured_solver = str(sampler_cfg.get("solver", "")).lower().replace("-", "_")
        if model_type in straight_flow_types:
            sampler = "pnp-flow" if configured_solver == "pnp_flow" else "flow-admm"
        elif configured_solver == "pnp_dm":
            sampler = "pnp-dm"
        elif configured_solver == "pnp_admm" or model_type in {
            "flow_matching",
            "velocity_flow",
        }:
            sampler = "pnp-admm"
        elif configured_solver == "pnp_cm" or model_type in {"consistency", "cm"}:
            sampler = "pnp-cm"
        else:
            sampler = "srf-rc"
    if sampler == "pnp-cm" and model_type not in {"consistency", "cm"}:
        raise ValueError("pnp-cm requires a checkpoint with model.type=consistency")
    if sampler == "pnp-admm" and model_type not in direct_prior_types:
        raise ValueError(
            "pnp-admm requires a direct clean or velocity-prior checkpoint"
        )
    if sampler in pnpdm_samplers and (
        model_type not in {"consistency", "cm"}
        or prior_method != "diffusion"
        or output_parameterization != "clean"
    ):
        raise ValueError(
            "pnp-dm requires a clean-output checkpoint trained with "
            "loss.type=diffusion"
        )
    if sampler in flow_solver_samplers and model_type not in straight_flow_types:
        raise ValueError(
            f"{sampler} requires a checkpoint with "
            "model.type=straight_flow_matching"
        )
    if sampler == "srf-rc" and model_type in direct_prior_types:
        raise ValueError("srf-rc expects a GMFlow checkpoint, not a direct prior")
    if sampler == "srf-rc" and model_type in straight_flow_types:
        raise ValueError("srf-rc cannot interpret a straight-flow checkpoint")
    steps = int(args.steps if args.steps is not None else sampler_cfg.get("steps", 20))
    outer_loops = int(
        args.outer_loops if args.outer_loops is not None else sampler_cfg.get("outer_loops", 1)
    )
    pbg_steps = int(
        args.pbg_steps if args.pbg_steps is not None else sampler_cfg.get("pbg_steps", 0)
    )
    pbg_step = float(
        args.pbg_step_size
        if args.pbg_step_size is not None
        else sampler_cfg.get("pbg_step_size", 0.15)
    )
    fig_mix = float(
        args.fig_plus_mix
        if args.fig_plus_mix is not None
        else sampler_cfg.get("fig_plus_mix", 0.0)
    )
    pnp_iterations = int(
        args.pnp_iterations
        if args.pnp_iterations is not None
        else sampler_cfg.get("pnp_iterations", 4)
    )
    iteration_sweep = args.pnp_iteration_grid is not None
    if iteration_sweep:
        if args.pnp_iterations is not None:
            raise ValueError(
                "Use either --pnp-iterations or --pnp-iteration-grid, not both"
            )
        if sampler not in iterative_samplers:
            raise ValueError(
                "--pnp-iteration-grid requires an iterative PnP/Flow solver"
            )
        iteration_grid = _parse_iteration_grid(args.pnp_iteration_grid)
        pnp_iterations = max(iteration_grid)
    else:
        iteration_grid = (pnp_iterations,)
    if pnp_iterations < 1:
        raise ValueError("--pnp-iterations must be positive")
    pnp_rho = float(
        args.pnp_rho if args.pnp_rho is not None else sampler_cfg.get("pnp_rho", 1.0)
    )
    pnp_lambda = float(
        args.pnp_lambda
        if args.pnp_lambda is not None
        else sampler_cfg.get("pnp_lambda", 0.1)
    )
    pnp_momentum = float(
        args.pnp_momentum
        if args.pnp_momentum is not None
        else sampler_cfg.get("pnp_momentum", 0.0)
    )
    pnpdm_rho_start = float(
        args.pnpdm_rho_start
        if args.pnpdm_rho_start is not None
        else sampler_cfg.get("pnpdm_rho_start", 1.0)
    )
    pnpdm_rho_min = float(
        args.pnpdm_rho_min
        if args.pnpdm_rho_min is not None
        else sampler_cfg.get("pnpdm_rho_min", 0.05)
    )
    pnpdm_rho_decay = float(
        args.pnpdm_rho_decay
        if args.pnpdm_rho_decay is not None
        else sampler_cfg.get("pnpdm_rho_decay", 0.9)
    )
    pnpdm_prior_steps = int(
        args.pnpdm_prior_steps
        if args.pnpdm_prior_steps is not None
        else sampler_cfg.get("pnpdm_prior_steps", 100)
    )
    pnpdm_schedule_power = float(
        args.pnpdm_schedule_power
        if args.pnpdm_schedule_power is not None
        else sampler_cfg.get("pnpdm_schedule_power", 7.0)
    )
    pnpdm_prior_solver = str(
        args.pnpdm_prior_solver
        if args.pnpdm_prior_solver is not None
        else sampler_cfg.get("pnpdm_prior_solver", "sde")
    ).replace("-", "_")
    if pnpdm_prior_solver == "ode":
        pnpdm_prior_solver = "pf_ode"
    pnpdm_initialization = str(
        args.pnpdm_initialization
        if args.pnpdm_initialization is not None
        else sampler_cfg.get("pnpdm_initialization", "ls")
    )
    pnpdm_estimate_mode = str(
        args.pnpdm_estimate_mode
        if args.pnpdm_estimate_mode is not None
        else sampler_cfg.get("pnpdm_estimate_mode", "posterior_mean")
    ).replace("-", "_")
    if pnpdm_estimate_mode == "mean":
        pnpdm_estimate_mode = "posterior_mean"
    pnpdm_burn_in = int(
        args.pnpdm_burn_in
        if args.pnpdm_burn_in is not None
        else sampler_cfg.get("pnpdm_burn_in", 40)
    )
    pnpdm_thin = int(
        args.pnpdm_thin
        if args.pnpdm_thin is not None
        else sampler_cfg.get("pnpdm_thin", 3)
    )
    pnpdm_seed = int(
        args.pnpdm_seed
        if args.pnpdm_seed is not None
        else sampler_cfg.get("pnpdm_seed", seed + 20_000_000)
    )
    if sampler in pnpdm_samplers:
        sigma_min = float(consistency_cfg.get("sigma_min", 0.002))
        sigma_max = float(consistency_cfg.get("sigma_max", 2.0))
        if not 0.0 < pnpdm_rho_min <= pnpdm_rho_start:
            raise ValueError("PnP-DM requires 0 < rho_min <= rho_start")
        if pnpdm_rho_min < sigma_min or pnpdm_rho_start > sigma_max:
            raise ValueError(
                "PnP-DM coupling schedule must lie inside the denoiser sigma range"
            )
        if not 0.0 < pnpdm_rho_decay <= 1.0:
            raise ValueError("PnP-DM rho decay must lie in (0,1]")
        if pnpdm_prior_steps < 2 or pnpdm_schedule_power <= 0.0:
            raise ValueError(
                "PnP-DM prior steps must be >= 2 and schedule power must be positive"
            )
        if pnpdm_prior_solver not in {"sde", "pf_ode"}:
            raise ValueError("PnP-DM prior solver must be sde or pf_ode")
        if pnpdm_burn_in < 0 or pnpdm_thin < 1:
            raise ValueError("PnP-DM requires burn_in >= 0 and thin >= 1")
        if pnpdm_estimate_mode == "posterior_mean":
            first_retained = pnpdm_burn_in + pnpdm_thin
            invalid_iterations = [
                value for value in iteration_grid if value < first_retained
            ]
            if invalid_iterations:
                raise ValueError(
                    "PnP-DM posterior_mean requires every reported K to retain "
                    "at least one state (K >= burn_in + thin); invalid K values: "
                    f"{invalid_iterations}"
                )
            retained_capacity = max(
                0, (pnp_iterations - pnpdm_burn_in) // pnpdm_thin
            )
            if (
                retained_capacity > 1
                and pnpdm_rho_decay < 1.0
                and pnpdm_rho_start
                * (pnpdm_rho_decay ** (first_retained - 1))
                > pnpdm_rho_min * (1.0 + 1.0e-12)
            ):
                raise ValueError(
                    "PnP-DM posterior_mean would mix changing rho values; "
                    "increase burn_in until retained states start at rho_min"
                )
    pnp_denoiser_relaxation = float(
        sampler_cfg.get("pnp_denoiser_relaxation", 1.0)
    )
    pnp_dual_damping = float(sampler_cfg.get("pnp_dual_damping", 1.0))
    pnp_final_data_step = bool(sampler_cfg.get("pnp_final_data_step", False))
    adaptive_rho = bool(
        args.adaptive_rho
        if args.adaptive_rho is not None
        else sampler_cfg.get("pnp_adaptive_rho", False)
    )
    pnp_rho_candidate_factors = tuple(
        float(value)
        for value in sampler_cfg.get(
            "pnp_rho_candidate_factors", (0.5, 1.0, 2.0)
        )
    )
    pnp_rho_min = float(sampler_cfg.get("pnp_rho_min", 1.0e-6))
    pnp_rho_max = float(sampler_cfg.get("pnp_rho_max", 1.0e6))
    pnp_whiteness_weight = float(
        sampler_cfg.get("pnp_whiteness_weight", 0.25)
    )
    flow_initialization = str(
        args.flow_initialization
        if args.flow_initialization is not None
        else sampler_cfg.get("flow_initialization", "ls")
    )
    flow_latent_std = float(
        args.flow_latent_std
        if args.flow_latent_std is not None
        else sampler_cfg.get("flow_latent_std", 1.0)
    )
    flow_latent_seed = int(
        args.flow_latent_seed
        if args.flow_latent_seed is not None
        else sampler_cfg.get("flow_latent_seed", seed + 10_000_000)
    )
    pnp_flow_step_scale = float(
        args.pnp_flow_step_scale
        if args.pnp_flow_step_scale is not None
        else sampler_cfg.get("pnp_flow_step_scale", 1.0)
    )
    pnp_flow_alpha = float(
        args.pnp_flow_alpha
        if args.pnp_flow_alpha is not None
        else sampler_cfg.get("pnp_flow_alpha", 1.0)
    )
    pnp_flow_time_min = float(
        args.pnp_flow_time_min
        if args.pnp_flow_time_min is not None
        else sampler_cfg.get("pnp_flow_time_min", 0.0)
    )
    pnp_flow_time_max = float(
        args.pnp_flow_time_max
        if args.pnp_flow_time_max is not None
        else sampler_cfg.get("pnp_flow_time_max", 1.0)
    )
    pnp_flow_latent_samples = int(
        args.pnp_flow_latent_samples
        if args.pnp_flow_latent_samples is not None
        else sampler_cfg.get("pnp_flow_latent_samples", 5)
    )
    flow_admm_tau = float(
        args.flow_admm_tau
        if args.flow_admm_tau is not None
        else sampler_cfg.get("flow_admm_tau", 1.0)
    )
    flow_admm_time_min = float(
        args.flow_admm_time_min
        if args.flow_admm_time_min is not None
        else sampler_cfg.get("flow_admm_time_min", 0.1)
    )
    flow_admm_time_max = float(
        args.flow_admm_time_max
        if args.flow_admm_time_max is not None
        else sampler_cfg.get("flow_admm_time_max", 0.95)
    )
    flow_admm_gamma = float(
        args.flow_admm_gamma
        if args.flow_admm_gamma is not None
        else sampler_cfg.get("flow_admm_gamma", 1.0)
    )
    flow_admm_k_avg = int(
        args.flow_admm_k_avg
        if args.flow_admm_k_avg is not None
        else sampler_cfg.get("flow_admm_k_avg", 5)
    )
    flow_admm_k_avg_schedule = str(
        args.flow_admm_k_avg_schedule
        if args.flow_admm_k_avg_schedule is not None
        else sampler_cfg.get("flow_admm_k_avg_schedule", "constant")
    ).replace("-", "_")
    flow_admm_k_avg_early = int(sampler_cfg.get("flow_admm_k_avg_early", 1))
    flow_admm_k_avg_mid = int(sampler_cfg.get("flow_admm_k_avg_mid", 2))
    flow_admm_k_avg_late = int(sampler_cfg.get("flow_admm_k_avg_late", 5))
    flow_admm_switch_frac = float(sampler_cfg.get("flow_admm_switch_frac", 0.5))
    flow_admm_switch_frac_2 = float(
        sampler_cfg.get("flow_admm_switch_frac_2", 0.8)
    )

    output_dir = _output_dir(args, checkpoint)
    if bool(getattr(runtime, "is_main_process", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
    runtime.wait_for_everyone()
    rows: list[dict[str, Any]] = []
    subspace_recorder = (SubspaceRecorder(output_dir, rtol=args.subspace_rank_rtol)
                         if args.subspace_metrics else None)
    evaluation_keys: tuple[int | None, ...] = (
        tuple(iteration_grid) if iteration_sweep else (None,)
    )

    for snr_index, snr_db in enumerate(args.snr_db):
        sums: dict[int | None, dict[int, float]] = {
            key: {} for key in evaluation_keys
        }
        residual_sums: dict[int | None, dict[int, float]] = {
            key: {} for key in evaluation_keys
        }
        counts: dict[int | None, dict[int, int]] = {
            key: {} for key in evaluation_keys
        }
        sampling_seconds: dict[int | None, float] = {
            key: 0.0 for key in evaluation_keys
        }
        denoiser_nfe_by_key: dict[int | None, int] = {}
        denoiser_nfe_per_iteration_by_key: dict[int | None, int | None] = {}
        data_evaluations_by_key: dict[int | None, int] = {}
        posterior_sample_count_by_key: dict[int | None, int] = {}
        pnpdm_start_sigma_first_by_key: dict[int | None, float] = {}
        pnpdm_start_sigma_at_key: dict[int | None, float] = {}
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        start_time = time.perf_counter()
        progress = tqdm(
            loader,
            desc=f"SNR {snr_db:g} dB",
            disable=not bool(getattr(runtime, "is_local_main_process", True)),
        )
        for batch_index, batch in enumerate(progress):
            truth = batch["H"].to(runtime.device)
            pilot = batch["P"].to(runtime.device)
            generator_device = runtime.device if runtime.device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(seed + 100000 * snr_index + batch_index)
            observation, noise_var = generate_observation(
                truth,
                pilot,
                snr_db=torch.full(
                    (truth.shape[0],),
                    float(snr_db),
                    device=runtime.device,
                    dtype=truth.real.dtype,
                ),
                generator=generator,
            )
            inference_batch = dict(batch)
            fingerprints = (problem_fingerprints(truth, pilot, observation, noise_var)
                            if subspace_recorder is not None else None)
            inference_batch.update({"H": truth, "P": pilot, "Y": observation, "noise_var": noise_var})
            if args.condition_mode == "mismatch":
                if condition_corruptor is None:
                    raise RuntimeError("Mismatch mode requires a resolved condition corruptor")
                inference_batch = stable_corrupt_batch(
                    inference_batch,
                    condition_corruptor,
                    condition=extract_condition(inference_batch),
                    seed=int(args.condition_corruption_seed),
                )
            if sampler in pnpdm_samplers:
                process_index = int(getattr(runtime, "process_index", 0))
                pnpdm_generator = torch.Generator(device=generator_device)
                pnpdm_generator.manual_seed(
                    pnpdm_seed
                    + 100_000 * snr_index
                    + batch_index
                    + 100_000_000 * process_index
                )
                estimate, diagnostics = sample_ckm_diffusion_pnpdm(
                    model,
                    inference_batch,
                    iterations=pnp_iterations,
                    rho_start=pnpdm_rho_start,
                    rho_min=pnpdm_rho_min,
                    rho_decay=pnpdm_rho_decay,
                    sigma_min=float(consistency_cfg.get("sigma_min", 0.002)),
                    sigma_max=float(consistency_cfg.get("sigma_max", 2.0)),
                    prior_steps=pnpdm_prior_steps,
                    schedule_power=pnpdm_schedule_power,
                    prior_solver=pnpdm_prior_solver,
                    initialization=pnpdm_initialization,
                    estimate_mode=pnpdm_estimate_mode,
                    burn_in=pnpdm_burn_in,
                    thin=pnpdm_thin,
                    generator=pnpdm_generator,
                    snapshot_iterations=iteration_grid if iteration_sweep else (),
                    record_snapshot_timing=iteration_sweep,
                )
                if iteration_sweep:
                    estimates = diagnostics["iteration_estimates"]
                    for iteration, seconds in diagnostics["iteration_seconds"].items():
                        key = int(iteration)
                        sampling_seconds[key] += float(seconds)
                        denoiser_nfe_by_key[key] = int(
                            diagnostics["iteration_denoiser_nfe"][key]
                        )
                        denoiser_nfe_per_iteration_by_key[key] = None
                        data_evaluations_by_key[key] = key
                        posterior_sample_count_by_key[key] = int(
                            diagnostics["iteration_sample_count"][key]
                        )
                        start_sigmas = diagnostics["prior_start_sigma_history"]
                        pnpdm_start_sigma_first_by_key[key] = float(start_sigmas[0])
                        pnpdm_start_sigma_at_key[key] = float(start_sigmas[key - 1])
                else:
                    estimates = {None: estimate}
                    sampling_seconds[None] += float(diagnostics["sampling_seconds"])
                    denoiser_nfe_by_key[None] = int(diagnostics["denoiser_nfe"])
                    denoiser_nfe_per_iteration_by_key[None] = None
                    data_evaluations_by_key[None] = int(
                        diagnostics["data_consistency_evaluations"]
                    )
                    posterior_sample_count_by_key[None] = int(
                        diagnostics["posterior_sample_count"]
                    )
                    start_sigmas = diagnostics["prior_start_sigma_history"]
                    pnpdm_start_sigma_first_by_key[None] = float(start_sigmas[0])
                    pnpdm_start_sigma_at_key[None] = float(start_sigmas[-1])
            elif sampler in denoiser_admm_samplers:
                estimate, diagnostics = sample_ckmcm_pnp(
                    model,
                    inference_batch,
                    iterations=pnp_iterations,
                    lambda_reg=pnp_lambda,
                    rho=pnp_rho,
                    adaptive_rho=adaptive_rho,
                    sigma_min=float(consistency_cfg.get("sigma_min", 0.002)),
                    sigma_max=float(consistency_cfg.get("sigma_max", 2.0)),
                    momentum=pnp_momentum,
                    denoiser_relaxation=pnp_denoiser_relaxation,
                    dual_damping=pnp_dual_damping,
                    final_data_step=pnp_final_data_step,
                    rho_candidate_factors=pnp_rho_candidate_factors,
                    rho_min=pnp_rho_min,
                    rho_max=pnp_rho_max,
                    whiteness_weight=pnp_whiteness_weight,
                    snapshot_iterations=iteration_grid if iteration_sweep else (),
                    record_snapshot_timing=iteration_sweep,
                )
                if iteration_sweep:
                    estimates = diagnostics["iteration_estimates"]
                    for iteration, seconds in diagnostics["iteration_seconds"].items():
                        sampling_seconds[int(iteration)] += float(seconds)
                        denoiser_nfe_by_key[int(iteration)] = int(iteration)
                        denoiser_nfe_per_iteration_by_key[int(iteration)] = 1
                        data_evaluations_by_key[int(iteration)] = int(iteration)
                else:
                    estimates = {None: estimate}
                    sampling_seconds[None] += float(
                        diagnostics["sampling_seconds"]
                    )
                    denoiser_nfe_by_key[None] = int(diagnostics["denoiser_nfe"])
                    denoiser_nfe_per_iteration_by_key[None] = 1
                    data_evaluations_by_key[None] = int(pnp_iterations)
            elif sampler in flow_solver_samplers:
                estimates = {}
                requested_iterations = (
                    tuple(iteration_grid) if iteration_sweep else (pnp_iterations,)
                )
                process_index = int(getattr(runtime, "process_index", 0))
                flow_seed_for_batch = (
                    flow_latent_seed
                    + 100_000 * snr_index
                    + batch_index
                    + 100_000_000 * process_index
                )
                for requested in requested_iterations:
                    # Reset the same latent seed for every independent K run.
                    # This shares random prefixes without incorrectly reusing
                    # a trajectory whose t schedule depends on total K.
                    flow_generator = torch.Generator(device=generator_device)
                    flow_generator.manual_seed(flow_seed_for_batch)
                    if sampler == "pnp-flow":
                        current_estimate, diagnostics = sample_pnp_flow(
                            model,
                            inference_batch,
                            iterations=int(requested),
                            step_scale=pnp_flow_step_scale,
                            alpha=pnp_flow_alpha,
                            time_min=pnp_flow_time_min,
                            time_max=pnp_flow_time_max,
                            latent_samples=pnp_flow_latent_samples,
                            latent_std=flow_latent_std,
                            initialization=flow_initialization,
                            generator=flow_generator,
                        )
                    else:
                        current_estimate, diagnostics = sample_flow_admm(
                            model,
                            inference_batch,
                            iterations=int(requested),
                            tau=flow_admm_tau,
                            time_min=flow_admm_time_min,
                            time_max=flow_admm_time_max,
                            gamma=flow_admm_gamma,
                            k_avg=flow_admm_k_avg,
                            k_avg_schedule=flow_admm_k_avg_schedule,
                            k_avg_early=flow_admm_k_avg_early,
                            k_avg_mid=flow_admm_k_avg_mid,
                            k_avg_late=flow_admm_k_avg_late,
                            switch_frac=flow_admm_switch_frac,
                            switch_frac_2=flow_admm_switch_frac_2,
                            latent_std=flow_latent_std,
                            initialization=flow_initialization,
                            generator=flow_generator,
                        )
                    evaluation_key = int(requested) if iteration_sweep else None
                    estimates[evaluation_key] = current_estimate
                    sampling_seconds[evaluation_key] += float(
                        diagnostics["sampling_seconds"]
                    )
                    denoiser_nfe_by_key[evaluation_key] = int(
                        diagnostics["denoiser_nfe"]
                    )
                    value = diagnostics.get("denoiser_nfe_per_iteration")
                    denoiser_nfe_per_iteration_by_key[evaluation_key] = (
                        None if value is None else int(value)
                    )
                    data_evaluations_by_key[evaluation_key] = int(
                        diagnostics["data_consistency_evaluations"]
                    )
            else:
                estimate, _ = sample_ckmsrf(
                    model,
                    inference_batch,
                    steps=steps,
                    outer_loops=outer_loops,
                    lambda_power=float(sampler_cfg.get("lambda_power", 1.0)),
                    beta_power=float(sampler_cfg.get("beta_power", 1.0)),
                    use_rc=True,
                    pbg_steps=pbg_steps,
                    pbg_step_size=pbg_step,
                    fig_plus_mix=fig_mix,
                    generator=generator,
                )
                estimates = {None: estimate}
            scene_ids = batch["scene_id"].to(runtime.device)
            denominator = truth.abs().square().flatten(1).sum(dim=1).clamp_min(1.0e-12)
            for evaluation_key, current_estimate in estimates.items():
                if subspace_recorder is not None:
                    subspace_recorder.add(
                        current_estimate, truth, pilot, batch, snr_db=snr_db,
                        iterations=(int(evaluation_key) if evaluation_key is not None
                                    else int(pnp_iterations) if sampler in iterative_samplers else 0),
                        fingerprints=fingerprints,
                    )
                numerator = (
                    (current_estimate - truth).abs().square().flatten(1).sum(dim=1)
                )
                nmse = numerator / denominator
                residual = (
                    torch.bmm(current_estimate, pilot) - observation
                ).abs().square()
                residual = (
                    residual.flatten(1).mean(dim=1)
                    / noise_var.clamp_min(1.0e-12)
                )
                gathered_nmse, gathered_residual, gathered_scene = (
                    runtime.gather_for_metrics((nmse, residual, scene_ids))
                )
                if bool(getattr(runtime, "is_main_process", True)):
                    for value, residual_value, scene in zip(
                        gathered_nmse.detach().cpu().tolist(),
                        gathered_residual.detach().cpu().tolist(),
                        gathered_scene.detach().cpu().tolist(),
                    ):
                        scene = int(scene)
                        sums[evaluation_key][scene] = (
                            sums[evaluation_key].get(scene, 0.0) + float(value)
                        )
                        residual_sums[evaluation_key][scene] = (
                            residual_sums[evaluation_key].get(scene, 0.0)
                            + float(residual_value)
                        )
                        counts[evaluation_key][scene] = (
                            counts[evaluation_key].get(scene, 0) + 1
                        )
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break

        peak_gpu_memory_mb: float | None = None
        if runtime.device.type == "cuda":
            local_peak = torch.tensor(
                [torch.cuda.max_memory_allocated(runtime.device)],
                device=runtime.device,
                dtype=torch.float64,
            )
            gathered_peak = runtime.gather_for_metrics(local_peak)
            peak_gpu_memory_mb = float(gathered_peak.max().cpu()) / (1024.0**2)
        runtime.wait_for_everyone()
        if bool(getattr(runtime, "is_main_process", True)):
            elapsed = time.perf_counter() - start_time
            for evaluation_key in evaluation_keys:
                current_sums = sums[evaluation_key]
                current_residual_sums = residual_sums[evaluation_key]
                current_counts = counts[evaluation_key]
                row_context: dict[str, Any] = {}
                if sampler in iterative_samplers:
                    current_iterations = (
                        int(evaluation_key)
                        if iteration_sweep
                        else int(pnp_iterations)
                    )
                    row_context = {
                        "pnp_iterations": current_iterations,
                        "data_consistency_evaluations": data_evaluations_by_key.get(
                            evaluation_key, current_iterations
                        ),
                        "denoiser_nfe": denoiser_nfe_by_key.get(
                            evaluation_key, current_iterations
                        ),
                        "denoiser_nfe_per_iteration": (
                            denoiser_nfe_per_iteration_by_key.get(
                                evaluation_key, 1
                            )
                        ),
                    }
                    if sampler in pnpdm_samplers:
                        row_context.update(
                            {
                                "posterior_sample_count": (
                                    posterior_sample_count_by_key.get(evaluation_key, 0)
                                ),
                                "pnpdm_prior_steps": pnpdm_prior_steps,
                                "pnpdm_prior_solver": pnpdm_prior_solver,
                                "pnpdm_rho_start": pnpdm_rho_start,
                                "pnpdm_rho_min": pnpdm_rho_min,
                                "pnpdm_rho_decay": pnpdm_rho_decay,
                                "pnpdm_start_sigma_first": (
                                    pnpdm_start_sigma_first_by_key.get(evaluation_key)
                                ),
                                "pnpdm_start_sigma_at_k": (
                                    pnpdm_start_sigma_at_key.get(evaluation_key)
                                ),
                            }
                        )
                    if iteration_sweep and sampler in (
                        denoiser_admm_samplers | pnpdm_samplers
                    ):
                        row_context["sampling_seconds_cumulative"] = (
                            sampling_seconds[evaluation_key]
                        )
                    if sampler in flow_solver_samplers or (
                        sampler in (denoiser_admm_samplers | pnpdm_samplers)
                        and not iteration_sweep
                    ):
                        row_context["sampling_seconds"] = sampling_seconds[
                            evaluation_key
                        ]
                    row_context["peak_gpu_memory_mb"] = peak_gpu_memory_mb
                scene_nmse = []
                for scene in sorted(current_counts):
                    linear = current_sums[scene] / max(current_counts[scene], 1)
                    scene_nmse.append(linear)
                    rows.append(
                        {
                            **row_context,
                            "snr_db": float(snr_db),
                            "scene_id": scene,
                            "num_samples": current_counts[scene],
                            "nmse_linear": linear,
                            "nmse_db": _to_db(linear),
                            "normalized_measurement_residual": current_residual_sums[scene]
                            / max(current_counts[scene], 1),
                            "seconds": elapsed,
                        }
                    )
                total_count = sum(current_counts.values())
                macro = sum(scene_nmse) / max(len(scene_nmse), 1)
                overall = sum(current_sums.values()) / max(total_count, 1)
                rows.extend(
                    [
                        {
                            **row_context,
                            "snr_db": float(snr_db),
                            "scene_id": "macro",
                            "num_samples": total_count,
                            "nmse_linear": macro,
                            "nmse_db": _to_db(macro),
                            "normalized_measurement_residual": None,
                            "seconds": elapsed,
                        },
                        {
                            **row_context,
                            "snr_db": float(snr_db),
                            "scene_id": "overall",
                            "num_samples": total_count,
                            "nmse_linear": overall,
                            "nmse_db": _to_db(overall),
                            "normalized_measurement_residual": sum(
                                current_residual_sums.values()
                            )
                            / max(total_count, 1),
                            "seconds": elapsed,
                        },
                    ]
                )
                prefix = (
                    f"SNR {snr_db:g} dB, K={evaluation_key}"
                    if iteration_sweep
                    else f"SNR {snr_db:g} dB"
                )
                print(f"{prefix}: overall NMSE={_to_db(overall):.3f} dB")

    if bool(getattr(runtime, "is_main_process", True)):
        csv_path = output_dir / "metrics.csv"
        if subspace_recorder is not None:
            subspace_recorder.finish()
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "prior": {
                "method": prior_method,
                "model_type": model_type,
                "output_parameterization": output_parameterization,
                "flow_path": (
                    "straight_latent_to_data"
                    if model_type in straight_flow_types
                    else "additive_ve"
                    if model_type in {"flow_matching", "velocity_flow"}
                    else None
                ),
                "flow_coupling": (
                    str(
                        config.get("straight_flow_matching", {}).get(
                            "coupling", "independent"
                        )
                    )
                    if model_type in straight_flow_types
                    else None
                ),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "weights_source": weights_source,
            },
            "evaluation": {
                "subspace_metrics": bool(args.subspace_metrics),
                "subspace_rank_rtol": args.subspace_rank_rtol if args.subspace_metrics else None,
                "validation_seed": seed,
                "max_samples": args.max_samples,
                "max_batches": args.max_batches,
                "eval_batch_size": int(
                    config.get("hardware", {}).get("eval_batch_size", 1)
                ),
                "precision": str(
                    config.get("hardware", {}).get("precision", "auto")
                ),
                "world_size": int(getattr(runtime, "num_processes", 1)),
                "normalization": (
                    None if stats is None else to_serializable(stats.to_dict())
                ),
                "condition": {
                    "mode": args.condition_mode,
                    "corruption_seed": int(args.condition_corruption_seed),
                    "profile_source": condition_profile_source,
                    "profile": condition_profile,
                    "profile_fingerprint": (
                        None
                        if condition_profile is None
                        else profile_fingerprint(condition_profile)
                    ),
                    "branch_order": list(BRANCH_ORDER),
                    "stable_identifier_fields": list(CONDITION_ID_FIELDS),
                    "snr_independent": True,
                    "batch_partition_independent": True,
                    "channel_pilot_observation_modified": False,
                },
            },
            "sampler": {
                "name": sampler,
                "steps": steps,
                "outer_loops": outer_loops,
                "pbg_steps": pbg_steps,
                "pbg_step_size": pbg_step,
                "fig_plus_mix": fig_mix,
                "pnp_iterations": pnp_iterations,
                "pnp_iteration_grid": (
                    list(iteration_grid) if iteration_sweep else None
                ),
                "solver_family": (
                    "pnp_flow_fbs"
                    if sampler == "pnp-flow"
                    else "flow_admm"
                    if sampler == "flow-admm"
                    else "split_gibbs_pnp_dm"
                    if sampler in pnpdm_samplers
                    else "direct_prior_pnp_admm"
                    if sampler in denoiser_admm_samplers
                    else "srf_rc"
                ),
                "trajectory_reuse": bool(
                    iteration_sweep
                    and sampler in (denoiser_admm_samplers | pnpdm_samplers)
                ),
                "timing_semantics": (
                    "trajectory_cumulative_snapshot"
                    if iteration_sweep
                    and sampler in (denoiser_admm_samplers | pnpdm_samplers)
                    else "independent_run_per_iteration_count"
                    if iteration_sweep and sampler in flow_solver_samplers
                    else "single_run"
                ),
                "iteration_schedule_depends_on_total_iterations": (
                    sampler in flow_solver_samplers
                ),
                "denoiser_nfe_per_iteration": (
                    1
                    if sampler in denoiser_admm_samplers
                    else None
                    if sampler in pnpdm_samplers
                    else pnp_flow_latent_samples
                    if sampler == "pnp-flow"
                    else flow_admm_k_avg
                    if sampler == "flow-admm"
                    and flow_admm_k_avg_schedule == "constant"
                    else None
                ),
                "pnp_rho": pnp_rho if sampler in denoiser_admm_samplers else None,
                "pnp_lambda": (
                    pnp_lambda if sampler in denoiser_admm_samplers else None
                ),
                "pnp_adaptive_rho": (
                    adaptive_rho if sampler in denoiser_admm_samplers else None
                ),
                "pnp_momentum": (
                    pnp_momentum if sampler in denoiser_admm_samplers else None
                ),
                "pnp_denoiser_relaxation": (
                    pnp_denoiser_relaxation
                    if sampler in denoiser_admm_samplers
                    else None
                ),
                "pnp_dual_damping": (
                    pnp_dual_damping
                    if sampler in denoiser_admm_samplers
                    else None
                ),
                "pnp_final_data_step": (
                    pnp_final_data_step
                    if sampler in denoiser_admm_samplers
                    else None
                ),
                "pnp_rho_candidate_factors": (
                    list(pnp_rho_candidate_factors)
                    if sampler in denoiser_admm_samplers
                    else None
                ),
                "pnp_rho_min": (
                    pnp_rho_min if sampler in denoiser_admm_samplers else None
                ),
                "pnp_rho_max": (
                    pnp_rho_max if sampler in denoiser_admm_samplers else None
                ),
                "pnp_whiteness_weight": (
                    pnp_whiteness_weight
                    if sampler in denoiser_admm_samplers
                    else None
                ),
                "pnp_dm": (
                    {
                        "rho_start": pnpdm_rho_start,
                        "rho_min": pnpdm_rho_min,
                        "rho_decay": pnpdm_rho_decay,
                        "prior_steps": pnpdm_prior_steps,
                        "schedule_power": pnpdm_schedule_power,
                        "prior_solver": pnpdm_prior_solver,
                        "initialization": pnpdm_initialization,
                        "estimate_mode": pnpdm_estimate_mode,
                        "burn_in": pnpdm_burn_in,
                        "thin": pnpdm_thin,
                        "first_retained_iteration": pnpdm_burn_in + pnpdm_thin,
                        "retained_states_start_at_rho_min": (
                            pnpdm_rho_decay < 1.0
                            and pnpdm_rho_start
                            * (
                                pnpdm_rho_decay
                                ** (pnpdm_burn_in + pnpdm_thin - 1)
                            )
                            <= pnpdm_rho_min * (1.0 + 1.0e-12)
                        ),
                        "seed": pnpdm_seed,
                        "coupling_semantics": "real_imag_component_std",
                        "likelihood_step": "exact_complex_gaussian_sample",
                        "prior_step": "partial_reverse_ve_diffusion",
                        "start_level_rule": "first_grid_sigma_le_rho",
                        "clean_boundary": 0.0,
                    }
                    if sampler in pnpdm_samplers
                    else None
                ),
                "flow_initialization": (
                    flow_initialization if sampler in flow_solver_samplers else None
                ),
                "flow_latent_std": (
                    flow_latent_std if sampler in flow_solver_samplers else None
                ),
                "flow_latent_seed": (
                    flow_latent_seed if sampler in flow_solver_samplers else None
                ),
                "latent_seed_policy": (
                    "observation_independent_same_seed_reset_per_K"
                    if sampler in flow_solver_samplers
                    else "separate_reproducible_chain_seed"
                    if sampler in pnpdm_samplers
                    else None
                ),
                "pnp_flow": (
                    {
                        "step_scale": pnp_flow_step_scale,
                        "alpha": pnp_flow_alpha,
                        "time_min": pnp_flow_time_min,
                        "time_max": pnp_flow_time_max,
                        "latent_samples": pnp_flow_latent_samples,
                        "time_schedule": "t_n=time_min+n/K*(time_max-time_min)",
                        "operator_normalization": "spectral_lipschitz",
                    }
                    if sampler == "pnp-flow"
                    else None
                ),
                "flow_admm": (
                    {
                        "tau": flow_admm_tau,
                        "time_min": flow_admm_time_min,
                        "time_max": flow_admm_time_max,
                        "gamma": flow_admm_gamma,
                        "k_avg": flow_admm_k_avg,
                        "k_avg_schedule": flow_admm_k_avg_schedule,
                        "k_avg_early": flow_admm_k_avg_early,
                        "k_avg_mid": flow_admm_k_avg_mid,
                        "k_avg_late": flow_admm_k_avg_late,
                        "switch_frac": flow_admm_switch_frac,
                        "switch_frac_2": flow_admm_switch_frac_2,
                        "time_schedule": (
                            "t_k=time_min+((k+1)/K)^gamma*(time_max-time_min)"
                        ),
                    }
                    if sampler == "flow-admm"
                    else None
                ),
            },
            "config": to_serializable(config),
            "rows": rows,
        }
        (output_dir / "results.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Evaluation results: {output_dir}")
    runtime.end_training()


if __name__ == "__main__":
    main()
