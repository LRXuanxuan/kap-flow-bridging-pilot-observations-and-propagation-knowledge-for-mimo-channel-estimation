"""Factories shared by the training, evaluation and data-preparation CLIs."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .data import (
    CKMSRFDataset,
    DatasetIndex,
    NormalizationStats,
    ObservationConfig,
    build_shard_index,
    compute_normalization,
)
from .losses import (
    ConsistencyObjective,
    DiffusionDenoisingObjective,
    FlowMatchingObjective,
    GMFlowObjective,
    RobustStraightFlowMatchingObjective,
    StraightFlowMatchingObjective,
)
from .models import (
    ConditionalConsistencyUNet,
    ConditionalFlowMatchingUNet,
    ConditionalGMUNet,
    ConditionalStraightFlowUNet,
    GMUNet,
    RobustConditionalStraightFlowUNet,
)


def parse_scene_ids(values: Iterable[int | str] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    result: list[int] = []
    for value in values:
        if isinstance(value, int):
            scene_id = value
        else:
            match = re.search(r"(\d+)$", str(value).strip())
            if match is None:
                raise ValueError(f"Cannot parse scene id from {value!r}")
            scene_id = int(match.group(1))
        if scene_id not in result:
            result.append(scene_id)
    return tuple(result)


def _scene_ids_for_split(config: Mapping[str, Any], split: str) -> tuple[int, ...] | None:
    data = config.get("data", {})
    key = {"train": "train_scenes", "val": "validation_scenes", "test": "test_scenes"}[split]
    return parse_scene_ids(data.get(key))


def _index_cache_path(
    config: Mapping[str, Any], split: str, scene_ids: Sequence[int] | None
) -> Path:
    data_root = Path(config["paths"]["data_root"])
    configured_value = str(config.get("data", {}).get("index_file", "index.json"))
    configured = Path(configured_value.format(split=split))
    parent = configured.parent if configured.is_absolute() else data_root / configured.parent
    stem = configured.stem
    suffix = configured.suffix or ".json"
    scene_tag = "all" if not scene_ids else "s" + "-".join(f"{value:02d}" for value in scene_ids)
    backend = str(config.get("data", {}).get("backend", "auto")).replace("_", "-")
    split_stem = stem if split in stem.lower().split("_") else f"{stem}_{split}"
    return parent / f"{split_stem}_{backend}_{scene_tag}{suffix}"


def load_or_build_index(
    config: Mapping[str, Any],
    split: str,
    *,
    refresh: bool = False,
) -> DatasetIndex:
    split = split.lower()
    if split == "validation":
        split = "val"
    scene_ids = _scene_ids_for_split(config, split)
    cache_path = _index_cache_path(config, split, scene_ids)
    data_root = Path(config["paths"]["data_root"])
    if cache_path.is_file() and not refresh:
        index = DatasetIndex.load(cache_path, root=data_root)
        if scene_ids is None or set(index.scene_ids) == set(scene_ids):
            return index
    index = build_shard_index(
        data_root,
        split,
        backend=str(config.get("data", {}).get("backend", "auto")),
        scene_ids=scene_ids,
    )
    index.save(cache_path)
    return index


def normalization_path(config: Mapping[str, Any]) -> Path:
    data_root = Path(config["paths"]["data_root"])
    value = Path(
        str(
            config.get("data", {})
            .get("normalization", {})
            .get("stats_file", "normalization.json")
        )
    )
    return value if value.is_absolute() else data_root / value


def load_or_compute_normalization(
    config: Mapping[str, Any],
    train_index: DatasetIndex,
    *,
    refresh: bool = False,
) -> NormalizationStats | None:
    normalization = config.get("data", {}).get("normalization", {})
    mode = str(normalization.get("mode", "global_rms")).lower()
    if mode in {"none", "off", "false"}:
        return None
    if mode != "global_rms":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    path = normalization_path(config)
    if path.is_file() and not refresh:
        return NormalizationStats.load(path)
    maximum = normalization.get("max_trajectories")
    stats = compute_normalization(
        train_index,
        chunk_trajectories=int(normalization.get("chunk_trajectories", 8)),
        max_trajectories=None if maximum is None else int(maximum),
    )
    stats.save(path)
    return stats


def build_dataset(
    config: Mapping[str, Any],
    split: str,
    *,
    index: DatasetIndex | None = None,
    normalization: NormalizationStats | None = None,
    include_observation: bool | None = None,
) -> CKMSRFDataset:
    canonical = "val" if split.lower() == "validation" else split.lower()
    index = index or load_or_build_index(config, canonical)
    data = config.get("data", {})
    if canonical == "train":
        sampling = str(data.get("sampling", "random_tf"))
    else:
        sampling = str(data.get(f"{canonical}_sampling", data.get("validation_sampling", "fixed")))
    fixed_pairs = data.get(f"{canonical}_fixed_pairs", data.get("fixed_pairs"))
    enabled = bool(data.get("include_observation", False)) if include_observation is None else bool(include_observation)
    observation = ObservationConfig(**dict(data.get("observation", {})))
    return CKMSRFDataset(
        index,
        sampling=sampling,
        fixed_pairs=fixed_pairs,
        normalization=normalization,
        include_observation=enabled,
        observation=observation,
        seed=int(config.get("training", {}).get("seed", 3407)),
        max_open_files=int(data.get("max_open_files", 2)),
        adp_shape=tuple(
            config.get("model", {})
            .get("condition", {})
            .get("adp_shape", (16, 16, 12))
        ),
    )


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = dict(config.get("model", {}))
    model_type = str(model_cfg.pop("type", "gmflow")).lower()
    if model_type == "cm":
        model_type = "consistency"
    condition_cfg = dict(model_cfg.pop("condition", {}))
    condition_enabled = bool(condition_cfg.get("enabled", True))
    use_presence_mask = bool(condition_cfg.get("use_presence_mask", False))
    common_kwargs = {
        "in_channels": int(model_cfg.get("in_channels", 2)),
        "out_channels": int(model_cfg.get("out_channels", 2)),
        "dim": int(model_cfg.get("dim", model_cfg.get("model_channels", 32))),
        "dim_mults": tuple(model_cfg.get("dim_mults", model_cfg.get("channel_mult", (1, 2, 4)))),
        "num_res_blocks": int(model_cfg.get("num_res_blocks", 2)),
        "groups": int(model_cfg.get("groups", 8)),
    }
    if use_presence_mask and model_type not in {
        "straight_flow_matching",
        "straight_flow",
    }:
        raise ValueError(
            "model.condition.use_presence_mask is currently supported only "
            "by model.type=straight_flow_matching"
        )
    if model_type == "consistency":
        if not condition_enabled:
            raise ValueError(
                "The current consistency model requires model.condition.enabled=true"
            )
        consistency = config.get("consistency", {})
        return ConditionalConsistencyUNet(
            **common_kwargs,
            # The inherited constructor creates a temporary GM head which is
            # replaced immediately by the direct two-channel CM head.
            num_components=1,
            condition=condition_cfg,
            sigma_min=float(consistency.get("sigma_min", 0.002)),
            sigma_data=float(consistency.get("sigma_data", 1.0)),
        )
    if model_type in {"flow_matching", "velocity_flow"}:
        if not condition_enabled:
            raise ValueError(
                "The current Flow Matching model requires "
                "model.condition.enabled=true"
            )
        consistency = config.get("consistency", {})
        return ConditionalFlowMatchingUNet(
            **common_kwargs,
            num_components=1,
            condition=condition_cfg,
            sigma_min=float(consistency.get("sigma_min", 0.002)),
            sigma_data=float(consistency.get("sigma_data", 1.0)),
        )
    if model_type in {"straight_flow_matching", "straight_flow"}:
        if not condition_enabled:
            raise ValueError(
                "The current straight Flow Matching model requires "
                "model.condition.enabled=true"
            )
        if use_presence_mask:
            return RobustConditionalStraightFlowUNet(
                **common_kwargs,
                num_components=1,
                condition=condition_cfg,
            )
        return ConditionalStraightFlowUNet(
            **common_kwargs,
            num_components=1,
            condition=condition_cfg,
        )
    if model_type != "gmflow":
        raise ValueError(f"Unsupported model.type: {model_type}")
    kwargs = {
        **common_kwargs,
        "num_components": int(model_cfg.get("num_components", model_cfg.get("gm_num_components", 2))),
        "gm_min_logstd": float(model_cfg.get("gm_min_logstd", -3.5)),
        "gm_max_logstd": float(model_cfg.get("gm_max_logstd", 0.5)),
        "gm_logstd_layers": int(model_cfg.get("gm_logstd_layers", 2)),
    }
    if condition_enabled:
        return ConditionalGMUNet(**kwargs, condition=condition_cfg)
    return GMUNet(**kwargs)


def load_initial_weights(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load raw legacy, Flower metadata, or native CKM-SRF checkpoints."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping, got {type(payload)!r}")
    state: Any
    if "model" in payload:
        state = payload["model"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    if strict:
        result = model.load_state_dict(state, strict=True)
    else:
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current
            and torch.is_tensor(value)
            and tuple(value.shape) == tuple(current[key].shape)
        }
        skipped = sorted(set(state) - set(compatible))
        result = model.load_state_dict(compatible, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys) + skipped
    return list(result.missing_keys), list(result.unexpected_keys)


def build_objective(config: Mapping[str, Any]) -> nn.Module:
    flow = config.get("flow", {})
    loss = config.get("loss", {})
    loss_type = str(loss.get("type", "gmflow")).lower()
    model_type = str(config.get("model", {}).get("type", "gmflow")).lower()
    if loss_type in {
        "straight_flow_matching",
        "straight_velocity_matching",
        "pnp_flow_matching",
    }:
        if model_type not in {"straight_flow_matching", "straight_flow"}:
            raise ValueError(
                "The straight Flow Matching objective requires "
                "model.type=straight_flow_matching"
            )
        straight_flow = config.get("straight_flow_matching", {})
        robust_condition = config.get("robust_condition", {})
        robust_enabled = bool(robust_condition.get("enabled", False))
        if robust_enabled and not bool(
            config.get("model", {}).get("condition", {}).get("use_presence_mask", False)
        ):
            raise ValueError(
                "robust_condition.enabled=true requires "
                "model.condition.use_presence_mask=true"
            )
        objective_type = (
            RobustStraightFlowMatchingObjective
            if robust_enabled
            else StraightFlowMatchingObjective
        )
        return objective_type(
            time_min=float(straight_flow.get("time_min", 0.0)),
            time_max=float(straight_flow.get("time_max", 1.0)),
            latent_std=float(straight_flow.get("latent_std", 1.0)),
            coupling=str(straight_flow.get("coupling", "independent")),
            **(
                {
                    "corruption": robust_condition,
                    "adp_shape": tuple(
                        config.get("model", {})
                        .get("condition", {})
                        .get("adp_shape", (16, 16, 12))
                    ),
                    "apply_in_eval": bool(
                        robust_condition.get("apply_in_eval", False)
                    ),
                }
                if robust_enabled
                else {}
            ),
        )
    if loss_type in {"flow_matching", "velocity_matching", "fm"}:
        if model_type not in {"flow_matching", "velocity_flow"}:
            raise ValueError(
                "The Flow Matching objective requires model.type=flow_matching"
            )
        consistency = config.get("consistency", {})
        flow_matching = config.get("flow_matching", {})
        return FlowMatchingObjective(
            sigma_min=float(consistency.get("sigma_min", 0.002)),
            sigma_max=float(consistency.get("sigma_max", 2.0)),
            p_mean=float(flow_matching.get("p_mean", -1.2)),
            p_std=float(flow_matching.get("p_std", 1.2)),
        )
    if loss_type in {"diffusion", "denoising_diffusion", "edm"}:
        if model_type not in {"consistency", "cm"}:
            raise ValueError(
                "The diffusion denoising objective requires the direct "
                "noise-conditioned consistency backbone"
            )
        consistency = config.get("consistency", {})
        diffusion = config.get("diffusion", {})
        return DiffusionDenoisingObjective(
            sigma_min=float(consistency.get("sigma_min", 0.002)),
            sigma_max=float(consistency.get("sigma_max", 2.0)),
            sigma_data=float(consistency.get("sigma_data", 1.0)),
            p_mean=float(diffusion.get("p_mean", -1.2)),
            p_std=float(diffusion.get("p_std", 1.2)),
            weighting=str(diffusion.get("weighting", "edm")),
        )
    if model_type in {"consistency", "cm"}:
        consistency = config.get("consistency", {})
        return ConsistencyObjective(
            sigma_min=float(consistency.get("sigma_min", 0.002)),
            sigma_max=float(consistency.get("sigma_max", 2.0)),
            rho=float(consistency.get("rho", 7.0)),
            num_scales=int(consistency.get("num_scales", 40)),
            consistency_weight=float(consistency.get("consistency_weight", 1.0)),
            endpoint_weight=float(consistency.get("endpoint_weight", 1.0)),
            reconstruction_weight=float(loss.get("reconstruction_weight", 0.0)),
            data_consistency_weight=float(loss.get("physics_weight", 0.0)),
            pseudo_huber_delta=float(consistency.get("pseudo_huber_c", 0.03)),
        )
    if model_type in {"flow_matching", "velocity_flow"}:
        raise ValueError(
            "model.type=flow_matching requires loss.type=flow_matching"
        )
    if model_type in {"straight_flow_matching", "straight_flow"}:
        raise ValueError(
            "model.type=straight_flow_matching requires "
            "loss.type=straight_flow_matching"
        )
    snr = flow.get("snr_db", {})
    return GMFlowObjective(
        snr_min_db=float(snr.get("min", flow.get("snr_min_db", -10.0))),
        snr_max_db=float(snr.get("max", flow.get("snr_max_db", 30.0))),
        snr_step_db=float(snr.get("step", flow.get("snr_step_db", 5.0))),
        time_mu=float(flow.get("time_mu", -0.4)),
        time_sigma=float(flow.get("time_sigma", 1.0)),
        transition_ratio=float(flow.get("transition_ratio", 0.5)),
        prior_weight=float(loss.get("gm_nll_weight", 1.0)),
        reconstruction_weight=float(loss.get("reconstruction_weight", 0.0)),
        data_consistency_weight=float(loss.get("physics_weight", 0.0)),
    )


def build_optimizer(config: Mapping[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    cfg = config.get("optimizer", {})
    name = str(cfg.get("name", "adamw")).lower()
    kwargs = {
        "lr": float(cfg.get("learning_rate", cfg.get("lr", 2.0e-4))),
        "weight_decay": float(cfg.get("weight_decay", 1.0e-4)),
        "betas": tuple(float(value) for value in cfg.get("betas", (0.9, 0.99))),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    cfg = config.get("scheduler", {})
    name = str(cfg.get("name", "cosine")).lower()
    if name in {"none", "off", "false"}:
        return None
    warmup = min(int(cfg.get("warmup_steps", 0)), max(total_steps - 1, 0))
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


__all__ = [
    "build_dataset",
    "build_model",
    "build_objective",
    "build_optimizer",
    "build_scheduler",
    "load_initial_weights",
    "load_or_build_index",
    "load_or_compute_normalization",
    "normalization_path",
    "parse_scene_ids",
]
