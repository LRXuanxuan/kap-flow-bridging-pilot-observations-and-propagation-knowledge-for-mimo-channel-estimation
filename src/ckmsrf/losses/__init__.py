"""Loss public API."""

from .consistency import ConsistencyMatchingLoss, ConsistencyObjective, pseudo_huber_loss
from .diffusion import DiffusionDenoisingObjective
from .flow_matching import FlowMatchingObjective
from .straight_flow import (
    RobustStraightFlowMatchingObjective,
    StraightFlowMatchingObjective,
)
from .gm import (
    gm_effective_variance,
    gm_global_stats,
    gm_logprob,
    gm_nll_loss,
    gm_second_moment,
    gm_tau2_from_x0_gm,
    gm_to_mean,
    normalize_gm_logweights,
    reverse_transition_additive,
    sample_gm,
    u_to_x0_gm,
    validate_gm,
)
from .gmflow import GMFlowMatchingLoss, GMFlowObjective, extract_condition

__all__ = [
    "ConsistencyMatchingLoss",
    "ConsistencyObjective",
    "DiffusionDenoisingObjective",
    "FlowMatchingObjective",
    "RobustStraightFlowMatchingObjective",
    "StraightFlowMatchingObjective",
    "GMFlowMatchingLoss",
    "GMFlowObjective",
    "extract_condition",
    "gm_effective_variance",
    "gm_global_stats",
    "gm_logprob",
    "gm_nll_loss",
    "gm_second_moment",
    "gm_tau2_from_x0_gm",
    "gm_to_mean",
    "normalize_gm_logweights",
    "pseudo_huber_loss",
    "reverse_transition_additive",
    "sample_gm",
    "u_to_x0_gm",
    "validate_gm",
]
