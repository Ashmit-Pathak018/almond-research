"""
Project Almond V3 — Lifecycle Decay & MemoryPolicy Subsystem
Canonical mathematical formulation of memory freshness, decay anchors, and tier state transitions.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

from core.memory_block import MemoryTag, MemoryTier


class DecayAnchorMode(str, Enum):
    """Specifies which timestamp serves as t_anchor in delta_t calculation."""
    DECAY_FROM_ACCESS = "DECAY_FROM_ACCESS"  # Default for user profiles, rules, durable facts
    DECAY_FROM_EVENT  = "DECAY_FROM_EVENT"   # Default for episodic memories (real-world event date)
    DECAY_FROM_UPDATE = "DECAY_FROM_UPDATE"  # Decays from when assertion was last modified


@dataclass(frozen=True)
class MemoryPolicy:
    """
    First-class policy governing memory lifecycle, decay rate, stability, and tier thresholds.
    """
    tag: MemoryTag
    decay_rate_lambda: float
    stability_divisor: float
    decay_anchor: DecayAnchorMode
    promotion_threshold: float     # P_eff required to promote L3 -> L2 (e.g. 5.0)
    demotion_threshold: float      # P_eff dropping below demotes L2 -> L3 (e.g. 2.5)
    archive_threshold: float       # P_eff dropping below moves L3 -> L4 (e.g. 1.0)
    deletion_threshold: float      # P_eff dropping below marks for deletion (e.g. 0.2)
    retention_period_days: Optional[float] = None
    extraction_enabled: bool = True
    conflict_resolution: str = "FLAG_CONFLICT"


# Canonical Default Memory Policies
DEFAULT_POLICIES: Dict[MemoryTag, MemoryPolicy] = {
    MemoryTag.CORE_RULE: MemoryPolicy(
        tag=MemoryTag.CORE_RULE,
        decay_rate_lambda=0.001,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_ACCESS,
        promotion_threshold=4.0,
        demotion_threshold=1.5,
        archive_threshold=0.5,
        deletion_threshold=0.05,
        retention_period_days=None,
    ),
    MemoryTag.USER_PROFILE: MemoryPolicy(
        tag=MemoryTag.USER_PROFILE,
        decay_rate_lambda=0.002,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_ACCESS,
        promotion_threshold=4.0,
        demotion_threshold=1.8,
        archive_threshold=0.8,
        deletion_threshold=0.1,
        retention_period_days=None,
    ),
    MemoryTag.PROJECT_FACT: MemoryPolicy(
        tag=MemoryTag.PROJECT_FACT,
        decay_rate_lambda=0.01,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_ACCESS,
        promotion_threshold=4.5,
        demotion_threshold=2.2,
        archive_threshold=1.0,
        deletion_threshold=0.2,
        retention_period_days=365.0,
    ),
    MemoryTag.TASK: MemoryPolicy(
        tag=MemoryTag.TASK,
        decay_rate_lambda=0.05,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_UPDATE,
        promotion_threshold=5.0,
        demotion_threshold=2.5,
        archive_threshold=1.0,
        deletion_threshold=0.2,
        retention_period_days=90.0,
    ),
    MemoryTag.EPISODIC: MemoryPolicy(
        tag=MemoryTag.EPISODIC,
        decay_rate_lambda=0.08,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_EVENT,
        promotion_threshold=5.0,
        demotion_threshold=2.5,
        archive_threshold=1.2,
        deletion_threshold=0.3,
        retention_period_days=180.0,
    ),
    MemoryTag.SMALL_TALK: MemoryPolicy(
        tag=MemoryTag.SMALL_TALK,
        decay_rate_lambda=0.20,
        stability_divisor=10.0,
        decay_anchor=DecayAnchorMode.DECAY_FROM_ACCESS,
        promotion_threshold=6.0,
        demotion_threshold=3.5,
        archive_threshold=1.5,
        deletion_threshold=0.5,
        retention_period_days=14.0,
    ),
}


def get_policy_for_tag(tag: MemoryTag) -> MemoryPolicy:
    """Retrieve the canonical policy for a given memory tag."""
    return DEFAULT_POLICIES[tag]


# ---------------------------------------------------------------------------
# Canonical Math Formulation
# ---------------------------------------------------------------------------

def compute_delta_t(reference_time: float, anchor_time: float) -> float:
    """
    Computes elapsed time in days between anchor_time and reference_time.
    Never returns negative delta_t (clamped to 0.0 for future-anchored events).
    """
    seconds_elapsed = max(0.0, reference_time - anchor_time)
    return seconds_elapsed / 86400.0


def compute_stability_factor(access_count: int, divisor: float = 10.0) -> float:
    """
    S = access_count / stability_divisor
    Higher access count -> higher stability -> slower decay.
    """
    count = max(1, access_count)
    div = max(0.1, divisor)
    return count / div


def compute_freshness(
    delta_t_days: float,
    lambda_decay: float,
    stability_factor: float
) -> float:
    """
    Canonical freshness formula:
        freshness = exp( -(lambda / S) * delta_t )

    Returns:
        float in range (0.0, 1.0]
    """
    if delta_t_days <= 0.0:
        return 1.0
    
    s = max(0.01, stability_factor)
    exponent = -(lambda_decay / s) * delta_t_days
    # Guard against underflow
    if exponent < -50.0:
        return 0.0
    return math.exp(exponent)


def compute_effective_priority(importance_score: float, freshness: float) -> float:
    """
    Canonical effective priority formula:
        P_eff = I_base * freshness
    """
    return importance_score * freshness


def resolve_anchor_timestamp(
    anchor_mode: DecayAnchorMode,
    event_time: Optional[float],
    last_accessed_at: float,
    updated_at: float,
    created_at: float
) -> float:
    """
    Resolves the exact epoch seconds timestamp based on the policy's DecayAnchorMode.
    """
    if anchor_mode == DecayAnchorMode.DECAY_FROM_EVENT:
        if event_time is not None and event_time > 0:
            return event_time
        # Fallback if no explicit event_time was extracted
        return created_at

    elif anchor_mode == DecayAnchorMode.DECAY_FROM_UPDATE:
        return updated_at

    elif anchor_mode == DecayAnchorMode.DECAY_FROM_ACCESS:
        return last_accessed_at

    return last_accessed_at


def evaluate_tier_transition(
    p_eff: float,
    current_tier: MemoryTier,
    policy: MemoryPolicy
) -> MemoryTier:
    """
    Determines whether a memory should transition storage tiers based on P_eff.
    """
    # L1 (Core/Hot) is immutable — never demoted automatically
    if current_tier == MemoryTier.L1_HOT_CACHE:
        return MemoryTier.L1_HOT_CACHE

    if current_tier == MemoryTier.L2_ACTIVE_RAM:
        if p_eff < policy.demotion_threshold:
            return MemoryTier.L3_VIRTUAL_SWAP
        return MemoryTier.L2_ACTIVE_RAM

    elif current_tier == MemoryTier.L3_VIRTUAL_SWAP:
        if p_eff >= policy.promotion_threshold:
            return MemoryTier.L2_ACTIVE_RAM
        elif p_eff < policy.archive_threshold:
            return MemoryTier.L4_ARCHIVE
        return MemoryTier.L3_VIRTUAL_SWAP

    elif current_tier == MemoryTier.L4_ARCHIVE:
        if p_eff >= policy.promotion_threshold:
            return MemoryTier.L2_ACTIVE_RAM
        return MemoryTier.L4_ARCHIVE

    return current_tier
