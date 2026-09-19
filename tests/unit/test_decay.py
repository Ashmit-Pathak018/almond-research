"""
Unit tests for core/lifecycle/decay.py
Validates canonical math, stability scaling, decay anchors, and tier transitions.
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.memory_block import MemoryTag, MemoryTier
from core.lifecycle.decay import (
    DecayAnchorMode, MemoryPolicy, DEFAULT_POLICIES,
    compute_delta_t, compute_stability_factor, compute_freshness,
    compute_effective_priority, resolve_anchor_timestamp, evaluate_tier_transition
)

def test_canonical_freshness_math():
    # delta_t = 10 days, lambda = 0.05, S = 1.0 (access_count = 10)
    # exponent = -(0.05 / 1.0) * 10 = -0.5
    # freshness = exp(-0.5) ~= 0.60653
    delta_t = 10.0
    lambda_ = 0.05
    s = 1.0
    f = compute_freshness(delta_t, lambda_, s)
    expected = math.exp(-0.5)
    assert abs(f - expected) < 1e-6, f"Expected {expected}, got {f}"

    # Zero or negative delta_t -> exactly 1.0
    assert compute_freshness(0.0, lambda_, s) == 1.0
    assert compute_freshness(-5.0, lambda_, s) == 1.0
    print("PASS: test_canonical_freshness_math")

def test_effective_priority():
    importance = 8.0
    freshness = 0.5
    p_eff = compute_effective_priority(importance, freshness)
    assert p_eff == 4.0
    print("PASS: test_effective_priority")

def test_decay_anchor_resolution():
    t_created = 1000.0
    t_updated = 1500.0
    t_access = 2000.0
    t_event = 500.0

    # DECAY_FROM_ACCESS -> t_access
    assert resolve_anchor_timestamp(DecayAnchorMode.DECAY_FROM_ACCESS, t_event, t_access, t_updated, t_created) == 2000.0
    # DECAY_FROM_EVENT -> t_event
    assert resolve_anchor_timestamp(DecayAnchorMode.DECAY_FROM_EVENT, t_event, t_access, t_updated, t_created) == 500.0
    # DECAY_FROM_UPDATE -> t_updated
    assert resolve_anchor_timestamp(DecayAnchorMode.DECAY_FROM_UPDATE, t_event, t_access, t_updated, t_created) == 1500.0
    print("PASS: test_decay_anchor_resolution")

def test_tier_transition_logic():
    policy = DEFAULT_POLICIES[MemoryTag.TASK] # demote: 2.5, promote: 5.0, archive: 1.0

    # L1 hot cache never demotes
    assert evaluate_tier_transition(0.1, MemoryTier.L1_HOT_CACHE, policy) == MemoryTier.L1_HOT_CACHE

    # L2 Active with low P_eff demotes to L3 Swap
    assert evaluate_tier_transition(2.0, MemoryTier.L2_ACTIVE_RAM, policy) == MemoryTier.L3_VIRTUAL_SWAP

    # L2 Active with high P_eff remains L2
    assert evaluate_tier_transition(4.5, MemoryTier.L2_ACTIVE_RAM, policy) == MemoryTier.L2_ACTIVE_RAM

    # L3 Swap with very low P_eff moves to L4 Archive
    assert evaluate_tier_transition(0.5, MemoryTier.L3_VIRTUAL_SWAP, policy) == MemoryTier.L4_ARCHIVE

    # L3 Swap with boosted P_eff (re-accessed) promotes back to L2
    assert evaluate_tier_transition(6.0, MemoryTier.L3_VIRTUAL_SWAP, policy) == MemoryTier.L2_ACTIVE_RAM
    print("PASS: test_tier_transition_logic")

if __name__ == "__main__":
    test_canonical_freshness_math()
    test_effective_priority()
    test_decay_anchor_resolution()
    test_tier_transition_logic()
    print("All decay unit tests passed successfully.")
