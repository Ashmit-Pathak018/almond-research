"""
Unit tests for V3 MemoryBlock schema, clock injection, and freshness calculations.
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.clock import VirtualClock, set_clock, SystemClock
from core.memory_block import MemoryBlock, MemoryTag, MemoryTier

def test_memory_block_virtual_clock_decay():
    # Freeze time at Jan 1 2024
    t0 = 1704067200.0 # Jan 1 2024 00:00:00 UTC
    vclk = VirtualClock(current_time=t0, reference_time=t0)
    set_clock(vclk)

    # Create memory block at Jan 1 2024
    mb = MemoryBlock(
        content="Testing temporal memory decay",
        tag=MemoryTag.EPISODIC,
        event_time=t0,
        created_at=t0,
        last_accessed_at=t0,
        importance_score=8.0,
        access_count=10 # S = 1.0
    )

    # At t0, delta_t is 0, freshness is 1.0, p_eff is 8.0
    assert mb.delta_t == 0.0
    assert mb.freshness == 1.0
    assert mb.p_eff == 8.0

    # Advance reference_time by 10 days to Jan 11 2024
    t10 = t0 + (10 * 86400.0)
    vclk.set_reference_time(t10)

    # delta_t is now 10 days, lambda = 0.08, S = 1.0
    # exponent = -(0.08 / 1.0) * 10 = -0.8
    expected_freshness = math.exp(-0.8)
    assert abs(mb.delta_t - 10.0) < 1e-4
    assert abs(mb.freshness - expected_freshness) < 1e-4
    assert abs(mb.p_eff - (8.0 * expected_freshness)) < 1e-4

    # Reset clock to default system clock
    set_clock(SystemClock())
    print("PASS: test_memory_block_virtual_clock_decay")

def test_memory_block_v3_fields():
    mb = MemoryBlock(
        content="Multi-tenant profile fact",
        tag=MemoryTag.USER_PROFILE,
        namespace_id="user_alpha",
        importance_score=9.0
    )
    assert mb.namespace_id == "user_alpha"
    assert mb.state == "ACTIVE"
    assert mb.tier == MemoryTier.L2_ACTIVE_RAM
    print("PASS: test_memory_block_v3_fields")

if __name__ == "__main__":
    test_memory_block_virtual_clock_decay()
    test_memory_block_v3_fields()
    print("All MemoryBlock V3 unit tests passed successfully.")
