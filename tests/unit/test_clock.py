"""
Unit tests for ClockProvider, SystemClock, and VirtualClock.
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.clock import SystemClock, VirtualClock, get_clock, set_clock

def test_system_clock():
    clk = SystemClock()
    t1 = clk.now()
    ref1 = clk.reference_time()
    assert abs(t1 - time.time()) < 0.5
    assert abs(ref1 - time.time()) < 0.5

    # Pinned reference time
    pinned_ts = 1704067200.0 # 2024-01-01
    clk.set_reference_time(pinned_ts)
    assert clk.reference_time() == pinned_ts
    assert abs(clk.now() - time.time()) < 0.5
    print("PASS: test_system_clock")

def test_virtual_clock_advancement():
    start_ts = 1704067200.0 # 2024-01-01 00:00:00 UTC
    vclk = VirtualClock(current_time=start_ts, mode="evaluation")
    assert vclk.now() == start_ts
    assert vclk.reference_time() == start_ts

    # Advance by 86400 seconds (1 day)
    vclk.advance(86400.0)
    assert vclk.now() == start_ts + 86400.0
    assert vclk.reference_time() == start_ts + 86400.0

    # Independent reference time
    future_ref = start_ts + (30 * 86400.0)
    vclk.set_reference_time(future_ref)
    assert vclk.now() == start_ts + 86400.0
    assert vclk.reference_time() == future_ref
    print("PASS: test_virtual_clock_advancement")

def test_global_clock_injection():
    original = get_clock()
    test_vclk = VirtualClock(1000.0, 2000.0)
    set_clock(test_vclk)
    assert get_clock().now() == 1000.0
    assert get_clock().reference_time() == 2000.0
    set_clock(original)
    print("PASS: test_global_clock_injection")

if __name__ == "__main__":
    test_system_clock()
    test_virtual_clock_advancement()
    test_global_clock_injection()
    print("All clock tests passed successfully.")
