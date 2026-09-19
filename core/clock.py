"""
Project Almond V3 — Clock Subsystem
Defines ClockProvider, SystemClock, and VirtualClock for bi-temporal and reference time management.
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Protocol, Optional


class ClockProvider(Protocol):
    """
    Protocol defining the clock interface for Almond V3.
    Decouples real physical wall-clock time from simulated evaluation time.
    """
    def now(self) -> float:
        """Returns current epoch timestamp in seconds."""
        ...

    def reference_time(self) -> float:
        """
        Returns the reference perspective timestamp in seconds.
        Decay, temporal reasoning, and freshness are evaluated relative to this timestamp.
        """
        ...

    def now_dt(self) -> datetime:
        """Returns current time as timezone-aware UTC datetime."""
        ...

    def reference_dt(self) -> datetime:
        """Returns reference time as timezone-aware UTC datetime."""
        ...


class SystemClock:
    """
    Production wall-clock provider.
    now() and reference_time() track physical system time by default.
    """
    def __init__(self, pinned_reference_time: Optional[float] = None):
        self._pinned_ref_time = pinned_reference_time

    def now(self) -> float:
        return time.time()

    def reference_time(self) -> float:
        if self._pinned_ref_time is not None:
            return self._pinned_ref_time
        return time.time()

    def set_reference_time(self, timestamp: Optional[float]) -> None:
        self._pinned_ref_time = timestamp

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.now(), tz=timezone.utc)

    def reference_dt(self) -> datetime:
        return datetime.fromtimestamp(self.reference_time(), tz=timezone.utc)

    def __repr__(self) -> str:
        return f"<SystemClock now={self.now():.2f} ref={self.reference_time():.2f}>"


class VirtualClock:
    """
    Simulated clock provider for evaluations, historical replays, and unit testing.
    current_time and reference_time can be independently controlled and advanced.
    """
    def __init__(
        self,
        current_time: float,
        reference_time: Optional[float] = None,
        mode: str = "simulation"
    ):
        self._current_time: float = float(current_time)
        self._reference_time: float = float(reference_time) if reference_time is not None else float(current_time)
        self.mode: str = mode

    def now(self) -> float:
        return self._current_time

    def reference_time(self) -> float:
        return self._reference_time

    def set_current_time(self, timestamp: float) -> None:
        self._current_time = float(timestamp)

    def set_reference_time(self, timestamp: float) -> None:
        self._reference_time = float(timestamp)

    def advance(self, seconds: float) -> None:
        """Advances current_time by seconds. If reference_time was equal to current_time, advances it too."""
        sync_ref = (self._reference_time == self._current_time)
        self._current_time += seconds
        if sync_ref:
            self._reference_time += seconds

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self._current_time, tz=timezone.utc)

    def reference_dt(self) -> datetime:
        return datetime.fromtimestamp(self._reference_time, tz=timezone.utc)

    def __repr__(self) -> str:
        return (
            f"<VirtualClock mode='{self.mode}' "
            f"now={self._current_time:.2f} ({self.now_dt().isoformat()}) "
            f"ref={self._reference_time:.2f} ({self.reference_dt().isoformat()})>"
        )


# Global default clock instance (SystemClock by default, replacable in testing/eval)
_DEFAULT_CLOCK: ClockProvider = SystemClock()


def get_clock() -> ClockProvider:
    """Get the active global ClockProvider."""
    return _DEFAULT_CLOCK


def set_clock(clock: ClockProvider) -> None:
    """Set the active global ClockProvider."""
    global _DEFAULT_CLOCK
    _DEFAULT_CLOCK = clock
