"""
Project Almond V3 — Typed Python SDK Package
Provides the standalone Almond client for external services and agents.
"""

from core.sdk.client import Almond
from core.sdk.models import (
    MemoryRecord,
    MemoryCreateResult,
    CandidateRecord,
    QueryResult,
    QueryTraceRecord,
    TimelineEventRecord,
    TimelineResult,
    EntityRecord,
    HealthResult,
)
from core.sdk.exceptions import (
    AlmondError,
    AlmondAPIError,
    AlmondConnectionError,
    AlmondNotFoundError,
    AlmondValidationError,
    AlmondConflictError,
)

__all__ = [
    "Almond",
    "MemoryRecord",
    "MemoryCreateResult",
    "CandidateRecord",
    "QueryResult",
    "QueryTraceRecord",
    "TimelineEventRecord",
    "TimelineResult",
    "EntityRecord",
    "HealthResult",
    "AlmondError",
    "AlmondAPIError",
    "AlmondConnectionError",
    "AlmondNotFoundError",
    "AlmondValidationError",
    "AlmondConflictError",
]
