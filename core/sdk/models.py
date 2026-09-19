"""
Project Almond V3 — SDK Typed Models
Clean, dataclass-based representations of Almond resources returned by the SDK.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    id: str
    namespace_id: str
    content: str
    tag: str
    tier: str
    state: str
    importance_score: float
    p_eff: float
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    event_time: Optional[float] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    last_accessed_at: float = 0.0
    access_count: int = 0


@dataclass
class MemoryCreateResult:
    memory_id: str
    job_id: Optional[str]
    status: str
    namespace: str


@dataclass
class CandidateRecord:
    memory_id: str
    final_score: float
    rank: int
    content: Optional[str] = None
    tag: Optional[str] = None
    tier: Optional[str] = None
    source_channels: List[str] = field(default_factory=list)
    event_time: Optional[float] = None


@dataclass
class QueryResult:
    query: str
    namespace: str
    context_text: str
    is_abstention: bool
    confidence: float
    memory_ids: List[str] = field(default_factory=list)
    candidates: List[CandidateRecord] = field(default_factory=list)
    trace_id: Optional[str] = None


@dataclass
class QueryTraceRecord:
    trace_id: str
    query: str
    namespace_id: str
    reference_time: float
    detected_intent: str
    intent_confidence: float
    channels_run: List[str]
    candidates_per_channel: Dict[str, int]
    fusion_merged_count: int
    temporal_constraints_detected: Dict[str, Any]
    rejected_count: int
    rejected_candidates: List[Dict[str, Any]]
    ranking_weights_used: Dict[str, float]
    ranked_candidates: List[Dict[str, Any]]
    final_context_ids: List[str]
    duration_ms: float
    created_at: float


@dataclass
class TimelineEventRecord:
    event_id: str
    namespace_id: str
    title: str
    event_type: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 1.0
    linked_memory_ids: List[str] = field(default_factory=list)


@dataclass
class TimelineResult:
    namespace: str
    events: List[TimelineEventRecord]
    total_count: int


@dataclass
class EntityRecord:
    entity_id: str
    namespace_id: str
    canonical_name: str
    entity_type: str
    summary: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    linked_memory_ids: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class HealthResult:
    status: str
    version: str
    components: Dict[str, Dict[str, Any]]
    clock: Dict[str, Any]
