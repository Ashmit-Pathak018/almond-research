"""
Project Almond V3 — Typed API Request and Response Schemas
Public contracts for the /v3 service boundary.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Memory Schemas
# ---------------------------------------------------------------------------

class CreateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Memory text content")
    namespace: str = Field(default="default", description="Tenant or namespace ID")
    event_time: Optional[float] = Field(default=None, description="Event timestamp in epoch seconds")
    tag: str = Field(default="EPISODIC", description="Memory tag (EPISODIC, PROJECT_FACT, USER_PROFILE, TASK, CORE_RULE)")
    importance_score: float = Field(default=5.0, ge=0.0, le=10.0, description="Base importance (0.0 to 10.0)")
    keywords: List[str] = Field(default_factory=list, description="Keywords for indexing")
    summary: Optional[str] = Field(default=None, description="Optional distillation summary")
    sync: bool = Field(default=False, description="Whether to execute synchronously (eval/benchmark) or queue asynchronously")


class CreateMemoryResponse(BaseModel):
    memory_id: str
    job_id: Optional[str] = None
    status: str
    namespace: str


class MemoryResponse(BaseModel):
    id: str
    namespace_id: str
    content: str
    summary: Optional[str] = None
    tag: str
    tier: str
    state: str
    importance_score: float
    p_eff: float
    keywords: List[str] = Field(default_factory=list)
    event_time: Optional[float] = None
    created_at: float
    updated_at: float
    last_accessed_at: float
    access_count: int


class DeleteMemoryResponse(BaseModel):
    deleted: bool
    memory_id: str


# ---------------------------------------------------------------------------
# Query & Retrieval Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Query text")
    namespace: str = Field(default="default", description="Tenant or namespace ID")
    reference_time: Optional[float] = Field(default=None, description="Perspective timestamp in epoch seconds")
    top_k: int = Field(default=10, ge=1, le=100, description="Max memories to rank and retrieve")


class CandidateResponse(BaseModel):
    memory_id: str
    content: Optional[str] = None
    tag: Optional[str] = None
    tier: Optional[str] = None
    final_score: float
    rank: int
    source_channels: List[str] = Field(default_factory=list)
    event_time: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    namespace: str
    context_text: str
    is_abstention: bool
    confidence: float
    memory_ids: List[str] = Field(default_factory=list)
    candidates: List[CandidateResponse] = Field(default_factory=list)
    trace_id: Optional[str] = None


class QueryTraceResponse(BaseModel):
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


# ---------------------------------------------------------------------------
# Timeline Schemas
# ---------------------------------------------------------------------------

class TimelineEventResponse(BaseModel):
    event_id: str
    namespace_id: str
    title: str
    event_type: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: float = 1.0
    linked_memory_ids: List[str] = Field(default_factory=list)


class TimelineResponse(BaseModel):
    namespace: str
    events: List[TimelineEventResponse]
    total_count: int


# ---------------------------------------------------------------------------
# Entity Schemas
# ---------------------------------------------------------------------------

class EntityResponse(BaseModel):
    entity_id: str
    namespace_id: str
    canonical_name: str
    entity_type: str
    summary: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    linked_memory_ids: List[str] = Field(default_factory=list)
    created_at: float
    updated_at: float


# ---------------------------------------------------------------------------
# Evaluation Schemas
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    evaluation_type: str = Field(default="golden", description="Type of evaluation: golden, etc.")
    case_id: Optional[str] = Field(default=None, description="Specific case ID to run, or None for all")


class EvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    status: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str = "v3"
    components: Dict[str, ComponentHealth]
    clock: Dict[str, Any]
