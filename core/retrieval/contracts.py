"""
Project Almond V3 — Retrieval Contracts
Core data types and interfaces for multi-channel retrieval, fusion, ranking, and tracing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Optional, List, Dict, Any


class RetrievalChannel(str, Enum):
    """Retrieval channel identifier."""
    SEMANTIC = "SEMANTIC"
    LEXICAL = "LEXICAL"
    ENTITY = "ENTITY"
    TEMPORAL = "TEMPORAL"


@dataclass
class RetrievalQuery:
    """
    Standardized request sent into the V3 retrieval subsystem.
    """
    query_text: str
    namespace_id: str = "default"
    reference_time: Optional[float] = None
    top_k: int = 10
    mode: str = "auto"  # auto, semantic, lexical, entity, temporal, hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)

    def effective_reference_time(self) -> float:
        """Returns the query's explicit reference time, or active clock's reference time."""
        if self.reference_time is not None and self.reference_time > 0:
            return self.reference_time
        from core.clock import get_clock
        return get_clock().reference_time()


@dataclass
class RetrievalCandidate:
    """
    A single memory candidate produced by one or more retrieval channels.
    Captures multi-channel evidence, raw and normalized scores, and lifecycle priority.
    """
    memory_id: str
    source_channels: List[RetrievalChannel] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    
    # Provenance and metadata
    content: Optional[str] = None
    tag: Optional[str] = None
    tier: Optional[str] = None
    event_time: Optional[float] = None
    last_accessed_at: Optional[float] = None
    importance_score: float = 5.0
    lifecycle_priority: float = 0.0  # P_eff = I_base * freshness
    freshness: float = 1.0
    
    # Entity & Temporal specific evidence
    entity_matches: List[str] = field(default_factory=list)
    entity_confidence: float = 0.0
    temporal_metadata: Dict[str, Any] = field(default_factory=dict)
    temporal_fit: float = 0.0
    
    # Post-ranking score
    final_score: float = 0.0
    rank: int = 0
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def add_channel_result(
        self,
        channel: RetrievalChannel,
        raw_score: float,
        normalized_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Merge a channel's score and evidence into this candidate."""
        if channel not in self.source_channels:
            self.source_channels.append(channel)
        ch_key = channel.value
        self.raw_scores[ch_key] = raw_score
        self.normalized_scores[ch_key] = normalized_score
        if metadata:
            if channel == RetrievalChannel.ENTITY:
                if "matched_alias" in metadata:
                    self.entity_matches.append(metadata["matched_alias"])
                if "confidence" in metadata:
                    self.entity_confidence = max(self.entity_confidence, metadata["confidence"])
            elif channel == RetrievalChannel.TEMPORAL:
                self.temporal_metadata.update(metadata)
                if "temporal_fit" in metadata:
                    self.temporal_fit = max(self.temporal_fit, metadata["temporal_fit"])


@dataclass
class RetrievalTrace:
    """
    Complete, inspectable decision trace for a single retrieval cycle.
    Provides total transparency into query routing, channel candidates, fusion,
    constraint reasoning, rejections, and ranking scores.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    namespace_id: str = "default"
    reference_time: float = 0.0
    
    # Step 1: Routing
    detected_intent: str = "HYBRID"
    intent_confidence: float = 1.0
    channels_run: List[str] = field(default_factory=list)
    
    # Step 2: Channel Generation
    candidates_per_channel: Dict[str, int] = field(default_factory=dict)
    raw_channel_hits: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    
    # Step 3: Fusion & Constraints
    fusion_merged_count: int = 0
    temporal_constraints_detected: Dict[str, Any] = field(default_factory=dict)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Step 4: Ranking
    ranking_weights_used: Dict[str, float] = field(default_factory=dict)
    ranked_candidates: List[Dict[str, Any]] = field(default_factory=list)
    final_context_ids: List[str] = field(default_factory=list)
    
    # Timing
    duration_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "namespace_id": self.namespace_id,
            "reference_time": self.reference_time,
            "detected_intent": self.detected_intent,
            "intent_confidence": self.intent_confidence,
            "channels_run": self.channels_run,
            "candidates_per_channel": self.candidates_per_channel,
            "fusion_merged_count": self.fusion_merged_count,
            "temporal_constraints_detected": self.temporal_constraints_detected,
            "rejected_count": len(self.rejected_candidates),
            "ranked_count": len(self.ranked_candidates),
            "final_context_ids": self.final_context_ids,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
        }


@dataclass
class RetrievalResult:
    """
    Final output of the retrieval pipeline returned to the caller / agent.
    """
    candidates: List[RetrievalCandidate]
    trace: RetrievalTrace
    context_text: str = ""
    is_abstention: bool = False
    confidence: float = 0.0

    @property
    def memory_ids(self) -> List[str]:
        return [c.memory_id for c in self.candidates]
