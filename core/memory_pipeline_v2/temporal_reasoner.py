"""
temporal_reasoner.py
---------------------
Phase 3 — Temporal Reasoning Engine

Computes chronological conclusions (Ordering, Latest, Earliest) from the TimelineIndex 
before the LLM sees the prompt. This isolates reasoning from text generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from core.memory_pipeline_v2.timeline_index import TimelineIndex, TimelineEvent, OrderingResult
from core.memory_pipeline_v2.entity_extractor import EntityRegistry

logger = logging.getLogger(__name__)

QueryType = Literal[
    "ORDERING",
    "LATEST",
    "EARLIEST",
    "BETWEEN",
    "AFTER",
    "BEFORE",
    "DURATION",
]

@dataclass
class ReasoningContext:
    query_type: QueryType
    events: list[TimelineEvent]
    ordering: Optional[OrderingResult]
    confidence: float
    reason: str

    def to_prompt_block(self, registry: EntityRegistry) -> str:
        """
        Formats the reasoning context into a deterministic text block for the LLM.
        """
        if self.query_type == "ORDERING" and self.ordering and not self.ordering.inconclusive:
            # Resolve names for readability
            entity_a = registry.find_by_id(self.ordering.entity_a_id)
            entity_b = registry.find_by_id(self.ordering.entity_b_id)
            name_a = entity_a.name if entity_a else self.ordering.entity_a_id
            name_b = entity_b.name if entity_b else self.ordering.entity_b_id
            
            first_name = name_a if self.ordering.first_id == self.ordering.entity_a_id else name_b
            second_name = name_b if self.ordering.first_id == self.ordering.entity_a_id else name_a

            lines = [
                "=== REASONING ===",
                "Detected Intent: ORDERING",
                "",
                "Known Events:"
            ]
            
            for ev in self.events:
                # Find the primary entity name for this event
                ev_entity_name = "Unknown"
                if ev.entity_ids:
                    ent = registry.find_by_id(ev.entity_ids[0])
                    if ent:
                        ev_entity_name = ent.name
                lines.append(f"- Event: {ev_entity_name} {ev.event_type.lower()} (Date: {ev.earliest.strftime('%Y-%m-%d')})")
            
            lines.extend([
                "",
                "Chronological Order:",
                f"1. {first_name}",
                f"2. {second_name}",
                "",
                f"Confidence: {self.confidence:.2f}",
                f"Reason: {self.reason}",
                "========================="
            ])
            return "\n".join(lines)
            
        return ""


class TemporalReasoner:
    """
    Reasons over the timeline index to answer temporal questions deterministically.
    """
    def __init__(self, timeline_index: TimelineIndex, entity_registry: EntityRegistry):
        self._timeline = timeline_index
        self._registry = entity_registry

    def compute_ordering(self, entity_a_id: str, entity_b_id: str, event_type: str = "") -> Optional[ReasoningContext]:
        """
        Computes the chronological order between two entities.
        """
        ordering = self._timeline.compare_order(entity_a_id, entity_b_id, event_type=event_type)
        
        if ordering.inconclusive:
            return None
            
        events = []
        if ordering.event_a:
            events.append(ordering.event_a)
        if ordering.event_b:
            events.append(ordering.event_b)
            
        # Sort events chronologically for the context
        events.sort(key=lambda e: e.earliest)
        
        # Determine confidence/reason based on data
        # If dates are explicit points in time, confidence is high.
        confidence = ordering.confidence
        reason = "Chronological comparison successful."
        if confidence > 0.8:
            reason = "Explicit temporal bounds established."
        else:
            reason = "Approximate or relative temporal bounds."
            
        return ReasoningContext(
            query_type="ORDERING",
            events=events,
            ordering=ordering,
            confidence=confidence,
            reason=reason
        )
