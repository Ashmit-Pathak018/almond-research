"""
Project Almond V3 — Temporal / Constraint Reasoner
Resolves structured temporal relationships and filters candidates against temporal boundaries.
Operates deterministically over timeline events and timestamps.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from core.retrieval.contracts import RetrievalCandidate, RetrievalQuery
from core.retrieval.intent_router import IntentAnalysis
from core.knowledge.timeline_store import TimelineStore, EventRecord

logger = logging.getLogger(__name__)


@dataclass
class ConstraintReasoningResult:
    """Structured output of constraint reasoning."""
    accepted_candidates: List[RetrievalCandidate]
    rejected_candidates: List[RetrievalCandidate]
    anchor_event: Optional[EventRecord] = None
    anchor_timestamp: Optional[float] = None
    constraints_applied: Dict[str, Any] = field(default_factory=dict)


class TemporalReasoner:
    """
    Evaluates candidates against explicit temporal constraints (before, after, between, chronological).
    Rejects candidates violating constraints and marks structured temporal fit.
    """

    def __init__(self, timeline_store: TimelineStore):
        self.timeline_store = timeline_store

    def resolve_constraints(
        self,
        candidates: List[RetrievalCandidate],
        query: RetrievalQuery,
        intent: IntentAnalysis,
    ) -> ConstraintReasoningResult:
        """
        Applies temporal constraint logic to fused candidates.
        """
        direction = intent.temporal_direction
        anchor_phrase = intent.temporal_anchor

        # If no temporal constraints apply, accept all candidates
        if not direction and not intent.temporal_markers:
            return ConstraintReasoningResult(
                accepted_candidates=candidates,
                rejected_candidates=[],
                constraints_applied={"mode": "none"}
            )

        anchor_event: Optional[EventRecord] = None
        anchor_ts: Optional[float] = None

        # 1. Resolve anchor event timestamp if direction is before/after
        if direction in ("before", "after") and anchor_phrase:
            anchor_event, anchor_ts = self._find_anchor_timestamp(
                anchor_phrase, query.namespace_id, candidates
            )

        accepted: List[RetrievalCandidate] = []
        rejected: List[RetrievalCandidate] = []
        constraints_log: Dict[str, Any] = {
            "direction": direction,
            "anchor_phrase": anchor_phrase,
            "anchor_timestamp": anchor_ts,
        }

        # 2. Filter candidates against anchor timestamp
        for cand in candidates:
            cand_time = cand.event_time
            if cand_time is None and cand.last_accessed_at is not None:
                cand_time = cand.last_accessed_at

            if anchor_ts is not None and cand_time is not None:
                if direction == "before":
                    # Must have occurred before anchor_ts
                    if cand_time >= anchor_ts:
                        cand.rejected = True
                        cand.rejection_reason = f"event_time ({cand_time}) >= anchor_time ({anchor_ts})"
                        rejected.append(cand)
                        continue
                    else:
                        # Closer to anchor event can have higher temporal fit
                        cand.temporal_fit = max(cand.temporal_fit, 0.95)
                        accepted.append(cand)
                        continue

                elif direction == "after":
                    # Must have occurred after anchor_ts
                    if cand_time <= anchor_ts:
                        cand.rejected = True
                        cand.rejection_reason = f"event_time ({cand_time}) <= anchor_time ({anchor_ts})"
                        rejected.append(cand)
                        continue
                    else:
                        cand.temporal_fit = max(cand.temporal_fit, 0.95)
                        accepted.append(cand)
                        continue

            # If no anchor timestamp was resolved or candidate has no time, accept by default
            accepted.append(cand)

        # 3. If chronological ordering is requested, sort accepted candidates by event_time
        if direction == "chronological" or intent.temporal_markers:
            # Sort by event_time ASC (None at the end)
            accepted.sort(
                key=lambda c: (c.event_time is None, c.event_time if c.event_time is not None else 0.0)
            )

        # For "before" queries: reverse chronological order (closest preceding events first)
        if direction == "before":
            accepted.sort(
                key=lambda c: (c.event_time is None, -(c.event_time if c.event_time is not None else 0.0))
            )

        return ConstraintReasoningResult(
            accepted_candidates=accepted,
            rejected_candidates=rejected,
            anchor_event=anchor_event,
            anchor_timestamp=anchor_ts,
            constraints_applied=constraints_log,
        )

    def _find_anchor_timestamp(
        self,
        anchor_phrase: str,
        namespace_id: str,
        candidates: List[RetrievalCandidate]
    ) -> Tuple[Optional[EventRecord], Optional[float]]:
        """
        Locates the event matching anchor_phrase in TimelineStore or candidates.
        """
        phrase_lower = anchor_phrase.lower()

        # Check timeline store events
        events = self.timeline_store.get_chronological(namespace_id, limit=50)
        for ev in events:
            if phrase_lower in ev.title.lower() or (ev.description and phrase_lower in ev.description.lower()):
                return ev, ev.start_time

        # Check candidate contents for matching phrase
        for cand in candidates:
            if cand.content and phrase_lower in cand.content.lower():
                if cand.event_time is not None:
                    return None, cand.event_time

        # Token overlap fallback
        anchor_tokens = [w for w in phrase_lower.split() if len(w) > 3]
        for ev in events:
            title_lower = ev.title.lower()
            if any(tok in title_lower for tok in anchor_tokens):
                return ev, ev.start_time

        return None, None
