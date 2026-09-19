"""
Project Almond V3 — Temporal Retriever
Retrieves candidates using structured timeline queries via TimelineStore.
Orders strictly by event_time (start_time), never by ingestion time.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any

from core.knowledge.timeline_store import TimelineStore, EventRecord
from core.retrieval.contracts import RetrievalCandidate, RetrievalChannel, RetrievalQuery

logger = logging.getLogger(__name__)


class TemporalRetriever:
    """
    Retrieves memories linked to timeline events satisfying temporal constraints.
    Chronological sorting uses event_time (start_time).
    """

    def __init__(self, timeline_store: TimelineStore):
        self.timeline_store = timeline_store

    def retrieve_chronological(
        self,
        query: RetrievalQuery,
        limit: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Retrieve events in strict chronological order by event_time."""
        k = limit or query.top_k
        events = self.timeline_store.get_chronological(query.namespace_id, limit=k)
        return self._events_to_candidates(events, query)

    def retrieve_before(
        self,
        query: RetrievalQuery,
        timestamp: float,
        limit: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Retrieve events occurring before timestamp, sorted chronologically."""
        events = self.timeline_store.get_before(query.namespace_id, timestamp)
        candidates = self._events_to_candidates(events, query)
        k = limit or query.top_k
        return candidates[:k]

    def retrieve_after(
        self,
        query: RetrievalQuery,
        timestamp: float,
        limit: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Retrieve events occurring after timestamp, sorted chronologically."""
        events = self.timeline_store.get_after(query.namespace_id, timestamp)
        candidates = self._events_to_candidates(events, query)
        k = limit or query.top_k
        return candidates[:k]

    def retrieve_between(
        self,
        query: RetrievalQuery,
        start_time: float,
        end_time: float,
        limit: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Retrieve events occurring within [start_time, end_time], sorted chronologically."""
        events = self.timeline_store.get_between(query.namespace_id, start_time, end_time)
        candidates = self._events_to_candidates(events, query)
        k = limit or query.top_k
        return candidates[:k]

    def retrieve_overlapping(
        self,
        query: RetrievalQuery,
        start_time: float,
        end_time: float,
        limit: Optional[int] = None
    ) -> List[RetrievalCandidate]:
        """Retrieve interval events that overlap [start_time, end_time]."""
        events = self.timeline_store.get_overlapping(query.namespace_id, start_time, end_time)
        candidates = self._events_to_candidates(events, query)
        k = limit or query.top_k
        return candidates[:k]

    def _events_to_candidates(
        self,
        events: List[EventRecord],
        query: RetrievalQuery
    ) -> List[RetrievalCandidate]:
        """
        Maps matched timeline events back to originating memory candidates.
        Preserves chronological event_time ordering.
        """
        candidates: List[RetrievalCandidate] = []
        seen_memory_ids: set[str] = set()

        for ev in events:
            mem_ids = self.timeline_store.get_memories_for_event(ev.event_id)
            for mid in mem_ids:
                if mid in seen_memory_ids:
                    continue
                seen_memory_ids.add(mid)

                cand = RetrievalCandidate(
                    memory_id=mid,
                    event_time=ev.start_time,
                    temporal_fit=ev.confidence,
                    temporal_metadata={
                        "event_id": ev.event_id,
                        "title": ev.title,
                        "event_type": ev.event_type,
                        "start_time": ev.start_time,
                        "end_time": ev.end_time,
                        "confidence": ev.confidence
                    }
                )
                cand.add_channel_result(
                    channel=RetrievalChannel.TEMPORAL,
                    raw_score=ev.start_time if ev.start_time is not None else 0.0,
                    normalized_score=ev.confidence,
                    metadata=cand.temporal_metadata
                )
                candidates.append(cand)

        return candidates
