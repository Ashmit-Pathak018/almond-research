"""
temporal_retriever.py
---------------------
Phase 3 — Temporal and ordering retrieval.

Handles queries where the answer depends on time, sequence, or chronology:
  "Which device did I get first?"
  "When did I buy my phone?"
  "What happened between January and March?"
  "How long after buying my laptop did I buy my phone?"

The critical design choice here is that this retriever does NOT start with
semantic similarity. It starts with the entity registry and the timeline
index, then fetches the parent memories for context.

Why this matters
----------------
"Which device did I get first?" typed into a vector store will retrieve
memories that are semantically similar to that question — mentions of
devices, purchasing, etc. — but ordered by similarity, not by when things
happened. A power bank memory that mentions both Samsung and Dell will rank
highly even though it has nothing to do with the answer.

This retriever routes around that entirely.

Retrieval flow
--------------
1. Extract entity mentions from the query (via EntityRegistry lookup)
2. Query the timeline index for those entities
3. Sort results chronologically
4. Optionally answer the ordering question directly (compare_order)
5. Fetch the parent memory texts for those events
6. Fall back to semantic search if timeline produces nothing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

from core.memory_pipeline_v2.query_analyzer import QueryIntent, IntentType
from core.memory_pipeline_v2.timeline_index import TimelineIndex, TimelineEvent, OrderingResult
from core.memory_pipeline_v2.entity_extractor import EntityRegistry, Entity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight result types
# (The full ranking engine receives these and applies final scoring)
# ---------------------------------------------------------------------------

@dataclass
class RetrievedMemory:
    memory_id:    str
    text:         str
    score:        float          # preliminary relevance score before ranking_engine
    source:       str            # "timeline" | "semantic_fallback"
    timeline_events: list[TimelineEvent] = field(default_factory=list)
    ordering_result: Optional[OrderingResult] = None


@dataclass
class TemporalRetrievalResult:
    """
    Complete result from the temporal retriever, handed to the ranking engine.
    """
    memories:          list[RetrievedMemory]
    ordering_result:   Optional[OrderingResult]   # set for ordering queries
    used_fallback:     bool                        # True if semantic fallback ran
    query_intent:      QueryIntent
    entity_ids_used:   list[str]

    @property
    def found_answer(self) -> bool:
        return bool(self.memories) or (
            self.ordering_result is not None and not self.ordering_result.inconclusive
        )


# ---------------------------------------------------------------------------
# Memory store protocol — plugs into Chroma or any vector store
# ---------------------------------------------------------------------------

class MemoryStore(Protocol):
    """
    Minimal interface the temporal retriever needs to fetch memory text.
    Implement this for your existing Chroma/SQLite setup.
    """
    def get_by_id(self, memory_id: str) -> Optional[str]:
        """Return the raw text of a memory, or None if not found."""
        ...

    def semantic_search(self, query: str, top_k: int = 5,
                        filters: Optional[dict] = None) -> list[tuple[str, str, float]]:
        """
        Fallback semantic search.
        Returns list of (memory_id, text, score) tuples.
        """
        ...


class NullMemoryStore:
    """
    No-op store for testing without Chroma.
    Returns empty strings for all lookups.
    """
    def __init__(self, memory_map: Optional[dict[str, str]] = None):
        self._map = memory_map or {}

    def get_by_id(self, memory_id: str) -> Optional[str]:
        return self._map.get(memory_id)

    def semantic_search(self, query: str, top_k: int = 5,
                        filters: Optional[dict] = None) -> list[tuple[str, str, float]]:
        return []


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class TemporalRetriever:
    """
    Retrieves memories relevant to temporal and ordering queries.

    Connect to your existing pipeline:
        retriever = TemporalRetriever(
            timeline_index=timeline_index,
            entity_registry=entity_registry,
            memory_store=chroma_adapter,   # your existing vector store
        )
        result = retriever.retrieve(query_intent)
    """

    def __init__(self,
                 timeline_index:   TimelineIndex,
                 entity_registry:  EntityRegistry,
                 memory_store:     MemoryStore,
                 fallback_top_k:   int = 5,
                 min_confidence:   float = 0.30):
        self._timeline  = timeline_index
        self._registry  = entity_registry
        self._store     = memory_store
        self._fallback_k = fallback_top_k
        self._min_conf  = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, intent: QueryIntent) -> TemporalRetrievalResult:
        """
        Main entry point. Routes to the right sub-strategy based on
        what the query is actually asking.
        """
        # Resolve entity names from the query to registry IDs
        entity_ids = self._resolve_entities(intent)

        # --- Ordering query: "which came first?" ---
        if self._is_ordering_query(intent) and len(entity_ids) >= 2:
            return self._retrieve_ordering(intent, entity_ids)

        # --- Timeline query: "when did X happen?", "what happened between..." ---
        if entity_ids:
            return self._retrieve_for_entities(intent, entity_ids)

        # --- No entities found: fall back to semantic search ---
        logger.debug("temporal_retriever: no entities resolved, using semantic fallback")
        return self._semantic_fallback(intent, entity_ids=[])

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_entities(self, intent: QueryIntent) -> list[str]:
        """
        Resolve entity names from the query intent to registry entity IDs.
        Tries entities_mentioned first, then comparison_targets.
        """
        names_to_try = list(dict.fromkeys(
            intent.entities_mentioned + intent.comparison_targets
        ))

        resolved: list[str] = []
        for name in names_to_try:
            entity = self._registry.find_by_name(name)
            if entity and entity.id not in resolved:
                resolved.append(entity.id)
                logger.debug("resolved %r → entity %s (%s)", name, entity.id[:8], entity.name)
            else:
                logger.debug("could not resolve entity name: %r", name)

        # DIAGNOSTIC: summary line for eval log parsing.
        # names_to_try=[] -> query_analyzer extracted no entities from the query
        #     (intent.entities_mentioned / comparison_targets both empty)
        # names_to_try non-empty but resolved=[] -> entities were named in the
        #     query but extraction never created registry entries for them
        #     (entity_extractor missed them during replay)
        logger.debug(
            "[ENTITY_RESOLVE] query=%r names_tried=%s resolved_count=%d",
            intent.raw_query[:60], names_to_try, len(resolved),
        )

        return resolved

    # ------------------------------------------------------------------
    # Ordering strategy
    # ------------------------------------------------------------------

    def _is_ordering_query(self, intent: QueryIntent) -> bool:
        ordering_markers = {"first", "before", "after", "earlier", "later",
                            "oldest", "newest", "which came"}
        query_lower = intent.raw_query.lower()
        return any(m in query_lower for m in ordering_markers)

    def _retrieve_ordering(self, intent: QueryIntent,
                           entity_ids: list[str]) -> TemporalRetrievalResult:
        """
        Handle "which came first?" style queries.
        Calls compare_order and fetches the winning memory for context.
        """
        # Use the most likely event type based on query language
        event_type = self._infer_event_type(intent.raw_query)

        # For 2+ entities, compare in pairs (first two are primary comparison)
        entity_a = entity_ids[0]
        entity_b = entity_ids[1]

        ordering = self._timeline.compare_order(entity_a, entity_b, event_type=event_type)

        memories: list[RetrievedMemory] = []

        if not ordering.inconclusive:
            # Fetch the memory text for both events to give context
            for event in [ordering.event_a, ordering.event_b]:
                if event:
                    text = self._store.get_by_id(event.memory_id)
                    if text:
                        # Higher score for the "first" entity's memory
                        is_first = (event == ordering.event_a and ordering.first_id == entity_a) or \
                                   (event == ordering.event_b and ordering.first_id == entity_b)
                        memories.append(RetrievedMemory(
                            memory_id=event.memory_id,
                            text=text,
                            score=0.95 if is_first else 0.80,
                            source="timeline",
                            timeline_events=[event],
                            ordering_result=ordering,
                        ))
        else:
            # Inconclusive — still fetch available memories for context
            for event in [ordering.event_a, ordering.event_b]:
                if event:
                    text = self._store.get_by_id(event.memory_id)
                    if text:
                        memories.append(RetrievedMemory(
                            memory_id=event.memory_id,
                            text=text,
                            score=0.60,
                            source="timeline",
                            timeline_events=[event],
                        ))
            logger.debug("compare_order inconclusive: %s", ordering.reason)

        # If no memories found via timeline, try semantic fallback
        if not memories:
            return self._semantic_fallback(intent, entity_ids)

        return TemporalRetrievalResult(
            memories=memories,
            ordering_result=ordering,
            used_fallback=False,
            query_intent=intent,
            entity_ids_used=entity_ids,
        )

    # ------------------------------------------------------------------
    # Entity-based timeline strategy
    # ------------------------------------------------------------------

    def _retrieve_for_entities(self, intent: QueryIntent,
                                entity_ids: list[str]) -> TemporalRetrievalResult:
        """
        Retrieve all timeline events for the resolved entities,
        then fetch their parent memories.
        """
        event_type = self._infer_event_type(intent.raw_query)

        events = self._timeline.get_events_for_entities(
            entity_ids,
            event_type=event_type if event_type else None,
        )

        if not events:
            # No events found — try without event_type filter
            events = self._timeline.get_events_for_entities(entity_ids)

        if not events:
            logger.debug("No timeline events for entities %s, falling back", entity_ids)
            return self._semantic_fallback(intent, entity_ids)

        # Group events by memory_id and fetch texts
        memory_event_map: dict[str, list[TimelineEvent]] = {}
        for event in events:
            memory_event_map.setdefault(event.memory_id, []).append(event)

        memories = []
        for memory_id, mem_events in memory_event_map.items():
            text = self._store.get_by_id(memory_id)
            if not text:
                continue
            # Score based on temporal confidence and recency of events
            avg_confidence = sum(e.temporal_confidence for e in mem_events) / len(mem_events)
            memories.append(RetrievedMemory(
                memory_id=memory_id,
                text=text,
                score=avg_confidence,
                source="timeline",
                timeline_events=mem_events,
            ))

        if not memories:
            return self._semantic_fallback(intent, entity_ids)

        # Sort by score descending
        memories.sort(key=lambda m: m.score, reverse=True)

        return TemporalRetrievalResult(
            memories=memories,
            ordering_result=None,
            used_fallback=False,
            query_intent=intent,
            entity_ids_used=entity_ids,
        )

    # ------------------------------------------------------------------
    # Semantic fallback
    # ------------------------------------------------------------------

    def _semantic_fallback(self, intent: QueryIntent,
                            entity_ids: list[str]) -> TemporalRetrievalResult:
        """
        Last resort: run a semantic search with the raw query.
        Only used when the timeline produces nothing useful.
        """
        logger.debug("temporal_retriever: semantic fallback for %r", intent.raw_query)

        results = self._store.semantic_search(
            query=intent.raw_query,
            top_k=self._fallback_k,
            filters={"memory_type": ["EVENT", "USER_FACT"]},
        )

        memories = [
            RetrievedMemory(memory_id=mid, text=text, score=score, source="semantic_fallback")
            for mid, text, score in results
        ]

        return TemporalRetrievalResult(
            memories=memories,
            ordering_result=None,
            used_fallback=True,
            query_intent=intent,
            entity_ids_used=entity_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_event_type(self, query: str) -> Optional[str]:
        """
        Guess the most relevant event_type from the query language.
        This narrows the timeline search to the right kind of event.
        """
        import re
        lower = query.lower()
        # Use word-boundary regex so "buy/bought/buying" all match
        if re.search(r'\b(bought|buy|purchase[d]?|order[ed]?|pick[ed]? up|got|get|receive[d]?)\b', lower):
            return "purchased"
        if re.search(r'\b(attend[ed]?|went to|visit[ed]?)\b', lower):
            return "attended"
        if re.search(r'\b(start[ed]?|began|begin|launch[ed]?)\b', lower):
            return "started"
        if re.search(r'\b(complet[ed]?|finish[ed]?|done|wrap[ped]? up)\b', lower):
            return "completed"
        if re.search(r'\b(mov[ed]?|relocat[ed]?)\b', lower):
            return "moved_to"
        if re.search(r'\b(first device|first laptop|first phone)\b', lower):
            return "purchased"
        return None