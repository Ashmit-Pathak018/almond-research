"""
comparison_retriever.py
-----------------------
Phase 4 — Multi-entity retrieval for comparison queries.

Handles queries like:
  "Samsung Galaxy S22 or Dell XPS 13 — which do I prefer?"
  "Compare my two laptops."
  "Which phone is better for me?"

The key insight: comparison queries need facts about BOTH (or all) entities
independently retrieved and then presented together. Running a single semantic
search for "Samsung vs Dell" will retrieve memories that happen to mention
both in the same sentence, which is rarely the most useful memory for each.

Retrieval flow
--------------
1. Resolve comparison targets from QueryIntent to entity IDs
2. For each entity independently:
   a. Fetch all memories referencing that entity (via EntityRegistry)
   b. Score each memory by how central the entity is to that memory
3. Merge results, deduplicate
4. If either entity has no memories → semantic fallback for that entity only
5. Return grouped results so the ranking engine knows which memories belong
   to which comparison target
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from core.memory_pipeline_v2.query_analyzer import QueryIntent
from core.memory_pipeline_v2.entity_extractor import EntityRegistry, Entity
from core.memory_pipeline_v2.temporal_retriever import RetrievedMemory   # reuse the same type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ComparisonGroup:
    """Memories relevant to one side of a comparison."""
    entity_id:   str
    entity_name: str
    memories:    list[RetrievedMemory]

    @property
    def has_memories(self) -> bool:
        return bool(self.memories)

    @property
    def best_score(self) -> float:
        return max((m.score for m in self.memories), default=0.0)


@dataclass
class ComparisonRetrievalResult:
    """
    Full result from the comparison retriever.
    groups:          one ComparisonGroup per comparison target, in query order
    flat_memories:   deduplicated union of all memories, for the ranking engine
    used_fallback:   True if any entity needed semantic fallback
    query_intent:    the original intent
    """
    groups:        list[ComparisonGroup]
    flat_memories: list[RetrievedMemory]
    used_fallback: bool
    query_intent:  QueryIntent

    @property
    def all_entity_ids(self) -> list[str]:
        return [g.entity_id for g in self.groups]

    @property
    def found_both(self) -> bool:
        return all(g.has_memories for g in self.groups)


# ---------------------------------------------------------------------------
# Memory store protocol (same interface as temporal_retriever)
# ---------------------------------------------------------------------------

class MemoryStore(Protocol):
    def get_by_id(self, memory_id: str) -> Optional[str]: ...
    def semantic_search(self, query: str, top_k: int = 5,
                        filters: Optional[dict] = None) -> list[tuple[str, str, float]]: ...
    # Optional — not part of the strict Protocol contract (NullMemoryStore and
    # other lightweight stubs don't need to implement it), but used when
    # present via getattr() to pull created_at for recency-aware selection.
    # def get_block_by_id(self, memory_id: str) -> Optional[MemoryBlock]: ...


class NullMemoryStore:
    def __init__(self, memory_map: Optional[dict[str, str]] = None):
        self._map = memory_map or {}
    def get_by_id(self, mid: str) -> Optional[str]:
        return self._map.get(mid)
    def semantic_search(self, query: str, top_k: int = 5,
                        filters: Optional[dict] = None) -> list[tuple[str, str, float]]:
        return []


# ---------------------------------------------------------------------------
# Comparison Retriever
# ---------------------------------------------------------------------------

class ComparisonRetriever:
    """
    Retrieves memories for each side of a comparison independently.

    Usage
    -----
    retriever = ComparisonRetriever(
        entity_registry=registry,
        memory_store=chroma_adapter,
    )
    result = retriever.retrieve(query_intent)
    # result.groups[0] → memories about entity A
    # result.groups[1] → memories about entity B
    # result.flat_memories → merged for the ranking engine
    """

    def __init__(self,
                 entity_registry: EntityRegistry,
                 memory_store:    MemoryStore,
                 max_per_entity:  int = 8,
                 fallback_top_k:  int = 4):
        self._registry      = entity_registry
        self._store         = memory_store
        self._max_per       = max_per_entity
        self._fallback_k    = fallback_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, intent: QueryIntent) -> ComparisonRetrievalResult:
        """
        Retrieve memories for all comparison targets in the query intent.
        """
        # Resolve target names → entities
        targets = self._resolve_targets(intent)

        if not targets:
            # No entities resolved — fall back to a single semantic search
            return self._full_fallback(intent)

        groups: list[ComparisonGroup] = []
        used_fallback = False
        seen_memory_ids: set[str] = set()

        for entity in targets:
            # Don't pass seen_memory_ids here — a memory that mentions both Samsung
            # and Dell should appear in BOTH groups. Deduplication happens in _merge_flat.
            group, fell_back = self._retrieve_for_entity(entity, intent, set())
            groups.append(group)
            used_fallback = used_fallback or fell_back

        # Flat list for the ranking engine: all memories, deduped
        flat = self._merge_flat(groups)

        return ComparisonRetrievalResult(
            groups=groups,
            flat_memories=flat,
            used_fallback=used_fallback,
            query_intent=intent,
        )

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def _resolve_targets(self, intent: QueryIntent) -> list[Entity]:
        """
        Resolve comparison_targets → Entity objects.
        Falls back to entities_mentioned if comparison_targets is empty.
        """
        names = intent.comparison_targets or intent.entities_mentioned
        entities: list[Entity] = []
        seen_ids: set[str] = set()

        for name in names:
            entity = self._registry.find_by_name(name)
            if entity and entity.id not in seen_ids:
                entities.append(entity)
                seen_ids.add(entity.id)
                logger.debug("comparison: resolved %r → %s", name, entity.name)
            else:
                logger.debug("comparison: could not resolve %r", name)

        return entities

    # ------------------------------------------------------------------
    # Per-entity retrieval
    # ------------------------------------------------------------------

    def _retrieve_for_entity(self,
                              entity: Entity,
                              intent: QueryIntent,
                              already_seen: set[str]
                              ) -> tuple[ComparisonGroup, bool]:
        """
        Retrieve memories for a single entity.
        Returns (ComparisonGroup, used_fallback).

        entity.memory_ids is populated in pure ingestion/append order (see
        entity_extractor.py) with no relevance or recency ordering at all.
        Previously this method took entity.memory_ids[:max_per_entity]
        directly — i.e. the first N memories that happened to mention this
        entity chronologically, regardless of whether those were the turns
        that actually establish when/how the entity was acquired or used.
        For entities mentioned often in passing (e.g. "my mesh network
        system" coming up repeatedly while shopping for an unrelated
        desktop computer), the dedicated "I set up X" turn could easily
        fall outside the first N mentions and never reach the prompt.

        Fix: pull ALL memory_ids for the entity (not just the first N),
        fetch full MemoryBlocks when the store supports it (via
        get_block_by_id), sort candidates by recency (most recent mention
        first) before truncating to max_per_entity. Recency is a reasonable
        proxy here — the LongMemEval benchmark's later-session turns are
        more likely to be the actual answer-bearing statement, since the
        most informative turn about "when did I do X" tends to appear when
        the user states it directly rather than every time the entity is
        mentioned in passing afterward. This is combined with the keyword-
        aware scoring in _score_memory_for_entity below.
        """
        used_fallback = False

        # Pull every memory_id for this entity (no longer truncated up front).
        all_ids = [mid for mid in entity.memory_ids if mid not in already_seen]

        candidates: list[tuple[str, str, Optional[object]]] = []  # (id, text, block_or_None)
        get_block = getattr(self._store, "get_block_by_id", None)

        for memory_id in all_ids:
            block = get_block(memory_id) if get_block else None
            if block is not None:
                text = getattr(block, "content", None)
            else:
                text = self._store.get_by_id(memory_id)
            if not text:
                continue
            candidates.append((memory_id, text, block))

        # Sort by recency (most recent first) when timestamp metadata is
        # available; falls back to original ingestion order otherwise
        # (e.g. NullMemoryStore in tests, or a store without get_block_by_id).
        if any(c[2] is not None for c in candidates):
            candidates.sort(
                key=lambda c: getattr(c[2], "created_at", None) or "",
                reverse=True,
            )

        # Pure recency-sort-then-truncate has its own failure mode: if the
        # turn that actually establishes WHEN/HOW the entity was acquired
        # is the OLDEST mention (common — people often state "I set up X"
        # once, then mention X in passing many times afterward), truncating
        # to the most-recent max_per_entity drops it just as easily as the
        # old ingestion-order truncation did, just in the opposite direction.
        #
        # Fix: partition into keyword-matched turns (see _EVENT_KEYWORDS)
        # and everything else. ALL keyword-matched turns are kept regardless
        # of truncation, since these are disproportionately likely to be the
        # actual answer-bearing content. Remaining slots are filled from the
        # recency-sorted remainder.
        keyword_matched = [c for c in candidates if self._has_event_keyword(c[1])]
        other = [c for c in candidates if not self._has_event_keyword(c[1])]

        remaining_slots = max(0, self._max_per - len(keyword_matched))
        selected = keyword_matched + other[:remaining_slots]
        # Cap total in case keyword matches alone exceed max_per (rare, but
        # don't silently balloon past the configured limit)
        selected = selected[: max(self._max_per, len(keyword_matched))]

        memories: list[RetrievedMemory] = []
        for memory_id, text, _block in selected:
            score = self._score_memory_for_entity(text, entity)
            memories.append(RetrievedMemory(
                memory_id=memory_id,
                text=text,
                score=score,
                source="entity_registry",
            ))

        # If the entity registry didn't have enough → semantic fallback for this entity
        if not memories:
            sem_results = self._store.semantic_search(
                query=f"{entity.name} {intent.raw_query}",
                top_k=self._fallback_k,
                filters={"memory_type": ["USER_FACT", "USER_PREFERENCE", "EVENT"]},
            )
            for mid, text, score in sem_results:
                if mid not in already_seen:
                    memories.append(RetrievedMemory(
                        memory_id=mid, text=text,
                        score=score * 0.85,  # discount: fallback quality is lower
                        source="semantic_fallback",
                    ))
            used_fallback = bool(sem_results)
            logger.debug("comparison: semantic fallback for entity %s → %d results",
                         entity.name, len(memories))

        # Sort by score, cap at max_per_entity
        memories.sort(key=lambda m: m.score, reverse=True)
        memories = memories[: self._max_per]

        return ComparisonGroup(
            entity_id=entity.id,
            entity_name=entity.name,
            memories=memories,
        ), used_fallback

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    _EVENT_KEYWORDS = (
        "set up", "set it up", "got my", "bought", "purchased", "ordered",
        "pre-ordered", "started using", "installed", "upgraded to",
        "switched to", "began", "started working with", "signed up",
        "moved in", "got a new", "picked up",
    )

    def _has_event_keyword(self, text: str) -> bool:
        """True if text contains acquisition/setup phrasing (see _EVENT_KEYWORDS)."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._EVENT_KEYWORDS)

    def _score_memory_for_entity(self, text: str, entity: Entity) -> float:
        """
        Score how central an entity is to a memory text, with a boost for
        event-establishing language (see _EVENT_KEYWORDS above).

        Base score uses name/alias occurrence count normalised by text
        length, same as before. An additional boost is applied when the
        text also contains acquisition/setup phrasing, since those turns
        are disproportionately likely to be what a comparison or temporal
        question is actually asking about - a plain mention-count can't
        distinguish "I set up my smart thermostat last month" from "by the
        way, my smart thermostat has been great" even though only the
        first answers a "when did I set this up" question.
        """
        text_lower  = text.lower()
        text_words  = max(len(text.split()), 1)
        hit_count   = 0

        for name in entity.all_names():
            occurrences = text_lower.count(name.lower())
            hit_count  += occurrences

        # Normalise: 1 hit in a 10-word sentence > 1 hit in a 100-word sentence
        raw_score = hit_count / (text_words ** 0.5)
        score = raw_score * 3.0

        # Event-establishing keyword boost
        if any(kw in text_lower for kw in self._EVENT_KEYWORDS):
            score += 0.25

        # Clamp to [0.40, 0.95] so registry memories always beat fallback (≤0.85)
        return min(0.95, max(0.40, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_flat(self, groups: list[ComparisonGroup]) -> list[RetrievedMemory]:
        """
        Merge all group memories into a flat deduplicated list.
        Memories appearing in multiple groups (e.g. a direct comparison
        memory) get their score boosted.
        """
        score_map:  dict[str, float] = {}
        memory_map: dict[str, RetrievedMemory] = {}

        for group in groups:
            for m in group.memories:
                if m.memory_id in score_map:
                    # Memory referenced by multiple entities → boost score
                    score_map[m.memory_id] = min(0.98, score_map[m.memory_id] + 0.10)
                else:
                    score_map[m.memory_id]  = m.score
                    memory_map[m.memory_id] = m

        # Apply boosted scores
        flat = []
        for mid, mem in memory_map.items():
            mem.score = score_map[mid]
            flat.append(mem)

        flat.sort(key=lambda m: m.score, reverse=True)
        return flat

    def _full_fallback(self, intent: QueryIntent) -> ComparisonRetrievalResult:
        """No entities resolved — run a single semantic search over the raw query."""
        results = self._store.semantic_search(
            query=intent.raw_query,
            top_k=self._fallback_k * 2,
            filters={"memory_type": ["USER_FACT", "USER_PREFERENCE", "EVENT"]},
        )
        flat = [
            RetrievedMemory(memory_id=mid, text=text, score=score,
                            source="semantic_fallback")
            for mid, text, score in results
        ]
        return ComparisonRetrievalResult(
            groups=[],
            flat_memories=flat,
            used_fallback=True,
            query_intent=intent,
        )