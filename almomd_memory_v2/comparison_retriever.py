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

from query_analyzer import QueryIntent
from entity_extractor import EntityRegistry, Entity
from temporal_retriever import RetrievedMemory   # reuse the same type

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
        """
        memories: list[RetrievedMemory] = []
        used_fallback = False

        # Pull all memory IDs from the entity registry
        for memory_id in entity.memory_ids[: self._max_per]:
            if memory_id in already_seen:
                continue
            text = self._store.get_by_id(memory_id)
            if not text:
                continue

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

    def _score_memory_for_entity(self, text: str, entity: Entity) -> float:
        """
        Score how central an entity is to a memory text.
        Uses name/alias occurrence count normalised by text length.
        """
        text_lower  = text.lower()
        text_words  = max(len(text.split()), 1)
        hit_count   = 0

        for name in entity.all_names():
            occurrences = text_lower.count(name.lower())
            hit_count  += occurrences

        # Normalise: 1 hit in a 10-word sentence > 1 hit in a 100-word sentence
        raw_score = hit_count / (text_words ** 0.5)

        # Clamp to [0.40, 0.95] so registry memories always beat fallback (≤0.85)
        return min(0.95, max(0.40, raw_score * 3.0))

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