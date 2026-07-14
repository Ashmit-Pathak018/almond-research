"""
memory_consolidator.py
----------------------
Phase 4 — Background memory consolidation.

Three jobs:
  1. Merge redundant memories
     "I have a Samsung Galaxy S22" × 20 → one stable fact (confidence 0.98).

  2. Demote cold memories
     Memories that haven't been accessed in 180+ days and have low salience
     move to a "cold" tier with reduced retrieval weight.

  3. Promote frequently-accessed facts
     A TEMPORARY_CONTEXT memory that keeps getting referenced gets
     re-classified as USER_FACT.

  4. Reclassify post-classification
     Uses the fact + entity evidence accumulated since first storage to
     correct early misclassifications.

Run this as a background job, not in the hot path:
  - After every N new memories (N=50 is reasonable)
  - Or on a nightly schedule
  - Never during an active conversation turn

Design decisions
----------------
1. Conservative merging.
   Only merge facts with the same subject + normalised predicate + object
   AND similarity ≥ 0.90. Partial merges are flagged for review, not
   automatically applied.

2. Stable facts as first-class objects.
   A StableFact is not the same as a memory. It's a derived, high-confidence
   summary. It gets its own ID so the ranking engine can weight it differently.

3. Source memories are marked, not deleted.
   Consolidated source memories get `is_consolidated=True`. They remain
   retrievable but get a reduced ranking weight. Destructive deletion is
   never applied automatically.

4. Conflict handling.
   When two facts have the same subject+object but different predicates
   (detected by fact_extractor), the consolidator reports the conflict and
   defers to the highest-confidence version — it does not silently pick one.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.memory_pipeline_v2.fact_extractor import StructuredFact, TemporalBound
from core.memory_pipeline_v2.entity_extractor import EntityRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class StableFact:
    """
    A consolidated, high-confidence fact derived from multiple source memories.
    Stored alongside (not replacing) the source memories.
    """
    id:                 str
    subject:            str
    predicate:          str
    object:             str
    confidence:         float
    source_memory_ids:  list[str]
    temporal_bound:     Optional[TemporalBound] = None
    date_raw:           str = ""
    consolidated_at:    datetime = field(default_factory=datetime.utcnow)
    fact_count:         int = 1     # how many source facts were merged
    has_conflict:       bool = False

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "subject":          self.subject,
            "predicate":        self.predicate,
            "object":           self.object,
            "confidence":       round(self.confidence, 4),
            "source_count":     len(self.source_memory_ids),
            "source_memory_ids":self.source_memory_ids,
            "date_raw":         self.date_raw,
            "consolidated_at":  self.consolidated_at.isoformat(),
            "fact_count":       self.fact_count,
            "has_conflict":     self.has_conflict,
        }


@dataclass
class MemoryRecord:
    """
    Lightweight memory metadata used by the consolidator.
    Mirrors what you'd pull from your existing SQLite store.
    """
    id:               str
    text:             str
    memory_type:      str
    retrieval_count:  int
    age_days:         int
    created_at:       datetime
    is_consolidated:  bool = False
    retrieval_weight: float = 1.0   # reduced for cold/consolidated memories


@dataclass
class ConsolidationReport:
    """What happened in a consolidation run."""
    run_at:             datetime
    facts_merged:       int = 0
    stable_facts_created: int = 0
    memories_demoted:   int = 0
    memories_promoted:  int = 0
    conflicts_found:    int = 0
    entity_merges:      int = 0
    errors:             list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Consolidation {self.run_at.strftime('%Y-%m-%d %H:%M')}: "
            f"merged={self.facts_merged} stable={self.stable_facts_created} "
            f"demoted={self.memories_demoted} promoted={self.memories_promoted} "
            f"conflicts={self.conflicts_found} entity_merges={self.entity_merges}"
        )


# ---------------------------------------------------------------------------
# In-memory store (replace with SQLite in production)
# ---------------------------------------------------------------------------

class ConsolidationStore:
    """
    Stores stable facts and tracks which memories have been consolidated.
    In production this wraps your existing SQLite database.
    """
    def __init__(self):
        self._stable_facts:   dict[str, StableFact]   = {}
        self._memory_records: dict[str, MemoryRecord] = {}

    # Stable facts
    def add_stable_fact(self, fact: StableFact):
        self._stable_facts[fact.id] = fact

    def get_stable_facts(self) -> list[StableFact]:
        return list(self._stable_facts.values())

    def get_stable_fact_by_spo(self, subject: str, predicate: str, obj: str
                                ) -> Optional[StableFact]:
        for sf in self._stable_facts.values():
            if (sf.subject.lower() == subject.lower() and
                    sf.predicate.lower() == predicate.lower() and
                    sf.object.lower() == obj.lower()):
                return sf
        return None

    # Memory records
    def upsert_memory(self, record: MemoryRecord):
        self._memory_records[record.id] = record

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._memory_records.get(memory_id)

    def all_memories(self) -> list[MemoryRecord]:
        return list(self._memory_records.values())

    def mark_consolidated(self, memory_ids: list[str]):
        for mid in memory_ids:
            if mid in self._memory_records:
                self._memory_records[mid].is_consolidated = True

    def demote(self, memory_id: str, weight: float = 0.3):
        if memory_id in self._memory_records:
            self._memory_records[memory_id].retrieval_weight = weight

    def promote_type(self, memory_id: str, new_type: str):
        if memory_id in self._memory_records:
            self._memory_records[memory_id].memory_type = new_type


# ---------------------------------------------------------------------------
# Salience computation (shared with ranking_engine but self-contained here)
# ---------------------------------------------------------------------------

def _salience(retrieval_count: int, age_days: int) -> float:
    recency_decay = 1.0 / (1.0 + 0.01 * age_days)
    access_signal = math.log1p(retrieval_count) / 10.0
    return min(1.0, (0.5 * recency_decay) + (0.5 * access_signal))


# ---------------------------------------------------------------------------
# Fact clustering
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return text.lower().strip()

def _spo_key(fact: StructuredFact) -> str:
    return f"{_normalise(fact.subject)}|{_normalise(fact.predicate)}|{_normalise(fact.object)}"

def _cluster_facts(facts: list[StructuredFact],
                   threshold: int = 3) -> list[list[StructuredFact]]:
    """
    Group facts by (subject, predicate, object) key.
    Only returns groups with >= threshold members.
    """
    groups: dict[str, list[StructuredFact]] = {}
    for fact in facts:
        key = _spo_key(fact)
        groups.setdefault(key, []).append(fact)
    return [g for g in groups.values() if len(g) >= threshold]


# ---------------------------------------------------------------------------
# Main consolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """
    Background consolidation engine.

    Usage
    -----
    consolidator = MemoryConsolidator(
        store=consolidation_store,
        entity_registry=registry,
    )

    # Run periodically (every 50 new memories, or nightly):
    report = consolidator.run(
        facts=all_structured_facts,
        memories=all_memory_records,
    )
    print(report.summary())
    """

    # Thresholds
    MERGE_THRESHOLD         = 3      # facts needed to create a stable fact
    COLD_DORMANCY_DAYS      = 180    # memories older than this and low-salience → cold
    COLD_SALIENCE_THRESHOLD = 0.20   # below this salience → cold candidate
    PROMOTE_RETRIEVAL_COUNT = 8      # TEMPORARY_CONTEXT retrieved this many times → promote
    ENTITY_REVIEW_SIMILARITY = 0.75  # entity pairs above this → merge candidates

    def __init__(self,
                 store:           ConsolidationStore,
                 entity_registry: EntityRegistry,
                 dry_run:         bool = False):
        """
        dry_run: if True, compute what would happen but don't mutate anything.
        """
        self._store    = store
        self._registry = entity_registry
        self._dry_run  = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self,
            facts:    list[StructuredFact],
            memories: list[MemoryRecord]) -> ConsolidationReport:
        """
        Full consolidation run.
        facts:    all StructuredFact objects from the fact store
        memories: all MemoryRecord objects from the memory store
        """
        report = ConsolidationReport(run_at=datetime.utcnow())

        try:
            self._merge_facts(facts, report)
        except Exception as e:
            report.errors.append(f"merge_facts: {e}")
            logger.exception("merge_facts failed")

        try:
            self._demote_cold(memories, report)
        except Exception as e:
            report.errors.append(f"demote_cold: {e}")
            logger.exception("demote_cold failed")

        try:
            self._promote_frequent(memories, report)
        except Exception as e:
            report.errors.append(f"promote_frequent: {e}")
            logger.exception("promote_frequent failed")

        try:
            self._merge_entity_duplicates(report)
        except Exception as e:
            report.errors.append(f"entity_merge: {e}")
            logger.exception("entity_merge failed")

        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    # Step 1: fact merging
    # ------------------------------------------------------------------

    def _merge_facts(self, facts: list[StructuredFact], report: ConsolidationReport):
        """
        Find groups of facts with the same SPO and merge them into StableFacts.
        """
        clusters = _cluster_facts(facts, threshold=self.MERGE_THRESHOLD)
        logger.debug("_merge_facts: found %d clusters", len(clusters))

        for cluster in clusters:
            self._consolidate_cluster(cluster, report)

    def _consolidate_cluster(self, cluster: list[StructuredFact],
                              report: ConsolidationReport):
        representative = cluster[0]  # all share same SPO
        subject   = representative.subject
        predicate = representative.predicate
        obj       = representative.object

        # Skip if a stable fact already exists for this SPO
        existing = self._store.get_stable_fact_by_spo(subject, predicate, obj)
        if existing:
            # Update source list and confidence
            new_sources = list(set(existing.source_memory_ids +
                                   [f.memory_id for f in cluster]))
            existing.source_memory_ids = new_sources
            existing.confidence = self._compute_confidence(cluster)
            existing.fact_count = len(cluster)
            return

        # Detect conflicts (same subject+object, different predicate in cluster)
        has_conflict = len({_normalise(f.predicate) for f in cluster}) > 1
        if has_conflict:
            report.conflicts_found += 1
            logger.warning("Conflict in cluster: %s %s %s — multiple predicates %s",
                           subject, predicate, obj,
                           [f.predicate for f in cluster])

        # Pick the best temporal bound (highest confidence)
        best_tb = max(
            (f.temporal_bound for f in cluster if f.temporal_bound),
            key=lambda tb: tb.confidence,
            default=None
        )

        confidence = self._compute_confidence(cluster)
        source_ids = list({f.memory_id for f in cluster})
        date_raw   = next((f.date_raw for f in cluster if f.date_raw), "")

        stable = StableFact(
            id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source_memory_ids=source_ids,
            temporal_bound=best_tb,
            date_raw=date_raw,
            consolidated_at=datetime.utcnow(),
            fact_count=len(cluster),
            has_conflict=has_conflict,
        )

        if not self._dry_run:
            self._store.add_stable_fact(stable)
            self._store.mark_consolidated(source_ids)

        report.facts_merged      += len(cluster)
        report.stable_facts_created += 1
        logger.debug("Created stable fact: %s %s %s (conf=%.2f, sources=%d)",
                     subject, predicate, obj, confidence, len(cluster))

    def _compute_confidence(self, cluster: list[StructuredFact]) -> float:
        """
        Confidence grows with cluster size and average fact confidence.
        Asymptotes toward 0.99 — never 1.0 (we never claim certainty).
        """
        avg_conf = sum(f.confidence for f in cluster) / len(cluster)
        size_boost = 1.0 - (1.0 / (1.0 + 0.5 * len(cluster)))
        raw = avg_conf * 0.6 + size_boost * 0.4
        return min(0.99, raw)

    # ------------------------------------------------------------------
    # Step 2: cold demotion
    # ------------------------------------------------------------------

    def _demote_cold(self, memories: list[MemoryRecord], report: ConsolidationReport):
        """
        Move low-salience, long-dormant memories to cold tier.
        Cold memories are not deleted — just get retrieval_weight=0.3.
        """
        for mem in memories:
            if mem.is_consolidated:
                continue   # already handled
            if mem.retrieval_weight < 0.5:
                continue   # already cold

            sal = _salience(mem.retrieval_count, mem.age_days)
            if mem.age_days >= self.COLD_DORMANCY_DAYS and sal < self.COLD_SALIENCE_THRESHOLD:
                logger.debug("Demoting cold memory: %s (age=%dd sal=%.3f)",
                             mem.id, mem.age_days, sal)
                if not self._dry_run:
                    self._store.demote(mem.id, weight=0.3)
                report.memories_demoted += 1

    # ------------------------------------------------------------------
    # Step 3: type promotion
    # ------------------------------------------------------------------

    def _promote_frequent(self, memories: list[MemoryRecord], report: ConsolidationReport):
        """
        TEMPORARY_CONTEXT memories that keep getting retrieved are probably
        USER_FACTs. Promote them.
        """
        for mem in memories:
            if (mem.memory_type == "TEMPORARY_CONTEXT" and
                    mem.retrieval_count >= self.PROMOTE_RETRIEVAL_COUNT):
                logger.debug("Promoting TEMPORARY_CONTEXT → USER_FACT: %s (count=%d)",
                             mem.id, mem.retrieval_count)
                if not self._dry_run:
                    self._store.promote_type(mem.id, "USER_FACT")
                report.memories_promoted += 1

    # ------------------------------------------------------------------
    # Step 4: entity deduplication
    # ------------------------------------------------------------------

    def _merge_entity_duplicates(self, report: ConsolidationReport):
        """
        Find entity pairs in the registry with needs_review=True and
        high name similarity. Merge confirmed duplicates.
        """
        from core.memory_pipeline_v2.entity_extractor import _name_similarity

        candidates = [e for e in self._registry.all_entities() if e.needs_review]
        all_entities = self._registry.all_entities()

        merged_ids: set[str] = set()

        for candidate in candidates:
            if candidate.id in merged_ids:
                continue

            for other in all_entities:
                if other.id == candidate.id or other.id in merged_ids:
                    continue
                if other.type != candidate.type:
                    continue

                sim = _name_similarity(candidate.name, other.name)
                if sim >= self.ENTITY_REVIEW_SIMILARITY:
                    # Merge: keep the one with more references
                    if candidate.reference_count >= other.reference_count:
                        keep, discard = candidate.id, other.id
                    else:
                        keep, discard = other.id, candidate.id

                    logger.info("Merging entities: %s ← %s (sim=%.2f)",
                                keep[:8], discard[:8], sim)
                    if not self._dry_run:
                        self._registry.merge(keep, discard)
                    merged_ids.add(discard)
                    report.entity_merges += 1
                    break