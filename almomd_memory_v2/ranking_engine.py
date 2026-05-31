"""
ranking_engine.py
-----------------
Phase 4 — Multi-signal ranking with intent-weighted scores.

Takes raw RetrievedMemory objects from any retriever and produces a ranked
list with a full, inspectable score breakdown for every memory.

Signals
-------
similarity        Semantic similarity to the query (0.0–1.0).
                  Passed in from the retriever or vector store.
entity_overlap    Fraction of query entities that appear in the memory.
timeline_relevance Whether the memory has timeline events for this query's entities.
fact_confidence   Average confidence of structured facts extracted from the memory.
type_match        How well the memory's type matches the query intent.
salience          Long-term importance score (recency × access frequency × centrality).

Weights
-------
Each intent type uses a different weight profile. The weights were chosen
so the primary retrieval signal dominates but secondary signals can break
ties and re-order near-equal candidates.

FACTUAL:       similarity leads (0.50), entity_overlap supports (0.25)
TEMPORAL:      timeline_relevance leads (0.45), entity_overlap supports (0.30)
EVENT:         timeline_relevance + entity_overlap share (0.35 + 0.30)
COMPARISON:    entity_overlap leads (0.45), similarity supports (0.20)
RELATIONSHIP:  entity_overlap leads (0.45), similarity supports (0.25)

These are starting points — tune them against real LongMemEval failures
using the audit log (see RetrievalAuditLog below).

Audit log
---------
Every retrieval is logged to a SQLite audit table with the full score
breakdown for each ranked memory. When a memory retrieves wrong, run:

  SELECT * FROM retrieval_audit WHERE query LIKE '%Samsung%'
  ORDER BY timestamp DESC LIMIT 5;

and you will immediately see which signal was over/under-weighted.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from query_analyzer import QueryIntent, IntentType
from temporal_retriever import RetrievedMemory
from timeline_index import TimelineIndex
from entity_extractor import EntityRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score breakdown and ranked result
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    similarity:         float = 0.0
    entity_overlap:     float = 0.0
    timeline_relevance: float = 0.0
    fact_confidence:    float = 0.0
    type_match:         float = 0.0
    salience:           float = 0.0
    weights_used:       dict  = field(default_factory=dict)
    intent_type:        str   = ""

    def weighted_total(self) -> float:
        w = self.weights_used
        return (
            self.similarity         * w.get("similarity",         0.0) +
            self.entity_overlap     * w.get("entity_overlap",     0.0) +
            self.timeline_relevance * w.get("timeline_relevance", 0.0) +
            self.fact_confidence    * w.get("fact_confidence",    0.0) +
            self.type_match         * w.get("type_match",         0.0) +
            self.salience           * w.get("salience",           0.0)
        )

    def to_dict(self) -> dict:
        return {
            "similarity":         round(self.similarity, 4),
            "entity_overlap":     round(self.entity_overlap, 4),
            "timeline_relevance": round(self.timeline_relevance, 4),
            "fact_confidence":    round(self.fact_confidence, 4),
            "type_match":         round(self.type_match, 4),
            "salience":           round(self.salience, 4),
            "weights_used":       self.weights_used,
            "intent_type":        self.intent_type,
            "final_score":        round(self.weighted_total(), 4),
        }


@dataclass
class RankedMemory:
    memory_id:    str
    text:         str
    final_score:  float
    rank:         int               # 1-based position in final ranked list
    breakdown:    ScoreBreakdown
    memory_type:  str = ""
    reasoning:    str = ""          # one-sentence human-readable explanation

    def to_dict(self) -> dict:
        return {
            "memory_id":   self.memory_id,
            "text":        self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "rank":        self.rank,
            "final_score": round(self.final_score, 4),
            "memory_type": self.memory_type,
            "reasoning":   self.reasoning,
            "breakdown":   self.breakdown.to_dict(),
        }


# ---------------------------------------------------------------------------
# Memory metadata (for salience and type info)
# ---------------------------------------------------------------------------

@dataclass
class MemoryMeta:
    """
    Lightweight metadata about a stored memory.
    Plug in whatever you have from your existing SQLite/Chroma store.
    """
    memory_id:       str
    memory_type:     str   = "USER_FACT"
    retrieval_count: int   = 0
    age_days:        int   = 0
    fact_confidences: list[float] = field(default_factory=list)

    @property
    def avg_fact_confidence(self) -> float:
        if not self.fact_confidences:
            return 0.5   # neutral default
        return sum(self.fact_confidences) / len(self.fact_confidences)


class MemoryMetaStore:
    """
    Simple in-process metadata store.
    Replace with a SQLite-backed version when connecting to Almond.
    """
    def __init__(self):
        self._data: dict[str, MemoryMeta] = {}

    def get(self, memory_id: str) -> Optional[MemoryMeta]:
        return self._data.get(memory_id)

    def set(self, meta: MemoryMeta):
        self._data[memory_id] = meta

    def increment_retrieval(self, memory_id: str):
        if memory_id in self._data:
            self._data[memory_id].retrieval_count += 1

    def upsert(self, memory_id: str, **kwargs) -> MemoryMeta:
        if memory_id not in self._data:
            self._data[memory_id] = MemoryMeta(memory_id=memory_id, **kwargs)
        else:
            for k, v in kwargs.items():
                setattr(self._data[memory_id], k, v)
        return self._data[memory_id]


# ---------------------------------------------------------------------------
# Weight profiles per intent type
# ---------------------------------------------------------------------------

_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    IntentType.FACTUAL.value: {
        "similarity":         0.50,
        "entity_overlap":     0.25,
        "timeline_relevance": 0.00,
        "fact_confidence":    0.15,
        "type_match":         0.05,
        "salience":           0.05,
    },
    IntentType.TEMPORAL.value: {
        "similarity":         0.10,
        "entity_overlap":     0.30,
        "timeline_relevance": 0.45,
        "fact_confidence":    0.10,
        "type_match":         0.00,
        "salience":           0.05,
    },
    IntentType.EVENT.value: {
        "similarity":         0.15,
        "entity_overlap":     0.30,
        "timeline_relevance": 0.35,
        "fact_confidence":    0.10,
        "type_match":         0.05,
        "salience":           0.05,
    },
    IntentType.COMPARISON.value: {
        "similarity":         0.20,
        "entity_overlap":     0.45,
        "timeline_relevance": 0.10,
        "fact_confidence":    0.15,
        "type_match":         0.05,
        "salience":           0.05,
    },
    IntentType.RELATIONSHIP.value: {
        "similarity":         0.25,
        "entity_overlap":     0.45,
        "timeline_relevance": 0.05,
        "fact_confidence":    0.10,
        "type_match":         0.10,
        "salience":           0.05,
    },
    IntentType.AMBIGUOUS.value: {
        "similarity":         0.35,
        "entity_overlap":     0.30,
        "timeline_relevance": 0.15,
        "fact_confidence":    0.10,
        "type_match":         0.05,
        "salience":           0.05,
    },
}

# Memory types that are a good match for each intent
_TYPE_MATCH_MAP: dict[str, set[str]] = {
    IntentType.FACTUAL.value:      {"USER_FACT", "PROJECT_FACT", "USER_PREFERENCE"},
    IntentType.TEMPORAL.value:     {"EVENT", "USER_FACT"},
    IntentType.EVENT.value:        {"EVENT", "TEMPORARY_CONTEXT"},
    IntentType.COMPARISON.value:   {"USER_FACT", "USER_PREFERENCE", "EVENT"},
    IntentType.RELATIONSHIP.value: {"USER_FACT", "PROJECT_FACT"},
    IntentType.AMBIGUOUS.value:    {"USER_FACT", "EVENT", "USER_PREFERENCE"},
}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class RetrievalAuditLog:
    """
    Persists every ranked retrieval for post-hoc debugging.

    Usage
    -----
    log = RetrievalAuditLog("almond_audit.db")   # or ":memory:"
    log.record(query="...", intent=intent, ranked=ranked_memories)

    # Then in sqlite3 CLI:
    # SELECT query, json(top5_breakdowns) FROM retrieval_audit
    # WHERE query LIKE '%Samsung%' ORDER BY timestamp DESC LIMIT 10;
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS retrieval_audit (
        id              TEXT PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        query           TEXT NOT NULL,
        intent_type     TEXT NOT NULL,
        intent_confidence REAL NOT NULL,
        top5_ids        TEXT NOT NULL,   -- JSON array
        top5_scores     TEXT NOT NULL,   -- JSON array of floats
        top5_breakdowns TEXT NOT NULL,   -- JSON array of breakdown dicts
        used_fallback   INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_audit_query     ON retrieval_audit(query);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON retrieval_audit(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_intent    ON retrieval_audit(intent_type);
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path   = db_path
        self._in_memory = (db_path == ":memory:")
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        if self._in_memory:
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.executescript(self._DDL)
        else:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.executescript(self._DDL)
            finally:
                conn.close()

    @contextmanager
    def _conn(self):
        if self._in_memory:
            conn = self._mem_conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def record(self, query: str, intent: QueryIntent,
               ranked: list[RankedMemory], used_fallback: bool = False):
        top5 = ranked[:5]
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO retrieval_audit
                (id, timestamp, query, intent_type, intent_confidence,
                 top5_ids, top5_scores, top5_breakdowns, used_fallback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                datetime.utcnow().isoformat(),
                query,
                intent.intent_type.value,
                round(intent.confidence, 4),
                json.dumps([m.memory_id for m in top5]),
                json.dumps([round(m.final_score, 4) for m in top5]),
                json.dumps([m.breakdown.to_dict() for m in top5]),
                int(used_fallback),
            ))

    def recent(self, n: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM retrieval_audit ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM retrieval_audit").fetchone()[0]


# ---------------------------------------------------------------------------
# Ranking Engine
# ---------------------------------------------------------------------------

class RankingEngine:
    """
    Scores and ranks a list of RetrievedMemory objects using multiple signals.

    Signals not already computed (entity_overlap, timeline_relevance, etc.)
    are computed here using the entity registry and timeline index.

    Usage
    -----
    engine = RankingEngine(
        entity_registry=registry,
        timeline_index=timeline,
        meta_store=meta_store,    # optional
        audit_log=audit_log,      # optional but strongly recommended
    )
    ranked = engine.rank(memories=retrieved, intent=query_intent)
    # ranked[0] is the most relevant memory
    """

    def __init__(self,
                 entity_registry: EntityRegistry,
                 timeline_index:  TimelineIndex,
                 meta_store:      Optional[MemoryMetaStore] = None,
                 audit_log:       Optional[RetrievalAuditLog] = None,
                 top_k:           int = 10):
        self._registry  = entity_registry
        self._timeline  = timeline_index
        self._meta      = meta_store or MemoryMetaStore()
        self._audit     = audit_log
        self._top_k     = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, memories: list[RetrievedMemory],
             intent: QueryIntent,
             used_fallback: bool = False) -> list[RankedMemory]:
        """
        Rank memories for a given query intent.
        Returns top_k RankedMemory objects sorted by final_score descending.
        """
        if not memories:
            return []

        weights    = self._get_weights(intent)
        query_eids = self._resolve_query_entities(intent)

        ranked: list[RankedMemory] = []
        for mem in memories:
            breakdown = self._compute_breakdown(mem, intent, query_eids, weights)
            final     = breakdown.weighted_total()
            reasoning = self._build_reasoning(breakdown, mem)
            meta      = self._meta.get(mem.memory_id)

            ranked.append(RankedMemory(
                memory_id=mem.memory_id,
                text=mem.text,
                final_score=round(final, 4),
                rank=0,           # set after sort
                breakdown=breakdown,
                memory_type=meta.memory_type if meta else "",
                reasoning=reasoning,
            ))

        # Sort descending, assign ranks
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        for i, r in enumerate(ranked, 1):
            r.rank = i

        ranked = ranked[: self._top_k]

        # Record to audit log
        if self._audit:
            self._audit.record(
                query=intent.raw_query,
                intent=intent,
                ranked=ranked,
                used_fallback=used_fallback,
            )

        # Increment retrieval counts in meta store
        for r in ranked:
            self._meta.increment_retrieval(r.memory_id)

        return ranked

    def merge_and_rank(self,
                       primary_memories:   list[RetrievedMemory],
                       secondary_memories: list[RetrievedMemory],
                       intent:             QueryIntent,
                       primary_weight:     float = 0.70) -> list[RankedMemory]:
        """
        Merge results from two retrievers (e.g. primary + fallback).
        primary_weight: how much to boost primary retriever scores.
        """
        # Boost primary, discount secondary, then rank normally
        boosted: list[RetrievedMemory] = []
        seen: set[str] = set()

        for m in primary_memories:
            m.score = min(1.0, m.score * primary_weight * (1 / primary_weight))
            boosted.append(m)
            seen.add(m.memory_id)

        for m in secondary_memories:
            if m.memory_id not in seen:
                m.score = m.score * (1.0 - primary_weight)
                boosted.append(m)

        return self.rank(boosted, intent)

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_breakdown(self, mem: RetrievedMemory, intent: QueryIntent,
                           query_eids: list[str], weights: dict) -> ScoreBreakdown:
        bd = ScoreBreakdown(
            weights_used=weights,
            intent_type=intent.intent_type.value,
        )

        # 1. Similarity — use score passed in from retriever
        bd.similarity = min(1.0, max(0.0, mem.score))

        # 2. Entity overlap
        bd.entity_overlap = self._entity_overlap(mem.memory_id, query_eids)

        # 3. Timeline relevance
        bd.timeline_relevance = self._timeline_relevance(mem, query_eids)

        # 4. Fact confidence
        meta = self._meta.get(mem.memory_id)
        bd.fact_confidence = meta.avg_fact_confidence if meta else 0.5

        # 5. Type match
        bd.type_match = self._type_match(mem, intent)

        # 6. Salience
        bd.salience = self._salience(meta)

        return bd

    def _entity_overlap(self, memory_id: str, query_entity_ids: list[str]) -> float:
        """Fraction of query entities that appear in this memory."""
        if not query_entity_ids:
            return 0.0
        memory_entity_ids = self._get_memory_entity_ids(memory_id)
        if not memory_entity_ids:
            return 0.0
        overlap = len(set(query_entity_ids) & set(memory_entity_ids))
        return overlap / len(query_entity_ids)

    def _timeline_relevance(self, mem: RetrievedMemory, query_eids: list[str]) -> float:
        """
        1.0 if the memory has a high-confidence timeline event for a query entity.
        Degrades with temporal confidence.
        """
        # If the retriever already attached timeline events, use their confidence
        # directly — the retriever already filtered by relevance.
        if mem.timeline_events:
            confidences = [e.temporal_confidence for e in mem.timeline_events]
            if confidences:
                return min(1.0, max(confidences))

        # Otherwise check the timeline index directly
        if not query_eids:
            return 0.0
        events = self._timeline.get_events_for_entities(query_eids)
        matching = [e for e in events if e.memory_id == mem.memory_id]
        if not matching:
            return 0.0
        return min(1.0, max(e.temporal_confidence for e in matching))

    def _type_match(self, mem: RetrievedMemory, intent: QueryIntent) -> float:
        meta = self._meta.get(mem.memory_id)
        if not meta:
            return 0.5   # neutral
        good_types = _TYPE_MATCH_MAP.get(intent.intent_type.value, set())
        return 1.0 if meta.memory_type in good_types else 0.2

    def _salience(self, meta: Optional[MemoryMeta]) -> float:
        """
        Combines recency decay + access frequency.
        Range: [0.0, 1.0]
        """
        if not meta:
            return 0.5   # neutral default

        import math
        recency_decay  = 1.0 / (1.0 + 0.01 * meta.age_days)
        access_signal  = math.log1p(meta.retrieval_count) / 10.0
        # Clamp to [0.0, 1.0]
        return min(1.0, (0.5 * recency_decay) + (0.5 * access_signal))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_weights(self, intent: QueryIntent) -> dict:
        profile = _WEIGHT_PROFILES.get(
            intent.intent_type.value,
            _WEIGHT_PROFILES[IntentType.FACTUAL.value]
        )
        # Make a copy so we can safely annotate it
        return dict(profile)

    def _resolve_query_entities(self, intent: QueryIntent) -> list[str]:
        """Resolve entity names in the query to entity IDs via the registry."""
        ids: list[str] = []
        seen: set[str] = set()
        names = list(dict.fromkeys(intent.entities_mentioned + intent.comparison_targets))
        for name in names:
            entity = self._registry.find_by_name(name)
            if entity and entity.id not in seen:
                ids.append(entity.id)
                seen.add(entity.id)
        return ids

    def _get_memory_entity_ids(self, memory_id: str) -> list[str]:
        """All entity IDs that reference this memory."""
        return [
            e.id for e in self._registry.all_entities()
            if memory_id in e.memory_ids
        ]

    def _build_reasoning(self, bd: ScoreBreakdown, mem: RetrievedMemory) -> str:
        """
        One-sentence explanation of why this memory ranked where it did.
        Identifies the top two contributing signals.
        """
        w = bd.weights_used
        signal_contributions = {
            "semantic similarity":   bd.similarity         * w.get("similarity", 0),
            "entity overlap":        bd.entity_overlap      * w.get("entity_overlap", 0),
            "timeline relevance":    bd.timeline_relevance  * w.get("timeline_relevance", 0),
            "fact confidence":       bd.fact_confidence     * w.get("fact_confidence", 0),
            "memory type match":     bd.type_match          * w.get("type_match", 0),
            "salience":              bd.salience            * w.get("salience", 0),
        }
        # Top two non-zero signals
        top = sorted(
            [(k, v) for k, v in signal_contributions.items() if v > 0.01],
            key=lambda x: x[1], reverse=True
        )[:2]

        source_note = f" (via {mem.source})" if mem.source != "entity_registry" else ""
        if not top:
            return f"Ranked by default signals{source_note}."

        top_labels = " and ".join(f"{k} ({v:.2f})" for k, v in top)
        return f"Ranked by {top_labels}{source_note}."