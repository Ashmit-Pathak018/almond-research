"""
timeline_index.py
-----------------
Phase 3 — Temporal event store.

Stores and retrieves events indexed by time. This is a dedicated SQLite
database — separate from the vector store — because temporal queries are
fundamentally different from similarity queries.

Questions it answers directly:
  "Which did I get first — Samsung or Dell?"
  "What happened between January and March?"
  "When did I buy my phone?"
  "How many days between the two purchases?"

Data model
----------
Each TimelineEvent maps one StructuredFact (from fact_extractor) to a
temporal record. A single memory can produce multiple events (e.g. "I
bought a Samsung in January and attended a conference in February" → 2
events).

TemporalBound → earliest/latest/confidence are all stored. Every query
operates on ranges, not point timestamps, which prevents false ordering
when dates are approximate.

Schema
------
events:
  id TEXT PK
  memory_id TEXT
  fact_id TEXT
  description TEXT
  earliest TEXT          -- ISO datetime
  latest TEXT            -- ISO datetime
  temporal_confidence REAL
  granularity TEXT
  date_raw TEXT
  event_type TEXT        -- purchased / attended / completed / works_at / etc.
  entities TEXT          -- JSON array of entity IDs
  created_at TEXT

entity_event_map:
  entity_id TEXT
  event_id TEXT
  PRIMARY KEY (entity_id, event_id)
  -- fast lookup: all events for a given entity
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from core.memory_pipeline_v2.fact_extractor import StructuredFact, TemporalBound, TemporalGranularity, FactType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    id:                   str
    memory_id:            str
    fact_id:              str
    description:          str          # human-readable: "user purchased Samsung Galaxy S22"
    earliest:             datetime
    latest:               datetime
    temporal_confidence:  float
    granularity:          TemporalGranularity
    date_raw:             str
    event_type:           str          # normalised predicate from fact
    entity_ids:           list[str]    # entity IDs involved
    created_at:           datetime

    @property
    def midpoint(self) -> datetime:
        delta = self.latest - self.earliest
        return self.earliest + delta / 2

    @property
    def temporal_bound(self) -> TemporalBound:
        return TemporalBound(
            earliest=self.earliest,
            latest=self.latest,
            confidence=self.temporal_confidence,
            granularity=self.granularity,
            raw=self.date_raw,
        )

    def to_dict(self) -> dict:
        return {
            "id":                  self.id,
            "memory_id":           self.memory_id,
            "fact_id":             self.fact_id,
            "description":         self.description,
            "earliest":            self.earliest.isoformat(),
            "latest":              self.latest.isoformat(),
            "temporal_confidence": self.temporal_confidence,
            "granularity":         self.granularity.value,
            "date_raw":            self.date_raw,
            "event_type":          self.event_type,
            "entity_ids":          self.entity_ids,
            "created_at":          self.created_at.isoformat(),
        }


@dataclass
class OrderingResult:
    """
    Result of a 'which came first?' query.
    """
    entity_a_id:   str
    entity_b_id:   str
    first_id:      Optional[str]     # ID of the entity that came first; None if inconclusive
    second_id:     Optional[str]
    event_a:       Optional[TimelineEvent]
    event_b:       Optional[TimelineEvent]
    delta_days:    Optional[int]     # approximate gap in days
    confidence:    float             # combined temporal confidence
    inconclusive:  bool = False
    reason:        str = ""


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS timeline_events (
    id                   TEXT PRIMARY KEY,
    memory_id            TEXT NOT NULL,
    fact_id              TEXT NOT NULL,
    description          TEXT NOT NULL,
    earliest             TEXT NOT NULL,
    latest               TEXT NOT NULL,
    temporal_confidence  REAL NOT NULL,
    granularity          TEXT NOT NULL,
    date_raw             TEXT NOT NULL DEFAULT '',
    event_type           TEXT NOT NULL,
    entities             TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_event_map (
    entity_id  TEXT NOT NULL,
    event_id   TEXT NOT NULL,
    PRIMARY KEY (entity_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_memory  ON timeline_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON timeline_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_earliest ON timeline_events(earliest);
CREATE INDEX IF NOT EXISTS idx_eem_entity     ON entity_event_map(entity_id);
"""

# Predicates that represent indexable events (have a meaningful temporal component)
_INDEXABLE_PREDICATES = {
    "purchased", "attended", "completed", "started", "finished",
    "moved_to", "works_at", "lives_in", "visited", "launched",
    "released", "joined", "left", "graduated", "hired",
    "booked",     # safety net: if LLM emits "booked" before normalization
    "set_up",     # "I set up my smart thermostat" - critical for Q19-type queries
    "installed",  # "I installed X" - similar setup context
}


# ---------------------------------------------------------------------------
# Timeline Index
# ---------------------------------------------------------------------------

class TimelineIndex:
    """
    SQLite-backed store for temporal events.

    Usage
    -----
    index = TimelineIndex(":memory:")          # in-memory for tests
    index = TimelineIndex("almond_timeline.db") # persistent

    # Store a fact
    event = index.store_fact(fact, entity_ids=["entity-uuid-1"])

    # Query ordering
    result = index.compare_order(entity_a_id, entity_b_id, event_type="purchased")
    if not result.inconclusive:
        print(f"{result.first_id} was first by ~{result.delta_days} days")
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path    = db_path
        self._in_memory  = (db_path == ":memory:")
        # For :memory: databases keep a single persistent connection so all
        # operations share the same database instance. File-backed databases
        # use per-operation connections which is safe for WAL mode.
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_db(self):
        if self._in_memory:
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys=ON")
            self._mem_conn.executescript(_DDL)
        else:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.executescript(_DDL)
            finally:
                conn.close()

    @contextmanager
    def _conn(self):
        if self._in_memory:
            # Reuse the persistent in-memory connection
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
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_fact(self, fact: StructuredFact,
                   entity_ids: Optional[list[str]] = None) -> Optional[TimelineEvent]:
        """
        Index a StructuredFact as a TimelineEvent.

        Returns None if:
          - The fact has no temporal bound (no date information)
          - The predicate is not in the indexable set
          - Temporal confidence is too low to be useful (< 0.25)
        """
        if not fact.temporal_bound:
            logger.debug("store_fact: skipping fact with no temporal_bound (fact_id=%s)", fact.id)
            return None

        if fact.temporal_bound.confidence < 0.25:
            logger.debug("store_fact: skipping low-confidence temporal fact (conf=%.2f)", fact.temporal_bound.confidence)
            return None

        # Only index predicates that represent real events
        if fact.predicate not in _INDEXABLE_PREDICATES:
            logger.debug("store_fact: predicate %r not indexable", fact.predicate)
            return None

        event = TimelineEvent(
            id=str(uuid.uuid4()),
            memory_id=fact.memory_id,
            fact_id=fact.id,
            description=f"{fact.subject} {fact.predicate} {fact.object}",
            earliest=fact.temporal_bound.earliest,
            latest=fact.temporal_bound.latest,
            temporal_confidence=fact.temporal_bound.confidence,
            granularity=fact.temporal_bound.granularity,
            date_raw=fact.date_raw,
            event_type=fact.predicate,
            entity_ids=entity_ids or [],
            created_at=datetime.utcnow(),
        )

        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO timeline_events
                (id, memory_id, fact_id, description, earliest, latest,
                 temporal_confidence, granularity, date_raw, event_type, entities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id, event.memory_id, event.fact_id, event.description,
                event.earliest.isoformat(), event.latest.isoformat(),
                event.temporal_confidence, event.granularity.value,
                event.date_raw, event.event_type,
                json.dumps(entity_ids or []),
                event.created_at.isoformat(),
            ))

            # Populate entity→event map for fast entity-based lookups
            for eid in (entity_ids or []):
                conn.execute("""
                    INSERT OR IGNORE INTO entity_event_map (entity_id, event_id)
                    VALUES (?, ?)
                """, (eid, event.id))

        logger.debug("store_fact: indexed event %s (%s)", event.id, event.description)
        return event

    def store_facts_batch(self, facts: list[StructuredFact],
                          entity_ids_map: Optional[dict[str, list[str]]] = None
                          ) -> list[TimelineEvent]:
        """
        Index multiple facts.
        entity_ids_map: {fact_id → [entity_ids]}
        """
        events = []
        for fact in facts:
            eids = (entity_ids_map or {}).get(fact.id, [])
            event = self.store_fact(fact, entity_ids=eids)
            if event:
                events.append(event)
        return events

    def delete_by_memory(self, memory_id: str):
        """Remove all events for a memory (used when a memory is retracted)."""
        with self._conn() as conn:
            # Get event IDs first for map cleanup
            rows = conn.execute(
                "SELECT id FROM timeline_events WHERE memory_id = ?", (memory_id,)
            ).fetchall()
            event_ids = [r["id"] for r in rows]
            for eid in event_ids:
                conn.execute("DELETE FROM entity_event_map WHERE event_id = ?", (eid,))
            conn.execute("DELETE FROM timeline_events WHERE memory_id = ?", (memory_id,))
        logger.debug("delete_by_memory: removed %d events for memory %s", len(event_ids), memory_id)

    # ------------------------------------------------------------------
    # Read — single event lookups
    # ------------------------------------------------------------------

    def get_by_id(self, event_id: str) -> Optional[TimelineEvent]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM timeline_events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    def get_by_memory(self, memory_id: str) -> list[TimelineEvent]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM timeline_events WHERE memory_id = ? ORDER BY earliest ASC",
                (memory_id,)
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------
    # Read — entity-based queries
    # ------------------------------------------------------------------

    def get_events_for_entity(self, entity_id: str,
                               event_type: Optional[str] = None) -> list[TimelineEvent]:
        """
        All timeline events referencing a given entity, sorted chronologically.
        Optionally filtered by event_type (e.g. "purchased").
        """
        with self._conn() as conn:
            if event_type:
                rows = conn.execute("""
                    SELECT te.* FROM timeline_events te
                    JOIN entity_event_map eem ON te.id = eem.event_id
                    WHERE eem.entity_id = ? AND te.event_type = ?
                    ORDER BY te.earliest ASC
                """, (entity_id, event_type)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT te.* FROM timeline_events te
                    JOIN entity_event_map eem ON te.id = eem.event_id
                    WHERE eem.entity_id = ?
                    ORDER BY te.earliest ASC
                """, (entity_id,)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_first_event_for_entity(self, entity_id: str,
                                    event_type: Optional[str] = None
                                    ) -> Optional[TimelineEvent]:
        """First (earliest) event for an entity, optionally filtered by type."""
        events = self.get_events_for_entity(entity_id, event_type)
        return events[0] if events else None

    def get_events_for_entities(self, entity_ids: list[str],
                                  event_type: Optional[str] = None) -> list[TimelineEvent]:
        """Union of events for multiple entities, deduplicated, sorted by earliest."""
        seen_ids: set[str] = set()
        all_events: list[TimelineEvent] = []
        for eid in entity_ids:
            for event in self.get_events_for_entity(eid, event_type):
                if event.id not in seen_ids:
                    seen_ids.add(event.id)
                    all_events.append(event)
        return sorted(all_events, key=lambda e: e.earliest)

    # ------------------------------------------------------------------
    # Read — range queries
    # ------------------------------------------------------------------

    def get_events_in_range(self, start: datetime, end: datetime,
                             event_type: Optional[str] = None) -> list[TimelineEvent]:
        """
        Events whose temporal range overlaps [start, end].
        Overlap condition: event.earliest <= end AND event.latest >= start
        """
        with self._conn() as conn:
            if event_type:
                rows = conn.execute("""
                    SELECT * FROM timeline_events
                    WHERE earliest <= ? AND latest >= ? AND event_type = ?
                    ORDER BY earliest ASC
                """, (end.isoformat(), start.isoformat(), event_type)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM timeline_events
                    WHERE earliest <= ? AND latest >= ?
                    ORDER BY earliest ASC
                """, (end.isoformat(), start.isoformat())).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_all_events(self, event_type: Optional[str] = None,
                       limit: int = 200) -> list[TimelineEvent]:
        """All events sorted chronologically. Mainly for debugging."""
        with self._conn() as conn:
            if event_type:
                rows = conn.execute(
                    "SELECT * FROM timeline_events WHERE event_type = ? ORDER BY earliest ASC LIMIT ?",
                    (event_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM timeline_events ORDER BY earliest ASC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------
    # Ordering queries — the main reason this module exists
    # ------------------------------------------------------------------

    def compare_order(self, entity_a_id: str, entity_b_id: str,
                      event_type: Optional[str] = None) -> OrderingResult:
        """
        Answer "which came first?" for two entities.

        event_type: if provided, only considers events of that type
                    (e.g. "purchased" to answer "which did I buy first?")
                    If None, uses the earliest event of any type.

        Returns an OrderingResult with:
          - first_id / second_id: entity IDs in temporal order
          - delta_days: approximate gap (midpoint to midpoint)
          - confidence: min(event_a.confidence, event_b.confidence)
          - inconclusive: True if dates overlap or confidence too low
        """
        event_a = self.get_first_event_for_entity(entity_a_id, event_type)
        event_b = self.get_first_event_for_entity(entity_b_id, event_type)

        if not event_a and not event_b:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=None, second_id=None,
                event_a=None, event_b=None,
                delta_days=None, confidence=0.0,
                inconclusive=True,
                reason="no_events_found_for_either_entity",
            )

        if not event_a:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=None, second_id=None,
                event_a=None, event_b=event_b,
                delta_days=None, confidence=0.0,
                inconclusive=True,
                reason=f"no_events_found_for_entity_a:{entity_a_id}",
            )

        if not event_b:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=None, second_id=None,
                event_a=event_a, event_b=None,
                delta_days=None, confidence=0.0,
                inconclusive=True,
                reason=f"no_events_found_for_entity_b:{entity_b_id}",
            )

        # Use TemporalBound.is_before for range-aware comparison
        tb_a = event_a.temporal_bound
        tb_b = event_b.temporal_bound
        combined_confidence = min(tb_a.confidence, tb_b.confidence)

        is_before = tb_a.is_before(tb_b)

        if is_before is None:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=None, second_id=None,
                event_a=event_a, event_b=event_b,
                delta_days=None,
                confidence=combined_confidence,
                inconclusive=True,
                reason="temporal_ranges_overlap_or_confidence_too_low",
            )

        delta = abs((event_a.midpoint - event_b.midpoint).days)

        if is_before:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=entity_a_id, second_id=entity_b_id,
                event_a=event_a, event_b=event_b,
                delta_days=delta,
                confidence=combined_confidence,
                inconclusive=False,
                reason="a_before_b",
            )
        else:
            return OrderingResult(
                entity_a_id=entity_a_id, entity_b_id=entity_b_id,
                first_id=entity_b_id, second_id=entity_a_id,
                event_a=event_a, event_b=event_b,
                delta_days=delta,
                confidence=combined_confidence,
                inconclusive=False,
                reason="b_before_a",
            )

    def build_timeline(self, entity_ids: Optional[list[str]] = None,
                       event_type: Optional[str] = None,
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None) -> list[TimelineEvent]:
        """
        Build a chronological timeline, optionally filtered by:
          - entity_ids: only events involving these entities
          - event_type: only events of this predicate type
          - start/end: only events in this date range
        """
        if entity_ids:
            events = self.get_events_for_entities(entity_ids, event_type)
        else:
            events = self.get_all_events(event_type)

        if start:
            events = [e for e in events if e.latest >= start]
        if end:
            events = [e for e in events if e.earliest <= end]

        return sorted(events, key=lambda e: e.earliest)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM timeline_events").fetchone()[0]

    def count_for_entity(self, entity_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM entity_event_map WHERE entity_id = ?", (entity_id,)
            ).fetchone()[0]

    # ------------------------------------------------------------------
    # Row deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TimelineEvent:
        try:
            granularity = TemporalGranularity(row["granularity"])
        except ValueError:
            granularity = TemporalGranularity.UNKNOWN

        try:
            entity_ids = json.loads(row["entities"])
        except (json.JSONDecodeError, TypeError):
            entity_ids = []

        return TimelineEvent(
            id=row["id"],
            memory_id=row["memory_id"],
            fact_id=row["fact_id"],
            description=row["description"],
            earliest=datetime.fromisoformat(row["earliest"]),
            latest=datetime.fromisoformat(row["latest"]),
            temporal_confidence=row["temporal_confidence"],
            granularity=granularity,
            date_raw=row["date_raw"],
            event_type=row["event_type"],
            entity_ids=entity_ids,
            created_at=datetime.fromisoformat(row["created_at"]),
        )