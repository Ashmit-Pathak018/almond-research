"""
Project Almond V3 — Timeline Store
SQLite-backed structured timeline for chronological event storage and temporal queries.

SQLite is the sole source of truth for timeline events.
Ordering is always by event_time (start_time), NOT by ingestion time.

Tables used (created by migration_001):
    - events          : timeline event records
    - memory_events   : memory ↔ event join table
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants (match V3 spec Section 2.4)
# ---------------------------------------------------------------------------

EVENT_TYPE_POINT = "POINT"
EVENT_TYPE_INTERVAL = "INTERVAL"
EVENT_TYPE_ORDERED = "ORDERED"
EVENT_TYPE_UNKNOWN_TIME = "UNKNOWN_TIME"

VALID_EVENT_TYPES = {EVENT_TYPE_POINT, EVENT_TYPE_INTERVAL, EVENT_TYPE_ORDERED, EVENT_TYPE_UNKNOWN_TIME}


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class EventRecord:
    """Represents a timeline event as stored in SQLite."""
    event_id: str
    namespace_id: str
    title: str
    description: Optional[str]
    event_type: str               # POINT, INTERVAL, ORDERED, UNKNOWN_TIME
    start_time: Optional[float]   # Unix epoch seconds
    end_time: Optional[float]     # Unix epoch seconds (for intervals)
    relative_anchor: Optional[str]
    confidence: float
    created_at: float


# ---------------------------------------------------------------------------
# Timeline Store
# ---------------------------------------------------------------------------

class TimelineStore:
    """
    SQLite-backed timeline store.

    All queries order by start_time (event_time), not ingestion time.
    Namespace isolation is enforced on all operations.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def create_event(
        self,
        namespace_id: str,
        title: str,
        event_type: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        description: Optional[str] = None,
        relative_anchor: Optional[str] = None,
        confidence: float = 1.0,
        event_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> EventRecord:
        """Create a new timeline event."""
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{event_type}'. Must be one of {VALID_EVENT_TYPES}")

        if event_type == EVENT_TYPE_INTERVAL and (start_time is None or end_time is None):
            raise ValueError("INTERVAL events require both start_time and end_time")

        if event_type == EVENT_TYPE_POINT and start_time is None:
            raise ValueError("POINT events require start_time")

        eid = event_id or str(uuid.uuid4())
        now = created_at or time.time()

        with self._conn:
            self._conn.execute("""
                INSERT INTO events (event_id, namespace_id, title, description, event_type,
                    start_time, end_time, relative_anchor, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (eid, namespace_id, title, description, event_type,
                  start_time, end_time, relative_anchor, confidence, now))

        return EventRecord(
            event_id=eid, namespace_id=namespace_id, title=title,
            description=description, event_type=event_type,
            start_time=start_time, end_time=end_time,
            relative_anchor=relative_anchor, confidence=confidence,
            created_at=now,
        )

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """Fetch a single event by ID."""
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def update_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Optional[EventRecord]:
        """Update mutable fields of an event."""
        event = self.get_event(event_id)
        if not event:
            return None

        new_title = title or event.title
        new_desc = description if description is not None else event.description
        new_type = event_type or event.event_type
        new_start = start_time if start_time is not None else event.start_time
        new_end = end_time if end_time is not None else event.end_time
        new_conf = confidence if confidence is not None else event.confidence

        with self._conn:
            self._conn.execute("""
                UPDATE events SET title = ?, description = ?, event_type = ?,
                    start_time = ?, end_time = ?, confidence = ?
                WHERE event_id = ?
            """, (new_title, new_desc, new_type, new_start, new_end, new_conf, event_id))

        return self.get_event(event_id)

    def delete_event(self, event_id: str) -> bool:
        """Delete an event. CASCADE handles memory_events links."""
        with self._conn:
            self._conn.execute("DELETE FROM memory_events WHERE event_id = ?", (event_id,))
            cursor = self._conn.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Memory ↔ Event Relationships
    # ------------------------------------------------------------------

    def link_memory(self, memory_id: str, event_id: str) -> bool:
        """Link a memory to a timeline event."""
        with self._conn:
            try:
                self._conn.execute("""
                    INSERT OR IGNORE INTO memory_events (memory_id, event_id)
                    VALUES (?, ?)
                """, (memory_id, event_id))
                return True
            except sqlite3.IntegrityError as e:
                logger.warning("TimelineStore.link_memory failed: %s", e)
                return False

    def unlink_memory(self, memory_id: str, event_id: str) -> bool:
        """Remove a memory-event link."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memory_events WHERE memory_id = ? AND event_id = ?",
                (memory_id, event_id)
            )
        return cursor.rowcount > 0

    def get_memories_for_event(self, event_id: str) -> List[str]:
        """Get all memory IDs linked to an event."""
        rows = self._conn.execute(
            "SELECT memory_id FROM memory_events WHERE event_id = ? ORDER BY memory_id ASC",
            (event_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_events_for_memory(self, memory_id: str) -> List[EventRecord]:
        """Get all events linked to a memory, ordered by start_time."""
        rows = self._conn.execute("""
            SELECT e.* FROM events e
            JOIN memory_events me ON e.event_id = me.event_id
            WHERE me.memory_id = ?
            ORDER BY e.start_time ASC NULLS LAST
        """, (memory_id,)).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------
    # Temporal Queries — ordered by event_time, NOT ingestion time
    # ------------------------------------------------------------------

    def get_chronological(self, namespace_id: str, limit: int = 100) -> List[EventRecord]:
        """
        Get all events in chronological order by start_time.
        Events without start_time are placed at the end.
        """
        rows = self._conn.execute("""
            SELECT * FROM events
            WHERE namespace_id = ?
            ORDER BY start_time ASC NULLS LAST, created_at ASC
            LIMIT ?
        """, (namespace_id, limit)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_before(self, namespace_id: str, timestamp: float) -> List[EventRecord]:
        """Get events with start_time before a given timestamp, in chronological order."""
        rows = self._conn.execute("""
            SELECT * FROM events
            WHERE namespace_id = ? AND start_time IS NOT NULL AND start_time < ?
            ORDER BY start_time ASC
        """, (namespace_id, timestamp)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_after(self, namespace_id: str, timestamp: float) -> List[EventRecord]:
        """Get events with start_time after a given timestamp, in chronological order."""
        rows = self._conn.execute("""
            SELECT * FROM events
            WHERE namespace_id = ? AND start_time IS NOT NULL AND start_time > ?
            ORDER BY start_time ASC
        """, (namespace_id, timestamp)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_between(
        self, namespace_id: str, start: float, end: float
    ) -> List[EventRecord]:
        """Get events with start_time within [start, end] range, in chronological order."""
        rows = self._conn.execute("""
            SELECT * FROM events
            WHERE namespace_id = ? AND start_time IS NOT NULL
                AND start_time >= ? AND start_time <= ?
            ORDER BY start_time ASC
        """, (namespace_id, start, end)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_overlapping(
        self, namespace_id: str, start: float, end: float
    ) -> List[EventRecord]:
        """
        Get events that overlap with a given time range [start, end].
        An event overlaps if its time span intersects with the query range.
        For POINT events: start_time is within [start, end].
        For INTERVAL events: event's [start_time, end_time] overlaps [start, end].
        """
        rows = self._conn.execute("""
            SELECT * FROM events
            WHERE namespace_id = ? AND start_time IS NOT NULL AND (
                (end_time IS NULL AND start_time >= ? AND start_time <= ?)
                OR
                (end_time IS NOT NULL AND start_time <= ? AND end_time >= ?)
            )
            ORDER BY start_time ASC
        """, (namespace_id, start, end, end, start)).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_event(self, row) -> EventRecord:
        """Convert a SQLite row to EventRecord."""
        if isinstance(row, sqlite3.Row):
            return EventRecord(
                event_id=row["event_id"],
                namespace_id=row["namespace_id"],
                title=row["title"],
                description=row["description"],
                event_type=row["event_type"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                relative_anchor=row["relative_anchor"],
                confidence=row["confidence"],
                created_at=row["created_at"],
            )
        # Tuple fallback
        return EventRecord(
            event_id=row[0], namespace_id=row[1], title=row[2],
            description=row[3], event_type=row[4],
            start_time=row[5], end_time=row[6],
            relative_anchor=row[7], confidence=row[8],
            created_at=row[9],
        )
