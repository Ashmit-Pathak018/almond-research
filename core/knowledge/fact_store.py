"""
Project Almond V3 — Fact State Store
SQLite-backed fact storage with state management: VERIFIED, INFERRED, CONFLICTING, SUPERSEDED.

Key invariants:
    - Never blindly deletes historical facts.
    - Superseded facts remain queryable with full provenance.
    - Newer verified facts supersede older facts with a pointer chain.
    - Temporal validity (valid_from, valid_to) tracks assertion windows.
    - SQLite is the sole source of truth.

Tables used (created by migration_001, enhanced by migration_002):
    - structured_facts  : fact records with state, supersession, temporal validity
    - memory_facts      : memory ↔ fact join table
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
# Fact state constants (match V3 spec Section 2.3)
# ---------------------------------------------------------------------------

FACT_STATE_VERIFIED = "VERIFIED"
FACT_STATE_INFERRED = "INFERRED"
FACT_STATE_CONFLICTING = "CONFLICTING"
FACT_STATE_SUPERSEDED = "SUPERSEDED"

VALID_FACT_STATES = {
    FACT_STATE_VERIFIED, FACT_STATE_INFERRED,
    FACT_STATE_CONFLICTING, FACT_STATE_SUPERSEDED,
}


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class FactRecord:
    """Represents a structured fact as stored in SQLite."""
    fact_id: str
    namespace_id: str
    memory_id: str              # Originating memory
    subject: str
    predicate: str
    object: str
    fact_type: str
    state: str                  # VERIFIED, INFERRED, CONFLICTING, SUPERSEDED
    confidence: float
    superseded_by: Optional[str]  # fact_id of the replacing fact
    valid_from: Optional[float]   # Unix epoch — when assertion became valid
    valid_to: Optional[float]     # Unix epoch — when assertion was invalidated
    created_at: float


# ---------------------------------------------------------------------------
# Fact Store
# ---------------------------------------------------------------------------

class FactStore:
    """
    SQLite-backed fact store with state management.

    Supports:
    - VERIFIED: Directly stated by user or corroborated.
    - INFERRED: Extracted via heuristic; lower confidence.
    - CONFLICTING: Contradicts an existing fact.
    - SUPERSEDED: Replaced by a more recent assertion.

    Never deletes historical facts. Supersession preserves the old fact
    with state=SUPERSEDED, superseded_by pointer, and closed valid_to.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")

    # ------------------------------------------------------------------
    # Fact CRUD
    # ------------------------------------------------------------------

    def create_fact(
        self,
        namespace_id: str,
        memory_id: str,
        subject: str,
        predicate: str,
        object_val: str,
        fact_type: str = "attribute",
        state: str = FACT_STATE_VERIFIED,
        confidence: float = 1.0,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        fact_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> FactRecord:
        """Create a new fact and link it to the originating memory."""
        if state not in VALID_FACT_STATES:
            raise ValueError(f"Invalid fact state '{state}'. Must be one of {VALID_FACT_STATES}")

        fid = fact_id or str(uuid.uuid4())
        now = created_at or time.time()

        with self._conn:
            self._conn.execute("""
                INSERT INTO structured_facts
                    (id, namespace_id, memory_id, subject, predicate, object, fact_type,
                     state, confidence, superseded_by, valid_from, valid_to, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """, (fid, namespace_id, memory_id, subject, predicate, object_val,
                  fact_type, state, confidence, valid_from, valid_to, now))

            # Auto-link via memory_facts join table
            self._conn.execute("""
                INSERT OR IGNORE INTO memory_facts (memory_id, fact_id)
                VALUES (?, ?)
            """, (memory_id, fid))

        return FactRecord(
            fact_id=fid, namespace_id=namespace_id, memory_id=memory_id,
            subject=subject, predicate=predicate, object=object_val,
            fact_type=fact_type, state=state, confidence=confidence,
            superseded_by=None, valid_from=valid_from, valid_to=valid_to,
            created_at=now,
        )

    def get_fact(self, fact_id: str) -> Optional[FactRecord]:
        """Fetch a single fact by ID."""
        row = self._conn.execute(
            "SELECT * FROM structured_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return self._row_to_fact(row) if row else None

    def get_facts_for_memory(self, memory_id: str) -> List[FactRecord]:
        """Get all facts linked to a memory."""
        rows = self._conn.execute("""
            SELECT sf.* FROM structured_facts sf
            JOIN memory_facts mf ON sf.id = mf.fact_id
            WHERE mf.memory_id = ?
            ORDER BY sf.created_at ASC
        """, (memory_id,)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_facts_by_subject_predicate(
        self, namespace_id: str, subject: str, predicate: str,
        include_superseded: bool = False,
    ) -> List[FactRecord]:
        """Get facts matching subject+predicate. Optionally includes superseded facts."""
        if include_superseded:
            rows = self._conn.execute("""
                SELECT * FROM structured_facts
                WHERE namespace_id = ? AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)
                ORDER BY created_at ASC
            """, (namespace_id, subject, predicate)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT * FROM structured_facts
                WHERE namespace_id = ? AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)
                    AND state != 'SUPERSEDED'
                ORDER BY created_at ASC
            """, (namespace_id, subject, predicate)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ------------------------------------------------------------------
    # State Transitions
    # ------------------------------------------------------------------

    def supersede_fact(self, old_fact_id: str, new_fact_id: str) -> bool:
        """
        Mark old_fact as SUPERSEDED and point superseded_by to new_fact.
        Closes the old fact's valid_to to the current time.
        Does NOT delete the old fact — it remains historically accessible.
        """
        old = self.get_fact(old_fact_id)
        new = self.get_fact(new_fact_id)
        if not old or not new:
            return False

        now = time.time()
        with self._conn:
            self._conn.execute("""
                UPDATE structured_facts
                SET state = 'SUPERSEDED', superseded_by = ?, valid_to = ?
                WHERE id = ?
            """, (new_fact_id, now, old_fact_id))

        logger.info("FactStore.supersede: %s SUPERSEDED by %s", old_fact_id, new_fact_id)
        return True

    def mark_conflicting(self, fact_id: str) -> bool:
        """Mark a fact as CONFLICTING."""
        with self._conn:
            cursor = self._conn.execute("""
                UPDATE structured_facts SET state = 'CONFLICTING', has_conflict = 1
                WHERE id = ?
            """, (fact_id,))
        return cursor.rowcount > 0

    def update_state(self, fact_id: str, new_state: str) -> bool:
        """Update the state of a fact."""
        if new_state not in VALID_FACT_STATES:
            raise ValueError(f"Invalid fact state '{new_state}'")
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE structured_facts SET state = ? WHERE id = ?",
                (new_state, fact_id)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Truth Resolution
    # ------------------------------------------------------------------

    def get_current_truth(
        self, namespace_id: str, subject: str, predicate: str
    ) -> Optional[FactRecord]:
        """
        Resolve the current ground truth for a subject+predicate.
        Returns the most recent VERIFIED fact, or None.
        """
        row = self._conn.execute("""
            SELECT * FROM structured_facts
            WHERE namespace_id = ? AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)
                AND state = 'VERIFIED'
            ORDER BY created_at DESC
            LIMIT 1
        """, (namespace_id, subject, predicate)).fetchone()
        return self._row_to_fact(row) if row else None

    def get_historical_facts(
        self, namespace_id: str, subject: str, predicate: str
    ) -> List[FactRecord]:
        """
        Get the full history of a subject+predicate including superseded facts.
        Returns in chronological order (oldest first).
        """
        rows = self._conn.execute("""
            SELECT * FROM structured_facts
            WHERE namespace_id = ? AND LOWER(subject) = LOWER(?) AND LOWER(predicate) = LOWER(?)
            ORDER BY created_at ASC
        """, (namespace_id, subject, predicate)).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ------------------------------------------------------------------
    # Memory Linking
    # ------------------------------------------------------------------

    def link_memory(self, memory_id: str, fact_id: str) -> bool:
        """Link a memory to a fact via memory_facts join table."""
        with self._conn:
            try:
                self._conn.execute("""
                    INSERT OR IGNORE INTO memory_facts (memory_id, fact_id)
                    VALUES (?, ?)
                """, (memory_id, fact_id))
                return True
            except sqlite3.IntegrityError as e:
                logger.warning("FactStore.link_memory failed: %s", e)
                return False

    def get_memories_for_fact(self, fact_id: str) -> List[str]:
        """Get all memory IDs linked to a fact."""
        rows = self._conn.execute(
            "SELECT memory_id FROM memory_facts WHERE fact_id = ? ORDER BY memory_id ASC",
            (fact_id,)
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Deletion (preserves history by default)
    # ------------------------------------------------------------------

    def delete_fact(self, fact_id: str) -> bool:
        """Hard-delete a fact. Use supersede_fact for normal lifecycle."""
        with self._conn:
            self._conn.execute("DELETE FROM memory_facts WHERE fact_id = ?", (fact_id,))
            cursor = self._conn.execute("DELETE FROM structured_facts WHERE id = ?", (fact_id,))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Namespace listing
    # ------------------------------------------------------------------

    def list_facts(self, namespace_id: str, state: Optional[str] = None) -> List[FactRecord]:
        """List all facts in a namespace, optionally filtered by state."""
        if state:
            rows = self._conn.execute(
                "SELECT * FROM structured_facts WHERE namespace_id = ? AND state = ? ORDER BY created_at ASC",
                (namespace_id, state)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM structured_facts WHERE namespace_id = ? ORDER BY created_at ASC",
                (namespace_id,)
            ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_fact(self, row) -> FactRecord:
        """Convert a SQLite row to FactRecord."""
        if isinstance(row, sqlite3.Row):
            keys = row.keys()
            return FactRecord(
                fact_id=row["id"],
                namespace_id=row["namespace_id"],
                memory_id=row["memory_id"],
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                fact_type=row["fact_type"],
                state=row["state"],
                confidence=row["confidence"],
                superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
                valid_from=row["valid_from"] if "valid_from" in keys else None,
                valid_to=row["valid_to"] if "valid_to" in keys else None,
                created_at=row["created_at"],
            )
        # Tuple fallback (positional, migration_001 structured_facts columns):
        # 0:id, 1:memory_id, 2:namespace_id, 3:subject, 4:predicate, 5:object,
        # 6:fact_type, 7:state, 8:confidence, 9:date_raw, 10:earliest, 11:latest,
        # 12:temporal_confidence, 13:granularity, 14:extraction_method,
        # 15:superseded_by, 16:needs_review, 17:has_conflict, 18:created_at,
        # 19:valid_from (if present), 20:valid_to (if present)
        return FactRecord(
            fact_id=row[0],
            namespace_id=row[2] if len(row) > 2 else "default",
            memory_id=row[1] if len(row) > 1 else "",
            subject=row[3] if len(row) > 3 else "",
            predicate=row[4] if len(row) > 4 else "",
            object=row[5] if len(row) > 5 else "",
            fact_type=row[6] if len(row) > 6 else "attribute",
            state=row[7] if len(row) > 7 else "VERIFIED",
            confidence=row[8] if len(row) > 8 else 1.0,
            superseded_by=row[15] if len(row) > 15 else None,
            valid_from=row[19] if len(row) > 19 else None,
            valid_to=row[20] if len(row) > 20 else None,
            created_at=row[18] if len(row) > 18 else 0.0,
        )
