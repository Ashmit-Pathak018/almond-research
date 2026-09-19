"""
Project Almond V3 — Entity Store
SQLite-backed entity registry with deterministic alias resolution.

This replaces the in-memory EntityRegistry for Phase 2+.
SQLite is the sole source of truth for entities, aliases, and memory-entity relationships.

Tables used (created by migration_001 and migration_002):
    - entities           : canonical entity records
    - entity_aliases     : normalized alias → entity mapping
    - memory_entities    : memory ↔ entity join with confidence
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class EntityRecord:
    """Represents an entity as stored in SQLite."""
    entity_id: str
    namespace_id: str
    canonical_name: str
    entity_type: str          # PERSON, ORGANIZATION, LOCATION, PROJECT, CONCEPT, TOOL
    summary: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata_json: str = "{}"


@dataclass
class AliasRecord:
    """Represents an alias mapping in SQLite."""
    alias_id: str
    entity_id: str
    alias_name: str
    created_at: float = 0.0


@dataclass
class AliasResolution:
    """Result of resolving an alias to a canonical entity."""
    entity: EntityRecord
    confidence: float               # 1.0 = exact canonical, 0.95 = exact alias, <0.95 = normalized
    matched_alias: str              # the alias string that matched
    resolution_type: str            # "canonical", "alias_exact", "alias_normalized"


# ---------------------------------------------------------------------------
# Entity Store
# ---------------------------------------------------------------------------

class EntityStore:
    """
    SQLite-backed entity store.

    All operations are scoped to a namespace_id for tenant isolation.
    The store is NOT the authoritative entity cache — SQLite is the truth.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entity(
        self,
        namespace_id: str,
        canonical_name: str,
        entity_type: str,
        summary: Optional[str] = None,
        metadata_json: str = "{}",
        entity_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> EntityRecord:
        """Create a new entity. Also inserts canonical_name as an alias."""
        eid = entity_id or str(uuid.uuid4())
        now = created_at or time.time()

        with self._conn:
            self._conn.execute("""
                INSERT INTO entities (id, namespace_id, name, type, summary, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (eid, namespace_id, canonical_name, entity_type, summary, now, now, metadata_json))

            # Auto-register canonical name as an alias for deterministic resolution
            alias_id = str(uuid.uuid4())
            self._conn.execute("""
                INSERT OR IGNORE INTO entity_aliases (alias_id, entity_id, alias_name, created_at)
                VALUES (?, ?, ?, ?)
            """, (alias_id, eid, self._normalize(canonical_name), now))

        logger.debug("EntityStore.create_entity: %s (%s) id=%s ns=%s", canonical_name, entity_type, eid, namespace_id)

        return EntityRecord(
            entity_id=eid,
            namespace_id=namespace_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            summary=summary,
            created_at=now,
            updated_at=now,
            metadata_json=metadata_json,
        )

    def get_entity(self, entity_id: str, namespace_id: Optional[str] = None) -> Optional[EntityRecord]:
        """Fetch an entity by ID. Optionally scope to namespace."""
        if namespace_id:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ? AND namespace_id = ?",
                (entity_id, namespace_id)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_name(self, canonical_name: str, namespace_id: str) -> Optional[EntityRecord]:
        """Fetch entity by exact canonical name within a namespace."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE name = ? AND namespace_id = ?",
            (canonical_name, namespace_id)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def update_entity(
        self,
        entity_id: str,
        canonical_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        summary: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> Optional[EntityRecord]:
        """Update mutable fields of an entity."""
        now = time.time()
        entity = self.get_entity(entity_id)
        if not entity:
            return None

        new_name = canonical_name or entity.canonical_name
        new_type = entity_type or entity.entity_type
        new_summary = summary if summary is not None else entity.summary
        new_meta = metadata_json if metadata_json is not None else entity.metadata_json

        with self._conn:
            self._conn.execute("""
                UPDATE entities SET name = ?, type = ?, summary = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
            """, (new_name, new_type, new_summary, new_meta, now, entity_id))

            # If canonical name changed, add new name as alias
            if canonical_name and canonical_name != entity.canonical_name:
                alias_id = str(uuid.uuid4())
                self._conn.execute("""
                    INSERT OR IGNORE INTO entity_aliases (alias_id, entity_id, alias_name, created_at)
                    VALUES (?, ?, ?, ?)
                """, (alias_id, entity_id, self._normalize(canonical_name), now))

        return self.get_entity(entity_id)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity. CASCADE handles aliases and memory_entities links."""
        with self._conn:
            # Delete aliases explicitly (in case FK cascade not active on older schema)
            self._conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,))
            self._conn.execute("DELETE FROM memory_entities WHERE entity_id = ?", (entity_id,))
            cursor = self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        return cursor.rowcount > 0

    def list_entities(self, namespace_id: str) -> List[EntityRecord]:
        """List all entities in a namespace."""
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE namespace_id = ? ORDER BY name ASC",
            (namespace_id,)
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Alias Management
    # ------------------------------------------------------------------

    def add_alias(self, entity_id: str, alias_name: str) -> Optional[AliasRecord]:
        """Add an alias for an entity. Returns None if entity doesn't exist."""
        entity = self.get_entity(entity_id)
        if not entity:
            return None

        normalized = self._normalize(alias_name)
        now = time.time()
        aid = str(uuid.uuid4())

        with self._conn:
            try:
                self._conn.execute("""
                    INSERT INTO entity_aliases (alias_id, entity_id, alias_name, created_at)
                    VALUES (?, ?, ?, ?)
                """, (aid, entity_id, normalized, now))
            except sqlite3.IntegrityError:
                # Duplicate alias for this entity — return existing
                row = self._conn.execute(
                    "SELECT * FROM entity_aliases WHERE entity_id = ? AND alias_name = ?",
                    (entity_id, normalized)
                ).fetchone()
                if row:
                    return AliasRecord(alias_id=row[0], entity_id=row[1], alias_name=row[2], created_at=row[3])
                return None

        return AliasRecord(alias_id=aid, entity_id=entity_id, alias_name=normalized, created_at=now)

    def get_aliases(self, entity_id: str) -> List[AliasRecord]:
        """Get all aliases for an entity."""
        rows = self._conn.execute(
            "SELECT alias_id, entity_id, alias_name, created_at FROM entity_aliases WHERE entity_id = ? ORDER BY alias_name ASC",
            (entity_id,)
        ).fetchall()
        return [AliasRecord(alias_id=r[0], entity_id=r[1], alias_name=r[2], created_at=r[3]) for r in rows]

    def remove_alias(self, entity_id: str, alias_name: str) -> bool:
        """Remove a specific alias from an entity."""
        normalized = self._normalize(alias_name)
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ? AND alias_name = ?",
                (entity_id, normalized)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Alias Resolution
    # ------------------------------------------------------------------

    def resolve_alias(self, alias_query: str, namespace_id: str) -> Optional[AliasResolution]:
        """
        Deterministic alias resolution.

        Resolution order:
        1. Exact canonical name match → confidence 1.0
        2. Exact alias match (normalized) → confidence 0.95
        3. Case-insensitive/whitespace-normalized alias match → confidence 0.90

        Returns None if no match found.
        """
        normalized_query = self._normalize(alias_query)

        # 1. Exact canonical name match
        row = self._conn.execute(
            "SELECT * FROM entities WHERE LOWER(TRIM(name)) = ? AND namespace_id = ?",
            (normalized_query, namespace_id)
        ).fetchone()
        if row:
            entity = self._row_to_entity(row)
            return AliasResolution(
                entity=entity,
                confidence=1.0,
                matched_alias=entity.canonical_name,
                resolution_type="canonical",
            )

        # 2. Exact normalized alias match — join with entities for namespace isolation
        row = self._conn.execute("""
            SELECT e.*, ea.alias_name
            FROM entity_aliases ea
            JOIN entities e ON ea.entity_id = e.id
            WHERE ea.alias_name = ? AND e.namespace_id = ?
        """, (normalized_query, namespace_id)).fetchone()
        if row:
            entity = self._row_to_entity(row)
            return AliasResolution(
                entity=entity,
                confidence=0.95,
                matched_alias=row["alias_name"] if isinstance(row, sqlite3.Row) else row[-1],
                resolution_type="alias_exact",
            )

        # 3. Broader normalized match (handles extra whitespace, case variations)
        rows = self._conn.execute("""
            SELECT e.*, ea.alias_name
            FROM entity_aliases ea
            JOIN entities e ON ea.entity_id = e.id
            WHERE e.namespace_id = ?
        """, (namespace_id,)).fetchall()

        for r in rows:
            alias_col = r["alias_name"] if isinstance(r, sqlite3.Row) else r[-1]
            if self._normalize(alias_col) == normalized_query:
                entity = self._row_to_entity(r)
                return AliasResolution(
                    entity=entity,
                    confidence=0.90,
                    matched_alias=alias_col,
                    resolution_type="alias_normalized",
                )

        return None

    # ------------------------------------------------------------------
    # Memory ↔ Entity Relationships
    # ------------------------------------------------------------------

    def link_memory(
        self,
        memory_id: str,
        entity_id: str,
        confidence: float = 1.0,
        mention_offset_start: Optional[int] = None,
        mention_offset_end: Optional[int] = None,
    ) -> bool:
        """Link a memory to an entity via memory_entities join table."""
        with self._conn:
            try:
                self._conn.execute("""
                    INSERT OR REPLACE INTO memory_entities
                        (memory_id, entity_id, mention_offset_start, mention_offset_end, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (memory_id, entity_id, mention_offset_start, mention_offset_end, confidence))
                return True
            except sqlite3.IntegrityError as e:
                logger.warning("EntityStore.link_memory failed: %s", e)
                return False

    def unlink_memory(self, memory_id: str, entity_id: str) -> bool:
        """Remove a memory-entity link."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memory_entities WHERE memory_id = ? AND entity_id = ?",
                (memory_id, entity_id)
            )
        return cursor.rowcount > 0

    def get_memories_for_entity(self, entity_id: str) -> List[dict]:
        """Get all memory links for an entity, with confidence."""
        rows = self._conn.execute("""
            SELECT memory_id, entity_id, confidence, mention_offset_start, mention_offset_end
            FROM memory_entities WHERE entity_id = ?
            ORDER BY memory_id ASC
        """, (entity_id,)).fetchall()
        return [
            {
                "memory_id": r[0],
                "entity_id": r[1],
                "confidence": r[2],
                "mention_offset_start": r[3],
                "mention_offset_end": r[4],
            }
            for r in rows
        ]

    def get_entities_for_memory(self, memory_id: str) -> List[EntityRecord]:
        """Get all entities linked to a memory."""
        rows = self._conn.execute("""
            SELECT e.* FROM entities e
            JOIN memory_entities me ON e.id = me.entity_id
            WHERE me.memory_id = ?
            ORDER BY e.name ASC
        """, (memory_id,)).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a name/alias for consistent matching.
        Lowercases, strips, and collapses whitespace."""
        return " ".join(name.lower().strip().split())

    def _row_to_entity(self, row) -> EntityRecord:
        """Convert a SQLite row to EntityRecord."""
        if isinstance(row, sqlite3.Row):
            return EntityRecord(
                entity_id=row["id"],
                namespace_id=row["namespace_id"],
                canonical_name=row["name"],
                entity_type=row["type"],
                summary=row["summary"] if "summary" in row.keys() else None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata_json=row["metadata_json"] if "metadata_json" in row.keys() else "{}",
            )
        # Tuple fallback (positional based on entities table schema from migration_001)
        # id, namespace_id, name, type, aliases, first_seen, last_seen,
        # mention_count, summary, metadata_json, created_at, updated_at
        return EntityRecord(
            entity_id=row[0],
            namespace_id=row[1],
            canonical_name=row[2],
            entity_type=row[3],
            summary=row[8] if len(row) > 8 else None,
            created_at=row[10] if len(row) > 10 else 0.0,
            updated_at=row[11] if len(row) > 11 else 0.0,
            metadata_json=row[9] if len(row) > 9 else "{}",
        )
