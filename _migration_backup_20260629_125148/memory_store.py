"""
Project Almond — Memory Store v2
Handles dual-persistence: SQLite for absolute truth, ChromaDB for semantic search.

Changes:
    - FIX 2: L1 (Hot Cache) excluded from ChromaDB semantic indexing.
    - SOTA: Replaced monolithic semantic_search with semantic_search_metadata and
            get_blocks_by_ids to support the Metadata-First Reranking pipeline.
    - PHASE 2: New tables for structured_facts, entities, entity_memory_map.
               New methods: save_fact(), save_entity(), get_facts_for_memory(),
               get_all_facts(), get_entities_for_memory(), load_entity_registry().
    - BUGFIX: memory_blocks table now includes 'summary' column so L3→L4
              archival summaries are correctly persisted and restored.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import chromadb

from memory_block import MemoryBlock, MemoryTag, MemoryTier

# ── Phase 2 types (only imported for type hints in new methods) ────────────
from memory_pipeline_v2.fact_extractor import (
    StructuredFact, TemporalBound, TemporalGranularity, FactType,
)
from memory_pipeline_v2.entity_extractor import Entity, EntityType, EntityRegistry

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, db_path: str | Path = "longmem_almond.db",
                 chroma_path: str | Path = "./almond_chroma_db"):
        self.db_path = str(db_path)

        # 1. SQLite — source of truth
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_sqlite()

        # 2. ChromaDB — semantic index
        # chroma_path is now a constructor parameter (previously hardcoded to
        # "./almond_chroma_db"). This lets callers like eval_unified.py give
        # each benchmark iteration its own unique directory instead of
        # deleting/recreating the same path every time, which raced against
        # Windows file-handle release and could leave a corrupted Chroma
        # tenant directory after a failed rmtree (manifesting as
        # "Could not connect to tenant default_tenant").
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self.chroma_client.get_or_create_collection(
            name="almond_memory_vault",
            metadata={"hnsw:space": "cosine"},
        )

    # -----------------------------------------------------------------------
    # Schema initialisation
    # -----------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        """Create all tables if they don't exist."""
        with self._conn:
            # ── Existing table (summary column added) ─────────────────────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_blocks (
                    id               TEXT PRIMARY KEY,
                    content          TEXT NOT NULL,
                    summary          TEXT,
                    tag              TEXT NOT NULL,
                    tier             TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    keywords         TEXT NOT NULL,
                    source           TEXT NOT NULL,
                    session_id       TEXT,
                    created_at       REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    access_count     INTEGER NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tier ON memory_blocks(tier)"
            )

            # ── Migrate: add summary column to existing databases ──────────
            # Safe no-op if the column already exists.
            try:
                self._conn.execute(
                    "ALTER TABLE memory_blocks ADD COLUMN summary TEXT"
                )
            except sqlite3.OperationalError:
                pass   # column already exists — fine

            # ── Phase 2: structured facts ──────────────────────────────────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS structured_facts (
                    id                  TEXT PRIMARY KEY,
                    memory_id           TEXT NOT NULL,
                    subject             TEXT NOT NULL,
                    predicate           TEXT NOT NULL,
                    object              TEXT NOT NULL,
                    fact_type           TEXT NOT NULL,
                    confidence          REAL NOT NULL,
                    date_raw            TEXT DEFAULT '',
                    earliest            TEXT,
                    latest              TEXT,
                    temporal_confidence REAL DEFAULT 0.0,
                    granularity         TEXT DEFAULT 'unknown',
                    extraction_method   TEXT DEFAULT 'heuristic',
                    needs_review        INTEGER DEFAULT 0,
                    has_conflict        INTEGER DEFAULT 0
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_memory "
                "ON structured_facts(memory_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_predicate "
                "ON structured_facts(predicate)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_object "
                "ON structured_facts(object)"
            )

            # ── Phase 2: entity registry ───────────────────────────────────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id              TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    type            TEXT NOT NULL,
                    aliases         TEXT DEFAULT '[]',
                    first_seen      TEXT,
                    last_seen       TEXT,
                    memory_ids      TEXT DEFAULT '[]',
                    fact_ids        TEXT DEFAULT '[]',
                    reference_count INTEGER DEFAULT 0,
                    needs_review    INTEGER DEFAULT 0
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)"
            )

            # ── Phase 2: entity-memory join ────────────────────────────────
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_memory_map (
                    entity_id  TEXT NOT NULL,
                    memory_id  TEXT NOT NULL,
                    PRIMARY KEY (entity_id, memory_id)
                )
            """)

    # -----------------------------------------------------------------------
    # Core CRUD — memory blocks (unchanged except summary column)
    # -----------------------------------------------------------------------

    def save(self, block: MemoryBlock) -> None:
        """
        Upserts the block to SQLite and ChromaDB.
        Enriches Chroma metadata with dynamic attributes for the reranker.
        """
        with self._conn:
            self._conn.execute("""
                INSERT INTO memory_blocks (
                    id, content, summary, tag, tier, importance_score,
                    keywords, source, session_id,
                    created_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary          = excluded.summary,
                    tier             = excluded.tier,
                    last_accessed_at = excluded.last_accessed_at,
                    access_count     = excluded.access_count
            """, (
                block.id,
                block.content,
                block.summary,
                block.tag.value,
                block.tier.value,
                block.importance_score,
                json.dumps(block.keywords),
                block.source,
                block.session_id,
                block.created_at,
                block.last_accessed_at,
                block.access_count,
            ))

        # FIX 2: Only L2 and L3 are semantically indexed.
        # L1 (Rules) and L4 (Archive) excluded — prevents semantic dominance.
        if block.tier in (MemoryTier.L2_ACTIVE_RAM, MemoryTier.L3_VIRTUAL_SWAP):
            self._collection.upsert(
                ids=[block.id],
                documents=[block.content],
                metadatas=[{
                    "tag":             block.tag.value,
                    "tier":            block.tier.value,
                    "p_eff":           float(block.p_eff),
                    "last_accessed_at":float(block.last_accessed_at),
                    "keywords":        json.dumps(block.keywords),
                }],
            )
        elif block.tier in (MemoryTier.L1_HOT_CACHE, MemoryTier.L4_ARCHIVE):
            try:
                self._collection.delete(ids=[block.id])
            except Exception:
                pass

    def delete(self, block_id: str) -> None:
        """Hard delete from both stores."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM memory_blocks WHERE id = ?", (block_id,)
            )
        try:
            self._collection.delete(ids=[block_id])
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Retrieval — memory blocks (unchanged)
    # -----------------------------------------------------------------------

    def get_by_id(self, block_id: str) -> Optional[MemoryBlock]:
        row = self._conn.execute(
            "SELECT * FROM memory_blocks WHERE id = ?", (block_id,)
        ).fetchone()
        return self._row_to_block(row) if row else None

    def get_all(self, tier: Optional[MemoryTier] = None) -> List[MemoryBlock]:
        # ORDER BY created_at, id is required here: without it, SQLite makes
        # no guarantee about row order, and since this feeds directly into
        # self._l2 (an insertion-ordered dict) during _rehydrate(), an
        # unordered fetch caused retrieval results to vary non-deterministically
        # between runs with zero code changes - the same query could surface
        # different "first" candidates on ties depending on row order alone.
        # id is a secondary sort key to break ties when created_at collides.
        if tier:
            rows = self._conn.execute(
                "SELECT * FROM memory_blocks WHERE tier = ? "
                "ORDER BY created_at ASC, id ASC",
                (tier.value,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_blocks ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def get_blocks_by_ids(self, block_ids: List[str]) -> List[MemoryBlock]:
        """
        Targeted hydration — only fetches blocks that survived ranking.
        Preserves the exact order requested by the ranking engine.
        """
        if not block_ids:
            return []
        placeholders = ",".join(["?"] * len(block_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memory_blocks WHERE id IN ({placeholders})", block_ids
        ).fetchall()
        block_map = {row["id"]: self._row_to_block(row) for row in rows}
        return [block_map[bid] for bid in block_ids if bid in block_map]

    def semantic_search_metadata(
        self,
        query: str,
        tier: MemoryTier = MemoryTier.L3_VIRTUAL_SWAP,
        n_results: int = 20,
    ) -> List[Dict]:
        """
        Phase 1 of Metadata-First Reranking.
        Queries ChromaDB and returns metadata dicts only — no SQLite touch.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"tier": tier.value},
            include=["distances", "metadatas"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for idx, block_id in enumerate(results["ids"][0]):
            meta         = results["metadatas"][0][idx]
            keywords_raw = meta.get("keywords", "[]")
            try:
                keywords = json.loads(keywords_raw) if isinstance(keywords_raw, str) else keywords_raw
            except Exception:
                keywords = []

            output.append({
                "id":             block_id,
                "distance":       results["distances"][0][idx],
                "p_eff":          float(meta.get("p_eff", 0.0)),
                "last_accessed_at":float(meta.get("last_accessed_at", 0.0)),
                "keywords":       keywords,
                "tag":            meta.get("tag", ""),
            })

        logger.debug(
            "[STORE] Semantic metadata query: %r → %d matches",
            query[:40], len(output),
        )
        return output

    def tier_counts(self) -> dict[str, int]:
        """Fast aggregation of tier sizes."""
        rows = self._conn.execute(
            "SELECT tier, COUNT(*) as count FROM memory_blocks GROUP BY tier"
        ).fetchall()
        return {row["tier"]: row["count"] for row in rows}

    # -----------------------------------------------------------------------
    # Phase 2: structured facts
    # -----------------------------------------------------------------------

    def save_fact(self, fact: StructuredFact) -> None:
        """Persist a StructuredFact to the structured_facts table."""
        tb = fact.temporal_bound
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO structured_facts (
                    id, memory_id, subject, predicate, object, fact_type,
                    confidence, date_raw, earliest, latest,
                    temporal_confidence, granularity, extraction_method,
                    needs_review, has_conflict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id,
                fact.memory_id,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.fact_type.value,
                fact.confidence,
                fact.date_raw,
                tb.earliest.isoformat() if tb else None,
                tb.latest.isoformat()   if tb else None,
                tb.confidence           if tb else 0.0,
                tb.granularity.value    if tb else "unknown",
                fact.extraction_method,
                int(fact.needs_review),
                int(fact.has_conflict),
            ))

    def get_facts_for_memory(self, memory_id: str) -> List[StructuredFact]:
        """Return all structured facts for a given memory ID."""
        rows = self._conn.execute(
            "SELECT * FROM structured_facts WHERE memory_id = ? ORDER BY id ASC",
            (memory_id,)
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_all_facts(self) -> List[StructuredFact]:
        """Return all structured facts — used by the consolidator."""
        rows = self._conn.execute(
            "SELECT * FROM structured_facts ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    # -----------------------------------------------------------------------
    # Phase 2: entity registry
    # -----------------------------------------------------------------------

    def save_entity(self, entity: Entity) -> None:
        """Upsert an entity into the entities table and update the join map."""
        with self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO entities (
                    id, name, type, aliases, first_seen, last_seen,
                    memory_ids, fact_ids, reference_count, needs_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id,
                entity.name,
                entity.type.value,
                json.dumps(sorted(entity.aliases)),
                entity.first_seen.isoformat() if entity.first_seen else None,
                entity.last_seen.isoformat()  if entity.last_seen  else None,
                json.dumps(entity.memory_ids),
                json.dumps(entity.fact_ids),
                entity.reference_count,
                int(entity.needs_review),
            ))

            # Keep entity_memory_map in sync
            for mid in entity.memory_ids:
                self._conn.execute("""
                    INSERT OR IGNORE INTO entity_memory_map (entity_id, memory_id)
                    VALUES (?, ?)
                """, (entity.id, mid))

    def get_entities_for_memory(self, memory_id: str) -> List[Entity]:
        """Return all entities that reference a given memory."""
        rows = self._conn.execute("""
            SELECT e.* FROM entities e
            JOIN entity_memory_map m ON e.id = m.entity_id
            WHERE m.memory_id = ?
            ORDER BY e.id ASC
        """, (memory_id,)).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def load_entity_registry(self, registry: EntityRegistry) -> None:
        """
        Hydrate an EntityRegistry from the entities table at startup.
        Call this from MemoryController.__init__ after _rehydrate() to
        restore the entity graph across sessions.
        """
        # ORDER BY first_seen, id: same fix as get_all() above - this feeds
        # registry._entities (an insertion-ordered dict), so an unordered
        # fetch here caused entity resolution and downstream retrieval
        # candidate ordering to vary between runs with identical input data.
        rows = self._conn.execute(
            "SELECT * FROM entities ORDER BY first_seen ASC, id ASC"
        ).fetchall()
        for row in rows:
            entity = self._row_to_entity(row)
            registry._entities[entity.id] = entity
        logger.info("[STORE] Loaded %d entities into registry.", len(rows))

    # -----------------------------------------------------------------------
    # Deserialisation helpers
    # -----------------------------------------------------------------------

    def _row_to_block(self, row: sqlite3.Row) -> MemoryBlock:
        block = MemoryBlock(
            content=row["content"],
            tag=MemoryTag(row["tag"]),
            importance_score=row["importance_score"],
            keywords=json.loads(row["keywords"]),
            source=row["source"],
            session_id=row["session_id"],
            tier=MemoryTier(row["tier"]),
            summary=row["summary"] if "summary" in row.keys() else None,
        )
        object.__setattr__(block, "id",               row["id"])
        object.__setattr__(block, "created_at",       row["created_at"])
        object.__setattr__(block, "last_accessed_at", row["last_accessed_at"])
        object.__setattr__(block, "access_count",     row["access_count"])
        return block

    def _row_to_fact(self, row: sqlite3.Row) -> StructuredFact:
        tb = None
        if row["earliest"] and row["latest"]:
            try:
                gran = TemporalGranularity(row["granularity"])
            except ValueError:
                gran = TemporalGranularity.UNKNOWN
            tb = TemporalBound(
                earliest=datetime.fromisoformat(row["earliest"]),
                latest=datetime.fromisoformat(row["latest"]),
                confidence=row["temporal_confidence"],
                granularity=gran,
                raw=row["date_raw"] or "",
            )
        try:
            ftype = FactType(row["fact_type"])
        except ValueError:
            ftype = FactType.UNKNOWN

        return StructuredFact(
            id=row["id"],
            memory_id=row["memory_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            fact_type=ftype,
            confidence=row["confidence"],
            date_raw=row["date_raw"] or "",
            temporal_bound=tb,
            needs_review=bool(row["needs_review"]),
            has_conflict=bool(row["has_conflict"]),
            extraction_method=row["extraction_method"] or "heuristic",
        )

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        def _dt(val):
            return datetime.fromisoformat(val) if val else None

        try:
            etype = EntityType(row["type"])
        except ValueError:
            etype = EntityType.UNKNOWN

        entity = Entity(
            id=row["id"],
            name=row["name"],
            type=etype,
            aliases=set(json.loads(row["aliases"] or "[]")),
            first_seen=_dt(row["first_seen"]),
            last_seen=_dt(row["last_seen"]),
            memory_ids=json.loads(row["memory_ids"] or "[]"),
            fact_ids=json.loads(row["fact_ids"] or "[]"),
            reference_count=row["reference_count"],
            needs_review=bool(row["needs_review"]),
        )
        return entity

    # -----------------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Gracefully close database connections. Safe no-op if already closed.

        WinError 32 fix: ChromaDB PersistentClient on Windows holds exclusive
        file handles on chroma.sqlite3 and data_level0.bin. shutil.rmtree()
        in reset_runtime() fails with WinError 32 if those handles are open.

        PersistentClient does NOT expose stop() or close() directly.
        The correct shutdown path is client._system.stop() which halts
        the internal Chroma server and releases all file handles.
        After that, delete the reference and force GC to release any
        remaining Python-side handles before the caller deletes the directory.
        """
        # 1. Close SQLite first
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()
                self._conn = None
        except Exception:
            pass

        # 2. Stop Chroma internal server (releases OS file handles on Windows)
        try:
            if hasattr(self, "chroma_client") and self.chroma_client:
                client = self.chroma_client
                # Primary path: internal system server stop
                system = getattr(client, "_system", None)
                if system and hasattr(system, "stop"):
                    system.stop()
                # Fallback: public stop/close if exposed by this version
                else:
                    for method in ("stop", "close", "reset"):
                        fn = getattr(client, method, None)
                        if fn and callable(fn):
                            try:
                                fn()
                            except Exception:
                                pass
                            break
                # 3. Delete reference and force GC so Python releases handles
                self.chroma_client = None
                self._collection   = None
        except Exception:
            pass

        # 4. Force garbage collection — critical on Windows for file handle release
        try:
            import gc
            gc.collect()
        except Exception:
            pass