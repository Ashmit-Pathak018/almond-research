"""
Project Almond — Memory Store v2
Handles dual-persistence: SQLite for absolute truth, ChromaDB for semantic search.

Changes:
    - FIX 2: L1 (Hot Cache) is officially excluded from ChromaDB semantic indexing.
    - SOTA: Replaced monolithic semantic_search with semantic_search_metadata and 
            get_blocks_by_ids to support the Metadata-First Reranking pipeline.
"""

import json
import logging
import sqlite3
from typing import Optional, List, Tuple, Dict

import chromadb
from chromadb.config import Settings

from memory_block import MemoryBlock, MemoryTag, MemoryTier

logger = logging.getLogger(__name__)

class MemoryStore:
    def __init__(self, db_path: str = "longmem_almond.db"):
        self.db_path = db_path
        
        # 1. Initialize SQLite (The Source of Truth)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_sqlite()

        # 2. Initialize ChromaDB (The Semantic Index)
        self.chroma_client = chromadb.PersistentClient(path="./almond_chroma_db")
        self._collection = self.chroma_client.get_or_create_collection(
            name="almond_memory_vault",
            metadata={"hnsw:space": "cosine"} # Standardize to Cosine Distance
        )

    def _init_sqlite(self) -> None:
        """Create the SQLite schema if it doesn't exist."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_blocks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    keywords TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_id TEXT,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    access_count INTEGER NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memory_blocks(tier)")

    # -----------------------------------------------------------------------
    # Core CRUD Operations
    # -----------------------------------------------------------------------

    def save(self, block: MemoryBlock) -> None:
        """
        Upserts the block to BOTH SQLite and ChromaDB.
        Enriches Chroma metadata with dynamic attributes for the reranker.
        """
        # 1. Save to SQLite
        with self._conn:
            self._conn.execute("""
                INSERT INTO memory_blocks (
                    id, content, tag, tier, importance_score, keywords,
                    source, session_id, created_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    tier=excluded.tier,
                    last_accessed_at=excluded.last_accessed_at,
                    access_count=excluded.access_count
            """, (
                block.id,
                block.content,
                block.tag.value,
                block.tier.value,
                block.importance_score,
                json.dumps(block.keywords),
                block.source,
                block.session_id,
                block.created_at,
                block.last_accessed_at,
                block.access_count
            ))

        # 2. Save to ChromaDB 
        # FIX 2: Only L2 and L3 are semantically indexed. 
        # L1 (Rules) and L4 (Archive) are excluded to prevent semantic dominance and pollution.
        if block.tier in (MemoryTier.L2_ACTIVE_RAM, MemoryTier.L3_VIRTUAL_SWAP):
            metadata = {
                "tag": block.tag.value,
                "tier": block.tier.value,
                "p_eff": float(block.p_eff),
                "last_accessed_at": float(block.last_accessed_at),
                "keywords": json.dumps(block.keywords)
            }
            
            self._collection.upsert(
                ids=[block.id],
                documents=[block.content],
                metadatas=[metadata]
            )
        elif block.tier in (MemoryTier.L1_HOT_CACHE, MemoryTier.L4_ARCHIVE):
            # Clean up dense vectors for items that should no longer be indexed
            try:
                self._collection.delete(ids=[block.id])
            except Exception:
                pass

    def delete(self, block_id: str) -> None:
        """Hard delete from both stores."""
        with self._conn:
            self._conn.execute("DELETE FROM memory_blocks WHERE id = ?", (block_id,))
        try:
            self._collection.delete(ids=[block_id])
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Retrieval Operations
    # -----------------------------------------------------------------------

    def get_by_id(self, block_id: str) -> Optional[MemoryBlock]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memory_blocks WHERE id = ?", (block_id,))
        row = cursor.fetchone()
        return self._row_to_block(row) if row else None

    def get_all(self, tier: Optional[MemoryTier] = None) -> List[MemoryBlock]:
        cursor = self._conn.cursor()
        if tier:
            cursor.execute("SELECT * FROM memory_blocks WHERE tier = ?", (tier.value,))
        else:
            cursor.execute("SELECT * FROM memory_blocks")
        return [self._row_to_block(row) for row in cursor.fetchall()]

    def semantic_search_metadata(
        self, 
        query: str, 
        tier: MemoryTier = MemoryTier.L3_VIRTUAL_SWAP,
        n_results: int = 20
    ) -> List[Dict]:
        """
        Phase 1 of Metadata-First Reranking.
        Queries ChromaDB and returns ONLY the metadata dicts.
        Does NOT touch SQLite.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"tier": tier.value},
            include=["distances", "metadatas"] 
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        raw_results = []
        for idx, block_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][idx]
            
            keywords_raw = meta.get("keywords", "[]")
            if isinstance(keywords_raw, str):
                try:
                    keywords = json.loads(keywords_raw)
                except Exception:
                    keywords = []
            else:
                keywords = keywords_raw

            raw_results.append({
                "id": block_id,
                "distance": results["distances"][0][idx],
                "p_eff": float(meta.get("p_eff", 0.0)),
                "last_accessed_at": float(meta.get("last_accessed_at", 0.0)),
                "keywords": keywords,
                "tag": meta.get("tag", "")
            })

        logger.debug(f"[STORE] Semantic metadata query: '{query[:30]}...' → {len(raw_results)} matches")
        return raw_results

    def get_blocks_by_ids(self, block_ids: List[str]) -> List[MemoryBlock]:
        """
        Phase 3 of Metadata-First Reranking.
        Targeted hydration of ONLY the blocks that survived the optimizer.
        """
        if not block_ids:
            return []
            
        placeholders = ",".join(["?"] * len(block_ids))
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM memory_blocks WHERE id IN ({placeholders})", block_ids)
        rows = cursor.fetchall()
        
        block_map = {row["id"]: self._row_to_block(row) for row in rows}
        
        # Return in the exact order requested by the reranker
        return [block_map[bid] for bid in block_ids if bid in block_map]

    def tier_counts(self) -> dict[str, int]:
        """Fast aggregation of tier sizes."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT tier, COUNT(*) as count FROM memory_blocks GROUP BY tier")
        return {row["tier"]: row["count"] for row in cursor.fetchall()}

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _row_to_block(self, row: sqlite3.Row) -> MemoryBlock:
        block = MemoryBlock(
            content=row["content"],
            tag=MemoryTag(row["tag"]),
            importance_score=row["importance_score"],
            keywords=json.loads(row["keywords"]),
            source=row["source"],
            session_id=row["session_id"],
            tier=MemoryTier(row["tier"])
        )
        object.__setattr__(block, "id", row["id"])
        object.__setattr__(block, "created_at", row["created_at"])
        object.__setattr__(block, "last_accessed_at", row["last_accessed_at"])
        object.__setattr__(block, "access_count", row["access_count"])
        return block
    
    def close(self):
        """
            Gracefully close database connections.
            Safe no-op if already closed.
        """
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()

        except Exception:
            pass