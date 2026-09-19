"""
Integration tests for MemoryStore V3:
Validates V3 persistence, cascading delete, and rebuild_indexes invariants.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_store import MemoryStore

def test_memory_store_v3_lifecycle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "test_store.db")
        chroma_path = os.path.join(tmpdir, "test_chroma")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)

        # 1. Save V3 Block
        mb = MemoryBlock(
            content="Testing V3 durable storage with cascading delete and rebuild.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=8.5,
            namespace_id="tenant_x",
            event_time=1704067200.0, # Jan 1 2024
            state="ACTIVE"
        )
        store.save(mb)

        # Verify retrieval from SQLite
        retrieved = store.get_by_id(mb.id)
        assert retrieved is not None
        assert retrieved.content == mb.content
        assert retrieved.namespace_id == "tenant_x"
        assert retrieved.event_time == 1704067200.0
        assert retrieved.state == "ACTIVE"
        assert retrieved.importance_score == 8.5

        # Verify indexed in Chroma
        assert store._collection.count() == 1

        # 2. Rebuild Indexes from SQLite
        rebuilt = store.rebuild_indexes()
        assert rebuilt == 1
        assert store._collection.count() == 1

        # 3. Cascading Delete
        store.delete(mb.id)

        # Verify SQLite deleted
        assert store.get_by_id(mb.id) is None
        # Verify Chroma deleted
        assert store._collection.count() == 0

        # Verify audit record written
        row = store._conn.execute("SELECT * FROM audit_events WHERE target_id = ?", (mb.id,)).fetchone()
        assert row is not None
        assert row["event_type"] == "CASCADE_DELETE"

        store.close()
        print("PASS: test_memory_store_v3_lifecycle")

if __name__ == "__main__":
    test_memory_store_v3_lifecycle()
    print("All storage V3 integration tests passed successfully.")
