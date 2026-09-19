"""
Integration test for V2 to V3 SQLite schema migration.
Verifies migration runs cleanly against V2 golden snapshot and preserves all data.
"""

import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations, get_current_version
# Ensure migration 001 is imported and registered
import core.storage.migrations.migration_001_v2_to_v3

def test_migration_on_v2_database():
    v2_golden_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../V2_GOLDEN/longmem_almond.db"))
    assert os.path.exists(v2_golden_db), f"V2 Golden DB not found at {v2_golden_db}"

    # Work in a temporary copy
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_migrated.db")
        shutil.copy2(v2_golden_db, test_db)

        conn = sqlite3.connect(test_db)
        
        # Count pre-migration records
        pre_count = conn.execute("SELECT COUNT(*) FROM memory_blocks").fetchone()[0]
        print(f"Pre-migration memory count: {pre_count}")

        # Check version before migration
        assert get_current_version(conn) == 0

        # Apply migrations
        latest_v = apply_migrations(conn)
        assert latest_v == 2, f"Expected version 2, got {latest_v}"
        assert get_current_version(conn) == 2

        # Verify all records preserved
        post_count = conn.execute("SELECT COUNT(*) FROM memory_blocks").fetchone()[0]
        assert post_count == pre_count, f"Record count changed from {pre_count} to {post_count}"

        # Verify new columns exist and have defaults
        row = conn.execute("SELECT id, namespace_id, state, updated_at FROM memory_blocks LIMIT 1").fetchone()
        assert row is not None
        assert row[1] == "default", f"Expected default namespace, got {row[1]}"
        assert row[2] == "ACTIVE", f"Expected ACTIVE state, got {row[2]}"
        assert row[3] is not None, "updated_at should be populated"

        # Verify new V3 tables exist
        for tbl in ["namespaces", "memory_entities", "memory_facts", "events", "memory_events", "ingestion_jobs", "retrieval_traces", "audit_events"]:
            c = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            assert c >= 0

        # Re-running apply_migrations should be a safe no-op
        noop_v = apply_migrations(conn)
        assert noop_v == 2

        conn.close()
        print("PASS: test_migration_on_v2_database")

def test_migration_on_fresh_database():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "test_fresh.db")
        conn = sqlite3.connect(test_db)

        assert get_current_version(conn) == 0
        latest_v = apply_migrations(conn)
        assert latest_v == 2
        assert get_current_version(conn) == 2

        # Check tables created
        for tbl in ["namespaces", "memory_blocks", "entities", "structured_facts", "events", "ingestion_jobs", "retrieval_traces"]:
            c = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            assert c >= 0

        conn.close()
        print("PASS: test_migration_on_fresh_database")

if __name__ == "__main__":
    test_migration_on_v2_database()
    test_migration_on_fresh_database()
    print("All migration integration tests passed successfully.")
