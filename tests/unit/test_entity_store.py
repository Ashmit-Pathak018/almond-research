"""
Unit tests for core/knowledge/entity_store.py
Covers: entity CRUD, alias resolution, memory-entity links, namespace isolation,
        deletion cascade, persistence, and CASE-002 golden case.
"""

import os
import sys
import sqlite3
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
from core.knowledge.entity_store import EntityStore, EntityRecord, AliasResolution


def _make_db():
    """Create a fresh in-memory V3 database with all migrations applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


# ------------------------------------------------------------------
# Entity Creation
# ------------------------------------------------------------------

def test_entity_creation():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(
        namespace_id="default",
        canonical_name="Robert Vance",
        entity_type="PERSON",
        summary="Software engineer on infrastructure team",
    )

    assert entity.entity_id is not None
    assert entity.canonical_name == "Robert Vance"
    assert entity.entity_type == "PERSON"
    assert entity.namespace_id == "default"
    assert entity.summary == "Software engineer on infrastructure team"
    assert entity.created_at > 0

    # Verify persisted in SQLite
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity.entity_id,)).fetchone()
    assert row is not None
    assert row["name"] == "Robert Vance"
    conn.close()
    print("PASS: test_entity_creation")


# ------------------------------------------------------------------
# Canonical Lookup
# ------------------------------------------------------------------

def test_canonical_lookup():
    conn = _make_db()
    store = EntityStore(conn)

    store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    found = store.get_entity_by_name("Robert Vance", "default")
    assert found is not None
    assert found.canonical_name == "Robert Vance"

    # Not found in different namespace
    not_found = store.get_entity_by_name("Robert Vance", "other_ns")
    assert not_found is None
    conn.close()
    print("PASS: test_canonical_lookup")


# ------------------------------------------------------------------
# Alias Management
# ------------------------------------------------------------------

def test_alias_add_and_lookup():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    # Add aliases
    alias1 = store.add_alias(entity.entity_id, "Bob")
    alias2 = store.add_alias(entity.entity_id, "Bobby V")
    assert alias1 is not None
    assert alias2 is not None
    assert alias1.alias_name == "bob"  # normalized

    # Get all aliases (should include canonical + manual aliases)
    aliases = store.get_aliases(entity.entity_id)
    alias_names = [a.alias_name for a in aliases]
    assert "bob" in alias_names
    assert "bobby v" in alias_names
    assert "robert vance" in alias_names  # canonical auto-registered
    conn.close()
    print("PASS: test_alias_add_and_lookup")


# ------------------------------------------------------------------
# Alias → Canonical Entity Resolution
# ------------------------------------------------------------------

def test_alias_resolution_canonical():
    conn = _make_db()
    store = EntityStore(conn)

    store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    result = store.resolve_alias("Robert Vance", "default")
    assert result is not None
    assert result.entity.canonical_name == "Robert Vance"
    assert result.confidence == 1.0
    assert result.resolution_type == "canonical"
    conn.close()
    print("PASS: test_alias_resolution_canonical")


def test_alias_resolution_alias():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")
    store.add_alias(entity.entity_id, "Bob")

    result = store.resolve_alias("Bob", "default")
    assert result is not None
    assert result.entity.canonical_name == "Robert Vance"
    assert result.confidence == 0.95
    assert result.resolution_type == "alias_exact"
    conn.close()
    print("PASS: test_alias_resolution_alias")


def test_alias_resolution_no_match():
    conn = _make_db()
    store = EntityStore(conn)

    store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    result = store.resolve_alias("Alice Smith", "default")
    assert result is None
    conn.close()
    print("PASS: test_alias_resolution_no_match")


# ------------------------------------------------------------------
# Confidence Value
# ------------------------------------------------------------------

def test_confidence_values():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")
    store.add_alias(entity.entity_id, "Bob")

    # Canonical → 1.0
    r1 = store.resolve_alias("Robert Vance", "default")
    assert r1.confidence == 1.0

    # Exact alias → 0.95
    r2 = store.resolve_alias("Bob", "default")
    assert r2.confidence == 0.95

    # Case-insensitive canonical → still 1.0 (canonical match is case-insensitive)
    r3 = store.resolve_alias("robert vance", "default")
    assert r3.confidence == 1.0
    conn.close()
    print("PASS: test_confidence_values")


# ------------------------------------------------------------------
# Duplicate / Normalized Aliases
# ------------------------------------------------------------------

def test_duplicate_alias():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    a1 = store.add_alias(entity.entity_id, "Bob")
    a2 = store.add_alias(entity.entity_id, "Bob")  # duplicate
    a3 = store.add_alias(entity.entity_id, "  Bob  ")  # normalized duplicate
    a4 = store.add_alias(entity.entity_id, "BOB")  # case duplicate

    assert a1 is not None
    assert a2 is not None  # returns existing
    assert a3 is not None
    assert a4 is not None

    # Should only have 2 alias rows: "robert vance" (canonical) + "bob"
    aliases = store.get_aliases(entity.entity_id)
    alias_names = [a.alias_name for a in aliases]
    assert alias_names.count("bob") == 1
    conn.close()
    print("PASS: test_duplicate_alias")


# ------------------------------------------------------------------
# Memory ↔ Entity Relationship
# ------------------------------------------------------------------

def test_memory_entity_link():
    conn = _make_db()
    store = EntityStore(conn)

    # Create a memory in memory_blocks first
    conn.execute("""
        INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
            importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
        VALUES (?, 'default', 'Robert Vance delivered the security audit.', 'PROJECT_FACT',
            'L2_ACTIVE_RAM', 'ACTIVE', 7.0, '[]', 'user', 1709900000.0, 1709900000.0, 1709900000.0, 1)
    """, ("mem_robert_vance",))

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")

    # Link
    result = store.link_memory("mem_robert_vance", entity.entity_id, confidence=0.95)
    assert result is True

    # Get memories for entity
    memories = store.get_memories_for_entity(entity.entity_id)
    assert len(memories) == 1
    assert memories[0]["memory_id"] == "mem_robert_vance"
    assert memories[0]["confidence"] == 0.95

    # Get entities for memory
    entities = store.get_entities_for_memory("mem_robert_vance")
    assert len(entities) == 1
    assert entities[0].canonical_name == "Robert Vance"
    conn.close()
    print("PASS: test_memory_entity_link")


# ------------------------------------------------------------------
# Entity → Associated Memories
# ------------------------------------------------------------------

def test_entity_associated_memories():
    conn = _make_db()
    store = EntityStore(conn)

    # Create two memories
    for mid, content in [
        ("mem_rv_1", "Robert Vance designed the architecture."),
        ("mem_rv_2", "Robert Vance reviewed the pull request."),
    ]:
        conn.execute("""
            INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
                importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
            VALUES (?, 'default', ?, 'PROJECT_FACT', 'L2_ACTIVE_RAM', 'ACTIVE',
                7.0, '[]', 'user', 1709900000.0, 1709900000.0, 1709900000.0, 1)
        """, (mid, content))

    entity = store.create_entity(namespace_id="default", canonical_name="Robert Vance", entity_type="PERSON")
    store.link_memory("mem_rv_1", entity.entity_id, confidence=0.90)
    store.link_memory("mem_rv_2", entity.entity_id, confidence=0.85)

    memories = store.get_memories_for_entity(entity.entity_id)
    assert len(memories) == 2
    mem_ids = [m["memory_id"] for m in memories]
    assert "mem_rv_1" in mem_ids
    assert "mem_rv_2" in mem_ids
    conn.close()
    print("PASS: test_entity_associated_memories")


# ------------------------------------------------------------------
# Namespace Isolation
# ------------------------------------------------------------------

def test_namespace_isolation():
    conn = _make_db()
    store = EntityStore(conn)

    # Create namespace entries
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns_a', 'NS A', 1000)")
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns_b', 'NS B', 1000)")

    e_a = store.create_entity(namespace_id="ns_a", canonical_name="Alice", entity_type="PERSON")
    e_b = store.create_entity(namespace_id="ns_b", canonical_name="Alice", entity_type="PERSON")

    store.add_alias(e_a.entity_id, "Ali")
    store.add_alias(e_b.entity_id, "Ali")

    # Resolution in ns_a should return ns_a's entity
    r_a = store.resolve_alias("Ali", "ns_a")
    assert r_a is not None
    assert r_a.entity.entity_id == e_a.entity_id

    # Resolution in ns_b should return ns_b's entity
    r_b = store.resolve_alias("Ali", "ns_b")
    assert r_b is not None
    assert r_b.entity.entity_id == e_b.entity_id

    # Listing is namespace-scoped
    assert len(store.list_entities("ns_a")) == 1
    assert len(store.list_entities("ns_b")) == 1
    conn.close()
    print("PASS: test_namespace_isolation")


# ------------------------------------------------------------------
# Deletion Cleanup
# ------------------------------------------------------------------

def test_deletion_cleanup():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(namespace_id="default", canonical_name="To Delete", entity_type="CONCEPT")
    store.add_alias(entity.entity_id, "delete me")

    # Create and link a memory
    conn.execute("""
        INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
            importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
        VALUES ('mem_del', 'default', 'Test deletion', 'SMALL_TALK', 'L2_ACTIVE_RAM', 'ACTIVE',
            3.0, '[]', 'user', 1000, 1000, 1000, 1)
    """)
    store.link_memory("mem_del", entity.entity_id)

    # Verify everything exists
    assert store.get_entity(entity.entity_id) is not None
    assert len(store.get_aliases(entity.entity_id)) >= 1
    assert len(store.get_memories_for_entity(entity.entity_id)) == 1

    # Delete entity
    deleted = store.delete_entity(entity.entity_id)
    assert deleted is True

    # Verify all cleaned up
    assert store.get_entity(entity.entity_id) is None
    assert len(store.get_aliases(entity.entity_id)) == 0
    assert len(store.get_memories_for_entity(entity.entity_id)) == 0

    # Memory itself still exists (not cascaded from entity deletion)
    row = conn.execute("SELECT * FROM memory_blocks WHERE id = 'mem_del'").fetchone()
    assert row is not None
    conn.close()
    print("PASS: test_deletion_cleanup")


# ------------------------------------------------------------------
# Persistence After Reopening SQLite
# ------------------------------------------------------------------

def test_persistence_after_reopen():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_entity_persist.db")

        # First session — create entity + alias
        conn1 = sqlite3.connect(db_path)
        conn1.row_factory = sqlite3.Row
        conn1.execute("PRAGMA foreign_keys = ON;")
        apply_migrations(conn1)
        store1 = EntityStore(conn1)

        entity = store1.create_entity(
            namespace_id="default",
            canonical_name="Persistent Entity",
            entity_type="PROJECT",
        )
        store1.add_alias(entity.entity_id, "PE")
        entity_id = entity.entity_id
        conn1.close()

        # Second session — reopen and verify
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA foreign_keys = ON;")
        apply_migrations(conn2)
        store2 = EntityStore(conn2)

        reloaded = store2.get_entity(entity_id)
        assert reloaded is not None
        assert reloaded.canonical_name == "Persistent Entity"

        resolution = store2.resolve_alias("PE", "default")
        assert resolution is not None
        assert resolution.entity.canonical_name == "Persistent Entity"

        aliases = store2.get_aliases(entity_id)
        alias_names = [a.alias_name for a in aliases]
        assert "pe" in alias_names
        conn2.close()
    print("PASS: test_persistence_after_reopen")


# ------------------------------------------------------------------
# CASE-002: Entity Alias Resolution (Golden Case)
# ------------------------------------------------------------------

def test_case_002_entity_alias_resolution():
    """
    CASE-002 from golden specification:
    Input: "What did Bob do on the project?"
    Precondition: Entity "Robert Vance" has alias "Bob".
    Expected: The canonical entity is resolved and associated memories can be retrieved.
    """
    conn = _make_db()
    store = EntityStore(conn)

    # Setup precondition: create entity "Robert Vance" with alias "Bob"
    entity = store.create_entity(
        namespace_id="default",
        canonical_name="Robert Vance",
        entity_type="PERSON",
    )
    store.add_alias(entity.entity_id, "Bob")

    # Create the precondition memory
    conn.execute("""
        INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
            importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
        VALUES ('mem_robert_vance', 'default',
            'Robert Vance delivered the client infrastructure security audit report on time.',
            'PROJECT_FACT', 'L2_ACTIVE_RAM', 'ACTIVE',
            7.0, '[]', 'user', 1709900000.0, 1709900000.0, 1709900000.0, 1)
    """)
    store.link_memory("mem_robert_vance", entity.entity_id, confidence=0.95)

    # Resolve "Bob" → should find "Robert Vance"
    resolution = store.resolve_alias("Bob", "default")
    assert resolution is not None, "CASE-002 FAIL: 'Bob' did not resolve to any entity"
    assert resolution.entity.canonical_name == "Robert Vance", \
        f"CASE-002 FAIL: Expected 'Robert Vance', got '{resolution.entity.canonical_name}'"
    assert resolution.confidence >= 0.90, \
        f"CASE-002 FAIL: Confidence {resolution.confidence} is too low"

    # Verify associated memories can be retrieved
    memories = store.get_memories_for_entity(resolution.entity.entity_id)
    assert len(memories) >= 1, "CASE-002 FAIL: No memories found for resolved entity"
    mem_ids = [m["memory_id"] for m in memories]
    assert "mem_robert_vance" in mem_ids, "CASE-002 FAIL: Expected memory not linked"

    conn.close()
    print("PASS: test_case_002_entity_alias_resolution (CASE-002 GOLDEN)")


# ------------------------------------------------------------------
# Update Entity
# ------------------------------------------------------------------

def test_update_entity():
    conn = _make_db()
    store = EntityStore(conn)

    entity = store.create_entity(
        namespace_id="default",
        canonical_name="Old Name",
        entity_type="PERSON",
    )

    updated = store.update_entity(
        entity.entity_id,
        canonical_name="New Name",
        summary="Updated summary",
    )

    assert updated is not None
    assert updated.canonical_name == "New Name"
    assert updated.summary == "Updated summary"

    # Old name should still be resolvable via alias
    # (canonical name change adds old as implicit alias via auto-registered canonical)
    # New name also resolvable
    r_new = store.resolve_alias("New Name", "default")
    assert r_new is not None
    assert r_new.entity.canonical_name == "New Name"
    conn.close()
    print("PASS: test_update_entity")


# ------------------------------------------------------------------
# Unlink Memory
# ------------------------------------------------------------------

def test_unlink_memory():
    conn = _make_db()
    store = EntityStore(conn)

    conn.execute("""
        INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
            importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
        VALUES ('mem_unlink', 'default', 'Test unlink', 'SMALL_TALK', 'L2_ACTIVE_RAM', 'ACTIVE',
            3.0, '[]', 'user', 1000, 1000, 1000, 1)
    """)

    entity = store.create_entity(namespace_id="default", canonical_name="Unlink Test", entity_type="CONCEPT")
    store.link_memory("mem_unlink", entity.entity_id)
    assert len(store.get_memories_for_entity(entity.entity_id)) == 1

    store.unlink_memory("mem_unlink", entity.entity_id)
    assert len(store.get_memories_for_entity(entity.entity_id)) == 0
    conn.close()
    print("PASS: test_unlink_memory")


if __name__ == "__main__":
    test_entity_creation()
    test_canonical_lookup()
    test_alias_add_and_lookup()
    test_alias_resolution_canonical()
    test_alias_resolution_alias()
    test_alias_resolution_no_match()
    test_confidence_values()
    test_duplicate_alias()
    test_memory_entity_link()
    test_entity_associated_memories()
    test_namespace_isolation()
    test_deletion_cleanup()
    test_persistence_after_reopen()
    test_case_002_entity_alias_resolution()
    test_update_entity()
    test_unlink_memory()
    print("\nAll EntityStore unit tests passed successfully.")
