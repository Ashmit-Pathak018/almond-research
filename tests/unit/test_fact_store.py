"""
Unit tests for core/knowledge/fact_store.py
Covers: fact state management, supersession tracking, historical preservation,
        current truth resolution, temporal validity, memory links,
        namespace isolation, and CASE-005 golden case.
"""

import os
import sys
import sqlite3
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
from core.knowledge.fact_store import (
    FactStore, FactRecord,
    FACT_STATE_VERIFIED, FACT_STATE_INFERRED,
    FACT_STATE_CONFLICTING, FACT_STATE_SUPERSEDED,
)


def _make_db():
    """Create a fresh in-memory V3 database with all migrations applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


def _insert_memory(conn, mid, content="test", event_time=None):
    """Helper to insert a memory for FK constraints."""
    conn.execute("""
        INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state,
            importance_score, keywords, source, event_time,
            created_at, updated_at, last_accessed_at, access_count)
        VALUES (?, 'default', ?, 'PROJECT_FACT', 'L2_ACTIVE_RAM', 'ACTIVE',
            5.0, '[]', 'user', ?, 1000, 1000, 1000, 1)
    """, (mid, content, event_time))


# ------------------------------------------------------------------
# Fact Creation & Basic States
# ------------------------------------------------------------------

def test_fact_creation():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem_fact")
    fact = store.create_fact(
        namespace_id="default",
        memory_id="mem_fact",
        subject="User",
        predicate="likes",
        object_val="apples",
        state=FACT_STATE_VERIFIED,
    )

    assert fact.fact_id is not None
    assert fact.subject == "User"
    assert fact.predicate == "likes"
    assert fact.object == "apples"
    assert fact.state == FACT_STATE_VERIFIED
    assert fact.memory_id == "mem_fact"

    fetched = store.get_fact(fact.fact_id)
    assert fetched is not None
    assert fetched.subject == "User"
    conn.close()
    print("PASS: test_fact_creation")


def test_fact_states():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem_state")
    
    # Inferred
    f1 = store.create_fact(
        namespace_id="default", memory_id="mem_state",
        subject="S", predicate="P", object_val="O1", state=FACT_STATE_INFERRED
    )
    assert f1.state == FACT_STATE_INFERRED

    # Update to Verified
    store.update_state(f1.fact_id, FACT_STATE_VERIFIED)
    assert store.get_fact(f1.fact_id).state == FACT_STATE_VERIFIED

    # Conflicting
    store.mark_conflicting(f1.fact_id)
    assert store.get_fact(f1.fact_id).state == FACT_STATE_CONFLICTING
    conn.close()
    print("PASS: test_fact_states")


# ------------------------------------------------------------------
# Supersession & Historical Preservation
# ------------------------------------------------------------------

def test_supersession_logic():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem_old")
    _insert_memory(conn, "mem_new")

    f_old = store.create_fact(
        namespace_id="default", memory_id="mem_old",
        subject="User", predicate="lives_in", object_val="Seattle",
        created_at=1000.0, valid_from=0.0
    )

    f_new = store.create_fact(
        namespace_id="default", memory_id="mem_new",
        subject="User", predicate="lives_in", object_val="Boston",
        created_at=2000.0, valid_from=2000.0
    )

    # Supersede old fact with new fact
    res = store.supersede_fact(f_old.fact_id, f_new.fact_id)
    assert res is True

    # Verify old fact is SUPERSEDED, has pointer, and has valid_to closed
    old_fetched = store.get_fact(f_old.fact_id)
    assert old_fetched.state == FACT_STATE_SUPERSEDED
    assert old_fetched.superseded_by == f_new.fact_id
    assert old_fetched.valid_to is not None
    assert old_fetched.valid_to > 0.0

    # Verify new fact is still VERIFIED
    new_fetched = store.get_fact(f_new.fact_id)
    assert new_fetched.state == FACT_STATE_VERIFIED
    assert new_fetched.superseded_by is None
    conn.close()
    print("PASS: test_supersession_logic")


# ------------------------------------------------------------------
# Current Truth Resolution
# ------------------------------------------------------------------

def test_current_truth_resolution():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem1")
    _insert_memory(conn, "mem2")
    _insert_memory(conn, "mem3")

    f1 = store.create_fact(namespace_id="default", memory_id="mem1",
                           subject="User", predicate="role", object_val="Junior", created_at=100)
    f2 = store.create_fact(namespace_id="default", memory_id="mem2",
                           subject="User", predicate="role", object_val="Senior", created_at=200)
    f3 = store.create_fact(namespace_id="default", memory_id="mem3",
                           subject="User", predicate="role", object_val="Lead", created_at=300)

    # Supersede chain: f1 -> f2 -> f3
    store.supersede_fact(f1.fact_id, f2.fact_id)
    store.supersede_fact(f2.fact_id, f3.fact_id)

    # get_current_truth should return the latest VERIFIED fact (f3)
    truth = store.get_current_truth("default", "User", "role")
    assert truth is not None
    assert truth.object == "Lead"
    assert truth.state == FACT_STATE_VERIFIED

    # get_facts_by_subject_predicate should ONLY return f3 by default
    active = store.get_facts_by_subject_predicate("default", "User", "role")
    assert len(active) == 1
    assert active[0].object == "Lead"

    # get_historical_facts should return all 3
    history = store.get_historical_facts("default", "User", "role")
    assert len(history) == 3
    assert [h.object for h in history] == ["Junior", "Senior", "Lead"]
    conn.close()
    print("PASS: test_current_truth_resolution")


# ------------------------------------------------------------------
# CASE-005: Contradiction & Superseded Truth (Golden Case)
# ------------------------------------------------------------------

def test_case_005_superseded_truth():
    """
    CASE-005: User lived in Seattle in 2022, moved to Boston in 2024.
    Expected: Seattle -> SUPERSEDED, Boston -> VERIFIED.
    Current ground truth should resolve to Boston.
    """
    conn = _make_db()
    store = FactStore(conn)

    T_2022 = 1640995200.0
    T_2024 = 1714521600.0

    _insert_memory(conn, "mem_city_old", "I live in Seattle", T_2022)
    _insert_memory(conn, "mem_city_new", "I relocated to Boston", T_2024)

    # Create the facts
    f_old = store.create_fact(
        namespace_id="default", memory_id="mem_city_old",
        subject="user", predicate="lives_in", object_val="Seattle",
        state=FACT_STATE_VERIFIED, created_at=T_2022
    )

    f_new = store.create_fact(
        namespace_id="default", memory_id="mem_city_new",
        subject="user", predicate="lives_in", object_val="Boston",
        state=FACT_STATE_VERIFIED, created_at=T_2024
    )

    # Consolidator logic would trigger supersession
    store.supersede_fact(f_old.fact_id, f_new.fact_id)

    # Verify CASE-005 conditions
    old_fetched = store.get_fact(f_old.fact_id)
    new_fetched = store.get_fact(f_new.fact_id)

    assert old_fetched.state == FACT_STATE_SUPERSEDED, \
        f"CASE-005 FAIL: Expected old fact to be SUPERSEDED, got {old_fetched.state}"
    assert new_fetched.state == FACT_STATE_VERIFIED, \
        f"CASE-005 FAIL: Expected new fact to be VERIFIED, got {new_fetched.state}"

    # Verify ground truth resolution
    truth = store.get_current_truth("default", "user", "lives_in")
    assert truth is not None, "CASE-005 FAIL: Current truth is None"
    assert truth.object == "Boston", \
        f"CASE-005 FAIL: Expected ground truth 'Boston', got '{truth.object}'"

    conn.close()
    print("PASS: test_case_005_superseded_truth (CASE-005 GOLDEN)")


# ------------------------------------------------------------------
# Provenance & Memory Links
# ------------------------------------------------------------------

def test_provenance_and_memory_links():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem_link_1")
    _insert_memory(conn, "mem_link_2")

    f1 = store.create_fact(namespace_id="default", memory_id="mem_link_1",
                           subject="A", predicate="B", object_val="C")

    # Manually link second memory
    store.link_memory("mem_link_2", f1.fact_id)

    # Verify memory->fact
    facts = store.get_facts_for_memory("mem_link_1")
    assert len(facts) == 1
    assert facts[0].fact_id == f1.fact_id

    # Verify fact->memory
    mem_ids = store.get_memories_for_fact(f1.fact_id)
    assert len(mem_ids) == 2
    assert "mem_link_1" in mem_ids
    assert "mem_link_2" in mem_ids
    conn.close()
    print("PASS: test_provenance_and_memory_links")


# ------------------------------------------------------------------
# Temporal Validity
# ------------------------------------------------------------------

def test_temporal_validity():
    conn = _make_db()
    store = FactStore(conn)

    _insert_memory(conn, "mem_valid")
    fact = store.create_fact(
        namespace_id="default", memory_id="mem_valid",
        subject="Project", predicate="status", object_val="Active",
        valid_from=1000.0, valid_to=2000.0
    )

    fetched = store.get_fact(fact.fact_id)
    assert fetched.valid_from == 1000.0
    assert fetched.valid_to == 2000.0
    conn.close()
    print("PASS: test_temporal_validity")


# ------------------------------------------------------------------
# Namespace Isolation
# ------------------------------------------------------------------

def test_namespace_isolation():
    conn = _make_db()
    store = FactStore(conn)

    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns1', 'NS 1', 1000)")
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns2', 'NS 2', 1000)")
    _insert_memory(conn, "mem_ns1")
    conn.execute("UPDATE memory_blocks SET namespace_id = 'ns1' WHERE id = 'mem_ns1'")
    _insert_memory(conn, "mem_ns2")
    conn.execute("UPDATE memory_blocks SET namespace_id = 'ns2' WHERE id = 'mem_ns2'")

    store.create_fact(namespace_id="ns1", memory_id="mem_ns1", subject="U", predicate="P", object_val="O1")
    store.create_fact(namespace_id="ns2", memory_id="mem_ns2", subject="U", predicate="P", object_val="O2")

    t1 = store.get_current_truth("ns1", "U", "P")
    assert t1.object == "O1"

    t2 = store.get_current_truth("ns2", "U", "P")
    assert t2.object == "O2"

    list_ns1 = store.list_facts("ns1")
    assert len(list_ns1) == 1
    assert list_ns1[0].object == "O1"
    conn.close()
    print("PASS: test_namespace_isolation")


if __name__ == "__main__":
    test_fact_creation()
    test_fact_states()
    test_supersession_logic()
    test_current_truth_resolution()
    test_case_005_superseded_truth()
    test_provenance_and_memory_links()
    test_temporal_validity()
    test_namespace_isolation()
    print("\nAll FactStore unit tests passed successfully.")
