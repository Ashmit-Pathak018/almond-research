"""
Unit tests for core/knowledge/timeline_store.py
Covers: event CRUD, event_time ordering, out-of-order ingestion, temporal queries,
        memory-event links, persistence, deletion cleanup, and CASE-009 golden case.
"""

import os
import sys
import sqlite3
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
from core.knowledge.timeline_store import (
    TimelineStore, EventRecord,
    EVENT_TYPE_POINT, EVENT_TYPE_INTERVAL, EVENT_TYPE_ORDERED, EVENT_TYPE_UNKNOWN_TIME,
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
        VALUES (?, 'default', ?, 'EPISODIC', 'L2_ACTIVE_RAM', 'ACTIVE',
            5.0, '[]', 'user', ?, 1000, 1000, 1000, 1)
    """, (mid, content, event_time))


# ------------------------------------------------------------------
# Point Event Creation
# ------------------------------------------------------------------

def test_point_event_creation():
    conn = _make_db()
    store = TimelineStore(conn)

    event = store.create_event(
        namespace_id="default",
        title="Deployed v1.0",
        event_type=EVENT_TYPE_POINT,
        start_time=1704067200.0,  # Jan 1 2024
        description="Initial production deployment",
        confidence=0.95,
    )

    assert event.event_id is not None
    assert event.title == "Deployed v1.0"
    assert event.event_type == EVENT_TYPE_POINT
    assert event.start_time == 1704067200.0
    assert event.end_time is None
    assert event.confidence == 0.95

    # Verify persisted
    fetched = store.get_event(event.event_id)
    assert fetched is not None
    assert fetched.title == "Deployed v1.0"
    conn.close()
    print("PASS: test_point_event_creation")


# ------------------------------------------------------------------
# Interval Event Creation
# ------------------------------------------------------------------

def test_interval_event_creation():
    conn = _make_db()
    store = TimelineStore(conn)

    event = store.create_event(
        namespace_id="default",
        title="Sprint 1",
        event_type=EVENT_TYPE_INTERVAL,
        start_time=1704067200.0,  # Jan 1 2024
        end_time=1705276800.0,    # Jan 15 2024
        description="Two-week sprint",
    )

    assert event.event_type == EVENT_TYPE_INTERVAL
    assert event.start_time == 1704067200.0
    assert event.end_time == 1705276800.0
    conn.close()
    print("PASS: test_interval_event_creation")


def test_interval_requires_both_times():
    conn = _make_db()
    store = TimelineStore(conn)

    try:
        store.create_event(
            namespace_id="default",
            title="Bad Interval",
            event_type=EVENT_TYPE_INTERVAL,
            start_time=1704067200.0,
            # Missing end_time
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "start_time and end_time" in str(e)
    conn.close()
    print("PASS: test_interval_requires_both_times")


# ------------------------------------------------------------------
# Event Time Ordering
# ------------------------------------------------------------------

def test_event_time_ordering():
    conn = _make_db()
    store = TimelineStore(conn)

    # Insert in non-chronological order
    store.create_event(namespace_id="default", title="March Event",
                       event_type=EVENT_TYPE_POINT, start_time=1709251200.0)
    store.create_event(namespace_id="default", title="January Event",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)
    store.create_event(namespace_id="default", title="May Event",
                       event_type=EVENT_TYPE_POINT, start_time=1714521600.0)

    timeline = store.get_chronological("default")
    assert len(timeline) == 3
    assert timeline[0].title == "January Event"
    assert timeline[1].title == "March Event"
    assert timeline[2].title == "May Event"
    conn.close()
    print("PASS: test_event_time_ordering")


# ------------------------------------------------------------------
# Start/End Interval Handling
# ------------------------------------------------------------------

def test_start_end_interval():
    conn = _make_db()
    store = TimelineStore(conn)

    event = store.create_event(
        namespace_id="default",
        title="Conference",
        event_type=EVENT_TYPE_INTERVAL,
        start_time=1704067200.0,
        end_time=1704326400.0,
    )

    fetched = store.get_event(event.event_id)
    assert fetched.start_time == 1704067200.0
    assert fetched.end_time == 1704326400.0
    assert fetched.event_type == EVENT_TYPE_INTERVAL
    conn.close()
    print("PASS: test_start_end_interval")


# ------------------------------------------------------------------
# Out-of-Order Ingestion (CASE-009 core logic)
# ------------------------------------------------------------------

def test_out_of_order_ingestion():
    """
    Ingest March event first, then January event.
    Timeline must return January → March based on event_time.
    """
    conn = _make_db()
    store = TimelineStore(conn)

    # Ingest March FIRST
    store.create_event(namespace_id="default", title="March Event",
                       event_type=EVENT_TYPE_POINT, start_time=1709251200.0,
                       created_at=1000.0)  # Earlier ingestion time
    # Ingest January SECOND
    store.create_event(namespace_id="default", title="January Event",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0,
                       created_at=2000.0)  # Later ingestion time

    timeline = store.get_chronological("default")
    assert len(timeline) == 2
    assert timeline[0].title == "January Event", \
        f"Expected 'January Event' first, got '{timeline[0].title}'"
    assert timeline[1].title == "March Event", \
        f"Expected 'March Event' second, got '{timeline[1].title}'"
    conn.close()
    print("PASS: test_out_of_order_ingestion")


# ------------------------------------------------------------------
# Memory-Event Relationship
# ------------------------------------------------------------------

def test_memory_event_relationship():
    conn = _make_db()
    store = TimelineStore(conn)

    _insert_memory(conn, "mem_deploy_v1", "Deployed v1.0", 1704067200.0)

    event = store.create_event(
        namespace_id="default", title="v1.0 Release",
        event_type=EVENT_TYPE_POINT, start_time=1704067200.0,
    )

    # Link
    result = store.link_memory("mem_deploy_v1", event.event_id)
    assert result is True

    # Get memories for event
    memories = store.get_memories_for_event(event.event_id)
    assert len(memories) == 1
    assert memories[0] == "mem_deploy_v1"

    # Get events for memory
    events = store.get_events_for_memory("mem_deploy_v1")
    assert len(events) == 1
    assert events[0].title == "v1.0 Release"
    conn.close()
    print("PASS: test_memory_event_relationship")


# ------------------------------------------------------------------
# Chronological Query
# ------------------------------------------------------------------

def test_chronological_query():
    conn = _make_db()
    store = TimelineStore(conn)

    times = [
        ("Alpha", 1704067200.0),
        ("Beta", 1706745600.0),
        ("Gamma", 1709251200.0),
        ("Delta", 1711929600.0),
    ]
    for title, t in times:
        store.create_event(namespace_id="default", title=title,
                           event_type=EVENT_TYPE_POINT, start_time=t)

    all_events = store.get_chronological("default")
    assert [e.title for e in all_events] == ["Alpha", "Beta", "Gamma", "Delta"]
    conn.close()
    print("PASS: test_chronological_query")


# ------------------------------------------------------------------
# Before / After Filtering
# ------------------------------------------------------------------

def test_before_filter():
    conn = _make_db()
    store = TimelineStore(conn)

    store.create_event(namespace_id="default", title="Jan",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)
    store.create_event(namespace_id="default", title="Mar",
                       event_type=EVENT_TYPE_POINT, start_time=1709251200.0)
    store.create_event(namespace_id="default", title="May",
                       event_type=EVENT_TYPE_POINT, start_time=1714521600.0)

    before_march = store.get_before("default", 1709251200.0)
    assert len(before_march) == 1
    assert before_march[0].title == "Jan"
    conn.close()
    print("PASS: test_before_filter")


def test_after_filter():
    conn = _make_db()
    store = TimelineStore(conn)

    store.create_event(namespace_id="default", title="Jan",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)
    store.create_event(namespace_id="default", title="Mar",
                       event_type=EVENT_TYPE_POINT, start_time=1709251200.0)
    store.create_event(namespace_id="default", title="May",
                       event_type=EVENT_TYPE_POINT, start_time=1714521600.0)

    after_march = store.get_after("default", 1709251200.0)
    assert len(after_march) == 1
    assert after_march[0].title == "May"
    conn.close()
    print("PASS: test_after_filter")


def test_between_filter():
    conn = _make_db()
    store = TimelineStore(conn)

    store.create_event(namespace_id="default", title="Jan",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)
    store.create_event(namespace_id="default", title="Mar",
                       event_type=EVENT_TYPE_POINT, start_time=1709251200.0)
    store.create_event(namespace_id="default", title="May",
                       event_type=EVENT_TYPE_POINT, start_time=1714521600.0)

    between = store.get_between("default", 1706000000.0, 1712000000.0)
    assert len(between) == 1
    assert between[0].title == "Mar"
    conn.close()
    print("PASS: test_between_filter")


# ------------------------------------------------------------------
# Overlap Handling
# ------------------------------------------------------------------

def test_overlap_with_interval():
    conn = _make_db()
    store = TimelineStore(conn)

    # Interval: Jan 1 - Jan 15
    store.create_event(namespace_id="default", title="Sprint 1",
                       event_type=EVENT_TYPE_INTERVAL,
                       start_time=1704067200.0, end_time=1705276800.0)
    # Point: Feb 1
    store.create_event(namespace_id="default", title="Feb Event",
                       event_type=EVENT_TYPE_POINT, start_time=1706745600.0)

    # Query range: Jan 10 - Jan 20 — should overlap with Sprint 1 only
    overlapping = store.get_overlapping("default", 1704844800.0, 1705708800.0)
    assert len(overlapping) == 1
    assert overlapping[0].title == "Sprint 1"

    # Query range: Jan 10 - Feb 15 — should overlap with both
    overlapping2 = store.get_overlapping("default", 1704844800.0, 1707955200.0)
    assert len(overlapping2) == 2
    conn.close()
    print("PASS: test_overlap_with_interval")


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def test_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_timeline_persist.db")

        # Session 1: create event
        conn1 = sqlite3.connect(db_path)
        conn1.row_factory = sqlite3.Row
        conn1.execute("PRAGMA foreign_keys = ON;")
        apply_migrations(conn1)
        store1 = TimelineStore(conn1)

        event = store1.create_event(
            namespace_id="default", title="Persistent Event",
            event_type=EVENT_TYPE_POINT, start_time=1704067200.0,
        )
        event_id = event.event_id
        conn1.close()

        # Session 2: reopen and verify
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA foreign_keys = ON;")
        apply_migrations(conn2)
        store2 = TimelineStore(conn2)

        reloaded = store2.get_event(event_id)
        assert reloaded is not None
        assert reloaded.title == "Persistent Event"
        assert reloaded.start_time == 1704067200.0
        conn2.close()
    print("PASS: test_persistence")


# ------------------------------------------------------------------
# Deletion Cleanup
# ------------------------------------------------------------------

def test_deletion_cleanup():
    conn = _make_db()
    store = TimelineStore(conn)

    _insert_memory(conn, "mem_del_event", "Event to delete", 1704067200.0)

    event = store.create_event(
        namespace_id="default", title="To Delete",
        event_type=EVENT_TYPE_POINT, start_time=1704067200.0,
    )
    store.link_memory("mem_del_event", event.event_id)

    # Verify link exists
    assert len(store.get_memories_for_event(event.event_id)) == 1

    # Delete event
    deleted = store.delete_event(event.event_id)
    assert deleted is True

    # Verify event and links cleaned up
    assert store.get_event(event.event_id) is None
    row = conn.execute(
        "SELECT * FROM memory_events WHERE event_id = ?", (event.event_id,)
    ).fetchone()
    assert row is None

    # Memory itself still exists
    mem_row = conn.execute("SELECT * FROM memory_blocks WHERE id = 'mem_del_event'").fetchone()
    assert mem_row is not None
    conn.close()
    print("PASS: test_deletion_cleanup")


# ------------------------------------------------------------------
# Namespace Isolation
# ------------------------------------------------------------------

def test_namespace_isolation():
    conn = _make_db()
    store = TimelineStore(conn)

    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns_x', 'NS X', 1000)")
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('ns_y', 'NS Y', 1000)")

    store.create_event(namespace_id="ns_x", title="Event X",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)
    store.create_event(namespace_id="ns_y", title="Event Y",
                       event_type=EVENT_TYPE_POINT, start_time=1704067200.0)

    x_events = store.get_chronological("ns_x")
    y_events = store.get_chronological("ns_y")
    assert len(x_events) == 1
    assert x_events[0].title == "Event X"
    assert len(y_events) == 1
    assert y_events[0].title == "Event Y"
    conn.close()
    print("PASS: test_namespace_isolation")


# ------------------------------------------------------------------
# CASE-009: Out-of-Order Ingestion Replay (Golden Case)
# ------------------------------------------------------------------

def test_case_009_out_of_order_ingestion():
    """
    CASE-009: March is ingested first, January is ingested second.
    Timeline must return January → March based on event_time.
    """
    conn = _make_db()
    store = TimelineStore(conn)

    JAN_1 = 1704067200.0   # January 1 2024
    MAR_1 = 1709251200.0   # March 1 2024
    MAY_1 = 1714521600.0   # May 1 2024

    # Insert memories in reverse ingestion order
    _insert_memory(conn, "mem_deploy_v3", "Deployed v1.2 release with bug fixes.", MAY_1)
    _insert_memory(conn, "mem_deploy_v2", "Deployed v1.1 release with dark mode.", MAR_1)
    _insert_memory(conn, "mem_deploy_v1", "Initial launch of v1.0 in production.", JAN_1)

    # Create events — ingested in reverse chronological order (May, March, January)
    ev3 = store.create_event(
        namespace_id="default", title="v1.2 Release",
        event_type=EVENT_TYPE_POINT, start_time=MAY_1,
        created_at=1000.0,  # ingested first
    )
    ev2 = store.create_event(
        namespace_id="default", title="v1.1 Release",
        event_type=EVENT_TYPE_POINT, start_time=MAR_1,
        created_at=2000.0,  # ingested second
    )
    ev1 = store.create_event(
        namespace_id="default", title="v1.0 Launch",
        event_type=EVENT_TYPE_POINT, start_time=JAN_1,
        created_at=3000.0,  # ingested third
    )

    # Link memories
    store.link_memory("mem_deploy_v1", ev1.event_id)
    store.link_memory("mem_deploy_v2", ev2.event_id)
    store.link_memory("mem_deploy_v3", ev3.event_id)

    # Timeline MUST order by event_time, NOT ingestion order
    timeline = store.get_chronological("default")
    assert len(timeline) == 3
    assert timeline[0].title == "v1.0 Launch", \
        f"CASE-009 FAIL: Expected 'v1.0 Launch' first, got '{timeline[0].title}'"
    assert timeline[1].title == "v1.1 Release", \
        f"CASE-009 FAIL: Expected 'v1.1 Release' second, got '{timeline[1].title}'"
    assert timeline[2].title == "v1.2 Release", \
        f"CASE-009 FAIL: Expected 'v1.2 Release' third, got '{timeline[2].title}'"

    # Verify event times are in ascending order
    assert timeline[0].start_time < timeline[1].start_time < timeline[2].start_time

    conn.close()
    print("PASS: test_case_009_out_of_order_ingestion (CASE-009 GOLDEN)")


if __name__ == "__main__":
    test_point_event_creation()
    test_interval_event_creation()
    test_interval_requires_both_times()
    test_event_time_ordering()
    test_start_end_interval()
    test_out_of_order_ingestion()
    test_memory_event_relationship()
    test_chronological_query()
    test_before_filter()
    test_after_filter()
    test_between_filter()
    test_overlap_with_interval()
    test_persistence()
    test_deletion_cleanup()
    test_namespace_isolation()
    test_case_009_out_of_order_ingestion()
    print("\nAll TimelineStore unit tests passed successfully.")
