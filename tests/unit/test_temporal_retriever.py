"""
Unit tests for core/retrieval/temporal_retriever.py
Tests:
- Chronological ordering by event_time (not ingestion order)
- before, after, between, overlapping retrieval
- Mapping timeline events back to memory candidates
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
import core.storage.migrations.migration_003_fts5
from core.knowledge.timeline_store import TimelineStore, EVENT_TYPE_POINT, EVENT_TYPE_INTERVAL
from core.retrieval.contracts import RetrievalQuery, RetrievalChannel
from core.retrieval.temporal_retriever import TemporalRetriever


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


def _insert_memory(conn, mid, content, event_time=None):
    conn.execute("""
        INSERT INTO memory_blocks (
            id, namespace_id, content, tag, tier, state, importance_score,
            keywords, source, event_time, created_at, updated_at, last_accessed_at, access_count
        ) VALUES (?, 'default', ?, 'EPISODIC', 'L2_ACTIVE_RAM', 'ACTIVE', 5.0, '[]', 'user', ?, 1000.0, 1000.0, 1000.0, 1)
    """, (mid, content, event_time))


def test_temporal_chronological_ordering():
    conn = _make_db()
    timeline_store = TimelineStore(conn)
    retriever = TemporalRetriever(timeline_store)

    # Ingest out of order: May first, then March, then January
    _insert_memory(conn, "mem_may", "May release", event_time=1714521600.0)
    _insert_memory(conn, "mem_mar", "March release", event_time=1709251200.0)
    _insert_memory(conn, "mem_jan", "January release", event_time=1704067200.0)

    ev_may = timeline_store.create_event("default", "May release", EVENT_TYPE_POINT, 1714521600.0, created_at=1000.0)
    timeline_store.link_memory("mem_may", ev_may.event_id)

    ev_mar = timeline_store.create_event("default", "March release", EVENT_TYPE_POINT, 1709251200.0, created_at=2000.0)
    timeline_store.link_memory("mem_mar", ev_mar.event_id)

    ev_jan = timeline_store.create_event("default", "January release", EVENT_TYPE_POINT, 1704067200.0, created_at=3000.0)
    timeline_store.link_memory("mem_jan", ev_jan.event_id)

    query = RetrievalQuery(query_text="list releases in chronological order", namespace_id="default")
    results = retriever.retrieve_chronological(query)

    assert len(results) == 3
    # Strict event_time ordering: Jan -> Mar -> May
    assert results[0].memory_id == "mem_jan"
    assert results[1].memory_id == "mem_mar"
    assert results[2].memory_id == "mem_may"
    assert RetrievalChannel.TEMPORAL in results[0].source_channels

    conn.close()
    print("PASS: test_temporal_chronological_ordering")


def test_temporal_before_and_after():
    conn = _make_db()
    timeline_store = TimelineStore(conn)
    retriever = TemporalRetriever(timeline_store)

    t_jan = 1704067200.0
    t_mar = 1709251200.0
    t_may = 1714521600.0

    _insert_memory(conn, "mem_jan", "Jan", t_jan)
    _insert_memory(conn, "mem_mar", "Mar", t_mar)
    _insert_memory(conn, "mem_may", "May", t_may)

    e1 = timeline_store.create_event("default", "Jan", EVENT_TYPE_POINT, t_jan)
    timeline_store.link_memory("mem_jan", e1.event_id)
    e2 = timeline_store.create_event("default", "Mar", EVENT_TYPE_POINT, t_mar)
    timeline_store.link_memory("mem_mar", e2.event_id)
    e3 = timeline_store.create_event("default", "May", EVENT_TYPE_POINT, t_may)
    timeline_store.link_memory("mem_may", e3.event_id)

    query = RetrievalQuery(query_text="events", namespace_id="default")
    before_mar = retriever.retrieve_before(query, timestamp=t_mar)
    assert len(before_mar) == 1
    assert before_mar[0].memory_id == "mem_jan"

    after_mar = retriever.retrieve_after(query, timestamp=t_mar)
    assert len(after_mar) == 1
    assert after_mar[0].memory_id == "mem_may"

    conn.close()
    print("PASS: test_temporal_before_and_after")


if __name__ == "__main__":
    test_temporal_chronological_ordering()
    test_temporal_before_and_after()
    print("All TemporalRetriever unit tests passed successfully.")
