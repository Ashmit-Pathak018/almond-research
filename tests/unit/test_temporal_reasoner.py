"""
Unit tests for core/retrieval/temporal_reasoner.py
Tests:
- Resolving 'before' constraint with anchor event
- Rejection of events occurring at or after anchor timestamp
- Chronological ordering
- Outputting inspectable ConstraintReasoningResult
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
import core.storage.migrations.migration_003_fts5
from core.knowledge.timeline_store import TimelineStore, EVENT_TYPE_POINT
from core.retrieval.contracts import RetrievalCandidate, RetrievalQuery
from core.retrieval.intent_router import IntentRouter
from core.retrieval.temporal_reasoner import TemporalReasoner


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


def test_temporal_reasoner_before_constraint():
    conn = _make_db()
    timeline_store = TimelineStore(conn)
    reasoner = TemporalReasoner(timeline_store)
    router = IntentRouter()

    # Anchor event: March 1 2024 (PostgreSQL cutover)
    t_mar = 1709294400.0
    ev_anchor = timeline_store.create_event(
        namespace_id="default",
        title="switching to PostgreSQL cutover",
        event_type=EVENT_TYPE_POINT,
        start_time=t_mar,
    )

    # Preceding event 1: Jan 10 2024
    t_jan = 1704888000.0
    # Preceding event 2: Feb 15 2024
    t_feb = 1707998400.0
    # Later event: April 1 2024
    t_apr = 1711972800.0

    c_jan = RetrievalCandidate(memory_id="mem_jan", content="MySQL initial setup", event_time=t_jan)
    c_feb = RetrievalCandidate(memory_id="mem_feb", content="Postgres migration plan", event_time=t_feb)
    c_mar = RetrievalCandidate(memory_id="mem_mar", content="Cutover to Postgres live", event_time=t_mar)
    c_apr = RetrievalCandidate(memory_id="mem_apr", content="Optimized postgres indexes", event_time=t_apr)

    query = RetrievalQuery(
        query_text="What did I do before switching to PostgreSQL?",
        namespace_id="default",
        reference_time=t_mar
    )
    intent = router.analyze(query.query_text)

    candidates = [c_jan, c_feb, c_mar, c_apr]
    res = reasoner.resolve_constraints(candidates, query, intent)

    accepted_ids = [c.memory_id for c in res.accepted_candidates]
    rejected_ids = [c.memory_id for c in res.rejected_candidates]

    # March (cutover itself) and April (after) must be rejected
    assert "mem_mar" in rejected_ids
    assert "mem_apr" in rejected_ids
    # Jan and Feb must be accepted
    assert "mem_feb" in accepted_ids
    assert "mem_jan" in accepted_ids

    # For 'before' queries, the order should place Feb (closer to cutover) before Jan
    assert accepted_ids.index("mem_feb") < accepted_ids.index("mem_jan")

    # Structured reasoner result is inspectable
    assert res.anchor_timestamp == t_mar
    assert res.constraints_applied["direction"] == "before"

    conn.close()
    print("PASS: test_temporal_reasoner_before_constraint")


if __name__ == "__main__":
    test_temporal_reasoner_before_constraint()
    print("All TemporalReasoner unit tests passed successfully.")
