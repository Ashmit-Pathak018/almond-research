"""
Unit tests for core/retrieval/lexical_retriever.py
Tests:
- exact term match
- partial/relevant lexical match
- no match
- namespace isolation
- FTS5 BM25 score normalization
- LIKE fallback behavior
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
import core.storage.migrations.migration_003_fts5
from core.retrieval.contracts import RetrievalQuery, RetrievalChannel
from core.retrieval.lexical_retriever import LexicalRetriever


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


def _insert_memory(conn, mid, content, namespace_id="default", keywords="[]"):
    conn.execute("""
        INSERT INTO memory_blocks (
            id, namespace_id, content, tag, tier, state, importance_score,
            keywords, source, created_at, updated_at, last_accessed_at, access_count
        ) VALUES (?, ?, ?, 'PROJECT_FACT', 'L2_ACTIVE_RAM', 'ACTIVE', 7.0, ?, 'user', 1000.0, 1000.0, 1000.0, 1)
    """, (mid, namespace_id, content, keywords))


def test_lexical_exact_match():
    conn = _make_db()
    retriever = LexicalRetriever(conn)

    _insert_memory(conn, "m1", "We deployed our application on Kubernetes cluster.")
    _insert_memory(conn, "m2", "Daily standup meeting notes about UI design.")

    query = RetrievalQuery(query_text="Kubernetes", namespace_id="default")
    results = retriever.retrieve(query)

    assert len(results) >= 1
    assert results[0].memory_id == "m1"
    assert RetrievalChannel.LEXICAL in results[0].source_channels
    assert results[0].normalized_scores[RetrievalChannel.LEXICAL.value] > 0.0
    conn.close()
    print("PASS: test_lexical_exact_match")


def test_lexical_partial_match():
    conn = _make_db()
    retriever = LexicalRetriever(conn)

    _insert_memory(conn, "m1", "Drafted architectural migration plan to switch from MySQL to PostgreSQL.")
    _insert_memory(conn, "m2", "Setting up local development environments with Docker.")

    query = RetrievalQuery(query_text="PostgreSQL architectural migration plan", namespace_id="default")
    results = retriever.retrieve(query)

    assert len(results) >= 1
    assert results[0].memory_id == "m1"
    score = results[0].normalized_scores[RetrievalChannel.LEXICAL.value]
    assert 0.0 < score <= 1.0
    conn.close()
    print("PASS: test_lexical_partial_match")


def test_lexical_no_match():
    conn = _make_db()
    retriever = LexicalRetriever(conn)

    _insert_memory(conn, "m1", "We deployed our application on Kubernetes cluster.")

    query = RetrievalQuery(query_text="unrelated quantum mechanical physics", namespace_id="default")
    results = retriever.retrieve(query)

    assert len(results) == 0
    conn.close()
    print("PASS: test_lexical_no_match")


def test_lexical_namespace_isolation():
    conn = _make_db()
    retriever = LexicalRetriever(conn)

    # Ensure namespaces exist
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('tenant_a', 'Tenant A', 1000.0)")
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('tenant_b', 'Tenant B', 1000.0)")

    _insert_memory(conn, "m_a", "Confidential payroll salary database for Tenant A.", namespace_id="tenant_a")
    _insert_memory(conn, "m_b", "Confidential payroll salary database for Tenant B.", namespace_id="tenant_b")

    query_a = RetrievalQuery(query_text="payroll salary database", namespace_id="tenant_a")
    results_a = retriever.retrieve(query_a)
    ids_a = [r.memory_id for r in results_a]
    assert "m_a" in ids_a
    assert "m_b" not in ids_a

    query_b = RetrievalQuery(query_text="payroll salary database", namespace_id="tenant_b")
    results_b = retriever.retrieve(query_b)
    ids_b = [r.memory_id for r in results_b]
    assert "m_b" in ids_b
    assert "m_a" not in ids_b
    conn.close()
    print("PASS: test_lexical_namespace_isolation")


def test_lexical_fallback_without_fts5():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Run only migration 1 & 2 without migration 3 (simulating no FTS5 table)
    core.storage.migrations.migration_001_v2_to_v3.migrate_v2_to_v3(conn)
    core.storage.migrations.migration_002_knowledge_layer.migrate_knowledge_layer(conn)

    _insert_memory(conn, "m1", "Redis caching layer configuration.")
    retriever = LexicalRetriever(conn)

    query = RetrievalQuery(query_text="Redis caching layer", namespace_id="default")
    results = retriever.retrieve(query)
    assert len(results) >= 1
    assert results[0].memory_id == "m1"
    conn.close()
    print("PASS: test_lexical_fallback_without_fts5")


if __name__ == "__main__":
    test_lexical_exact_match()
    test_lexical_partial_match()
    test_lexical_no_match()
    test_lexical_namespace_isolation()
    test_lexical_fallback_without_fts5()
    print("All LexicalRetriever unit tests passed successfully.")
