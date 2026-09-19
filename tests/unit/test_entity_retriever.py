"""
Unit tests for core/retrieval/entity_retriever.py
Tests:
- Entity alias resolution (Bob -> Robert Vance)
- Memory retrieval linked to canonical entity
- Confidence propagation
- Namespace isolation
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.storage.migrations.migration_manager import apply_migrations
import core.storage.migrations.migration_001_v2_to_v3
import core.storage.migrations.migration_002_knowledge_layer
import core.storage.migrations.migration_003_fts5
from core.knowledge.entity_store import EntityStore
from core.retrieval.contracts import RetrievalQuery, RetrievalChannel
from core.retrieval.entity_retriever import EntityRetriever


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_migrations(conn)
    return conn


def _insert_memory(conn, mid, content, namespace_id="default"):
    conn.execute("""
        INSERT INTO memory_blocks (
            id, namespace_id, content, tag, tier, state, importance_score,
            keywords, source, created_at, updated_at, last_accessed_at, access_count
        ) VALUES (?, ?, ?, 'PROJECT_FACT', 'L2_ACTIVE_RAM', 'ACTIVE', 7.0, '[]', 'user', 1000.0, 1000.0, 1000.0, 1)
    """, (mid, namespace_id, content))


def test_entity_retriever_alias_resolution():
    conn = _make_db()
    entity_store = EntityStore(conn)
    retriever = EntityRetriever(entity_store)

    # 1. Create canonical entity Robert Vance with alias Bob
    ent = entity_store.create_entity(
        namespace_id="default",
        canonical_name="Robert Vance",
        entity_type="PERSON"
    )
    entity_store.add_alias(ent.entity_id, "Bob")

    # 2. Insert memory and link
    _insert_memory(conn, "mem_robert_report", "Robert Vance delivered the infrastructure audit report on time.")
    entity_store.link_memory("mem_robert_report", ent.entity_id, confidence=0.95)

    # 3. Query using alias "Bob"
    query = RetrievalQuery(query_text="What did Bob work on?", namespace_id="default")
    results = retriever.retrieve(query)

    assert len(results) >= 1
    top = results[0]
    assert top.memory_id == "mem_robert_report"
    assert RetrievalChannel.ENTITY in top.source_channels
    assert "Robert Vance" in top.entity_matches
    assert top.entity_confidence >= 0.90
    conn.close()
    print("PASS: test_entity_retriever_alias_resolution")


def test_entity_retriever_namespace_isolation():
    conn = _make_db()
    entity_store = EntityStore(conn)
    retriever = EntityRetriever(entity_store)

    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('team_a', 'Team A', 1000.0)")
    conn.execute("INSERT OR IGNORE INTO namespaces (namespace_id, name, created_at) VALUES ('team_b', 'Team B', 1000.0)")

    ent_a = entity_store.create_entity(namespace_id="team_a", canonical_name="Alice", entity_type="PERSON")
    entity_store.add_alias(ent_a.entity_id, "Ali")
    _insert_memory(conn, "mem_team_a", "Alice leads Team A frontend.", namespace_id="team_a")
    entity_store.link_memory("mem_team_a", ent_a.entity_id)

    ent_b = entity_store.create_entity(namespace_id="team_b", canonical_name="Alice", entity_type="PERSON")
    entity_store.add_alias(ent_b.entity_id, "Ali")
    _insert_memory(conn, "mem_team_b", "Alice leads Team B backend.", namespace_id="team_b")
    entity_store.link_memory("mem_team_b", ent_b.entity_id)

    query_a = RetrievalQuery(query_text="Where is Ali working?", namespace_id="team_a")
    results_a = retriever.retrieve(query_a)
    ids_a = [r.memory_id for r in results_a]
    assert "mem_team_a" in ids_a
    assert "mem_team_b" not in ids_a

    conn.close()
    print("PASS: test_entity_retriever_namespace_isolation")


if __name__ == "__main__":
    test_entity_retriever_alias_resolution()
    test_entity_retriever_namespace_isolation()
    print("All EntityRetriever unit tests passed successfully.")
