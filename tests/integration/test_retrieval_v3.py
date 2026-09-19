"""
Integration tests for Almond V3 Retrieval Engine:
Validates multi-channel candidate generation, fusion, temporal constraint reasoning,
entity resolution, ranking, context assembly, and RetrievalTrace observability.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_store import MemoryStore
from core.knowledge.entity_store import EntityStore
from core.knowledge.timeline_store import TimelineStore, EVENT_TYPE_POINT
from core.retrieval.contracts import RetrievalQuery, RetrievalChannel
from core.retrieval.retrieval_engine import RetrievalEngine


def test_retrieval_v3_multi_channel_and_fusion():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "retrieval_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_test")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        entity_store = EntityStore(store._conn)
        timeline_store = TimelineStore(store._conn)
        engine = RetrievalEngine(store, entity_store, timeline_store)

        # 1. Seed entity: Robert Vance with alias "Bob"
        e = entity_store.create_entity(
            namespace_id="default",
            canonical_name="Robert Vance",
            entity_type="PERSON"
        )
        entity_store.add_alias(e.entity_id, "Bob")

        # 2. Seed memories
        # Mem A: Mentioning Bob on a project
        mem_a = MemoryBlock(
            id="mem_bob_project",
            content="Robert Vance delivered the client infrastructure security audit report on time.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=8.0,
            event_time=1709900000.0,
        )
        store.save(mem_a)
        entity_store.link_memory(mem_a.id, e.entity_id, confidence=0.95)

        # Mem B: Python API development
        mem_b = MemoryBlock(
            id="mem_fastapi",
            content="We built the core microservices using FastAPI and SQLite.",
            tag=MemoryTag.PROJECT_FACT,
            tier=MemoryTier.L2_ACTIVE_RAM,
            importance_score=7.0,
            event_time=1708000000.0,
        )
        store.save(mem_b)

        # 3. Query: Entity alias "Bob"
        q1 = RetrievalQuery(query_text="How is Bob doing on the project?", namespace_id="default")
        res1 = engine.query(q1)

        assert not res1.is_abstention
        assert len(res1.candidates) >= 1
        top_cand = res1.candidates[0]
        assert top_cand.memory_id == "mem_bob_project"
        assert "Robert Vance" in top_cand.entity_matches
        assert res1.trace.detected_intent in ("ENTITY", "HYBRID")
        assert "ENTITY" in res1.trace.channels_run
        assert "Robert Vance delivered" in res1.context_text
        print("PASS: test_retrieval_v3_multi_channel_and_fusion (Entity alias)")

        # 4. Query: Lexical exact term
        q2 = RetrievalQuery(query_text="FastAPI microservices architecture", namespace_id="default")
        res2 = engine.query(q2)
        assert not res2.is_abstention
        matched_ids = [c.memory_id for c in res2.candidates]
        assert "mem_fastapi" in matched_ids
        print("PASS: test_retrieval_v3_multi_channel_and_fusion (Lexical/Semantic)")

        store.close()


def test_retrieval_v3_temporal_constraint_reasoning():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "temporal_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_temporal")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        timeline_store = TimelineStore(store._conn)
        engine = RetrievalEngine(store, timeline_store=timeline_store)

        # Jan 10 2024: MySQL setup
        t_jan = 1704888000.0
        mem_jan = MemoryBlock(
            id="mem_mysql",
            content="I set up our primary MySQL database with replication.",
            tag=MemoryTag.PROJECT_FACT,
            event_time=t_jan,
            importance_score=7.0,
        )
        store.save(mem_jan)
        ev_jan = timeline_store.create_event(
            namespace_id="default", title="MySQL setup", event_type=EVENT_TYPE_POINT, start_time=t_jan
        )
        timeline_store.link_memory(mem_jan.id, ev_jan.event_id)

        # Feb 15 2024: Migration plan
        t_feb = 1707998400.0
        mem_feb = MemoryBlock(
            id="mem_plan",
            content="Drafted the architectural migration plan to switch from MySQL to PostgreSQL.",
            tag=MemoryTag.PROJECT_FACT,
            event_time=t_feb,
            importance_score=8.0,
        )
        store.save(mem_feb)
        ev_feb = timeline_store.create_event(
            namespace_id="default", title="PostgreSQL migration plan", event_type=EVENT_TYPE_POINT, start_time=t_feb
        )
        timeline_store.link_memory(mem_feb.id, ev_feb.event_id)

        # March 1 2024: PostgreSQL live
        t_mar = 1709294400.0
        mem_mar = MemoryBlock(
            id="mem_live",
            content="Completed the cutover: PostgreSQL is now live in production.",
            tag=MemoryTag.PROJECT_FACT,
            event_time=t_mar,
            importance_score=9.0,
        )
        store.save(mem_mar)
        ev_mar = timeline_store.create_event(
            namespace_id="default", title="switching to PostgreSQL live", event_type=EVENT_TYPE_POINT, start_time=t_mar
        )
        timeline_store.link_memory(mem_mar.id, ev_mar.event_id)

        # Query: "What did I do before switching to PostgreSQL?"
        q = RetrievalQuery(
            query_text="What did I do before switching to PostgreSQL?",
            namespace_id="default",
            reference_time=t_mar
        )
        res = engine.query(q)

        assert not res.is_abstention
        matched_ids = [c.memory_id for c in res.candidates]
        
        # mem_live (March 1) MUST be rejected by temporal reasoner because event_time >= anchor
        assert "mem_live" not in matched_ids, "mem_live occurred at/after anchor and must be filtered out"
        # mem_plan (Feb) and mem_mysql (Jan) MUST be retrieved
        assert "mem_plan" in matched_ids
        assert "mem_mysql" in matched_ids
        # Order should place Feb closer to cutover before Jan
        assert matched_ids.index("mem_plan") < matched_ids.index("mem_mysql")
        print("PASS: test_retrieval_v3_temporal_constraint_reasoning (CASE-001 behavior)")

        store.close()


def test_retrieval_v3_abstention():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "abstain_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_abstain")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        engine = RetrievalEngine(store)

        # Seed unrelated memory
        mb = MemoryBlock(
            content="My favorite food is spicy vegetarian ramen.",
            tag=MemoryTag.USER_PROFILE,
            importance_score=5.0
        )
        store.save(mb)

        # Query completely unrelated
        q = RetrievalQuery(query_text="What did I say my favorite musical artist was?")
        res = engine.query(q)

        assert res.is_abstention is True
        assert res.context_text == "NO_RELEVANT_MEMORIES"
        assert len(res.candidates) == 0
        print("PASS: test_retrieval_v3_abstention (CASE-004 behavior)")

        store.close()


def test_retrieval_v3_namespace_isolation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "ns_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_ns")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        engine = RetrievalEngine(store)

        # Memory in tenant A
        mem_a = MemoryBlock(
            id="mem_tenant_a",
            content="Confidential financial report for Tenant A.",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=5.0,
            namespace_id="tenant_a",
        )
        store.save(mem_a)

        # Memory in tenant B
        mem_b = MemoryBlock(
            id="mem_tenant_b",
            content="Public marketing materials for Tenant B.",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=5.0,
            namespace_id="tenant_b",
        )
        store.save(mem_b)

        # Query tenant A
        q_a = RetrievalQuery(query_text="Confidential financial report", namespace_id="tenant_a")
        res_a = engine.query(q_a)
        assert any(c.memory_id == "mem_tenant_a" for c in res_a.candidates)
        assert not any(c.memory_id == "mem_tenant_b" for c in res_a.candidates)

        # Query tenant B
        q_b = RetrievalQuery(query_text="Confidential financial report", namespace_id="tenant_b")
        res_b = engine.query(q_b)
        assert not any(c.memory_id == "mem_tenant_a" for c in res_b.candidates)
        print("PASS: test_retrieval_v3_namespace_isolation")

        store.close()


def test_retrieval_v3_missing_chroma_fallback():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = os.path.join(tmpdir, "fallback_test.db")
        chroma_path = os.path.join(tmpdir, "chroma_empty")

        store = MemoryStore(db_path=db_path, chroma_path=chroma_path)
        engine = RetrievalEngine(store)

        # Save memory only to SQLite (simulate Chroma missing/wiped)
        mb = MemoryBlock(
            id="mem_sql_only",
            content="Critical system recovery procedure using runbooks.",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=8.0,
        )
        # Insert directly to SQLite without Chroma indexing
        with store._conn:
            store._conn.execute("""
                INSERT INTO memory_blocks (id, namespace_id, content, tag, tier, state, importance_score, keywords, source, created_at, updated_at, last_accessed_at, access_count)
                VALUES (?, 'default', ?, ?, 'L2_ACTIVE_RAM', 'ACTIVE', 8.0, '[]', 'user', 1000.0, 1000.0, 1000.0, 1)
            """, (mb.id, mb.content, mb.tag.value))

        # Reset Chroma collection to simulate complete vector outage
        try:
            store.chroma_client.delete_collection("almond_memory_vault")
        except Exception:
            pass
        store._collection = None

        q = RetrievalQuery(query_text="system recovery runbooks")
        res = engine.query(q)

        # Must still find via LexicalRetriever directly from SQLite!
        matched_ids = [c.memory_id for c in res.candidates]
        assert "mem_sql_only" in matched_ids
        assert not res.is_abstention
        print("PASS: test_retrieval_v3_missing_chroma_fallback")

        store.close()


if __name__ == "__main__":
    test_retrieval_v3_multi_channel_and_fusion()
    test_retrieval_v3_temporal_constraint_reasoning()
    test_retrieval_v3_abstention()
    test_retrieval_v3_namespace_isolation()
    test_retrieval_v3_missing_chroma_fallback()
    print("All Retrieval V3 integration tests passed successfully.")
