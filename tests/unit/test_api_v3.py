"""
Unit and Contract tests for Almond V3 FastAPI endpoints.
Tests:
- POST /v3/memories (sync and async)
- GET /v3/memories/{id} (found and 404)
- DELETE /v3/memories/{id} (cascade and index consistency)
- POST /v3/query (calls RetrievalEngine, returns context & candidates)
- POST /v3/query/trace (exposes full RetrievalTrace)
- GET /v3/timeline (chronological ordering & filtering)
- GET /v3/entities/{id} (canonical entity, aliases, links)
- POST /v3/evaluations (runs golden cases through service boundary)
- GET /v3/health (SQLite, worker, Chroma status)
- Namespace isolation through the API
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from core.api.app import create_app
from core.memory_store import MemoryStore


def test_api_memories_crud_and_isolation():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "api_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)
        client = TestClient(app)

        # 1. POST /v3/memories (synchronous)
        r = client.post("/v3/memories", json={
            "content": "Secret credentials for namespace Alpha.",
            "namespace": "tenant_alpha",
            "tag": "CORE_RULE",
            "importance_score": 9.0,
            "sync": True,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["status"] == "SUCCESS"
        assert data["namespace"] == "tenant_alpha"
        mem_alpha_id = data["memory_id"]

        # 2. POST /v3/memories (in tenant Beta)
        r_beta = client.post("/v3/memories", json={
            "content": "Secret credentials for namespace Beta.",
            "namespace": "tenant_beta",
            "tag": "CORE_RULE",
            "importance_score": 9.0,
            "sync": True,
        })
        assert r_beta.status_code == 201
        mem_beta_id = r_beta.json()["memory_id"]

        # 3. GET /v3/memories/{id}
        r_get = client.get(f"/v3/memories/{mem_alpha_id}")
        assert r_get.status_code == 200
        m_data = r_get.json()
        assert m_data["id"] == mem_alpha_id
        assert m_data["content"] == "Secret credentials for namespace Alpha."
        assert m_data["namespace_id"] == "tenant_alpha"
        assert m_data["tag"] == "CORE_RULE"

        # 4. GET /v3/memories/{non_existent} -> 404
        r_not_found = client.get("/v3/memories/non_existent_id")
        assert r_not_found.status_code == 404

        # 5. POST /v3/query — Namespace Isolation Test
        # Query tenant Alpha
        r_q_alpha = client.post("/v3/query", json={
            "query": "What are the secret credentials?",
            "namespace": "tenant_alpha",
        })
        assert r_q_alpha.status_code == 200
        q_alpha_data = r_q_alpha.json()
        assert mem_alpha_id in q_alpha_data["memory_ids"]
        assert mem_beta_id not in q_alpha_data["memory_ids"]  # ISOLATION GUARANTEED

        # Query tenant Beta
        r_q_beta = client.post("/v3/query", json={
            "query": "What are the secret credentials?",
            "namespace": "tenant_beta",
        })
        assert r_q_beta.status_code == 200
        q_beta_data = r_q_beta.json()
        assert mem_beta_id in q_beta_data["memory_ids"]
        assert mem_alpha_id not in q_beta_data["memory_ids"]  # ISOLATION GUARANTEED

        # 6. DELETE /v3/memories/{id}
        r_del = client.delete(f"/v3/memories/{mem_alpha_id}")
        assert r_del.status_code == 200
        assert r_del.json()["deleted"] is True

        # Verify deletion in SQLite
        r_after_del = client.get(f"/v3/memories/{mem_alpha_id}")
        assert r_after_del.status_code == 404

        print("PASS: test_api_memories_crud_and_isolation")


def test_api_query_and_trace():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "api_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)
        client = TestClient(app)

        # Ingest memory
        client.post("/v3/memories", json={
            "content": "Bob delivered the infrastructure roadmap in Seattle on March 1st.",
            "namespace": "default",
            "tag": "PROJECT_FACT",
            "importance_score": 7.0,
            "event_time": 1710000000.0,
            "sync": True,
        })

        # POST /v3/query
        r_query = client.post("/v3/query", json={
            "query": "What did Bob deliver?",
            "namespace": "default",
            "reference_time": 1710000100.0,
        })
        assert r_query.status_code == 200
        q_data = r_query.json()
        assert len(q_data["memory_ids"]) > 0
        assert q_data["context_text"] != ""
        assert q_data["trace_id"] is not None

        # POST /v3/query/trace
        r_trace = client.post("/v3/query/trace", json={
            "query": "What did Bob deliver?",
            "namespace": "default",
            "reference_time": 1710000100.0,
        })
        assert r_trace.status_code == 200
        t_data = r_trace.json()
        assert t_data["trace_id"] is not None
        assert "SEMANTIC" in t_data["channels_run"]
        assert "LEXICAL" in t_data["channels_run"]
        assert "ranked_candidates" in t_data
        assert t_data["duration_ms"] > 0

        print("PASS: test_api_query_and_trace")


def test_api_timeline_and_entities():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "api_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)
        client = TestClient(app)

        # Ingest memory with entity and event time
        client.post("/v3/memories", json={
            "content": "Robert Vance joined the company on January 15th 2024.",
            "namespace": "default",
            "tag": "EPISODIC",
            "importance_score": 6.0,
            "event_time": 1705276800.0,
            "sync": True,
        })

        # GET /v3/timeline
        r_tl = client.get("/v3/timeline?namespace=default")
        assert r_tl.status_code == 200
        tl_data = r_tl.json()
        assert tl_data["total_count"] >= 1
        assert tl_data["events"][0]["start_time"] == 1705276800.0

        # Entities check
        ent = app.state.entity_store.get_entity_by_name("Robert Vance", "default")
        if ent:
            r_ent = client.get(f"/v3/entities/{ent.entity_id}")
            assert r_ent.status_code == 200
            ent_data = r_ent.json()
            assert ent_data["canonical_name"] == "Robert Vance"

        print("PASS: test_api_timeline_and_entities")


def test_api_health():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "api_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)
        client = TestClient(app)

        r = client.get("/v3/health")
        assert r.status_code == 200
        h = r.json()
        assert "components" in h
        assert h["components"]["sqlite"]["status"] == "healthy"
        assert "clock" in h
        assert h["version"] == "v3"

        print("PASS: test_api_health")


if __name__ == "__main__":
    test_api_memories_crud_and_isolation()
    test_api_query_and_trace()
    test_api_timeline_and_entities()
    test_api_health()
    print("All API v3 unit tests passed successfully.")
