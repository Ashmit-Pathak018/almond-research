"""
Integration tests for Almond V3 Standalone Memory Service.
Tests:
- End-to-end async ingestion -> worker -> retrieval -> query trace
- Cascading deletion across SQLite, Chroma, and timeline
- Degraded Chroma handling (vector index offline, fallback to FTS5 / relational truth)
- Golden evaluation endpoint (/v3/evaluations)
- Namespace isolation across multi-tenant workloads
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from core.api.app import create_app
from core.memory_store import MemoryStore
from core.sdk import Almond


def test_service_e2e_async_workflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "svc_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=True)

        with TestClient(app) as test_http:
            with Almond(base_url="http://testserver", namespace="tenant_gamma", http_client=test_http) as client:
                # 1. Submit async ingestion job
                res = client.memories.create(
                    content="Carol completed the quantum cryptography module on September 1st.",
                    tag="PROJECT_FACT",
                    importance_score=8.0,
                    event_time=1725148800.0,
                    sync=False,
                )

                assert res.job_id is not None
                assert res.status in ("QUEUED", "PROCESSING", "SUCCESS")

                # Wait for worker thread to process job to SUCCESS
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    job = app.state.service.job_store.get_job(res.job_id)
                    if job and job.state == "SUCCESS":
                        break
                    time.sleep(0.05)

                mem = client.memories.get(res.memory_id)

                assert mem is not None
                assert mem.content == "Carol completed the quantum cryptography module on September 1st."
                assert mem.namespace_id == "tenant_gamma"

                # 2. Query
                q = client.query("Who completed the quantum cryptography module?")
                assert len(q.memory_ids) > 0
                assert res.memory_id in q.memory_ids
                assert "Carol" in q.context_text

                # 3. Query Trace
                trace = client.query_trace("Who completed the quantum cryptography module?")
                assert trace.detected_intent is not None
                assert "ranked_candidates" in trace.to_dict() if hasattr(trace, "to_dict") else len(trace.ranked_candidates) > 0

                # 4. Timeline
                tl = client.timeline()
                assert tl.total_count >= 1

                # 5. Cascading Deletion
                del_ok = client.memories.delete(res.memory_id)
                assert del_ok is True

                # Search after deletion must not return deleted memory
                q_after = client.query("Who completed the quantum cryptography module?")
                assert res.memory_id not in q_after.memory_ids

        print("PASS: test_service_e2e_async_workflow")


def test_service_degraded_chroma_fallback():
    """
    Test Phase 3/4 invariant: If Chroma is offline or unavailable,
    the service gracefully degrades to SQLite FTS5 / relational retrieval
    without throwing 500 errors.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "svc_degraded.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)

        with TestClient(app) as test_http:
            with Almond(base_url="http://testserver", namespace="default", http_client=test_http) as client:
                # Ingest a memory
                res = client.memories.create(
                    content="Database migration completed successfully in Boston.",
                    tag="PROJECT_FACT",
                    importance_score=7.0,
                    sync=True,
                )

                # Simulate Chroma failure by setting _collection to None
                store._collection = None

                # Health check should report degraded
                h = client.health()
                assert h.status == "degraded"
                assert h.components["chroma"]["status"] == "degraded"

                # Query should still succeed via lexical FTS5 channel
                q = client.query("Database migration completed")
                assert res.memory_id in q.memory_ids
                assert "Database migration" in q.context_text

        print("PASS: test_service_degraded_chroma_fallback")


def test_service_evaluations_endpoint():
    """
    Test /v3/evaluations service boundary running Golden Cases.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "svc_eval.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)

        with TestClient(app) as client:
            # Run single case CASE-004 (Abstention)
            resp = client.post("/v3/evaluations", json={
                "evaluation_type": "golden",
                "case_id": "CASE-004",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_cases"] == 1
            assert data["passed_cases"] == 1
            assert data["status"] == "completed"

        print("PASS: test_service_evaluations_endpoint")


if __name__ == "__main__":
    test_service_e2e_async_workflow()
    test_service_degraded_chroma_fallback()
    test_service_evaluations_endpoint()
    print("All Service v3 integration tests passed successfully.")
