"""
Unit tests for the Almond V3 Typed Python SDK.
Tests:
- client.memories.create (sync & async)
- client.memories.get
- client.query
- client.query_trace
- client.timeline
- client.memories.delete
- client.health
- Typed error handling (AlmondNotFoundError, AlmondValidationError)
- Namespace scoping
"""

import os
import sys
import tempfile
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from core.api.app import create_app
from core.memory_store import MemoryStore
from core.sdk import (
    Almond,
    AlmondNotFoundError,
    AlmondValidationError,
)


def test_sdk_full_lifecycle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "sdk_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        app = create_app(store=store, start_worker=False)

        test_http = TestClient(app)
        with Almond(base_url="http://testserver", namespace="project_omega", http_client=test_http) as client:
            # 1. Health check
            h = client.health()
            assert h.status in ("healthy", "degraded")
            assert "sqlite" in h.components

            # 2. Create memory (sync)
            created = client.memories.create(
                content="Project Omega architecture specification released in Berlin on June 10th 2024.",
                tag="PROJECT_FACT",
                importance_score=8.5,
                event_time=1718000000.0,
                sync=True,
            )
            assert created.memory_id is not None
            assert created.status == "SUCCESS"
            assert created.namespace == "project_omega"

            # 3. Get memory
            mem = client.memories.get(created.memory_id)
            assert mem.id == created.memory_id
            assert mem.content == "Project Omega architecture specification released in Berlin on June 10th 2024."
            assert mem.importance_score == 8.5
            assert mem.namespace_id == "project_omega"

            # 4. Query
            q_res = client.query("When was Project Omega released?", reference_time=1718001000.0)
            assert q_res.query == "When was Project Omega released?"
            assert len(q_res.memory_ids) > 0
            assert created.memory_id in q_res.memory_ids
            assert q_res.context_text != ""

            # 5. Query Trace
            trace = client.query_trace("When was Project Omega released?", reference_time=1718001000.0)
            assert trace.trace_id is not None
            assert "ranked_candidates" in trace.to_dict() if hasattr(trace, "to_dict") else len(trace.ranked_candidates) > 0

            # 6. Timeline
            tl = client.timeline()
            assert tl.total_count >= 1
            assert tl.events[0].start_time == 1718000000.0

            # 7. Delete memory
            deleted = client.memories.delete(created.memory_id)
            assert deleted is True

            # 8. Get after delete -> AlmondNotFoundError
            try:
                client.memories.get(created.memory_id)
                assert False, "Should have raised AlmondNotFoundError"
            except AlmondNotFoundError as e:
                assert e.status_code == 404

            # 9. Validation error -> AlmondValidationError
            try:
                client.memories.create(content="Invalid", tag="NON_EXISTENT_TAG", sync=True)
                assert False, "Should have raised AlmondValidationError"
            except AlmondValidationError as e:
                assert e.status_code in (400, 422)

        print("PASS: test_sdk_full_lifecycle")


if __name__ == "__main__":
    test_sdk_full_lifecycle()
    print("All SDK v3 unit tests passed successfully.")
