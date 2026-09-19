"""
Unit tests for IngestionService worker execution.
Tests:
- Synchronous ingestion path (eval/benchmark)
- Asynchronous worker loop execution (submit_job -> SUCCESS)
- Failure retry with exponential backoff
- Dead-letter transition on exhausted retries
- Idempotent ingestion (same memory_id ingested twice)
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.memory_block import MemoryTag, MemoryTier
from core.memory_store import MemoryStore
from core.workers.ingestion_service import IngestionService, JobState


def test_sync_ingestion():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "worker_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        service = IngestionService(store)

        block = service.ingest_sync(
            content="Bob delivered the infrastructure roadmap in Seattle on March 1st.",
            namespace_id="default",
            tag=MemoryTag.PROJECT_FACT,
            importance_score=7.5,
            event_time=1710000000.0,
        )

        assert block.id is not None
        assert block.content == "Bob delivered the infrastructure roadmap in Seattle on March 1st."
        assert block.importance_score == 7.5

        # Verify saved in SQLite canonical ground truth
        stored_block = store.get_by_id(block.id)
        assert stored_block is not None
        assert stored_block.content == block.content
        assert stored_block.event_time == 1710000000.0

        # Verify timeline event linked
        events = service.timeline_store.get_events_for_memory(block.id)
        assert len(events) >= 1
        assert events[0].start_time == 1710000000.0

        print("PASS: test_sync_ingestion")


def test_async_worker_execution():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "worker_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        service = IngestionService(store, poll_interval=0.05)

        # Start worker thread
        service.start()

        try:
            job = service.submit_job(
                content="Alice started the Almond research project.",
                namespace_id="tenant_a",
                tag=MemoryTag.EPISODIC,
                importance_score=8.0,
            )

            assert job.state == JobState.QUEUED

            # Wait for worker thread to process job
            t0 = time.time()
            completed_job = None
            while time.time() - t0 < 5.0:
                completed_job = service.job_store.get_job(job.job_id)
                if completed_job.state in (JobState.SUCCESS, JobState.DEAD_LETTER):
                    break
                time.sleep(0.05)

            assert completed_job is not None
            assert completed_job.state == JobState.SUCCESS
            assert completed_job.memory_id is not None

            # Verify memory block exists in store
            mb = store.get_by_id(completed_job.memory_id)
            assert mb is not None
            assert mb.content == "Alice started the Almond research project."
            assert mb.namespace_id == "tenant_a"

        finally:
            service.stop()

        print("PASS: test_async_worker_execution")


def test_worker_retry_and_dead_letter():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "worker_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        service = IngestionService(store, poll_interval=0.05)

        # Deliberately submit invalid tag to trigger validation error and retries
        job = service.submit_job(
            content="Corrupted payload test",
            tag="INVALID_TAG_NAME",
            max_retries=2,
        )

        service.start()
        try:
            t0 = time.time()
            finished_job = None
            while time.time() - t0 < 8.0:
                finished_job = service.job_store.get_job(job.job_id)
                if finished_job.state == JobState.DEAD_LETTER:
                    break
                time.sleep(0.1)

            assert finished_job is not None
            assert finished_job.state == JobState.DEAD_LETTER
            assert finished_job.attempts >= 2
            assert finished_job.last_error is not None
        finally:
            service.stop()

        print("PASS: test_worker_retry_and_dead_letter")


def test_idempotent_ingestion():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "worker_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        service = IngestionService(store)

        fixed_mem_id = "mem_fixed_id_123"

        # First ingestion
        b1 = service.ingest_sync(
            content="The user purchased a MacBook Pro in March 2024.",
            memory_id=fixed_mem_id,
            tag=MemoryTag.PROJECT_FACT,
            event_time=1000.0,
        )

        # Second ingestion with same request (retry simulation)
        b2 = service.ingest_sync(
            content="The user purchased a MacBook Pro in March 2024.",
            memory_id=fixed_mem_id,
            tag=MemoryTag.PROJECT_FACT,
            event_time=1000.0,
        )

        # SQLite count must be 1, not 2
        cur = store._conn.execute("SELECT COUNT(*) FROM memory_blocks WHERE id = ?", (fixed_mem_id,))
        count = cur.fetchone()[0]
        assert count == 1

        stored = store.get_by_id(fixed_mem_id)
        assert stored.content == "The user purchased a MacBook Pro in March 2024."
        assert stored.event_time == 1000.0

        # Verify timeline events and facts are not duplicated
        ev_count = store._conn.execute("SELECT COUNT(*) FROM memory_events WHERE memory_id = ?", (fixed_mem_id,)).fetchone()[0]
        assert ev_count == 1


        print("PASS: test_idempotent_ingestion")


if __name__ == "__main__":
    test_sync_ingestion()
    test_async_worker_execution()
    test_worker_retry_and_dead_letter()
    test_idempotent_ingestion()
    print("All IngestionService worker unit tests passed successfully.")
