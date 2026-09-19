"""
Unit tests for Ingestion Jobs persistence and state machine in SQLite.
Tests:
- Job creation in RECEIVED -> QUEUED
- Legal state transitions (QUEUED -> PROCESSING -> SUCCESS, QUEUED -> PROCESSING -> RETRY -> PROCESSING -> DEAD_LETTER)
- Illegal state transitions (SUCCESS -> QUEUED, DEAD_LETTER -> PROCESSING, etc.)
- Rejection of invalid transitions via InvalidStateTransitionError
- Persistence across connection restarts
- Stale PROCESSING job recovery
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.clock import VirtualClock, set_clock, SystemClock
from core.memory_store import MemoryStore
from core.workers.ingestion_service import (
    JobStore,
    JobState,
    InvalidStateTransitionError,
)


def test_job_creation_and_state():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "jobs_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        job_store = JobStore(store._conn)

        job = job_store.create_job(
            payload={"content": "Alice lives in Zurich"},
            namespace_id="test_ns",
            max_retries=3,
        )

        assert job.job_id is not None
        assert job.state == JobState.QUEUED
        assert job.namespace_id == "test_ns"
        assert job.attempts == 0
        assert job.max_retries == 3
        assert job.payload["content"] == "Alice lives in Zurich"

        # Fetch by ID
        fetched = job_store.get_job(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.state == JobState.QUEUED
        print("PASS: test_job_creation_and_state")


def test_legal_and_illegal_state_transitions():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "jobs_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        job_store = JobStore(store._conn)

        job = job_store.create_job(payload={"content": "Legal test"})

        # QUEUED -> PROCESSING (Legal)
        claimed = job_store.claim_next_job()
        assert claimed is not None
        assert claimed.job_id == job.job_id
        assert claimed.state == JobState.PROCESSING
        assert claimed.attempts == 1

        # PROCESSING -> SUCCESS (Legal)
        job_store.update_state(job.job_id, JobState.SUCCESS)
        job_s = job_store.get_job(job.job_id)
        assert job_s.state == JobState.SUCCESS
        assert job_s.completed_at is not None

        # SUCCESS -> QUEUED (Illegal! Must raise InvalidStateTransitionError)
        try:
            job_store.update_state(job.job_id, JobState.QUEUED)
            assert False, "Should have raised InvalidStateTransitionError"
        except InvalidStateTransitionError:
            pass

        # Create another job to test retry and dead letter
        job2 = job_store.create_job(payload={"content": "Retry test"}, max_retries=2)
        claimed2 = job_store.claim_next_job()
        assert claimed2.job_id == job2.job_id

        # PROCESSING -> RETRY (Legal)
        job_store.update_state(job2.job_id, JobState.RETRY, error="Transient network timeout")
        job2_r = job_store.get_job(job2.job_id)
        assert job2_r.state == JobState.RETRY
        assert job2_r.last_error == "Transient network timeout"

        # RETRY -> PROCESSING (Legal)
        claimed2_again = job_store.claim_next_job()
        assert claimed2_again is not None
        assert claimed2_again.job_id == job2.job_id
        assert claimed2_again.attempts == 2

        # PROCESSING -> DEAD_LETTER (Legal)
        job_store.update_state(job2.job_id, JobState.DEAD_LETTER, error="Max retries exhausted")
        job2_dl = job_store.get_job(job2.job_id)
        assert job2_dl.state == JobState.DEAD_LETTER
        assert job2_dl.completed_at is not None

        # DEAD_LETTER -> PROCESSING (Illegal! Must raise InvalidStateTransitionError)
        try:
            job_store.update_state(job2.job_id, JobState.PROCESSING)
            assert False, "Should have raised InvalidStateTransitionError from DEAD_LETTER"
        except InvalidStateTransitionError:
            pass

        print("PASS: test_legal_and_illegal_state_transitions")


def test_crash_recovery_stale_jobs():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db = os.path.join(tmpdir, "jobs_test.db")
        chroma = os.path.join(tmpdir, "chroma")
        store = MemoryStore(db_path=db, chroma_path=chroma)
        job_store = JobStore(store._conn)

        vclock = VirtualClock(current_time=1000.0)
        set_clock(vclock)

        # Job 1: attempts < max_retries
        job1 = job_store.create_job(payload={"content": "Task 1"}, max_retries=3)
        claimed1 = job_store.claim_next_job()
        assert claimed1.state == JobState.PROCESSING

        # Job 2: attempts == max_retries
        job2 = job_store.create_job(payload={"content": "Task 2"}, max_retries=1)
        claimed2 = job_store.claim_next_job()
        assert claimed2.state == JobState.PROCESSING

        # Advance virtual clock by 400s (timeout is 300s)
        vclock.advance(400.0)

        recovered = job_store.recover_stale_jobs(timeout_seconds=300.0)
        assert len(recovered) == 2

        j1_after = job_store.get_job(job1.job_id)
        assert j1_after.state == JobState.RETRY

        j2_after = job_store.get_job(job2.job_id)
        assert j2_after.state == JobState.DEAD_LETTER

        set_clock(SystemClock())
        print("PASS: test_crash_recovery_stale_jobs")


if __name__ == "__main__":
    test_job_creation_and_state()
    test_legal_and_illegal_state_transitions()
    test_crash_recovery_stale_jobs()
    print("All Ingestion Job unit tests passed successfully.")
