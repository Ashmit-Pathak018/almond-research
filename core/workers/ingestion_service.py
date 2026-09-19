"""
Project Almond V3 — Durable Ingestion Service & Job State Machine
Persists ingestion jobs in SQLite and guarantees transactional state transitions,
deterministic retry backoff, dead-lettering, restart recovery, and idempotency.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.clock import get_clock
from core.knowledge.entity_store import EntityStore
from core.knowledge.fact_store import FactStore
from core.knowledge.timeline_store import TimelineStore, EVENT_TYPE_POINT
from core.memory_block import MemoryBlock, MemoryTag, MemoryTier
from core.memory_pipeline_v2.entity_extractor import EntityExtractor
from core.memory_pipeline_v2.fact_extractor import FactExtractor
from core.memory_store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job State Machine Definitions
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    RECEIVED = "RECEIVED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    DEAD_LETTER = "DEAD_LETTER"


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal state machine transition is attempted."""
    pass


# Legal state transitions
VALID_TRANSITIONS: Dict[JobState, set[JobState]] = {
    JobState.RECEIVED: {JobState.QUEUED},
    JobState.QUEUED: {JobState.PROCESSING},
    JobState.PROCESSING: {JobState.SUCCESS, JobState.RETRY, JobState.DEAD_LETTER, JobState.QUEUED},
    JobState.RETRY: {JobState.PROCESSING},
    JobState.SUCCESS: set(),       # Terminal state
    JobState.DEAD_LETTER: set(),  # Terminal state
}


# ---------------------------------------------------------------------------
# Ingestion Job Model
# ---------------------------------------------------------------------------

class IngestionJob:
    def __init__(
        self,
        job_id: str,
        namespace_id: str,
        payload_json: str,
        state: JobState,
        attempts: int = 0,
        max_retries: int = 3,
        last_error: Optional[str] = None,
        created_at: float = 0.0,
        scheduled_at: float = 0.0,
        completed_at: Optional[float] = None,
        memory_id: Optional[str] = None,
        updated_at: Optional[float] = None,
    ):
        self.job_id = job_id
        self.namespace_id = namespace_id
        self.payload_json = payload_json
        self.state = state if isinstance(state, JobState) else JobState(state)
        self.attempts = attempts
        self.max_retries = max_retries
        self.last_error = last_error
        self.created_at = created_at
        self.scheduled_at = scheduled_at
        self.completed_at = completed_at
        self.memory_id = memory_id
        self.updated_at = updated_at if updated_at is not None else created_at

    @property
    def status(self) -> str:
        return self.state.value

    @property
    def attempt_count(self) -> int:
        return self.attempts

    @property
    def max_attempts(self) -> int:
        return self.max_retries

    @property
    def available_at(self) -> float:
        return self.scheduled_at

    @property
    def payload(self) -> Dict[str, Any]:
        try:
            return json.loads(self.payload_json)
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "memory_id": self.memory_id,
            "namespace_id": self.namespace_id,
            "status": self.state.value,
            "attempt_count": self.attempts,
            "max_attempts": self.max_retries,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "available_at": self.scheduled_at,
            "completed_at": self.completed_at,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# Job Store (SQLite Persistence)
# ---------------------------------------------------------------------------

class JobStore:
    """
    SQLite-backed persistent job repository enforcing the state machine.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.Lock()

    def create_job(
        self,
        payload: Dict[str, Any],
        namespace_id: str = "default",
        memory_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> IngestionJob:
        """Create a new job in RECEIVED state and immediately transition to QUEUED."""
        job_id = str(uuid.uuid4())
        now = get_clock().now()
        payload_str = json.dumps(payload)

        with self._lock:
            with self._conn:
                # 1. Store in RECEIVED
                self._conn.execute("""
                    INSERT INTO ingestion_jobs (
                        job_id, namespace_id, payload_json, state, attempts,
                        max_retries, last_error, created_at, scheduled_at,
                        completed_at, memory_id, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, NULL, ?, ?, NULL, ?, ?)
                """, (
                    job_id, namespace_id, payload_str, JobState.RECEIVED.value,
                    max_retries, now, now, memory_id, now
                ))

                # 2. Transition to QUEUED
                self._conn.execute("""
                    UPDATE ingestion_jobs
                    SET state = ?, updated_at = ?
                    WHERE job_id = ?
                """, (JobState.QUEUED.value, now, job_id))

        return IngestionJob(
            job_id=job_id,
            namespace_id=namespace_id,
            payload_json=payload_str,
            state=JobState.QUEUED,
            attempts=0,
            max_retries=max_retries,
            created_at=now,
            scheduled_at=now,
            memory_id=memory_id,
            updated_at=now,
        )

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Fetch job by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            )
            row = cursor.fetchone()
        return self._row_to_job(row) if row else None

    def update_state(
        self,
        job_id: str,
        new_state: JobState,
        error: Optional[str] = None,
        scheduled_at: Optional[float] = None,
        memory_id: Optional[str] = None,
        increment_attempts: bool = False,
    ) -> IngestionJob:
        """
        Transition job to new_state while strictly verifying legal transitions.
        """
        now = get_clock().now()
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
                )
                row = cursor.fetchone()
                if not row:
                    raise KeyError(f"Job '{job_id}' not found")

                current_job = self._row_to_job(row)
                curr_state = current_job.state

                # Check transition validity
                if new_state not in VALID_TRANSITIONS.get(curr_state, set()):
                    raise InvalidStateTransitionError(
                        f"Illegal state transition for job '{job_id}': cannot move from {curr_state.value} to {new_state.value}"
                    )

                new_attempts = current_job.attempts + (1 if increment_attempts else 0)
                completed_at = now if new_state in (JobState.SUCCESS, JobState.DEAD_LETTER) else current_job.completed_at
                new_sched = scheduled_at if scheduled_at is not None else current_job.scheduled_at
                new_mem_id = memory_id or current_job.memory_id
                new_err = error if error is not None else current_job.last_error

                self._conn.execute("""
                    UPDATE ingestion_jobs
                    SET state = ?, attempts = ?, last_error = ?, scheduled_at = ?,
                        completed_at = ?, memory_id = ?, updated_at = ?
                    WHERE job_id = ?
                """, (
                    new_state.value, new_attempts, new_err, new_sched,
                    completed_at, new_mem_id, now, job_id
                ))

        return self.get_job(job_id)

    def claim_next_job(self, namespace_id: Optional[str] = None) -> Optional[IngestionJob]:
        """
        Atomically claim the next eligible QUEUED or RETRY job whose scheduled_at <= now.
        Transitions state to PROCESSING and increments attempt count.
        """
        now = get_clock().now()
        with self._lock:
            with self._conn:
                query = """
                    SELECT * FROM ingestion_jobs
                    WHERE state IN ('QUEUED', 'RETRY') AND scheduled_at <= ?
                """
                params: List[Any] = [now]
                if namespace_id:
                    query += " AND namespace_id = ?"
                    params.append(namespace_id)
                query += " ORDER BY scheduled_at ASC, created_at ASC LIMIT 1"

                cursor = self._conn.execute(query, params)
                row = cursor.fetchone()
                if not row:
                    return None

                job = self._row_to_job(row)
                # Verify transition
                if JobState.PROCESSING not in VALID_TRANSITIONS.get(job.state, set()):
                    return None

                new_attempts = job.attempts + 1
                self._conn.execute("""
                    UPDATE ingestion_jobs
                    SET state = ?, attempts = ?, updated_at = ?
                    WHERE job_id = ?
                """, (JobState.PROCESSING.value, new_attempts, now, job.job_id))

                job.state = JobState.PROCESSING
                job.attempts = new_attempts
                job.updated_at = now
                return job

    def recover_stale_jobs(self, timeout_seconds: float = 300.0) -> List[IngestionJob]:
        """
        Recover jobs stuck in PROCESSING due to worker crash / abnormal termination.
        If attempts < max_retries -> RETRY / QUEUED with backoff.
        If attempts >= max_retries -> DEAD_LETTER.
        """
        now = get_clock().now()
        threshold = now - timeout_seconds
        recovered: List[IngestionJob] = []

        with self._lock:
            with self._conn:
                cursor = self._conn.execute("""
                    SELECT * FROM ingestion_jobs
                    WHERE state = 'PROCESSING' AND updated_at < ?
                """, (threshold,))
                stale_rows = cursor.fetchall()

                for row in stale_rows:
                    job = self._row_to_job(row)
                    if job.attempts >= job.max_retries:
                        # Exhausted
                        self._conn.execute("""
                            UPDATE ingestion_jobs
                            SET state = ?, last_error = ?, completed_at = ?, updated_at = ?
                            WHERE job_id = ?
                        """, (
                            JobState.DEAD_LETTER.value,
                            "Crash recovery: attempts exhausted during abnormal process termination",
                            now, now, job.job_id
                        ))
                        job.state = JobState.DEAD_LETTER
                    else:
                        # Schedule retry
                        retry_time = now + (2 ** job.attempts)
                        self._conn.execute("""
                            UPDATE ingestion_jobs
                            SET state = ?, last_error = ?, scheduled_at = ?, updated_at = ?
                            WHERE job_id = ?
                        """, (
                            JobState.RETRY.value,
                            "Crash recovery: recovered from interrupted processing",
                            retry_time, now, job.job_id
                        ))
                        job.state = JobState.RETRY
                        job.scheduled_at = retry_time

                    job.updated_at = now
                    recovered.append(job)

        if recovered:
            logger.info("Recovered %d stale PROCESSING jobs.", len(recovered))
        return recovered

    def list_jobs(
        self,
        namespace_id: Optional[str] = None,
        state: Optional[JobState] = None,
        limit: int = 50,
    ) -> List[IngestionJob]:
        """List ingestion jobs."""
        with self._lock:
            query = "SELECT * FROM ingestion_jobs WHERE 1=1"
            params: List[Any] = []
            if namespace_id:
                query += " AND namespace_id = ?"
                params.append(namespace_id)
            if state:
                query += " AND state = ?"
                params.append(state.value if isinstance(state, JobState) else state)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            return [self._row_to_job(r) for r in cursor.fetchall()]

    def _row_to_job(self, row: sqlite3.Row) -> IngestionJob:
        if isinstance(row, sqlite3.Row):
            d = dict(row)
        else:
            cols = [
                "job_id", "namespace_id", "payload_json", "state", "attempts",
                "max_retries", "last_error", "created_at", "scheduled_at",
                "completed_at", "memory_id", "updated_at"
            ]
            d = dict(zip(cols, row))

        return IngestionJob(
            job_id=d["job_id"],
            namespace_id=d["namespace_id"],
            payload_json=d["payload_json"],
            state=JobState(d["state"]),
            attempts=d.get("attempts", 0),
            max_retries=d.get("max_retries", 3),
            last_error=d.get("last_error"),
            created_at=d["created_at"],
            scheduled_at=d["scheduled_at"],
            completed_at=d.get("completed_at"),
            memory_id=d.get("memory_id"),
            updated_at=d.get("updated_at", d["created_at"]),
        )


# ---------------------------------------------------------------------------
# Hardened Ingestion Service
# ---------------------------------------------------------------------------

class IngestionService:
    """
    Coordinates durable job persistence, knowledge extraction, canonical memory storage,
    derived index updates, retry backoff, and crash recovery.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        entity_store: Optional[EntityStore] = None,
        timeline_store: Optional[TimelineStore] = None,
        fact_store: Optional[FactStore] = None,
        poll_interval: float = 0.5,
    ):
        self.store = memory_store
        self.conn = memory_store._conn
        self.entity_store = entity_store or EntityStore(self.conn)
        self.timeline_store = timeline_store or TimelineStore(self.conn)
        self.fact_store = fact_store or FactStore(self.conn)
        self.job_store = JobStore(self.conn)

        # Local extraction helpers (heuristic fallback without remote LLM required)
        self.fact_extractor = FactExtractor(llm=None)
        from core.memory_pipeline_v2.entity_extractor import EntityRegistry
        self.entity_extractor = EntityExtractor(registry=EntityRegistry(), llm=None)

        # Worker thread lifecycle
        self.poll_interval = poll_interval
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # Worker Lifecycle
    # -----------------------------------------------------------------------

    def start(self, recover_stale: bool = True) -> None:
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        if recover_stale:
            # Step 8: Crash recovery on startup
            self.job_store.recover_stale_jobs()

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="IngestionServiceWorker"
        )
        self._worker_thread.start()
        logger.info("[IngestionService] Background worker started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal worker to stop and wait for termination."""
        if self._worker_thread is not None:
            self._shutdown_event.set()
            self._worker_thread.join(timeout=timeout)
            logger.info("[IngestionService] Background worker stopped.")

    # -----------------------------------------------------------------------
    # Synchronous Evaluation Ingestion Path
    # -----------------------------------------------------------------------

    def ingest_sync(
        self,
        content: str,
        namespace_id: str = "default",
        tag: str | MemoryTag = MemoryTag.EPISODIC,
        importance_score: float = 5.0,
        keywords: Optional[List[str]] = None,
        event_time: Optional[float] = None,
        memory_id: Optional[str] = None,
        summary: Optional[str] = None,
        tier: MemoryTier = MemoryTier.L2_ACTIVE_RAM,
    ) -> MemoryBlock:
        """
        Synchronous evaluation ingestion path.
        Executes extraction, canonical storage, and derived index updates immediately.
        Preserves deterministic benchmark execution without queue latency.
        """
        block_id = memory_id or str(uuid.uuid4())
        mem_tag = MemoryTag(tag) if isinstance(tag, str) else tag

        block = MemoryBlock(
            id=block_id,
            namespace_id=namespace_id,
            content=content,
            summary=summary,
            tag=mem_tag,
            tier=tier,
            importance_score=importance_score,
            keywords=keywords or [],
            event_time=event_time,
            source="sync_ingest",
        )

        self._execute_pipeline(block)
        return block

    # -----------------------------------------------------------------------
    # Asynchronous Ingestion Job Submission
    # -----------------------------------------------------------------------

    def submit_job(
        self,
        content: str,
        namespace_id: str = "default",
        tag: str | MemoryTag = MemoryTag.EPISODIC,
        importance_score: float = 5.0,
        keywords: Optional[List[str]] = None,
        event_time: Optional[float] = None,
        memory_id: Optional[str] = None,
        summary: Optional[str] = None,
        max_retries: int = 3,
    ) -> IngestionJob:
        """
        Durable asynchronous ingestion submission.
        Persists job in SQLite and queues it for the background worker.
        """
        target_memory_id = memory_id or str(uuid.uuid4())
        tag_str = tag.value if isinstance(tag, MemoryTag) else str(tag)

        payload = {
            "content": content,
            "namespace_id": namespace_id,
            "tag": tag_str,
            "importance_score": importance_score,
            "keywords": keywords or [],
            "event_time": event_time,
            "summary": summary,
            "memory_id": target_memory_id,
        }

        job = self.job_store.create_job(
            payload=payload,
            namespace_id=namespace_id,
            memory_id=target_memory_id,
            max_retries=max_retries,
        )
        return job

    # -----------------------------------------------------------------------
    # Worker Execution Loop
    # -----------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                job = self.job_store.claim_next_job()
                if job is None:
                    time.sleep(self.poll_interval)
                    continue

                self._process_claimed_job(job)

            except Exception as e:
                logger.error("[IngestionService] Worker loop error: %s", e, exc_info=True)
                time.sleep(self.poll_interval)

    def _process_claimed_job(self, job: IngestionJob) -> None:
        """Execute a claimed job with retry and dead-letter handling."""
        try:
            p = job.payload
            mem_tag = MemoryTag(p.get("tag", "EPISODIC"))
            block = MemoryBlock(
                id=job.memory_id or p.get("memory_id") or str(uuid.uuid4()),
                namespace_id=job.namespace_id,
                content=p["content"],
                summary=p.get("summary"),
                tag=mem_tag,
                tier=MemoryTier.L2_ACTIVE_RAM,
                importance_score=float(p.get("importance_score", 5.0)),
                keywords=p.get("keywords") or [],
                event_time=p.get("event_time"),
                source="async_worker",
            )

            # Execute pipeline
            self._execute_pipeline(block)

            # Mark SUCCESS
            self.job_store.update_state(job.job_id, JobState.SUCCESS, memory_id=block.id)
            logger.info("[IngestionService] Job '%s' succeeded for memory '%s'.", job.job_id, block.id)

        except Exception as e:
            err_msg = str(e)
            logger.warning("[IngestionService] Job '%s' failed on attempt %d: %s", job.job_id, job.attempts, err_msg)
            now = get_clock().now()

            if job.attempts < job.max_retries:
                # Exponential retry backoff
                retry_delay = 2 ** job.attempts
                scheduled_at = now + retry_delay
                self.job_store.update_state(
                    job.job_id,
                    JobState.RETRY,
                    error=err_msg,
                    scheduled_at=scheduled_at,
                )
                logger.info("[IngestionService] Job '%s' scheduled for retry in %ds.", job.job_id, retry_delay)
            else:
                # Retries exhausted -> DEAD_LETTER
                self.job_store.update_state(
                    job.job_id,
                    JobState.DEAD_LETTER,
                    error=f"Exhausted {job.max_retries} attempts. Last error: {err_msg}",
                )
                logger.error("[IngestionService] Job '%s' DEAD-LETTERED after %d attempts.", job.job_id, job.attempts)

    # -----------------------------------------------------------------------
    # Canonical Pipeline Execution (Idempotent)
    # -----------------------------------------------------------------------

    def _execute_pipeline(self, block: MemoryBlock) -> None:
        """
        Executes idempotent ingestion:
        1. Clean any partial previous relational linkages for this memory_id.
        2. Save canonical MemoryBlock to SQLite + derived Chroma/FTS5 index.
        3. Extract and link structured facts, timeline events, and entity mentions.
        """
        # 1. Idempotency cleanup in case of retries
        with self.conn:
            self.conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", (block.id,))
            self.conn.execute("DELETE FROM memory_facts WHERE memory_id = ?", (block.id,))
            self.conn.execute("DELETE FROM memory_events WHERE memory_id = ?", (block.id,))
            self.conn.execute("DELETE FROM structured_facts WHERE memory_id = ?", (block.id,))

        # 2. Canonical memory block storage
        self.store.save(block)

        # 3. Timeline event extraction
        ts = datetime.fromtimestamp(block.created_at)
        if block.event_time is not None:
            title = block.content[:30]
            ev = self.timeline_store.create_event(
                namespace_id=block.namespace_id,
                title=title,
                event_type=EVENT_TYPE_POINT,
                start_time=block.event_time,
            )
            self.timeline_store.link_memory(block.id, ev.event_id)

        # 4. Structured fact extraction
        try:
            facts = self.fact_extractor.extract(
                block.id, block.content, "observation", ts
            )
            for fact in facts:
                valid_from = fact.temporal_bound.earliest.timestamp() if fact.temporal_bound and fact.temporal_bound.earliest else None
                valid_to = fact.temporal_bound.latest.timestamp() if fact.temporal_bound and fact.temporal_bound.latest else None
                f_rec = self.fact_store.create_fact(
                    namespace_id=block.namespace_id,
                    memory_id=block.id,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_val=fact.object,
                    fact_type=fact.fact_type,
                    confidence=fact.confidence,
                    valid_from=valid_from,
                    valid_to=valid_to,
                )
        except Exception as e:
            logger.debug("Fact extraction skipped or failed: %s", e)

        # 5. Entity extraction and linking
        try:
            linked_entities = self.entity_extractor.extract_and_link(
                block.id, block.content, ts
            )
            for le in linked_entities:
                ent = self.entity_store.get_entity_by_name(le.entity.canonical_name, block.namespace_id)
                if not ent:
                    ent = self.entity_store.create_entity(
                        namespace_id=block.namespace_id,
                        canonical_name=le.entity.canonical_name,
                        entity_type=le.entity.type.upper(),
                    )
                for alias in le.entity.aliases:
                    self.entity_store.add_alias(ent.entity_id, alias)
                self.entity_store.link_memory(block.id, ent.entity_id, confidence=0.90)
        except Exception as e:
            logger.debug("Entity extraction skipped or failed: %s", e)
