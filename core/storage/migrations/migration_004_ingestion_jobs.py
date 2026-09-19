"""
Project Almond V3 — Migration 004: Ingestion Jobs Hardening & Indices
Upgrades ingestion_jobs table to support durable state machine and tracking.
Safe, idempotent, and backwards-compatible with Migration 001.
"""

from __future__ import annotations
import logging
import sqlite3
from core.storage.migrations.migration_manager import register_migration

logger = logging.getLogger(__name__)


@register_migration(version=4, name="ingestion_jobs_hardening")
def migrate_ingestion_jobs(conn: sqlite3.Connection) -> None:
    # 1. Ensure table exists with base schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id       TEXT PRIMARY KEY,
            namespace_id TEXT NOT NULL DEFAULT 'default',
            payload_json TEXT NOT NULL,
            state        TEXT NOT NULL,
            attempts     INTEGER DEFAULT 0,
            max_retries  INTEGER DEFAULT 3,
            last_error   TEXT,
            created_at   REAL NOT NULL,
            scheduled_at REAL NOT NULL,
            completed_at REAL
        );
    """)

    # 2. Check for missing columns and add safely
    cursor = conn.execute("PRAGMA table_info(ingestion_jobs)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "memory_id" not in existing_cols:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN memory_id TEXT")
    if "updated_at" not in existing_cols:
        conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN updated_at REAL")
        conn.execute("UPDATE ingestion_jobs SET updated_at = created_at WHERE updated_at IS NULL")

    # 3. Create indices for worker polling, namespace isolation, and memory tracing
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_state_scheduled
        ON ingestion_jobs (state, scheduled_at);
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_namespace
        ON ingestion_jobs (namespace_id);
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_memory_id
        ON ingestion_jobs (memory_id);
    """)

    logger.info("Migration 004 (ingestion_jobs_hardening) applied successfully.")
