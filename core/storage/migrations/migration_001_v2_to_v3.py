"""
Project Almond V3 — Migration 001: V2 to V3 Baseline Migration
Upgrades V2 memory tables into V3 schema with namespaces, bi-temporal timestamps, state, and link tables.
"""

import sqlite3
from core.storage.migrations.migration_manager import register_migration

@register_migration(version=1, name="v2_to_v3_baseline")
def migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    # 1. Namespaces
    conn.execute("""
        CREATE TABLE IF NOT EXISTS namespaces (
            namespace_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at REAL NOT NULL,
            metadata_json TEXT DEFAULT '{}'
        );
    """)
    conn.execute("""
        INSERT OR IGNORE INTO namespaces (namespace_id, name, description, created_at)
        VALUES ('default', 'Default Namespace', 'Primary tenant namespace', strftime('%s', 'now'));
    """)

    # 2. Upgrade memory_blocks table if it exists
    # Check existing columns in memory_blocks
    cursor = conn.execute("PRAGMA table_info(memory_blocks)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if existing_cols:
        # Columns to add safely
        cols_to_add = [
            ("namespace_id", "TEXT DEFAULT 'default'"),
            ("event_time", "REAL"),
            ("updated_at", "REAL"),
            ("state", "TEXT DEFAULT 'ACTIVE'")
        ]
        for col_name, col_def in cols_to_add:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE memory_blocks ADD COLUMN {col_name} {col_def}")

        # Backfill updated_at from created_at where NULL
        conn.execute("UPDATE memory_blocks SET updated_at = created_at WHERE updated_at IS NULL")
    else:
        # Create full V3 memory_blocks table if starting fresh
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_blocks (
                id               TEXT PRIMARY KEY,
                namespace_id     TEXT NOT NULL DEFAULT 'default',
                content          TEXT NOT NULL,
                summary          TEXT,
                tag              TEXT NOT NULL,
                tier             TEXT NOT NULL,
                state            TEXT NOT NULL DEFAULT 'ACTIVE',
                importance_score REAL NOT NULL,
                keywords         TEXT NOT NULL,
                source           TEXT NOT NULL,
                session_id       TEXT,
                event_time       REAL,
                created_at       REAL NOT NULL,
                updated_at       REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                access_count     INTEGER NOT NULL,
                FOREIGN KEY (namespace_id) REFERENCES namespaces(namespace_id) ON DELETE CASCADE
            )
        """)

    # Indexes on memory_blocks
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memory_blocks(namespace_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tier ON memory_blocks(tier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_state ON memory_blocks(state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_event_time ON memory_blocks(event_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memory_blocks(last_accessed_at)")

    # 3. Entities & Entity Aliases
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id              TEXT PRIMARY KEY,
            namespace_id    TEXT NOT NULL DEFAULT 'default',
            name            TEXT NOT NULL,
            type            TEXT NOT NULL,
            aliases         TEXT DEFAULT '[]',
            first_seen      TEXT,
            last_seen       TEXT,
            mention_count   INTEGER DEFAULT 1,
            summary         TEXT,
            metadata_json   TEXT DEFAULT '{}',
            created_at      REAL NOT NULL DEFAULT 0.0,
            updated_at      REAL NOT NULL DEFAULT 0.0
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id            TEXT NOT NULL,
            entity_id            TEXT NOT NULL,
            mention_offset_start INTEGER,
            mention_offset_end   INTEGER,
            confidence           REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (memory_id, entity_id),
            FOREIGN KEY (memory_id) REFERENCES memory_blocks(id) ON DELETE CASCADE,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_memory_map (
            entity_id  TEXT NOT NULL,
            memory_id  TEXT NOT NULL,
            PRIMARY KEY (entity_id, memory_id)
        );
    """)

    # 4. Structured Facts & Link Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structured_facts (
            id                  TEXT PRIMARY KEY,
            memory_id           TEXT NOT NULL,
            namespace_id        TEXT NOT NULL DEFAULT 'default',
            subject             TEXT NOT NULL,
            predicate           TEXT NOT NULL,
            object              TEXT NOT NULL,
            fact_type           TEXT NOT NULL,
            state               TEXT NOT NULL DEFAULT 'VERIFIED',
            confidence          REAL NOT NULL,
            date_raw            TEXT DEFAULT '',
            earliest            TEXT,
            latest              TEXT,
            temporal_confidence REAL DEFAULT 0.0,
            granularity         TEXT DEFAULT 'unknown',
            extraction_method   TEXT DEFAULT 'heuristic',
            superseded_by       TEXT,
            needs_review        INTEGER DEFAULT 0,
            has_conflict        INTEGER DEFAULT 0,
            created_at          REAL NOT NULL DEFAULT 0.0
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_memory ON structured_facts(memory_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON structured_facts(predicate)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
            memory_id TEXT NOT NULL,
            fact_id   TEXT NOT NULL,
            PRIMARY KEY (memory_id, fact_id),
            FOREIGN KEY (memory_id) REFERENCES memory_blocks(id) ON DELETE CASCADE,
            FOREIGN KEY (fact_id) REFERENCES structured_facts(id) ON DELETE CASCADE
        );
    """)

    # 5. Timeline Events
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id        TEXT PRIMARY KEY,
            namespace_id    TEXT NOT NULL DEFAULT 'default',
            title           TEXT NOT NULL,
            description     TEXT,
            event_type      TEXT NOT NULL DEFAULT 'POINT',
            start_time      REAL,
            end_time        REAL,
            relative_anchor TEXT,
            confidence      REAL NOT NULL DEFAULT 1.0,
            created_at      REAL NOT NULL DEFAULT 0.0
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_start_time ON events(start_time)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_events (
            memory_id TEXT NOT NULL,
            event_id  TEXT NOT NULL,
            PRIMARY KEY (memory_id, event_id),
            FOREIGN KEY (memory_id) REFERENCES memory_blocks(id) ON DELETE CASCADE,
            FOREIGN KEY (event_id) REFERENCES events(event_id) ON DELETE CASCADE
        );
    """)

    # 6. Persistent Ingestion Jobs Table (for hardened MemoryWorker)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state_sched ON ingestion_jobs(state, scheduled_at)")

    # 7. Retrieval Traces & Audit Events
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_traces (
            trace_id                 TEXT PRIMARY KEY,
            namespace_id             TEXT NOT NULL DEFAULT 'default',
            query                    TEXT NOT NULL,
            intent_detected          TEXT NOT NULL,
            reference_time           REAL NOT NULL,
            channels_used            TEXT NOT NULL,
            candidates_retrieved     INTEGER NOT NULL,
            candidates_ranked        INTEGER NOT NULL,
            selected_memory_ids_json TEXT NOT NULL,
            trace_payload_json       TEXT NOT NULL,
            duration_ms              REAL NOT NULL,
            created_at               REAL NOT NULL
        );
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_events (
            audit_id     TEXT PRIMARY KEY,
            namespace_id TEXT NOT NULL DEFAULT 'default',
            event_type   TEXT NOT NULL,
            target_id    TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at   REAL NOT NULL
        );
    """)
