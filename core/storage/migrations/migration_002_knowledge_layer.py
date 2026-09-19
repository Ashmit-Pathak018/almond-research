"""
Project Almond V3 — Migration 002: Knowledge Layer Schema Enhancements
Adds entity_aliases table, fact supersession columns, and namespace isolation indexes.
Additive migration — no existing data is destroyed or restructured.
"""

import json
import sqlite3
import uuid
import time
from core.storage.migrations.migration_manager import register_migration


@register_migration(version=2, name="knowledge_layer_enhancement")
def migrate_knowledge_layer(conn: sqlite3.Connection) -> None:
    # -----------------------------------------------------------------------
    # 1. entity_aliases — Normalized alias table per V3 spec
    # -----------------------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_aliases (
            alias_id   TEXT PRIMARY KEY,
            entity_id  TEXT NOT NULL,
            alias_name TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
            UNIQUE(entity_id, alias_name)
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_aliases_name ON entity_aliases(alias_name);")

    # -----------------------------------------------------------------------
    # 2. Backfill entity_aliases from existing JSON aliases column
    # -----------------------------------------------------------------------
    cursor = conn.execute("SELECT id, aliases FROM entities WHERE aliases IS NOT NULL AND aliases != '[]'")
    now = time.time()
    for row in cursor.fetchall():
        entity_id = row[0]
        try:
            aliases = json.loads(row[1]) if row[1] else []
        except (json.JSONDecodeError, TypeError):
            aliases = []
        for alias_name in aliases:
            if alias_name and isinstance(alias_name, str) and alias_name.strip():
                alias_id = str(uuid.uuid4())
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO entity_aliases (alias_id, entity_id, alias_name, created_at) VALUES (?, ?, ?, ?)",
                        (alias_id, entity_id, alias_name.strip(), now)
                    )
                except sqlite3.IntegrityError:
                    pass  # Skip duplicates

    # -----------------------------------------------------------------------
    # 3. structured_facts — Add supersession and temporal validity columns
    # -----------------------------------------------------------------------
    cursor = conn.execute("PRAGMA table_info(structured_facts)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    cols_to_add = [
        ("namespace_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("state", "TEXT NOT NULL DEFAULT 'VERIFIED'"),
        ("superseded_by", "TEXT REFERENCES structured_facts(id)"),
        ("valid_from", "REAL"),
        ("valid_to", "REAL"),
    ]
    for col_name, col_def in cols_to_add:
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE structured_facts ADD COLUMN {col_name} {col_def}")

    # -----------------------------------------------------------------------
    # 4. Add namespace index on structured_facts if not already present
    # -----------------------------------------------------------------------
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_namespace ON structured_facts(namespace_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_state ON structured_facts(state);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject_pred ON structured_facts(subject, predicate);")

    # -----------------------------------------------------------------------
    # 5. Namespace FK index on events table
    # -----------------------------------------------------------------------
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_namespace ON events(namespace_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
