"""
Project Almond V3 — Migration 003: SQLite FTS5 Lexical Search Index
Creates memory_blocks_fts virtual table with automated synchronization triggers.
Fully idempotent migration — safely rerunnable without data duplication or disruption.
"""

from __future__ import annotations
import logging
import sqlite3
from core.storage.migrations.migration_manager import register_migration

logger = logging.getLogger(__name__)


@register_migration(version=3, name="fts5_lexical_index")
def migrate_fts5_lexical_index(conn: sqlite3.Connection) -> None:
    # -----------------------------------------------------------------------
    # 1. Create FTS5 virtual table for memory content and keywords
    # -----------------------------------------------------------------------
    # Using 'porter unicode61' tokenizer for stemming and case folding
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_blocks_fts USING fts5(
            id UNINDEXED,
            namespace_id UNINDEXED,
            content,
            keywords,
            tokenize = 'porter unicode61'
        );
    """)

    # -----------------------------------------------------------------------
    # 2. Idempotently create synchronization triggers
    # -----------------------------------------------------------------------
    conn.execute("DROP TRIGGER IF EXISTS trg_memory_blocks_fts_insert;")
    conn.execute("""
        CREATE TRIGGER trg_memory_blocks_fts_insert AFTER INSERT ON memory_blocks BEGIN
            INSERT INTO memory_blocks_fts (id, namespace_id, content, keywords)
            VALUES (new.id, new.namespace_id, new.content, COALESCE(new.keywords, ''));
        END;
    """)

    conn.execute("DROP TRIGGER IF EXISTS trg_memory_blocks_fts_update;")
    conn.execute("""
        CREATE TRIGGER trg_memory_blocks_fts_update AFTER UPDATE ON memory_blocks BEGIN
            DELETE FROM memory_blocks_fts WHERE id = old.id;
            INSERT INTO memory_blocks_fts (id, namespace_id, content, keywords)
            VALUES (new.id, new.namespace_id, new.content, COALESCE(new.keywords, ''));
        END;
    """)

    conn.execute("DROP TRIGGER IF EXISTS trg_memory_blocks_fts_delete;")
    conn.execute("""
        CREATE TRIGGER trg_memory_blocks_fts_delete AFTER DELETE ON memory_blocks BEGIN
            DELETE FROM memory_blocks_fts WHERE id = old.id;
        END;
    """)

    # -----------------------------------------------------------------------
    # 3. Idempotently backfill existing memory_blocks into memory_blocks_fts
    # -----------------------------------------------------------------------
    conn.execute("""
        INSERT INTO memory_blocks_fts (id, namespace_id, content, keywords)
        SELECT id, namespace_id, content, COALESCE(keywords, '')
        FROM memory_blocks
        WHERE id NOT IN (SELECT id FROM memory_blocks_fts);
    """)
    logger.info("Migration 003 (fts5_lexical_index) applied successfully.")
