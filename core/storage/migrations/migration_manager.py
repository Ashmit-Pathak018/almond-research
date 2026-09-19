"""
Project Almond V3 — Migration Manager
Executes versioned migrations within transactions to ensure zero data loss.
"""

from __future__ import annotations
import hashlib
import logging
import sqlite3
import time
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)

MigrationFunc = Callable[[sqlite3.Connection], None]

class Migration:
    def __init__(self, version: int, name: str, func: MigrationFunc):
        self.version = version
        self.name = name
        self.func = func

    @property
    def checksum(self) -> str:
        return hashlib.sha256(f"{self.version}:{self.name}".encode()).hexdigest()[:16]


# Registry of all V3 migrations in execution order
_MIGRATIONS: List[Migration] = []


def register_migration(version: int, name: str):
    """Decorator to register a migration function."""
    def decorator(fn: MigrationFunc):
        _MIGRATIONS.append(Migration(version=version, name=name, func=fn))
        _MIGRATIONS.sort(key=lambda m: m.version)
        return fn
    return decorator


def init_schema_version_table(conn: sqlite3.Connection) -> None:
    """Ensure schema_version table exists."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                migration_name TEXT NOT NULL,
                applied_at REAL NOT NULL,
                checksum TEXT NOT NULL
            );
        """)


def get_current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version, or 0 if none applied."""
    init_schema_version_table(conn)
    cursor = conn.execute("SELECT MAX(version) FROM schema_version")
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(row[0])
    return 0


def apply_migrations(conn: sqlite3.Connection) -> int:
    """
    Applies all pending migrations in order.
    Returns the latest applied schema version.
    """
    init_schema_version_table(conn)
    current_v = get_current_version(conn)

    for m in _MIGRATIONS:
        if m.version > current_v:
            logger.info("Applying Almond migration %03d: %s...", m.version, m.name)
            t0 = time.time()
            with conn:
                # Run the migration inside transaction
                m.func(conn)
                # Record migration version
                conn.execute(
                    "INSERT INTO schema_version (version, migration_name, applied_at, checksum) VALUES (?, ?, ?, ?)",
                    (m.version, m.name, t0, m.checksum)
                )
            logger.info("Applied migration %03d (%s) in %.3fs", m.version, m.name, time.time() - t0)
            current_v = m.version

    return current_v
