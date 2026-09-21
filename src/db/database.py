"""
Navigators IDR - Database Engine & Connection Management
Manages SQLite database connections, schema migrations, and connection lifecycles.
"""

import sqlite3
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

# Default database location within the project repository
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "navigators.db"
SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_db_path(custom_path: str | Path | None = None) -> Path:
    """Resolve active database file path, ensuring parent directory exists."""
    path = Path(custom_path) if custom_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """
    Establish a connection to SQLite database with foreign keys enabled
    and row factory configured for dictionary-like column access.
    """
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database transactions.
    Automatically commits on success or rolls back on exception.
    """
    conn = connect_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """
    Execute DDL schema migrations and seed initial roles, permissions,
    and role-permission mappings idempotently.
    """
    if not SCHEMA_SQL_PATH.exists():
        raise FileNotFoundError(f"Schema definition not found at {SCHEMA_SQL_PATH}")

    with open(SCHEMA_SQL_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db(db_path) as conn:
        # Migrate existing contributions table if new columns are missing before executing schema script
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='contributions'"
        ).fetchone()
        if table_check:
            columns = [row["name"] for row in conn.execute("PRAGMA table_info(contributions)").fetchall()]
            if "target_resource_id" not in columns:
                conn.execute("ALTER TABLE contributions ADD COLUMN target_resource_id TEXT REFERENCES places(id) ON DELETE SET NULL")
            if "action" not in columns:
                conn.execute("ALTER TABLE contributions ADD COLUMN action TEXT NOT NULL DEFAULT 'create'")

        conn.executescript(schema_sql)
