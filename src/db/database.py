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
        # Migrate existing contributions table if check constraints or columns are outdated
        table_check = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='contributions'"
        ).fetchone()
        if table_check and table_check["sql"]:
            current_sql = table_check["sql"]
            if "changes_requested" not in current_sql or "published_at" not in current_sql:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS contributions_migrated (
                        id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        resource_type TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'submitted', 'pending_review', 'pending', 'changes_requested', 'approved', 'published', 'rejected', 'withdrawn')),
                        title TEXT NOT NULL,
                        data_json TEXT NOT NULL DEFAULT '{}',
                        target_resource_id TEXT REFERENCES places(id) ON DELETE SET NULL,
                        action TEXT NOT NULL DEFAULT 'create' CHECK (action IN ('create', 'update', 'delete')),
                        reviewed_by TEXT REFERENCES users(id),
                        review_notes TEXT,
                        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                        updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                        reviewed_at TEXT,
                        published_at TEXT
                    )
                """)
                cols = [row["name"] for row in conn.execute("PRAGMA table_info(contributions)").fetchall()]
                target_cols = ['id', 'owner_id', 'resource_type', 'status', 'title', 'data_json', 'target_resource_id', 'action', 'reviewed_by', 'review_notes', 'created_at', 'updated_at', 'reviewed_at', 'published_at']
                common_cols = [c for c in target_cols if c in cols]
                cols_str = ", ".join(common_cols)
                conn.execute(f"INSERT INTO contributions_migrated ({cols_str}) SELECT {cols_str} FROM contributions")
                conn.execute("DROP TABLE contributions")
                conn.execute("ALTER TABLE contributions_migrated RENAME TO contributions")
                conn.execute("PRAGMA foreign_keys = ON")

        conn.executescript(schema_sql)
