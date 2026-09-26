from __future__ import annotations
"""
Navigators IDR - Platform Audit Logging Engine
Provides immutable audit record persistence for all governance actions,
moderation events, role reassignments, and canonical map updates.
"""

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.db.database import get_db, DEFAULT_DB_PATH


@dataclass
class AuditEntry:
    id: str
    actor_id: Optional[str]
    action: str
    resource_type: str
    resource_id: str
    old_state: Optional[str] = None
    new_state: Optional[str] = None
    metadata_json: str = "{}"
    ip_hash: Optional[str] = None
    created_at: str = ""

    @property
    def metadata(self) -> Dict[str, Any]:
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "old_state": self.old_state,
            "new_state": self.new_state,
            "metadata": self.metadata,
            "ip_hash": self.ip_hash,
            "created_at": self.created_at,
        }


class AuditRepository:
    """Repository managing immutable audit log entries."""

    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is not None and isinstance(db_path, str):
            self._db_path = Path(db_path)
        else:
            self._db_path = db_path or DEFAULT_DB_PATH

    @property
    def db_path(self) -> Path:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Optional[str | Path]) -> None:
        if val is not None and isinstance(val, str):
            self._db_path = Path(val)
        else:
            self._db_path = val or DEFAULT_DB_PATH

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        actor_id: Optional[str] = None,
        old_state: Optional[str] = None,
        new_state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_hash: Optional[str] = None,
    ) -> AuditEntry:
        """Create and persist an immutable audit log record."""
        log_id = f"aud_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})

        with get_db(self.db_path) as conn:

            valid_actor_id = None
            if actor_id:
                cur = conn.execute("SELECT id FROM users WHERE id = ?", (actor_id,))
                if cur.fetchone():
                    valid_actor_id = actor_id

            conn.execute(
                """
                INSERT INTO audit_logs (
                    id, actor_id, action, resource_type, resource_id,
                    old_state, new_state, metadata_json, ip_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    valid_actor_id,
                    action,
                    resource_type,
                    resource_id,
                    old_state,
                    new_state,
                    meta_str,
                    ip_hash,
                    now,
                ),
            )

        return AuditEntry(
            id=log_id,
            actor_id=valid_actor_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_state=old_state,
            new_state=new_state,
            metadata_json=meta_str,
            ip_hash=ip_hash,
            created_at=now,
        )

    def get_log(self, log_id: str) -> Optional[AuditEntry]:
        """Fetch single audit log record by ID."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM audit_logs WHERE id = ?",
                (log_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return AuditEntry(**dict(row))

    def list_logs(
        self,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit log records with optional filtering."""
        clauses = []
        params: List[Any] = []

        if actor_id is not None:
            clauses.append("actor_id = ?")
            params.append(actor_id)
        if resource_type is not None:
            clauses.append("resource_type = ?")
            params.append(resource_type)
        if resource_id is not None:
            clauses.append("resource_id = ?")
            params.append(resource_id)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM audit_logs {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            rows = cur.fetchall()
            return [AuditEntry(**dict(r)) for r in rows]

    def count_logs(
        self,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        action: Optional[str] = None,
    ) -> int:
        """Count total matching audit log records."""
        clauses = []
        params: List[Any] = []

        if actor_id is not None:
            clauses.append("actor_id = ?")
            params.append(actor_id)
        if resource_type is not None:
            clauses.append("resource_type = ?")
            params.append(resource_type)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT COUNT(*) as cnt FROM audit_logs {where_sql}"

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            row = cur.fetchone()
            return row["cnt"] if row else 0

    def get_audit_summary(self) -> Dict[str, Any]:
        """Return aggregate audit statistics breakdown by resource type, action, and actor."""
        with get_db(self.db_path) as conn:
            total_cur = conn.execute("SELECT COUNT(*) as n FROM audit_logs")
            total = total_cur.fetchone()["n"]

            res_cur = conn.execute(
                "SELECT resource_type, COUNT(*) as n FROM audit_logs GROUP BY resource_type"
            )
            res_rows = res_cur.fetchall()

            act_cur = conn.execute(
                "SELECT action, COUNT(*) as n FROM audit_logs GROUP BY action"
            )
            act_rows = act_cur.fetchall()

            actor_cur = conn.execute(
                "SELECT actor_id, COUNT(*) as n FROM audit_logs WHERE actor_id IS NOT NULL GROUP BY actor_id ORDER BY n DESC LIMIT 10"
            )
            actor_rows = actor_cur.fetchall()

        return {
            "total_logs": total,
            "by_resource_type": {row["resource_type"]: row["n"] for row in res_rows},
            "by_action": {row["action"]: row["n"] for row in act_rows},
            "by_actor": {row["actor_id"]: row["n"] for row in actor_rows},
        }

