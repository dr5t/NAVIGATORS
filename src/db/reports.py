"""
Navigators IDR - Community Reporting Repository
Manages community-reported flags and issues on places, contributions, and metadata,
feeding into the moderator triage queue.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.db.database import get_db, DEFAULT_DB_PATH


@dataclass
class Report:
    id: str
    reporter_id: str
    target_type: str
    target_id: str
    reason: str
    details: Optional[str] = None
    status: str = "pending"
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    created_at: str = ""
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "reporter_id": self.reporter_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "reason": self.reason,
            "details": self.details,
            "status": self.status,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


class ReportRepository:
    """Repository managing community reports and moderation resolutions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH

    def create_report(
        self,
        reporter_id: str,
        target_type: str,
        target_id: str,
        reason: str,
        details: Optional[str] = None,
    ) -> Report:
        """Create and submit a new community issue report."""
        report_id = f"rep_{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO reports (
                    id, reporter_id, target_type, target_id, reason,
                    details, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (report_id, reporter_id, target_type, target_id, reason.strip(), details, now),
            )

        return Report(
            id=report_id,
            reporter_id=reporter_id,
            target_type=target_type,
            target_id=target_id,
            reason=reason.strip(),
            details=details,
            status="pending",
            created_at=now,
        )

    def get_report(self, report_id: str) -> Optional[Report]:
        """Fetch report by unique ID."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM reports WHERE id = ?",
                (report_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Report(**dict(row))

    def list_reports(
        self,
        status: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Report]:
        """List reports filtered by status and target type."""
        clauses = []
        params: List[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status.lower())
        if target_type:
            clauses.append("target_type = ?")
            params.append(target_type.lower())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM reports {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            rows = cur.fetchall()
            return [Report(**dict(r)) for r in rows]

    def resolve_report(
        self,
        report_id: str,
        resolved_by: str,
        decision: str,
        notes: Optional[str] = None,
    ) -> Optional[Report]:
        """Moderator action to resolve or dismiss a report."""
        clean_status = decision.lower().strip()
        if clean_status not in ("resolved", "dismissed"):
            raise ValueError(f"Invalid decision '{decision}'. Expected 'resolved' or 'dismissed'.")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE reports
                SET status = ?, resolved_by = ?, resolution_notes = ?, resolved_at = ?
                WHERE id = ?
                """,
                (clean_status, resolved_by, notes, now, report_id),
            )

        return self.get_report(report_id)
