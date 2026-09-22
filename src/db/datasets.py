"""
Navigators IDR - Dataset Sessions Repository (Phase 13)
Manages internal contributor dataset session submissions, lifecycle state machine,
validation workflow, and audit logging.

State machine:
  UPLOADED → VALIDATING → VALIDATED
       ↓           ↓
    REJECTED    REJECTED
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from src.db.database import get_db, DEFAULT_DB_PATH
from src.db.audit import AuditRepository


ACTIVITY_TYPES = frozenset({"walking", "driving", "gnss_imu", "gnss_outage", "validation"})


VALID_TRANSITIONS: Dict[str, frozenset] = {
    "uploaded":   frozenset({"validating", "rejected"}),
    "validating": frozenset({"validated", "rejected"}),
    "validated":  frozenset(),
    "rejected":   frozenset(),
}


@dataclass
class DatasetSession:
    id: str
    contributor_id: str
    activity_type: str
    device: str
    duration_seconds: float
    sensor_data_path: Optional[str]
    gnss_available: bool
    consent: bool
    status: str
    rejection_reason: Optional[str]
    validated_by: Optional[str]
    validated_at: Optional[str]
    notes: Optional[str]
    created_at: str
    updated_at: str
    contributor_name: Optional[str] = None
    contributor_email: Optional[str] = None
    validator_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "contributor_id": self.contributor_id,
            "contributor_name": self.contributor_name,
            "contributor_email": self.contributor_email,
            "activity_type": self.activity_type,
            "device": self.device,
            "duration_seconds": self.duration_seconds,
            "sensor_data_path": self.sensor_data_path,
            "gnss_available": self.gnss_available,
            "consent": self.consent,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "validated_by": self.validated_by,
            "validator_name": self.validator_name,
            "validated_at": self.validated_at,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class DatasetRepository:
    """Repository managing dataset session submissions and the validation lifecycle."""

    _SELECT_FULL = """
        SELECT
            s.id, s.contributor_id, s.activity_type, s.device,
            s.duration_seconds, s.sensor_data_path,
            s.gnss_available, s.consent, s.status,
            s.rejection_reason, s.validated_by, s.validated_at,
            s.notes, s.created_at, s.updated_at,
            u.name  AS contributor_name,
            u.email AS contributor_email,
            v.name  AS validator_name
        FROM dataset_sessions s
        LEFT JOIN users u ON s.contributor_id = u.id
        LEFT JOIN users v ON s.validated_by   = v.id
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self.audit_repo = AuditRepository(self._db_path)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Path) -> None:
        self._db_path = val
        self.audit_repo.db_path = val





    def _row_to_obj(self, row) -> DatasetSession:
        return DatasetSession(
            id=row["id"],
            contributor_id=row["contributor_id"],
            activity_type=row["activity_type"],
            device=row["device"],
            duration_seconds=float(row["duration_seconds"]),
            sensor_data_path=row["sensor_data_path"],
            gnss_available=bool(row["gnss_available"]),
            consent=bool(row["consent"]),
            status=row["status"],
            rejection_reason=row["rejection_reason"],
            validated_by=row["validated_by"],
            validated_at=row["validated_at"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            contributor_name=row["contributor_name"],
            contributor_email=row["contributor_email"],
            validator_name=row["validator_name"],
        )

    def _assert_transition(self, current: str, target: str) -> None:
        allowed = VALID_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise ValueError(
                f"Invalid state transition: '{current}' → '{target}'. "
                f"Allowed from '{current}': {sorted(allowed) or 'none (terminal state)'}"
            )





    def submit_session(
        self,
        contributor_id: str,
        activity_type: str,
        device: str,
        duration_seconds: float = 0.0,
        gnss_available: bool = True,
        consent: bool = True,
        sensor_data_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> DatasetSession:
        """
        Submit a new dataset session.
        Status starts as 'uploaded'. Requires explicit consent=True.
        """
        if activity_type not in ACTIVITY_TYPES:
            raise ValueError(
                f"Invalid activity_type '{activity_type}'. "
                f"Must be one of: {sorted(ACTIVITY_TYPES)}"
            )
        if not device.strip():
            raise ValueError("Device identifier cannot be empty")
        if not consent:
            raise ValueError("Contributor must provide explicit data consent to submit a session")
        if duration_seconds < 0:
            raise ValueError("Duration cannot be negative")

        session_id = f"ds_{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc).isoformat()

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO dataset_sessions (
                    id, contributor_id, activity_type, device,
                    duration_seconds, sensor_data_path,
                    gnss_available, consent, status,
                    notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploaded', ?, ?, ?)
                """,
                (
                    session_id, contributor_id, activity_type, device.strip(),
                    duration_seconds, sensor_data_path,
                    int(gnss_available), int(consent),
                    notes, now, now,
                ),
            )

        self.audit_repo.log(
            actor_id=contributor_id,
            action="SUBMIT_DATASET_SESSION",
            resource_type="dataset_session",
            resource_id=session_id,
            old_state=None,
            new_state="uploaded",
            metadata={
                "activity_type": activity_type,
                "device": device,
                "gnss_available": gnss_available,
            },
        )

        session = self.get_session(session_id)
        if not session:
            raise RuntimeError(f"Failed to load newly created session {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[DatasetSession]:
        """Fetch single session by ID with contributor and validator metadata."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                f"{self._SELECT_FULL} WHERE s.id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            return self._row_to_obj(row) if row else None

    def list_sessions(
        self,
        status: Optional[str] = None,
        contributor_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[DatasetSession], int]:
        """List sessions with optional filters. Returns (items, total_count)."""
        clauses: List[str] = []
        params: List[Any] = []

        if status:
            clauses.append("s.status = ?")
            params.append(status)
        if contributor_id:
            clauses.append("s.contributor_id = ?")
            params.append(contributor_id)
        if activity_type:
            clauses.append("s.activity_type = ?")
            params.append(activity_type)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with get_db(self.db_path) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM dataset_sessions s {where}", params
            ).fetchone()["n"]

            rows = conn.execute(
                f"{self._SELECT_FULL} {where} ORDER BY s.created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        return [self._row_to_obj(r) for r in rows], total

    def start_validation(self, session_id: str, validator_id: str) -> DatasetSession:
        """
        Transition uploaded → validating.
        Called when a team admin opens a session for review.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Dataset session '{session_id}' not found")
        self._assert_transition(session.status, "validating")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE dataset_sessions
                SET status = 'validating', validated_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (validator_id, now, session_id),
            )

        self.audit_repo.log(
            actor_id=validator_id,
            action="START_DATASET_VALIDATION",
            resource_type="dataset_session",
            resource_id=session_id,
            old_state="uploaded",
            new_state="validating",
            metadata={"contributor_id": session.contributor_id},
        )

        updated = self.get_session(session_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve session {session_id} after start_validation")
        return updated

    def validate_session(self, session_id: str, validator_id: str) -> DatasetSession:
        """
        Transition validating → validated.
        Makes the session eligible for inclusion in training runs.
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Dataset session '{session_id}' not found")
        self._assert_transition(session.status, "validated")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE dataset_sessions
                SET status = 'validated', validated_by = ?, validated_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (validator_id, now, now, session_id),
            )

        self.audit_repo.log(
            actor_id=validator_id,
            action="VALIDATE_DATASET_SESSION",
            resource_type="dataset_session",
            resource_id=session_id,
            old_state="validating",
            new_state="validated",
            metadata={"contributor_id": session.contributor_id},
        )

        updated = self.get_session(session_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve session {session_id} after validate_session")
        return updated

    def reject_session(
        self,
        session_id: str,
        validator_id: str,
        rejection_reason: str,
    ) -> DatasetSession:
        """
        Transition uploaded|validating → rejected.
        Rejection reason is mandatory.
        """
        clean_reason = rejection_reason.strip()
        if not clean_reason:
            raise ValueError("Rejection reason cannot be empty")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Dataset session '{session_id}' not found")
        self._assert_transition(session.status, "rejected")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE dataset_sessions
                SET status = 'rejected',
                    rejection_reason = ?,
                    validated_by = ?,
                    validated_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (clean_reason, validator_id, now, now, session_id),
            )

        self.audit_repo.log(
            actor_id=validator_id,
            action="REJECT_DATASET_SESSION",
            resource_type="dataset_session",
            resource_id=session_id,
            old_state=session.status,
            new_state="rejected",
            metadata={
                "contributor_id": session.contributor_id,
                "rejection_reason": clean_reason,
            },
        )

        updated = self.get_session(session_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve session {session_id} after reject_session")
        return updated

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate counts by status and activity type for dashboard."""
        with get_db(self.db_path) as conn:
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM dataset_sessions GROUP BY status"
            ).fetchall()
            activity_rows = conn.execute(
                "SELECT activity_type, COUNT(*) AS n FROM dataset_sessions GROUP BY activity_type"
            ).fetchall()
            validated_total = conn.execute(
                "SELECT COUNT(*) AS n FROM dataset_sessions WHERE status = 'validated'"
            ).fetchone()["n"]

        by_status = {row["status"]: row["n"] for row in status_rows}
        by_activity = {row["activity_type"]: row["n"] for row in activity_rows}

        return {
            "total": sum(by_status.values()),
            "by_status": {
                "uploaded":   by_status.get("uploaded", 0),
                "validating": by_status.get("validating", 0),
                "validated":  by_status.get("validated", 0),
                "rejected":   by_status.get("rejected", 0),
            },
            "by_activity": {t: by_activity.get(t, 0) for t in sorted(ACTIVITY_TYPES)},
            "validated_for_training": validated_total,
        }
