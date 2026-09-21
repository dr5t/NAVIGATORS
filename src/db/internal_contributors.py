"""
Navigators IDR - Internal Contributor Requests Repository
Manages application submissions, team reviews, role escalations,
and audit logging for internal contributor privileges.
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.db.database import get_db, DEFAULT_DB_PATH
from src.db.rbac import RBACRepository
from src.db.audit import AuditRepository


@dataclass
class InternalContributorRequest:
    id: str
    user_id: str
    reason: str
    experience: str
    requested_scope: str
    status: str
    rejection_reason: Optional[str]
    reviewed_by: Optional[str]
    reviewed_at: Optional[str]
    created_at: str
    updated_at: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    reviewer_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "reason": self.reason,
            "experience": self.experience,
            "requested_scope": self.requested_scope,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "reviewer_name": self.reviewer_name,
        }


class InternalContributorRepository:
    """Repository managing internal contributor applications and approvals."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.rbac_repo = RBACRepository(self.db_path)
        self.audit_repo = AuditRepository(self.db_path)

    def submit_request(
        self,
        user_id: str,
        reason: str,
        experience: str,
        requested_scope: str = "trajectories_and_models",
    ) -> InternalContributorRequest:
        """
        Submit a new application for internal contributor role.
        Prevents duplicate pending submissions and checks if role is already granted.
        """
        clean_reason = reason.strip()
        clean_experience = experience.strip()
        if not clean_reason:
            raise ValueError("Application reason cannot be empty")
        if not clean_experience:
            raise ValueError("Experience details cannot be empty")

        # Check existing user roles
        existing_roles = {r.id for r in self.rbac_repo.get_user_roles(user_id)}
        if "internal_contributor" in existing_roles:
            raise ValueError("User already holds the internal contributor role")

        now = datetime.now(timezone.utc).isoformat()
        request_id = f"icr_{secrets.token_hex(8)}"

        with get_db(self.db_path) as conn:
            # Check for existing pending request
            cur = conn.execute(
                "SELECT id FROM internal_contributor_requests WHERE user_id = ? AND status = 'pending'",
                (user_id,),
            )
            if cur.fetchone():
                raise ValueError("An active pending internal contributor application already exists for this user")

            conn.execute(
                """
                INSERT INTO internal_contributor_requests (
                    id, user_id, reason, experience, requested_scope,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (request_id, user_id, clean_reason, clean_experience, requested_scope, now, now),
            )

        self.audit_repo.log(
            actor_id=user_id,
            action="SUBMIT_INTERNAL_ACCESS_REQUEST",
            resource_type="internal_contributor_request",
            resource_id=request_id,
            old_state=None,
            new_state="pending",
            metadata={"requested_scope": requested_scope},
        )

        req = self.get_request(request_id)
        if not req:
            raise RuntimeError(f"Failed to load newly created request {request_id}")
        return req

    def get_request(self, request_id: str) -> Optional[InternalContributorRequest]:
        """Fetch single application by ID with applicant and reviewer metadata."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT
                    r.id, r.user_id, r.reason, r.experience, r.requested_scope,
                    r.status, r.rejection_reason, r.reviewed_by, r.reviewed_at,
                    r.created_at, r.updated_at,
                    u.name AS user_name, u.email AS user_email,
                    rev.name AS reviewer_name
                FROM internal_contributor_requests r
                LEFT JOIN users u ON r.user_id = u.id
                LEFT JOIN users rev ON r.reviewed_by = rev.id
                WHERE r.id = ?
                """,
                (request_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            return InternalContributorRequest(
                id=row["id"],
                user_id=row["user_id"],
                reason=row["reason"],
                experience=row["experience"],
                requested_scope=row["requested_scope"],
                status=row["status"],
                rejection_reason=row["rejection_reason"],
                reviewed_by=row["reviewed_by"],
                reviewed_at=row["reviewed_at"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                user_name=row["user_name"],
                user_email=row["user_email"],
                reviewer_name=row["reviewer_name"],
            )

    def get_user_latest_request(self, user_id: str) -> Optional[InternalContributorRequest]:
        """Retrieve latest application for a specific user to determine status in client UI."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT id FROM internal_contributor_requests
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self.get_request(row["id"])

    def list_requests(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[InternalContributorRequest], int]:
        """List applications with optional status filtering for team admin triage."""
        with get_db(self.db_path) as conn:
            where_clauses = []
            params: List[Any] = []

            if status:
                where_clauses.append("r.status = ?")
                params.append(status)

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            count_cur = conn.execute(
                f"SELECT COUNT(*) AS total FROM internal_contributor_requests r {where_str}",
                params,
            )
            total = count_cur.fetchone()["total"]

            cur = conn.execute(
                f"""
                SELECT
                    r.id, r.user_id, r.reason, r.experience, r.requested_scope,
                    r.status, r.rejection_reason, r.reviewed_by, r.reviewed_at,
                    r.created_at, r.updated_at,
                    u.name AS user_name, u.email AS user_email,
                    rev.name AS reviewer_name
                FROM internal_contributor_requests r
                LEFT JOIN users u ON r.user_id = u.id
                LEFT JOIN users rev ON r.reviewed_by = rev.id
                {where_str}
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            )

            items = [
                InternalContributorRequest(
                    id=row["id"],
                    user_id=row["user_id"],
                    reason=row["reason"],
                    experience=row["experience"],
                    requested_scope=row["requested_scope"],
                    status=row["status"],
                    rejection_reason=row["rejection_reason"],
                    reviewed_by=row["reviewed_by"],
                    reviewed_at=row["reviewed_at"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    user_name=row["user_name"],
                    user_email=row["user_email"],
                    reviewer_name=row["reviewer_name"],
                )
                for row in cur.fetchall()
            ]

            return items, total

    def approve_request(self, request_id: str, reviewer_id: str) -> InternalContributorRequest:
        """
        Approve an application:
        1. Transitions status to 'approved'.
        2. Assigns 'internal_contributor' role to the applicant.
        3. Writes immutable audit log record.
        """
        req = self.get_request(request_id)
        if not req:
            raise ValueError(f"Application '{request_id}' not found")
        if req.status != "pending":
            raise ValueError(f"Cannot approve application in '{req.status}' state (must be pending)")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE internal_contributor_requests
                SET status = 'approved',
                    reviewed_by = ?,
                    reviewed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (reviewer_id, now, now, request_id),
            )

        # Escalate role to internal_contributor
        self.rbac_repo.assign_role_to_user(req.user_id, "internal_contributor")

        # Record audit log
        self.audit_repo.log(
            actor_id=reviewer_id,
            action="APPROVE_INTERNAL_ACCESS",
            resource_type="internal_contributor_request",
            resource_id=request_id,
            old_state="pending",
            new_state="approved",
            metadata={
                "applicant_id": req.user_id,
                "assigned_role": "internal_contributor",
                "requested_scope": req.requested_scope,
            },
        )

        updated = self.get_request(request_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve updated request {request_id}")
        return updated

    def reject_request(
        self,
        request_id: str,
        reviewer_id: str,
        rejection_reason: str,
    ) -> InternalContributorRequest:
        """
        Reject an application:
        1. Transitions status to 'rejected'.
        2. Stores required rejection explanation.
        3. Writes immutable audit log record.
        """
        clean_reason = rejection_reason.strip()
        if not clean_reason:
            raise ValueError("Rejection reason cannot be empty")

        req = self.get_request(request_id)
        if not req:
            raise ValueError(f"Application '{request_id}' not found")
        if req.status != "pending":
            raise ValueError(f"Cannot reject application in '{req.status}' state (must be pending)")

        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                UPDATE internal_contributor_requests
                SET status = 'rejected',
                    rejection_reason = ?,
                    reviewed_by = ?,
                    reviewed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (clean_reason, reviewer_id, now, now, request_id),
            )

        self.audit_repo.log(
            actor_id=reviewer_id,
            action="REJECT_INTERNAL_ACCESS",
            resource_type="internal_contributor_request",
            resource_id=request_id,
            old_state="pending",
            new_state="rejected",
            metadata={
                "applicant_id": req.user_id,
                "rejection_reason": clean_reason,
            },
        )

        updated = self.get_request(request_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve updated request {request_id}")
        return updated
