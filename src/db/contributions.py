"""
Navigators IDR - Community Contribution Repository & Data Layer
Manages contribution entities, drafts, pending reviews, and moderation decisions
enforced strictly via the ContributionStateMachine.
"""

import json
import sqlite3
import secrets
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.db.database import get_db
from src.db.state_machine import ContributionStateMachine, ContributionState, StateTransitionError


@dataclass
class Contribution:
    id: str
    owner_id: str
    resource_type: str
    status: str
    title: str
    data_json: str
    reviewed_by: Optional[str]
    review_notes: Optional[str]
    created_at: str
    updated_at: str
    reviewed_at: Optional[str]
    published_at: Optional[str] = None
    target_resource_id: Optional[str] = None
    action: str = "create"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        try:
            data["data"] = json.loads(self.data_json)
        except (ValueError, TypeError):
            data["data"] = {}
        return data


class ContributionRepository:
    """Data access layer for community map contributions."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self._db_path = db_path

    @property
    def db_path(self) -> Optional[str | Path]:
        return self._db_path

    @db_path.setter
    def db_path(self, val: Optional[str | Path]):
        self._db_path = val

    def create(
        self,
        contribution_id: str,
        owner_id: str,
        resource_type: str,
        title: str,
        data: Optional[Dict[str, Any]] = None,
        status: str = ContributionState.DRAFT,
        target_resource_id: Optional[str] = None,
        action: str = "create",
    ) -> Contribution:
        """Create a new contribution record (default status is 'draft')."""
        now = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(data or {})

        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO contributions (
                    id, owner_id, resource_type, status, title, data_json,
                    target_resource_id, action, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (contribution_id, owner_id, resource_type, status, title.strip(), payload_str, target_resource_id, action, now, now),
            )

        item = self.get(contribution_id)
        if not item:
            raise RuntimeError(f"Failed to create contribution {contribution_id}.")

        try:
            from src.db.audit import AuditRepository
            audit_repo = AuditRepository(self.db_path)
            audit_repo.log(
                action="CREATE",
                resource_type="contribution",
                resource_id=contribution_id,
                actor_id=owner_id,
                old_state=None,
                new_state=status,
                metadata={"title": title, "action": action, "target_resource_id": target_resource_id},
            )
        except Exception:
            pass

        return item

    def create_contribution(
        self,
        owner_id: str,
        resource_type: str,
        title: str,
        data: Optional[Dict[str, Any]] = None,
        status: str = ContributionState.DRAFT,
        target_resource_id: Optional[str] = None,
        action: str = "create",
        contribution_id: Optional[str] = None,
    ) -> Contribution:
        """Convenience factory method generating a unique ID if omitted."""
        cid = contribution_id or f"contrib_{secrets.token_hex(6)}"
        return self.create(
            contribution_id=cid,
            owner_id=owner_id,
            resource_type=resource_type,
            title=title,
            data=data,
            status=status,
            target_resource_id=target_resource_id,
            action=action,
        )

    def get(self, contribution_id: str) -> Optional[Contribution]:
        """Fetch contribution by unique ID."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM contributions WHERE id = ?",
                (contribution_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Contribution(**dict(row))

    def get_contribution(self, contribution_id: str) -> Optional[Contribution]:
        """Alias for get()."""
        return self.get(contribution_id)

    def update_content(
        self,
        contribution_id: str,
        title: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Contribution]:
        """
        Update mutable content (title, data payload) of a contribution.
        Does NOT allow changing status directly.
        """
        fields = []
        values = []
        if title is not None:
            fields.append("title = ?")
            values.append(title.strip())
        if data is not None:
            fields.append("data_json = ?")
            values.append(json.dumps(data))

        if not fields:
            return self.get(contribution_id)

        values.append(contribution_id)
        query = f"UPDATE contributions SET {', '.join(fields)} WHERE id = ?"
        with get_db(self.db_path) as conn:
            conn.execute(query, tuple(values))

        return self.get(contribution_id)

    def update(
        self,
        contribution_id: str,
        title: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Contribution]:
        """Alias for update_content."""
        return self.update_content(contribution_id, title=title, data=data)

    def transition_state(
        self,
        contribution_id: str,
        target_state: str,
        user: Any,
        notes: Optional[str] = None,
    ) -> Contribution:
        """
        Execute an atomic state transition validated by ContributionStateMachine.
        Prevents arbitrary or out-of-order state transitions.
        """
        item = self.get(contribution_id)
        if not item:
            raise ValueError(f"Contribution '{contribution_id}' not found.")

        # Validate with formal state machine
        allowed, reason, code = ContributionStateMachine.can_transition(
            current_state=item.status,
            target_state=target_state,
            user=user,
            resource=item,
            db_path=self.db_path,
        )
        if not allowed:
            raise StateTransitionError(reason, code=code)

        now = datetime.now(timezone.utc).isoformat()
        user_id = None
        if hasattr(user, "user") and user.user:
            user_id = user.user.id
        elif hasattr(user, "id"):
            user_id = user.id
        elif isinstance(user, dict):
            user_id = user.get("id") or user.get("user_id")
        elif isinstance(user, str):
            user_id = user

        with get_db(self.db_path) as conn:
            if target_state in (ContributionState.APPROVED, ContributionState.REJECTED, ContributionState.CHANGES_REQUESTED):
                conn.execute(
                    """
                    UPDATE contributions
                    SET status = ?, reviewed_by = ?, review_notes = ?, reviewed_at = ?
                    WHERE id = ?
                    """,
                    (target_state, user_id, notes, now, contribution_id),
                )
            elif target_state == ContributionState.PUBLISHED:
                conn.execute(
                    """
                    UPDATE contributions
                    SET status = ?, published_at = ?
                    WHERE id = ?
                    """,
                    (target_state, now, contribution_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE contributions
                    SET status = ?
                    WHERE id = ?
                    """,
                    (target_state, contribution_id),
                )

        updated = self.get(contribution_id)
        if not updated:
            raise RuntimeError(f"State transition to '{target_state}' failed.")

        # Record immutable audit log
        try:
            from src.db.audit import AuditRepository
            audit_repo = AuditRepository(self.db_path)
            audit_repo.log(
                action=f"contribution:{target_state}",
                resource_type="contribution",
                resource_id=contribution_id,
                actor_id=user_id,
                old_state=item.status,
                new_state=target_state,
                metadata={"title": item.title, "notes": notes, "action": item.action},
            )
        except Exception:
            pass

        if target_state == ContributionState.PUBLISHED and updated.resource_type == "place":
            try:
                from src.db.places import PlaceRepository
                place_repo = PlaceRepository(self.db_path)
                published_place = place_repo.publish_from_contribution(updated, publisher=user)
                if published_place and not updated.target_resource_id:
                    with get_db(self.db_path) as conn:
                        conn.execute(
                            "UPDATE contributions SET target_resource_id = ? WHERE id = ?",
                            (published_place.id, updated.id),
                        )
                    updated.target_resource_id = published_place.id
            except Exception:
                pass

        return updated

    # -------------------------------------------------------------------------
    # High-level state transition helpers
    # -------------------------------------------------------------------------

    def submit(self, contribution_id: str, user: Optional[Any] = None) -> Contribution:
        """Author submits draft for moderation review."""
        if user is None:
            item = self.get(contribution_id)
            if item:
                user = item.owner_id
        return self.transition_state(contribution_id, ContributionState.PENDING_REVIEW, user)

    def queue_for_review(self, contribution_id: str, user: Optional[Any] = None) -> Contribution:
        """Queue submitted contribution for reviewer triage."""
        if user is None:
            item = self.get(contribution_id)
            if item:
                user = item.owner_id
        return self.transition_state(contribution_id, ContributionState.PENDING_REVIEW, user)

    def approve(self, contribution_id: str, reviewer: Any, notes: Optional[str] = None) -> Contribution:
        """Reviewer approves contribution."""
        return self.transition_state(contribution_id, ContributionState.APPROVED, reviewer, notes=notes)

    def reject(self, contribution_id: str, reviewer: Any, notes: Optional[str] = None) -> Contribution:
        """Reviewer rejects contribution with rationale."""
        return self.transition_state(contribution_id, ContributionState.REJECTED, reviewer, notes=notes)

    def request_changes(self, contribution_id: str, reviewer: Any, notes: Optional[str] = None) -> Contribution:
        """Reviewer requests revisions on contribution with feedback notes."""
        return self.transition_state(contribution_id, ContributionState.CHANGES_REQUESTED, reviewer, notes=notes)

    def get_diff_summary(self, contribution_id: str) -> Dict[str, Any]:
        """
        Compute a structured diff comparing the contribution's data payload
        against its target canonical place (or all proposed fields for new additions).
        """
        item = self.get(contribution_id)
        if not item:
            return {}

        proposed_data = item.to_dict().get("data", {})
        diff: Dict[str, Any] = {
            "action": item.action,
            "target_resource_id": item.target_resource_id,
            "changes": {},
        }

        if item.action == "create" or not item.target_resource_id:
            for k, v in proposed_data.items():
                diff["changes"][k] = {"old": None, "new": v}
            return diff

        try:
            from src.db.places import PlaceRepository
            place_repo = PlaceRepository(self.db_path)
            canonical = place_repo.get_place(item.target_resource_id, include_deleted=True)
            if not canonical:
                for k, v in proposed_data.items():
                    diff["changes"][k] = {"old": None, "new": v}
                return diff

            canonical_dict = canonical.to_dict()
            for k, new_v in proposed_data.items():
                old_v = canonical_dict.get(k)
                if old_v != new_v:
                    diff["changes"][k] = {"old": old_v, "new": new_v}
        except Exception:
            for k, v in proposed_data.items():
                diff["changes"][k] = {"old": None, "new": v}

        return diff

    def review(
        self,
        contribution_id: str,
        reviewed_by: Any,
        decision: str,
        review_notes: Optional[str] = None,
    ) -> Contribution:
        """Reviewer approves or rejects a pending submission."""
        decision_clean = str(decision).lower()
        target = ContributionState.APPROVED if decision_clean in ("approved", "approve") else ContributionState.REJECTED
        return self.transition_state(contribution_id, target, user=reviewed_by, notes=review_notes)

    def publish(self, contribution_id: str, staff: Any) -> Contribution:
        """Publish approved contribution to canonical map data."""
        return self.transition_state(contribution_id, ContributionState.PUBLISHED, staff)

    def withdraw(self, contribution_id: str, user: Optional[Any] = None) -> Contribution:
        """Author withdraws contribution."""
        if user is None:
            item = self.get(contribution_id)
            if item:
                user = item.owner_id
        return self.transition_state(contribution_id, ContributionState.WITHDRAWN, user)

    def list(
        self,
        owner_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Contribution]:
        """Query contributions filtered by owner and/or status."""
        clauses = []
        params: List[Any] = []

        if owner_id:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        if status:
            clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM contributions {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with get_db(self.db_path) as conn:
            cur = conn.execute(query, tuple(params))
            return [Contribution(**dict(row)) for row in cur.fetchall()]

    def delete(self, contribution_id: str) -> bool:
        """Permanently delete a contribution record."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("DELETE FROM contributions WHERE id = ?", (contribution_id,))
            return cur.rowcount > 0
