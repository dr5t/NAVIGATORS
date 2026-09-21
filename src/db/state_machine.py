"""
Navigators IDR - Contribution Resource State Machine
Enforces deterministic lifecycle transitions and role-gated state changes:
  DRAFT -> SUBMITTED -> PENDING_REVIEW -> APPROVED -> PUBLISHED
  Alternative: PENDING_REVIEW -> REJECTED
  Cancellation: DRAFT / SUBMITTED / PENDING_REVIEW -> WITHDRAWN

Arbitrary state jumps (e.g. User -> APPROVED or DRAFT -> APPROVED) are strictly prevented.
"""

from typing import Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass


class StateTransitionError(ValueError):
    """Raised when an illegal or unauthorized state transition is attempted."""
    def __init__(self, message: str, code: str = "INVALID_STATE_TRANSITION"):
        super().__init__(message)
        self.message = message
        self.code = code


class ContributionState:
    """Canonical states for community contributions."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PENDING_REVIEW = "pending_review"
    PENDING = "pending"  # Recognized alias for pending_review
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    WITHDRAWN = "withdrawn"

    ALL = {
        DRAFT,
        SUBMITTED,
        PENDING_REVIEW,
        PENDING,
        APPROVED,
        PUBLISHED,
        REJECTED,
        CHANGES_REQUESTED,
        WITHDRAWN,
    }


@dataclass(frozen=True)
class TransitionRule:
    source: str
    target: str
    required_permission: str
    owner_only: bool = False
    reviewer_only: bool = False
    staff_only: bool = False
    description: str = ""


class ContributionStateMachine:
    """
    Deterministic state machine engine.
    Ensures that only authorized actors can trigger allowed state transitions.
    """

    # Legal state transition rules table
    RULES: Dict[Tuple[str, str], TransitionRule] = {
        # 1. Author submits draft for ingestion
        (ContributionState.DRAFT, ContributionState.SUBMITTED): TransitionRule(
            source=ContributionState.DRAFT,
            target=ContributionState.SUBMITTED,
            required_permission="contribution:update",
            owner_only=True,
            description="Author submits drafted contribution for triage",
        ),
        # Direct submission to review queue
        (ContributionState.DRAFT, ContributionState.PENDING_REVIEW): TransitionRule(
            source=ContributionState.DRAFT,
            target=ContributionState.PENDING_REVIEW,
            required_permission="contribution:update",
            owner_only=True,
            description="Author submits drafted contribution directly to review queue",
        ),
        # 2. Ingestion pipeline / triage queues submitted contribution for moderation
        (ContributionState.SUBMITTED, ContributionState.PENDING_REVIEW): TransitionRule(
            source=ContributionState.SUBMITTED,
            target=ContributionState.PENDING_REVIEW,
            required_permission="contribution:read",
            description="Queues submitted contribution for moderation review",
        ),
        # 3. Reviewer approves submission (Moderator only)
        (ContributionState.PENDING_REVIEW, ContributionState.APPROVED): TransitionRule(
            source=ContributionState.PENDING_REVIEW,
            target=ContributionState.APPROVED,
            required_permission="contribution:approve",
            reviewer_only=True,
            description="Reviewer approves contribution for map inclusion",
        ),
        # 4. Reviewer rejects submission (Moderator only)
        (ContributionState.PENDING_REVIEW, ContributionState.REJECTED): TransitionRule(
            source=ContributionState.PENDING_REVIEW,
            target=ContributionState.REJECTED,
            required_permission="contribution:reject",
            reviewer_only=True,
            description="Reviewer rejects contribution with rationale",
        ),
        # 4b. Reviewer requests changes from author
        (ContributionState.PENDING_REVIEW, ContributionState.CHANGES_REQUESTED): TransitionRule(
            source=ContributionState.PENDING_REVIEW,
            target=ContributionState.CHANGES_REQUESTED,
            required_permission="contribution:request_changes",
            reviewer_only=True,
            description="Reviewer requests revisions on contribution from author",
        ),
        # 4c. Author resubmits after making requested changes
        (ContributionState.CHANGES_REQUESTED, ContributionState.PENDING_REVIEW): TransitionRule(
            source=ContributionState.CHANGES_REQUESTED,
            target=ContributionState.PENDING_REVIEW,
            required_permission="contribution:update",
            owner_only=True,
            description="Author resubmits revised contribution for moderation review",
        ),
        # 4d. Author withdraws contribution in changes_requested status
        (ContributionState.CHANGES_REQUESTED, ContributionState.WITHDRAWN): TransitionRule(
            source=ContributionState.CHANGES_REQUESTED,
            target=ContributionState.WITHDRAWN,
            required_permission="contribution:withdraw",
            owner_only=True,
            description="Author withdraws contribution after revision request",
        ),
        # 5. Canonical map publisher synchronizes approved data to live map
        (ContributionState.APPROVED, ContributionState.PUBLISHED): TransitionRule(
            source=ContributionState.APPROVED,
            target=ContributionState.PUBLISHED,
            required_permission="place:create",
            staff_only=True,
            description="Synchronizes approved contribution into canonical map dataset",
        ),
        # 6. Cancellation / withdrawal by author
        (ContributionState.DRAFT, ContributionState.WITHDRAWN): TransitionRule(
            source=ContributionState.DRAFT,
            target=ContributionState.WITHDRAWN,
            required_permission="contribution:withdraw",
            owner_only=True,
            description="Author withdraws draft contribution",
        ),
        (ContributionState.SUBMITTED, ContributionState.WITHDRAWN): TransitionRule(
            source=ContributionState.SUBMITTED,
            target=ContributionState.WITHDRAWN,
            required_permission="contribution:withdraw",
            owner_only=True,
            description="Author withdraws submitted contribution",
        ),
        (ContributionState.PENDING_REVIEW, ContributionState.WITHDRAWN): TransitionRule(
            source=ContributionState.PENDING_REVIEW,
            target=ContributionState.WITHDRAWN,
            required_permission="contribution:withdraw",
            owner_only=True,
            description="Author withdraws pending contribution from review",
        ),
    }

    @classmethod
    def can_transition(
        cls,
        current_state: str,
        target_state: str,
        user: Any,
        resource: Any,
        db_path: Optional[Any] = None,
    ) -> Tuple[bool, str, str]:
        """
        Validate whether the given user is authorized to perform the transition
        from current_state to target_state on the given resource.

        Returns:
            (allowed: bool, reason: str, code: str)
        """
        current_norm = current_state.lower()
        target_norm = target_state.lower()

        # Normalize 'pending' alias to canonical 'pending_review'
        if current_norm == "pending":
            current_norm = ContributionState.PENDING_REVIEW
        if target_norm == "pending":
            target_norm = ContributionState.PENDING_REVIEW

        # 1. State validity check
        if current_norm not in ContributionState.ALL:
            return False, f"Unknown current state '{current_state}'.", "INVALID_STATE"
        if target_norm not in ContributionState.ALL:
            return False, f"Unknown target state '{target_state}'.", "INVALID_STATE"

        # 2. No-op transition
        if current_norm == target_norm:
            return True, f"Resource is already in state '{target_norm}'.", "NO_CHANGE"

        # 3. Check legal transition rule in state machine
        transition_key = (current_norm, target_norm)
        rule = cls.RULES.get(transition_key)

        if not rule:
            return (
                False,
                f"Illegal state transition: Cannot transition from '{current_norm}' to '{target_norm}'. "
                f"Follow the defined lifecycle: DRAFT -> SUBMITTED -> PENDING_REVIEW -> APPROVED -> PUBLISHED.",
                "INVALID_STATE_TRANSITION",
            )

        # 4. Extract actor context
        user_id = None
        user_perms: Set[str] = set()
        user_roles: Set[str] = set()
        is_guest = False

        if hasattr(user, "user") and user.user:
            user_id = str(user.user.id)
            user_perms = {str(p) for p in (user.permissions or [])}
            user_roles = {str(r.id if hasattr(r, "id") else r) for r in (user.roles or [])}
            is_guest = getattr(user, "is_guest", False)
        elif isinstance(user, str):
            user_id = user
            try:
                from src.db.rbac import RBACRepository
                rbac = RBACRepository(db_path)
                roles = rbac.get_user_roles(user_id)
                user_roles = {r.id for r in roles}
                user_perms = rbac.get_user_permissions(user_id)
            except Exception:
                pass
            is_guest = (user_id.startswith("guest") or "guest" in user_roles)
        elif hasattr(user, "id"):
            user_id = str(user.id)
            try:
                from src.db.rbac import RBACRepository
                rbac = RBACRepository(db_path)
                roles = rbac.get_user_roles(user_id)
                user_roles = {r.id for r in roles}
                user_perms = rbac.get_user_permissions(user_id)
            except Exception:
                pass
            is_guest = False
        elif isinstance(user, dict):
            user_id = str(user.get("id") or user.get("user_id") or "")
            user_perms = {str(p) for p in (user.get("permissions") or [])}
            user_roles = {
                str(r.get("id") or "") if isinstance(r, dict) else str(r)
                for r in user.get("roles", [])
            }
            user_roles.discard("")
            is_guest = bool(user.get("is_guest", False))

        is_super_admin = "super_admin" in user_roles
        is_moderator = "moderator" in user_roles or "team_admin" in user_roles or is_super_admin

        # Guests cannot trigger state transitions
        if is_guest or not user_id:
            return False, "Authentication is required to trigger state transitions.", "UNAUTHENTICATED"

        # 5. Extract resource owner
        resource_owner_id = None
        if hasattr(resource, "owner_id"):
            resource_owner_id = str(resource.owner_id)
        elif hasattr(resource, "user_id"):
            resource_owner_id = str(resource.user_id)
        elif isinstance(resource, dict):
            resource_owner_id = str(resource.get("owner_id") or resource.get("user_id") or "")

        is_owner = bool(resource_owner_id and resource_owner_id == user_id)

        # 6. Enforce Owner-Only constraint
        if rule.owner_only and not is_owner and not is_super_admin:
            return (
                False,
                f"Transition to '{target_norm}' is restricted exclusively to the resource owner.",
                "NOT_OWNER",
            )

        # 7. Enforce Reviewer-Only constraint (Approve / Reject)
        # Standard user CANNOT approve (User -> APPROVED is strictly impossible)
        if rule.reviewer_only:
            if rule.required_permission not in user_perms and not is_super_admin:
                return (
                    False,
                    f"Only authorized reviewer roles can perform '{rule.required_permission}'. "
                    f"Standard contributors cannot transition submissions to '{target_norm}'.",
                    "PERMISSION_DENIED",
                )
            if not is_moderator:
                return (
                    False,
                    f"Reviewer role required for transition to '{target_norm}'.",
                    "PERMISSION_DENIED",
                )

        # 8. Enforce Staff-Only constraint (Publishing to canonical map data)
        if rule.staff_only:
            if not is_moderator and not is_super_admin:
                return (
                    False,
                    f"Staff role required to publish contributions to canonical map data.",
                    "PERMISSION_DENIED",
                )

        # 9. Required permission check
        if rule.required_permission not in user_perms and not is_super_admin:
            return (
                False,
                f"Missing required permission '{rule.required_permission}'.",
                "PERMISSION_DENIED",
            )

        return True, f"Transition from '{current_norm}' to '{target_norm}' authorized.", "AUTHORIZED"
