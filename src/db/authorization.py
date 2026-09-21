"""
Navigators IDR - Central Authorization Service & Middleware
Single point of truth for evaluating authorization decisions across the application:
  Can user X perform action Y on resource Z?
  can(user, "action", resource)

Evaluates in sequence:
  1. Authentication: Active account vs guest vs suspended
  2. Role & Permissions: Granted resource:action tokens via dynamic RBAC
  3. Ownership: Author/owner verification for personal resource actions
  4. Resource State: Lifecycle constraints (pending, approved, rejected, archived)
"""

from typing import Optional, List, Set, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from src.db.rbac import RBACRepository, User, Role, Permission
from src.db.auth_service import SessionContext


@dataclass
class AuthorizationResult:
    """Standardized decision returned by the AuthorizationService."""
    allowed: bool
    reason: str
    code: str  # AUTHORIZED, UNAUTHENTICATED, ACCOUNT_INACTIVE, PERMISSION_DENIED, NOT_OWNER, INVALID_RESOURCE_STATE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuthorizationService:
    """
    Centralized authority answering:
    Can user X perform action Y on resource Z?
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        rbac_repo: Optional[RBACRepository] = None,
    ):
        self.db_path = db_path
        self.rbac = rbac_repo or RBACRepository(db_path)

    def can(
        self,
        user: Optional[Any],
        action: str,
        resource: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationResult:
        """
        Evaluate if a user is authorized to perform action on resource.
        
        Parameters:
            user: SessionContext, User, dict, or None (anonymous visitor)
            action: Permission action string (e.g. 'place:update', 'contribution:withdraw')
            resource: Target entity (dict, dataclass, or object)
            context: Additional environmental context (optional)
        """
        # =====================================================================
        # 1. Authentication Check
        # =====================================================================
        user_id = None
        user_status = "active"
        is_guest = False
        user_perms: Set[str] = set()
        user_roles: List[str] = []

        if user is None:
            is_guest = True
            guest_perms = self.rbac.get_role_permissions("guest")
            user_perms = {p.id for p in guest_perms} if guest_perms else {"place:read"}
            user_roles = ["guest"]
        elif isinstance(user, SessionContext):
            is_guest = user.is_guest
            user_perms = set(user.permissions)
            user_roles = [r.id for r in user.roles]
            if user.user:
                user_id = user.user.id
                user_status = user.user.status
            else:
                user_id = f"guest_{user.session_id}"
        elif isinstance(user, User):
            user_id = user.id
            user_status = user.status
            user_perms = self.rbac.get_user_permissions(user.id)
            user_roles = [r.id for r in self.rbac.get_user_roles(user.id)]
        elif isinstance(user, dict):
            user_id = user.get("id") or user.get("user_id")
            user_status = user.get("status", "active")
            is_guest = bool(user.get("is_guest", False))
            if "permissions" in user and isinstance(user["permissions"], (list, set)):
                user_perms = set(user["permissions"])
            elif user_id and not is_guest:
                user_perms = self.rbac.get_user_permissions(user_id)
            else:
                user_perms = {"place:read"}
            if "roles" in user and isinstance(user["roles"], list):
                user_roles = [r.get("id") if isinstance(r, dict) else str(r) for r in user["roles"]]
        else:
            return AuthorizationResult(
                allowed=False,
                reason="Invalid user object provided to authorization service.",
                code="UNAUTHENTICATED",
            )

        # Check account lifecycle
        if user_status != "active":
            return AuthorizationResult(
                allowed=False,
                reason=f"User account status is '{user_status}'. Access denied.",
                code="ACCOUNT_INACTIVE",
            )

        # Unauthenticated guest check
        if is_guest and action != "place:read":
            return AuthorizationResult(
                allowed=False,
                reason=f"Action '{action}' requires an authenticated account.",
                code="UNAUTHENTICATED",
            )

        # =====================================================================
        # 2. Role & Permission Check
        # =====================================================================
        # Check if user holds super_admin role for universal access
        is_super_admin = "super_admin" in user_roles

        if not is_super_admin and action not in user_perms:
            return AuthorizationResult(
                allowed=False,
                reason=f"User lacks required permission '{action}'.",
                code="PERMISSION_DENIED",
            )

        # If no resource is specified, permission grant is sufficient
        if resource is None:
            return AuthorizationResult(
                allowed=True,
                reason=f"User has permission '{action}'.",
                code="AUTHORIZED",
            )

        # =====================================================================
        # 3. Ownership Check
        # =====================================================================
        resource_owner_id = self._extract_owner_id(resource)
        is_owner = (resource_owner_id is not None) and (str(resource_owner_id) == str(user_id))

        # Draft contribution privacy: Unsubmitted drafts are strictly private to author
        res_state_initial = self._extract_state(resource)
        if action == "contribution:read" and res_state_initial:
            if str(res_state_initial).lower() == "draft":
                has_staff_view = ("moderator" in user_roles) or ("team_admin" in user_roles) or is_super_admin
                if not is_owner and not has_staff_view:
                    return AuthorizationResult(
                        allowed=False,
                        reason="Unsubmitted draft contributions are private to the author.",
                        code="NOT_OWNER",
                    )

        # Actions strictly constrained to resource author/owner
        author_only_actions = {
            "contribution:withdraw",
            "contribution:update",
        }

        if action in author_only_actions:
            if not is_owner:
                return AuthorizationResult(
                    allowed=False,
                    reason=f"Only the author/owner can perform '{action}' on this resource.",
                    code="NOT_OWNER",
                )

        # Mutating existing user records (unless administrator)
        if action == "user:update" and resource_owner_id:
            has_admin_override = ("team_admin" in user_roles) or is_super_admin
            if not is_owner and not has_admin_override:
                return AuthorizationResult(
                    allowed=False,
                    reason="Cannot update profile of another user without administrative rights.",
                    code="NOT_OWNER",
                )

        # Deleting or modifying datasets (owner or admin override)
        if action in ("dataset:update", "dataset:delete") and resource_owner_id:
            has_admin_override = ("team_admin" in user_roles) or is_super_admin
            if not is_owner and not has_admin_override:
                return AuthorizationResult(
                    allowed=False,
                    reason="User is not the owner of this dataset.",
                    code="NOT_OWNER",
                )

        # =====================================================================
        # 4. Resource State Validation
        # =====================================================================
        res_state = self._extract_state(resource)

        if res_state:
            res_state_norm = str(res_state).lower()

            # Contribution state transitions
            if action in ("contribution:update", "contribution:withdraw"):
                if res_state_norm in ("approved", "rejected", "withdrawn"):
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Cannot perform '{action}' on contribution with status '{res_state}'.",
                        code="INVALID_RESOURCE_STATE",
                    )

            if action in ("contribution:approve", "contribution:reject"):
                if res_state_norm not in ("pending", "pending_review"):
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Cannot review contribution with status '{res_state}'. Must be pending.",
                        code="INVALID_RESOURCE_STATE",
                    )

            # Model deployment state constraints
            if action == "model:deploy":
                if res_state_norm not in ("approved", "qualified", "ready"):
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Cannot deploy candidate model with status '{res_state}'. Must be approved or qualified.",
                        code="INVALID_RESOURCE_STATE",
                    )

            # Dataset mutation state constraints
            if action in ("dataset:update", "dataset:delete"):
                if res_state_norm in ("archived", "locked", "read_only"):
                    return AuthorizationResult(
                        allowed=False,
                        reason=f"Cannot mutate dataset in '{res_state}' state.",
                        code="INVALID_RESOURCE_STATE",
                    )

        # =====================================================================
        # 5. Final Decision
        # =====================================================================
        return AuthorizationResult(
            allowed=True,
            reason=f"Action '{action}' is authorized.",
            code="AUTHORIZED",
        )

    # -------------------------------------------------------------------------
    # Helper extractors
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_owner_id(resource: Any) -> Optional[str]:
        """Extract resource owner/author identifier."""
        if isinstance(resource, dict):
            for key in ("user_id", "owner_id", "created_by", "author_id"):
                if resource.get(key) is not None:
                    return str(resource[key])
            if resource.get("type") == "user" and resource.get("id"):
                return str(resource["id"])
            return None
        for attr in ("user_id", "owner_id", "created_by", "author_id"):
            if hasattr(resource, attr):
                val = getattr(resource, attr)
                if val is not None:
                    return str(val)
        return None

    @staticmethod
    def _extract_state(resource: Any) -> Optional[str]:
        """Extract resource lifecycle state or status."""
        if isinstance(resource, dict):
            return resource.get("status") or resource.get("state")
        for attr in ("status", "state"):
            if hasattr(resource, attr):
                val = getattr(resource, attr)
                if val is not None:
                    return str(val)
        return None
