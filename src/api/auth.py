"""
Navigators IDR - RBAC Authentication & Session Management API Router
Exposes authentication (registration, login, guest sessions), token verification,
session management, and dynamic permission checking.
"""

from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService, AuthorizationResult

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Ensure DB is initialized on module load
init_db()
repo = RBACRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


# =============================================================================
# Request & Response Models
# =============================================================================

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    avatar: Optional[str] = None
    role_id: str = "user"


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    id: str
    email: str
    name: str
    avatar: Optional[str] = None
    status: str = "active"
    role_ids: Optional[List[str]] = Field(default_factory=lambda: ["user"])


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None
    status: Optional[str] = None


class AssignRoleRequest(BaseModel):
    role_id: str


class CheckPermissionRequest(BaseModel):
    user_id: str
    permission_id: str


class AuthorizeRequest(BaseModel):
    action: str
    resource: Optional[Dict[str, Any]] = None


# =============================================================================
# Security Dependencies
# Server-side authentication and dynamic authorization gates.
# Never trust client-supplied role or permission headers.
# =============================================================================

def extract_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """Extract raw bearer token string from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication credentials were not provided in Authorization header.",
        )
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
        )
    return parts[1].strip()


def get_current_session(authorization: Optional[str] = Header(None)) -> SessionContext:
    """
    FastAPI dependency: Resolves session token to server-verified SessionContext.
    Enforces flow: Token -> Session -> User Record -> Role Lookup -> Permissions Loaded.
    """
    raw_token = extract_bearer_token(authorization)
    context = auth_service.resolve_session(raw_token)
    if not context:
        raise HTTPException(
            status_code=401,
            detail="Session token is invalid, expired, or has been revoked.",
        )
    return context


def require_permission(permission_id: str):
    """
    FastAPI dependency factory: Enforces that the authenticated session holds
    the requested permission, derived strictly from database roles.
    """
    def _dependency(context: SessionContext = Depends(get_current_session)) -> SessionContext:
        if permission_id not in context.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: operation requires '{permission_id}'.",
            )
        return context
    return _dependency


def require_authz(action: str, resource_loader: Optional[Any] = None):
    """
    FastAPI dependency factory: Central authorization gate.
    Evaluates authentication + role + permission + ownership + resource state.
    """
    def _dependency(context: SessionContext = Depends(get_current_session)) -> SessionContext:
        resource = resource_loader(context) if callable(resource_loader) else None
        result = authz_service.can(user=context, action=action, resource=resource)
        if not result.allowed:
            status_code = 401 if result.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
            raise HTTPException(status_code=status_code, detail=result.reason)
        return context
    return _dependency



# =============================================================================
# Authentication Endpoints
# =============================================================================

@router.post("/register", status_code=201)
def register(
    req: RegisterRequest,
    user_agent: Optional[str] = Header(None),
):
    """
    Register a new user account with local password identity.
    Issues session token and returns resolved user and permissions.
    """
    try:
        user, session_context, raw_token = auth_service.register(
            email=req.email,
            password=req.password,
            name=req.name,
            avatar=req.avatar,
            role_id=req.role_id,
            user_agent=user_agent,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": "Account successfully registered.",
        "token": raw_token,
        "session": session_context.to_dict(),
    }


@router.post("/login")
def login(
    req: LoginRequest,
    user_agent: Optional[str] = Header(None),
):
    """
    Authenticate user credentials, verify identity, create session,
    and dynamically load roles and permissions from database.
    """
    try:
        user, session_context, raw_token = auth_service.login(
            email=req.email,
            password=req.password,
            user_agent=user_agent,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return {
        "message": "Login successful.",
        "token": raw_token,
        "session": session_context.to_dict(),
    }


@router.post("/guest")
def create_guest(user_agent: Optional[str] = Header(None)):
    """
    Issue an anonymous guest session with no account creation.
    Guest sessions receive read-only navigation permissions.
    """
    session_context, raw_token = auth_service.create_guest_session(user_agent=user_agent)
    return {
        "message": "Guest session created.",
        "token": raw_token,
        "session": session_context.to_dict(),
    }


@router.get("/me")
def get_current_user_profile(context: SessionContext = Depends(get_current_session)):
    """
    Return currently authenticated user profile, roles, and effective permissions.
    Resolved exclusively by server database from session token.
    """
    return context.to_dict()


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """
    Revoke the current session token immediately.
    """
    raw_token = extract_bearer_token(authorization)
    revoked = auth_service.revoke_session(raw_token)
    return {"message": "Session revoked successfully.", "revoked": revoked}


@router.post("/authorize")
def authorize_action(
    req: AuthorizeRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Central authorization evaluation endpoint:
    Answers: Can user X perform action Y on resource Z?
    Evaluates: authentication + role + permission + ownership + resource state.
    """
    session_context = None
    if authorization:
        try:
            raw_token = extract_bearer_token(authorization)
            session_context = auth_service.resolve_session(raw_token)
        except HTTPException:
            session_context = None

    decision = authz_service.can(
        user=session_context,
        action=req.action,
        resource=req.resource,
    )
    return decision.to_dict()



# =============================================================================
# Roles, Permissions & RBAC Catalog Endpoints
# =============================================================================

@router.get("/roles")
def get_roles():
    """List all available roles in the system."""
    roles = repo.list_roles()
    return {"roles": [r.to_dict() for r in roles]}


@router.get("/roles/{role_id}/permissions")
def get_role_permissions(role_id: str):
    """List all permissions granted to a specific role."""
    role = repo.get_role(role_id)
    if not role:
        raise HTTPException(status_code=404, detail=f"Role '{role_id}' not found")
    perms = repo.get_role_permissions(role_id)
    return {"role": role.to_dict(), "permissions": [p.to_dict() for p in perms]}


@router.get("/permissions")
def get_permissions(resource: Optional[str] = Query(None, description="Filter by resource")):
    """List all defined permissions in the catalog."""
    perms = repo.list_permissions(resource=resource)
    return {"permissions": [p.to_dict() for p in perms]}


@router.get("/matrix")
def get_rbac_matrix():
    """Export complete dynamic RBAC matrix for offline client synchronization."""
    return repo.export_rbac_matrix()


# =============================================================================
# User Administration Endpoints
# =============================================================================

@router.post("/users", status_code=201)
def create_user(req: CreateUserRequest):
    """Register or create a new user profile with initial roles."""
    existing = repo.get_user(req.id) or repo.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="User with this ID or email already exists")
    user = repo.create_user(
        user_id=req.id,
        email=req.email,
        name=req.name,
        avatar=req.avatar,
        status=req.status,
        initial_role_ids=req.role_ids,
    )
    roles = repo.get_user_roles(user.id)
    perms = repo.get_user_permissions(user.id)
    return {
        "user": user.to_dict(),
        "roles": [r.to_dict() for r in roles],
        "permissions": sorted(list(perms)),
    }


@router.get("/users/{user_id}")
def get_user(user_id: str):
    """Get user profile, assigned roles, and effective permissions."""
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    roles = repo.get_user_roles(user_id)
    perms = repo.get_user_permissions(user_id)
    return {
        "user": user.to_dict(),
        "roles": [r.to_dict() for r in roles],
        "permissions": sorted(list(perms)),
    }


@router.patch("/users/{user_id}")
def update_user(user_id: str, req: UpdateUserRequest):
    """Update user profile attributes or status."""
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    updated = repo.update_user(user_id, name=req.name, avatar=req.avatar, status=req.status)
    return {"user": updated.to_dict() if updated else None}


@router.post("/users/{user_id}/roles")
def assign_user_role(user_id: str, req: AssignRoleRequest):
    """Assign a role to a user."""
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    role = repo.get_role(req.role_id)
    if not role:
        raise HTTPException(status_code=404, detail=f"Role '{req.role_id}' not found")
    repo.assign_role_to_user(user_id, req.role_id)
    roles = repo.get_user_roles(user_id)
    perms = repo.get_user_permissions(user_id)
    return {
        "message": f"Assigned role '{role.name}' to user '{user.name}'",
        "roles": [r.to_dict() for r in roles],
        "permissions": sorted(list(perms)),
    }


@router.delete("/users/{user_id}/roles/{role_id}")
def remove_user_role(user_id: str, role_id: str):
    """Revoke a role from a user."""
    user = repo.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    removed = repo.remove_role_from_user(user_id, role_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Role was not assigned to user")
    roles = repo.get_user_roles(user_id)
    perms = repo.get_user_permissions(user_id)
    return {
        "message": f"Removed role '{role_id}' from user '{user.name}'",
        "roles": [r.to_dict() for r in roles],
        "permissions": sorted(list(perms)),
    }


@router.post("/check-permission")
def check_permission(req: CheckPermissionRequest):
    """
    Security evaluation endpoint:
    Checks if a user is authorized for a specific resource:action permission.
    Evaluates dynamically via database joins without hardcoded role checking.
    """
    user = repo.get_user(req.user_id)
    if not user:
        return {"allowed": False, "reason": "User not found"}
    if user.status != "active":
        return {"allowed": False, "reason": f"User account is {user.status}"}

    allowed = repo.has_permission(req.user_id, req.permission_id)
    return {
        "user_id": req.user_id,
        "permission_id": req.permission_id,
        "allowed": allowed,
    }
