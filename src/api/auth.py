"""
Navigators IDR - RBAC Authentication & Authorization API Router
Exposes dynamic permission checking, role management, and user permissions endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.rbac import RBACRepository

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Ensure DB is initialized on module load
init_db()
repo = RBACRepository()


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
    Core security evaluation endpoint:
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
