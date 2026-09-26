from __future__ import annotations
"""
Navigators IDR - Dynamic Role-Based Access Control (RBAC) Repository
Enforces permissions through relational mapping without hardcoded role names.
"""

import sqlite3
from typing import Optional, List, Set, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from src.db.database import get_db, connect_db


@dataclass
class User:
    id: str
    email: str
    name: str
    avatar: Optional[str]
    status: str
    created_at: str
    updated_at: str
    last_login_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Role:
    id: str
    name: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Permission:
    id: str
    resource: str
    action: str
    description: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RBACRepository:
    """Data access layer for Users, Roles, Permissions, and dynamic evaluation."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = db_path

    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        with get_db(self.db_path) as conn:
            return conn.execute(query, params)





    def create_user(
        self,
        user_id: str,
        email: str,
        name: str,
        avatar: Optional[str] = None,
        status: str = "active",
        initial_role_ids: Optional[List[str]] = None,
    ) -> User:
        """Create a new user and optionally assign initial roles."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, name, avatar, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, email.lower().strip(), name.strip(), avatar, status, now, now),
            )
            if initial_role_ids:
                for role_id in initial_role_ids:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO user_roles (user_id, role_id, assigned_at)
                        VALUES (?, ?, ?)
                        """,
                        (user_id, role_id, now),
                    )

        user = self.get_user(user_id)
        if not user:
            raise RuntimeError(f"User {user_id} creation failed.")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Fetch user by unique identifier."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return User(**dict(row))

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Fetch user by unique email address."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))
            row = cur.fetchone()
            if not row:
                return None
            return User(**dict(row))

    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        avatar: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[User]:
        """Update mutable fields of a user record."""
        fields = []
        values = []
        if name is not None:
            fields.append("name = ?")
            values.append(name.strip())
        if avatar is not None:
            fields.append("avatar = ?")
            values.append(avatar)
        if status is not None:
            fields.append("status = ?")
            values.append(status)

        if not fields:
            return self.get_user(user_id)

        values.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
        with get_db(self.db_path) as conn:
            conn.execute(query, tuple(values))

        return self.get_user(user_id)

    def record_login(self, user_id: str) -> None:
        """Update last_login_at timestamp for user."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))

    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Paginated user listing."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [User(**dict(row)) for row in cur.fetchall()]





    def create_role(self, role_id: str, name: str, description: str) -> Role:
        """Create a new role definition."""
        with get_db(self.db_path) as conn:
            conn.execute(
                "INSERT INTO roles (id, name, description) VALUES (?, ?, ?)",
                (role_id, name, description),
            )
        return Role(id=role_id, name=name, description=description)

    def get_role(self, role_id: str) -> Optional[Role]:
        """Retrieve role by identifier."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM roles WHERE id = ?", (role_id,))
            row = cur.fetchone()
            return Role(**dict(row)) if row else None

    def list_roles(self) -> List[Role]:
        """Return all defined roles."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM roles ORDER BY id ASC")
            return [Role(**dict(row)) for row in cur.fetchall()]

    def delete_role(self, role_id: str) -> bool:
        """Delete role and cascade remove role_permissions and user_roles."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("DELETE FROM roles WHERE id = ?", (role_id,))
            return cur.rowcount > 0





    def create_permission(
        self, permission_id: str, resource: str, action: str, description: Optional[str] = None
    ) -> Permission:
        """Create a new permission entry."""
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO permissions (id, resource, action, description)
                VALUES (?, ?, ?, ?)
                """,
                (permission_id, resource, action, description),
            )
        return Permission(
            id=permission_id, resource=resource, action=action, description=description
        )

    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Fetch permission by identifier."""
        with get_db(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM permissions WHERE id = ?", (permission_id,))
            row = cur.fetchone()
            return Permission(**dict(row)) if row else None

    def list_permissions(self, resource: Optional[str] = None) -> List[Permission]:
        """List permissions, optionally filtered by resource."""
        with get_db(self.db_path) as conn:
            if resource:
                cur = conn.execute(
                    "SELECT * FROM permissions WHERE resource = ? ORDER BY id ASC", (resource,)
                )
            else:
                cur = conn.execute("SELECT * FROM permissions ORDER BY resource, action ASC")
            return [Permission(**dict(row)) for row in cur.fetchall()]





    def assign_permission_to_role(self, role_id: str, permission_id: str) -> None:
        """Grant a permission to a role."""
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
                VALUES (?, ?)
                """,
                (role_id, permission_id),
            )

    def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """Revoke a permission from a role."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?",
                (role_id, permission_id),
            )
            return cur.rowcount > 0

    def get_role_permissions(self, role_id: str) -> List[Permission]:
        """Fetch all permissions associated with a given role."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT p.* FROM permissions p
                JOIN role_permissions rp ON rp.permission_id = p.id
                WHERE rp.role_id = ?
                ORDER BY p.id ASC
                """,
                (role_id,),
            )
            return [Permission(**dict(row)) for row in cur.fetchall()]





    def assign_role_to_user(self, user_id: str, role_id: str) -> None:
        """Assign a role to a user."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO user_roles (user_id, role_id, assigned_at)
                VALUES (?, ?, ?)
                """,
                (user_id, role_id, now),
            )

    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """Remove a role assignment from a user."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?", (user_id, role_id)
            )
            return cur.rowcount > 0

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Fetch all roles currently assigned to a user."""
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT r.* FROM roles r
                JOIN user_roles ur ON ur.role_id = r.id
                WHERE ur.user_id = ?
                ORDER BY r.id ASC
                """,
                (user_id,),
            )
            return [Role(**dict(row)) for row in cur.fetchall()]





    def get_user_permissions(self, user_id: str) -> Set[str]:
        """
        Dynamically calculate the distinct set of permission IDs granted to a user
        across all of their assigned roles. Suspended/deleted users have 0 permissions.
        """
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT DISTINCT rp.permission_id
                FROM role_permissions rp
                JOIN user_roles ur ON ur.role_id = rp.role_id
                JOIN users u ON u.id = ur.user_id
                WHERE ur.user_id = ? AND u.status = 'active'
                """,
                (user_id,),
            )
            return {row["permission_id"] for row in cur.fetchall()}

    def has_permission(self, user_id: str, permission_id: str) -> bool:
        """
        Core security check: Verify if an active user has the requested permission
        via ANY of their assigned roles.

        This method completely avoids hardcoding role names in application logic.
        """
        with get_db(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT 1
                FROM role_permissions rp
                JOIN user_roles ur ON ur.role_id = rp.role_id
                JOIN users u ON u.id = ur.user_id
                WHERE ur.user_id = ? AND rp.permission_id = ? AND u.status = 'active'
                LIMIT 1
                """,
                (user_id, permission_id),
            )
            return cur.fetchone() is not None

    def export_rbac_matrix(self) -> Dict[str, Any]:
        """
        Export complete RBAC matrix definition for offline client sync.
        """
        roles = self.list_roles()
        permissions = self.list_permissions()
        matrix = {
            "version": "1.0",
            "roles": [r.to_dict() for r in roles],
            "permissions": [p.to_dict() for p in permissions],
            "role_permissions": {},
        }
        for role in roles:
            perms = self.get_role_permissions(role.id)
            matrix["role_permissions"][role.id] = [p.id for p in perms]
        return matrix
