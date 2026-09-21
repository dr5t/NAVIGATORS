"""
Navigators IDR - Database Schema & Dynamic RBAC Test Suite
Validates relational integrity, seed matrices, user lifecycle, and dynamic permission evaluation.
"""

import sqlite3
import pytest
from pathlib import Path

from src.db.database import init_db, connect_db
from src.db.rbac import RBACRepository, User, Role, Permission
from src.api.auth import (
    get_roles,
    get_permissions,
    create_user as api_create_user,
    check_permission as api_check_permission,
    assign_user_role as api_assign_user_role,
    remove_user_role as api_remove_user_role,
    CreateUserRequest,
    CheckPermissionRequest,
    AssignRoleRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database for each test."""
    db_file = tmp_path / "navigators_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def repo(temp_db: Path):
    """Provide an RBACRepository bound to the temporary database."""
    return RBACRepository(temp_db)


# =============================================================================
# 1. Schema Initialization & Idempotency
# =============================================================================

def test_schema_tables_exist(temp_db: Path):
    """Verify all 5 core RBAC tables exist with primary keys and foreign keys."""
    conn = connect_db(temp_db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}
    conn.close()

    expected = {"users", "roles", "permissions", "role_permissions", "user_roles"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


def test_schema_initialization_is_idempotent(temp_db: Path):
    """Running init_db multiple times must not crash or duplicate seed records."""
    init_db(temp_db)
    init_db(temp_db)

    conn = connect_db(temp_db)
    role_count = conn.execute("SELECT COUNT(*) as c FROM roles").fetchone()["c"]
    perm_count = conn.execute("SELECT COUNT(*) as c FROM permissions").fetchone()["c"]
    conn.close()

    assert role_count == 7
    assert perm_count >= 21


# =============================================================================
# 2. Initial Roles & Permissions Seeds
# =============================================================================

def test_initial_roles_catalog(repo: RBACRepository):
    """Ensure all 7 required initial roles are seeded correctly."""
    roles = {r.id: r for r in repo.list_roles()}
    required_roles = {
        "guest",
        "user",
        "local_contributor",
        "internal_contributor",
        "moderator",
        "team_admin",
        "super_admin",
    }
    assert required_roles.issubset(roles.keys())


def test_permissions_catalog_and_resources(repo: RBACRepository):
    """Ensure granular resource:action permissions are cataloged."""
    perms = repo.list_permissions()
    perm_ids = {p.id for p in perms}

    required_permissions = {
        "place:create",
        "place:read",
        "place:update",
        "place:delete",
        "contribution:create",
        "contribution:read",
        "contribution:update",
        "contribution:withdraw",
        "contribution:approve",
        "contribution:reject",
        "dataset:create",
        "dataset:read",
        "dataset:update",
        "dataset:delete",
        "training:create",
        "training:read",
        "model:create",
        "model:read",
        "model:update",
        "model:approve",
        "model:deploy",
        "user:read",
        "user:update",
        "role:assign",
    }
    assert required_permissions.issubset(perm_ids)


def test_role_permissions_hierarchy(repo: RBACRepository):
    """Verify standard role capabilities adhere to security hierarchy."""
    guest_perms = {p.id for p in repo.get_role_permissions("guest")}
    assert guest_perms == {"place:read"}

    user_perms = {p.id for p in repo.get_role_permissions("user")}
    assert "place:read" in user_perms
    assert "contribution:create" in user_perms
    assert "contribution:approve" not in user_perms

    local_perms = {p.id for p in repo.get_role_permissions("local_contributor")}
    assert "place:create" in local_perms
    assert "contribution:update" in local_perms
    assert "model:deploy" not in local_perms

    moderator_perms = {p.id for p in repo.get_role_permissions("moderator")}
    assert "contribution:approve" in moderator_perms
    assert "contribution:reject" in moderator_perms
    assert "place:delete" in moderator_perms
    assert "dataset:delete" not in moderator_perms

    admin_perms = {p.id for p in repo.get_role_permissions("team_admin")}
    assert "model:deploy" in admin_perms
    assert "role:assign" in admin_perms
    assert "training:create" in admin_perms

    super_perms = {p.id for p in repo.get_role_permissions("super_admin")}
    all_perms = {p.id for p in repo.list_permissions()}
    assert super_perms == all_perms


# =============================================================================
# 3. User Lifecycle & Constraints
# =============================================================================

def test_user_creation_and_query(repo: RBACRepository):
    """Create user and retrieve by ID and Email."""
    user = repo.create_user(
        user_id="usr_101",
        email="navigator@example.org",
        name="Navigator Field Tester",
        avatar="https://navigators.example/avatar.png",
        status="active",
        initial_role_ids=["user"],
    )
    assert user.id == "usr_101"
    assert user.email == "navigator@example.org"
    assert user.status == "active"

    by_id = repo.get_user("usr_101")
    assert by_id is not None
    assert by_id.name == "Navigator Field Tester"

    by_email = repo.get_user_by_email("navigator@example.org")
    assert by_email is not None
    assert by_email.id == "usr_101"


def test_user_duplicate_email_rejected(repo: RBACRepository):
    """Ensure email uniqueness constraint is enforced."""
    repo.create_user("usr_01", "unique@example.org", "User One")
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_user("usr_02", "unique@example.org", "User Two")


def test_user_update_and_login_timestamp(repo: RBACRepository):
    """Updating user updates status and record_login updates last_login_at."""
    repo.create_user("usr_update", "update@example.org", "Original Name")

    updated = repo.update_user("usr_update", name="Modified Name", status="suspended")
    assert updated.name == "Modified Name"
    assert updated.status == "suspended"

    user_before = repo.get_user("usr_update")
    assert user_before.last_login_at is None

    repo.record_login("usr_update")
    user_after = repo.get_user("usr_update")
    assert user_after.last_login_at is not None


# =============================================================================
# 4. Dynamic Permission Evaluation (No Hardcoded Roles)
# =============================================================================

def test_dynamic_permission_evaluation(repo: RBACRepository):
    """
    Validate that has_permission evaluates directly against assigned role permissions.
    No hardcoded role names are used in the check.
    """
    user = repo.create_user("usr_eval", "eval@example.org", "Evaluator", initial_role_ids=["user"])

    # Base user permissions
    assert repo.has_permission(user.id, "place:read") is True
    assert repo.has_permission(user.id, "contribution:create") is True
    assert repo.has_permission(user.id, "place:create") is False
    assert repo.has_permission(user.id, "model:deploy") is False

    # Promote to local_contributor dynamically
    repo.assign_role_to_user(user.id, "local_contributor")
    assert repo.has_permission(user.id, "place:create") is True
    assert repo.has_permission(user.id, "model:deploy") is False

    # Promote to team_admin dynamically
    repo.assign_role_to_user(user.id, "team_admin")
    assert repo.has_permission(user.id, "model:deploy") is True
    assert repo.has_permission(user.id, "role:assign") is True

    # Revoke team_admin dynamically
    repo.remove_role_from_user(user.id, "team_admin")
    assert repo.has_permission(user.id, "model:deploy") is False
    assert repo.has_permission(user.id, "place:create") is True


def test_inactive_or_suspended_users_have_zero_permissions(repo: RBACRepository):
    """Suspended or deleted users must fail all permission checks."""
    user = repo.create_user(
        "usr_suspended", "suspend@example.org", "Suspended Admin", initial_role_ids=["super_admin"]
    )
    # Active super admin has all permissions
    assert repo.has_permission(user.id, "model:deploy") is True

    # Suspend user
    repo.update_user(user.id, status="suspended")
    assert repo.has_permission(user.id, "model:deploy") is False
    assert repo.has_permission(user.id, "place:read") is False
    assert repo.get_user_permissions(user.id) == set()


def test_foreign_key_cascades(repo: RBACRepository, temp_db: Path):
    """Deleting a user removes user_roles entries; deleting role cascades role_permissions."""
    user = repo.create_user("usr_cascade", "cascade@example.org", "Cascade", initial_role_ids=["user", "local_contributor"])
    assert len(repo.get_user_roles(user.id)) == 2

    # Delete user directly
    conn = connect_db(temp_db)
    conn.execute("DELETE FROM users WHERE id = ?", (user.id,))
    conn.commit()

    roles_after = conn.execute("SELECT * FROM user_roles WHERE user_id = ?", (user.id,)).fetchall()
    conn.close()
    assert len(roles_after) == 0


def test_export_rbac_matrix(repo: RBACRepository):
    """Verify export_rbac_matrix formats the full RBAC state for client sync."""
    matrix = repo.export_rbac_matrix()
    assert matrix["version"] == "1.0"
    assert len(matrix["roles"]) == 7
    assert len(matrix["permissions"]) >= 21
    assert "super_admin" in matrix["role_permissions"]
    assert "place:read" in matrix["role_permissions"]["guest"]


# =============================================================================
# 5. FastAPI RBAC Router Endpoints Direct Testing
# =============================================================================

def test_api_roles_and_permissions_endpoints():
    """Verify router functions get_roles and get_permissions."""
    roles_res = get_roles()
    assert "roles" in roles_res
    role_ids = {r["id"] for r in roles_res["roles"]}
    assert "team_admin" in role_ids
    assert "local_contributor" in role_ids

    perms_res = get_permissions(resource="model")
    assert "permissions" in perms_res
    model_perms = {p["id"] for p in perms_res["permissions"]}
    assert "model:deploy" in model_perms


def test_api_user_creation_and_permission_check_endpoints():
    """Verify router user creation, role assignment, and dynamic permission evaluation."""
    import time
    uid = f"usr_api_{int(time.time() * 1000)}"
    req = CreateUserRequest(
        id=uid,
        email=f"{uid}@navigators.example",
        name="API Tester",
        role_ids=["user"],
    )
    user_res = api_create_user(req)
    assert user_res["user"]["id"] == uid

    # Check permission place:read -> True
    chk1 = api_check_permission(CheckPermissionRequest(user_id=uid, permission_id="place:read"))
    assert chk1["allowed"] is True

    # Check permission model:deploy -> False
    chk2 = api_check_permission(CheckPermissionRequest(user_id=uid, permission_id="model:deploy"))
    assert chk2["allowed"] is False

    # Promote user to team_admin
    promote_res = api_assign_user_role(uid, AssignRoleRequest(role_id="team_admin"))
    assert any(r["id"] == "team_admin" for r in promote_res["roles"])

    # Check permission model:deploy -> Now True!
    chk3 = api_check_permission(CheckPermissionRequest(user_id=uid, permission_id="model:deploy"))
    assert chk3["allowed"] is True

    # Revoke team_admin
    revoke_res = api_remove_user_role(uid, "team_admin")
    assert not any(r["id"] == "team_admin" for r in revoke_res["roles"])

    # Check permission model:deploy -> False again!
    chk4 = api_check_permission(CheckPermissionRequest(user_id=uid, permission_id="model:deploy"))
    assert chk4["allowed"] is False
