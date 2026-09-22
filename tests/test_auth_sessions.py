"""
Navigators IDR - Authentication & Session Management Test Suite
Validates PBKDF2 password hashing, multi-provider identity mapping,
session token hashing, expiration, revocation, guest sessions, and dynamic RBAC resolution.
"""

import sqlite3
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.db.database import init_db, connect_db
from src.db.rbac import RBACRepository
from src.db.auth_service import (
    AuthService,
    hash_password,
    verify_password,
    generate_session_token,
    compute_token_hash,
)
from src.api.auth import (
    register as api_register,
    login as api_login,
    create_guest as api_create_guest,
    get_current_user_profile as api_get_current_user_profile,
    logout as api_logout,
    RegisterRequest,
    LoginRequest,
    get_current_session,
    require_permission,
)
from fastapi import HTTPException


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_auth_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def rbac_repo(temp_db: Path):
    """Provide an RBACRepository instance."""
    return RBACRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path, rbac_repo: RBACRepository):
    """Provide an AuthService instance."""
    return AuthService(temp_db, rbac_repo)






def test_password_hashing_and_verification():
    """Verify PBKDF2-HMAC-SHA256 hashing format and constant-time match."""
    password = "NavigatorsSecurePass2026!"
    hashed = hash_password(password)


    parts = hashed.split("$")
    assert len(parts) == 4
    assert parts[0] == "pbkdf2_sha256"
    assert int(parts[1]) == 600_000
    assert len(parts[2]) == 32


    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123!", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "") is False
    assert verify_password(password, "malformed$hash") is False


def test_password_salt_uniqueness():
    """Identical passwords must produce distinct salted hashes."""
    pwd = "IdenticalPassword123!"
    h1 = hash_password(pwd)
    h2 = hash_password(pwd)
    assert h1 != h2
    assert verify_password(pwd, h1) is True
    assert verify_password(pwd, h2) is True


def test_password_length_constraint():
    """Passwords shorter than 8 characters must be rejected."""
    with pytest.raises(ValueError, match="at least 8 characters"):
        hash_password("short")






def test_user_registration_flow(auth_service: AuthService, temp_db: Path):
    """
    Registering a user must:
    1. Create record in users table with active status.
    2. Create record in auth_identities with provider 'local_password'.
    3. Assign default 'user' role in user_roles.
    4. Issue session token stored as SHA-256 in sessions table.
    5. Return loaded permissions.
    """
    user, session, token = auth_service.register(
        email="explorer@navigators.dev",
        password="SecureNavPassword123!",
        name="Alex River",
    )

    assert user.email == "explorer@navigators.dev"
    assert user.name == "Alex River"
    assert user.status == "active"
    assert session.is_guest is False


    conn = connect_db(temp_db)

    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user.id,)).fetchone()
    assert user_row is not None
    assert user_row["email"] == "explorer@navigators.dev"


    identity = conn.execute("SELECT * FROM auth_identities WHERE user_id = ?", (user.id,)).fetchone()
    assert identity is not None
    assert identity["provider"] == "local_password"
    assert identity["identifier"] == "explorer@navigators.dev"
    assert verify_password("SecureNavPassword123!", identity["credential_hash"]) is True


    token_hash = compute_token_hash(token)
    session_row = conn.execute("SELECT * FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
    assert session_row is not None
    assert session_row["user_id"] == user.id
    assert session_row["is_guest"] == 0
    conn.close()


    assert "place:read" in session.permissions
    assert "contribution:create" in session.permissions
    assert "training:create" not in session.permissions


def test_registration_duplicate_rejected(auth_service: AuthService):
    """Cannot register multiple accounts with the same email."""
    auth_service.register(
        email="duplicate@navigators.dev",
        password="ValidPassword123!",
        name="User One",
    )
    with pytest.raises(ValueError, match="already exists"):
        auth_service.register(
            email="duplicate@navigators.dev",
            password="ValidPassword123!",
            name="User Two",
        )







def test_login_success(auth_service: AuthService):
    """Successful login verifies identity, updates last_login_at, and issues session."""
    auth_service.register(
        email="member@navigators.dev",
        password="MySecretPassword99!",
        name="Team Member",
    )

    user, session, token = auth_service.login(
        email="member@navigators.dev",
        password="MySecretPassword99!",
        user_agent="NavigatorsTestClient/1.0",
    )

    assert user.email == "member@navigators.dev"
    assert session.is_guest is False
    assert user.last_login_at is not None
    assert "place:read" in session.permissions


def test_login_invalid_credentials(auth_service: AuthService):
    """Login with invalid password or non-existent email must fail."""
    auth_service.register(
        email="pilot@navigators.dev",
        password="CorrectPassword123!",
        name="Nav Pilot",
    )

    with pytest.raises(ValueError, match="Invalid email or password"):
        auth_service.login(email="pilot@navigators.dev", password="WrongPassword!")

    with pytest.raises(ValueError, match="Invalid email or password"):
        auth_service.login(email="nonexistent@navigators.dev", password="SomePassword123!")


def test_login_suspended_account_rejected(auth_service: AuthService, rbac_repo: RBACRepository):
    """Suspended accounts cannot log in."""
    user, _, _ = auth_service.register(
        email="suspended@navigators.dev",
        password="Password123456!",
        name="Suspended Account",
    )
    rbac_repo.update_user(user.id, status="suspended")

    with pytest.raises(ValueError, match="Access denied"):
        auth_service.login(email="suspended@navigators.dev", password="Password123456!")







def test_session_resolution_and_last_seen_update(auth_service: AuthService):
    """Resolving session updates last_seen_at and loads user permissions."""
    _, initial_session, token = auth_service.register(
        email="tracker@navigators.dev",
        password="TrackPassword123!",
        name="Tracker User",
    )

    resolved = auth_service.resolve_session(token)
    assert resolved is not None
    assert resolved.user is not None
    assert resolved.user.email == "tracker@navigators.dev"
    assert resolved.session_id == initial_session.session_id
    assert resolved.last_seen_at is not None


def test_session_expiration(auth_service: AuthService, temp_db: Path):
    """Expired sessions must not be resolved."""
    _, _, token = auth_service.register(
        email="expiring@navigators.dev",
        password="ExpiringPassword123!",
        name="Expiring User",
    )

    token_hash = compute_token_hash(token)
    past_iso = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    conn = connect_db(temp_db)
    conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?", (past_iso, token_hash))
    conn.commit()
    conn.close()

    assert auth_service.resolve_session(token) is None


def test_session_revocation(auth_service: AuthService, temp_db: Path):
    """Revoked sessions must immediately become invalid."""
    _, _, token = auth_service.register(
        email="revokeme@navigators.dev",
        password="RevokePassword123!",
        name="Revoke User",
    )


    assert auth_service.resolve_session(token) is not None


    revoked = auth_service.revoke_session(token)
    assert revoked is True


    assert auth_service.resolve_session(token) is None


    conn = connect_db(temp_db)
    token_hash = compute_token_hash(token)
    row = conn.execute("SELECT revoked_at FROM sessions WHERE token_hash = ?", (token_hash,)).fetchone()
    conn.close()
    assert row["revoked_at"] is not None


def test_revoke_all_user_sessions(auth_service: AuthService):
    """Revoking all sessions for a user invalidates multiple active devices."""
    user, _, token1 = auth_service.register(
        email="multidevice@navigators.dev",
        password="MultiDevicePass123!",
        name="Multi Device",
    )
    _, _, token2 = auth_service.login(
        email="multidevice@navigators.dev",
        password="MultiDevicePass123!",
    )

    assert auth_service.resolve_session(token1) is not None
    assert auth_service.resolve_session(token2) is not None

    count = auth_service.revoke_all_user_sessions(user.id)
    assert count == 2

    assert auth_service.resolve_session(token1) is None
    assert auth_service.resolve_session(token2) is None






def test_guest_session_flow(auth_service: AuthService, temp_db: Path):
    """
    Guest flow:
    1. No entry created in users table.
    2. Session entry has is_guest = 1, user_id = NULL.
    3. Session resolves with virtual guest user and 'guest' role.
    4. Permissions contain 'place:read' only.
    """
    conn = connect_db(temp_db)
    initial_user_count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    conn.close()

    guest_context, guest_token = auth_service.create_guest_session(user_agent="MobileBrowser/1.0")

    assert guest_context.is_guest is True
    assert guest_context.user is not None
    assert guest_context.user.name == "Guest Explorer"
    assert "place:read" in guest_context.permissions
    assert "contribution:create" not in guest_context.permissions


    conn = connect_db(temp_db)
    user_count = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    conn.close()
    assert user_count == initial_user_count


    resolved = auth_service.resolve_session(guest_token)
    assert resolved is not None
    assert resolved.is_guest is True
    assert "place:read" in resolved.permissions







def test_server_determines_permissions_dynamically(
    auth_service: AuthService,
    rbac_repo: RBACRepository,
):
    """
    Permissions are loaded dynamically from the database.
    Granting or revoking roles in the database reflects immediately in the session
    without trusting any client role claims.
    """
    user, _, token = auth_service.register(
        email="dynamicrole@navigators.dev",
        password="DynamicRolePass123!",
        name="Dynamic User",
    )


    session = auth_service.resolve_session(token)
    assert session is not None
    assert "training:create" not in session.permissions


    rbac_repo.assign_role_to_user(user.id, "internal_contributor")


    updated_session = auth_service.resolve_session(token)
    assert updated_session is not None
    assert "training:create" in updated_session.permissions
    assert "dataset:create" in updated_session.permissions


    rbac_repo.remove_role_from_user(user.id, "internal_contributor")

    revoked_session = auth_service.resolve_session(token)
    assert revoked_session is not None
    assert "training:create" not in revoked_session.permissions






def test_api_auth_endpoints(monkeypatch, temp_db: Path):
    """Test register, login, guest, me, and logout API handlers."""

    test_auth_service = AuthService(temp_db)
    import src.api.auth as api_mod
    monkeypatch.setattr(api_mod, "auth_service", test_auth_service)
    monkeypatch.setattr(api_mod, "repo", test_auth_service.rbac)


    reg_resp = api_register(
        req=RegisterRequest(
            email="api_user@navigators.dev",
            password="ApiUserPass123!",
            name="API User",
        ),
        user_agent="TestRunner",
    )
    token = reg_resp["token"]
    assert token is not None
    assert reg_resp["session"]["is_guest"] is False


    auth_header = f"Bearer {token}"
    context = get_current_session(authorization=auth_header)
    assert context.user is not None
    assert context.user.email == "api_user@navigators.dev"

    profile_resp = api_get_current_user_profile(context=context)
    assert profile_resp["user"]["email"] == "api_user@navigators.dev"


    check_read = require_permission("place:read")
    assert check_read(context) is context

    check_deploy = require_permission("model:deploy")
    with pytest.raises(HTTPException) as exc_info:
        check_deploy(context)
    assert exc_info.value.status_code == 403


    login_resp = api_login(
        req=LoginRequest(
            email="api_user@navigators.dev",
            password="ApiUserPass123!",
        ),
        user_agent="TestRunner",
    )
    new_token = login_resp["token"]
    assert new_token is not None


    guest_resp = api_create_guest(user_agent="TestRunner")
    assert guest_resp["session"]["is_guest"] is True
    assert "place:read" in guest_resp["session"]["permissions"]


    logout_resp = api_logout(authorization=auth_header)
    assert logout_resp["revoked"] is True


    with pytest.raises(HTTPException) as exc_info:
        get_current_session(authorization=auth_header)
    assert exc_info.value.status_code == 401
