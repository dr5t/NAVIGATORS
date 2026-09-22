"""
Navigators IDR - Phase 16 Mac Dashboard RBAC & Role Gating Tests
Validates the role permission matrix and access control rules for:
  - User & Local Contributor (No dashboard privileges)
  - Internal Contributor (Datasets, Experiments, Training Jobs)
  - Moderator (Moderation Tools & Triage)
  - Team Admin (Full Engineering & Governance Controls)
"""

import pytest
from pathlib import Path

from src.db.database import get_db, init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService
from src.db.authorization import AuthorizationService
from src.api.server import get_dashboard


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a clean database initialized with canonical schema."""
    db_file = tmp_path / "test_mac_dashboard_rbac.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def auth_roles(temp_db):
    """Fixture providing test users and resolved SessionContext for each role."""
    auth_service = AuthService(temp_db)


    user, user_session, _ = auth_service.register(
        email="user@navigators.test", password="Password123!", name="User", role_id="user"
    )


    lc_user, lc_session, _ = auth_service.register(
        email="lc@navigators.test", password="Password123!", name="Local Contributor", role_id="local_contributor"
    )


    ic_user, ic_session, _ = auth_service.register(
        email="ic@navigators.test", password="Password123!", name="Internal Contributor", role_id="internal_contributor"
    )


    mod_user, mod_session, _ = auth_service.register(
        email="mod@navigators.test", password="Password123!", name="Moderator", role_id="moderator"
    )


    admin_user, admin_session, _ = auth_service.register(
        email="admin@navigators.test", password="Password123!", name="Team Admin", role_id="team_admin"
    )

    return {
        "user_session": user_session,
        "lc_session": lc_session,
        "ic_session": ic_session,
        "mod_session": mod_session,
        "admin_session": admin_session,
    }


def test_dashboard_route_serves_html():
    """Verify get_dashboard() serves the Mac Dashboard UI file."""
    response = get_dashboard()
    assert response is not None
    assert Path(response.path).exists()
    content = Path(response.path).read_text()
    assert "MAC DASHBOARD" in content
    assert "engine/auth.js" in content
    assert "mac_dashboard.js" in content


def test_user_and_local_contributor_have_no_engineering_permissions(temp_db, auth_roles):
    """Verify 'user', 'local_contributor', and 'guest' roles cannot dispatch training or deploy models."""
    authz = AuthorizationService(temp_db)

    for session in (auth_roles["user_session"], auth_roles["lc_session"]):

        assert not authz.can(user=session, action="training:create", resource="training").allowed


        assert not authz.can(user=session, action="model:deploy", resource="model").allowed


        assert not authz.can(user=session, action="dataset:validate", resource="dataset").allowed


        assert not authz.can(user=session, action="internal_contributor:review", resource="contributor").allowed


def test_internal_contributor_permissions(temp_db, auth_roles):
    """Verify 'internal_contributor' has datasets & training permissions, but no admin deployment."""
    authz = AuthorizationService(temp_db)
    session = auth_roles["ic_session"]


    assert authz.can(user=session, action="dataset:create", resource="dataset").allowed
    assert authz.can(user=session, action="dataset:read", resource="dataset").allowed


    assert authz.can(user=session, action="training:create", resource="training").allowed
    assert authz.can(user=session, action="training:read", resource="training").allowed


    assert authz.can(user=session, action="model:read", resource="model").allowed


    assert not authz.can(user=session, action="model:deploy", resource="model").allowed
    assert not authz.can(user=session, action="dataset:validate", resource="dataset").allowed
    assert not authz.can(user=session, action="internal_contributor:approve", resource="contributor").allowed


def test_moderator_permissions(temp_db, auth_roles):
    """Verify 'moderator' has moderation permissions, but no ML training or model deployment."""
    authz = AuthorizationService(temp_db)
    session = auth_roles["mod_session"]


    assert authz.can(user=session, action="contribution:approve", resource="contribution").allowed
    assert authz.can(user=session, action="contribution:reject", resource="contribution").allowed


    assert not authz.can(user=session, action="training:create", resource="training").allowed
    assert not authz.can(user=session, action="model:deploy", resource="model").allowed


def test_team_admin_permissions(temp_db, auth_roles):
    """Verify 'team_admin' has full engineering & governance permissions."""
    authz = AuthorizationService(temp_db)
    session = auth_roles["admin_session"]


    assert authz.can(user=session, action="dataset:create", resource="dataset").allowed
    assert authz.can(user=session, action="dataset:validate", resource="dataset").allowed
    assert authz.can(user=session, action="training:create", resource="training").allowed
    assert authz.can(user=session, action="model:review", resource="model").allowed
    assert authz.can(user=session, action="model:approve", resource="model").allowed
    assert authz.can(user=session, action="model:deploy", resource="model").allowed
    assert authz.can(user=session, action="internal_contributor:approve", resource="contributor").allowed
