"""
Navigators IDR - Central Authorization Service Test Suite
Validates:
  can(user, action, resource)
evaluating the 5-step pipeline:
  1. Authentication
  2. Role & Permissions
  3. Ownership
  4. Resource State
  5. Central Authorization Decision
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService, AuthorizationResult
from src.api.auth import (
    authorize_action as api_authorize_action,
    AuthorizeRequest,
    require_authz,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_authz_test.db"
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


@pytest.fixture
def authz_service(temp_db: Path, rbac_repo: RBACRepository):
    """Provide an AuthorizationService instance."""
    return AuthorizationService(temp_db, rbac_repo)






def test_unauthenticated_and_guest_capabilities(authz_service: AuthorizationService):
    """
    Guests (unauthenticated visitors) can read places,
    but cannot create places or submit contributions.
    """

    res_read = authz_service.can(user=None, action="place:read")
    assert res_read.allowed is True
    assert res_read.code == "AUTHORIZED"

    res_create = authz_service.can(user=None, action="place:create")
    assert res_create.allowed is False
    assert res_create.code == "UNAUTHENTICATED"


    guest_dict = {"is_guest": True, "status": "active", "permissions": ["place:read"]}
    res_contrib = authz_service.can(user=guest_dict, action="contribution:create")
    assert res_contrib.allowed is False
    assert res_contrib.code == "UNAUTHENTICATED"


def test_inactive_account_rejection(
    auth_service: AuthService,
    authz_service: AuthorizationService,
    rbac_repo: RBACRepository,
):
    """Suspended or deleted accounts are blocked from all actions."""
    user, _, _ = auth_service.register(
        email="inactive@navigators.dev",
        password="ValidPassword123!",
        name="Inactive Account",
    )

    rbac_repo.update_user(user.id, status="suspended")
    suspended_user = rbac_repo.get_user(user.id)

    res = authz_service.can(user=suspended_user, action="place:read")
    assert res.allowed is False
    assert res.code == "ACCOUNT_INACTIVE"






def test_permission_enforcement(
    auth_service: AuthService,
    authz_service: AuthorizationService,
    rbac_repo: RBACRepository,
):
    """Users can only perform actions granted by their assigned roles."""
    user, session, _ = auth_service.register(
        email="standard@navigators.dev",
        password="StandardPass123!",
        name="Standard User",
    )


    assert authz_service.can(user=session, action="place:read").allowed is True
    assert authz_service.can(user=session, action="contribution:create").allowed is True


    res_deploy = authz_service.can(user=session, action="model:deploy")
    assert res_deploy.allowed is False
    assert res_deploy.code == "PERMISSION_DENIED"

    res_approve = authz_service.can(user=session, action="contribution:approve")
    assert res_approve.allowed is False
    assert res_approve.code == "PERMISSION_DENIED"


    rbac_repo.assign_role_to_user(user.id, "moderator")
    user_mod = rbac_repo.get_user(user.id)


    res_mod = authz_service.can(user=user_mod, action="contribution:approve")
    assert res_mod.allowed is True
    assert res_mod.code == "AUTHORIZED"






def test_ownership_enforcement_for_contributions(
    auth_service: AuthService,
    authz_service: AuthorizationService,
    rbac_repo: RBACRepository,
):
    """
    Authors can update/withdraw their own contributions in pending state.
    Other users (even with contribution:update permission) cannot mutate others' submissions.
    """
    user_a, session_a, _ = auth_service.register(
        email="author_a@navigators.dev",
        password="AuthorPassword123!",
        name="Author Alice",
    )
    user_b, session_b, _ = auth_service.register(
        email="author_b@navigators.dev",
        password="AuthorPassword123!",
        name="Author Bob",
    )


    contribution_a = {
        "id": "contrib_101",
        "user_id": user_a.id,
        "resource_type": "place",
        "status": "pending",
    }


    res_alice_withdraw = authz_service.can(
        user=session_a,
        action="contribution:withdraw",
        resource=contribution_a,
    )
    assert res_alice_withdraw.allowed is True
    assert res_alice_withdraw.code == "AUTHORIZED"


    res_bob_withdraw = authz_service.can(
        user=session_b,
        action="contribution:withdraw",
        resource=contribution_a,
    )
    assert res_bob_withdraw.allowed is False
    assert res_bob_withdraw.code == "NOT_OWNER"


    res_bob_update = authz_service.can(
        user=session_b,
        action="contribution:update",
        resource=contribution_a,
    )
    assert res_bob_update.allowed is False
    assert res_bob_update.code == "NOT_OWNER"


def test_moderation_override_does_not_require_ownership(
    auth_service: AuthService,
    authz_service: AuthorizationService,
    rbac_repo: RBACRepository,
):
    """Moderators can approve or reject contributions created by any user."""
    user_a, _, _ = auth_service.register(
        email="alice@navigators.dev",
        password="AlicePass123!",
        name="Alice",
    )
    moderator, session_mod, _ = auth_service.register(
        email="mod@navigators.dev",
        password="ModPassword123!",
        name="Mod Charlie",
        role_id="moderator",
    )

    contribution = {
        "id": "contrib_202",
        "user_id": user_a.id,
        "status": "pending",
    }


    res_approve = authz_service.can(
        user=session_mod,
        action="contribution:approve",
        resource=contribution,
    )
    assert res_approve.allowed is True
    assert res_approve.code == "AUTHORIZED"






def test_contribution_lifecycle_state_constraints(
    auth_service: AuthService,
    authz_service: AuthorizationService,
):
    """
    Pending contributions can be updated or withdrawn.
    Approved or rejected contributions are locked against editing and withdrawal.
    """
    author, session_author, _ = auth_service.register(
        email="contributor@navigators.dev",
        password="ContributorPass123!",
        name="Contributor Dave",
    )


    pending_contrib = {"id": "c_pending", "user_id": author.id, "status": "pending"}
    assert authz_service.can(user=session_author, action="contribution:update", resource=pending_contrib).allowed is True
    assert authz_service.can(user=session_author, action="contribution:withdraw", resource=pending_contrib).allowed is True


    approved_contrib = {"id": "c_approved", "user_id": author.id, "status": "approved"}
    res_edit_approved = authz_service.can(user=session_author, action="contribution:update", resource=approved_contrib)
    assert res_edit_approved.allowed is False
    assert res_edit_approved.code == "INVALID_RESOURCE_STATE"

    res_withdraw_approved = authz_service.can(user=session_author, action="contribution:withdraw", resource=approved_contrib)
    assert res_withdraw_approved.allowed is False
    assert res_withdraw_approved.code == "INVALID_RESOURCE_STATE"


    rejected_contrib = {"id": "c_rejected", "user_id": author.id, "status": "rejected"}
    res_edit_rejected = authz_service.can(user=session_author, action="contribution:update", resource=rejected_contrib)
    assert res_edit_rejected.allowed is False
    assert res_edit_rejected.code == "INVALID_RESOURCE_STATE"


def test_model_deployment_state_constraints(
    auth_service: AuthService,
    authz_service: AuthorizationService,
):
    """
    Models can only be deployed if they are approved or qualified.
    Candidate or rejected models cannot be deployed into navigation runtime.
    """
    admin, session_admin, _ = auth_service.register(
        email="lead@navigators.dev",
        password="AdminPassword123!",
        name="Team Admin",
        role_id="team_admin",
    )

    candidate_model = {"id": "mdl_candidate", "status": "candidate"}
    res_candidate = authz_service.can(user=session_admin, action="model:deploy", resource=candidate_model)
    assert res_candidate.allowed is False
    assert res_candidate.code == "INVALID_RESOURCE_STATE"

    approved_model = {"id": "mdl_approved", "status": "approved"}
    res_approved = authz_service.can(user=session_admin, action="model:deploy", resource=approved_model)
    assert res_approved.allowed is True
    assert res_approved.code == "AUTHORIZED"


def test_dataset_mutation_state_constraints(
    auth_service: AuthService,
    authz_service: AuthorizationService,
):
    """Locked or archived datasets cannot be deleted or mutated."""
    admin, session_admin, _ = auth_service.register(
        email="data_lead@navigators.dev",
        password="DataLeadPass123!",
        name="Data Lead",
        role_id="team_admin",
    )

    locked_dataset = {"id": "ds_locked", "user_id": admin.id, "status": "locked"}
    res = authz_service.can(user=session_admin, action="dataset:delete", resource=locked_dataset)
    assert res.allowed is False
    assert res.code == "INVALID_RESOURCE_STATE"






def test_api_authorize_endpoint(monkeypatch, temp_db: Path, auth_service: AuthService):
    """Test POST /api/v1/auth/authorize endpoint with session token."""
    import src.api.auth as api_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(api_mod, "auth_service", auth_service)
    monkeypatch.setattr(api_mod, "authz_service", test_authz)


    resp_guest = api_authorize_action(
        req=AuthorizeRequest(action="place:read"),
        authorization=None,
    )
    assert resp_guest["allowed"] is True
    assert resp_guest["code"] == "AUTHORIZED"


    user, _, token = auth_service.register(
        email="query_user@navigators.dev",
        password="QueryPass123!",
        name="Query User",
    )
    auth_header = f"Bearer {token}"

    resp_auth = api_authorize_action(
        req=AuthorizeRequest(
            action="contribution:update",
            resource={"user_id": user.id, "status": "pending"},
        ),
        authorization=auth_header,
    )
    assert resp_auth["allowed"] is True
    assert resp_auth["code"] == "AUTHORIZED"


    resp_not_owner = api_authorize_action(
        req=AuthorizeRequest(
            action="contribution:update",
            resource={"user_id": "other_user_999", "status": "pending"},
        ),
        authorization=auth_header,
    )
    assert resp_not_owner["allowed"] is False
    assert resp_not_owner["code"] == "NOT_OWNER"


def test_fastapi_require_authz_dependency(monkeypatch, temp_db: Path, auth_service: AuthService):
    """Test require_authz dependency enforcing authorization gates."""
    import src.api.auth as api_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(api_mod, "auth_service", auth_service)
    monkeypatch.setattr(api_mod, "authz_service", test_authz)

    user, session, token = auth_service.register(
        email="dep_user@navigators.dev",
        password="DepPass123456!",
        name="Dep User",
    )


    checker = require_authz("place:read")
    assert checker(session) is session


    deploy_checker = require_authz("model:deploy")
    with pytest.raises(HTTPException) as exc_info:
        deploy_checker(session)
    assert exc_info.value.status_code == 403
    assert "lacks required permission" in exc_info.value.detail

