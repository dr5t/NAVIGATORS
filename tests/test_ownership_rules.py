"""
Navigators IDR - Phase 5 Ownership Rules Test Suite
Validates:
  - User 47 creates Contribution #182 (owner_id = 47).
  - User 47 can read, update, delete/withdraw their own draft.
  - User 82 cannot edit, withdraw, or view User 47's unsubmitted draft (NOT_OWNER).
  - Moderator can review (approve/reject) pending submission.
  - Moderator cannot edit the author's content.
  - Owner cannot self-approve without moderation rights.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService, AuthorizationResult
from src.db.contributions import ContributionRepository, Contribution
from src.api.contributions import (
    create_contribution as api_create_contribution,
    get_contribution as api_get_contribution,
    update_contribution as api_update_contribution,
    withdraw_contribution as api_withdraw_contribution,
    submit_contribution as api_submit_contribution,
    review_contribution as api_review_contribution,
    list_contributions as api_list_contributions,
    CreateContributionRequest,
    UpdateContributionRequest,
    ReviewContributionRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_ownership_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def rbac_repo(temp_db: Path):
    return RBACRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path, rbac_repo: RBACRepository):
    return AuthService(temp_db, rbac_repo)


@pytest.fixture
def authz_service(temp_db: Path, rbac_repo: RBACRepository):
    return AuthorizationService(temp_db, rbac_repo)


@pytest.fixture
def contrib_repo(temp_db: Path):
    return ContributionRepository(temp_db)


# =============================================================================
# 1. Direct Scenario: User 47 vs User 82 vs Moderator
# =============================================================================

def test_user_47_and_user_82_ownership_scenario(
    auth_service: AuthService,
    authz_service: AuthorizationService,
    contrib_repo: ContributionRepository,
):
    """
    Scenario:
      A normal user creates:
        Contribution #182
        owner_id = 47
      User 47 can:
        - read
        - update
        - delete/withdraw
        their own draft.
      User 82 cannot edit it.
      But a moderator can review it.
    """
    # Register User 47 (Normal registered user)
    user_47, session_47, _ = auth_service.register(
        email="user47@navigators.dev",
        password="Password4747!",
        name="User 47",
    )

    # Register User 82 (Different registered user)
    user_82, session_82, _ = auth_service.register(
        email="user82@navigators.dev",
        password="Password8282!",
        name="User 82",
    )

    # Register Moderator
    moderator, session_mod, _ = auth_service.register(
        email="moderator@navigators.dev",
        password="ModPassword99!",
        name="Community Moderator",
        role_id="moderator",
    )

    # User 47 creates Contribution #182 (draft)
    contrib_182 = contrib_repo.create(
        contribution_id="contrib_182",
        owner_id=user_47.id,
        resource_type="place",
        title="EV Charging Hub Proposal",
        data={"amenity": "charging_station", "plugs": 6},
        status="draft",
    )
    assert contrib_182.owner_id == user_47.id
    assert contrib_182.status == "draft"

    # -------------------------------------------------------------------------
    # Rule 1: User 47 (Owner) has own resource permissions
    # User 47 can read their own draft
    can_read_47 = authz_service.can(user=session_47, action="contribution:read", resource=contrib_182)
    assert can_read_47.allowed is True
    assert can_read_47.code == "AUTHORIZED"

    # User 47 can update their own draft
    can_update_47 = authz_service.can(user=session_47, action="contribution:update", resource=contrib_182)
    assert can_update_47.allowed is True
    assert can_update_47.code == "AUTHORIZED"

    # User 47 can withdraw/delete their own draft
    can_withdraw_47 = authz_service.can(user=session_47, action="contribution:withdraw", resource=contrib_182)
    assert can_withdraw_47.allowed is True
    assert can_withdraw_47.code == "AUTHORIZED"

    # -------------------------------------------------------------------------
    # Rule 2: User 82 (Non-owner) cannot edit or view User 47's draft
    # User 82 cannot read User 47's unsubmitted draft
    can_read_82 = authz_service.can(user=session_82, action="contribution:read", resource=contrib_182)
    assert can_read_82.allowed is False
    assert can_read_82.code == "NOT_OWNER"

    # User 82 cannot edit User 47's draft
    can_update_82 = authz_service.can(user=session_82, action="contribution:update", resource=contrib_182)
    assert can_update_82.allowed is False
    assert can_update_82.code == "NOT_OWNER"

    # User 82 cannot withdraw User 47's draft
    can_withdraw_82 = authz_service.can(user=session_82, action="contribution:withdraw", resource=contrib_182)
    assert can_withdraw_82.allowed is False
    assert can_withdraw_82.code == "NOT_OWNER"

    # -------------------------------------------------------------------------
    # Rule 3: Moderator cannot edit User 47's content
    can_mod_edit = authz_service.can(user=session_mod, action="contribution:update", resource=contrib_182)
    assert can_mod_edit.allowed is False
    assert can_mod_edit.code == "NOT_OWNER"

    # -------------------------------------------------------------------------
    # Rule 4: Review requires 'pending' status
    # In 'draft' status, moderator cannot review yet
    can_mod_review_draft = authz_service.can(user=session_mod, action="contribution:approve", resource=contrib_182)
    assert can_mod_review_draft.allowed is False
    assert can_mod_review_draft.code == "INVALID_RESOURCE_STATE"

    # User 47 submits contribution to pending review
    contrib_pending = contrib_repo.submit("contrib_182")
    assert contrib_pending.status in ("pending", "pending_review")

    # Now Moderator can review (approve/reject)
    can_mod_approve = authz_service.can(user=session_mod, action="contribution:approve", resource=contrib_pending)
    assert can_mod_approve.allowed is True
    assert can_mod_approve.code == "AUTHORIZED"

    can_mod_reject = authz_service.can(user=session_mod, action="contribution:reject", resource=contrib_pending)
    assert can_mod_reject.allowed is True
    assert can_mod_reject.code == "AUTHORIZED"

    # -------------------------------------------------------------------------
    # Rule 5: User 47 cannot self-approve without moderator rights
    can_self_approve = authz_service.can(user=session_47, action="contribution:approve", resource=contrib_pending)
    assert can_self_approve.allowed is False
    assert can_self_approve.code == "PERMISSION_DENIED"


# =============================================================================
# 2. API Endpoints Integration Test
# =============================================================================

def test_api_contributions_ownership_flow(monkeypatch, temp_db: Path, auth_service: AuthService):
    """Test REST API layer enforcing ownership and moderation boundaries."""
    import src.api.contributions as contrib_mod
    import src.api.auth as auth_mod

    test_repo = ContributionRepository(temp_db)
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(contrib_mod, "contrib_repo", test_repo)
    monkeypatch.setattr(contrib_mod, "auth_service", auth_service)
    monkeypatch.setattr(contrib_mod, "authz_service", test_authz)

    # 1. User 47 registers and creates a draft contribution
    user_47, session_47, token_47 = auth_service.register(
        email="owner47@navigators.dev",
        password="Password47!",
        name="Owner 47",
    )
    user_82, session_82, token_82 = auth_service.register(
        email="other82@navigators.dev",
        password="Password82!",
        name="Other 82",
    )
    mod, session_mod, token_mod = auth_service.register(
        email="modadmin@navigators.dev",
        password="ModPassword123!",
        name="Mod Staff",
        role_id="moderator",
    )

    # User 47 creates draft contribution
    create_resp = api_create_contribution(
        req=CreateContributionRequest(
            resource_type="place",
            title="City Library Amenity",
            data={"books": True},
            submit_now=False,
        ),
        context=session_47,
    )
    contrib_id = create_resp["contribution"]["id"]
    assert create_resp["contribution"]["status"] == "draft"

    # User 47 can read it
    get_resp = api_get_contribution(contrib_id=contrib_id, context=session_47)
    assert get_resp["contribution"]["title"] == "City Library Amenity"

    # User 82 cannot read User 47's draft (HTTP 403)
    with pytest.raises(HTTPException) as exc_get:
        api_get_contribution(contrib_id=contrib_id, context=session_82)
    assert exc_get.value.status_code == 403

    # User 82 cannot update User 47's draft (HTTP 403)
    with pytest.raises(HTTPException) as exc_update:
        api_update_contribution(
            contrib_id=contrib_id,
            req=UpdateContributionRequest(title="Tampered Title"),
            context=session_82,
        )
    assert exc_update.value.status_code == 403
    assert "Only the author/owner" in exc_update.value.detail

    # User 47 can update title
    update_resp = api_update_contribution(
        contrib_id=contrib_id,
        req=UpdateContributionRequest(title="Updated City Library"),
        context=session_47,
    )
    assert update_resp["contribution"]["title"] == "Updated City Library"

    # User 47 submits draft for review
    submit_resp = api_submit_contribution(contrib_id=contrib_id, context=session_47)
    assert submit_resp["contribution"]["status"] in ("pending", "pending_review")

    # Moderator reviews and approves
    review_resp = api_review_contribution(
        contrib_id=contrib_id,
        req=ReviewContributionRequest(decision="approved", notes="Verified location and data"),
        context=session_mod,
    )
    assert review_resp["contribution"]["status"] == "approved"
    assert review_resp["contribution"]["reviewed_by"] == mod.id

    # Once approved, User 47 can no longer edit or withdraw
    with pytest.raises(HTTPException) as exc_locked:
        api_update_contribution(
            contrib_id=contrib_id,
            req=UpdateContributionRequest(title="Post-Approval Tamper"),
            context=session_47,
        )
    assert exc_locked.value.status_code == 403
    assert "Cannot perform 'contribution:update'" in exc_locked.value.detail
