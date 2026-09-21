"""
Navigators IDR - Phase 6 Resource State Machine Test Suite
Validates the community contribution lifecycle:
  DRAFT -> SUBMITTED -> PENDING_REVIEW -> APPROVED -> PUBLISHED
  Alternative: PENDING_REVIEW -> REJECTED
  Withdrawal: DRAFT / SUBMITTED / PENDING_REVIEW -> WITHDRAWN

Key guarantees:
  1. Arbitrary state jumps (e.g. DRAFT -> APPROVED) are strictly impossible.
  2. Standard users cannot self-approve (User -> APPROVED is impossible).
  3. Only reviewer roles (moderator, team_admin, super_admin) can approve or reject.
  4. Only staff roles can publish approved contributions to canonical map data.
  5. Direct mutation cannot bypass the state machine.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.db.contributions import ContributionRepository, Contribution
from src.db.state_machine import (
    ContributionStateMachine,
    ContributionState,
    StateTransitionError,
)
from src.api.contributions import (
    create_contribution as api_create_contribution,
    submit_contribution as api_submit_contribution,
    approve_contribution as api_approve_contribution,
    reject_contribution as api_reject_contribution,
    publish_contribution as api_publish_contribution,
    withdraw_contribution as api_withdraw_contribution,
    CreateContributionRequest,
    ReviewNotesRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_state_machine_test.db"
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
# 1. State Machine Engine Unit Tests
# =============================================================================

def test_state_machine_happy_path(auth_service: AuthService, contrib_repo: ContributionRepository):
    """
    Test full happy path lifecycle:
    DRAFT -> SUBMITTED -> PENDING_REVIEW -> APPROVED -> PUBLISHED
    """
    author, session_author, _ = auth_service.register(
        email="author@navigators.dev",
        password="Password123!",
        name="Map Contributor",
    )
    moderator, session_mod, _ = auth_service.register(
        email="moderator@navigators.dev",
        password="Password123!",
        name="Map Moderator",
        role_id="moderator",
    )

    # 1. Author creates draft
    item = contrib_repo.create(
        contribution_id="contrib_happy_1",
        owner_id=author.id,
        resource_type="place",
        title="New Bicycle Parking Station",
        data={"capacity": 20},
    )
    assert item.status == ContributionState.DRAFT
    assert item.published_at is None

    # 2. Author transitions DRAFT -> SUBMITTED
    submitted = contrib_repo.transition_state(
        contribution_id=item.id,
        target_state=ContributionState.SUBMITTED,
        user=session_author,
    )
    assert submitted.status == ContributionState.SUBMITTED

    # 3. Ingestion/triage queues SUBMITTED -> PENDING_REVIEW
    pending = contrib_repo.transition_state(
        contribution_id=item.id,
        target_state=ContributionState.PENDING_REVIEW,
        user=session_mod,
    )
    assert pending.status == ContributionState.PENDING_REVIEW

    # 4. Reviewer approves PENDING_REVIEW -> APPROVED
    approved = contrib_repo.approve(
        contribution_id=item.id,
        reviewer=session_mod,
        notes="Verified bicycle racks coordinates on site.",
    )
    assert approved.status == ContributionState.APPROVED
    assert approved.reviewed_by == moderator.id
    assert approved.review_notes == "Verified bicycle racks coordinates on site."
    assert approved.reviewed_at is not None

    # 5. Staff publishes APPROVED -> PUBLISHED
    published = contrib_repo.publish(
        contribution_id=item.id,
        staff=session_mod,
    )
    assert published.status == ContributionState.PUBLISHED
    assert published.published_at is not None


def test_state_machine_direct_submission_path(auth_service: AuthService, contrib_repo: ContributionRepository):
    """Author can submit directly from DRAFT to PENDING_REVIEW."""
    author, session_author, _ = auth_service.register(
        email="author2@navigators.dev",
        password="Password123!",
        name="Direct Submitter",
    )
    item = contrib_repo.create(
        contribution_id="contrib_direct_1",
        owner_id=author.id,
        resource_type="amenity",
        title="Public Water Fountain",
    )
    assert item.status == ContributionState.DRAFT

    pending = contrib_repo.submit(item.id, user=session_author)
    assert pending.status == ContributionState.PENDING_REVIEW


def test_state_machine_rejection_path(auth_service: AuthService, contrib_repo: ContributionRepository):
    """
    Test rejection lifecycle:
    DRAFT -> PENDING_REVIEW -> REJECTED
    """
    author, session_author, _ = auth_service.register(
        email="author3@navigators.dev",
        password="Password123!",
        name="Author Three",
    )
    moderator, session_mod, _ = auth_service.register(
        email="mod3@navigators.dev",
        password="Password123!",
        name="Moderator Three",
        role_id="moderator",
    )

    item = contrib_repo.create(
        contribution_id="contrib_reject_1",
        owner_id=author.id,
        resource_type="place",
        title="Commercial Spam Listing",
    )
    contrib_repo.submit(item.id, user=session_author)

    rejected = contrib_repo.reject(
        contribution_id=item.id,
        reviewer=session_mod,
        notes="Duplicate listing and invalid coordinates.",
    )
    assert rejected.status == ContributionState.REJECTED
    assert rejected.reviewed_by == moderator.id
    assert "Duplicate listing" in rejected.review_notes


def test_state_machine_withdrawal_paths(auth_service: AuthService, contrib_repo: ContributionRepository):
    """
    Author can withdraw from DRAFT, SUBMITTED, and PENDING_REVIEW states.
    """
    author, session_author, _ = auth_service.register(
        email="author4@navigators.dev",
        password="Password123!",
        name="Withdraw Tester",
    )

    # From DRAFT -> WITHDRAWN
    c1 = contrib_repo.create("contrib_w1", author.id, "place", "Draft to Cancel")
    w1 = contrib_repo.withdraw(c1.id, user=session_author)
    assert w1.status == ContributionState.WITHDRAWN

    # From SUBMITTED -> WITHDRAWN
    c2 = contrib_repo.create("contrib_w2", author.id, "place", "Submitted to Cancel")
    contrib_repo.transition_state(c2.id, ContributionState.SUBMITTED, user=session_author)
    w2 = contrib_repo.withdraw(c2.id, user=session_author)
    assert w2.status == ContributionState.WITHDRAWN

    # From PENDING_REVIEW -> WITHDRAWN
    c3 = contrib_repo.create("contrib_w3", author.id, "place", "Pending to Cancel")
    contrib_repo.submit(c3.id, user=session_author)
    w3 = contrib_repo.withdraw(c3.id, user=session_author)
    assert w3.status == ContributionState.WITHDRAWN


# =============================================================================
# 2. Arbitrary State Jump Prevention
# =============================================================================

def test_arbitrary_state_jumps_are_strictly_prevented(
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Arbitrary state jumps must raise StateTransitionError with code INVALID_STATE_TRANSITION.
    """
    author, session_author, _ = auth_service.register(
        email="hacker@navigators.dev",
        password="Password123!",
        name="Curious Contributor",
    )
    moderator, session_mod, _ = auth_service.register(
        email="mod_sec@navigators.dev",
        password="Password123!",
        name="Security Moderator",
        role_id="moderator",
    )

    item = contrib_repo.create("contrib_sec_1", author.id, "place", "Security Test Item")

    # 1. Cannot jump DRAFT -> APPROVED
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.transition_state(item.id, ContributionState.APPROVED, user=session_mod)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"
    assert "Illegal state transition" in exc_info.value.message

    # 2. Cannot jump DRAFT -> PUBLISHED
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.transition_state(item.id, ContributionState.PUBLISHED, user=session_mod)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"

    # Move to pending
    contrib_repo.submit(item.id, user=session_author)

    # 3. Cannot jump PENDING_REVIEW -> PUBLISHED directly (must be approved first)
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.transition_state(item.id, ContributionState.PUBLISHED, user=session_mod)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"

    # Reject item
    contrib_repo.reject(item.id, reviewer=session_mod, notes="Rejected for testing")

    # 4. Cannot transition REJECTED -> PUBLISHED
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.transition_state(item.id, ContributionState.PUBLISHED, user=session_mod)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"

    # 5. Cannot transition REJECTED -> APPROVED
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.transition_state(item.id, ContributionState.APPROVED, user=session_mod)
    assert exc_info.value.code == "INVALID_STATE_TRANSITION"


# =============================================================================
# 3. Role & Permission Guards
# =============================================================================

def test_standard_user_cannot_self_approve(
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Guarantee: User -> APPROVED must be strictly impossible.
    Even though the user owns the contribution, they cannot approve it.
    """
    user, session_user, _ = auth_service.register(
        email="self_approver@navigators.dev",
        password="Password123!",
        name="Sneaky User",
    )
    item = contrib_repo.create("contrib_self_app_1", user.id, "place", "Self-Promotion POI")
    contrib_repo.submit(item.id, user=session_user)

    # Standard user attempting approve
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.approve(item.id, reviewer=session_user)
    assert exc_info.value.code == "PERMISSION_DENIED"
    assert "Only authorized reviewer roles" in exc_info.value.message


def test_standard_user_cannot_publish(
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Standard users cannot publish to canonical map data.
    """
    author, session_author, _ = auth_service.register(
        email="author_pub@navigators.dev",
        password="Password123!",
        name="Author Pub",
    )
    moderator, session_mod, _ = auth_service.register(
        email="mod_pub@navigators.dev",
        password="Password123!",
        name="Mod Pub",
        role_id="moderator",
    )

    item = contrib_repo.create("contrib_pub_guard_1", author.id, "place", "Guarded Publish Place")
    contrib_repo.submit(item.id, user=session_author)
    contrib_repo.approve(item.id, reviewer=session_mod)

    # Author attempts publish
    with pytest.raises(StateTransitionError) as exc_info:
        contrib_repo.publish(item.id, staff=session_author)
    assert exc_info.value.code == "PERMISSION_DENIED"
    assert "Staff role required" in exc_info.value.message


def test_non_owner_cannot_submit_or_withdraw(
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    User B cannot submit or withdraw User A's draft.
    """
    user_a, session_a, _ = auth_service.register(
        email="usera@navigators.dev", password="Password123!", name="User A"
    )
    user_b, session_b, _ = auth_service.register(
        email="userb@navigators.dev", password="Password123!", name="User B"
    )

    item = contrib_repo.create("contrib_owner_guard_1", user_a.id, "place", "User A Draft")

    # User B attempts submit
    with pytest.raises(StateTransitionError) as exc_submit:
        contrib_repo.submit(item.id, user=session_b)
    assert exc_submit.value.code == "NOT_OWNER"

    # User B attempts withdraw
    with pytest.raises(StateTransitionError) as exc_withdraw:
        contrib_repo.withdraw(item.id, user=session_b)
    assert exc_withdraw.value.code == "NOT_OWNER"


def test_unauthenticated_actor_cannot_trigger_transitions(
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Unauthenticated guest or None actor cannot trigger state transitions.
    """
    author, session_author, _ = auth_service.register(
        email="auth_req@navigators.dev", password="Password123!", name="Author"
    )
    item = contrib_repo.create("contrib_unauth_1", author.id, "place", "Draft Place")

    # Anonymous guest context
    guest_session = auth_service.create_guest_session()

    with pytest.raises(StateTransitionError) as exc_guest:
        contrib_repo.submit(item.id, user=guest_session)
    assert exc_guest.value.code == "UNAUTHENTICATED"


# =============================================================================
# 4. REST API Transition Endpoints
# =============================================================================

def test_api_state_transition_endpoints_flow(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
):
    """
    Test REST endpoints:
      POST /api/v1/contributions/{id}/submit
      POST /api/v1/contributions/{id}/approve
      POST /api/v1/contributions/{id}/publish
      POST /api/v1/contributions/{id}/withdraw
    """
    import src.api.contributions as contrib_mod

    test_repo = ContributionRepository(temp_db)
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(contrib_mod, "contrib_repo", test_repo)
    monkeypatch.setattr(contrib_mod, "auth_service", auth_service)
    monkeypatch.setattr(contrib_mod, "authz_service", test_authz)

    author, session_author, _ = auth_service.register(
        email="api_author@navigators.dev", password="Password123!", name="API Author"
    )
    other_user, session_other, _ = auth_service.register(
        email="api_other@navigators.dev", password="Password123!", name="API Other"
    )
    moderator, session_mod, _ = auth_service.register(
        email="api_mod@navigators.dev", password="Password123!", name="API Mod", role_id="moderator"
    )

    # 1. Author creates draft via API
    create_resp = api_create_contribution(
        req=CreateContributionRequest(
            resource_type="place",
            title="Solar Bench Hub",
            data={"solar": True, "usb_ports": 4},
        ),
        context=session_author,
    )
    contrib_id = create_resp["contribution"]["id"]
    assert create_resp["contribution"]["status"] == ContributionState.DRAFT

    # 2. Other user cannot submit User A's draft (HTTP 403 NOT_OWNER)
    with pytest.raises(HTTPException) as exc_other_submit:
        api_submit_contribution(contrib_id=contrib_id, context=session_other)
    assert exc_other_submit.value.status_code == 403

    # 3. Moderator cannot approve draft before it is submitted (HTTP 403/400)
    with pytest.raises(HTTPException) as exc_early_approve:
        api_approve_contribution(
            contrib_id=contrib_id,
            req=ReviewNotesRequest(notes="Premature approval"),
            context=session_mod,
        )
    assert exc_early_approve.value.status_code in (400, 403)

    # 4. Author submits draft for review
    submit_resp = api_submit_contribution(contrib_id=contrib_id, context=session_author)
    assert submit_resp["contribution"]["status"] == ContributionState.PENDING_REVIEW

    # 5. Author cannot self-approve (HTTP 403)
    with pytest.raises(HTTPException) as exc_self_app:
        api_approve_contribution(
            contrib_id=contrib_id,
            req=ReviewNotesRequest(notes="Self approval attempt"),
            context=session_author,
        )
    assert exc_self_app.value.status_code == 403

    # 6. Moderator approves contribution
    approve_resp = api_approve_contribution(
        contrib_id=contrib_id,
        req=ReviewNotesRequest(notes="Approved by urban mobility staff"),
        context=session_mod,
    )
    assert approve_resp["contribution"]["status"] == ContributionState.APPROVED
    assert approve_resp["contribution"]["reviewed_by"] == moderator.id

    # 7. Author cannot withdraw already approved contribution (HTTP 403)
    with pytest.raises(HTTPException) as exc_post_app_withdraw:
        api_withdraw_contribution(contrib_id=contrib_id, context=session_author)
    assert exc_post_app_withdraw.value.status_code == 403

    # 8. Staff publishes approved contribution to canonical live map data
    pub_resp = api_publish_contribution(contrib_id=contrib_id, context=session_mod)
    assert pub_resp["contribution"]["status"] == ContributionState.PUBLISHED
    assert pub_resp["contribution"]["published_at"] is not None
