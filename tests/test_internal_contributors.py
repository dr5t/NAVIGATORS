"""
Tests for Phase 11 (Contributor System) and Phase 12 (Internal Contributor Approval)
Verifies that:
1. Users can submit internal contributor access requests.
2. Duplicate pending requests are blocked (409 Conflict).
3. Team admins can approve requests and promote the user role.
4. Team admins can reject requests with mandatory rationale.
5. Unauthorized users cannot access review or approval endpoints.
6. Road change suggestions flow through the community contribution pipeline.
7. Engineering controls (training, dataset, model deploy) are denied for normal users.
"""

import pytest
import secrets

from src.db.database import init_db
from src.db.internal_contributors import InternalContributorRepository
from src.db.auth_service import AuthService
from src.db.rbac import RBACRepository
from src.db.audit import AuditRepository
from src.db.authorization import AuthorizationService
from src.api.internal_contributors import (
    api_submit_internal_request,
    api_get_my_latest_request,
    api_list_internal_requests,
    api_get_internal_request,
    api_approve_internal_request,
    api_reject_internal_request,
    SubmitInternalRequestModel,
    RejectInternalRequestModel,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def _register_user(role: str = "user", tag: str = ""):
    auth_service = AuthService()
    suffix = tag or secrets.token_hex(4)
    user, session, _ = auth_service.register(
        email=f"icr_{role}_{suffix}@example.com",
        password="SecurePassword123!",
        name=f"ICR {role.title()} {suffix}",
        role_id=role,
    )
    return user, session


def test_user_can_submit_internal_contributor_request():
    """User submits request with valid reason and experience; status is pending."""
    tag = secrets.token_hex(4)
    user, session = _register_user("user", tag)
    icr_repo = InternalContributorRepository()

    req = icr_repo.submit_request(
        user_id=user.id,
        reason="I want to collect IMU sensor trajectories for urban canyon testing",
        experience="2 years embedded systems, Pixel 7 Pro with Android 14",
        requested_scope="trajectories_and_models",
    )

    assert req.status == "pending"
    assert req.user_id == user.id
    assert req.rejection_reason is None
    assert req.reviewed_by is None
    assert "trajectories" in req.requested_scope


def test_prevent_duplicate_pending_requests():
    """Submitting a second request while one is pending raises ValueError."""
    tag = secrets.token_hex(4)
    user, session = _register_user("user", tag)
    icr_repo = InternalContributorRepository()

    icr_repo.submit_request(
        user_id=user.id,
        reason="First application for sensor data collection",
        experience="Graduate researcher in inertial navigation",
    )

    with pytest.raises(ValueError, match="already exists"):
        icr_repo.submit_request(
            user_id=user.id,
            reason="Duplicate application",
            experience="Same experience",
        )


def test_team_admin_can_approve_request_and_promote_role():
    """Admin approves request; user gains internal_contributor role; audit logged."""
    tag = secrets.token_hex(4)
    user, user_session = _register_user("user", tag)
    admin, admin_session = _register_user("team_admin", f"admin_{tag}")
    icr_repo = InternalContributorRepository()
    rbac_repo = RBACRepository()
    audit_repo = AuditRepository()


    req = icr_repo.submit_request(
        user_id=user.id,
        reason="I want to contribute driving session datasets",
        experience="Automotive engineer with MEMS sensor experience",
    )


    roles_before = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles_before


    approved = icr_repo.approve_request(request_id=req.id, reviewer_id=admin.id)

    assert approved.status == "approved"
    assert approved.reviewed_by == admin.id
    assert approved.reviewed_at is not None


    roles_after = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" in roles_after


    logs = audit_repo.list_logs(resource_type="internal_contributor_request", resource_id=req.id)
    actions = [log.action for log in logs]
    assert "APPROVE_INTERNAL_ACCESS" in actions


def test_team_admin_can_reject_request_with_reason():
    """Admin rejects request with reason; user does not get promoted; audit logged."""
    tag = secrets.token_hex(4)
    user, user_session = _register_user("user", tag)
    admin, admin_session = _register_user("team_admin", f"admin_{tag}")
    icr_repo = InternalContributorRepository()
    rbac_repo = RBACRepository()
    audit_repo = AuditRepository()

    req = icr_repo.submit_request(
        user_id=user.id,
        reason="I want to help with model training",
        experience="Self-taught ML engineer",
    )

    rejected = icr_repo.reject_request(
        request_id=req.id,
        reviewer_id=admin.id,
        rejection_reason="Insufficient verifiable sensor hardware experience for field data collection",
    )

    assert rejected.status == "rejected"
    assert rejected.rejection_reason is not None
    assert "Insufficient" in rejected.rejection_reason
    assert rejected.reviewed_by == admin.id


    roles = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles


    logs = audit_repo.list_logs(resource_type="internal_contributor_request", resource_id=req.id)
    actions = [log.action for log in logs]
    assert "REJECT_INTERNAL_ACCESS" in actions


def test_unauthorized_user_cannot_access_review_or_approval_endpoints():
    """Regular user calling review/approve endpoints gets 403 Forbidden."""
    tag = secrets.token_hex(4)
    user, user_session = _register_user("user", tag)
    applicant, _ = _register_user("user", f"app_{tag}")
    icr_repo = InternalContributorRepository()

    req = icr_repo.submit_request(
        user_id=applicant.id,
        reason="Testing unauthorized access scenario",
        experience="Testing experience field",
    )


    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        api_list_internal_requests(context=user_session)
    assert exc_info.value.status_code == 403


    with pytest.raises(HTTPException) as exc_info:
        api_approve_internal_request(request_id=req.id, context=user_session)
    assert exc_info.value.status_code == 403


    with pytest.raises(HTTPException) as exc_info:
        api_reject_internal_request(
            request_id=req.id,
            body=RejectInternalRequestModel(rejection_reason="Trying to reject"),
            context=user_session,
        )
    assert exc_info.value.status_code == 403


def test_road_change_suggestion_workflow():
    """Community user submits road change suggestion via contribution pipeline."""
    from src.db.contributions import ContributionRepository

    tag = secrets.token_hex(4)
    user, session = _register_user("user", tag)
    contrib_repo = ContributionRepository()

    contrib = contrib_repo.create_contribution(
        owner_id=user.id,
        resource_type="road_change",
        title="Speed Limit Update: MG Road",
        data={
            "road_name": "MG Road",
            "change_type": "speed_limit_update",
            "current_speed_limit": 60,
            "proposed_speed_limit": 40,
            "reason": "School zone proximity",
            "coordinates": {"lat": 12.9716, "lng": 77.5946},
        },
    )

    assert contrib.status in ("draft", "submitted")
    assert contrib.owner_id == user.id
    fetched = contrib_repo.get_contribution(contrib.id)
    assert fetched is not None
    assert fetched.resource_type == "road_change"


def test_engineering_controls_hidden_for_normal_users():
    """Authorization service denies training:create, dataset:create, model:deploy for normal user."""
    tag = secrets.token_hex(4)
    user, session = _register_user("user", tag)
    authz = AuthorizationService()

    user_context = session.to_dict()


    decision_train = authz.can(user=user_context, action="training:create", resource="training")
    assert not decision_train.allowed


    decision_dataset = authz.can(user=user_context, action="dataset:create", resource="dataset")
    assert not decision_dataset.allowed


    decision_deploy = authz.can(user=user_context, action="model:deploy", resource="model")
    assert not decision_deploy.allowed


    decision_request = authz.can(user=user_context, action="internal_contributor:request", resource="contributor")
    assert decision_request.allowed






def test_api_submit_and_approve_full_flow(monkeypatch):
    """
    Full HTTP flow:
      api_submit_internal_request → api_approve_internal_request
    Verifies:
      - Submission returns wrapped {"request": {...}, "message": ...}
      - Status starts as 'pending'
      - After approval, user gains 'internal_contributor' role
      - Role was NOT present before approval
    """
    import src.api.internal_contributors as ic_mod

    tag = secrets.token_hex(4)
    auth_service = AuthService()
    user, user_session, _ = auth_service.register(
        email=f"api_submit_{tag}@example.com",
        password="Password123!",
        name=f"API Submit User {tag}",
        role_id="user",
    )
    admin, admin_session, _ = auth_service.register(
        email=f"api_admin_{tag}@example.com",
        password="Password123!",
        name=f"API Admin {tag}",
        role_id="team_admin",
    )

    test_icr_repo = InternalContributorRepository()
    test_authz = AuthorizationService()
    monkeypatch.setattr(ic_mod, "icr_repo", test_icr_repo)
    monkeypatch.setattr(ic_mod, "authz_service", test_authz)


    submit_resp = api_submit_internal_request(
        body=SubmitInternalRequestModel(
            reason="I collect IMU trajectories for urban navigation research",
            experience="3 years Android sensor development, published datasets",
            requested_scope="trajectories_and_models",
        ),
        context=user_session,
    )
    assert "request" in submit_resp
    assert "message" in submit_resp
    assert submit_resp["request"]["status"] == "pending"
    assert submit_resp["request"]["user_id"] == user.id
    request_id = submit_resp["request"]["id"]


    rbac_repo = RBACRepository()
    roles_before = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles_before


    approved = api_approve_internal_request(request_id=request_id, context=admin_session)
    assert approved["status"] == "approved"
    assert approved["reviewed_by"] == admin.id


    roles_after = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" in roles_after


def test_api_submit_and_reject_full_flow(monkeypatch):
    """
    Full HTTP flow:
      api_submit_internal_request → api_reject_internal_request
    Verifies:
      - After rejection, user does NOT gain 'internal_contributor' role
      - Rejection reason is stored
      - Status is 'rejected'
    """
    import src.api.internal_contributors as ic_mod
    from fastapi import HTTPException as FHTTPException

    tag = secrets.token_hex(4)
    auth_service = AuthService()
    user, user_session, _ = auth_service.register(
        email=f"api_reject_user_{tag}@example.com",
        password="Password123!",
        name=f"Reject User {tag}",
        role_id="user",
    )
    admin, admin_session, _ = auth_service.register(
        email=f"api_reject_admin_{tag}@example.com",
        password="Password123!",
        name=f"Reject Admin {tag}",
        role_id="team_admin",
    )

    test_icr_repo = InternalContributorRepository()
    test_authz = AuthorizationService()
    monkeypatch.setattr(ic_mod, "icr_repo", test_icr_repo)
    monkeypatch.setattr(ic_mod, "authz_service", test_authz)


    submit_resp = api_submit_internal_request(
        body=SubmitInternalRequestModel(
            reason="I want to upload training datasets",
            experience="Hobbyist, no verifiable hardware experience",
        ),
        context=user_session,
    )
    request_id = submit_resp["request"]["id"]


    rejected = api_reject_internal_request(
        request_id=request_id,
        body=RejectInternalRequestModel(
            rejection_reason="Insufficient verifiable sensor hardware experience for field data collection"
        ),
        context=admin_session,
    )
    assert rejected["status"] == "rejected"
    assert "Insufficient" in rejected["rejection_reason"]


    rbac_repo = RBACRepository()
    roles = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles


def test_api_duplicate_submission_returns_409(monkeypatch):
    """
    Submitting a second internal contributor request while one is pending
    must return HTTP 409 Conflict - not silently create a second record.
    """
    import src.api.internal_contributors as ic_mod
    from fastapi import HTTPException as FHTTPException

    tag = secrets.token_hex(4)
    auth_service = AuthService()
    user, user_session, _ = auth_service.register(
        email=f"api_dup_{tag}@example.com",
        password="Password123!",
        name=f"Dup User {tag}",
        role_id="user",
    )

    test_icr_repo = InternalContributorRepository()
    test_authz = AuthorizationService()
    monkeypatch.setattr(ic_mod, "icr_repo", test_icr_repo)
    monkeypatch.setattr(ic_mod, "authz_service", test_authz)

    body = SubmitInternalRequestModel(
        reason="First application for sensor data collection",
        experience="Graduate researcher in inertial navigation",
    )


    first = api_submit_internal_request(body=body, context=user_session)
    assert first["request"]["status"] == "pending"


    with pytest.raises(FHTTPException) as exc_info:
        api_submit_internal_request(body=body, context=user_session)
    assert exc_info.value.status_code == 409


def test_api_get_my_request_status(monkeypatch):
    """
    api_get_my_latest_request returns {"request": {...}} when a request exists,
    and {"request": None} when no request has been submitted.
    """
    import src.api.internal_contributors as ic_mod

    tag = secrets.token_hex(4)
    auth_service = AuthService()
    user_no_req, session_no_req, _ = auth_service.register(
        email=f"api_norq_{tag}@example.com",
        password="Password123!",
        name=f"No Request User {tag}",
        role_id="user",
    )
    user_with_req, session_with_req, _ = auth_service.register(
        email=f"api_withrq_{tag}@example.com",
        password="Password123!",
        name=f"With Request User {tag}",
        role_id="user",
    )

    test_icr_repo = InternalContributorRepository()
    test_authz = AuthorizationService()
    monkeypatch.setattr(ic_mod, "icr_repo", test_icr_repo)
    monkeypatch.setattr(ic_mod, "authz_service", test_authz)


    resp_none = api_get_my_latest_request(context=session_no_req)
    assert "request" in resp_none
    assert resp_none["request"] is None


    api_submit_internal_request(
        body=SubmitInternalRequestModel(
            reason="I want to contribute trajectory datasets for my MSc thesis",
            experience="Android developer with 2 years IMU sensor integration",
        ),
        context=session_with_req,
    )
    resp_found = api_get_my_latest_request(context=session_with_req)
    assert "request" in resp_found
    assert resp_found["request"] is not None
    assert resp_found["request"]["status"] == "pending"
    assert resp_found["request"]["user_id"] == user_with_req.id
