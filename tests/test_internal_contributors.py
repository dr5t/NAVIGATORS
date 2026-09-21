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

    # Submit request
    req = icr_repo.submit_request(
        user_id=user.id,
        reason="I want to contribute driving session datasets",
        experience="Automotive engineer with MEMS sensor experience",
    )

    # Verify user does NOT have internal_contributor role yet
    roles_before = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles_before

    # Approve
    approved = icr_repo.approve_request(request_id=req.id, reviewer_id=admin.id)

    assert approved.status == "approved"
    assert approved.reviewed_by == admin.id
    assert approved.reviewed_at is not None

    # Verify user NOW has internal_contributor role
    roles_after = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" in roles_after

    # Verify audit log
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

    # Verify user does NOT have internal_contributor role
    roles = {r.id for r in rbac_repo.get_user_roles(user.id)}
    assert "internal_contributor" not in roles

    # Verify audit log
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

    # Try list_requests via API (requires internal_contributor:review)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        api_list_internal_requests(context=user_session)
    assert exc_info.value.status_code == 403

    # Try approve via API (requires internal_contributor:approve)
    with pytest.raises(HTTPException) as exc_info:
        api_approve_internal_request(request_id=req.id, context=user_session)
    assert exc_info.value.status_code == 403

    # Try reject via API (requires internal_contributor:reject)
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

    # Normal user cannot create training jobs
    decision_train = authz.can(user=user_context, action="training:create", resource="training")
    assert not decision_train.allowed

    # Normal user cannot create datasets
    decision_dataset = authz.can(user=user_context, action="dataset:create", resource="dataset")
    assert not decision_dataset.allowed

    # Normal user cannot deploy models
    decision_deploy = authz.can(user=user_context, action="model:deploy", resource="model")
    assert not decision_deploy.allowed

    # But normal user CAN request internal contributor access
    decision_request = authz.can(user=user_context, action="internal_contributor:request", resource="contributor")
    assert decision_request.allowed
