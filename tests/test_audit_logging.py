"""
Navigators IDR - Phase 17: Platform Audit Logging & Traceability Tests
Verifies immutable audit logging across Place CRUD, Contribution state transitions,
Dataset sessions, Model approvals & deployments, Internal contributor requests,
and Audit REST API permission gating.
"""

import tempfile
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.audit import AuditRepository
from src.db.places import PlaceRepository
from src.db.contributions import ContributionRepository
from src.db.datasets import DatasetRepository
from src.db.model_registry import ModelRegistryRepository
from src.db.internal_contributors import InternalContributorRepository
from src.db.auth_service import AuthService
from src.db.rbac import RBACRepository
from src.api.audit import list_audit_logs, get_audit_log_detail, get_audit_summary


@pytest.fixture
def temp_db():
    """Create an isolated temporary database for test execution."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


def test_place_crud_audit_logging(temp_db):
    """Test that place creation, update, soft delete, and restore write audit entries."""
    audit_repo = AuditRepository(temp_db)
    place_repo = PlaceRepository(temp_db)
    auth_service = AuthService(temp_db)

    # Seed actor user
    user, _, _ = auth_service.register(email="place_editor@example.com", password="Password123!", name="Place Editor")

    # 1. Create Place
    place = place_repo.create_place(
        name="Central Station",
        category="transportation",
        latitude=40.7128,
        longitude=-74.0060,
        created_by=user.id,
    )
    logs = audit_repo.list_logs(resource_type="place", resource_id=place.id)
    assert len(logs) == 1
    assert logs[0].action == "CREATE"
    assert logs[0].actor_id == user.id
    assert logs[0].new_state == "published"

    # 2. Update Place
    updated = place_repo.update_place(
        place_id=place.id,
        changed_by=user.id,
        name="Grand Central Station",
        change_summary="Renamed station",
    )
    logs = audit_repo.list_logs(resource_type="place", resource_id=place.id)
    assert len(logs) == 2
    assert logs[0].action == "UPDATE"
    assert logs[0].old_state == "v1"
    assert logs[0].new_state == "v2"

    # 3. Soft Delete Place
    archived = place_repo.soft_delete_place(place.id, changed_by=user.id, reason="Under renovation")
    logs = audit_repo.list_logs(resource_type="place", resource_id=place.id)
    assert len(logs) == 3
    assert logs[0].action == "SOFT_DELETE"
    assert logs[0].old_state == "published"
    assert logs[0].new_state == "archived"

    # 4. Restore Place
    restored = place_repo.restore_place(place.id, changed_by=user.id)
    logs = audit_repo.list_logs(resource_type="place", resource_id=place.id)
    assert len(logs) == 4
    assert logs[0].action == "RESTORE"
    assert logs[0].old_state == "archived"
    assert logs[0].new_state == "published"


def test_contribution_lifecycle_audit_logging(temp_db):
    """Test that contribution creation and state machine transitions write audit entries."""
    audit_repo = AuditRepository(temp_db)
    contrib_repo = ContributionRepository(temp_db)
    auth_service = AuthService(temp_db)

    author, _, _ = auth_service.register(email="contrib_author@example.com", password="Password123!", name="Contrib Author")
    reviewer, _, _ = auth_service.register(email="reviewer@example.com", password="Password123!", name="Reviewer User", role_id="moderator")

    # Create contribution
    contrib = contrib_repo.create_contribution(
        owner_id=author.id,
        resource_type="place",
        title="Add Local Park",
        data={"name": "Local Park", "category": "park", "latitude": 12.97, "longitude": 77.59},
    )

    create_logs = audit_repo.list_logs(resource_type="contribution", resource_id=contrib.id, action="CREATE")
    assert len(create_logs) == 1
    assert create_logs[0].actor_id == author.id

    # Submit for review
    contrib_repo.submit(contrib.id, user=author.id)
    submit_logs = audit_repo.list_logs(resource_type="contribution", resource_id=contrib.id, action="contribution:pending_review")
    assert len(submit_logs) == 1
    assert submit_logs[0].old_state == "draft"
    assert submit_logs[0].new_state == "pending_review"

    # Approve contribution
    contrib_repo.approve(contrib.id, reviewer=reviewer.id, notes="Looks accurate")
    approve_logs = audit_repo.list_logs(resource_type="contribution", resource_id=contrib.id, action="contribution:approved")
    assert len(approve_logs) == 1
    assert approve_logs[0].actor_id == reviewer.id


def test_dataset_session_audit_logging(temp_db):
    """Test that dataset submission and validation workflow write audit entries."""
    audit_repo = AuditRepository(temp_db)
    dataset_repo = DatasetRepository(temp_db)
    auth_service = AuthService(temp_db)

    contrib_user, _, _ = auth_service.register(email="sensor_driver@example.com", password="Password123!", name="Sensor Driver", role_id="internal_contributor")
    admin_user, _, _ = auth_service.register(email="dataset_admin@example.com", password="Password123!", name="Dataset Admin", role_id="team_admin")

    # Submit session
    session = dataset_repo.submit_session(
        contributor_id=contrib_user.id,
        activity_type="driving",
        device="iPhone14Pro",
        duration_seconds=120.0,
        consent=True,
    )
    submit_logs = audit_repo.list_logs(resource_type="dataset_session", resource_id=session.id, action="SUBMIT_DATASET_SESSION")
    assert len(submit_logs) == 1
    assert submit_logs[0].actor_id == contrib_user.id
    assert submit_logs[0].new_state == "uploaded"

    # Start validation
    dataset_repo.start_validation(session.id, validator_id=admin_user.id)
    val_logs = audit_repo.list_logs(resource_type="dataset_session", resource_id=session.id, action="START_DATASET_VALIDATION")
    assert len(val_logs) == 1
    assert val_logs[0].actor_id == admin_user.id

    # Validate session
    dataset_repo.validate_session(session.id, validator_id=admin_user.id)
    validated_logs = audit_repo.list_logs(resource_type="dataset_session", resource_id=session.id, action="VALIDATE_DATASET_SESSION")
    assert len(validated_logs) == 1
    assert validated_logs[0].new_state == "validated"


def test_model_approval_audit_logging(temp_db):
    """Test model registry candidate registration, review, approval, and deployment audit logs."""
    audit_repo = AuditRepository(temp_db)
    model_repo = ModelRegistryRepository(temp_db)
    auth_service = AuthService(temp_db)

    engineer, _, _ = auth_service.register(email="ml_eng@example.com", password="Password123!", name="ML Engineer", role_id="team_admin")

    # 1. Register candidate
    model = model_repo.register_candidate(
        name="TCN Experimental v2.1",
        registered_by=engineer.id,
        architecture="TCNVelocityEstimator",
    )
    reg_logs = audit_repo.list_logs(resource_type="model_registry", resource_id=model.id, action="REGISTER_MODEL_CANDIDATE")
    assert len(reg_logs) == 1

    # 2. Record evaluation
    model_repo.record_evaluation(model.id, {"candidate_test_mae": 3.85, "candidate_test_rmse": 5.42})
    eval_logs = audit_repo.list_logs(resource_type="model_registry", resource_id=model.id, action="RECORD_MODEL_EVALUATION")
    assert len(eval_logs) == 1

    # 3. Open review & Approve
    model_repo.open_for_review(model.id, reviewer_id=engineer.id)
    model_repo.approve_model(model.id, reviewer_id=engineer.id, notes="Significantly reduces error")
    appr_logs = audit_repo.list_logs(resource_type="model_registry", resource_id=model.id, action="APPROVE_MODEL")
    assert len(appr_logs) == 1
    assert appr_logs[0].actor_id == engineer.id


def test_internal_contributor_request_audit_logging(temp_db):
    """Test internal contributor access request and approval audit logging."""
    audit_repo = AuditRepository(temp_db)
    icr_repo = InternalContributorRepository(temp_db)
    auth_service = AuthService(temp_db)

    applicant, _, _ = auth_service.register(email="applicant@example.com", password="Password123!", name="Applicant User", role_id="user")
    admin, _, _ = auth_service.register(email="admin_approver@example.com", password="Password123!", name="Admin Approver", role_id="team_admin")

    # Submit application
    req = icr_repo.submit_request(
        user_id=applicant.id,
        reason="Recording IMU data for research",
        experience="2 years field testing",
    )
    sub_logs = audit_repo.list_logs(resource_type="internal_contributor_request", resource_id=req.id, action="SUBMIT_INTERNAL_ACCESS_REQUEST")
    assert len(sub_logs) == 1

    # Approve application
    icr_repo.approve_request(req.id, reviewer_id=admin.id)
    appr_logs = audit_repo.list_logs(resource_type="internal_contributor_request", resource_id=req.id, action="APPROVE_INTERNAL_ACCESS")
    assert len(appr_logs) == 1
    assert appr_logs[0].actor_id == admin.id
    assert appr_logs[0].new_state == "approved"


def test_audit_api_permission_gating_and_querying(temp_db):
    """Test that audit REST API routes enforce permission gating and return expected summary stats."""
    audit_repo = AuditRepository(temp_db)
    auth_service = AuthService(temp_db)

    # 1. Unauthenticated guest request -> 401
    guest_session, _ = auth_service.create_session(user_id=None, is_guest=True)
    with pytest.raises(HTTPException) as exc_info:
        list_audit_logs(session=guest_session)
    assert exc_info.value.status_code == 401

    # 2. Registered user without audit:read -> 403
    norm_user, norm_session, _ = auth_service.register(email="normal_user@example.com", password="Password123!", name="Normal User", role_id="user")
    with pytest.raises(HTTPException) as exc_info:
        list_audit_logs(session=norm_session)
    assert exc_info.value.status_code == 403

    # 3. Team admin with audit:read -> 200 OK
    admin_user, admin_session, _ = auth_service.register(email="team_admin_audit@example.com", password="Password123!", name="Admin User", role_id="team_admin")

    # Log dummy entry
    audit_repo.log(
        action="TEST_ACTION",
        resource_type="test_res",
        resource_id="res_123",
        actor_id=admin_user.id,
    )

    res = list_audit_logs(session=admin_session)
    assert "items" in res
    assert res["total"] >= 1

    summary = get_audit_summary(session=admin_session)
    assert "total_logs" in summary
    assert "by_resource_type" in summary
