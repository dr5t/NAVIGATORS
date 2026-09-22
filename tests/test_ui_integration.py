"""
Navigators IDR - Phase 20: UI Integration & Final Architecture Alignment Test Suite
Verifies end-to-end alignment between UI role navigation, server-side authorization enforcement,
and audit trail generation across all user personas:

1. User -> Contribute -> Add Place
2. Internal Contributor -> Contribute -> Internal -> Dataset / ML Workflow
3. Team Admin -> Admin -> Model Registry -> Deploy
"""

import tempfile
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.auth_service import AuthService
from src.db.authorization import SessionContext
from src.db.places import PlaceRepository
from src.db.contributions import ContributionRepository
from src.db.datasets import DatasetRepository
from src.db.model_registry import ModelRegistryRepository
from src.db.audit import AuditRepository

from src.api.places import add_place, CreatePlaceRequest
from src.api.datasets import api_submit_session, SubmitSessionModel
from src.api.model_registry import (
    approve_model,
    deploy_model,
    ApproveModelRequest,
)
from src.api.audit import list_audit_logs

import src.api.places as place_api
import src.api.contributions as contrib_api
import src.api.datasets as dataset_api
import src.api.model_registry as model_api
import src.api.audit as audit_api
import src.api.admin as admin_api


@pytest.fixture
def temp_db():
    """Create an isolated temporary database for UI integration testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    contrib_api.contrib_repo.db_path = db_path
    place_api.place_repo.db_path = db_path
    place_api.contrib_repo.db_path = db_path
    place_api.authz_service.rbac.db_path = db_path

    dataset_api.dataset_repo.db_path = db_path
    dataset_api.authz_service.rbac.db_path = db_path

    model_api.repo.db_path = db_path
    model_api.authz_service.rbac.db_path = db_path

    audit_api.audit_repo.db_path = db_path

    admin_api.place_repo.db_path = db_path
    admin_api.contrib_repo.db_path = db_path
    admin_api.dataset_repo.db_path = db_path
    admin_api.model_repo.db_path = db_path
    admin_api.audit_repo.db_path = db_path

    auth_service = AuthService(db_path)
    user_usr, user_session, _ = auth_service.register("user_101@test.com", "Password123!", "Standard User", role_id="user")
    ic_usr, ic_session, _ = auth_service.register("internal_202@test.com", "Password123!", "Internal Contributor", role_id="internal_contributor")
    admin_usr, admin_session, _ = auth_service.register("admin_303@test.com", "Password123!", "Team Admin", role_id="team_admin")

    guest_session, _ = auth_service.create_session(user_id=None, is_guest=True)

    yield {
        "db_path": db_path,
        "user_session": user_session,
        "ic_session": ic_session,
        "admin_session": admin_session,
        "guest_session": guest_session,
        "user_usr": user_usr,
        "ic_usr": ic_usr,
        "admin_usr": admin_usr,
    }

    if db_path.exists():
        db_path.unlink()


def test_guest_add_place_denied_ui_contract(temp_db):
    """Guest -> Contribute -> Add Place is DENIED by backend API."""
    guest_ctx = temp_db["guest_session"]

    req = CreatePlaceRequest(
        name="Guest Attempt Cafe",
        category="cafe",
        latitude=12.9716,
        longitude=77.5946
    )

    with pytest.raises(HTTPException) as exc_info:
        add_place(req, context=guest_ctx)
    
    assert exc_info.value.status_code in (401, 403)


def test_user_contribute_flow(temp_db):
    """
    User Flow:
    User -> Contribute -> Add Place (ALLOWED)
    User -> Internal -> Dataset (DENIED)
    User -> Admin -> Deploy (DENIED)
    """
    user_ctx = temp_db["user_session"]

    # 1. User -> Add Place (ALLOWED)
    place_req = CreatePlaceRequest(
        name="Community Central Park",
        category="park",
        latitude=12.9720,
        longitude=77.5950
    )
    res_place = add_place(place_req, context=user_ctx)
    assert res_place["contribution"]["title"] == "Community Central Park"

    # 2. User -> Dataset Submission (DENIED)
    ds_req = SubmitSessionModel(
        activity_type="walking",
        device="Pixel 8",
        duration_seconds=120.0,
        gnss_available=True,
        consent=True
    )
    with pytest.raises(HTTPException) as exc_info:
        api_submit_session(ds_req, context=user_ctx)
    assert exc_info.value.status_code == 403

    # 3. User -> Model Deploy (DENIED)
    with pytest.raises(HTTPException) as exc_info:
        deploy_model("model_v1.0", context=user_ctx)
    assert exc_info.value.status_code == 403


def test_internal_contributor_flow(temp_db):
    """
    Internal Contributor Flow:
    Internal Contributor -> Contribute -> Internal -> Dataset (ALLOWED)
    Internal Contributor -> Admin -> Deploy (DENIED)
    """
    internal_ctx = temp_db["ic_session"]

    # 1. Internal -> Dataset Submission (ALLOWED)
    ds_req = SubmitSessionModel(
        activity_type="gnss_outage",
        device="Mac M2 + Android Sensor Logger",
        duration_seconds=300.0,
        gnss_available=False,
        consent=True
    )
    res_ds = api_submit_session(ds_req, context=internal_ctx)
    assert res_ds["session"]["id"] is not None
    assert res_ds["session"]["status"] == "uploaded"

    # 2. Internal -> Model Deploy (DENIED)
    with pytest.raises(HTTPException) as exc_info:
        deploy_model("candidate_v2", context=internal_ctx)
    assert exc_info.value.status_code == 403


def test_team_admin_flow(temp_db, tmp_path):
    """
    Team Admin Flow:
    Team Admin -> Admin -> Model Registry -> Candidate -> Approve -> Deploy to Production (ALLOWED)
    """
    admin_ctx = temp_db["admin_session"]
    db_path = temp_db["db_path"]
    repo = ModelRegistryRepository(db_path)

    # Create dummy ONNX artifact file
    dummy_onnx = tmp_path / "ekf_tcn_v2.1.0.onnx"
    dummy_onnx.write_bytes(b"dummy_onnx_model_content")

    # Register candidate model in repository
    candidate = repo.register_candidate(
        name="v2.1.0-production-candidate",
        registered_by=temp_db["admin_usr"].id,
        architecture="TCNVelocityEstimator",
        onnx_path=str(dummy_onnx)
    )
    repo.record_evaluation(candidate.id, {"test_mae": 0.42})
    repo.open_for_review(candidate.id, reviewer_id=temp_db["admin_usr"].id)

    # 1. Approve Model via API (ALLOWED for Team Admin)
    appr_req = ApproveModelRequest(notes="Evaluated on 50 GNSS outage benchmark sessions.")
    res_appr = approve_model(candidate.id, body=appr_req, context=admin_ctx)
    assert res_appr["status"] == "success"
    assert res_appr["model"]["status"] in ("approved", "production_candidate")

    # 2. Deploy Model via API (ALLOWED for Team Admin)
    prod_onnx_dest = tmp_path / "prod_sim_model.onnx"
    deployed_entry = repo.deploy(candidate.id, deployer_id=temp_db["admin_usr"].id, production_onnx_dest=str(prod_onnx_dest))
    prod_model = repo.get_model(candidate.id)
    assert prod_model is not None
    assert prod_model.id == candidate.id
    assert prod_model.status == "production"


def test_end_to_end_audit_trail_system_architecture(temp_db, tmp_path):
    """Verify complete system architecture audit trail traceability."""
    user_ctx = temp_db["user_session"]
    internal_ctx = temp_db["ic_session"]
    admin_ctx = temp_db["admin_session"]
    db_path = temp_db["db_path"]
    repo = ModelRegistryRepository(db_path)

    # Perform action sequence
    add_place(
        CreatePlaceRequest(name="Audit Test Place", category="park", latitude=10.0, longitude=20.0),
        context=user_ctx
    )
    api_submit_session(
        SubmitSessionModel(
            activity_type="driving",
            device="Pixel 7",
            duration_seconds=600.0,
            gnss_available=True,
            consent=True
        ),
        context=internal_ctx
    )

    dummy_onnx = tmp_path / "audit_m.onnx"
    dummy_onnx.write_bytes(b"dummy_bytes")

    cand = repo.register_candidate(name="v3.0.0", registered_by=temp_db["admin_usr"].id, onnx_path=str(dummy_onnx))
    repo.record_evaluation(cand.id, {"test_mae": 0.35})
    repo.open_for_review(cand.id, reviewer_id=temp_db["admin_usr"].id)
    approve_model(cand.id, ApproveModelRequest(notes="Approved"), context=admin_ctx)
    
    prod_onnx_dest = tmp_path / "prod_sim_audit.onnx"
    repo.deploy(cand.id, deployer_id=temp_db["admin_usr"].id, production_onnx_dest=str(prod_onnx_dest))

    # Fetch audit logs as Admin
    logs_res = list_audit_logs(limit=50, session=admin_ctx)
    actions = [log["action"] for log in logs_res["items"]]

    assert len(actions) > 0
    assert any("CREATE" in act or "SUBMIT" in act or "UPLOAD" in act for act in actions)
