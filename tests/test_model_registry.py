"""
Navigators IDR - Phase 15 Model Registry & Governance Tests
Tests the model registry repository state machine, atomic deployment,
RBAC enforcement, and API router functions.
"""

import os
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import get_db, init_db
from src.db.rbac import RBACRepository
from src.db.model_registry import (
    ModelRegistryRepository,
    ModelEntry,
    VALID_TRANSITIONS,
)
from src.db.auth_service import AuthService, SessionContext
from src.api.model_registry import (
    list_models as api_list_models,
    get_production_model as api_get_production_model,
    get_model as api_get_model,
    open_model_review as api_open_model_review,
    approve_model as api_approve_model,
    reject_model as api_reject_model,
    deploy_model as api_deploy_model,
    ApproveModelRequest,
    RejectModelRequest,
    repo as global_repo,
)


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a clean database for testing."""
    db_file = tmp_path / "test_model_registry.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def repo(temp_db):
    """Fixture providing ModelRegistryRepository connected to temp_db."""
    return ModelRegistryRepository(temp_db)


@pytest.fixture
def test_users(temp_db):
    """Create test users in database so foreign key constraints on users(id) pass."""
    rbac = RBACRepository(temp_db)
    admin_user = rbac.create_user(
        user_id="usr_admin",
        email="admin@navigators.test",
        name="Admin User",
        status="active",
        initial_role_ids=["team_admin"]
    )
    reg_user = rbac.create_user(
        user_id="usr_test123",
        email="test@navigators.test",
        name="Regular User",
        status="active",
        initial_role_ids=["user"]
    )
    return {"admin": admin_user, "user": reg_user}


@pytest.fixture
def auth_setup(temp_db):
    """Helper fixture to create test users with valid session contexts."""
    auth_service = AuthService(temp_db)

    admin_user, admin_ctx, _ = auth_service.register(
        email="admin_ctx@navigators.test",
        password="AdminPassword123!",
        name="Admin Context User",
        role_id="team_admin"
    )

    reg_user, user_ctx, _ = auth_service.register(
        email="user_ctx@navigators.test",
        password="UserPassword123!",
        name="User Context User",
        role_id="user"
    )

    return {
        "admin_ctx": admin_ctx,
        "user_ctx": user_ctx,
        "auth_service": auth_service,
    }






def test_seed_production_model(repo):
    """Verify idempotent seeding of the verified production baseline model."""
    prod = repo.seed_production_model()
    assert prod is not None
    assert prod.status == "production"
    assert prod.name == "TCN v1.0 - Verified Production (4.2039 m/s)"
    assert prod.test_mae == 4.2039


    prod2 = repo.seed_production_model()
    assert prod2.id == prod.id


def test_register_candidate(repo, test_users):
    """Verify registering a new model candidate in candidate_training status."""
    cand = repo.register_candidate(
        name="TCN Candidate v2",
        architecture="TCNVelocityEstimator",
        registered_by="usr_test123"
    )
    assert cand is not None
    assert cand.status == "candidate_training"
    assert cand.name == "TCN Candidate v2"
    assert cand.registered_by == "usr_test123"


def test_state_machine_happy_path(repo, test_users, tmp_path):
    """Verify full state machine progression: candidate_training -> evaluating -> review -> approved/production_candidate -> production."""

    cand = repo.register_candidate(name="TCN Candidate Happy", registered_by="usr_admin")
    model_id = cand.id
    assert cand.status == "candidate_training"


    metrics = {
        "candidate_test_mae": 3.85,
        "candidate_test_rmse": 5.21,
        "best_val_loss": 0.012,
        "onnx_parity_max_diff": 1e-7,
        "error_reduction_pct": 12.5,
        "total_epochs": 40,
        "batch_size": 256,
        "window_size": 200,
        "total_time_s": 120.5,
    }
    eval_entry = repo.record_evaluation(model_id, metrics)
    assert eval_entry.status == "evaluating"
    assert eval_entry.test_mae == 3.85


    rev_entry = repo.open_for_review(model_id, reviewer_id="usr_admin")
    assert rev_entry.status == "review"
    assert rev_entry.reviewed_by == "usr_admin"


    appr_entry = repo.approve_model(model_id, reviewer_id="usr_admin", notes="Looks great!")
    assert appr_entry.status == "production_candidate"


    pt_file = tmp_path / "best_model.pt"
    onnx_file = tmp_path / "model.onnx"
    stats_file = tmp_path / "norm_stats.json"
    pt_file.write_text("dummy_pt_content")
    onnx_file.write_text("dummy_onnx_content")
    stats_file.write_text("{}")


    with get_db(repo.db_path) as conn:
        conn.execute(
            "UPDATE model_registry SET checkpoint_path=?, onnx_path=?, norm_stats_path=? WHERE id=?",
            (str(pt_file), str(onnx_file), str(stats_file), model_id)
        )


    dest_pt = tmp_path / "prod" / "best_model.pt"
    dest_onnx = tmp_path / "prod" / "model.onnx"
    dest_stats = tmp_path / "prod" / "norm_stats.json"
    dest_sim_stats = tmp_path / "prod" / "sim_norm_stats.json"

    prod_entry = repo.deploy(
        model_id,
        deployer_id="usr_admin",
        production_checkpoint_dest=str(dest_pt),
        production_onnx_dest=str(dest_onnx),
        production_stats_dest=str(dest_stats),
        production_sim_stats_dest=str(dest_sim_stats),
    )
    assert prod_entry.status == "production"
    assert dest_pt.read_text() == "dummy_pt_content"
    assert dest_onnx.read_text() == "dummy_onnx_content"


def test_illegal_state_transitions(repo, test_users):
    """Verify that invalid state transitions raise ValueError."""
    cand = repo.register_candidate(name="TCN Candidate Invalid")
    model_id = cand.id


    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.approve_model(model_id, reviewer_id="usr_admin")


    repo.record_evaluation(model_id, {"candidate_test_mae": 4.0})


    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.deploy(model_id, deployer_id="usr_admin")


def test_rejection_flow(repo, test_users):
    """Verify review -> rejected transition requires reason."""
    cand = repo.register_candidate(name="TCN Bad Model")
    model_id = cand.id
    repo.record_evaluation(model_id, {"candidate_test_mae": 15.0})
    repo.open_for_review(model_id, reviewer_id="usr_admin")


    with pytest.raises(ValueError, match="Rejection reason cannot be empty"):
        repo.reject_model(model_id, reviewer_id="usr_admin", reason="   ")


    rej_entry = repo.reject_model(model_id, reviewer_id="usr_admin", reason="High MAE degradation")
    assert rej_entry.status == "rejected"
    assert rej_entry.rejection_reason == "High MAE degradation"


    with pytest.raises(ValueError, match="terminal state"):
        repo.open_for_review(model_id, reviewer_id="usr_admin")






def test_api_list_and_get_models(temp_db, monkeypatch):
    """Test API list and get functions with default DB path."""
    repo = ModelRegistryRepository(temp_db)
    repo.seed_production_model()
    monkeypatch.setattr(global_repo, "db_path", temp_db)

    list_res = api_list_models(status=None, limit=50, offset=0)
    assert "models" in list_res
    assert list_res["total"] >= 1

    prod_res = api_get_production_model()
    assert prod_res["model"] is not None
    assert prod_res["model"]["status"] == "production"

    single_res = api_get_model(prod_res["model"]["id"])
    assert single_res["model"]["id"] == prod_res["model"]["id"]


def test_api_rbac_governance(temp_db, auth_setup, monkeypatch):
    """Test full API governance lifecycle with admin session vs user/unauthenticated sessions."""
    repo = ModelRegistryRepository(temp_db)
    monkeypatch.setattr(global_repo, "db_path", temp_db)

    admin_ctx = auth_setup["admin_ctx"]
    user_ctx = auth_setup["user_ctx"]


    cand = repo.register_candidate(name="TCN API Test Candidate")
    model_id = cand.id
    repo.record_evaluation(model_id, {"candidate_test_mae": 3.9})


    with pytest.raises(HTTPException) as exc_info:
        api_open_model_review(model_id, context=None)
    assert exc_info.value.status_code == 401


    with pytest.raises(HTTPException) as exc_info:
        api_open_model_review(model_id, context=user_ctx)
    assert exc_info.value.status_code == 403


    res_open = api_open_model_review(model_id, context=admin_ctx)
    assert res_open["status"] == "success"
    assert res_open["model"]["status"] == "review"


    with pytest.raises(HTTPException) as exc_info:
        api_approve_model(model_id, body=ApproveModelRequest(notes="fine"), context=user_ctx)
    assert exc_info.value.status_code == 403


    res_appr = api_approve_model(model_id, body=ApproveModelRequest(notes="Approved by admin"), context=admin_ctx)
    assert res_appr["status"] == "success"
    assert res_appr["model"]["status"] == "production_candidate"
