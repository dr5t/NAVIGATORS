"""
Navigators IDR - Phase 15 Model Registry & Governance Tests
Tests the model registry repository state machine, atomic deployment,
RBAC enforcement, and API router endpoints.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.db.database import get_db, init_db
from src.db.model_registry import (
    ModelRegistryRepository,
    ModelEntry,
    VALID_TRANSITIONS,
)
from src.db.auth_service import AuthService
from src.db.authorization import AuthorizationService
from src.api.server import app

client = TestClient(app)


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
def auth_setup(temp_db):
    """Helper fixture to create test users with different roles and valid session tokens."""
    auth_service = AuthService(temp_db)

    # Register admin user
    admin_res = auth_service.register_user(
        name="Admin User",
        email="admin@navigators.test",
        password="AdminPassword123!",
        role="team_admin"
    )

    # Register regular user
    user_res = auth_service.register_user(
        name="Regular User",
        email="user@navigators.test",
        password="UserPassword123!",
        role="user"
    )

    return {
        "admin": admin_res,
        "user": user_res,
        "auth_service": auth_service,
    }


# =============================================================================
# Unit Tests: ModelRegistryRepository State Machine & Operations
# =============================================================================

def test_seed_production_model(repo):
    """Verify idempotent seeding of the verified production baseline model."""
    prod = repo.seed_production_model()
    assert prod is not None
    assert prod.status == "production"
    assert prod.name == "TCN v1.0 — Verified Production (4.2039 m/s)"
    assert prod.test_mae == 4.2039

    # Second call should be idempotent and return identical entry
    prod2 = repo.seed_production_model()
    assert prod2.id == prod.id


def test_register_candidate(repo):
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


def test_state_machine_happy_path(repo, tmp_path):
    """Verify full state machine progression: candidate_training -> evaluating -> review -> approved/production_candidate -> production."""
    # 1. Register candidate
    cand = repo.register_candidate(name="TCN Candidate Happy")
    model_id = cand.id
    assert cand.status == "candidate_training"

    # 2. Record evaluation
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

    # 3. Open for review
    rev_entry = repo.open_for_review(model_id, reviewer_id="usr_admin")
    assert rev_entry.status == "review"
    assert rev_entry.reviewed_by == "usr_admin"

    # 4. Approve model (moves to production_candidate)
    appr_entry = repo.approve_model(model_id, reviewer_id="usr_admin", notes="Looks great!")
    assert appr_entry.status == "production_candidate"

    # 5. Deploy model
    # Create fake source artifact files
    pt_file = tmp_path / "best_model.pt"
    onnx_file = tmp_path / "model.onnx"
    stats_file = tmp_path / "norm_stats.json"
    pt_file.write_text("dummy_pt_content")
    onnx_file.write_text("dummy_onnx_content")
    stats_file.write_text("{}")

    # Set checkpoint paths in DB
    with get_db(repo.db_path) as conn:
        conn.execute(
            "UPDATE model_registry SET checkpoint_path=?, onnx_path=?, norm_stats_path=? WHERE id=?",
            (str(pt_file), str(onnx_file), str(stats_file), model_id)
        )

    # Destination paths for testing deploy
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


def test_illegal_state_transitions(repo):
    """Verify that invalid state transitions raise ValueError."""
    cand = repo.register_candidate(name="TCN Candidate Invalid")
    model_id = cand.id

    # Trying to jump candidate_training -> approved should fail
    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.approve_model(model_id, reviewer_id="usr_admin")

    # Move to evaluating
    repo.record_evaluation(model_id, {"candidate_test_mae": 4.0})

    # Trying to jump evaluating -> production should fail
    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.deploy(model_id, deployer_id="usr_admin")


def test_rejection_flow(repo):
    """Verify review -> rejected transition requires reason."""
    cand = repo.register_candidate(name="TCN Bad Model")
    model_id = cand.id
    repo.record_evaluation(model_id, {"candidate_test_mae": 15.0})
    repo.open_for_review(model_id, reviewer_id="usr_admin")

    # Empty rejection reason must fail
    with pytest.raises(ValueError, match="Rejection reason cannot be empty"):
        repo.reject_model(model_id, reviewer_id="usr_admin", reason="   ")

    # Valid rejection
    rej_entry = repo.reject_model(model_id, reviewer_id="usr_admin", reason="High MAE degradation")
    assert rej_entry.status == "rejected"
    assert rej_entry.rejection_reason == "High MAE degradation"

    # Terminal state check: cannot transition from rejected
    with pytest.raises(ValueError, match="terminal state"):
        repo.open_for_review(model_id, reviewer_id="usr_admin")


# =============================================================================
# API Endpoint Integration Tests
# =============================================================================

def test_api_list_and_get_models(temp_db):
    """Test GET /api/v1/models and GET /api/v1/models/production."""
    repo = ModelRegistryRepository(temp_db)
    repo.seed_production_model()

    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["total"] >= 1

    prod_res = client.get("/api/v1/models/production")
    assert prod_res.status_code == 200
    prod_data = prod_res.json()
    assert prod_data["model"] is not None
    assert prod_data["model"]["status"] == "production"


def test_api_rbac_governance(temp_db, auth_setup):
    """Test full API governance lifecycle with admin token vs unauthenticated/user token."""
    repo = ModelRegistryRepository(temp_db)
    admin_token = auth_setup["admin"]["token"]
    user_token = auth_setup["user"]["token"]

    # Register candidate
    cand = repo.register_candidate(name="TCN API Test Candidate")
    model_id = cand.id
    repo.record_evaluation(model_id, {"candidate_test_mae": 3.9})

    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    user_headers = {"Authorization": f"Bearer {user_token}"}

    # 1. Open review as regular user (Denied)
    r1 = client.post(f"/api/v1/models/{model_id}/open-review", headers=user_headers)
    assert r1.status_code == 403

    # Open review as admin (Allowed)
    r2 = client.post(f"/api/v1/models/{model_id}/open-review", headers=admin_headers)
    assert r2.status_code == 200
    assert r2.json()["model"]["status"] == "review"

    # 2. Approve as regular user (Denied)
    r3 = client.post(f"/api/v1/models/{model_id}/approve", headers=user_headers, json={"notes": "looks fine"})
    assert r3.status_code == 403

    # Approve as admin (Allowed)
    r4 = client.post(f"/api/v1/models/{model_id}/approve", headers=admin_headers, json={"notes": "looks fine"})
    assert r4.status_code == 200
    assert r4.json()["model"]["status"] == "production_candidate"
