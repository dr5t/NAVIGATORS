"""
Navigators IDR - Phase 19: Security Testing & Comprehensive Permission Matrix Suite
Verifies backend authorization enforcement across all roles (Guest, User, Local Contributor,
Internal Contributor, Moderator, Team Admin, Super Admin) and direct API route enforcement.
Confirms that the API itself rejects unauthorized requests (HTTP 401 / HTTP 403)
without relying on UI button hiding.
"""

import tempfile
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService
from src.db.authorization import AuthorizationService
from src.db.places import PlaceRepository
from src.db.contributions import ContributionRepository
from src.db.datasets import DatasetRepository
from src.db.model_registry import ModelRegistryRepository
from src.db.internal_contributors import InternalContributorRepository

from src.api.places import add_place, direct_update_place, CreatePlaceRequest
from src.api.contributions import approve_contribution, review_contribution, ReviewContributionRequest
from src.api.datasets import api_validate_session
from src.api.model_registry import deploy_model
from src.api.audit import list_audit_logs
from src.api.internal_contributors import api_approve_internal_request
import src.api.contributions as contrib_api
import src.api.places as place_api
import src.api.datasets as dataset_api
import src.api.model_registry as model_api
import src.api.internal_contributors as icr_api
import src.api.reports as report_api
import src.api.audit as audit_api
import src.api.admin as admin_api


@pytest.fixture
def temp_db():
    """Create an isolated temporary database for security matrix testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    contrib_api.contrib_repo.db_path = db_path
    place_api.place_repo.db_path = db_path
    dataset_api.dataset_repo.db_path = db_path
    model_api.repo.db_path = db_path
    icr_api.icr_repo.db_path = db_path
    report_api.report_repo.db_path = db_path
    audit_api.audit_repo.db_path = db_path
    admin_api.place_repo.db_path = db_path
    admin_api.contrib_repo.db_path = db_path
    admin_api.dataset_repo.db_path = db_path
    admin_api.model_repo.db_path = db_path
    admin_api.audit_repo.db_path = db_path
    admin_api.report_repo.db_path = db_path
    admin_api.rbac_repo.db_path = db_path

    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def role_sessions(temp_db):
    """Fixture initializing accounts and active sessions for every role."""
    auth_service = AuthService(temp_db)

    guest_session, _ = auth_service.create_session(user_id=None, is_guest=True)
    user_usr, user_session, _ = auth_service.register("user_sec@test.com", "Password123!", "User Sec", role_id="user")
    lc_usr, lc_session, _ = auth_service.register("lc_sec@test.com", "Password123!", "LC Sec", role_id="local_contributor")
    ic_usr, ic_session, _ = auth_service.register("ic_sec@test.com", "Password123!", "IC Sec", role_id="internal_contributor")
    mod_usr, mod_session, _ = auth_service.register("mod_sec@test.com", "Password123!", "Mod Sec", role_id="moderator")
    admin_usr, admin_session, _ = auth_service.register("admin_sec@test.com", "Password123!", "Admin Sec", role_id="team_admin")
    super_usr, super_session, _ = auth_service.register("super_sec@test.com", "Password123!", "Super Sec", role_id="super_admin")

    return {
        "guest": (None, guest_session),
        "user": (user_usr, user_session),
        "local_contributor": (lc_usr, lc_session),
        "internal_contributor": (ic_usr, ic_session),
        "moderator": (mod_usr, mod_session),
        "team_admin": (admin_usr, admin_session),
        "super_admin": (super_usr, super_session),
    }






def test_authz_permission_matrix(temp_db, role_sessions):
    """Test explicit ALLOW / DENY permissions matrix across all roles."""
    authz = AuthorizationService(temp_db)


    g_ctx = role_sessions["guest"][1]
    assert authz.can(g_ctx, "place:read").allowed is True
    assert authz.can(g_ctx, "contribution:create").allowed is False
    assert authz.can(g_ctx, "contribution:approve").allowed is False
    assert authz.can(g_ctx, "dataset:validate").allowed is False
    assert authz.can(g_ctx, "model:deploy").allowed is False
    assert authz.can(g_ctx, "audit:read").allowed is False


    u_ctx = role_sessions["user"][1]
    assert authz.can(u_ctx, "place:read").allowed is True
    assert authz.can(u_ctx, "contribution:create").allowed is True
    assert authz.can(u_ctx, "contribution:approve").allowed is False
    assert authz.can(u_ctx, "dataset:validate").allowed is False
    assert authz.can(u_ctx, "model:deploy").allowed is False
    assert authz.can(u_ctx, "audit:read").allowed is False


    lc_ctx = role_sessions["local_contributor"][1]
    assert authz.can(lc_ctx, "place:create").allowed is True
    assert authz.can(lc_ctx, "contribution:approve").allowed is False
    assert authz.can(lc_ctx, "dataset:validate").allowed is False
    assert authz.can(lc_ctx, "model:deploy").allowed is False


    ic_ctx = role_sessions["internal_contributor"][1]
    assert authz.can(ic_ctx, "dataset:create").allowed is True
    assert authz.can(ic_ctx, "training:create").allowed is True
    assert authz.can(ic_ctx, "model:create").allowed is True
    assert authz.can(ic_ctx, "dataset:validate").allowed is False
    assert authz.can(ic_ctx, "model:deploy").allowed is False
    assert authz.can(ic_ctx, "audit:read").allowed is False


    m_ctx = role_sessions["moderator"][1]
    assert authz.can(m_ctx, "contribution:approve").allowed is True
    assert authz.can(m_ctx, "report:resolve").allowed is True
    assert authz.can(m_ctx, "audit:read").allowed is True
    assert authz.can(m_ctx, "dataset:validate").allowed is False
    assert authz.can(m_ctx, "model:deploy").allowed is False


    ta_ctx = role_sessions["team_admin"][1]
    assert authz.can(ta_ctx, "contribution:approve").allowed is True
    assert authz.can(ta_ctx, "dataset:validate").allowed is True
    assert authz.can(ta_ctx, "model:approve").allowed is True
    assert authz.can(ta_ctx, "model:deploy").allowed is True
    assert authz.can(ta_ctx, "audit:read").allowed is True
    assert authz.can(ta_ctx, "role:assign").allowed is True


    sa_ctx = role_sessions["super_admin"][1]
    assert authz.can(sa_ctx, "model:deploy").allowed is True
    assert authz.can(sa_ctx, "dataset:validate").allowed is True






def test_api_rejection_guest_place_creation(role_sessions):
    """Guest -> add place endpoint raises HTTP 401 Unauthorized."""
    guest_ctx = role_sessions["guest"][1]
    req = CreatePlaceRequest(name="Forbidden POI", category="fuel", latitude=12.0, longitude=77.0)

    with pytest.raises(HTTPException) as exc_info:
        add_place(req=req, context=guest_ctx)
    assert exc_info.value.status_code == 401


def test_api_rejection_user_contribution_approval(temp_db, role_sessions):
    """User / Contributor -> approve contribution endpoint raises HTTP 403 Forbidden."""
    contrib_repo = ContributionRepository(temp_db)
    author_usr = role_sessions["user"][0]
    user_ctx = role_sessions["user"][1]
    mod_ctx = role_sessions["moderator"][1]

    contrib = contrib_repo.create_contribution(
        owner_id=author_usr.id,
        resource_type="place",
        title="Test Place",
        data={"name": "Test Place", "category": "park", "latitude": 12.0, "longitude": 77.0},
        status="pending_review",
    )


    with pytest.raises(HTTPException) as exc_info:
        approve_contribution(contrib_id=contrib.id, context=user_ctx)
    assert exc_info.value.status_code == 403


    lc_ctx = role_sessions["local_contributor"][1]
    with pytest.raises(HTTPException) as exc_info:
        review_contribution(contrib_id=contrib.id, req=ReviewContributionRequest(decision="approved"), context=lc_ctx)
    assert exc_info.value.status_code == 403


    res = review_contribution(contrib_id=contrib.id, req=ReviewContributionRequest(decision="approved"), context=mod_ctx)
    assert res["contribution"]["status"] == "approved"


def test_api_rejection_internal_contributor_dataset_validation(temp_db, role_sessions):
    """Internal Contributor -> dataset validation endpoint raises HTTP 403 Forbidden."""
    dataset_repo = DatasetRepository(temp_db)
    ic_usr = role_sessions["internal_contributor"][0]
    ic_ctx = role_sessions["internal_contributor"][1]
    admin_ctx = role_sessions["team_admin"][1]

    session = dataset_repo.submit_session(
        contributor_id=ic_usr.id,
        activity_type="driving",
        device="iPhone14Pro",
        duration_seconds=60.0,
        consent=True,
    )


    with pytest.raises(HTTPException) as exc_info:
        api_validate_session(session_id=session.id, context=ic_ctx)
    assert exc_info.value.status_code == 403


    dataset_repo.start_validation(session.id, validator_id=role_sessions["team_admin"][0].id)
    res = api_validate_session(session_id=session.id, context=admin_ctx)
    assert res["session"]["status"] == "validated"


def test_api_rejection_internal_contributor_model_deployment(temp_db, role_sessions):
    """Internal Contributor / Moderator -> deploy model endpoint raises HTTP 403 Forbidden."""
    model_repo = ModelRegistryRepository(temp_db)
    ic_ctx = role_sessions["internal_contributor"][1]
    mod_ctx = role_sessions["moderator"][1]
    admin_ctx = role_sessions["team_admin"][1]

    model = model_repo.register_candidate(name="TCN Security Test", architecture="TCNVelocityEstimator")
    model_repo.record_evaluation(model.id, {"candidate_test_mae": 4.1})
    model_repo.open_for_review(model.id, reviewer_id=role_sessions["team_admin"][0].id)
    model_repo.approve_model(model.id, reviewer_id=role_sessions["team_admin"][0].id)


    with pytest.raises(HTTPException) as exc_info:
        deploy_model(model_id=model.id, context=ic_ctx)
    assert exc_info.value.status_code == 403


    with pytest.raises(HTTPException) as exc_info:
        deploy_model(model_id=model.id, context=mod_ctx)
    assert exc_info.value.status_code == 403


    res = deploy_model(model_id=model.id, context=admin_ctx)
    assert res["status"] == "success"
    assert res["model"]["status"] == "production"


def test_api_rejection_user_audit_log_inspection(role_sessions):
    """User / Local Contributor -> audit log listing raises HTTP 403 Forbidden."""
    user_ctx = role_sessions["user"][1]
    lc_ctx = role_sessions["local_contributor"][1]
    admin_ctx = role_sessions["team_admin"][1]

    with pytest.raises(HTTPException) as exc_info:
        list_audit_logs(session=user_ctx)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        list_audit_logs(session=lc_ctx)
    assert exc_info.value.status_code == 403

    res = list_audit_logs(session=admin_ctx)
    assert "items" in res


def test_api_rejection_moderator_internal_access_approval(temp_db, role_sessions):
    """User / Moderator -> approve internal contributor request raises HTTP 403 Forbidden."""
    icr_repo = InternalContributorRepository(temp_db)
    user_usr = role_sessions["user"][0]
    user_ctx = role_sessions["user"][1]
    mod_ctx = role_sessions["moderator"][1]
    admin_ctx = role_sessions["team_admin"][1]

    req = icr_repo.submit_request(
        user_id=user_usr.id,
        reason="Field testing trajectory capture",
        experience="3 years GNSS research",
    )


    with pytest.raises(HTTPException) as exc_info:
        api_approve_internal_request(request_id=req.id, context=user_ctx)
    assert exc_info.value.status_code == 403


    with pytest.raises(HTTPException) as exc_info:
        api_approve_internal_request(request_id=req.id, context=mod_ctx)
    assert exc_info.value.status_code == 403


    res = api_approve_internal_request(request_id=req.id, context=admin_ctx)
    assert res["status"] == "approved"


def test_api_rejection_user_admin_dashboard_users_list(role_sessions):
    """User / Local Contributor -> list admin users endpoint raises HTTP 403 Forbidden."""
    user_ctx = role_sessions["user"][1]
    lc_ctx = role_sessions["local_contributor"][1]
    admin_ctx = role_sessions["team_admin"][1]

    with pytest.raises(HTTPException) as exc_info:
        admin_api.list_admin_users(session=user_ctx)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException) as exc_info:
        admin_api.list_admin_users(session=lc_ctx)
    assert exc_info.value.status_code == 403

    res = admin_api.list_admin_users(session=admin_ctx)
    assert "items" in res
    assert res["total"] >= 1
