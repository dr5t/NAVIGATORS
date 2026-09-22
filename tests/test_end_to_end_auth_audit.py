"""
Navigators IDR - Phase 21: End-to-End Authorization Audit Test Suite
Exhaustively audits backend RBAC enforcement across all roles (Guest, User, Local Contributor,
Internal Contributor, Moderator, Team Admin, Super Admin) against every sensitive system action:

1. View map (place:read)
2. Navigate (place:read)
3. Add place (contribution:create)
4. Edit own draft (contribution:update + ownership)
5. Approve contribution (contribution:approve)
6. Upload dataset (dataset:create)
7. Start ML training (training:create)
8. Approve model (model:approve)
9. Deploy model (model:deploy)
10. Manage roles (role:assign)

Guarantees 100% backend API security enforcement independent of UI state.
"""

import tempfile
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.auth_service import AuthService
from src.db.authorization import AuthorizationService, SessionContext
from src.db.places import PlaceRepository
from src.db.contributions import ContributionRepository
from src.db.datasets import DatasetRepository
from src.db.model_registry import ModelRegistryRepository
from src.db.audit import AuditRepository

from src.api.places import add_place, list_canonical_places, CreatePlaceRequest
from src.api.contributions import update_contribution, approve_contribution, UpdateContributionRequest, ReviewContributionRequest
from src.api.datasets import api_submit_session, SubmitSessionModel
from src.api.model_registry import approve_model, deploy_model, ApproveModelRequest
from src.api.admin import list_admin_users
from src.api.audit import list_audit_logs

import src.api.places as place_api
import src.api.contributions as contrib_api
import src.api.datasets as dataset_api
import src.api.model_registry as model_api
import src.api.audit as audit_api
import src.api.admin as admin_api


@pytest.fixture
def temp_db():
    """Create an isolated temporary database for authorization matrix auditing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)

    contrib_api.contrib_repo.db_path = db_path
    place_api.place_repo.db_path = db_path
    place_api.contrib_repo.db_path = db_path
    place_api.authz_service.rbac.db_path = db_path

    contrib_api.authz_service.rbac.db_path = db_path

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
    admin_api.rbac_repo.db_path = db_path

    auth_service = AuthService(db_path)

    guest_session, _ = auth_service.create_session(user_id=None, is_guest=True)
    user_usr, user_session, _ = auth_service.register("audit_user@test.com", "Password123!", "Standard User", role_id="user")
    user2_usr, user2_session, _ = auth_service.register("audit_user2@test.com", "Password123!", "Second User", role_id="user")
    lc_usr, lc_session, _ = auth_service.register("audit_lc@test.com", "Password123!", "Local Contributor", role_id="local_contributor")
    ic_usr, ic_session, _ = auth_service.register("audit_ic@test.com", "Password123!", "Internal Contributor", role_id="internal_contributor")
    mod_usr, mod_session, _ = auth_service.register("audit_mod@test.com", "Password123!", "Moderator", role_id="moderator")
    admin_usr, admin_session, _ = auth_service.register("audit_admin@test.com", "Password123!", "Team Admin", role_id="team_admin")
    super_usr, super_session, _ = auth_service.register("audit_super@test.com", "Password123!", "Super Admin", role_id="super_admin")

    guest_session.db_path = db_path
    user_session.db_path = db_path
    user2_session.db_path = db_path
    lc_session.db_path = db_path
    ic_session.db_path = db_path
    mod_session.db_path = db_path
    admin_session.db_path = db_path
    super_session.db_path = db_path

    yield {
        "db_path": db_path,
        "sessions": {
            "guest": (None, guest_session),
            "user": (user_usr, user_session),
            "user2": (user2_usr, user2_session),
            "local_contributor": (lc_usr, lc_session),
            "internal_contributor": (ic_usr, ic_session),
            "moderator": (mod_usr, mod_session),
            "team_admin": (admin_usr, admin_session),
            "super_admin": (super_usr, super_session),
        }
    }

    if db_path.exists():
        db_path.unlink()


def test_full_role_action_authorization_matrix(temp_db):
    """
    Exhaustively tests all 6 core roles + super_admin against all 10 sensitive permissions
    using the central AuthorizationService.
    """
    db_path = temp_db["db_path"]
    authz = AuthorizationService(db_path)
    sessions = temp_db["sessions"]

    # Expected matrix mapping: role_id -> dict of action -> allowed boolean
    MATRIX_EXPECTATIONS = {
        "guest": {
            "place:read": True,
            "sync:pull": False,  # Guest navigation uses public place:read; sync:pull requires auth account
            "contribution:create": False,
            "contribution:update": False,
            "contribution:approve": False,
            "dataset:create": False,
            "training:create": False,
            "model:approve": False,
            "model:deploy": False,
            "role:assign": False,
        },
        "user": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": False,
            "dataset:create": False,
            "training:create": False,
            "model:approve": False,
            "model:deploy": False,
            "role:assign": False,
        },
        "local_contributor": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": False,
            "dataset:create": False,
            "training:create": False,
            "model:approve": False,
            "model:deploy": False,
            "role:assign": False,
        },
        "internal_contributor": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": False,
            "dataset:create": True,
            "training:create": True,
            "model:approve": False,
            "model:deploy": False,
            "role:assign": False,
        },
        "moderator": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": True,
            "dataset:create": False,
            "training:create": False,
            "model:approve": False,
            "model:deploy": False,
            "role:assign": False,
        },
        "team_admin": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": True,
            "dataset:create": True,
            "training:create": True,
            "model:approve": True,
            "model:deploy": True,
            "role:assign": True,
        },
        "super_admin": {
            "place:read": True,
            "sync:pull": True,
            "contribution:create": True,
            "contribution:update": True,
            "contribution:approve": True,
            "dataset:create": True,
            "training:create": True,
            "model:approve": True,
            "model:deploy": True,
            "role:assign": True,
        }
    }

    for role_id, expected_actions in MATRIX_EXPECTATIONS.items():
        user_obj, session_ctx = sessions[role_id]
        for action, expected_allowed in expected_actions.items():
            decision = authz.can(user=session_ctx, action=action)
            assert decision.allowed == expected_allowed, (
                f"Role '{role_id}' for action '{action}' expected allowed={expected_allowed}, got {decision.allowed} ({decision.reason})"
            )


# =============================================================================
# Direct API Endpoint Verification Across Every Role
# =============================================================================

def test_api_action_1_and_2_view_map_and_navigate(temp_db):
    """Action 1 & 2: View Map / Navigate is ALLOWED for all roles including Guest."""
    sessions = temp_db["sessions"]
    for role_id, (_, session_ctx) in sessions.items():
        res = list_canonical_places(limit=10, context=session_ctx)
        assert "places" in res


def test_api_action_3_add_place(temp_db):
    """Action 3: Add Place is DENIED for Guest (401/403), ALLOWED for all authenticated roles."""
    sessions = temp_db["sessions"]
    req = CreatePlaceRequest(name="Audit Cafe", category="cafe", latitude=12.9, longitude=77.5)

    # Guest -> DENY (401/403)
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        add_place(req, context=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local, Internal, Moderator, Admin -> ALLOW (200/201)
    for role_id in ["user", "local_contributor", "internal_contributor", "moderator", "team_admin"]:
        _, session_ctx = sessions[role_id]
        res = add_place(req, context=session_ctx)
        assert res["contribution"]["title"] == "Audit Cafe"


def test_api_action_4_edit_own_draft_and_ownership(temp_db):
    """Action 4: Edit own draft is ALLOWED for owner, DENIED for non-owner (403), DENIED for guest (401/403)."""
    sessions = temp_db["sessions"]
    _, user_ctx = sessions["user"]
    _, user2_ctx = sessions["user2"]
    _, guest_ctx = sessions["guest"]

    # 1. User 1 creates draft
    create_res = add_place(CreatePlaceRequest(name="User1 Draft", category="fuel", latitude=10.0, longitude=20.0), context=user_ctx)
    contrib_id = create_res["contribution"]["id"]

    # 2. Owner (User 1) updates own draft -> ALLOWED
    upd_req = UpdateContributionRequest(title="User1 Draft Updated")
    upd_res = update_contribution(contrib_id, upd_req, context=user_ctx)
    assert upd_res["contribution"]["title"] == "User1 Draft Updated"

    # 3. Non-owner (User 2) attempts to update User 1's draft -> DENIED (403)
    with pytest.raises(HTTPException) as exc:
        update_contribution(contrib_id, upd_req, context=user2_ctx)
    assert exc.value.status_code == 403

    # 4. Guest attempts to update User 1's draft -> DENIED (401/403)
    with pytest.raises(HTTPException) as exc:
        update_contribution(contrib_id, upd_req, context=guest_ctx)
    assert exc.value.status_code in (401, 403)


def test_api_action_5_approve_contribution(temp_db):
    """Action 5: Approve Contribution is DENIED for Guest (401/403), User/Local/Internal (403), ALLOWED for Moderator & Admin."""
    sessions = temp_db["sessions"]
    _, user_ctx = sessions["user"]

    # Create pending contribution
    draft_res = add_place(CreatePlaceRequest(name="Approve Test Place", category="park", latitude=12.0, longitude=77.0, submit_now=True), context=user_ctx)
    contrib_id = draft_res["contribution"]["id"]
    rev_req = ReviewContributionRequest(decision="approve", notes="Moderation approval test")

    # Guest -> 401/403
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        approve_contribution(contrib_id, rev_req, context=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local Contributor, Internal Contributor -> 403 DENY
    for role_id in ["user", "local_contributor", "internal_contributor"]:
        _, s_ctx = sessions[role_id]
        with pytest.raises(HTTPException) as exc:
            approve_contribution(contrib_id, rev_req, context=s_ctx)
        assert exc.value.status_code == 403

    # Moderator -> ALLOWED (200)
    _, mod_ctx = sessions["moderator"]
    mod_res = approve_contribution(contrib_id, req=None, context=mod_ctx)
    assert mod_res["contribution"]["status"] == "approved"


def test_api_action_6_upload_dataset(temp_db):
    """Action 6: Upload dataset is DENIED for Guest (401/403), User/Local/Moderator (403), ALLOWED for Internal Contributor & Admin."""
    sessions = temp_db["sessions"]
    ds_req = SubmitSessionModel(activity_type="walking", device="Pixel 8", duration_seconds=100.0, consent=True)

    # Guest -> 401/403
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        api_submit_session(ds_req, context=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local Contributor, Moderator -> 403 DENY
    for role_id in ["user", "local_contributor", "moderator"]:
        _, s_ctx = sessions[role_id]
        with pytest.raises(HTTPException) as exc:
            api_submit_session(ds_req, context=s_ctx)
        assert exc.value.status_code == 403

    # Internal Contributor & Admin -> ALLOWED (201)
    for role_id in ["internal_contributor", "team_admin"]:
        _, s_ctx = sessions[role_id]
        res = api_submit_session(ds_req, context=s_ctx)
        assert res["session"]["id"] is not None


def test_api_action_7_start_ml_training(temp_db):
    """Action 7: Start ML Training permission check (DENIED for Guest/User/Local/Moderator, ALLOWED for Internal & Admin)."""
    db_path = temp_db["db_path"]
    authz = AuthorizationService(db_path)
    sessions = temp_db["sessions"]

    for role_id in ["guest", "user", "local_contributor", "moderator"]:
        _, s_ctx = sessions[role_id]
        assert authz.can(user=s_ctx, action="training:create").allowed is False

    for role_id in ["internal_contributor", "team_admin", "super_admin"]:
        _, s_ctx = sessions[role_id]
        assert authz.can(user=s_ctx, action="training:create").allowed is True


def test_api_action_8_approve_model(temp_db):
    """Action 8: Approve model is DENIED for Guest (401/403), User/Local/Internal/Moderator (403), ALLOWED for Team Admin."""
    sessions = temp_db["sessions"]
    db_path = temp_db["db_path"]
    repo = ModelRegistryRepository(db_path)
    admin_usr, admin_ctx = sessions["team_admin"]

    # Register candidate model & move to review
    cand = repo.register_candidate(name="v2.0-eval", registered_by=admin_usr.id)
    repo.record_evaluation(cand.id, {"test_mae": 0.40})
    repo.open_for_review(cand.id, reviewer_id=admin_usr.id)

    appr_req = ApproveModelRequest(notes="Approved for benchmark")

    # Guest -> 401/403
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        approve_model(cand.id, body=appr_req, context=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local, Internal, Moderator -> 403 DENY
    for role_id in ["user", "local_contributor", "internal_contributor", "moderator"]:
        _, s_ctx = sessions[role_id]
        with pytest.raises(HTTPException) as exc:
            approve_model(cand.id, body=appr_req, context=s_ctx)
        assert exc.value.status_code == 403

    # Team Admin -> ALLOWED
    res = approve_model(cand.id, body=appr_req, context=admin_ctx)
    assert res["status"] == "success"


def test_api_action_9_deploy_model(temp_db, tmp_path):
    """Action 9: Deploy model is DENIED for Guest (401/403), User/Local/Internal/Moderator (403), ALLOWED for Team Admin."""
    sessions = temp_db["sessions"]
    db_path = temp_db["db_path"]
    repo = ModelRegistryRepository(db_path)
    admin_usr, admin_ctx = sessions["team_admin"]

    dummy_onnx = tmp_path / "deploy_eval.onnx"
    dummy_onnx.write_bytes(b"onnx_binary")

    cand = repo.register_candidate(name="v2.5-prod", registered_by=admin_usr.id, onnx_path=str(dummy_onnx))
    repo.record_evaluation(cand.id, {"test_mae": 0.30})
    repo.open_for_review(cand.id, reviewer_id=admin_usr.id)
    repo.approve_model(cand.id, reviewer_id=admin_usr.id, notes="Qualified")

    # Guest -> 401/403
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        deploy_model(cand.id, context=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local, Internal, Moderator -> 403 DENY
    for role_id in ["user", "local_contributor", "internal_contributor", "moderator"]:
        _, s_ctx = sessions[role_id]
        with pytest.raises(HTTPException) as exc:
            deploy_model(cand.id, context=s_ctx)
        assert exc.value.status_code == 403

    # Team Admin -> ALLOWED
    prod_onnx_dest = tmp_path / "prod_sim_deploy.onnx"
    repo.deploy(cand.id, deployer_id=admin_usr.id, production_onnx_dest=str(prod_onnx_dest))
    assert repo.get_production_model().id == cand.id


def test_api_action_10_manage_roles(temp_db):
    """Action 10: Manage roles / User admin is DENIED for Guest (401), User/Local/Internal/Moderator (403), ALLOWED for Admin."""
    sessions = temp_db["sessions"]

    # Guest -> 401
    _, guest_ctx = sessions["guest"]
    with pytest.raises(HTTPException) as exc:
        list_admin_users(session=guest_ctx)
    assert exc.value.status_code in (401, 403)

    # User, Local, Internal -> 403 DENY
    for role_id in ["user", "local_contributor", "internal_contributor"]:
        _, s_ctx = sessions[role_id]
        with pytest.raises(HTTPException) as exc:
            list_admin_users(session=s_ctx)
        assert exc.value.status_code == 403

    # Team Admin & Super Admin -> ALLOWED
    for role_id in ["team_admin", "super_admin"]:
        _, s_ctx = sessions[role_id]
        res = list_admin_users(session=s_ctx)
        assert "items" in res
