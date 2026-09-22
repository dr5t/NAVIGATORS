"""
Tests for Phase 13 (Internal Dataset Contribution) and Phase 14 (ML Training Workflow)

Verifies:
1.  internal_contributor can submit dataset sessions.
2.  Normal user is denied dataset:create (403).
3.  State machine: uploaded → validating.
4.  State machine: validating → validated.
5.  Rejection from uploaded state.
6.  Rejection from validating state.
7.  Invalid transitions raise ValueError.
8.  Terminal state (validated) cannot be transitioned further.
9.  Team admin can call validate/reject API endpoints.
10. Normal user is denied validate/reject API endpoints (403).
11. List filtering by status works correctly.
12. Stats endpoint returns correct aggregate counts.
13. Training does not equal deployment (RBAC enforced).
14. api_submit_session endpoint returns wrapped response with status=uploaded.
"""

import pytest
import secrets

from src.db.database import init_db
from src.db.datasets import DatasetRepository, ACTIVITY_TYPES, VALID_TRANSITIONS
from src.db.auth_service import AuthService
from src.db.authorization import AuthorizationService
from fastapi import HTTPException


# =============================================================================
# Fixtures & Helpers
# =============================================================================

@pytest.fixture()
def tmp_db(tmp_path):
    """Isolated DB per test - no shared state pollution."""
    db_file = tmp_path / "test_datasets.db"
    init_db(db_file)
    return db_file


def _register(db_file, role: str, tag: str = ""):
    auth = AuthService(db_file)
    suffix = tag or secrets.token_hex(4)
    user, session, _ = auth.register(
        email=f"ds_{role}_{suffix}@example.com",
        password="Password123!",
        name=f"DS {role.title()} {suffix}",
        role_id=role,
    )
    return user, session


def _submit(repo: DatasetRepository, contributor_id: str, activity: str = "walking") -> object:
    return repo.submit_session(
        contributor_id=contributor_id,
        activity_type=activity,
        device="Pixel 7 Pro / Android 14",
        duration_seconds=300.0,
        gnss_available=True,
        consent=True,
        notes="Urban canyon test route",
    )


# =============================================================================
# Phase 13 - Repository (State Machine) Tests
# =============================================================================

def test_internal_contributor_can_submit_session(tmp_db):
    """internal_contributor submits a session; status starts as 'uploaded'."""
    user, _ = _register(tmp_db, "internal_contributor")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, user.id, "driving")

    assert session.status == "uploaded"
    assert session.contributor_id == user.id
    assert session.activity_type == "driving"
    assert session.gnss_available is True
    assert session.consent is True
    assert session.rejection_reason is None
    assert session.validated_by is None


def test_submit_requires_consent(tmp_db):
    """Submitting without consent raises ValueError."""
    user, _ = _register(tmp_db, "internal_contributor")
    repo = DatasetRepository(tmp_db)

    with pytest.raises(ValueError, match="consent"):
        repo.submit_session(
            contributor_id=user.id,
            activity_type="walking",
            device="iPhone 15",
            consent=False,
        )


def test_submit_invalid_activity_type(tmp_db):
    """Unknown activity_type is rejected."""
    user, _ = _register(tmp_db, "internal_contributor")
    repo = DatasetRepository(tmp_db)

    with pytest.raises(ValueError, match="activity_type"):
        repo.submit_session(
            contributor_id=user.id,
            activity_type="flying",
            device="iPhone 15",
            consent=True,
        )


def test_all_activity_types_accepted(tmp_db):
    """All five canonical activity types can be submitted."""
    user, _ = _register(tmp_db, "internal_contributor")
    repo = DatasetRepository(tmp_db)

    for activity in sorted(ACTIVITY_TYPES):
        tag = secrets.token_hex(3)
        session = repo.submit_session(
            contributor_id=user.id,
            activity_type=activity,
            device=f"Device_{tag}",
            consent=True,
        )
        assert session.activity_type == activity
        assert session.status == "uploaded"


def test_state_machine_uploaded_to_validating(tmp_db):
    """uploaded → validating transition succeeds."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    assert session.status == "uploaded"

    reviewing = repo.start_validation(session_id=session.id, validator_id=admin.id)
    assert reviewing.status == "validating"
    assert reviewing.validated_by == admin.id


def test_state_machine_validating_to_validated(tmp_db):
    """validating → validated transition succeeds; session becomes training-eligible."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    repo.start_validation(session_id=session.id, validator_id=admin.id)
    validated = repo.validate_session(session_id=session.id, validator_id=admin.id)

    assert validated.status == "validated"
    assert validated.validated_by == admin.id
    assert validated.validated_at is not None


def test_reject_from_uploaded_state(tmp_db):
    """Rejection directly from uploaded state is permitted."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    rejected = repo.reject_session(
        session_id=session.id,
        validator_id=admin.id,
        rejection_reason="Sensor data is corrupted beyond recovery",
    )

    assert rejected.status == "rejected"
    assert "corrupted" in rejected.rejection_reason
    assert rejected.validated_by == admin.id


def test_reject_from_validating_state(tmp_db):
    """Rejection from validating state is also permitted."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    repo.start_validation(session_id=session.id, validator_id=admin.id)
    rejected = repo.reject_session(
        session_id=session.id,
        validator_id=admin.id,
        rejection_reason="GNSS timestamps do not align with IMU data",
    )

    assert rejected.status == "rejected"


def test_invalid_transition_raises_value_error(tmp_db):
    """
    Attempting an invalid state transition must raise ValueError.
    Examples: uploaded → validated (skipping validating)
              validating → uploaded  (backward move)
    """
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    # uploaded → validated (must go through validating first)
    session = _submit(repo, contributor.id)
    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.validate_session(session_id=session.id, validator_id=admin.id)


def test_terminal_validated_state_cannot_transition(tmp_db):
    """'validated' is terminal - no further transitions allowed."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    repo.start_validation(session_id=session.id, validator_id=admin.id)
    repo.validate_session(session_id=session.id, validator_id=admin.id)

    # Cannot reject after validating
    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.reject_session(
            session_id=session.id,
            validator_id=admin.id,
            rejection_reason="Attempting to reject an already validated session",
        )

    # Cannot start validation again
    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.start_validation(session_id=session.id, validator_id=admin.id)


def test_terminal_rejected_state_cannot_transition(tmp_db):
    """'rejected' is terminal - cannot be re-opened."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    session = _submit(repo, contributor.id)
    repo.reject_session(session_id=session.id, validator_id=admin.id, rejection_reason="Bad data")

    with pytest.raises(ValueError, match="Invalid state transition"):
        repo.start_validation(session_id=session.id, validator_id=admin.id)


# =============================================================================
# Phase 13 - List & Stats Tests
# =============================================================================

def test_list_sessions_filtered_by_status(tmp_db):
    """list_sessions correctly filters by status."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    # Create 3 uploaded, validate 1
    s1 = _submit(repo, contributor.id, "walking")
    s2 = _submit(repo, contributor.id, "driving")
    s3 = _submit(repo, contributor.id, "gnss_imu")

    repo.start_validation(s1.id, admin.id)
    repo.validate_session(s1.id, admin.id)

    uploaded_items, uploaded_total = repo.list_sessions(status="uploaded")
    validated_items, validated_total = repo.list_sessions(status="validated")

    assert uploaded_total == 2
    assert validated_total == 1
    assert validated_items[0].id == s1.id


def test_list_sessions_scoped_by_contributor(tmp_db):
    """list_sessions filtered by contributor_id returns only that user's sessions."""
    userA, _ = _register(tmp_db, "internal_contributor", "A")
    userB, _ = _register(tmp_db, "internal_contributor", "B")
    repo = DatasetRepository(tmp_db)

    _submit(repo, userA.id)
    _submit(repo, userA.id)
    _submit(repo, userB.id)

    items_A, total_A = repo.list_sessions(contributor_id=userA.id)
    items_B, total_B = repo.list_sessions(contributor_id=userB.id)

    assert total_A == 2
    assert total_B == 1


def test_stats_returns_correct_counts(tmp_db):
    """get_stats() returns accurate counts per status and activity."""
    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, _ = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)

    w = _submit(repo, contributor.id, "walking")
    d = _submit(repo, contributor.id, "driving")
    g = _submit(repo, contributor.id, "gnss_outage")

    # Validate walking session
    repo.start_validation(w.id, admin.id)
    repo.validate_session(w.id, admin.id)

    # Reject driving session
    repo.reject_session(d.id, admin.id, "Poor GNSS coverage")

    stats = repo.get_stats()

    assert stats["total"] == 3
    assert stats["by_status"]["uploaded"] == 1     # gnss_outage
    assert stats["by_status"]["validated"] == 1    # walking
    assert stats["by_status"]["rejected"] == 1     # driving
    assert stats["validated_for_training"] == 1
    assert stats["by_activity"]["walking"] == 1
    assert stats["by_activity"]["driving"] == 1


# =============================================================================
# Phase 13 - API-Level Tests
# =============================================================================

def test_api_normal_user_denied_dataset_submit(tmp_db, monkeypatch):
    """Normal 'user' role lacks dataset:create permission → 403."""
    import src.api.datasets as ds_mod
    from src.api.datasets import api_submit_session, SubmitSessionModel

    user, session = _register(tmp_db, "user")
    monkeypatch.setattr(ds_mod, "dataset_repo", DatasetRepository(tmp_db))
    monkeypatch.setattr(ds_mod, "authz_service", AuthorizationService(tmp_db))

    with pytest.raises(HTTPException) as exc:
        api_submit_session(
            body=SubmitSessionModel(
                activity_type="walking",
                device="iPhone 15",
                consent=True,
            ),
            context=session,
        )
    assert exc.value.status_code == 403


def test_api_internal_contributor_can_submit(tmp_db, monkeypatch):
    """internal_contributor submits via API; response is wrapped with status=uploaded."""
    import src.api.datasets as ds_mod
    from src.api.datasets import api_submit_session, SubmitSessionModel

    user, session = _register(tmp_db, "internal_contributor")
    monkeypatch.setattr(ds_mod, "dataset_repo", DatasetRepository(tmp_db))
    monkeypatch.setattr(ds_mod, "authz_service", AuthorizationService(tmp_db))

    resp = api_submit_session(
        body=SubmitSessionModel(
            activity_type="gnss_imu",
            device="Pixel 7 Pro",
            duration_seconds=180.0,
            gnss_available=True,
            consent=True,
            notes="Highway test with tunnel",
        ),
        context=session,
    )
    assert "session" in resp
    assert "message" in resp
    assert resp["session"]["status"] == "uploaded"
    assert resp["session"]["contributor_id"] == user.id


def test_api_team_admin_can_validate_session(tmp_db, monkeypatch):
    """team_admin calls validate endpoints; state machine progresses correctly."""
    import src.api.datasets as ds_mod
    from src.api.datasets import (
        api_start_validation, api_validate_session, api_reject_session,
        RejectSessionModel,
    )

    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, admin_session = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)
    monkeypatch.setattr(ds_mod, "dataset_repo", repo)
    monkeypatch.setattr(ds_mod, "authz_service", AuthorizationService(tmp_db))

    session = _submit(repo, contributor.id, "driving")

    # Start validation
    r1 = api_start_validation(session_id=session.id, context=admin_session)
    assert r1["session"]["status"] == "validating"

    # Validate
    r2 = api_validate_session(session_id=session.id, context=admin_session)
    assert r2["session"]["status"] == "validated"
    assert "message" in r2


def test_api_team_admin_can_reject_session(tmp_db, monkeypatch):
    """team_admin rejects a session; reason is stored."""
    import src.api.datasets as ds_mod
    from src.api.datasets import api_reject_session, RejectSessionModel

    contributor, _ = _register(tmp_db, "internal_contributor")
    admin, admin_session = _register(tmp_db, "team_admin", "admin")
    repo = DatasetRepository(tmp_db)
    monkeypatch.setattr(ds_mod, "dataset_repo", repo)
    monkeypatch.setattr(ds_mod, "authz_service", AuthorizationService(tmp_db))

    session = _submit(repo, contributor.id, "gnss_outage")

    resp = api_reject_session(
        session_id=session.id,
        body=RejectSessionModel(rejection_reason="GNSS dropout too frequent for this benchmark"),
        context=admin_session,
    )
    assert resp["session"]["status"] == "rejected"
    assert "frequent" in resp["session"]["rejection_reason"]


def test_api_normal_user_denied_validate_endpoints(tmp_db, monkeypatch):
    """Normal user calling validate/reject endpoints receives 403."""
    import src.api.datasets as ds_mod
    from src.api.datasets import api_start_validation, api_reject_session, RejectSessionModel

    contributor, _ = _register(tmp_db, "internal_contributor", "contrib")
    user, user_session = _register(tmp_db, "user", "plain")
    repo = DatasetRepository(tmp_db)
    monkeypatch.setattr(ds_mod, "dataset_repo", repo)
    monkeypatch.setattr(ds_mod, "authz_service", AuthorizationService(tmp_db))

    session = _submit(repo, contributor.id)

    with pytest.raises(HTTPException) as exc:
        api_start_validation(session_id=session.id, context=user_session)
    assert exc.value.status_code == 403

    with pytest.raises(HTTPException) as exc:
        api_reject_session(
            session_id=session.id,
            body=RejectSessionModel(rejection_reason="Trying to reject"),
            context=user_session,
        )
    assert exc.value.status_code == 403


# =============================================================================
# Phase 14 - Training ≠ Deployment (RBAC Enforced)
# =============================================================================

def test_training_does_not_equal_deployment(tmp_db):
    """
    PRODUCT RULE: Training does not equal production deployment.

    internal_contributor CAN dispatch training jobs (training:create).
    internal_contributor CANNOT deploy models (model:deploy).
    Only team_admin and super_admin hold model:deploy.
    """
    contributor, contrib_session = _register(tmp_db, "internal_contributor")
    admin, admin_session = _register(tmp_db, "team_admin", "admin")
    authz = AuthorizationService(tmp_db)

    contrib_ctx = contrib_session.to_dict()
    admin_ctx   = admin_session.to_dict()

    # internal_contributor CAN start training
    assert authz.can(contrib_ctx, "training:create", "training").allowed is True

    # internal_contributor CANNOT deploy
    assert authz.can(contrib_ctx, "model:deploy", "model").allowed is False

    # internal_contributor CANNOT validate datasets
    assert authz.can(contrib_ctx, "dataset:validate", "dataset").allowed is False

    # team_admin CAN deploy
    assert authz.can(admin_ctx, "model:deploy", "model").allowed is True

    # team_admin CAN validate datasets
    assert authz.can(admin_ctx, "dataset:validate", "dataset").allowed is True


def test_normal_user_blocked_from_all_engineering_controls(tmp_db):
    """
    A normal registered 'user' must be denied all engineering-tier permissions:
    training:create, dataset:create, dataset:validate, model:deploy.
    """
    user, session = _register(tmp_db, "user")
    authz = AuthorizationService(tmp_db)
    ctx = session.to_dict()

    assert authz.can(ctx, "training:create", "training").allowed is False
    assert authz.can(ctx, "dataset:create",  "dataset").allowed  is False
    assert authz.can(ctx, "dataset:validate", "dataset").allowed is False
    assert authz.can(ctx, "model:deploy",     "model").allowed   is False
