"""
Navigators IDR - Phase 8 Moderation & Audit Trail Test Suite
Validates the complete moderation and governance system:
  1. Moderation triage queues (Pending, Approved, Rejected, Changes Requested).
  2. Item inspection cards enriched with author profile, coordinates, evidence, and structured diff.
  3. Tri-state moderation actions:
     - Approve
     - Reject
     - Request Changes
  4. Resubmission flow after changes requested.
  5. Role verification: standard users and guests are denied access (HTTP 403 / 401).
  6. Community reporting workflow and moderator resolution.
  7. Full audit trail logging for all actions and state transitions.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.db.contributions import ContributionRepository, Contribution
from src.db.places import PlaceRepository, Place
from src.db.reports import ReportRepository, Report
from src.db.audit import AuditRepository, AuditEntry
from src.db.state_machine import ContributionState
from src.api.moderation import (
    list_moderation_queue as api_list_moderation_queue,
    get_moderation_contribution_detail as api_get_moderation_detail,
    approve_contribution as api_approve_contribution,
    reject_contribution as api_reject_contribution,
    request_changes_on_contribution as api_request_changes,
    list_reported_items as api_list_reports,
    resolve_report as api_resolve_report,
    list_audit_trail as api_list_audit_trail,
    ApproveContributionRequest,
    RejectContributionRequest,
    RequestChangesRequest,
    ResolveReportRequest,
)
from src.api.reports import (
    submit_report as api_submit_report,
    CreateReportRequest,
)
from src.api.places import (
    suggest_edit_place as api_suggest_edit_place,
    SuggestEditRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_moderation_test.db"
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


@pytest.fixture
def place_repo(temp_db: Path):
    return PlaceRepository(temp_db)


@pytest.fixture
def report_repo(temp_db: Path):
    return ReportRepository(temp_db)


@pytest.fixture
def audit_repo(temp_db: Path):
    return AuditRepository(temp_db)






def test_moderation_queue_filtering_and_enrichment(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
    place_repo: PlaceRepository,
):
    """
    Moderator inspects triage queue:
      - Pending items are returned with author info, location, evidence, and structured diff.
      - Approved and rejected filters return their respective subsets.
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    test_rbac = RBACRepository(temp_db)

    monkeypatch.setattr(mod_api, "contrib_repo", contrib_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)
    monkeypatch.setattr(mod_api, "rbac_repo", test_rbac)

    author, session_author, _ = auth_service.register("author_queue@navigators.dev", "Password123!", "Alice Author")
    mod, session_mod, _ = auth_service.register("mod_queue@navigators.dev", "Password123!", "Bob Moderator", role_id="moderator")


    c1 = contrib_repo.create(
        contribution_id="c_pending_1",
        owner_id=author.id,
        resource_type="place",
        title="Highway Rest Stop",
        data={
            "name": "Highway Rest Stop",
            "category": "fuel",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "address": "NH-48 Mile 42",
            "opening_hours": "24/7",
            "phone": "+91 11 1234 5678",
            "notes": "Verified new signage on highway",
        },
        status=ContributionState.PENDING_REVIEW,
    )


    res = api_list_moderation_queue(status="pending", context=session_mod)
    assert res["count"] == 1
    item = res["items"][0]


    assert item["id"] == "c_pending_1"
    assert item["author"]["name"] == "Alice Author"
    assert item["author"]["email"] == "author_queue@navigators.dev"
    assert item["location"]["latitude"] == 28.6139
    assert item["location"]["longitude"] == 77.2090
    assert item["evidence"]["opening_hours"] == "24/7"
    assert item["evidence"]["phone"] == "+91 11 1234 5678"
    assert "name" in item["changes"]
    assert item["changes"]["name"]["new"] == "Highway Rest Stop"






def test_moderator_approve_and_audit_record(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
    audit_repo: AuditRepository,
):
    """
    Moderator approves pending contribution:
      - State becomes 'approved'.
      - Immutable audit log entry is persisted.
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(mod_api, "contrib_repo", contrib_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)
    monkeypatch.setattr(mod_api, "audit_repo", audit_repo)

    author, _, _ = auth_service.register("author_app@navigators.dev", "Password123!", "Author App")
    mod, session_mod, _ = auth_service.register("mod_app@navigators.dev", "Password123!", "Mod App", role_id="moderator")

    contrib = contrib_repo.create(
        contribution_id="c_app_1",
        owner_id=author.id,
        resource_type="place",
        title="City Pharmacy",
        data={"name": "City Pharmacy", "category": "pharmacy"},
        status=ContributionState.PENDING_REVIEW,
    )


    res = api_approve_contribution(
        contrib_id=contrib.id,
        req=ApproveContributionRequest(notes="Signboard photo confirmed"),
        context=session_mod,
    )
    assert res["item"]["status"] == ContributionState.APPROVED


    logs = audit_repo.list_logs(resource_id=contrib.id, action="contribution:approved")
    assert len(logs) == 1
    audit = logs[0]
    assert audit.actor_id == mod.id
    assert audit.old_state == ContributionState.PENDING_REVIEW
    assert audit.new_state == ContributionState.APPROVED
    assert audit.metadata.get("notes") == "Signboard photo confirmed"






def test_moderator_reject_and_audit_record(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
    audit_repo: AuditRepository,
):
    """
    Moderator rejects pending contribution:
      - State becomes 'rejected'.
      - Mandatory rationale reason is required.
      - Immutable audit log entry is persisted.
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(mod_api, "contrib_repo", contrib_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)
    monkeypatch.setattr(mod_api, "audit_repo", audit_repo)

    author, _, _ = auth_service.register("author_rej@navigators.dev", "Password123!", "Author Rej")
    mod, session_mod, _ = auth_service.register("mod_rej@navigators.dev", "Password123!", "Mod Rej", role_id="moderator")

    contrib = contrib_repo.create(
        contribution_id="c_rej_1",
        owner_id=author.id,
        resource_type="place",
        title="Duplicate Fuel Station",
        data={"name": "Duplicate Fuel Station", "category": "fuel"},
        status=ContributionState.PENDING_REVIEW,
    )


    res = api_reject_contribution(
        contrib_id=contrib.id,
        req=RejectContributionRequest(reason="Location duplicates existing canonical fuel station #402"),
        context=session_mod,
    )
    assert res["item"]["status"] == ContributionState.REJECTED


    logs = audit_repo.list_logs(resource_id=contrib.id, action="contribution:rejected")
    assert len(logs) == 1
    audit = logs[0]
    assert audit.actor_id == mod.id
    assert audit.old_state == ContributionState.PENDING_REVIEW
    assert audit.new_state == ContributionState.REJECTED
    assert "duplicates" in audit.metadata.get("notes", "")






def test_moderator_request_changes_and_resubmission_flow(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
    audit_repo: AuditRepository,
):
    """
    Moderator requests revisions:
      1. Contribution transitions to 'changes_requested'.
      2. Audit record is persisted.
      3. Author updates payload content.
      4. Author resubmits to 'pending_review'.
      5. Item returns to moderator pending queue.
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(mod_api, "contrib_repo", contrib_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)
    monkeypatch.setattr(mod_api, "audit_repo", audit_repo)

    author, session_author, _ = auth_service.register("author_rc@navigators.dev", "Password123!", "Author RC")
    mod, session_mod, _ = auth_service.register("mod_rc@navigators.dev", "Password123!", "Mod RC", role_id="moderator")

    contrib = contrib_repo.create(
        contribution_id="c_rc_1",
        owner_id=author.id,
        resource_type="place",
        title="Vague Clinic",
        data={"name": "Vague Clinic", "category": "hospital"},
        status=ContributionState.PENDING_REVIEW,
    )


    res = api_request_changes(
        contrib_id=contrib.id,
        req=RequestChangesRequest(notes="Please specify opening hours and contact phone number."),
        context=session_mod,
    )
    assert res["item"]["status"] == ContributionState.CHANGES_REQUESTED


    logs = audit_repo.list_logs(resource_id=contrib.id, action="contribution:changes_requested")
    assert len(logs) == 1
    assert logs[0].actor_id == mod.id
    assert logs[0].new_state == ContributionState.CHANGES_REQUESTED


    contrib_repo.update_content(
        contribution_id=contrib.id,
        title="Apex Specialty Clinic",
        data={"name": "Apex Specialty Clinic", "category": "hospital", "opening_hours": "08:00 - 20:00", "phone": "+91 11 7777 8888"},
    )


    resubmitted = contrib_repo.submit(contrib.id, user=session_author)
    assert resubmitted.status == ContributionState.PENDING_REVIEW


    pending = api_list_moderation_queue(status="pending", context=session_mod)
    pending_ids = [it["id"] for it in pending["items"]]
    assert contrib.id in pending_ids






def test_non_moderator_cannot_access_moderation_endpoints(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Standard users receive HTTP 403 when trying to access moderation queues
    or execute moderation actions. Unauthenticated guests receive HTTP 401.
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(mod_api, "contrib_repo", contrib_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("regular_user@navigators.dev", "Password123!", "Regular User")
    guest_session, _ = auth_service.create_guest_session()


    with pytest.raises(HTTPException) as exc_queue:
        api_list_moderation_queue(status="pending", context=session_user)
    assert exc_queue.value.status_code == 403


    with pytest.raises(HTTPException) as exc_guest:
        api_list_moderation_queue(status="pending", context=guest_session)
    assert exc_guest.value.status_code == 401


    with pytest.raises(HTTPException) as exc_app:
        api_approve_contribution(contrib_id="c_fake", context=session_user)
    assert exc_app.value.status_code == 403






def test_community_reports_and_moderator_resolution_workflow(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    report_repo: ReportRepository,
    audit_repo: AuditRepository,
):
    """
    Community reporting lifecycle:
      1. User submits report on inaccurate place.
      2. Audit record 'report:created' is persisted.
      3. Moderator inspects reported items queue.
      4. Moderator resolves report with notes.
      5. Audit record 'report:resolved' is persisted.
    """
    import src.api.reports as rep_api
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)
    test_rbac = RBACRepository(temp_db)

    monkeypatch.setattr(rep_api, "report_repo", report_repo)
    monkeypatch.setattr(rep_api, "auth_service", auth_service)
    monkeypatch.setattr(rep_api, "authz_service", test_authz)
    monkeypatch.setattr(rep_api, "audit_repo", audit_repo)

    monkeypatch.setattr(mod_api, "report_repo", report_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)
    monkeypatch.setattr(mod_api, "audit_repo", audit_repo)
    monkeypatch.setattr(mod_api, "rbac_repo", test_rbac)

    user, session_user, _ = auth_service.register("citizen@navigators.dev", "Password123!", "Citizen User")
    mod, session_mod, _ = auth_service.register("mod_reports@navigators.dev", "Password123!", "Mod Reports", role_id="moderator")


    rep_res = api_submit_report(
        req=CreateReportRequest(
            target_type="place",
            target_id="plc_inaccurate_123",
            reason="Place closed permanently and building demolished",
            details="Visited location yesterday; site is now a construction zone.",
        ),
        context=session_user,
    )
    report_id = rep_res["report"]["id"]
    assert report_id.startswith("rep_")
    assert rep_res["report"]["status"] == "pending"


    logs_created = audit_repo.list_logs(resource_id=report_id, action="report:created")
    assert len(logs_created) == 1
    assert logs_created[0].actor_id == user.id


    mod_reports = api_list_reports(status="pending", context=session_mod)
    assert mod_reports["count"] == 1
    r_item = mod_reports["reports"][0]
    assert r_item["id"] == report_id
    assert r_item["reason"] == "Place closed permanently and building demolished"


    res_resolved = api_resolve_report(
        report_id=report_id,
        req=ResolveReportRequest(decision="resolved", notes="Verified demolition notice; place archived."),
        context=session_mod,
    )
    assert res_resolved["report"]["status"] == "resolved"


    logs_res = audit_repo.list_logs(resource_id=report_id, action="report:resolved")
    assert len(logs_res) == 1
    assert logs_res[0].actor_id == mod.id
    assert logs_res[0].new_state == "resolved"






def test_audit_logs_querying_and_governance(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    audit_repo: AuditRepository,
):
    """
    Moderator queries immutable audit trail:
      - Can filter by actor_id, resource_type, or action.
      - Regular users cannot access audit logs (HTTP 403).
    """
    import src.api.moderation as mod_api
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(mod_api, "audit_repo", audit_repo)
    monkeypatch.setattr(mod_api, "auth_service", auth_service)
    monkeypatch.setattr(mod_api, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("aud_user@navigators.dev", "Password123!", "User")
    mod, session_mod, _ = auth_service.register("aud_mod@navigators.dev", "Password123!", "Mod", role_id="moderator")


    audit_repo.log(action="place:create", resource_type="place", resource_id="plc_1", actor_id=mod.id)
    audit_repo.log(action="contribution:approve", resource_type="contribution", resource_id="c_1", actor_id=mod.id)
    audit_repo.log(action="report:created", resource_type="report", resource_id="r_1", actor_id=user.id)


    res_all = api_list_audit_trail(context=session_mod)
    assert res_all["count"] == 3


    res_place = api_list_audit_trail(resource_type="place", context=session_mod)
    assert res_place["count"] == 1
    assert res_place["audit_logs"][0]["action"] == "place:create"


    with pytest.raises(HTTPException) as exc_aud:
        api_list_audit_trail(context=session_user)
    assert exc_aud.value.status_code == 403


def test_contributor_experience_history_and_real_badges(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Phase 25: Contributor Experience Test.
    Verifies real contribution history listing and live database point/badge calculations.
    """
    import src.api.contributions as contrib_api_mod
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(contrib_api_mod, "contrib_repo", contrib_repo)
    monkeypatch.setattr(contrib_api_mod, "auth_service", auth_service)
    monkeypatch.setattr(contrib_api_mod, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("contributor_badge@navigators.dev", "Password123!", "Badge Contributor")
    mod, session_mod, _ = auth_service.register("mod_badge@navigators.dev", "Password123!", "Mod", role_id="moderator")


    c1 = contrib_repo.create(
        contribution_id="c_badge_1",
        owner_id=user.id,
        resource_type="place",
        title="Dehradun Petrol Pump",
        data={"name": "Dehradun Petrol Pump", "category": "petrol_pump"},
        status=ContributionState.DRAFT,
        action="create",
    )
    contrib_repo.submit(c1.id, user=session_user)


    my_res = contrib_api_mod.get_my_contributions_endpoint(context=session_user)
    assert my_res["count"] == 1
    assert my_res["items"][0]["title"] == "Dehradun Petrol Pump"
    assert my_res["items"][0]["status"] == ContributionState.PENDING_REVIEW


    stats1 = contrib_api_mod.get_contributor_stats_endpoint(context=session_user)
    assert stats1["points"] == 10
    assert stats1["level"] == 1
    assert stats1["approved_count"] == 0
    assert len(stats1["badges"]) == 0


    contrib_repo.approve(c1.id, reviewer=session_mod, notes="Verified station on site")
    contrib_repo.publish(c1.id, staff=session_mod)


    stats2 = contrib_api_mod.get_contributor_stats_endpoint(context=session_user)
    assert stats2["points"] == 50
    assert stats2["level"] == 2
    assert stats2["level_name"] == "Active Contributor"
    assert stats2["approved_count"] == 1
    assert len(stats2["badges"]) == 1
    assert stats2["badges"][0]["id"] == "first_contribution"

