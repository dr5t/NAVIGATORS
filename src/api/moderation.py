"""
Navigators IDR - Moderation & Audit Interface API Router
Provides comprehensive moderation workflow for reviewing community submissions:
  - Triage Queues: Pending, Approved, Rejected, Reported
  - Detailed Cards: Place, Location, Submitted By, Created, Evidence, Changes (diff)
  - Tri-State Actions: Approve, Reject, Request Changes
  - Audit Trail: Immutable logging for every action and state mutation
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.contributions import ContributionRepository, Contribution
from src.db.state_machine import ContributionState, StateTransitionError
from src.db.reports import ReportRepository, Report
from src.db.audit import AuditRepository, AuditEntry
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/moderation", tags=["moderation"])

init_db()
contrib_repo = ContributionRepository()
report_repo = ReportRepository()
audit_repo = AuditRepository()
rbac_repo = RBACRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


# =============================================================================
# Request Models
# =============================================================================

class ApproveContributionRequest(BaseModel):
    notes: Optional[str] = Field(None, description="Optional reviewer approval rationale")


class RejectContributionRequest(BaseModel):
    reason: str = Field(..., min_length=3, description="Mandatory rejection rationale")


class RequestChangesRequest(BaseModel):
    notes: str = Field(..., min_length=3, description="Detailed revision guidance for author")


class ResolveReportRequest(BaseModel):
    decision: str = Field(..., description="'resolved' or 'dismissed'")
    notes: Optional[str] = Field(None, description="Resolution rationale")


def _extract_query_val(val: Any, default: Any = None) -> Any:
    """Extract actual value if val is a FastAPI Query object, or return default."""
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return default if val is None else val


def _enrich_contribution(c: Contribution) -> Dict[str, Any]:
    """Enrich contribution model with author profile, location, evidence, and structured diff."""
    base = c.to_dict()
    data = base.get("data", {})

    # 1. Author profile
    author_info = {"id": c.owner_id, "name": "Unknown", "email": ""}
    try:
        user = rbac_repo.get_user(c.owner_id)
        if user:
            author_info["name"] = user.name
            author_info["email"] = user.email
            author_info["avatar"] = user.avatar
            author_info["roles"] = [r.name for r in rbac_repo.get_user_roles(user.id)]
    except Exception:
        pass

    # 2. Location
    location = {
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "address": data.get("address"),
    }

    # 3. Evidence
    evidence = {
        "notes": data.get("notes") or c.review_notes,
        "signage_verified": data.get("signage_verified", False),
        "source": data.get("source", "community_submission"),
        "phone": data.get("phone"),
        "website": data.get("website"),
        "opening_hours": data.get("opening_hours"),
    }

    # 4. Changes diff
    diff = contrib_repo.get_diff_summary(c.id)

    return {
        **base,
        "author": author_info,
        "location": location,
        "evidence": evidence,
        "changes": diff.get("changes", {}),
    }


# =============================================================================
# 1. Contributions Triage Queues (Pending, Approved, Rejected, Changes Requested)
# =============================================================================

@router.get("/contributions")
def list_moderation_queue(
    status: Optional[str] = Query("pending", description="Queue status: pending, approved, rejected, changes_requested, or all"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: SessionContext = Depends(get_current_session),
):
    """
    Moderator triage queue:
      - Pending: Awaiting review
      - Approved: Reviewed and accepted
      - Rejected: Reviewed and rejected
      - Changes Requested: Sent back to author for revisions
    """
    decision = authz_service.can(user=context, action="contribution:approve")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    status_val = _extract_query_val(status, "pending")
    lim = _extract_query_val(limit, 50)
    off = _extract_query_val(offset, 0)

    items: List[Contribution] = []
    if status_val == "all":
        items = contrib_repo.list(limit=lim, offset=off)
    elif status_val == "pending":
        p_items = contrib_repo.list(status=ContributionState.PENDING_REVIEW, limit=lim, offset=off)
        s_items = contrib_repo.list(status=ContributionState.SUBMITTED, limit=lim, offset=off)
        items = p_items + s_items
    elif status_val in ("approved", "rejected", "changes_requested", "published", "draft"):
        items = contrib_repo.list(status=status_val, limit=lim, offset=off)
    else:
        items = contrib_repo.list(status=status_val, limit=lim, offset=off)

    enriched = [_enrich_contribution(c) for c in items]
    return {
        "status_filter": status_val,
        "count": len(enriched),
        "items": enriched,
    }


@router.get("/contributions/{contrib_id}")
def get_moderation_contribution_detail(
    contrib_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """Inspect detailed contribution card with diff and evidence for moderation triage."""
    decision = authz_service.can(user=context, action="contribution:approve")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    c = contrib_repo.get(contrib_id)
    if not c:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    return {
        "item": _enrich_contribution(c),
    }


# =============================================================================
# 2. Moderation Actions: Approve, Reject, Request Changes
# =============================================================================

@router.post("/contributions/{contrib_id}/approve")
def approve_contribution(
    contrib_id: str,
    req: Optional[ApproveContributionRequest] = None,
    context: SessionContext = Depends(get_current_session),
):
    """
    Moderator approves community contribution.
    Records audit record and transitions status to 'approved'.
    """
    decision = authz_service.can(user=context, action="contribution:approve")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    c = contrib_repo.get(contrib_id)
    if not c:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    notes = req.notes if req else None
    try:
        updated = contrib_repo.approve(contrib_id, reviewer=context, notes=notes)
    except StateTransitionError as err:
        code = err.code
        if code in ("UNAUTHENTICATED",):
            raise HTTPException(status_code=401, detail=err.message)
        if code in ("NOT_OWNER", "PERMISSION_DENIED"):
            raise HTTPException(status_code=403, detail=err.message)
        raise HTTPException(status_code=400, detail=err.message)

    return {
        "message": "Contribution approved successfully.",
        "item": _enrich_contribution(updated),
    }


@router.post("/contributions/{contrib_id}/reject")
def reject_contribution(
    contrib_id: str,
    req: RejectContributionRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Moderator rejects community contribution with required rationale.
    Records audit record and transitions status to 'rejected'.
    """
    decision = authz_service.can(user=context, action="contribution:reject")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    c = contrib_repo.get(contrib_id)
    if not c:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    try:
        updated = contrib_repo.reject(contrib_id, reviewer=context, notes=req.reason)
    except StateTransitionError as err:
        code = err.code
        if code in ("UNAUTHENTICATED",):
            raise HTTPException(status_code=401, detail=err.message)
        if code in ("NOT_OWNER", "PERMISSION_DENIED"):
            raise HTTPException(status_code=403, detail=err.message)
        raise HTTPException(status_code=400, detail=err.message)

    return {
        "message": "Contribution rejected.",
        "item": _enrich_contribution(updated),
    }


@router.post("/contributions/{contrib_id}/request-changes")
def request_changes_on_contribution(
    contrib_id: str,
    req: RequestChangesRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Moderator requests revisions on contribution with feedback notes.
    Transitions status to 'changes_requested' and records audit record.
    Author can revise draft payload and re-submit for review.
    """
    decision = authz_service.can(user=context, action="contribution:request_changes")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    c = contrib_repo.get(contrib_id)
    if not c:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    try:
        updated = contrib_repo.request_changes(contrib_id, reviewer=context, notes=req.notes)
    except StateTransitionError as err:
        code = err.code
        if code in ("UNAUTHENTICATED",):
            raise HTTPException(status_code=401, detail=err.message)
        if code in ("NOT_OWNER", "PERMISSION_DENIED"):
            raise HTTPException(status_code=403, detail=err.message)
        raise HTTPException(status_code=400, detail=err.message)

    return {
        "message": "Changes requested from author.",
        "item": _enrich_contribution(updated),
    }


# =============================================================================
# 3. Community Reports Triage & Resolution
# =============================================================================

@router.get("/reports")
def list_reported_items(
    status: Optional[str] = Query("pending", description="Report status: pending, resolved, dismissed, all"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: SessionContext = Depends(get_current_session),
):
    """Moderator view of user-reported issues and flags on map data and contributions."""
    decision = authz_service.can(user=context, action="report:read")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    status_val = _extract_query_val(status, "pending")
    status_filter = None if status_val == "all" else status_val
    lim = _extract_query_val(limit, 50)
    off = _extract_query_val(offset, 0)

    reports = report_repo.list_reports(status=status_filter, limit=lim, offset=off)
    results = []
    for rep in reports:
        rep_dict = rep.to_dict()
        try:
            reporter = rbac_repo.get_user(rep.reporter_id)
            rep_dict["reporter"] = {"name": reporter.name, "email": reporter.email} if reporter else {"name": "User"}
        except Exception:
            rep_dict["reporter"] = {"name": "User"}
        results.append(rep_dict)

    return {
        "status_filter": status_val,
        "count": len(results),
        "reports": results,
    }


@router.post("/reports/{report_id}/resolve")
def resolve_report(
    report_id: str,
    req: ResolveReportRequest,
    context: SessionContext = Depends(get_current_session),
):
    """Moderator resolves or dismisses a reported community issue."""
    decision = authz_service.can(user=context, action="report:resolve")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    rep = report_repo.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")

    reviewer_id = context.user.id if context.user else "moderator"
    resolved = report_repo.resolve_report(
        report_id=report_id,
        resolved_by=reviewer_id,
        decision=req.decision,
        notes=req.notes,
    )

    # Record audit log
    try:
        audit_repo.log(
            action=f"report:{req.decision}",
            resource_type="report",
            resource_id=report_id,
            actor_id=reviewer_id,
            old_state=rep.status,
            new_state=req.decision,
            metadata={"notes": req.notes, "target_type": rep.target_type, "target_id": rep.target_id},
        )
    except Exception:
        pass

    if not resolved:
        raise HTTPException(status_code=500, detail="Report resolution failed: record not found after update.")
    return {
        "message": f"Report marked as {req.decision}.",
        "report": resolved.to_dict(),
    }


# =============================================================================
# 4. Immutable Audit Logs Inspection
# =============================================================================

@router.get("/audit-logs")
def list_audit_trail(
    actor_id: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: SessionContext = Depends(get_current_session),
):
    """Query immutable platform audit logs for moderation and governance inspection."""
    decision = authz_service.can(user=context, action="audit:read")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    lim = _extract_query_val(limit, 50)
    off = _extract_query_val(offset, 0)
    act_id = _extract_query_val(actor_id)
    res_type = _extract_query_val(resource_type)
    res_id = _extract_query_val(resource_id)
    act = _extract_query_val(action)

    logs = audit_repo.list_logs(
        actor_id=act_id,
        resource_type=res_type,
        resource_id=res_id,
        action=act,
        limit=lim,
        offset=off,
    )
    total = audit_repo.count_logs(actor_id=act_id, resource_type=res_type, action=act)

    return {
        "count": len(logs),
        "total": total,
        "audit_logs": [entry.to_dict() for entry in logs],
    }
