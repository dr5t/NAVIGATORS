"""
Navigators IDR - Internal Contributor Approval API Router
Provides endpoints for submitting internal contributor applications,
team admin review queue, approvals with role escalation, and rejections.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.internal_contributors import InternalContributorRepository, InternalContributorRequest
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/internal-contributors", tags=["internal-contributors"])

init_db()
icr_repo = InternalContributorRepository()
auth_service = AuthService()
authz_service = AuthorizationService()






class SubmitInternalRequestModel(BaseModel):
    reason: str = Field(..., min_length=5, description="Applicant motivation and project objectives")
    experience: str = Field(..., min_length=5, description="Sensors, hardware, or navigation experience")
    requested_scope: Optional[str] = Field(default="trajectories_and_models", description="Requested access scope")


class RejectInternalRequestModel(BaseModel):
    rejection_reason: str = Field(..., min_length=3, description="Mandatory rationale for rejection")


def _extract_query_val(val: Any, default: Any = None) -> Any:
    """Extract actual value if val is a FastAPI Query object, or return default."""
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return val if val is not None else default






@router.post("/requests", status_code=201)
def api_submit_internal_request(
    body: SubmitInternalRequestModel,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Submit an application for internal contributor access.
    Gated by internal_contributor:request.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to request internal contributor access")

    user_context = context.to_dict()
    decision = authz_service.can(user=user_context, action="internal_contributor:request", resource="contributor")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    try:
        req = icr_repo.submit_request(
            user_id=context.user.id,
            reason=body.reason,
            experience=body.experience,
            requested_scope=body.requested_scope or "trajectories_and_models",
        )
        return {
            "message": "Internal contributor application submitted successfully.",
            "request": req.to_dict(),
        }
    except ValueError as e:
        err_msg = str(e)
        if "already exists" in err_msg or "already holds" in err_msg:
            raise HTTPException(status_code=409, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)


@router.get("/requests/me")
def api_get_my_latest_request(
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Retrieve the current user's latest internal contributor application status.
    Accessible to any authenticated user.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required")

    req = icr_repo.get_user_latest_request(context.user.id)
    return {"request": req.to_dict() if req else None}


@router.get("/requests")
def api_list_internal_requests(
    status: Optional[str] = Query(None, description="Filter by application status: pending, approved, rejected"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    List internal contributor applications for team admin triage.
    Gated by internal_contributor:review.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="internal_contributor:review", resource="contributor")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    stat = _extract_query_val(status, None)
    lim = int(_extract_query_val(limit, 50))
    off = int(_extract_query_val(offset, 0))

    items, total = icr_repo.list_requests(status=stat, limit=lim, offset=off)

    return {
        "requests": [i.to_dict() for i in items],
        "total": total,
        "limit": lim,
        "offset": off,
    }


@router.get("/requests/{request_id}")
def api_get_internal_request(
    request_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Fetch single application by ID.
    Accessible to team reviewers or the applicant themselves.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required")

    req = icr_repo.get_request(request_id)
    if not req:
        raise HTTPException(status_code=404, detail=f"Application '{request_id}' not found")


    if req.user_id != context.user.id:
        user_context = context.to_dict()
        decision = authz_service.can(user=user_context, action="internal_contributor:review", resource="contributor")
        if not decision.allowed:
            raise HTTPException(status_code=403, detail="Permission denied")

    return req.to_dict()


@router.post("/requests/{request_id}/approve")
def api_approve_internal_request(
    request_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Approve internal contributor application and promote user role.
    Gated by internal_contributor:approve.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="internal_contributor:approve", resource="contributor")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    reviewer_id = context.user.id if context and context.user else "usr_admin"

    try:
        updated = icr_repo.approve_request(request_id=request_id, reviewer_id=reviewer_id)
        return updated.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/requests/{request_id}/reject")
def api_reject_internal_request(
    request_id: str,
    body: RejectInternalRequestModel,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Reject internal contributor application with required rationale.
    Gated by internal_contributor:reject.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="internal_contributor:reject", resource="contributor")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    reviewer_id = context.user.id if context and context.user else "usr_admin"

    try:
        updated = icr_repo.reject_request(
            request_id=request_id,
            reviewer_id=reviewer_id,
            rejection_reason=body.rejection_reason,
        )
        return updated.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
