"""
Navigators IDR - Dataset Sessions API Router (Phase 13)
Provides endpoints for submitting dataset sessions, managing the validation
lifecycle, and exposing aggregate stats to the Mac Dashboard.

Access control:
  - dataset:create  → submit a session (internal_contributor+)
  - dataset:read    → list / fetch sessions (internal_contributor+)
  - dataset:validate→ start_validation, validate, reject (team_admin+)
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Any

from src.db.database import init_db
from src.db.datasets import DatasetRepository, ACTIVITY_TYPES
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

init_db()
dataset_repo = DatasetRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


# =============================================================================
# Request Models
# =============================================================================

class SubmitSessionModel(BaseModel):
    activity_type: str = Field(
        ...,
        description=f"Session type. One of: {sorted(ACTIVITY_TYPES)}",
    )
    device: str = Field(..., min_length=2, description="Device model/identifier used for recording")
    duration_seconds: float = Field(0.0, ge=0.0, description="Recording duration in seconds")
    gnss_available: bool = Field(True, description="Whether GNSS was available during the session")
    consent: bool = Field(..., description="Explicit contributor consent for data use (required)")
    sensor_data_path: Optional[str] = Field(None, description="Relative path to raw sensor data file")
    notes: Optional[str] = Field(None, description="Optional notes about recording conditions")


class RejectSessionModel(BaseModel):
    rejection_reason: str = Field(..., min_length=5, description="Mandatory reason for rejecting this session")


def _extract_query_val(val: Any, default: Any = None) -> Any:
    """Extract actual value if val is a FastAPI Query object, or return default."""
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return val if val is not None else default


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/sessions", status_code=201)
def api_submit_session(
    body: SubmitSessionModel,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Submit a new dataset session for validation.
    Gated by dataset:create. Status begins as 'uploaded'.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to submit dataset sessions")

    user_context = context.to_dict()
    decision = authz_service.can(user=user_context, action="dataset:create", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    try:
        session = dataset_repo.submit_session(
            contributor_id=context.user.id,
            activity_type=body.activity_type,
            device=body.device,
            duration_seconds=body.duration_seconds,
            gnss_available=body.gnss_available,
            consent=body.consent,
            sensor_data_path=body.sensor_data_path,
            notes=body.notes,
        )
        return {
            "message": "Dataset session submitted successfully. Awaiting validation.",
            "session": session.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sessions/stats")
def api_get_session_stats(
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Aggregate counts by status and activity type for the Mac Dashboard.
    Gated by dataset:read.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="dataset:read", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    return dataset_repo.get_stats()


@router.get("/sessions")
def api_list_sessions(
    status: Optional[str] = Query(None, description="Filter by status: uploaded, validating, validated, rejected"),
    activity_type: Optional[str] = Query(None, description="Filter by activity type"),
    contributor_id: Optional[str] = Query(None, description="Filter by contributor (team_admin only)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    List dataset sessions.
    internal_contributor sees only their own. team_admin sees all.
    Gated by dataset:read.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_context = context.to_dict()
    decision = authz_service.can(user=user_context, action="dataset:read", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    # Scope: non-admins can only see their own sessions
    can_validate = authz_service.can(user=user_context, action="dataset:validate", resource="dataset")
    effective_contributor_id = (
        _extract_query_val(contributor_id, None)
        if can_validate.allowed
        else context.user.id       # restrict to own sessions
    )

    stat = _extract_query_val(status, None)
    act  = _extract_query_val(activity_type, None)
    lim  = int(_extract_query_val(limit, 50))
    off  = int(_extract_query_val(offset, 0))

    items, total = dataset_repo.list_sessions(
        status=stat,
        contributor_id=effective_contributor_id,
        activity_type=act,
        limit=lim,
        offset=off,
    )

    return {
        "sessions": [s.to_dict() for s in items],
        "total": total,
        "limit": lim,
        "offset": off,
    }


@router.get("/sessions/{session_id}")
def api_get_session(
    session_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Fetch a single session by ID.
    Owner or team_admin can access.
    Gated by dataset:read.
    """
    if not context or not context.user:
        raise HTTPException(status_code=401, detail="Authentication required")

    session = dataset_repo.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Dataset session '{session_id}' not found")

    # Access: owner always allowed; others need dataset:validate
    if session.contributor_id != context.user.id:
        user_context = context.to_dict()
        decision = authz_service.can(user=user_context, action="dataset:validate", resource="dataset")
        if not decision.allowed:
            raise HTTPException(status_code=403, detail="Permission denied")

    return {"session": session.to_dict()}


@router.post("/sessions/{session_id}/start-validation")
def api_start_validation(
    session_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Transition uploaded → validating.
    Gated by dataset:validate (team_admin+).
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="dataset:validate", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    validator_id = context.user.id if context and context.user else "usr_admin"
    try:
        updated = dataset_repo.start_validation(session_id=session_id, validator_id=validator_id)
        return {"session": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sessions/{session_id}/validate")
def api_validate_session(
    session_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Transition validating → validated.
    Makes session eligible for training. Gated by dataset:validate.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="dataset:validate", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    validator_id = context.user.id if context and context.user else "usr_admin"
    try:
        updated = dataset_repo.validate_session(session_id=session_id, validator_id=validator_id)
        return {
            "message": "Session validated and eligible for training.",
            "session": updated.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sessions/{session_id}/reject")
def api_reject_session(
    session_id: str,
    body: RejectSessionModel,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Transition uploaded|validating → rejected.
    Rejection reason is mandatory. Gated by dataset:validate.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="dataset:validate", resource="dataset")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    validator_id = context.user.id if context and context.user else "usr_admin"
    try:
        updated = dataset_repo.reject_session(
            session_id=session_id,
            validator_id=validator_id,
            rejection_reason=body.rejection_reason,
        )
        return {"session": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
