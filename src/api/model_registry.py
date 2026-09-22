"""
Navigators IDR - Model Registry API Router (Phase 15)
Provides endpoints for listing model candidates, retrieving production models,
opening candidates for review, approving, rejecting, and deploying models to production.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.model_registry import ModelRegistryRepository, ModelEntry
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/models", tags=["models"])

init_db()
repo = ModelRegistryRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


try:
    repo.seed_production_model()
except Exception:
    pass




class ApproveModelRequest(BaseModel):
    notes: Optional[str] = Field(None, description="Optional approval notes or review summary")


class RejectModelRequest(BaseModel):
    rejection_reason: str = Field(..., min_length=3, description="Mandatory reason for model candidate rejection")




def _require_permission(context: Optional[SessionContext], permission: str) -> str:
    """Helper to enforce authentication and authorization."""
    if not context or not context.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    decision = authz_service.can(user=context.to_dict(), action=permission, resource="model")
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {decision.reason}"
        )
    return context.user.id




@router.get("", response_model=Dict[str, Any])
@router.get("/", response_model=Dict[str, Any])
def list_models(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List model entries in registry with optional status filtering and pagination."""
    entries, total = repo.list_models(status=status, limit=limit, offset=offset)
    return {
        "models": [e.to_dict() for e in entries],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/production", response_model=Dict[str, Any])
def get_production_model():
    """Get the current active production model."""
    prod = repo.get_production_model()
    return {
        "model": prod.to_dict() if prod else None
    }


@router.get("/{model_id}", response_model=Dict[str, Any])
def get_model(model_id: str):
    """Get detailed model registry entry by ID."""
    entry = repo.get_model(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {"model": entry.to_dict()}


@router.post("/{model_id}/open-review", response_model=Dict[str, Any])
def open_model_review(
    model_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Transition model candidate evaluating → review.
    Requires permission: model:review
    """
    user_id = _require_permission(context, "model:review")
    try:
        updated = repo.open_for_review(model_id, reviewer_id=user_id)
        return {"status": "success", "model": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/approve", response_model=Dict[str, Any])
def approve_model(
    model_id: str,
    body: ApproveModelRequest,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Approve candidate and transition review → production_candidate.
    Requires permission: model:approve
    """
    user_id = _require_permission(context, "model:approve")
    try:
        updated = repo.approve_model(model_id, reviewer_id=user_id, notes=body.notes)
        return {"status": "success", "model": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/reject", response_model=Dict[str, Any])
def reject_model(
    model_id: str,
    body: RejectModelRequest,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Reject candidate and transition review → rejected.
    Requires permission: model:review
    """
    user_id = _require_permission(context, "model:review")
    try:
        updated = repo.reject_model(model_id, reviewer_id=user_id, reason=body.rejection_reason)
        return {"status": "success", "model": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/deploy", response_model=Dict[str, Any])
def deploy_model(
    model_id: str,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Deploy model candidate to production and transition production_candidate → production.
    Atomically updates production model files.
    Requires permission: model:deploy
    """
    user_id = _require_permission(context, "model:deploy")
    try:
        updated = repo.deploy(model_id, deployer_id=user_id)
        return {"status": "success", "model": updated.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
