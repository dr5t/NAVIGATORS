"""
Navigators IDR - Community Contributions API Router
Guarded by central AuthorizationService to enforce strict ownership rules:
  - Owners have exclusive rights to read, update, and withdraw their own drafts.
  - Non-owners cannot edit or withdraw others' submissions.
  - Moderators have moderation rights (approve, reject) on pending submissions.
"""

from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.db.database import init_db
from src.db.contributions import ContributionRepository, Contribution
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session, extract_bearer_token

router = APIRouter(prefix="/api/v1/contributions", tags=["contributions"])

init_db()
contrib_repo = ContributionRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


# =============================================================================
# Request Models
# =============================================================================

class CreateContributionRequest(BaseModel):
    resource_type: str = "place"
    title: str
    data: Optional[Dict[str, Any]] = None
    submit_now: bool = False


class UpdateContributionRequest(BaseModel):
    title: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ReviewContributionRequest(BaseModel):
    decision: str  # 'approved' or 'rejected'
    notes: Optional[str] = None


# =============================================================================
# Helper to resolve optional caller session
# =============================================================================

def get_optional_session(authorization: Optional[str] = Header(None)) -> Optional[SessionContext]:
    """Resolve session context if bearer token is provided, otherwise return None."""
    if not authorization:
        return None
    try:
        raw_token = extract_bearer_token(authorization)
        return auth_service.resolve_session(raw_token)
    except HTTPException:
        return None


# =============================================================================
# Contribution Endpoints
# =============================================================================

@router.post("", status_code=201)
def create_contribution(
    req: CreateContributionRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Create a new community contribution.
    Requires an authenticated account with 'contribution:create' permission.
    """
    decision = authz_service.can(user=context, action="contribution:create")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    import secrets
    contrib_id = f"contrib_{secrets.token_hex(6)}"
    owner_id = context.user.id if context.user else "guest_session"
    status = "pending" if req.submit_now else "draft"

    item = contrib_repo.create(
        contribution_id=contrib_id,
        owner_id=owner_id,
        resource_type=req.resource_type,
        title=req.title,
        data=req.data,
        status=status,
    )
    return {
        "message": f"Contribution created in '{status}' status.",
        "contribution": item.to_dict(),
    }


@router.get("/{contrib_id}")
def get_contribution(
    contrib_id: str,
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Read a contribution by ID.
    Enforces draft privacy: Draft contributions are visible exclusively to the author.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:read", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    return {"contribution": item.to_dict()}


@router.patch("/{contrib_id}")
def update_contribution(
    contrib_id: str,
    req: UpdateContributionRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Update contribution title or data payload.
    Ownership Rule: Only the author can update their own contribution.
    State Rule: Only draft or pending contributions can be edited.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:update", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    updated = contrib_repo.update(
        contribution_id=contrib_id,
        title=req.title,
        data=req.data,
    )
    return {
        "message": "Contribution updated successfully.",
        "contribution": updated.to_dict() if updated else None,
    }


@router.delete("/{contrib_id}")
def withdraw_contribution(
    contrib_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Withdraw a contribution.
    Ownership Rule: Only the creator can withdraw their own submission.
    State Rule: Cannot withdraw already approved or rejected contributions.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:withdraw", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    withdrawn = contrib_repo.withdraw(contrib_id)
    return {
        "message": "Contribution withdrawn successfully.",
        "contribution": withdrawn.to_dict() if withdrawn else None,
    }


@router.post("/{contrib_id}/submit")
def submit_contribution(
    contrib_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Transition draft contribution to pending review.
    Ownership Rule: Only the creator can submit their draft.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:update", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    submitted = contrib_repo.submit(contrib_id)
    return {
        "message": "Contribution submitted for moderation review.",
        "contribution": submitted.to_dict() if submitted else None,
    }


@router.post("/{contrib_id}/review")
def review_contribution(
    contrib_id: str,
    req: ReviewContributionRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Review a community submission (approve or reject).
    Moderation Rule: Requires 'contribution:approve' or 'contribution:reject' permission.
    State Rule: Can only review submissions in 'pending' status.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    action = "contribution:approve" if req.decision == "approved" else "contribution:reject"
    decision = authz_service.can(user=context, action=action, resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    moderator_id = context.user.id if context.user else "moderator"
    reviewed = contrib_repo.review(
        contribution_id=contrib_id,
        reviewed_by=moderator_id,
        decision=req.decision,
        review_notes=req.notes,
    )
    return {
        "message": f"Contribution review completed with decision '{req.decision}'.",
        "contribution": reviewed.to_dict() if reviewed else None,
    }


@router.get("")
def list_contributions(
    owner_id: Optional[str] = Query(None, description="Filter by creator user ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    List contributions.
    Drafts belonging to other users are automatically filtered out unless caller is staff.
    """
    items = contrib_repo.list(owner_id=owner_id, status=status, limit=limit, offset=offset)

    # Filter private drafts
    caller_id = context.user.id if (context and context.user) else None
    caller_roles = [r.id for r in context.roles] if (context and context.roles) else []
    is_staff = ("moderator" in caller_roles) or ("team_admin" in caller_roles) or ("super_admin" in caller_roles)

    visible = []
    for it in items:
        if it.status == "draft" and it.owner_id != caller_id and not is_staff:
            continue
        visible.append(it.to_dict())

    return {"contributions": visible, "count": len(visible)}
