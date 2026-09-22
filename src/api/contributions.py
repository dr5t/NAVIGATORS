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
from src.db.state_machine import ContributionState, StateTransitionError
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session, extract_bearer_token

router = APIRouter(prefix="/api/v1/contributions", tags=["contributions"])

init_db()
contrib_repo = ContributionRepository()
auth_service = AuthService()
authz_service = AuthorizationService()






class CreateContributionRequest(BaseModel):
    resource_type: str = "place"
    title: str
    data: Optional[Dict[str, Any]] = None
    submit_now: bool = False


class UpdateContributionRequest(BaseModel):
    title: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ReviewContributionRequest(BaseModel):
    decision: str
    notes: Optional[str] = None


class ReviewNotesRequest(BaseModel):
    notes: Optional[str] = None


def handle_transition_error(err: Exception) -> HTTPException:
    """Format StateTransitionError into appropriate HTTP error response."""
    if isinstance(err, StateTransitionError):
        code = err.code
        if code in ("UNAUTHENTICATED",):
            return HTTPException(status_code=401, detail=f"[{code}] {err.message}")
        if code in ("NOT_OWNER", "PERMISSION_DENIED"):
            return HTTPException(status_code=403, detail=f"[{code}] {err.message}")
        return HTTPException(status_code=400, detail=f"[{code}] {err.message}")
    return HTTPException(status_code=400, detail=str(err))






def get_optional_session(authorization: Optional[str] = Header(None)) -> Optional[SessionContext]:
    """Resolve session context if bearer token is provided, otherwise return None."""
    if not authorization:
        return None
    try:
        raw_token = extract_bearer_token(authorization)
        return auth_service.resolve_session(raw_token)
    except HTTPException:
        return None






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


@router.get("/stats")
def get_contributor_stats_endpoint(
    context: SessionContext = Depends(get_current_session),
):
    """
    Fetch contributor points, level, and badges calculated strictly from real database records.
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return contrib_repo.get_contributor_stats(context.user.id)


@router.get("/my")
def get_my_contributions_endpoint(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    context: SessionContext = Depends(get_current_session),
):
    """
    List contributions belonging to the authenticated caller.
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    items = contrib_repo.list(owner_id=context.user.id, status=status, limit=limit, offset=offset)
    return {
        "count": len(items),
        "items": [item.to_dict() for item in items],
    }


@router.get("/user/{target_user_id}")
def get_user_contributions_endpoint(
    target_user_id: str,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    List contributions for a specific target user ID.
    IDOR & Privacy Protection:
      - If User A calls GET /api/v1/contributions/user/USER_B_ID, drafts and non-public
        unapproved contributions belonging to User B are hidden unless caller is staff.
    """
    items = contrib_repo.list(owner_id=target_user_id, status=status, limit=limit, offset=offset)

    caller_id = context.user.id if (context and context.user) else None
    caller_roles = [r.id for r in context.roles] if (context and context.roles) else []
    is_staff = ("moderator" in caller_roles) or ("team_admin" in caller_roles) or ("super_admin" in caller_roles)

    visible = []
    for it in items:

        if it.status == "draft" and it.owner_id != caller_id and not is_staff:
            continue
        visible.append(it.to_dict())

    return {"contributions": visible, "count": len(visible)}



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
    if not updated:
        raise HTTPException(status_code=500, detail="Contribution update failed.")
    return {
        "message": "Contribution updated successfully.",
        "contribution": updated.to_dict(),
    }


@router.delete("/{contrib_id}")
@router.post("/{contrib_id}/withdraw")
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

    try:
        withdrawn = contrib_repo.withdraw(contrib_id, user=context)
    except Exception as e:
        raise handle_transition_error(e)

    if not withdrawn:
        raise HTTPException(status_code=500, detail="Contribution withdraw failed: record not found after transition.")
    return {
        "message": "Contribution withdrawn successfully.",
        "contribution": withdrawn.to_dict(),
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

    try:
        submitted = contrib_repo.submit(contrib_id, user=context)
    except Exception as e:
        raise handle_transition_error(e)

    if not submitted:
        raise HTTPException(status_code=500, detail="Contribution submit failed: record not found after transition.")
    return {
        "message": "Contribution submitted for moderation review.",
        "contribution": submitted.to_dict(),
    }


@router.post("/{contrib_id}/approve")
def approve_contribution(
    contrib_id: str,
    req: Optional[ReviewNotesRequest | ReviewContributionRequest] = None,
    context: SessionContext = Depends(get_current_session),
):
    """
    Reviewer approves contribution for map inclusion.
    Requires moderator role and 'contribution:approve' permission.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:approve", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    notes = req.notes if req else None
    try:
        approved = contrib_repo.approve(contrib_id, reviewer=context, notes=notes)
    except Exception as e:
        raise handle_transition_error(e)

    if not approved:
        raise HTTPException(status_code=500, detail="Contribution approve failed: record not found after transition.")
    return {
        "message": "Contribution approved successfully.",
        "contribution": approved.to_dict(),
    }


@router.post("/{contrib_id}/reject")
def reject_contribution(
    contrib_id: str,
    req: Optional[ReviewNotesRequest | ReviewContributionRequest] = None,
    context: SessionContext = Depends(get_current_session),
):
    """
    Reviewer rejects contribution with rationale notes.
    Requires moderator role and 'contribution:reject' permission.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="contribution:reject", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    notes = req.notes if req else None
    try:
        rejected = contrib_repo.reject(contrib_id, reviewer=context, notes=notes)
    except Exception as e:
        raise handle_transition_error(e)

    if not rejected:
        raise HTTPException(status_code=500, detail="Contribution reject failed: record not found after transition.")
    return {
        "message": "Contribution rejected with review notes.",
        "contribution": rejected.to_dict(),
    }


@router.post("/{contrib_id}/publish")
def publish_contribution(
    contrib_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Synchronizes approved contribution into canonical live map data.
    Requires staff role and 'place:create' permission.
    """
    item = contrib_repo.get(contrib_id)
    if not item:
        raise HTTPException(status_code=404, detail="Contribution not found.")

    decision = authz_service.can(user=context, action="place:create", resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    try:
        published = contrib_repo.publish(contrib_id, staff=context)
    except Exception as e:
        raise handle_transition_error(e)

    if not published:
        raise HTTPException(status_code=500, detail="Contribution publish failed: record not found after transition.")
    return {
        "message": "Contribution published to canonical map dataset.",
        "contribution": published.to_dict(),
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

    action = "contribution:approve" if req.decision.lower() in ("approved", "approve") else "contribution:reject"
    decision = authz_service.can(user=context, action=action, resource=item)
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    try:
        reviewed = contrib_repo.review(
            contribution_id=contrib_id,
            reviewed_by=context,
            decision=req.decision,
            review_notes=req.notes,
        )
    except Exception as e:
        raise handle_transition_error(e)

    if not reviewed:
        raise HTTPException(status_code=500, detail="Contribution review failed: record not found after transition.")
    return {
        "message": f"Contribution review completed with decision '{req.decision}'.",
        "contribution": reviewed.to_dict(),
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


    caller_id = context.user.id if (context and context.user) else None
    caller_roles = [r.id for r in context.roles] if (context and context.roles) else []
    is_staff = ("moderator" in caller_roles) or ("team_admin" in caller_roles) or ("super_admin" in caller_roles)

    visible = []
    for it in items:
        if it.status == "draft" and it.owner_id != caller_id and not is_staff:
            continue
        visible.append(it.to_dict())

    return {"contributions": visible, "count": len(visible)}
