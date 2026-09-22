"""
Navigators IDR - Community Places & Canonical POI API Router
Manages canonical map places, versioned audit history, soft delete / archiving,
and community contribution workflows:
  - User: Add Place (creates DRAFT contribution)
  - User: "My Contributions" view
  - Moderator: "Pending Contributions" triage view
  - Public users: Read PUBLISHED canonical places only
  - Owner: Update draft, withdraw pending
  - Moderator: Review, edit pending, direct canonical place updates
  - Normal users: Suggest Edit on published places (creates linked contribution)
  - Staff: Soft delete / archive canonical places (never physical casual delete)
"""

import secrets
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field

from src.db.database import init_db
from src.db.places import PlaceRepository, Place, PlaceHistory, POI_TAXONOMY, CATEGORY_ALIASES
from src.db.contributions import ContributionRepository, Contribution
from src.db.state_machine import ContributionState
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session, extract_bearer_token

router = APIRouter(prefix="/api/v1/places", tags=["places"])

init_db()
place_repo = PlaceRepository()
contrib_repo = ContributionRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


# =============================================================================
# Request Models
# =============================================================================

class CreatePlaceRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Place or amenity name")
    category: str = Field(..., description="Category (e.g. fuel, hospital, ev_charging, atm, pharmacy)")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    address: Optional[str] = None
    opening_hours: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    submit_now: bool = False


class SuggestEditRequest(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    opening_hours: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    submit_now: bool = True


class DirectUpdatePlaceRequest(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    opening_hours: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    change_summary: Optional[str] = None


class DeletePlaceRequest(BaseModel):
    reason: Optional[str] = None


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


def _extract_query_val(val: Any, default: Any = None) -> Any:
    """Extract actual value if val is a FastAPI Query object, or return default."""
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return default if val is None else val


# =============================================================================
# 1. Place Creation & Contribution Flow
# =============================================================================

@router.post("", status_code=201)
def add_place(
    req: CreatePlaceRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    User adds a new place to the map.
    Backend creates a place contribution in 'draft' (or 'pending_review') status.
    Requires 'contribution:create' permission.
    """
    decision = authz_service.can(user=context, action="contribution:create")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    owner_id = context.user.id if context.user else "guest_session"
    contrib_id = f"contrib_plc_{secrets.token_hex(6)}"
    initial_status = ContributionState.PENDING_REVIEW if req.submit_now else ContributionState.DRAFT

    data_payload = {
        "name": req.name.strip(),
        "category": req.category.strip().lower(),
        "latitude": req.latitude,
        "longitude": req.longitude,
        "address": req.address.strip() if req.address else None,
        "opening_hours": req.opening_hours.strip() if req.opening_hours else None,
        "phone": req.phone.strip() if req.phone else None,
        "website": req.website.strip() if req.website else None,
        "metadata": req.metadata or {},
    }

    item = contrib_repo.create(
        contribution_id=contrib_id,
        owner_id=owner_id,
        resource_type="place",
        title=req.name.strip(),
        data=data_payload,
        status=initial_status,
        target_resource_id=None,
        action="create",
    )
    return {
        "message": f"Place contribution created in '{initial_status}' status.",
        "contribution": item.to_dict(),
    }


# =============================================================================
# 2. Public Canonical Places Reading & POI Taxonomy
# =============================================================================

@router.get("/poi/categories")
def get_poi_categories():
    """Expose official canonical POI taxonomy categories and aliases."""
    return {
        "categories": POI_TAXONOMY,
        "aliases": CATEGORY_ALIASES,
    }


@router.get("")
def list_canonical_places(
    category: Optional[str] = Query(None, description="Filter by place category (e.g. fuel, hospital, atm)"),
    q: Optional[str] = Query(None, description="Search term matching place name or address"),
    min_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    max_lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    min_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    max_lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    lat: Optional[float] = Query(None, ge=-90.0, le=90.0, description="Center latitude for proximity search"),
    lon: Optional[float] = Query(None, ge=-180.0, le=180.0, description="Center longitude for proximity search"),
    radius_km: Optional[float] = Query(None, gt=0, description="Proximity search radius in kilometers"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Query canonical map places.
    Publicly accessible to all users (including guests).
    Returns ONLY active published places (excludes soft-deleted and unapproved items).
    Supports category taxonomy, text search, bounding box, and proximity radius search.
    """
    decision = authz_service.can(user=context, action="place:read")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    places = place_repo.list_places(
        category=_extract_query_val(category),
        search=_extract_query_val(q),
        min_lat=_extract_query_val(min_lat),
        max_lat=_extract_query_val(max_lat),
        min_lon=_extract_query_val(min_lon),
        max_lon=_extract_query_val(max_lon),
        lat=_extract_query_val(lat),
        lon=_extract_query_val(lon),
        radius_km=_extract_query_val(radius_km),
        limit=_extract_query_val(limit, 50),
        offset=_extract_query_val(offset, 0),
    )
    return {
        "places": [p.to_dict() for p in places],
        "count": len(places),
    }


# =============================================================================
# 3. Community Views: My Contributions & Pending Triage
# =============================================================================

@router.get("/contributions/my")
def get_my_contributions(
    context: SessionContext = Depends(get_current_session),
):
    """
    User view: 'My Contributions'.
    Returns all contributions created by the current user across all lifecycle states
    (draft, submitted, pending_review, approved, published, rejected, withdrawn).
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to view your contributions.")

    items = contrib_repo.list(owner_id=context.user.id, limit=100)
    return {
        "contributions": [it.to_dict() for it in items],
        "count": len(items),
    }


@router.get("/contributions/pending")
def get_pending_contributions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    context: SessionContext = Depends(get_current_session),
):
    """
    Moderator view: 'Pending Contributions'.
    Returns community contributions awaiting moderation review.
    Requires moderator role or 'contribution:approve' permission.
    """
    decision = authz_service.can(user=context, action="contribution:approve")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    lim = _extract_query_val(limit, 50)
    off = _extract_query_val(offset, 0)
    pending_items = contrib_repo.list(status=ContributionState.PENDING_REVIEW, limit=lim, offset=off)
    # Also include 'submitted' if any
    submitted_items = contrib_repo.list(status=ContributionState.SUBMITTED, limit=lim, offset=off)
    all_pending = pending_items + submitted_items

    return {
        "pending_contributions": [it.to_dict() for it in all_pending],
        "count": len(all_pending),
    }


# =============================================================================
# 4. Canonical Place Details & Version History
# =============================================================================

@router.get("/{place_id}")
def get_place_detail(
    place_id: str,
    include_deleted: bool = False,
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Read canonical place details.
    Publicly accessible. Soft-deleted places are hidden unless requested by staff.
    """
    decision = authz_service.can(user=context, action="place:read")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    place = place_repo.get_place(place_id, include_deleted=include_deleted)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found or has been archived.")

    return {"place": place.to_dict()}


@router.get("/{place_id}/history")
def get_place_version_history(
    place_id: str,
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Read immutable version history snapshots for a canonical place.
    """
    decision = authz_service.can(user=context, action="place:read")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    place = place_repo.get_place(place_id, include_deleted=True)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found.")

    history = place_repo.get_place_history(place_id)
    return {
        "place_id": place_id,
        "history": [h.to_dict() for h in history],
        "count": len(history),
    }


# =============================================================================
# 5. Suggest Edit (Normal User Workflow for Published Places)
# =============================================================================

@router.post("/{place_id}/suggest-edit", status_code=201)
def suggest_edit_place(
    place_id: str,
    req: SuggestEditRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Normal users cannot directly update published canonical places.
    Instead, 'Suggest Edit' creates a new contribution linked to the canonical place.
    The proposal enters the community moderation review workflow.
    """
    decision = authz_service.can(user=context, action="contribution:create")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    place = place_repo.get_place(place_id, include_deleted=False)
    if not place:
        raise HTTPException(status_code=404, detail="Target place not found or has been archived.")

    owner_id = context.user.id if context.user else "guest_session"
    contrib_id = f"contrib_edit_{secrets.token_hex(6)}"
    initial_status = ContributionState.PENDING_REVIEW if req.submit_now else ContributionState.DRAFT

    data_payload: Dict[str, Any] = {"target_place_id": place_id}
    if req.name is not None:
        data_payload["name"] = req.name.strip()
    if req.category is not None:
        data_payload["category"] = req.category.strip().lower()
    if req.latitude is not None:
        data_payload["latitude"] = req.latitude
    if req.longitude is not None:
        data_payload["longitude"] = req.longitude
    if req.address is not None:
        data_payload["address"] = req.address.strip()
    if req.opening_hours is not None:
        data_payload["opening_hours"] = req.opening_hours.strip()
    if req.phone is not None:
        data_payload["phone"] = req.phone.strip()
    if req.website is not None:
        data_payload["website"] = req.website.strip()
    if req.metadata is not None:
        data_payload["metadata"] = req.metadata
    if req.notes:
        data_payload["notes"] = req.notes.strip()

    contrib = contrib_repo.create(
        contribution_id=contrib_id,
        owner_id=owner_id,
        resource_type="place",
        title=f"Edit: {place.name}",
        data=data_payload,
        status=initial_status,
        target_resource_id=place_id,
        action="update",
    )
    return {
        "message": f"Suggested edit created in '{initial_status}' status.",
        "contribution": contrib.to_dict(),
    }


# =============================================================================
# 6. Direct Staff Updates & Soft Delete / Archiving
# =============================================================================

@router.patch("/{place_id}")
def direct_update_place(
    place_id: str,
    req: DirectUpdatePlaceRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Direct update on published canonical place.
    Restricted to staff with 'place:update' permission (moderator / admin).
    Normal contributors must use the 'Suggest Edit' workflow instead.
    """
    decision = authz_service.can(user=context, action="place:update")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(
            status_code=status_code,
            detail="Normal contributors cannot directly modify canonical map records. Use 'Suggest Edit' to submit changes for review.",
        )

    place = place_repo.get_place(place_id, include_deleted=False)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found or has been archived.")

    changed_by = context.user.id if context.user else "staff"
    updated = place_repo.update_place(
        place_id=place_id,
        changed_by=changed_by,
        name=req.name,
        category=req.category,
        latitude=req.latitude,
        longitude=req.longitude,
        address=req.address,
        opening_hours=req.opening_hours,
        phone=req.phone,
        website=req.website,
        metadata=req.metadata,
        change_summary=req.change_summary or "Direct staff update",
    )
    return {
        "message": "Canonical place updated successfully.",
        "place": updated.to_dict(),
    }


@router.delete("/{place_id}")
def soft_delete_place(
    place_id: str,
    req: Optional[DeletePlaceRequest] = None,
    context: SessionContext = Depends(get_current_session),
):
    """
    Soft delete / archive a canonical place.
    Canonical map records are NEVER physically deleted casually.
    Sets is_deleted = 1 and status = 'archived', recording full version history.
    Requires 'place:delete' permission (moderator / admin).
    """
    decision = authz_service.can(user=context, action="place:delete")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    place = place_repo.get_place(place_id, include_deleted=False)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found or already archived.")

    changed_by = context.user.id if context.user else "staff"
    reason = req.reason if req else "Archived per moderation review"
    archived = place_repo.soft_delete_place(place_id, changed_by=changed_by, reason=reason)

    return {
        "message": "Canonical place archived and soft-deleted. Record preserved in audit history.",
        "place": archived.to_dict(),
    }


@router.post("/{place_id}/restore")
def restore_place(
    place_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Restore an archived / soft-deleted place to active canonical status.
    Requires staff role ('place:update' or 'place:delete').
    """
    decision = authz_service.can(user=context, action="place:update")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    place = place_repo.get_place(place_id, include_deleted=True)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found.")

    changed_by = context.user.id if context.user else "staff"
    restored = place_repo.restore_place(place_id, changed_by=changed_by)

    return {
        "message": "Place restored to active published status.",
        "place": restored.to_dict(),
    }
