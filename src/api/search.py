"""
Navigators IDR - Unified Search & Saved Places API Router
Supports top search field queries across Places, Addresses, Coordinates, Saved Places, and Recent Searches.
Handles Online vs Offline operation cleanly with honest offline capability notices.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field

from src.db.database import init_db
from src.db.search import SearchEngine, SavedPlacesRepository, RecentSearchesRepository
from src.db.auth_service import AuthService, SessionContext
from src.api.auth import get_current_session, extract_bearer_token

router = APIRouter(prefix="/api/v1/search", tags=["search"])

init_db()
search_engine = SearchEngine()
saved_repo = SavedPlacesRepository()
recent_repo = RecentSearchesRepository()
auth_service = AuthService()






class SavePlaceRequest(BaseModel):
    name: str = Field(..., min_length=1)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    place_id: Optional[str] = None
    category: str = "landmark"
    address: Optional[str] = None
    notes: Optional[str] = None


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
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return default if val is None else val






@router.get("")
def execute_search(
    q: Optional[str] = Query(None, description="Search query string (place, address, coordinates)"),
    lat: Optional[float] = Query(None, ge=-90.0, le=90.0),
    lon: Optional[float] = Query(None, ge=-180.0, le=180.0),
    radius_km: Optional[float] = Query(None, gt=0),
    is_online: bool = Query(True, description="Whether network is online for web search provider lookup"),
    limit: int = Query(20, ge=1, le=100),
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Unified multi-type search across:
      - Places
      - Addresses
      - Coordinates
      - Saved places
      - Recent searches

    In offline mode (is_online=False), transparently queries local POI / map database
    and returns an explicit capability notice.
    """
    user_id = context.user.id if (context and context.user) else None

    res = search_engine.search(
        query=_extract_query_val(q, "") or "",
        user_id=user_id,
        lat=_extract_query_val(lat),
        lon=_extract_query_val(lon),
        radius_km=_extract_query_val(radius_km),
        is_online=_extract_query_val(is_online, True),
        limit=_extract_query_val(limit, 20),
    )
    return res






@router.get("/saved")
def list_saved_places(
    q: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    context: SessionContext = Depends(get_current_session),
):
    """List saved places / bookmarks belonging to the authenticated user."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to access saved places.")

    items = saved_repo.list_saved_places(
        user_id=context.user.id,
        search=_extract_query_val(q),
        limit=_extract_query_val(limit, 50),
    )
    return {"saved_places": items, "count": len(items)}


@router.post("/saved", status_code=201)
def add_saved_place(
    req: SavePlaceRequest,
    context: SessionContext = Depends(get_current_session),
):
    """Save a place or bookmark to the user's profile."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to save places.")

    item = saved_repo.add_saved_place(
        user_id=context.user.id,
        name=req.name,
        latitude=req.latitude,
        longitude=req.longitude,
        place_id=req.place_id,
        category=req.category,
        address=req.address,
        notes=req.notes,
    )
    return {"message": "Place saved successfully.", "saved_place": item}


@router.delete("/saved/{saved_place_id}")
def delete_saved_place(
    saved_place_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """Remove a saved place / bookmark."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    success = saved_repo.delete_saved_place(context.user.id, saved_place_id)
    if not success:
        raise HTTPException(status_code=404, detail="Saved place not found or not owned by user.")
    return {"message": "Saved place deleted successfully."}






@router.get("/recent")
def list_recent_searches(
    limit: int = Query(10, ge=1, le=50),
    context: SessionContext = Depends(get_current_session),
):
    """Retrieve user's recent search query history."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    items = recent_repo.list_recent_searches(context.user.id, limit=_extract_query_val(limit, 10))
    return {"recent_searches": items, "count": len(items)}


@router.delete("/recent")
def clear_recent_searches(
    context: SessionContext = Depends(get_current_session),
):
    """Clear user's recent search query history."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    recent_repo.clear_recent_searches(context.user.id)
    return {"message": "Recent search history cleared."}
