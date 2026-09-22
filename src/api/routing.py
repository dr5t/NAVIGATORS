"""
Navigators IDR — Turn-by-Turn Routing API Router (Phase 31)
Exposes the turn-by-turn routing engine calculating recommended,
alternative, and offline routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Tuple, Optional, Dict, Any

from src.navigation.routing import RoutingEngine
from src.db.auth_service import SessionContext
from src.api.auth import get_optional_session

router = APIRouter(prefix="/api/v1/routing", tags=["routing"])
engine = RoutingEngine()


class CalculateRouteRequest(BaseModel):
    origin: Tuple[float, float] = Field(..., description="(latitude, longitude) of start point")
    destination: Tuple[float, float] = Field(..., description="(latitude, longitude) of target point")
    is_offline: bool = Field(False, description="Set True if request is performed during offline mode")


@router.post("/calculate")
def calculate_route_endpoint(
    req: CalculateRouteRequest,
    context: Optional[SessionContext] = Depends(get_optional_session),
):
    """
    Calculate turn-by-turn route between origin and destination coordinates.
    Returns recommended, alternative, and offline route options.
    """
    try:
        res = engine.compute_route(
            origin=req.origin,
            destination=req.destination,
            is_offline=req.is_offline,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Routing calculation failed: {str(e)}")
