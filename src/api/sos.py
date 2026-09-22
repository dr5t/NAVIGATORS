"""
Navigators IDR - Phase 34 Non-Modal SOS Emergency API Router
Flow: SOS -> Confirm -> Emergency Options (Call emergency services, Share live location, Emergency contact)
Guarantees:
  1. No automatic consequential action.
  2. Map / navigation session continues underneath.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field

from src.db.database import init_db
from src.db.sos import SOSRepository, EmergencyContactsRepository
from src.db.auth_service import AuthService, SessionContext
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/sos", tags=["sos"])

init_db()
sos_repo = SOSRepository()
contacts_repo = EmergencyContactsRepository()
auth_service = AuthService()


# =============================================================================
# Request Models
# =============================================================================

class TriggerSOSRequest(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    nav_state: str = "NORMAL"


class SelectOptionRequest(BaseModel):
    option: str = Field(..., description="Chosen emergency option: call_emergency_services, share_live_location, or emergency_contact")
    metadata: Optional[Dict[str, Any]] = None


class AddEmergencyContactRequest(BaseModel):
    name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=1)
    relationship: Optional[str] = None


# =============================================================================
# SOS Flow Endpoints
# =============================================================================

@router.post("/trigger", status_code=202)
def trigger_sos(
    req: TriggerSOSRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Step 1: Initiate SOS trigger.
    Creates an SOS session requiring explicit user confirmation.
    No automatic consequential action is taken.
    Navigation session continues underneath.
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required to trigger SOS.")

    res = sos_repo.trigger_sos(
        user_id=context.user.id,
        latitude=req.latitude,
        longitude=req.longitude,
        nav_state=req.nav_state,
    )
    return res


@router.post("/{sos_id}/confirm")
def confirm_sos(
    sos_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Step 2: User confirms SOS trigger.
    Reveals explicit emergency options: Call Emergency Services, Share Live Location, Emergency Contact.
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        return sos_repo.confirm_sos(user_id=context.user.id, sos_id=sos_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{sos_id}/option")
def select_emergency_option(
    sos_id: str,
    req: SelectOptionRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Step 3: User selects an emergency option.
    Options:
      - call_emergency_services
      - share_live_location
      - emergency_contact
    Navigation session continues underneath uninterrupted.
    """
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        return sos_repo.select_emergency_option(
            user_id=context.user.id,
            sos_id=sos_id,
            option=req.option,
            metadata=req.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{sos_id}/cancel")
def cancel_sos(
    sos_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """Cancel active SOS session."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    return sos_repo.cancel_sos(user_id=context.user.id, sos_id=sos_id)


@router.get("/live/{token}")
def get_live_location_public(token: str):
    """Public tracking endpoint for shared live location token."""
    res = sos_repo.get_live_location(token)
    if not res:
        raise HTTPException(status_code=404, detail="Live location token not found.")
    return res


# =============================================================================
# Emergency Contacts Endpoints
# =============================================================================

@router.get("/contacts")
def list_emergency_contacts(
    context: SessionContext = Depends(get_current_session),
):
    """List registered emergency contacts for current user."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    contacts = contacts_repo.list_contacts(context.user.id)
    return {"contacts": [c.to_dict() for c in contacts], "count": len(contacts)}


@router.post("/contacts", status_code=201)
def add_emergency_contact(
    req: AddEmergencyContactRequest,
    context: SessionContext = Depends(get_current_session),
):
    """Add a new emergency contact."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    contact = contacts_repo.add_contact(
        user_id=context.user.id,
        name=req.name,
        phone=req.phone,
        relationship=req.relationship,
    )
    return {"message": "Emergency contact added successfully.", "contact": contact.to_dict()}


@router.delete("/contacts/{contact_id}")
def delete_emergency_contact(
    contact_id: str,
    context: SessionContext = Depends(get_current_session),
):
    """Remove an emergency contact."""
    if not context.user:
        raise HTTPException(status_code=401, detail="Authentication required.")

    success = contacts_repo.delete_contact(context.user.id, contact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Emergency contact not found.")
    return {"message": "Emergency contact removed successfully."}
