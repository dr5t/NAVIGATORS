"""
Navigators IDR - Phase 35 Privacy & Data Controls REST API Router
Provides endpoints to manage domain privacy settings (stored_locally, synced, sharing_level),
execute domain-specific physical data purges, and perform full account deletions.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from src.db.database import init_db
from src.db.privacy import PrivacyRepository, VALID_DOMAINS, VALID_SHARING_LEVELS
from src.db.auth_service import SessionContext
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/privacy", tags=["privacy"])

init_db()
privacy_repo = PrivacyRepository()






class UpdateDomainSettingRequest(BaseModel):
    stored_locally: Optional[bool] = None
    synced: Optional[bool] = None
    sharing_level: Optional[str] = Field(
        None,
        description="Sharing level: 'private', 'anonymous', 'team', or 'public'",
    )


class DomainSettingResponse(BaseModel):
    domain: str
    stored_locally: bool
    synced: bool
    sharing_level: str
    updated_at: str


class PrivacySettingsSummaryResponse(BaseModel):
    user_id: str
    settings: List[DomainSettingResponse]


class PurgeDomainResponse(BaseModel):
    user_id: str
    domain: str
    purged: bool
    deleted_records: int
    deleted_files: int
    timestamp: str


class DeleteAccountResponse(BaseModel):
    user_id: str
    account_deleted: bool
    deleted_records: int
    deleted_files: int
    timestamp: str






def _ensure_authenticated_user(context: SessionContext) -> str:
    if context.is_guest or not context.user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access or modify privacy settings",
        )
    return context.user_id






@router.get("/settings", response_model=PrivacySettingsSummaryResponse)
def get_privacy_settings(
    context: SessionContext = Depends(get_current_session),
):
    """
    Retrieves privacy settings for all 6 data domains for the current user.
    """
    user_id = _ensure_authenticated_user(context)
    settings = privacy_repo.get_privacy_settings(user_id)
    return PrivacySettingsSummaryResponse(user_id=user_id, settings=settings)


@router.patch("/settings/{domain}", response_model=DomainSettingResponse)
def update_domain_setting(
    domain: str,
    req: UpdateDomainSettingRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Updates stored_locally, synced, or sharing_level preference for a specific data domain.
    """
    user_id = _ensure_authenticated_user(context)
    if domain not in VALID_DOMAINS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid privacy domain '{domain}'. Must be one of {VALID_DOMAINS}",
        )

    if req.sharing_level is not None and req.sharing_level not in VALID_SHARING_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sharing level '{req.sharing_level}'. Must be one of {VALID_SHARING_LEVELS}",
        )

    try:
        updated = privacy_repo.update_domain_setting(
            user_id=user_id,
            domain=domain,
            stored_locally=req.stored_locally,
            synced=req.synced,
            sharing_level=req.sharing_level,
        )
        return DomainSettingResponse(**updated)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/purge/{domain}", response_model=PurgeDomainResponse)
def purge_domain_data(
    domain: str,
    context: SessionContext = Depends(get_current_session),
):
    """
    Triggers an immediate physical purge for the specified data domain.
    Deletes database entries and unlinks physical disk files where applicable.
    """
    user_id = _ensure_authenticated_user(context)
    if domain not in VALID_DOMAINS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid privacy domain '{domain}'. Must be one of {VALID_DOMAINS}",
        )

    try:
        summary = privacy_repo.purge_domain_data(user_id=user_id, domain=domain)
        return PurgeDomainResponse(**summary)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/account", response_model=DeleteAccountResponse)
def delete_account(
    context: SessionContext = Depends(get_current_session),
):
    """
    Performs a complete user account deletion.
    Cascades removal across all database entities and revokes all active session tokens.
    """
    user_id = _ensure_authenticated_user(context)
    summary = privacy_repo.delete_user_account(user_id=user_id)
    return DeleteAccountResponse(**summary)
