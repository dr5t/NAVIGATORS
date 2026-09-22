"""
Navigators IDR - Offline Synchronization API Router
Provides endpoints for incremental delta synchronization, offline device queue uploads,
conflict detection, and offline map package distribution.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.db.database import init_db
from src.db.canonical import CanonicalRepository, OfflinePackage
from src.db.sync import SyncRepository, SyncReceipt
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/sync", tags=["sync"])

init_db()
canonical_repo = CanonicalRepository()
sync_repo = SyncRepository()
auth_service = AuthService()
authz_service = AuthorizationService()






class SyncItemModel(BaseModel):
    client_id: Optional[str] = Field(None, description="Client generated UUID for idempotent deduplication")
    client_sequence: int = Field(1, description="Sequence of change on client device")
    operation: str = Field(..., description="Operation type: add_place, suggest_edit, or report")
    payload: Dict[str, Any] = Field(..., description="Operation payload")
    base_version: Optional[int] = Field(None, description="Expected base version of target resource for conflict detection")


class PushSyncRequest(BaseModel):
    device_id: str = Field(..., min_length=3, description="Unique client device identifier")
    client_timestamp: Optional[str] = Field(None, description="Client timestamp when batch was prepared")
    items: List[SyncItemModel] = Field(..., description="List of queued offline changes to push")


class BuildPackageRequest(BaseModel):
    region: str = Field("global", description="Geographic region name")
    format: str = Field("sqlite", description="Package format: sqlite or json_bundle")


def _extract_query_val(val: Any, default: Any = None) -> Any:
    """Extract actual value if val is a FastAPI Query object, or return default."""
    if hasattr(val, "default"):
        res = val.default
        return default if res is ... else res
    return val if val is not None else default






@router.get("/status")
def api_sync_status():
    """
    Get synchronization status, server changelog sequence, and latest offline map package info.
    Accessible to all users and offline clients checking for updates.
    """
    latest_seq = canonical_repo.get_latest_sequence()
    latest_pkg = canonical_repo.get_latest_package(region="global")

    return {
        "status": "online",
        "latest_sequence": latest_seq,
        "latest_package": latest_pkg.to_dict() if latest_pkg else None,
    }


@router.get("/pull")
def api_sync_pull(
    since_sequence: int = Query(0, ge=0, description="Retrieve changelog entries after this sequence ID"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of changelog deltas to return"),
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Pull canonical map updates and changelog deltas.
    Returns delta list, latest sequence ID, and has_more flag.
    Gated by sync:pull.
    """
    since_seq = int(_extract_query_val(since_sequence, 0))
    lim = int(_extract_query_val(limit, 100))

    user_context = context.to_dict() if context else {"user": None, "roles": ["guest"], "permissions": ["place:read"]}
    decision = authz_service.can(user=user_context, action="sync:pull", resource="sync")
    if not decision.allowed:
        alt_decision = authz_service.can(user=user_context, action="place:read", resource="place")
        if not alt_decision.allowed:
            raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    entries, latest_sequence, has_more = canonical_repo.get_changes(
        since_sequence=since_seq,
        limit=lim,
    )

    return {
        "since_sequence": since_seq,
        "latest_sequence": latest_sequence,
        "count": len(entries),
        "has_more": has_more,
        "changes": [e.to_dict() for e in entries],
    }


@router.post("/push")
def api_sync_push(
    body: PushSyncRequest,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Upload a batch of offline changes queued by a device.
    Performs conflict detection, creates canonical contribution drafts, and returns sync receipts.
    Gated by sync:push.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="sync:push", resource="sync")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    actor_id = context.user.id if context and context.user else "usr_guest"
    raw_items = [item.model_dump() for item in body.items]

    receipts = sync_repo.push_device_changes(
        device_id=body.device_id,
        items=raw_items,
        actor_id=actor_id,
        client_timestamp=body.client_timestamp,
    )

    synced_count = sum(1 for r in receipts if r.status == "synced")
    conflict_count = sum(1 for r in receipts if r.status == "conflict")

    return {
        "device_id": body.device_id,
        "total_items": len(receipts),
        "synced_count": synced_count,
        "conflict_count": conflict_count,
        "receipts": [r.to_dict() for r in receipts],
    }


@router.get("/package/latest")
def api_get_latest_package(
    region: str = Query("global", description="Geographic region name"),
):
    """
    Retrieve metadata for the latest pre-compiled offline map package.
    Accessible to all users and devices.
    """
    region_str = str(_extract_query_val(region, "global")).strip().lower()
    pkg = canonical_repo.get_latest_package(region=region_str)
    if not pkg:

        pkg = canonical_repo.build_offline_package(region=region_str, format="sqlite")

    return {
        "package": pkg.to_dict(),
        "download_url": f"/api/v1/sync/package/download/{pkg.id}",
    }


@router.get("/package/download/{package_id}")
def api_download_package(package_id: str):
    """
    Download a pre-compiled offline map package file.
    """
    pkg = canonical_repo.get_package(package_id)
    if not pkg:
        raise HTTPException(status_code=404, detail=f"Offline map package '{package_id}' not found")

    file_path = Path(pkg.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Package file does not exist on storage")

    media_type = "application/x-sqlite3" if pkg.format == "sqlite" else "application/json"
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
    )


@router.post("/package/build")
def api_build_package(
    body: BuildPackageRequest,
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Manually trigger compilation of a new standalone offline map package.
    Gated by package:build.
    """
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="package:build", resource="package")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    try:
        pkg = canonical_repo.build_offline_package(
            region=body.region,
            format=body.format,
        )
        return {
            "message": "Offline map package successfully compiled",
            "package": pkg.to_dict(),
            "download_url": f"/api/v1/sync/package/download/{pkg.id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build offline package: {str(e)}")


@router.get("/device-queue/{device_id}")
def api_get_device_queue(
    device_id: str,
    limit: int = Query(50, ge=1, le=200),
    context: Optional[SessionContext] = Depends(get_current_session),
):
    """
    Query processed or pending offline sync items for a device.
    Gated by sync:push.
    """
    lim = int(_extract_query_val(limit, 50))
    user_context = context.to_dict() if context else None
    decision = authz_service.can(user=user_context, action="sync:push", resource="sync")
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Permission denied: {decision.reason}")

    items = sync_repo.get_device_queue(device_id=device_id, limit=lim)
    return {
        "device_id": device_id,
        "count": len(items),
        "items": [item.to_dict() for item in items],
    }
