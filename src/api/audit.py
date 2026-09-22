"""
Navigators IDR - Platform Audit Logging REST API Router (Phase 17)
Provides secure audit query endpoints, audit log detail inspection,
and platform governance summaries gated strictly by the 'audit:read' permission.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends, Header

from src.db.database import init_db
from src.db.audit import AuditRepository, AuditEntry
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])

init_db()
audit_repo = AuditRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


def _require_audit_permission(session: SessionContext) -> None:
    """Helper to verify that the active session holds 'audit:read' permission."""
    if session.is_guest or not session.user:
        raise HTTPException(status_code=401, detail="Authentication required to inspect platform audit logs.")
    
    decision = authz_service.can(session, "audit:read")
    if not decision.allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: Account lacks 'audit:read' permission required to view audit records. ({decision.reason})"
        )


@router.get("/logs", response_model=Dict[str, Any])
def list_audit_logs(
    actor_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: SessionContext = Depends(get_current_session),
):
    """
    List platform audit logs with optional filtering by actor, resource type, resource ID, or action.
    Requires 'audit:read' permission (Moderators, Team Admins, Super Admins).
    """
    _require_audit_permission(session)

    logs = audit_repo.list_logs(
        actor_id=actor_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        limit=limit,
        offset=offset,
    )
    total = audit_repo.count_logs(
        actor_id=actor_id,
        resource_type=resource_type,
        action=action,
    )

    return {
        "items": [log.to_dict() for log in logs],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/logs/{log_id}", response_model=Dict[str, Any])
def get_audit_log_detail(
    log_id: str,
    session: SessionContext = Depends(get_current_session),
):
    """
    Fetch a single audit log entry by ID.
    Requires 'audit:read' permission.
    """
    _require_audit_permission(session)

    log = audit_repo.get_log(log_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"Audit log record '{log_id}' not found.")
    
    return log.to_dict()


@router.get("/summary", response_model=Dict[str, Any])
def get_audit_summary(
    session: SessionContext = Depends(get_current_session),
):
    """
    Retrieve platform audit activity breakdown (total counts, by resource type, by action, top actors).
    Requires 'audit:read' permission.
    """
    _require_audit_permission(session)

    return audit_repo.get_audit_summary()
