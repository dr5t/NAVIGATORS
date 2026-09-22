"""
Navigators IDR - Platform Admin API Router (Phase 18)
Exposes aggregated executive metrics, real-time system health diagnostics,
and user administration endpoints for the standalone Admin Dashboard.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends

from src.db.database import get_db, init_db
from src.db.rbac import RBACRepository
from src.db.places import PlaceRepository
from src.db.contributions import ContributionRepository
from src.db.datasets import DatasetRepository
from src.db.model_registry import ModelRegistryRepository
from src.db.audit import AuditRepository
from src.db.reports import ReportRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

init_db()
rbac_repo = RBACRepository()
place_repo = PlaceRepository()
contrib_repo = ContributionRepository()
dataset_repo = DatasetRepository()
model_repo = ModelRegistryRepository()
audit_repo = AuditRepository()
report_repo = ReportRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


def _require_admin_permission(session: SessionContext, permission_id: str = "role:assign") -> None:
    """Verify that the authenticated session possesses required administration permission or admin role."""
    if session.is_guest or not session.user:
        raise HTTPException(status_code=401, detail="Authentication required for administrative controls.")

    user_roles = {r.id for r in session.roles}
    has_admin_role = bool(user_roles & {"moderator", "team_admin", "super_admin"})
    has_perm = permission_id in session.permissions

    if not (has_admin_role or has_perm):
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: Account lacks administrative permission '{permission_id}'."
        )


@router.get("/overview", response_model=Dict[str, Any])
def get_admin_overview(
    session: SessionContext = Depends(get_current_session),
):
    """
    Return executive system KPI stats across all platform subsystems.
    Requires administrative authentication.
    """
    _require_admin_permission(session, "user:read")

    # DB Connection counts
    with get_db() as conn:
        total_users = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        active_users = conn.execute("SELECT COUNT(*) AS n FROM users WHERE status = 'active'").fetchone()["n"]
        suspended_users = conn.execute("SELECT COUNT(*) AS n FROM users WHERE status = 'suspended'").fetchone()["n"]
        
        total_places = conn.execute("SELECT COUNT(*) AS n FROM places").fetchone()["n"]
        published_places = conn.execute("SELECT COUNT(*) AS n FROM places WHERE is_deleted = 0 AND status = 'published'").fetchone()["n"]
        archived_places = conn.execute("SELECT COUNT(*) AS n FROM places WHERE is_deleted = 1 OR status = 'archived'").fetchone()["n"]

        pending_contribs = conn.execute("SELECT COUNT(*) AS n FROM contributions WHERE status = 'pending_review'").fetchone()["n"]
        total_contribs = conn.execute("SELECT COUNT(*) AS n FROM contributions").fetchone()["n"]

        pending_reports = conn.execute("SELECT COUNT(*) AS n FROM reports WHERE status = 'pending'").fetchone()["n"]

    dataset_stats = dataset_repo.get_stats()
    audit_summary = audit_repo.get_audit_summary()
    prod_model = model_repo.get_production_model()

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "suspended": suspended_users,
        },
        "places": {
            "total": total_places,
            "published": published_places,
            "archived": archived_places,
        },
        "contributions": {
            "total": total_contribs,
            "pending_review": pending_contribs,
        },
        "datasets": dataset_stats,
        "models": {
            "production_model": prod_model.to_dict() if prod_model else None,
        },
        "audit": {
            "total_logs": audit_summary.get("total_logs", 0),
            "by_resource_type": audit_summary.get("by_resource_type", {}),
        },
        "reports": {
            "pending": pending_reports,
        },
        "system_health": {
            "status": "healthy",
            "database": "connected",
        }
    }


@router.get("/health", response_model=Dict[str, Any])
def get_system_health():
    """
    Return real-time diagnostic health metrics for backend services.
    Publicly accessible health endpoint for monitoring probes.
    """
    db_connected = False
    table_count = 0
    try:
        with get_db() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table'").fetchone()
            table_count = row["n"] if row else 0
            db_connected = True
    except Exception:
        db_connected = False

    prod_model = model_repo.get_production_model()

    return {
        "status": "healthy" if db_connected else "degraded",
        "service": "Navigators IDR Backend",
        "database": {
            "connected": db_connected,
            "tables_count": table_count,
        },
        "ml_model": {
            "loaded": prod_model is not None,
            "active_version": prod_model.name if prod_model else "Baseline EKF/DR Engine",
            "test_mae": prod_model.test_mae if prod_model else None,
        },
        "version": "v1.0.0",
    }


@router.get("/users", response_model=Dict[str, Any])
def list_admin_users(
    search: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: SessionContext = Depends(get_current_session),
):
    """
    List user accounts with assigned roles and permission metadata.
    Requires 'user:read' permission.
    """
    _require_admin_permission(session, "user:read")

    clauses: List[str] = []
    params: List[Any] = []

    if search:
        clauses.append("(name LIKE ? OR email LIKE ?)")
        term = f"%{search.strip()}%"
        params.extend([term, term])
    if status:
        clauses.append("status = ?")
        params.append(status.strip())

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM users {where_sql}", params).fetchone()["n"]
        cur = conn.execute(
            f"SELECT * FROM users {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        )
        user_rows = cur.fetchall()

    users_data = []
    for u in user_rows:
        u_dict = dict(u)
        roles = rbac_repo.get_user_roles(u_dict["id"])
        u_dict["roles"] = [r.to_dict() for r in roles]
        users_data.append(u_dict)

    return {
        "items": users_data,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
