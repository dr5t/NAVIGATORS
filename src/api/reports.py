"""
Navigators IDR - Community Reporting API Router
Allows authenticated users to submit flags and issue reports on map data and contributions.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.db.database import init_db
from src.db.reports import ReportRepository, Report
from src.db.audit import AuditRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.api.auth import get_current_session

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

init_db()
report_repo = ReportRepository()
audit_repo = AuditRepository()
auth_service = AuthService()
authz_service = AuthorizationService()


class CreateReportRequest(BaseModel):
    target_type: str = Field(..., description="Entity type: 'place', 'contribution', or 'user'")
    target_id: str = Field(..., description="ID of the entity being flagged")
    reason: str = Field(..., min_length=3, description="Summary reason for reporting")
    details: Optional[str] = Field(None, description="Detailed explanation or evidence")


@router.post("", status_code=201)
def submit_report(
    req: CreateReportRequest,
    context: SessionContext = Depends(get_current_session),
):
    """
    Submit a community report/flag against a place, contribution, or user.
    Requires authenticated user account with 'report:create' permission.
    """
    decision = authz_service.can(user=context, action="report:create")
    if not decision.allowed:
        status_code = 401 if decision.code in ("UNAUTHENTICATED", "ACCOUNT_INACTIVE") else 403
        raise HTTPException(status_code=status_code, detail=decision.reason)

    if req.target_type not in ("place", "contribution", "user"):
        raise HTTPException(status_code=400, detail="Invalid target_type. Expected 'place', 'contribution', or 'user'.")

    reporter_id = context.user.id if context.user else "anonymous"
    report = report_repo.create_report(
        reporter_id=reporter_id,
        target_type=req.target_type,
        target_id=req.target_id,
        reason=req.reason,
        details=req.details,
    )


    try:
        audit_repo.log(
            action="report:created",
            resource_type="report",
            resource_id=report.id,
            actor_id=reporter_id,
            new_state="pending",
            metadata={"target_type": req.target_type, "target_id": req.target_id, "reason": req.reason},
        )
    except Exception:
        pass

    return {
        "message": "Report submitted successfully.",
        "report": report.to_dict(),
    }
