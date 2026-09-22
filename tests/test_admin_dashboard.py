"""
Navigators IDR - Phase 18: Admin Dashboard Tests
Verifies standalone admin area routes, executive KPI metrics API,
system health diagnostics, user administration endpoints, and RBAC permission gating.
"""

import tempfile
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService
from src.db.places import PlaceRepository
from src.api.admin import get_admin_overview, get_system_health, list_admin_users
from src.api.server import app, get_admin_page


@pytest.fixture
def temp_db():
    """Create an isolated temporary database for test execution."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    if db_path.exists():
        db_path.unlink()


def test_admin_overview_api(temp_db):
    """Test that /api/v1/admin/overview returns expected executive KPI stats."""
    auth_service = AuthService(temp_db)
    place_repo = PlaceRepository(temp_db)

    admin_user, admin_session, _ = auth_service.register(
        email="overview_admin@example.com",
        password="Password123!",
        name="Overview Admin",
        role_id="team_admin",
    )

    place_repo.create_place(
        name="Admin Test POI",
        category="fuel",
        latitude=12.9716,
        longitude=77.5946,
        created_by=admin_user.id,
    )

    overview = get_admin_overview(session=admin_session)
    assert "users" in overview
    assert "places" in overview
    assert "contributions" in overview
    assert "datasets" in overview
    assert "models" in overview
    assert "audit" in overview
    assert "reports" in overview
    assert overview["users"]["total"] >= 1


def test_admin_system_health_api():
    """Test that /api/v1/admin/health returns valid system diagnostic telemetry."""
    health = get_system_health()
    assert health["status"] in ("healthy", "degraded")
    assert "database" in health
    assert health["database"]["connected"] is True
    assert "ml_model" in health


def test_admin_users_list_and_rbac_gating(temp_db):
    """Test user administration API listing and permission gating."""
    auth_service = AuthService(temp_db)


    guest_session, _ = auth_service.create_session(user_id=None, is_guest=True)
    with pytest.raises(HTTPException) as exc_info:
        list_admin_users(session=guest_session)
    assert exc_info.value.status_code == 401


    _, norm_session, _ = auth_service.register(
        email="user_no_admin@example.com",
        password="Password123!",
        name="No Admin User",
        role_id="user",
    )
    with pytest.raises(HTTPException) as exc_info:
        list_admin_users(session=norm_session)
    assert exc_info.value.status_code == 403


    admin_user, admin_session, _ = auth_service.register(
        email="users_admin@example.com",
        password="Password123!",
        name="Users Admin",
        role_id="team_admin",
    )
    users_resp = list_admin_users(session=admin_session)
    assert "items" in users_resp
    assert users_resp["total"] >= 2


def test_admin_page_route_serving():
    """Test that /admin route serves the admin.html file response."""
    response = get_admin_page()
    assert response is not None
    assert response.path.endswith("admin.html")
