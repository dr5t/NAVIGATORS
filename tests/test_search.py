"""
Navigators IDR - Phase 33 Search Test Suite
Validates unified top search field functionality across:
  1. Places (Canonical POIs)
  2. Addresses (Street / address text matching)
  3. Coordinates (Multi-format GPS coordinate parsing)
  4. Saved Places (User bookmarks)
  5. Recent Searches (Query history)

And verifies Online vs Offline operation:
  - Online mode: Searches online geocoder & local database
  - Offline mode: Relies strictly on local POI / map database
  - Honest capability indicators: Explicit offline warnings provided when offline
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.places import PlaceRepository
from src.db.auth_service import AuthService
from src.db.search import (
    SearchEngine,
    SavedPlacesRepository,
    RecentSearchesRepository,
    CoordinateParser,
)
from src.api.search import (
    execute_search as api_execute_search,
    list_saved_places as api_list_saved_places,
    add_saved_place as api_add_saved_place,
    delete_saved_place as api_delete_saved_place,
    list_recent_searches as api_list_recent_searches,
    clear_recent_searches as api_clear_recent_searches,
    SavePlaceRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "navigators_search_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def place_repo(temp_db: Path):
    return PlaceRepository(temp_db)


@pytest.fixture
def search_engine(temp_db: Path):
    return SearchEngine(temp_db)


@pytest.fixture
def saved_repo(temp_db: Path):
    return SavedPlacesRepository(temp_db)


@pytest.fixture
def recent_repo(temp_db: Path):
    return RecentSearchesRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path):
    return AuthService(temp_db)


# =============================================================================
# 1. Coordinate Parser Tests
# =============================================================================

def test_coordinate_parser_formats():
    """Validates parsing of multiple standard coordinate formats."""
    # 1. Decimal format
    c1 = CoordinateParser.parse("28.6139, 77.2090")
    assert c1 == (28.6139, 77.2090)

    c2 = CoordinateParser.parse("-12.34 56.78")
    assert c2 == (-12.34, 56.78)

    # 2. Hemisphere suffix
    c3 = CoordinateParser.parse("28.6139 N, 77.2090 E")
    assert c3 == (28.6139, 77.2090)

    c4 = CoordinateParser.parse("12.34 S, 56.78 W")
    assert c4 == (-12.34, -56.78)

    # 3. DMS format: 28°36'50"N 77°12'32"E
    c5 = CoordinateParser.parse("""28°36'50"N 77°12'32"E""")
    assert c5 is not None
    assert pytest.approx(c5[0], rel=1e-3) == 28.6138
    assert pytest.approx(c5[1], rel=1e-3) == 77.2088

    # Invalid coordinates
    assert CoordinateParser.parse("random text query") is None
    assert CoordinateParser.parse("95.0, 180.0") is None  # Latitude out of bounds
    assert CoordinateParser.parse("") is None


# =============================================================================
# 2. Saved Places Repository Tests
# =============================================================================

def test_saved_places_crud(saved_repo: SavedPlacesRepository, auth_service: AuthService):
    """Validates adding, listing, searching, and deleting saved places."""
    user, _, _ = auth_service.register("saved_user@navigators.dev", "Password123!", "Saved User")

    # Add saved places
    sp1 = saved_repo.add_saved_place(
        user_id=user.id,
        name="Home",
        latitude=28.6139,
        longitude=77.2090,
        category="landmark",
        address="Sector 1, New Delhi",
        notes="Home apartment",
    )
    assert sp1["name"] == "Home"

    sp2 = saved_repo.add_saved_place(
        user_id=user.id,
        name="Office",
        latitude=28.5355,
        longitude=77.3910,
        category="hotel",
        address="Cyber City, Gurugram",
    )

    # List saved places
    places = saved_repo.list_saved_places(user.id)
    assert len(places) == 2

    # Search filter in saved places
    filtered = saved_repo.list_saved_places(user.id, search="Cyber")
    assert len(filtered) == 1
    assert filtered[0]["name"] == "Office"

    # Delete saved place
    assert saved_repo.delete_saved_place(user.id, sp1["id"]) is True
    assert len(saved_repo.list_saved_places(user.id)) == 1


# =============================================================================
# 3. Recent Searches Repository Tests
# =============================================================================

def test_recent_searches_tracking(recent_repo: RecentSearchesRepository, auth_service: AuthService):
    """Validates query history recording, deduplication, and clearing."""
    user, _, _ = auth_service.register("recent_user@navigators.dev", "Password123!", "Recent User")

    recent_repo.add_recent_search(user.id, "Hospital near me")
    recent_repo.add_recent_search(user.id, "Connaught Place petrol pump")
    recent_repo.add_recent_search(user.id, "Hospital near me")  # Duplicate query

    searches = recent_repo.list_recent_searches(user.id)
    assert len(searches) == 2
    assert searches[0]["query_text"] == "Hospital near me"  # Most recent first

    # Clear recent searches
    recent_repo.clear_recent_searches(user.id)
    assert len(recent_repo.list_recent_searches(user.id)) == 0


# =============================================================================
# 4. Unified Search Engine & Offline Mode Tests
# =============================================================================

def test_unified_search_types_and_offline_notice(
    search_engine: SearchEngine,
    place_repo: PlaceRepository,
    saved_repo: SavedPlacesRepository,
    auth_service: AuthService,
):
    """
    Validates that unified search returns:
      - Places
      - Addresses
      - Coordinates
      - Saved Places
      - Recent Searches
    And provides honest offline notices when offline.
    """
    user, session_user, _ = auth_service.register("search_user@navigators.dev", "Password123!", "Search User")

    # Seed POI & Address
    place_repo.create_place(
        name="Apollo Hospital Saket",
        category="hospital",
        latitude=28.5284,
        longitude=77.2185,
        address="Press Enclave Marg, Saket",
    )

    # Seed Saved Place
    saved_repo.add_saved_place(
        user_id=user.id,
        name="Favorite Apollo Clinic",
        latitude=28.5284,
        longitude=77.2185,
        address="Saket, New Delhi",
    )

    # 1. Search coordinates
    res_coord = search_engine.search("28.5284, 77.2185", user_id=user.id, is_online=True)
    coord_types = [r["result_type"] for r in res_coord["results"]]
    assert "coordinates" in coord_types

    # 2. Search POIs and Addresses in Online mode
    res_online = search_engine.search("Apollo", user_id=user.id, is_online=True)
    assert res_online["is_online"] is True
    assert res_online["offline_mode"] is False
    assert res_online["offline_notice"] is None
    assert res_online["count"] >= 1

    types_online = {r["result_type"] for r in res_online["results"]}
    assert "place" in types_online or "saved_place" in types_online

    # 3. Search in Offline mode (is_online=False)
    res_offline = search_engine.search("Press Enclave", user_id=user.id, is_online=False)
    assert res_offline["is_online"] is False
    assert res_offline["offline_mode"] is True
    assert "Offline Mode" in res_offline["offline_notice"]
    assert res_offline["capabilities"]["online_geocoder"] == "unavailable_offline"
    assert res_offline["capabilities"]["places"] == "available_offline"

    types_offline = {r["result_type"] for r in res_offline["results"]}
    assert "address" in types_offline or "place" in types_offline


# =============================================================================
# 5. Search API Endpoints Integration Tests
# =============================================================================

def test_search_api_endpoints(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    search_engine: SearchEngine,
    saved_repo: SavedPlacesRepository,
    recent_repo: RecentSearchesRepository,
):
    """
    Validates REST API endpoints for execute_search, saved places, and recent searches.
    """
    import src.api.search as search_mod
    monkeypatch.setattr(search_mod, "search_engine", search_engine)
    monkeypatch.setattr(search_mod, "saved_repo", saved_repo)
    monkeypatch.setattr(search_mod, "recent_repo", recent_repo)
    monkeypatch.setattr(search_mod, "auth_service", auth_service)

    user, session_user, _ = auth_service.register("api_search@navigators.dev", "Password123!", "API Search User")

    # 1. Save place via API
    req_save = SavePlaceRequest(
        name="Gym",
        latitude=28.6000,
        longitude=77.2000,
        category="landmark",
        address="Sector 15",
    )
    res_save = api_add_saved_place(req_save, context=session_user)
    assert res_save["saved_place"]["name"] == "Gym"

    # 2. List saved places via API
    res_list_saved = api_list_saved_places(context=session_user)
    assert res_list_saved["count"] == 1
    sp_id = res_list_saved["saved_places"][0]["id"]

    # 3. Execute unified search API
    res_search = api_execute_search(
        q="Gym",
        is_online=False,
        context=session_user,
    )
    assert res_search["offline_mode"] is True
    assert len(res_search["results"]) >= 1

    # 4. List recent searches API
    res_recent = api_list_recent_searches(context=session_user)
    assert res_recent["count"] >= 1

    # 5. Clear recent searches API
    api_clear_recent_searches(context=session_user)
    res_recent_after = api_list_recent_searches(context=session_user)
    assert res_recent_after["count"] == 0

    # 6. Delete saved place API
    api_delete_saved_place(saved_place_id=sp_id, context=session_user)
    res_saved_after = api_list_saved_places(context=session_user)
    assert res_saved_after["count"] == 0
