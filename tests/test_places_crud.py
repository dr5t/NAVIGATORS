"""
Navigators IDR - Phase 7 Community Place CRUD Test Suite
Validates the complete lifecycle for community places:
  1. User adds a place -> Creates place contribution in DRAFT status.
  2. "My Contributions" view for users.
  3. "Pending Contributions" triage view for moderators.
  4. Public users see only PUBLISHED canonical places.
  5. Draft updates by author vs non-owner rejection.
  6. Moderator review and approval of pending submissions.
  7. Publishing contribution creates canonical place with version 1 audit history.
  8. Normal users cannot directly mutate published canonical places (HTTP 403).
  9. Normal users use "Suggest Edit" -> Creates linked contribution.
  10. Publishing suggested edit updates canonical place, increments version to 2, and logs history.
  11. Soft delete / archiving preserves history (never casual physical delete).
  12. Restoring soft-deleted canonical place.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.authorization import AuthorizationService
from src.db.contributions import ContributionRepository, Contribution
from src.db.places import PlaceRepository, Place, POI_TAXONOMY, CATEGORY_ALIASES, normalize_category
from src.db.state_machine import ContributionState
from src.api.places import (
    add_place as api_add_place,
    list_canonical_places as api_list_canonical_places,
    get_place_detail as api_get_place_detail,
    get_place_version_history as api_get_place_version_history,
    suggest_edit_place as api_suggest_edit_place,
    direct_update_place as api_direct_update_place,
    soft_delete_place as api_soft_delete_place,
    restore_place as api_restore_place,
    get_my_contributions as api_get_my_contributions,
    get_pending_contributions as api_get_pending_contributions,
    get_poi_categories as api_get_poi_categories,
    CreatePlaceRequest,
    SuggestEditRequest,
    DirectUpdatePlaceRequest,
    DeletePlaceRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = tmp_path / "navigators_places_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def rbac_repo(temp_db: Path):
    return RBACRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path, rbac_repo: RBACRepository):
    return AuthService(temp_db, rbac_repo)


@pytest.fixture
def authz_service(temp_db: Path, rbac_repo: RBACRepository):
    return AuthorizationService(temp_db, rbac_repo)


@pytest.fixture
def contrib_repo(temp_db: Path):
    return ContributionRepository(temp_db)


@pytest.fixture
def place_repo(temp_db: Path):
    return PlaceRepository(temp_db)


# =============================================================================
# 1. Place Creation & Draft Lifecycle
# =============================================================================

def test_create_place_contribution_as_draft(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
):
    """
    User creates a place:
      Name: ABC Fuel Station
      Category: fuel
      Location: 28.6139, 77.2090
    Backend creates place contribution in DRAFT status with owner_id = current_user.
    """
    import src.api.places as places_mod
    test_place_repo = PlaceRepository(temp_db)
    test_contrib_repo = ContributionRepository(temp_db)
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(places_mod, "place_repo", test_place_repo)
    monkeypatch.setattr(places_mod, "contrib_repo", test_contrib_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    user, session_user, _ = auth_service.register(
        email="contributor@navigators.dev",
        password="Password123!",
        name="Local Contributor",
    )

    req = CreatePlaceRequest(
        name="ABC Fuel Station",
        category="fuel",
        latitude=28.6139,
        longitude=77.2090,
        address="Outer Ring Road, Sector 4",
        opening_hours="24/7",
        phone="+91 98765 43210",
        website="https://abcfuel.example",
        submit_now=False,
    )

    res = api_add_place(req, context=session_user)
    contrib = res["contribution"]

    assert contrib["status"] == ContributionState.DRAFT
    assert contrib["owner_id"] == user.id
    assert contrib["resource_type"] == "place"
    assert contrib["title"] == "ABC Fuel Station"
    assert contrib["data"]["category"] == "fuel"
    assert contrib["data"]["latitude"] == 28.6139
    assert contrib["data"]["longitude"] == 77.2090
    assert contrib["data"]["opening_hours"] == "24/7"


# =============================================================================
# 2. Reading Contributions: User vs Moderator vs Public
# =============================================================================

def test_my_contributions_and_moderator_pending_views(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
):
    """
    User sees only 'My Contributions'.
    Moderator sees 'Pending Contributions'.
    Public users see only PUBLISHED places.
    """
    import src.api.places as places_mod
    test_place_repo = PlaceRepository(temp_db)
    test_contrib_repo = ContributionRepository(temp_db)
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(places_mod, "place_repo", test_place_repo)
    monkeypatch.setattr(places_mod, "contrib_repo", test_contrib_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    user1, session1, _ = auth_service.register("user1@navigators.dev", "Password123!", "User 1")
    user2, session2, _ = auth_service.register("user2@navigators.dev", "Password123!", "User 2")
    moderator, session_mod, _ = auth_service.register("mod@navigators.dev", "Password123!", "Moderator", role_id="moderator")

    # User 1 creates 1 draft and 1 submitted place
    c1 = test_contrib_repo.create(
        contribution_id="c1", owner_id=user1.id, resource_type="place",
        title="User 1 Draft", data={"category": "pharmacy"}, status=ContributionState.DRAFT
    )
    c2 = test_contrib_repo.create(
        contribution_id="c2", owner_id=user1.id, resource_type="place",
        title="User 1 Pending", data={"category": "atm"}, status=ContributionState.PENDING_REVIEW
    )

    # User 2 creates 1 draft
    c3 = test_contrib_repo.create(
        contribution_id="c3", owner_id=user2.id, resource_type="place",
        title="User 2 Draft", data={"category": "hospital"}, status=ContributionState.DRAFT
    )

    # User 1 checks "My Contributions" -> sees c1 and c2, but NOT c3
    my_contribs = api_get_my_contributions(context=session1)
    assert my_contribs["count"] == 2
    my_ids = {item["id"] for item in my_contribs["contributions"]}
    assert my_ids == {"c1", "c2"}

    # Moderator checks "Pending Contributions" -> sees c2, but NOT c1 or c3 (which are drafts)
    pending = api_get_pending_contributions(context=session_mod)
    pending_ids = {item["id"] for item in pending["pending_contributions"]}
    assert "c2" in pending_ids
    assert "c1" not in pending_ids
    assert "c3" not in pending_ids

    # Regular user cannot access moderator pending view
    with pytest.raises(HTTPException) as exc_pending:
        api_get_pending_contributions(context=session1)
    assert exc_pending.value.status_code == 403


def test_public_users_see_only_published_canonical_places(
    monkeypatch,
    temp_db: Path,
    place_repo: PlaceRepository,
    contrib_repo: ContributionRepository,
    auth_service: AuthService,
):
    """
    Public users see only active published canonical places.
    Drafts, pending reviews, and archived places are not shown.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)

    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "contrib_repo", contrib_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    # 1. Create a draft contribution (should NOT be visible to public)
    dummy_author, _, _ = auth_service.register("dummy_unpub@navigators.dev", "Password123!", "Dummy Author")
    contrib_repo.create("contrib_unpub", dummy_author.id, "place", "Unpublished Draft", status=ContributionState.DRAFT)

    # 2. Create a published canonical place
    p1 = place_repo.create_place(
        name="Metro Fuel Station",
        category="fuel",
        latitude=28.6139,
        longitude=77.2090,
        address="Ring Road",
        opening_hours="24/7",
    )

    # 3. Create a second published place and then soft-delete it
    p2 = place_repo.create_place(
        name="Old Pharmacy",
        category="pharmacy",
        latitude=28.6150,
        longitude=77.2100,
    )
    place_repo.soft_delete_place(p2.id, changed_by=dummy_author.id, reason="Decommissioned")

    # Public user queries places
    guest_session, _ = auth_service.create_guest_session()
    res = api_list_canonical_places(context=guest_session)

    assert res["count"] == 1
    place_names = [p["name"] for p in res["places"]]
    assert "Metro Fuel Station" in place_names
    assert "Old Pharmacy" not in place_names
    assert "Unpublished Draft" not in place_names


# =============================================================================
# 3. Canonical Publication & Versioned Audit History
# =============================================================================

def test_publish_creates_canonical_place_and_version_1_history(
    place_repo: PlaceRepository,
    contrib_repo: ContributionRepository,
    auth_service: AuthService,
):
    """
    When a place contribution is published:
      - Canonical place is created in 'places' table.
      - Version 1 snapshot is recorded in 'place_history'.
    """
    author, session_author, _ = auth_service.register("place_author@navigators.dev", "Password123!", "Author")
    moderator, session_mod, _ = auth_service.register("place_mod@navigators.dev", "Password123!", "Mod", role_id="moderator")

    # Author creates draft place
    contrib = contrib_repo.create(
        contribution_id="contrib_pub_1",
        owner_id=author.id,
        resource_type="place",
        title="Central EV Charging Hub",
        data={
            "name": "Central EV Charging Hub",
            "category": "ev_charging",
            "latitude": 28.5355,
            "longitude": 77.3910,
            "opening_hours": "24/7",
            "phone": "+91 11 2345 6789",
        },
        status=ContributionState.DRAFT,
    )

    # Move through state machine: DRAFT -> PENDING_REVIEW -> APPROVED -> PUBLISHED
    contrib_repo.submit(contrib.id, user=session_author)
    contrib_repo.approve(contrib.id, reviewer=session_mod, notes="Verified charging station on site")
    published = contrib_repo.publish(contrib.id, staff=session_mod)

    assert published.status == ContributionState.PUBLISHED

    # Verify canonical place was created
    places = place_repo.list_places(category="ev_charging")
    assert len(places) == 1
    canonical = places[0]
    assert canonical.name == "Central EV Charging Hub"
    assert canonical.category in ("ev_charging", "charging_station")
    assert canonical.version == 1
    assert canonical.status == "published"
    assert canonical.is_deleted == 0

    # Verify version 1 history record was logged
    history = place_repo.get_place_history(canonical.id)
    assert len(history) == 1
    h1 = history[0]
    assert h1.version == 1
    assert h1.action == "created"
    assert h1.to_dict()["snapshot"]["name"] == "Central EV Charging Hub"


# =============================================================================
# 4. Direct Staff Updates vs Normal User "Suggest Edit"
# =============================================================================

def test_normal_user_cannot_directly_update_published_place(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
):
    """
    Normal users cannot directly modify published canonical places.
    Direct PATCH on /api/v1/places/{id} must return HTTP 403.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("regular@navigators.dev", "Password123!", "Regular User")

    canonical = place_repo.create_place(
        name="City Hospital",
        category="hospital",
        latitude=28.7041,
        longitude=77.1025,
    )

    # Regular user attempts direct PATCH
    with pytest.raises(HTTPException) as exc_direct:
        api_direct_update_place(
            place_id=canonical.id,
            req=DirectUpdatePlaceRequest(name="Tampered Name"),
            context=session_user,
        )
    assert exc_direct.value.status_code == 403
    assert "Suggest Edit" in exc_direct.value.detail


def test_suggest_edit_workflow_and_version_increment(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
    contrib_repo: ContributionRepository,
):
    """
    Normal user suggests an edit on an existing published place:
      1. Suggest Edit creates a contribution with action='update' and target_resource_id=place.id.
      2. Moderator approves and staff publishes the contribution.
      3. Canonical place version increments to 2 and history is recorded.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "contrib_repo", contrib_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("editor@navigators.dev", "Password123!", "Community Editor")
    moderator, session_mod, _ = auth_service.register("mod_edit@navigators.dev", "Password123!", "Mod", role_id="moderator")

    # 1. Canonical place exists (Version 1)
    canonical = place_repo.create_place(
        name="Apex Pharmacy",
        category="pharmacy",
        latitude=28.5000,
        longitude=77.2000,
        opening_hours="09:00 - 18:00",
        phone="+91 11 1111 2222",
    )
    assert canonical.version == 1

    # 2. Normal user suggests edit: updated opening hours to 24/7
    sugg_res = api_suggest_edit_place(
        place_id=canonical.id,
        req=SuggestEditRequest(
            opening_hours="24/7",
            phone="+91 11 9999 8888",
            notes="Pharmacy recently expanded to 24/7 operations",
            submit_now=True,
        ),
        context=session_user,
    )
    contrib = sugg_res["contribution"]
    assert contrib["action"] == "update"
    assert contrib["target_resource_id"] == canonical.id
    assert contrib["status"] == ContributionState.PENDING_REVIEW

    # Before publication, canonical place remains unchanged
    unchanged = place_repo.get_place(canonical.id)
    assert unchanged is not None
    assert unchanged.opening_hours == "09:00 - 18:00"
    assert unchanged.version == 1

    # 3. Reviewer approves and staff publishes the suggested edit
    contrib_repo.approve(contrib["id"], reviewer=session_mod, notes="Confirmed 24/7 signboard")
    contrib_repo.publish(contrib["id"], staff=session_mod)

    # 4. Canonical place has been updated to Version 2
    updated = place_repo.get_place(canonical.id)
    assert updated is not None
    assert updated.version == 2
    assert updated.opening_hours == "24/7"
    assert updated.phone == "+91 11 9999 8888"

    # 5. Place history now contains both Version 1 and Version 2 records
    history = place_repo.get_place_history(canonical.id)
    assert len(history) == 2
    assert history[0].version == 2
    assert history[0].action == "updated"
    assert history[1].version == 1
    assert history[1].action == "created"


# =============================================================================
# 5. Soft Delete / Archiving & Restoration
# =============================================================================

def test_soft_delete_and_restore_cycle(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
):
    """
    Validates:
      - Place is soft deleted / archived (never casual physical delete).
      - History is preserved with 'soft_deleted' action.
      - Public queries hide the archived place.
      - Staff can restore the archived place.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    moderator, session_mod, _ = auth_service.register("staff_arch@navigators.dev", "Password123!", "Staff", role_id="moderator")
    guest_session, _ = auth_service.create_guest_session()

    place = place_repo.create_place(
        name="Historic Landmark",
        category="landmark",
        latitude=28.6000,
        longitude=77.3000,
    )
    assert place.version == 1

    # 1. Moderator soft-deletes the place
    del_res = api_soft_delete_place(
        place_id=place.id,
        req=DeletePlaceRequest(reason="Renovation and temporary closure"),
        context=session_mod,
    )
    archived = del_res["place"]
    assert archived["is_deleted"] == 1
    assert archived["status"] == "archived"
    assert archived["version"] == 2

    # 2. Public query excludes archived place
    pub_list = api_list_canonical_places(context=guest_session)
    assert not any(p["id"] == place.id for p in pub_list["places"])

    # 3. Public detail query returns HTTP 404
    with pytest.raises(HTTPException) as exc_get:
        api_get_place_detail(place_id=place.id, context=guest_session)
    assert exc_get.value.status_code == 404

    # 4. History retains all actions
    hist_res = api_get_place_version_history(place_id=place.id, context=session_mod)
    assert hist_res["count"] == 2
    actions = [h["action"] for h in hist_res["history"]]
    assert "soft_deleted" in actions
    assert "created" in actions

    # 5. Moderator restores the place
    res_restore = api_restore_place(place_id=place.id, context=session_mod)
    restored = res_restore["place"]
    assert restored["is_deleted"] == 0
    assert restored["status"] == "published"
    assert restored["version"] == 3

    # 6. Public query now sees the restored place again
    pub_list_after = api_list_canonical_places(context=guest_session)
    assert any(p["id"] == place.id for p in pub_list_after["places"])


def test_draft_update_owner_vs_non_owner(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    contrib_repo: ContributionRepository,
):
    """
    Validates:
      - Owner can update their own draft contribution.
      - Non-owner cannot update someone else's draft contribution.
    """
    import src.api.contributions as contrib_mod
    from src.api.contributions import update_contribution, UpdateContributionRequest
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(contrib_mod, "contrib_repo", contrib_repo)
    monkeypatch.setattr(contrib_mod, "auth_service", auth_service)
    monkeypatch.setattr(contrib_mod, "authz_service", test_authz)

    owner, session_owner, _ = auth_service.register("owner_draft@navigators.dev", "Password123!", "Draft Owner")
    other_user, session_other, _ = auth_service.register("intruder@navigators.dev", "Password123!", "Intruder")

    contrib = contrib_repo.create(
        contribution_id="contrib_draft_edit",
        owner_id=owner.id,
        resource_type="place",
        title="Original Draft Name",
        data={"category": "fuel", "name": "Original Station"},
        status=ContributionState.DRAFT,
    )

    # 1. Direct authorization service check
    decision_owner = test_authz.can(user=session_owner, action="contribution:update", resource=contrib)
    assert decision_owner.allowed is True

    decision_intruder = test_authz.can(user=session_other, action="contribution:update", resource=contrib)
    assert decision_intruder.allowed is False
    assert decision_intruder.code == "NOT_OWNER"

    # 2. Owner can successfully update their draft via API
    res = update_contribution(
        contrib_id=contrib.id,
        req=UpdateContributionRequest(
            title="Updated Draft Name",
            data={"category": "fuel", "name": "Updated Station", "opening_hours": "08:00 - 22:00"},
        ),
        context=session_owner,
    )
    assert res is not None
    assert res["contribution"] is not None
    assert res["contribution"]["title"] == "Updated Draft Name"
    assert res["contribution"]["data"]["opening_hours"] == "08:00 - 22:00"

    # 3. Non-owner cannot update the draft via API (HTTP 403)
    with pytest.raises(HTTPException) as exc_authz:
        update_contribution(
            contrib_id=contrib.id,
            req=UpdateContributionRequest(title="Hacked Title"),
            context=session_other,
        )
    assert exc_authz.value.status_code == 403


def test_direct_staff_update_and_history_recording(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
):
    """
    Moderator or admin with place:update permission can directly PATCH a published place.
    The mutation increments version to 2 and writes an immutable history snapshot.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    mod, session_mod, _ = auth_service.register("staff_patcher@navigators.dev", "Password123!", "Staff Patcher", role_id="moderator")

    canonical = place_repo.create_place(
        name="Apollo Clinic",
        category="hospital",
        latitude=28.6200,
        longitude=77.2100,
        phone="+91 11 4444 5555",
        created_by=mod.id,
    )
    assert canonical.version == 1

    # Staff directly updates phone and website
    res = api_direct_update_place(
        place_id=canonical.id,
        req=DirectUpdatePlaceRequest(
            phone="+91 11 8888 9999",
            website="https://apolloclinic.example.com",
            change_summary="Updated emergency contact details",
        ),
        context=session_mod,
    )
    patched = res["place"]
    assert patched["version"] == 2
    assert patched["phone"] == "+91 11 8888 9999"
    assert patched["website"] == "https://apolloclinic.example.com"

    # History snapshot inspection
    history = place_repo.get_place_history(canonical.id)
    assert len(history) == 2
    assert history[0].version == 2
    assert history[0].action == "updated"
    assert history[0].change_summary == "Updated emergency contact details"
    assert history[0].changed_by == mod.id


# =============================================================================
# 6. Phase 32 POI System & Pipeline Integration Tests
# =============================================================================

def test_poi_taxonomy_all_nine_categories_and_aliases(place_repo: PlaceRepository):
    """
    Validates all 9 canonical POI taxonomy categories:
      - Petrol Pump (petrol_pump)
      - Hospital (hospital)
      - Pharmacy (pharmacy)
      - ATM (atm)
      - Parking (parking)
      - Restaurant (restaurant)
      - Hotel (hotel)
      - Charging Station (charging_station)
      - Landmark (landmark)
    And verifies alias mapping resolution.
    """
    required_keys = {
        "petrol_pump", "hospital", "pharmacy", "atm", "parking",
        "restaurant", "hotel", "charging_station", "landmark"
    }
    assert required_keys.issubset(set(POI_TAXONOMY.keys()))

    # Alias normalization checks
    assert normalize_category("fuel") == "petrol_pump"
    assert normalize_category("gas_station") == "petrol_pump"
    assert normalize_category("ev_charging") == "charging_station"
    assert normalize_category("cafe") == "restaurant"
    assert normalize_category("food") == "restaurant"
    assert normalize_category("lodging") == "hotel"
    assert normalize_category("park") == "landmark"
    assert normalize_category("amenity") == "landmark"
    assert normalize_category("Petrol Pump") == "petrol_pump"
    assert normalize_category("Charging Station") == "charging_station"

    # Create one place for each of the 9 categories in POI database
    categories_created = []
    for key, display_name in POI_TAXONOMY.items():
        p = place_repo.create_place(
            name=f"Test {display_name}",
            category=key,
            latitude=28.6000 + len(categories_created) * 0.01,
            longitude=77.2000 + len(categories_created) * 0.01,
        )
        categories_created.append(p.category)

    assert len(categories_created) >= 9
    for key in required_keys:
        places_of_cat = place_repo.list_places(category=key)
        assert len(places_of_cat) == 1
        assert places_of_cat[0].category == key


def test_poi_categories_api_endpoint():
    """
    Validates GET /api/v1/places/poi/categories endpoint.
    """
    res = api_get_poi_categories()
    assert "categories" in res
    assert "aliases" in res
    assert res["categories"]["petrol_pump"] == "Petrol Pump"
    assert res["categories"]["charging_station"] == "Charging Station"
    assert res["aliases"]["fuel"] == "petrol_pump"


def test_poi_proximity_search_and_indexing(place_repo: PlaceRepository):
    """
    Validates spatial indexing and proximity search (lat, lon, radius_km)
    which orders POIs by Haversine distance ascending.
    """
    # Create 3 POIs at increasing distance from (28.6139, 77.2090)
    p_close = place_repo.create_place("Near ATM", "atm", 28.6145, 77.2095) # ~0.1 km
    p_mid = place_repo.create_place("Mid Pharmacy", "pharmacy", 28.6300, 77.2200) # ~2 km
    p_far = place_repo.create_place("Far Hotel", "hotel", 28.9000, 77.5000) # ~40 km

    # Query within 5 km radius
    results_5km = place_repo.list_places(lat=28.6139, lon=77.2090, radius_km=5.0)
    assert len(results_5km) == 2
    assert results_5km[0].id == p_close.id
    assert results_5km[1].id == p_mid.id

    # Query within 50 km radius
    results_50km = place_repo.list_places(lat=28.6139, lon=77.2090, radius_km=50.0)
    assert len(results_50km) == 3
    assert results_50km[0].id == p_close.id
    assert results_50km[1].id == p_mid.id
    assert results_50km[2].id == p_far.id


def test_search_to_poi_database_to_map_pipeline(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
):
    """
    Verifies Pipeline: Search -> POI Database -> Map
    1. POI Database holds indexed amenity POIs across categories.
    2. Search queries API / DB with category alias ("fuel"), search text ("Shell"), and radius.
    3. Results are returned cleanly to Map with full coordinate and amenity metadata.
    """
    import src.api.places as places_mod
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)

    # Populate POI Database
    place_repo.create_place(
        name="Shell Fuel Station",
        category="petrol_pump",
        latitude=28.6139,
        longitude=77.2090,
        address="Connaught Place Ring Road",
        opening_hours="24/7",
    )
    place_repo.create_place(
        name="Apollo Pharmacy",
        category="pharmacy",
        latitude=28.6150,
        longitude=77.2100,
        address="Connaught Place Block A",
    )

    guest_session, _ = auth_service.create_guest_session()

    # Search query for "fuel" category alias near CP
    res = api_list_canonical_places(
        category="fuel",
        q="Shell",
        lat=28.6139,
        lon=77.2090,
        radius_km=2.0,
        context=guest_session,
    )
    assert res["count"] == 1
    found = res["places"][0]
    assert found["name"] == "Shell Fuel Station"
    assert found["category"] == "petrol_pump"
    assert found["latitude"] == 28.6139
    assert found["longitude"] == 77.2090


def test_contribution_to_moderation_to_poi_database_pipeline(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    place_repo: PlaceRepository,
    contrib_repo: ContributionRepository,
):
    """
    Verifies Pipeline: Contribution -> Moderation -> POI Database
    1. Contributor submits new amenity POI ("Max Healthcare Hospital").
    2. Moderation workspace approves submission.
    3. Approved contribution is published into POI Database (`places` table).
    4. POI Database indexes place, which immediately becomes readable by Map & Search.
    """
    import src.api.places as places_mod
    test_authz = AuthorizationService(temp_db)
    monkeypatch.setattr(places_mod, "place_repo", place_repo)
    monkeypatch.setattr(places_mod, "contrib_repo", contrib_repo)
    monkeypatch.setattr(places_mod, "auth_service", auth_service)
    monkeypatch.setattr(places_mod, "authz_service", test_authz)

    user, session_user, _ = auth_service.register("contrib_user@navigators.dev", "Password123!", "User")
    mod, session_mod, _ = auth_service.register("mod_user@navigators.dev", "Password123!", "Moderator", role_id="moderator")

    # Step 1: Contribution created & submitted
    req = CreatePlaceRequest(
        name="Max Healthcare Hospital",
        category="hospital",
        latitude=28.5284,
        longitude=77.2185,
        address="Saket, New Delhi",
        phone="+91 11 2651 5050",
        submit_now=True,
    )
    res_contrib = api_add_place(req, context=session_user)
    contrib_id = res_contrib["contribution"]["id"]
    assert res_contrib["contribution"]["status"] == ContributionState.PENDING_REVIEW

    # Step 2: Moderation workspace approves & publishes
    approved = contrib_repo.approve(contrib_id, reviewer=session_mod, notes="Verified hospital details")
    assert approved.status == ContributionState.APPROVED

    published = contrib_repo.publish(contrib_id, staff=session_mod)
    assert published.status == ContributionState.PUBLISHED

    # Step 3: POI Database verification
    places = place_repo.list_places(category="hospital", search="Max Healthcare")
    assert len(places) == 1
    poi = places[0]
    assert poi.name == "Max Healthcare Hospital"
    assert poi.category == "hospital"
    assert poi.version == 1
    assert poi.status == "published"

