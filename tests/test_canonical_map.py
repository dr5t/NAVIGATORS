"""
Tests for Phase 9: Canonical Map Pipeline
Verifies that:
1. Community submissions do not directly become map data until moderated and published.
2. Canonical mutations append monotonically increasing changelog entries.
3. Online map API immediately reflects canonical state.
4. Standalone offline map packages are compiled with valid schema and SHA-256 checksums.
"""

import os
import tempfile
import sqlite3
import hashlib
from pathlib import Path

from src.db.database import get_db, init_db
from src.db.places import PlaceRepository
from src.db.canonical import CanonicalRepository
from src.db.contributions import ContributionRepository
from src.db.state_machine import ContributionState
from src.db.auth_service import AuthService, SessionContext
from src.api.places import list_canonical_places as api_list_canonical_places
from src.api.moderation import (
    approve_contribution as api_approve_contribution,
    ApproveContributionRequest,
)
from src.api.sync import (
    api_get_latest_package,
    api_download_package,
)


def test_user_contribution_to_canonical_map_pipeline():
    """
    Verifies full lifecycle:
    User adds petrol pump -> Pending -> Moderator approves -> Canonical Petrol Pump -> Online Map.
    """
    init_db()
    auth_service = AuthService()
    contrib_repo = ContributionRepository()
    place_repo = PlaceRepository()
    canonical_repo = CanonicalRepository()


    import secrets
    tag = secrets.token_hex(4)
    user, user_session, _ = auth_service.register(
        email=f"shaurya_canon_{tag}@example.com",
        password="SecurePassword123!",
        name="Shaurya Tiwari",
        role_id="user",
    )

    mod, mod_session, _ = auth_service.register(
        email=f"mod_canon_{tag}@example.com",
        password="SecurePassword123!",
        name="Moderator Lead",
        role_id="moderator",
    )


    pump_name = f"ABC Fuel Station {tag}"
    pump_data = {
        "name": pump_name,
        "category": "fuel",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "address": "Connaught Place Outer Circle, New Delhi",
        "opening_hours": "24/7",
        "phone": "+91-11-23456789",
    }
    contrib = contrib_repo.create_contribution(
        owner_id=user.id,
        resource_type="place",
        title=f"Add {pump_name}",
        data=pump_data,
        action="create",
    )


    contrib_repo.transition_state(
        contribution_id=contrib.id,
        target_state=ContributionState.PENDING_REVIEW,
        user=user_session,
    )


    public_res = api_list_canonical_places(q=pump_name, context=None)
    places_found = [p for p in public_res["places"] if p["name"] == pump_name]
    assert len(places_found) == 0, "Pending contribution must NOT directly appear in canonical map!"


    approve_res = api_approve_contribution(
        contrib_id=contrib.id,
        req=ApproveContributionRequest(notes="Verified fuel station signage and GPS coordinates"),
        context=mod_session,
    )
    assert approve_res["item"]["status"] == "approved"


    canonical_place = place_repo.publish_from_contribution(
        contribution=contrib_repo.get_contribution(contrib.id),
        publisher=mod.id,
    )
    assert canonical_place is not None
    assert canonical_place.name == pump_name
    assert canonical_place.category in ("fuel", "petrol_pump")
    assert canonical_place.version == 1


    latest_seq = canonical_repo.get_latest_sequence()
    assert latest_seq > 0
    changes, max_seq, _ = canonical_repo.get_changes(since_sequence=latest_seq - 1)
    assert len(changes) >= 1
    last_change = changes[-1]
    assert last_change.resource_id == canonical_place.id
    assert last_change.action == "create"
    assert last_change.version == 1


    online_res = api_list_canonical_places(q=pump_name, context=None)
    online_places = [p for p in online_res["places"] if p["id"] == canonical_place.id]
    assert len(online_places) == 1
    assert online_places[0]["name"] == pump_name


def test_canonical_mutations_and_changelog_tracking():
    """
    Verifies that canonical updates, soft deletes, and restores increment version
    and append corresponding entries to the canonical changelog stream.
    """
    init_db()
    place_repo = PlaceRepository()
    canonical_repo = CanonicalRepository()


    place = place_repo.create_place(
        name="Metro Hospital",
        category="hospital",
        latitude=28.5355,
        longitude=77.3910,
        address="Sector 12, Noida",
        phone="+91-120-123456",
        created_by=None,
    )
    v1_version = place.version
    assert v1_version == 1


    updated = place_repo.update_place(
        place_id=place.id,
        phone="+91-120-999999",
        opening_hours="24 Hours Emergency",
        change_summary="Emergency desk updated",
    )
    assert updated.version == 2
    assert updated.phone == "+91-120-999999"


    deleted = place_repo.soft_delete_place(
        place_id=place.id,
        reason="Facility under renovation",
    )
    assert deleted.version == 3
    assert deleted.is_deleted == 1
    assert deleted.status == "archived"


    restored = place_repo.restore_place(
        place_id=place.id,
    )
    assert restored.version == 4
    assert restored.is_deleted == 0
    assert restored.status == "published"


    changes, _, _ = canonical_repo.get_changes(since_sequence=0, limit=500)
    resource_changes = [c for c in changes if c.resource_id == place.id]
    assert len(resource_changes) >= 4

    actions = [c.action for c in resource_changes]
    assert "create" in actions
    assert "update" in actions
    assert "delete" in actions
    assert "restore" in actions


def test_standalone_offline_map_package_generation():
    """
    Verifies offline map package compilation, SQLite schema, place records,
    SHA-256 checksum integrity, and download endpoint.
    """
    init_db()
    canonical_repo = CanonicalRepository()
    place_repo = PlaceRepository()


    place_repo.create_place(
        name="Central Park Plaza",
        category="landmark",
        latitude=28.6289,
        longitude=77.2065,
        created_by=None,
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        pkg = canonical_repo.build_offline_package(
            region="delhi_ncr",
            format="sqlite",
            export_dir=temp_dir,
        )

        assert pkg.id.startswith("pkg_")
        assert pkg.region == "delhi_ncr"
        assert pkg.format == "sqlite"
        assert pkg.record_count >= 1
        assert Path(pkg.file_path).exists()
        assert pkg.size_bytes > 0


        hasher = hashlib.sha256()
        with open(pkg.file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        expected_checksum = hasher.hexdigest()
        assert pkg.checksum_sha256 == expected_checksum


        conn = sqlite3.connect(pkg.file_path)
        try:
            meta = conn.execute("SELECT * FROM package_metadata").fetchone()
            assert meta is not None

            places = conn.execute("SELECT name, category FROM places").fetchall()
            names = [p[0] for p in places]
            assert "Central Park Plaza" in names
        finally:
            conn.close()


        dl_res = api_download_package(package_id=pkg.id)
        assert dl_res.path == pkg.file_path
        assert Path(dl_res.path).stat().st_size == pkg.size_bytes
