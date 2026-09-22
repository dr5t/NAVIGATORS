"""
Navigators IDR - Phase 35 Privacy and Data Controls Test Suite
Validates domain privacy preferences management, actual file unlinking on disk,
database record purges, and cascading full account deletion with session revocation.
"""

import os
import uuid
import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db, connect_db
from src.db.rbac import RBACRepository
from src.db.auth_service import AuthService, SessionContext
from src.db.privacy import PrivacyRepository, VALID_DOMAINS
from src.api.privacy import (
    UpdateDomainSettingRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "navigators_privacy_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def rbac_repo(temp_db: Path):
    return RBACRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path):
    return AuthService(temp_db)


@pytest.fixture
def privacy_repo(temp_db: Path):
    return PrivacyRepository(temp_db)


def _create_user_and_context(rbac_svc: RBACRepository, auth_svc: AuthService, email: str = "privacy.user@example.com", name: str = "Privacy Tester", role: str = "user"):
    user_id = str(uuid.uuid4())
    user = rbac_svc.create_user(user_id=user_id, email=email, name=name)
    rbac_svc.assign_role_to_user(user.id, role)
    ctx, raw_token = auth_svc.create_session(user_id=user.id)
    return user, ctx, raw_token






def test_get_privacy_settings_defaults(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "defaults@example.com", "Defaults User")

    settings = privacy_repo.get_privacy_settings(user.id)
    assert len(settings) == 6

    domain_names = [s["domain"] for s in settings]
    for dom in VALID_DOMAINS:
        assert dom in domain_names

    for s in settings:
        assert s["stored_locally"] is True
        assert s["synced"] is True
        assert s["sharing_level"] == "private"






def test_update_domain_setting_valid(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "update.setting@example.com", "Setting User")

    updated = privacy_repo.update_domain_setting(
        user_id=user.id,
        domain="location_data",
        stored_locally=False,
        synced=False,
        sharing_level="anonymous",
    )
    assert updated["domain"] == "location_data"
    assert updated["stored_locally"] is False
    assert updated["synced"] is False
    assert updated["sharing_level"] == "anonymous"


    get_res = privacy_repo.get_privacy_settings(user.id)
    all_settings = {s["domain"]: s for s in get_res}
    assert all_settings["location_data"]["stored_locally"] is False
    assert all_settings["location_data"]["synced"] is False
    assert all_settings["location_data"]["sharing_level"] == "anonymous"


def test_update_domain_setting_invalid_domain(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "invalid.dom@example.com", "Invalid Domain User")

    with pytest.raises(ValueError) as exc_info:
        privacy_repo.update_domain_setting(
            user_id=user.id,
            domain="non_existent_domain",
            sharing_level="public",
        )
    assert "Invalid privacy domain" in str(exc_info.value)


def test_update_domain_setting_invalid_sharing_level(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "invalid.share@example.com", "Invalid Share User")

    with pytest.raises(ValueError) as exc_info:
        privacy_repo.update_domain_setting(
            user_id=user.id,
            domain="sensor_data",
            sharing_level="super_public",
        )
    assert "Invalid sharing level" in str(exc_info.value)






def test_purge_location_data(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "purge.loc@example.com", "Purge Location User")


    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute(
        "INSERT INTO saved_places (id, user_id, name, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), user.id, "Home", 12.97, 77.59),
    )
    c.execute(
        "INSERT INTO recent_searches (id, user_id, query_text) VALUES (?, ?, ?)",
        (str(uuid.uuid4()), user.id, "Bengaluru Station"),
    )
    conn.commit()
    conn.close()


    res = privacy_repo.purge_domain_data(user_id=user.id, domain="location_data")
    assert res["domain"] == "location_data"
    assert res["purged"] is True
    assert res["deleted_records"] == 2


    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM saved_places WHERE user_id = ?", (user.id,))
    assert c.fetchone()[0] == 0
    c.execute("SELECT COUNT(*) FROM recent_searches WHERE user_id = ?", (user.id,))
    assert c.fetchone()[0] == 0
    conn.close()


def test_purge_dataset_contributions_physical_file_deletion(temp_db, rbac_repo, auth_service, privacy_repo, tmp_path):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "purge.dataset@example.com", "Purge Dataset User", role="internal_contributor")


    dummy_file = tmp_path / "sensor_recording_test.bin"
    dummy_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    assert dummy_file.exists()


    ds_id = str(uuid.uuid4())
    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO dataset_sessions (id, contributor_id, activity_type, device, sensor_data_path)
        VALUES (?, ?, 'driving', 'iPhone 15', ?)
        """,
        (ds_id, user.id, str(dummy_file)),
    )
    conn.commit()
    conn.close()


    res = privacy_repo.purge_domain_data(user_id=user.id, domain="dataset_contributions")
    assert res["domain"] == "dataset_contributions"
    assert res["purged"] is True
    assert res["deleted_records"] == 1
    assert res["deleted_files"] == 1


    assert not dummy_file.exists()


    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM dataset_sessions WHERE contributor_id = ?", (user.id,))
    assert c.fetchone()[0] == 0
    conn.close()


def test_purge_contribution_data(temp_db, rbac_repo, auth_service, privacy_repo):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "purge.contrib@example.com", "Purge Contrib User")


    contrib_id = str(uuid.uuid4())
    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO contributions (id, owner_id, resource_type, title, status)
        VALUES (?, ?, 'place', 'New Cafe', 'draft')
        """,
        (contrib_id, user.id),
    )
    conn.commit()
    conn.close()

    res = privacy_repo.purge_domain_data(user_id=user.id, domain="contribution_data")
    assert res["deleted_records"] == 1

    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM contributions WHERE owner_id = ?", (user.id,))
    assert c.fetchone()[0] == 0
    conn.close()






def test_delete_account_full_wipe(temp_db, rbac_repo, auth_service, privacy_repo, tmp_path):
    user, ctx, raw_token = _create_user_and_context(rbac_repo, auth_service, "delete.account@example.com", "Delete Account User")


    dummy_file = tmp_path / "account_dataset.bin"
    dummy_file.write_bytes(b"account data file")
    assert dummy_file.exists()

    conn = connect_db(temp_db)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO dataset_sessions (id, contributor_id, activity_type, device, sensor_data_path)
        VALUES (?, ?, 'walking', 'Pixel 8', ?)
        """,
        (str(uuid.uuid4()), user.id, str(dummy_file)),
    )

    c.execute(
        "INSERT INTO saved_places (id, user_id, name, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), user.id, "Work", 12.98, 77.60),
    )
    conn.commit()
    conn.close()


    res = privacy_repo.delete_user_account(user_id=user.id)
    assert res["user_id"] == user.id
    assert res["account_deleted"] is True
    assert res["deleted_files"] == 1


    assert not dummy_file.exists()


    conn = connect_db(temp_db)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE id = ?", (user.id,))
    assert c.fetchone()[0] == 0

    c.execute("SELECT COUNT(*) FROM saved_places WHERE user_id = ?", (user.id,))
    assert c.fetchone()[0] == 0

    c.execute("SELECT revoked_at FROM sessions WHERE user_id = ?", (user.id,))
    sess_rows = c.fetchall()
    for s in sess_rows:
        assert s[0] is not None
    conn.close()
