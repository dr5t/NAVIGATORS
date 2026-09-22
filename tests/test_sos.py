"""
Navigators IDR - Phase 34 SOS Flow Test Suite
Validates:
  1. SOS -> Confirm -> Emergency Options (Call emergency services, Share live location, Emergency contact).
  2. No automatic consequential action taking place upon trigger.
  3. Map / navigation session continues underneath uninterrupted (nav_state preserved).
  4. Public live location tracking link resolution.
  5. Emergency contacts management and alert dispatch.
"""

import pytest
from pathlib import Path
from fastapi import HTTPException

from src.db.database import init_db
from src.db.auth_service import AuthService
from src.db.sos import SOSRepository, EmergencyContactsRepository
from src.api.sos import (
    trigger_sos as api_trigger_sos,
    confirm_sos as api_confirm_sos,
    select_emergency_option as api_select_emergency_option,
    cancel_sos as api_cancel_sos,
    get_live_location_public as api_get_live_location_public,
    list_emergency_contacts as api_list_emergency_contacts,
    add_emergency_contact as api_add_emergency_contact,
    delete_emergency_contact as api_delete_emergency_contact,
    TriggerSOSRequest,
    SelectOptionRequest,
    AddEmergencyContactRequest,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "navigators_sos_test.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def sos_repo(temp_db: Path):
    return SOSRepository(temp_db)


@pytest.fixture
def contacts_repo(temp_db: Path):
    return EmergencyContactsRepository(temp_db)


@pytest.fixture
def auth_service(temp_db: Path):
    return AuthService(temp_db)


# =============================================================================
# 1. Full SOS Flow Tests
# =============================================================================

def test_full_sos_flow_call_emergency_services(sos_repo: SOSRepository, auth_service: AuthService):
    """
    Validates: SOS -> Confirm -> Call emergency services option.
    Ensures no automatic consequential action and navigation continues underneath.
    """
    user, session_user, _ = auth_service.register("sos_user1@navigators.dev", "Password123!", "SOS User 1")

    # Step 1: Trigger SOS during active Dead Reckoning (DR) navigation
    trig_res = sos_repo.trigger_sos(
        user_id=user.id,
        latitude=28.6139,
        longitude=77.2090,
        nav_state="DR",
    )
    assert trig_res["status"] == "triggered"
    assert trig_res["requires_confirmation"] is True
    assert trig_res["automatic_action_taken"] is False
    assert trig_res["nav_session_active"] is True
    assert trig_res["nav_state"] == "DR"
    sos_id = trig_res["sos_id"]

    # Step 2: User confirms SOS
    conf_res = sos_repo.confirm_sos(user_id=user.id, sos_id=sos_id)
    assert conf_res["status"] == "confirmed"
    assert len(conf_res["emergency_options"]) == 3
    option_keys = [opt["key"] for opt in conf_res["emergency_options"]]
    assert "call_emergency_services" in option_keys
    assert "share_live_location" in option_keys
    assert "emergency_contact" in option_keys
    assert conf_res["automatic_action_taken"] is False
    assert conf_res["nav_session_active"] is True

    # Step 3: User explicitly selects "call_emergency_services" option
    opt_res = sos_repo.select_emergency_option(
        user_id=user.id,
        sos_id=sos_id,
        option="call_emergency_services",
    )
    assert opt_res["status"] == "action_selected"
    assert opt_res["selected_option"] == "call_emergency_services"
    assert opt_res["action_details"]["phone_number"] == "112"
    assert opt_res["action_details"]["dial_uri"] == "tel:112"
    assert opt_res["nav_session_active"] is True
    assert opt_res["nav_state"] == "DR"


def test_sos_share_live_location_and_public_tracking(sos_repo: SOSRepository, auth_service: AuthService):
    """
    Validates: SOS -> Confirm -> Share live location option.
    Generates tracking URL and verifies public endpoint.
    """
    user, session_user, _ = auth_service.register("sos_user2@navigators.dev", "Password123!", "SOS User 2")

    trig = sos_repo.trigger_sos(user.id, latitude=28.5284, longitude=77.2185, nav_state="NORMAL")
    sos_id = trig["sos_id"]
    sos_repo.confirm_sos(user.id, sos_id)

    # Select Share Live Location option
    opt_res = sos_repo.select_emergency_option(user.id, sos_id, option="share_live_location")
    details = opt_res["action_details"]
    assert details["action_type"] == "share_live_location"
    assert "live_location_token" in details
    assert "tracking_url" in details

    token = details["live_location_token"]

    # Public tracking lookup
    public_res = sos_repo.get_live_location(token)
    assert public_res is not None
    assert public_res["expired"] is False
    assert public_res["latitude"] == 28.5284
    assert public_res["longitude"] == 77.2185
    assert public_res["nav_state"] == "NORMAL"


def test_sos_emergency_contact_dispatch(
    sos_repo: SOSRepository,
    contacts_repo: EmergencyContactsRepository,
    auth_service: AuthService,
):
    """
    Validates: SOS -> Confirm -> Emergency contact alert dispatch.
    """
    user, session_user, _ = auth_service.register("sos_user3@navigators.dev", "Password123!", "SOS User 3")

    # Add emergency contact
    c1 = contacts_repo.add_contact(user.id, "Emergency Contact 1", "+91 98765 43210", "Parent")

    trig = sos_repo.trigger_sos(user.id, latitude=28.6000, longitude=77.2000, nav_state="GNSS_DEGRADED")
    sos_id = trig["sos_id"]
    sos_repo.confirm_sos(user.id, sos_id)

    # Select Emergency Contact option
    opt_res = sos_repo.select_emergency_option(user.id, sos_id, option="emergency_contact")
    details = opt_res["action_details"]
    assert details["action_type"] == "emergency_contact"
    assert details["contact_count"] == 1
    assert details["contacts_notified"][0]["phone"] == "+91 98765 43210"
    assert opt_res["nav_state"] == "GNSS_DEGRADED"


def test_no_automatic_consequential_action_guarantee(sos_repo: SOSRepository, auth_service: AuthService):
    """
    Guarantees that triggering SOS without confirm/select option performs NO consequential actions.
    """
    user, _, _ = auth_service.register("sos_user4@navigators.dev", "Password123!", "SOS User 4")

    trig = sos_repo.trigger_sos(user.id, latitude=28.7000, longitude=77.1000, nav_state="NORMAL")
    assert trig["automatic_action_taken"] is False

    # Cancel SOS session without selecting option
    cancel_res = sos_repo.cancel_sos(user.id, trig["sos_id"])
    assert cancel_res["status"] == "cancelled"
    assert cancel_res["nav_session_active"] is True


# =============================================================================
# 2. REST API Integration Tests
# =============================================================================

def test_sos_api_endpoints(
    monkeypatch,
    temp_db: Path,
    auth_service: AuthService,
    sos_repo: SOSRepository,
    contacts_repo: EmergencyContactsRepository,
):
    """
    Validates REST API endpoints for SOS trigger, confirm, option selection,
    cancel, live tracking, and emergency contacts.
    """
    import src.api.sos as sos_api_mod
    monkeypatch.setattr(sos_api_mod, "sos_repo", sos_repo)
    monkeypatch.setattr(sos_api_mod, "contacts_repo", contacts_repo)
    monkeypatch.setattr(sos_api_mod, "auth_service", auth_service)

    user, session_user, _ = auth_service.register("api_sos@navigators.dev", "Password123!", "API SOS User")

    # 1. Add Emergency Contact API
    add_req = AddEmergencyContactRequest(name="Doctor", phone="+91 11 2222 3333", relationship="Physician")
    res_add_c = api_add_emergency_contact(add_req, context=session_user)
    assert res_add_c["contact"]["name"] == "Doctor"
    contact_id = res_add_c["contact"]["id"]

    # 2. List Emergency Contacts API
    res_list_c = api_list_emergency_contacts(context=session_user)
    assert res_list_c["count"] == 1

    # 3. Trigger SOS API
    trig_req = TriggerSOSRequest(latitude=28.6139, longitude=77.2090, nav_state="NORMAL")
    res_trig = api_trigger_sos(trig_req, context=session_user)
    assert res_trig["status"] == "triggered"
    sos_id = res_trig["sos_id"]

    # 4. Confirm SOS API
    res_conf = api_confirm_sos(sos_id=sos_id, context=session_user)
    assert res_conf["status"] == "confirmed"

    # 5. Select Emergency Option API
    opt_req = SelectOptionRequest(option="share_live_location")
    res_opt = api_select_emergency_option(sos_id=sos_id, req=opt_req, context=session_user)
    assert res_opt["status"] == "action_selected"
    token = res_opt["action_details"]["live_location_token"]

    # 6. Public Live Location Tracking API
    res_live = api_get_live_location_public(token=token)
    assert res_live["latitude"] == 28.6139
    assert res_live["longitude"] == 77.2090

    # 7. Cancel SOS API
    res_cancel = api_cancel_sos(sos_id=sos_id, context=session_user)
    assert res_cancel["status"] == "cancelled"

    # 8. Delete Emergency Contact API
    api_delete_emergency_contact(contact_id=contact_id, context=session_user)
    res_contacts_after = api_list_emergency_contacts(context=session_user)
    assert res_contacts_after["count"] == 0
