"""
Tests for Phase 29: Navigation Reliability State Machine
Verifies that:
1. Navigation states transition strictly according to valid transitions.
2. The UI receives real state emitted from NavigationStateEngine (STANDBY -> NORMAL -> DR -> REACQUIRING -> NORMAL).
3. Invalid state jumps raise ValueError.
4. Process sensor health drives transitions automatically.
"""

import pytest
from src.navigation.nav_state import NavigationStateEngine, NavState, VALID_NAV_TRANSITIONS


def test_navigation_state_engine_initial_state():
    """Engine initializes in STANDBY state."""
    engine = NavigationStateEngine()
    assert engine.current_state == NavState.STANDBY
    payload = engine.get_status_payload()
    assert payload["state"] == "STANDBY"


def test_navigation_state_engine_happy_path_transitions():
    """
    Test exact pipeline:
    Start Navigation -> STANDBY -> GNSS Available -> NORMAL -> GNSS Lost -> DR -> GNSS Restored -> REACQUIRING -> Converged -> NORMAL
    """
    engine = NavigationStateEngine(NavState.STANDBY)

    # 1. GNSS Available -> NORMAL
    state1 = engine.process_sensor_health(gnss_available=True, hdop=1.2)
    assert state1 == NavState.NORMAL

    # 2. GNSS Lost -> DR
    state2 = engine.process_sensor_health(gnss_available=False)
    assert state2 == NavState.DR

    # 3. GNSS Restored -> REACQUIRING
    state3 = engine.process_sensor_health(gnss_available=True, hdop=1.5, reacquiring_progress=0.4)
    assert state3 == NavState.REACQUIRING

    # 4. Filter Converging (95%) -> NORMAL
    state4 = engine.process_sensor_health(gnss_available=True, hdop=1.2, reacquiring_progress=0.96)
    assert state4 == NavState.NORMAL


def test_invalid_state_transition_raises_value_error():
    """Direct invalid state jumps (e.g. REACQUIRING -> STANDBY) raise ValueError."""
    engine = NavigationStateEngine(NavState.STANDBY)
    engine.transition(NavState.NORMAL)
    engine.process_sensor_health(gnss_available=False)  # Moves to DR

    assert engine.current_state == NavState.DR

    # Invalid jump: DR cannot jump directly to NORMAL without REACQUIRING
    with pytest.raises(ValueError, match="Invalid navigation state transition"):
        engine.transition(NavState.NORMAL)


def test_sensor_failure_triggers_error_state():
    """IMU hardware error forces ERROR state."""
    engine = NavigationStateEngine(NavState.NORMAL)
    err_state = engine.process_sensor_health(gnss_available=True, imu_healthy=False)
    assert err_state == NavState.ERROR


def test_minimalist_gnss_dr_notifications():
    """Verify clean minimalist toast notifications for GNSS loss and GNSS recovery."""
    engine = NavigationStateEngine(NavState.NORMAL)

    # 1. GNSS Lost -> DR Notification
    engine.process_sensor_health(gnss_available=False)
    assert engine.current_state == NavState.DR

    notif_dr = engine.get_ui_notification()
    assert notif_dr is not None
    assert notif_dr["title"] == "Dead Reckoning Active"
    assert notif_dr["body"] == "GNSS signal lost. Navigation continues."

    # 2. GNSS Restored -> REACQUIRING -> NORMAL Notification
    engine.process_sensor_health(gnss_available=True, reacquiring_progress=0.96)
    notif_restored = engine.get_ui_notification()
    assert notif_restored is not None
    assert notif_restored["title"] == "GNSS Restored"
    assert notif_restored["body"] == "Position correction resumed."
