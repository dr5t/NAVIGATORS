import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import GNSSTrustState, GNSSTrustMetric
from navigation.gnss_trust import GNSSTrustEngine, GNSSTrustConfig
from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from navigation.nav_state import NavigationStateEngine, NavState


def test_healthy_gnss():
    engine = GNSSTrustEngine()
    gnss_data = {
        "position_enu": [100.0, 200.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([100.0, 199.5]),
        current_velocity=np.array([0.0, 9.8]),
        current_heading=0.0,
        position_uncertainty=2.0,
    )
    assert metric.state == GNSSTrustState.TRUSTED
    assert metric.is_trusted is True
    assert metric.trust_score >= 0.75
    assert metric.position_variance_scale == 1.0
    assert metric.velocity_variance_scale == 1.0


def test_noisy_gnss():
    engine = GNSSTrustEngine()
    gnss_data = {
        "position_enu": [100.0, 200.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 18.0,
        "timestamp": 100.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([100.0, 190.0]),
        current_velocity=np.array([0.0, 10.0]),
        current_heading=0.0,
        position_uncertainty=5.0,
    )
    assert metric.state == GNSSTrustState.DEGRADED
    assert metric.is_trusted is True
    assert 0.35 <= metric.trust_score <= 0.75
    assert metric.position_variance_scale >= 4.0


def test_sudden_position_jump():
    engine = GNSSTrustEngine()
    initial_gnss = {
        "position_enu": [0.0, 0.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    _ = engine.evaluate_trust(
        gnss_data=initial_gnss,
        dt=1.0,
        current_position=np.array([0.0, 0.0]),
    )

    jumped_gnss = {
        "position_enu": [0.0, 250.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 101.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=jumped_gnss,
        dt=1.0,
        current_position=np.array([0.0, 10.0]),
        position_uncertainty=3.0,
    )
    assert metric.state == GNSSTrustState.UNUSABLE
    assert metric.is_trusted is False
    assert metric.trust_score == 0.0
    assert metric.position_variance_scale >= 1000.0

    moderate_jump_gnss = {
        "position_enu": [0.0, 60.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 4.0,
        "timestamp": 101.0,
    }
    engine.reset()
    _ = engine.evaluate_trust(initial_gnss, dt=1.0, current_position=np.array([0.0, 0.0]))
    metric_mod = engine.evaluate_trust(
        gnss_data=moderate_jump_gnss,
        dt=1.0,
        current_position=np.array([0.0, 10.0]),
        position_uncertainty=3.0,
    )
    assert metric_mod.state in (GNSSTrustState.SUSPICIOUS, GNSSTrustState.UNUSABLE)
    assert metric_mod.position_variance_scale >= 25.0


def test_impossible_velocity():
    engine = GNSSTrustEngine()
    gnss_data = {
        "position_enu": [10.0, 20.0, 0.0],
        "velocity_enu": [0.0, 120.0, 0.0],
        "speed": 120.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([10.0, 20.0]),
        current_velocity=np.array([0.0, 15.0]),
    )
    assert metric.state == GNSSTrustState.UNUSABLE
    assert metric.is_trusted is False
    assert metric.trust_score == 0.0


def test_heading_inconsistency():
    engine = GNSSTrustEngine()
    gnss_data = {
        "position_enu": [0.0, 100.0, 0.0],
        "velocity_enu": [15.0, 0.0, 0.0],
        "speed": 15.0,
        "heading": np.pi / 2,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([0.0, 100.0]),
        current_velocity=np.array([0.0, 15.0]),
        current_heading=0.0,
        position_uncertainty=2.0,
    )
    assert metric.state == GNSSTrustState.SUSPICIOUS
    assert metric.is_trusted is False
    assert metric.position_variance_scale >= 25.0
    assert any("heading" in r for r in metric.diagnostics.get("reasons", []))


def test_degraded_accuracy():
    engine = GNSSTrustEngine()
    gnss_data = {
        "position_enu": [10.0, 20.0, 0.0],
        "velocity_enu": [0.0, 5.0, 0.0],
        "speed": 5.0,
        "heading": 0.0,
        "accuracy": 40.0,
        "timestamp": 100.0,
    }
    metric = engine.evaluate_trust(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([10.0, 20.0]),
        current_velocity=np.array([0.0, 5.0]),
        current_heading=0.0,
        position_uncertainty=5.0,
    )
    assert metric.state == GNSSTrustState.DEGRADED
    assert metric.is_trusted is True
    assert metric.position_variance_scale >= 5.0

    gnss_data_poor = {
        "position_enu": [10.0, 20.0, 0.0],
        "velocity_enu": [0.0, 5.0, 0.0],
        "speed": 5.0,
        "heading": 0.0,
        "accuracy": 65.0,
        "timestamp": 101.0,
    }
    metric_poor = engine.evaluate_trust(
        gnss_data=gnss_data_poor,
        dt=1.0,
        current_position=np.array([10.0, 20.0]),
        current_velocity=np.array([0.0, 5.0]),
        current_heading=0.0,
        position_uncertainty=5.0,
    )
    assert metric_poor.state == GNSSTrustState.SUSPICIOUS
    assert metric_poor.is_trusted is False


def test_complete_gnss_loss():
    engine = GNSSTrustEngine()
    metric_none = engine.evaluate_trust(None, dt=1.0)
    assert metric_none.state == GNSSTrustState.UNUSABLE
    assert metric_none.is_trusted is False
    assert metric_none.trust_score == 0.0
    assert metric_none.position_variance_scale == 10000.0

    metric_empty = engine.evaluate_trust({}, dt=1.0)
    assert metric_empty.state == GNSSTrustState.UNUSABLE
    assert metric_empty.is_trusted is False

    metric_stale = engine.evaluate_trust(
        {"position_enu": [0, 0, 0], "accuracy": 3.0, "timestamp": 100.0},
        dt=5.0,
    )
    assert metric_stale.state == GNSSTrustState.UNUSABLE


def test_gnss_recovery():
    engine = GNSSTrustEngine(GNSSTrustConfig(min_recovery_fixes=3))
    engine.evaluate_trust(None, dt=1.0)
    assert engine.in_recovery is True

    fix1 = {
        "position_enu": [10.0, 10.0, 0.0],
        "velocity_enu": [0.0, 5.0, 0.0],
        "speed": 5.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 10.0,
    }
    m1 = engine.evaluate_trust(
        fix1,
        dt=1.0,
        current_position=np.array([10.0, 10.0]),
        current_velocity=np.array([0.0, 5.0]),
        current_heading=0.0,
        position_uncertainty=3.0,
    )
    assert m1.state == GNSSTrustState.DEGRADED
    assert engine.consecutive_valid_fixes == 1

    fix2 = {
        "position_enu": [10.0, 15.0, 0.0],
        "velocity_enu": [0.0, 5.0, 0.0],
        "speed": 5.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 11.0,
    }
    m2 = engine.evaluate_trust(
        fix2,
        dt=1.0,
        current_position=np.array([10.0, 15.0]),
        current_velocity=np.array([0.0, 5.0]),
        current_heading=0.0,
        position_uncertainty=3.0,
    )
    assert m2.state == GNSSTrustState.DEGRADED
    assert engine.consecutive_valid_fixes == 2

    fix3 = {
        "position_enu": [10.0, 20.0, 0.0],
        "velocity_enu": [0.0, 5.0, 0.0],
        "speed": 5.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 12.0,
    }
    m3 = engine.evaluate_trust(
        fix3,
        dt=1.0,
        current_position=np.array([10.0, 20.0]),
        current_velocity=np.array([0.0, 5.0]),
        current_heading=0.0,
        position_uncertainty=2.0,
    )
    assert m3.state == GNSSTrustState.TRUSTED
    assert engine.consecutive_valid_fixes == 3
    assert engine.in_recovery is False


def test_ekf_fusion_with_trust_engine():
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.initialize_from_gnss(np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))

    unusable_metric = GNSSTrustMetric(
        trust_score=0.0,
        is_trusted=False,
        position_variance_scale=10000.0,
        velocity_variance_scale=10000.0,
        state=GNSSTrustState.UNUSABLE,
    )
    initial_pos = ekf.get_position()
    ekf.update_gnss(
        gnss_position=np.array([500.0, 500.0, 0.0]),
        trust_metric=unusable_metric,
    )
    pos_after_unusable = ekf.get_position()
    assert np.allclose(initial_pos, pos_after_unusable)

    suspicious_metric = GNSSTrustMetric(
        trust_score=0.2,
        is_trusted=False,
        position_variance_scale=50.0,
        velocity_variance_scale=50.0,
        state=GNSSTrustState.SUSPICIOUS,
    )
    ekf.update_gnss(
        gnss_position=np.array([5.0, 0.0, 0.0]),
        trust_metric=suspicious_metric,
    )
    pos_after_suspicious = ekf.get_position()
    assert abs(pos_after_suspicious[0]) < 1.0


def test_navigation_state_machine_with_trust_engine():
    nav_engine = NavigationStateEngine(NavState.NORMAL)

    state_suspicious = nav_engine.process_sensor_health(
        gnss_available=True,
        hdop=1.0,
        trust_state=GNSSTrustState.SUSPICIOUS,
    )
    assert state_suspicious == NavState.GNSS_DEGRADED

    state_trusted = nav_engine.process_sensor_health(
        gnss_available=True,
        hdop=1.0,
        trust_state=GNSSTrustState.TRUSTED,
    )
    assert state_trusted == NavState.NORMAL

    state_unusable = nav_engine.process_sensor_health(
        gnss_available=True,
        hdop=1.0,
        trust_state=GNSSTrustState.UNUSABLE,
    )
    assert state_unusable == NavState.DR
