import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    AdaptiveNoiseParameters,
    GNSSTrustState,
    GNSSTrustMetric,
    AIVelocityMeasurement,
    RoadHypothesis,
    MapConstraint,
)
from navigation.adaptive_fusion import (
    AdaptiveFusionEngine,
    AdaptiveFusionConfig,
)
from navigation.ekf import ExtendedKalmanFilter, NavigationMode


def test_gnss_healthy():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.95,
        is_trusted=True,
        position_variance_scale=1.0,
        velocity_variance_scale=1.0,
        state=GNSSTrustState.TRUSTED,
    )
    ai_vel = AIVelocityMeasurement(
        velocity_north=10.0,
        velocity_east=0.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
        status="VALID",
        confidence=0.95,
    )
    map_hyp = RoadHypothesis(
        segment_id="road_1",
        probability=0.9,
        cross_track_distance_m=2.0,
        along_track_distance_m=50.0,
        heading_difference_rad=0.0,
        snapped_point=np.array([0.0, 50.0]),
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=ai_vel,
        road_hypothesis=map_hyp,
    )

    assert params.measurement_noise_scale_gnss_pos == 1.0
    assert params.measurement_noise_scale_gnss_vel == 1.0
    assert params.telemetry["gnss_weight"] == 1.0
    assert params.telemetry["gnss_status"] == "HEALTHY"
    assert params.measurement_noise_scale_ai_vel == 1.5
    assert params.measurement_noise_scale_map >= 1.0
    assert len(params.downweighted_reasons) == 0
    assert len(params.rejected_reasons) == 0


def test_gnss_degraded():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.55,
        is_trusted=True,
        position_variance_scale=6.5,
        velocity_variance_scale=3.25,
        state=GNSSTrustState.DEGRADED,
        diagnostics={"reasons": ["degraded_accuracy"]},
    )
    ai_vel = AIVelocityMeasurement(
        velocity_north=10.0,
        velocity_east=0.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
        status="VALID",
        confidence=0.95,
    )
    map_hyp = RoadHypothesis(
        segment_id="road_1",
        probability=0.95,
        cross_track_distance_m=3.0,
        along_track_distance_m=50.0,
        heading_difference_rad=0.0,
        snapped_point=np.array([0.0, 50.0]),
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_DEGRADED",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=ai_vel,
        road_hypothesis=map_hyp,
    )

    assert params.measurement_noise_scale_gnss_pos == 6.5
    assert params.telemetry["gnss_weight"] < 0.25
    assert params.telemetry["gnss_status"] == "DEGRADED"
    assert params.measurement_noise_scale_ai_vel == 1.0
    assert params.telemetry["ai_weight"] == 1.0
    assert params.measurement_noise_scale_map < 1.0
    assert any("degraded" in r.lower() for r in params.downweighted_reasons)


def test_gnss_suspicious():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.25,
        is_trusted=False,
        position_variance_scale=50.0,
        velocity_variance_scale=50.0,
        state=GNSSTrustState.SUSPICIOUS,
        diagnostics={"reasons": ["heading_inconsistency"]},
    )
    ai_vel = AIVelocityMeasurement(
        velocity_north=10.0,
        velocity_east=0.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
        status="VALID",
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_DEGRADED",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=ai_vel,
    )

    assert params.measurement_noise_scale_gnss_pos >= 50.0
    assert params.telemetry["gnss_weight"] <= 0.02
    assert params.telemetry["gnss_status"] == "SUSPICIOUS"
    assert any("suspicious" in r.lower() for r in params.downweighted_reasons)


def test_gnss_lost():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.0,
        is_trusted=False,
        position_variance_scale=10000.0,
        velocity_variance_scale=10000.0,
        state=GNSSTrustState.UNUSABLE,
        diagnostics={"reasons": ["signal_outage"]},
    )
    ai_vel = AIVelocityMeasurement(
        velocity_north=10.0,
        velocity_east=0.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
        status="VALID",
        confidence=0.95,
    )
    map_hyp = RoadHypothesis(
        segment_id="road_1",
        probability=0.9,
        cross_track_distance_m=2.0,
        along_track_distance_m=50.0,
        heading_difference_rad=0.0,
        snapped_point=np.array([0.0, 50.0]),
    )

    params = engine.compute_adaptive_noise(
        current_mode="DEAD_RECKONING",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=ai_vel,
        road_hypothesis=map_hyp,
    )

    assert params.measurement_noise_scale_gnss_pos >= 10000.0
    assert params.telemetry["gnss_weight"] == 0.0
    assert params.telemetry["gnss_status"] == "UNUSABLE"
    assert any("gnss rejected" in r.lower() for r in params.rejected_reasons)
    assert params.measurement_noise_scale_ai_vel == 0.8
    assert params.measurement_noise_scale_map < 1.0
    assert params.process_noise_scale_pos > 1.0


def test_ai_unavailable():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.9,
        is_trusted=True,
        position_variance_scale=1.0,
        velocity_variance_scale=1.0,
        state=GNSSTrustState.TRUSTED,
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=None,
    )

    assert params.measurement_noise_scale_ai_vel >= 10000.0
    assert params.telemetry["ai_weight"] == 0.0
    assert params.telemetry["ai_status"] == "UNAVAILABLE"
    assert any("ai velocity rejected" in r.lower() for r in params.rejected_reasons)


def test_ai_degraded():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.9,
        is_trusted=True,
        position_variance_scale=1.0,
        velocity_variance_scale=1.0,
        state=GNSSTrustState.TRUSTED,
    )
    ai_stale = AIVelocityMeasurement(
        velocity_north=0.0,
        velocity_east=0.0,
        variance_north=100.0,
        variance_east=100.0,
        latency_ms=1.0,
        is_valid=False,
        status="STALE_PREDICTION",
        reason="prediction_stale",
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        ai_velocity=ai_stale,
    )

    assert params.measurement_noise_scale_ai_vel >= 10000.0
    assert params.telemetry["ai_weight"] == 0.0
    assert any("ai velocity rejected" in r.lower() for r in params.rejected_reasons)


def test_map_unavailable():
    engine = AdaptiveFusionEngine()
    gnss_trust = GNSSTrustMetric(
        trust_score=0.9,
        is_trusted=True,
        position_variance_scale=1.0,
        velocity_variance_scale=1.0,
        state=GNSSTrustState.TRUSTED,
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        map_constraint=None,
    )

    assert params.measurement_noise_scale_map >= 10000.0
    assert params.telemetry["map_weight"] == 0.0
    assert any("map constraint rejected" in r.lower() for r in params.rejected_reasons)

    off_road = RoadHypothesis(
        segment_id="far_road",
        probability=0.2,
        cross_track_distance_m=50.0,
        along_track_distance_m=0.0,
        heading_difference_rad=0.0,
        snapped_point=np.array([0.0, 0.0]),
    )
    params_off = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="MOVING",
        gnss_trust=gnss_trust,
        road_hypothesis=off_road,
    )
    assert params_off.measurement_noise_scale_map >= 10000.0
    assert params_off.telemetry["map_status"] == "REJECTED"


def test_multiple_simultaneous_degradations():
    engine = AdaptiveFusionEngine()

    params = engine.compute_adaptive_noise(
        current_mode="DEAD_RECKONING",
        motion_state="MOVING",
        gnss_trust=None,
        ai_velocity=None,
        map_constraint=None,
    )

    assert params.measurement_noise_scale_gnss_pos >= 10000.0
    assert params.measurement_noise_scale_ai_vel >= 10000.0
    assert params.measurement_noise_scale_map >= 10000.0
    assert params.telemetry["gnss_weight"] == 0.0
    assert params.telemetry["ai_weight"] == 0.0
    assert params.telemetry["map_weight"] == 0.0

    rejected_lower = [r.lower() for r in params.rejected_reasons]
    assert any("gnss" in r for r in rejected_lower)
    assert any("ai" in r for r in rejected_lower)
    assert any("map" in r for r in rejected_lower)


def test_stationary_damping():
    engine = AdaptiveFusionEngine()
    ai_stationary = AIVelocityMeasurement(
        velocity_north=0.0,
        velocity_east=0.0,
        variance_north=0.01,
        variance_east=0.01,
        latency_ms=1.5,
        is_valid=True,
        status="VALID_STATIONARY",
    )

    params = engine.compute_adaptive_noise(
        current_mode="GNSS_INS",
        motion_state="STATIONARY",
        ai_velocity=ai_stationary,
    )

    assert params.process_noise_scale_pos <= 0.1
    assert params.process_noise_scale_vel <= 0.1
    assert params.telemetry["is_stationary"] is True
    assert params.measurement_noise_scale_ai_vel <= 0.15


def test_ekf_adaptive_integration():
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.initialize_from_gnss(np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))
    engine = AdaptiveFusionEngine()

    noise_params = engine.compute_adaptive_noise(
        current_mode=ekf.mode.value,
        motion_state="MOVING",
        gnss_trust=GNSSTrustMetric(0.9, True, 1.0, 1.0, GNSSTrustState.TRUSTED),
    )
    ekf.apply_adaptive_noise(noise_params)

    ekf.predict(accel_body=np.array([0.0, 0.0, 9.81]), gyro_body=np.array([0.0, 0.0, 0.0]))
    summary = ekf.get_state_summary()
    assert "adaptive_fusion" in summary
    assert summary["adaptive_fusion"]["measurement_noise_scale_gnss_pos"] == 1.0

    ekf.mode = NavigationMode.DEAD_RECKONING
    pos_before_snap = ekf.get_position().copy()
    ekf.update_map_constraint(snapped_position=np.array([0.0, 5.0]), confidence=0.9, covariance_scale=0.5)
    pos_after_snap = ekf.get_position()
    assert np.all(np.isfinite(pos_after_snap))
