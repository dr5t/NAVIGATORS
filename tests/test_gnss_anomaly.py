import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    GNSSAnomalyStatus,
    GNSSAnomalyType,
    GNSSAnomalyReport,
    GNSSTrustState,
    RoadHypothesis,
)
from navigation.gnss_anomaly import (
    GNSSAnomalyDetector,
    GNSSAnomalyDetectorConfig,
    GNSS_POSITION_JUMP,
    GNSS_IMPOSSIBLE_DISPLACEMENT,
    GNSS_SPEED_INCONSISTENCY,
    GNSS_VELOCITY_DISAGREEMENT,
    GNSS_HEADING_INCONSISTENCY,
    GNSS_NAVIGATION_RESIDUAL_HIGH,
    GNSS_MAP_INCONSISTENCY,
)
from navigation.gnss_trust import GNSSTrustEngine, GNSSTrustConfig
from navigation.ekf import ExtendedKalmanFilter


def test_normal_trajectory():
    detector = GNSSAnomalyDetector()
    for i in range(5):
        gnss_data = {
            "position_enu": [0.0, float(i * 10), 0.0],
            "velocity_enu": [0.0, 10.0, 0.0],
            "speed": 10.0,
            "heading": 0.0,
            "accuracy": 3.0,
            "timestamp": 100.0 + float(i),
        }
        report = detector.detect_anomalies(
            gnss_data=gnss_data,
            dt=1.0,
            current_position=np.array([0.0, float(i * 10)]),
            current_velocity=np.array([0.0, 10.0]),
            current_heading=0.0,
            position_uncertainty=2.0,
        )
        if i == 0:
            assert report.status in (GNSSAnomalyStatus.NORMAL, GNSSAnomalyStatus.UNKNOWN)
        else:
            assert report.status == GNSSAnomalyStatus.NORMAL
            assert report.detected is False
            assert len(report.reason_codes) == 0
            assert report.confidence >= 0.9


def test_synthetic_gnss_jump():
    detector = GNSSAnomalyDetector()
    fix1 = {
        "position_enu": [0.0, 10.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    detector.detect_anomalies(fix1, dt=1.0, current_position=np.array([0.0, 10.0]))

    jumped_fix = {
        "position_enu": [0.0, 210.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 101.0,
    }
    report = detector.detect_anomalies(
        jumped_fix,
        dt=1.0,
        current_position=np.array([0.0, 20.0]),
        position_uncertainty=3.0,
    )
    assert report.status == GNSSAnomalyStatus.ANOMALOUS
    assert report.detected is True
    assert GNSS_POSITION_JUMP in report.reason_codes
    assert GNSS_IMPOSSIBLE_DISPLACEMENT in report.reason_codes
    assert report.metrics["jump_speed_mps"] == 200.0


def test_synthetic_gnss_drift():
    detector = GNSSAnomalyDetector()
    fix1 = {
        "position_enu": [0.0, 10.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 4.0,
        "timestamp": 100.0,
    }
    detector.detect_anomalies(fix1, dt=1.0, current_position=np.array([0.0, 10.0]))

    drifting_fix = {
        "position_enu": [45.0, 20.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 4.0,
        "timestamp": 101.0,
    }
    report = detector.detect_anomalies(
        drifting_fix,
        dt=1.0,
        current_position=np.array([0.0, 20.0]),
        position_uncertainty=3.0,
    )
    assert report.status == GNSSAnomalyStatus.ANOMALOUS
    assert GNSS_NAVIGATION_RESIDUAL_HIGH in report.reason_codes
    assert report.metrics["normalized_innovation"] > 4.0


def test_parallel_road_disagreement():
    detector = GNSSAnomalyDetector()
    gnss_data = {
        "position_enu": [35.0, 100.0, 0.0],
        "velocity_enu": [0.0, 15.0, 0.0],
        "speed": 15.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    road_hyp = RoadHypothesis(
        segment_id="road_main",
        probability=0.9,
        cross_track_distance_m=35.0,
        along_track_distance_m=100.0,
        heading_difference_rad=0.0,
        snapped_point=np.array([0.0, 100.0]),
    )
    report = detector.detect_anomalies(
        gnss_data=gnss_data,
        dt=1.0,
        current_position=np.array([0.0, 100.0]),
        current_velocity=np.array([0.0, 15.0]),
        road_hypothesis=road_hyp,
    )
    assert report.status == GNSSAnomalyStatus.ANOMALOUS
    assert GNSS_MAP_INCONSISTENCY in report.reason_codes
    assert report.metrics["cross_track_m"] == 35.0


def test_sudden_speed_jump():
    detector = GNSSAnomalyDetector()
    fix1 = {
        "position_enu": [0.0, 10.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    detector.detect_anomalies(fix1, dt=1.0)

    fix2 = {
        "position_enu": [0.0, 25.0, 0.0],
        "velocity_enu": [0.0, 50.0, 0.0],
        "speed": 50.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.5,
    }
    report = detector.detect_anomalies(fix2, dt=0.5)
    assert report.status == GNSSAnomalyStatus.ANOMALOUS
    assert GNSS_SPEED_INCONSISTENCY in report.reason_codes
    assert report.metrics["implied_accel_mps2"] == 80.0


def test_velocity_and_heading_disagreements():
    detector = GNSSAnomalyDetector()
    gnss_data = {
        "position_enu": [0.0, 50.0, 0.0],
        "velocity_enu": [0.0, 25.0, 0.0],
        "speed": 25.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    rep_vel = detector.detect_anomalies(
        gnss_data=gnss_data,
        dt=1.0,
        inertial_velocity=np.array([0.0, 5.0]),
    )
    assert rep_vel.status == GNSSAnomalyStatus.ANOMALOUS
    assert GNSS_VELOCITY_DISAGREEMENT in rep_vel.reason_codes

    detector.reset()
    gnss_heading_divergent = {
        "position_enu": [0.0, 50.0, 0.0],
        "velocity_enu": [12.0, 0.0, 0.0],
        "speed": 12.0,
        "heading": np.pi / 2,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    rep_hdg = detector.detect_anomalies(
        gnss_data=gnss_heading_divergent,
        dt=1.0,
        current_heading=0.0,
        current_velocity=np.array([0.0, 12.0]),
    )
    assert rep_hdg.status == GNSSAnomalyStatus.ANOMALOUS
    assert GNSS_HEADING_INCONSISTENCY in rep_hdg.reason_codes


def test_normal_recovery():
    detector = GNSSAnomalyDetector(GNSSAnomalyDetectorConfig(min_consecutive_normal_fixes=3))
    fix1 = {
        "position_enu": [0.0, 10.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    detector.detect_anomalies(fix1, dt=1.0)

    jump = {
        "position_enu": [0.0, 200.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 101.0,
    }
    rep_jump = detector.detect_anomalies(jump, dt=1.0)
    assert rep_jump.status == GNSSAnomalyStatus.ANOMALOUS
    assert detector.is_in_anomaly is True

    norm1 = {
        "position_enu": [0.0, 210.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 102.0,
    }
    r1 = detector.detect_anomalies(norm1, dt=1.0)
    assert r1.status == GNSSAnomalyStatus.NORMAL
    assert detector.consecutive_normal_fixes == 1
    assert detector.is_in_anomaly is True

    norm2 = {
        "position_enu": [0.0, 220.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 103.0,
    }
    r2 = detector.detect_anomalies(norm2, dt=1.0)
    assert r2.status == GNSSAnomalyStatus.NORMAL
    assert detector.consecutive_normal_fixes == 2
    assert detector.is_in_anomaly is True

    norm3 = {
        "position_enu": [0.0, 230.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 104.0,
    }
    r3 = detector.detect_anomalies(norm3, dt=1.0)
    assert r3.status == GNSSAnomalyStatus.NORMAL
    assert detector.consecutive_normal_fixes == 3
    assert detector.is_in_anomaly is False


def test_unknown_status():
    detector = GNSSAnomalyDetector()
    rep_none = detector.detect_anomalies(None)
    assert rep_none.status == GNSSAnomalyStatus.UNKNOWN
    assert rep_none.detected is False

    rep_empty = detector.detect_anomalies({})
    assert rep_empty.status == GNSSAnomalyStatus.UNKNOWN


def test_anomaly_to_trust_engine_integration():
    trust_engine = GNSSTrustEngine()
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.initialize_from_gnss(np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))

    normal_fix = {
        "position_enu": [0.0, 10.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 100.0,
    }
    metric_normal = trust_engine.evaluate_trust(
        gnss_data=normal_fix,
        dt=1.0,
        current_position=ekf.get_position()[:2],
        current_velocity=ekf.get_velocity()[:2],
        current_heading=ekf.get_heading(),
        position_uncertainty=3.0,
    )
    assert metric_normal.state == GNSSTrustState.TRUSTED
    assert metric_normal.position_variance_scale == 1.0

    jump_fix = {
        "position_enu": [0.0, 250.0, 0.0],
        "velocity_enu": [0.0, 10.0, 0.0],
        "speed": 10.0,
        "heading": 0.0,
        "accuracy": 3.0,
        "timestamp": 101.0,
    }
    metric_jump = trust_engine.evaluate_trust(
        gnss_data=jump_fix,
        dt=1.0,
        current_position=ekf.get_position()[:2],
        current_velocity=ekf.get_velocity()[:2],
        current_heading=ekf.get_heading(),
        position_uncertainty=3.0,
    )
    assert metric_jump.state == GNSSTrustState.UNUSABLE
    assert metric_jump.is_trusted is False
    assert GNSS_POSITION_JUMP in metric_jump.diagnostics["reasons"]
    assert len(metric_jump.diagnostics["anomaly_reasons"]) > 0

    pos_before = ekf.get_position().copy()
    ekf.update_gnss(
        gnss_position=np.array(jump_fix["position_enu"]),
        gnss_velocity=np.array(jump_fix["velocity_enu"]),
        timestamp=jump_fix["timestamp"],
        trust_metric=metric_jump,
    )
    pos_after = ekf.get_position()
    assert np.allclose(pos_before, pos_after)


def test_replay_with_gnss_anomalies():
    detector = GNSSAnomalyDetector()
    trust_engine = GNSSTrustEngine(anomaly_detector=detector)
    trajectory_length = 20
    dt = 1.0
    detected_anomalies = []

    for k in range(trajectory_length):
        t = 1000.0 + k * dt
        y = float(k * 10)
        x = 0.0
        s = 10.0
        h = 0.0

        if k == 7:
            y += 180.0
        elif k == 12:
            s = 55.0
        elif k == 16:
            x = 40.0

        gnss_fix = {
            "position_enu": [x, y, 0.0],
            "velocity_enu": [s * np.sin(h), s * np.cos(h), 0.0],
            "speed": s,
            "heading": h,
            "accuracy": 3.0,
            "timestamp": t,
        }

        est_pos = np.array([0.0, float(k * 10)])
        est_vel = np.array([0.0, 10.0])

        road_h = None
        if k == 16:
            road_h = RoadHypothesis(
                segment_id="road_seg",
                probability=0.9,
                cross_track_distance_m=40.0,
                along_track_distance_m=y,
                heading_difference_rad=0.0,
                snapped_point=np.array([0.0, y]),
            )

        metric = trust_engine.evaluate_trust(
            gnss_data=gnss_fix,
            dt=dt,
            current_position=est_pos,
            current_velocity=est_vel,
            current_heading=0.0,
            position_uncertainty=2.0,
            road_distance=road_h.cross_track_distance_m if road_h is not None else None,
        )

        if metric.state in (GNSSTrustState.SUSPICIOUS, GNSSTrustState.UNUSABLE):
            detected_anomalies.append({
                "epoch": k,
                "state": metric.state,
                "reasons": metric.diagnostics.get("anomaly_reasons", []),
            })

    anomaly_epochs = [a["epoch"] for a in detected_anomalies]
    assert 7 in anomaly_epochs
    assert 12 in anomaly_epochs
    assert 16 in anomaly_epochs

    epoch_7 = next(a for a in detected_anomalies if a["epoch"] == 7)
    assert GNSS_POSITION_JUMP in epoch_7["reasons"] or GNSS_IMPOSSIBLE_DISPLACEMENT in epoch_7["reasons"]

    epoch_12 = next(a for a in detected_anomalies if a["epoch"] == 12)
    assert GNSS_SPEED_INCONSISTENCY in epoch_12["reasons"]

    epoch_16 = next(a for a in detected_anomalies if a["epoch"] == 16)
    assert GNSS_MAP_INCONSISTENCY in epoch_16["reasons"] or GNSS_NAVIGATION_RESIDUAL_HIGH in epoch_16["reasons"]
