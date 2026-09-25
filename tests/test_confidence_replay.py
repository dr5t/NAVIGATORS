import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from navigation.adaptive_fusion import AdaptiveFusionEngine
from navigation.gnss_trust import GNSSTrustEngine, GNSSTrustConfig
from navigation.gnss_anomaly import GNSSAnomalyDetector, GNSSAnomalyDetectorConfig
from navigation.confidence import ConfidenceEstimator, ConfidenceConfig
from navigation.interfaces import (
    GNSSTrustState,
    AIVelocityMeasurement,
    RoadHypothesis,
)


def create_navigation_rig():
    ekf = ExtendedKalmanFilter()
    trust_engine = GNSSTrustEngine()
    anomaly_detector = GNSSAnomalyDetector()
    adaptive_engine = AdaptiveFusionEngine()
    confidence_estimator = ConfidenceEstimator()

    initial_pos = np.array([0.0, 0.0, 0.0])
    initial_vel = np.array([0.0, 10.0, 0.0])
    initial_heading = 0.0
    ekf.initialize_from_gnss(initial_pos, initial_vel, initial_heading)

    return {
        "ekf": ekf,
        "trust": trust_engine,
        "anomaly": anomaly_detector,
        "adaptive": adaptive_engine,
        "confidence": confidence_estimator,
    }


def step_rig(rig, dt, accel, gyro, gnss_data=None, ai_vel=None, road_hyp=None, t=0.0):
    ekf = rig["ekf"]
    trust_engine = rig["trust"]
    anomaly_detector = rig["anomaly"]
    adaptive_engine = rig["adaptive"]
    confidence_estimator = rig["confidence"]

    ekf.dt = dt
    pos_2d = ekf.get_position()[:2]
    vel_2d = ekf.get_velocity()[:2]
    heading = ekf.get_heading()
    pos_unc = ekf.get_position_uncertainty()

    anomaly_rep = None
    trust_metric = None
    if gnss_data is not None:
        anomaly_rep = anomaly_detector.detect_anomalies(
            gnss_data=gnss_data,
            dt=dt,
            current_position=pos_2d,
            current_velocity=vel_2d,
            current_heading=heading,
            position_uncertainty=pos_unc,
        )
        trust_metric = trust_engine.evaluate_trust(
            gnss_data=gnss_data,
            dt=dt,
            current_position=pos_2d,
            current_velocity=vel_2d,
            current_heading=heading,
            position_uncertainty=pos_unc,
            anomaly_report=anomaly_rep,
        )

    adaptive_noise = adaptive_engine.compute_adaptive_noise(
        current_mode=ekf.mode.value,
        motion_state="MOVING",
        gnss_trust=trust_metric,
        ai_velocity=ai_vel,
        road_hypothesis=road_hyp,
    )
    ekf.apply_adaptive_noise(adaptive_noise)

    ai_vel_vec = None
    if ai_vel is not None and ai_vel.is_valid:
        ai_vel_vec = np.array([ai_vel.velocity_east, ai_vel.velocity_north], dtype=np.float64)

    ekf.predict(accel, gyro, ai_velocity=ai_vel_vec)

    if gnss_data is not None:
        if trust_metric is not None and trust_metric.state != GNSSTrustState.UNUSABLE and trust_metric.is_trusted:
            pos_enu = np.array([float(gnss_data["east"]), float(gnss_data["north"]), 0.0], dtype=np.float64)
            vel_enu = np.array([float(gnss_data.get("speed", 0.0)) * np.sin(heading),
                                float(gnss_data.get("speed", 0.0)) * np.cos(heading), 0.0], dtype=np.float64)
            ekf.update_gnss(pos_enu, vel_enu, timestamp=t)
    else:
        ekf.set_gnss_denied(timestamp=t)

    if road_hyp is not None and road_hyp.probability > 0.5:
        map_scale = adaptive_noise.measurement_noise_scale_map
        ekf.update_map_constraint(road_hyp.snapped_point, road_hyp.probability, map_scale)

    conf_est = confidence_estimator.compute_confidence(
        covariance=ekf.P,
        dr_duration=float(t - ekf.gnss_outage_start) if ekf.gnss_outage_start is not None else 0.0,
        map_confidence=road_hyp.probability if road_hyp else 0.0,
        gnss_trust=trust_metric,
        ai_velocity=ai_vel,
        road_hypothesis=road_hyp,
        ekf_mode=ekf.mode.value,
        distance_traveled=float(np.linalg.norm(ekf.get_position()[:2])),
        timestamp=t,
    )

    return conf_est


def test_replay_healthy_navigation():
    rig = create_navigation_rig()
    dt = 0.1
    history = []

    for step in range(50):
        t = step * dt
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        gnss = {
            "east": 0.0,
            "north": 10.0 * t,
            "accuracy": 2.0,
            "speed": 10.0,
            "heading": 0.0,
            "timestamp": t,
        }
        ai_vel = AIVelocityMeasurement(
            velocity_north=10.0,
            velocity_east=0.0,
            variance_north=0.16,
            variance_east=0.16,
            latency_ms=2.2,
            is_valid=True,
            status="VALID",
        )
        conf = step_rig(rig, dt, accel, gyro, gnss_data=gnss, ai_vel=ai_vel, t=t)
        history.append(conf)

    final_conf = history[-1]
    assert final_conf.horizontal_accuracy_m < 3.5
    assert final_conf.velocity_uncertainty_mps < 0.6
    assert final_conf.heading_accuracy_deg < 50.0
    assert final_conf.integrity_status == "NOMINAL"
    assert final_conf.confidence_object["sensors"]["gnss"]["status"] == "TRUSTED"
    assert final_conf.confidence_object["units"]["position_horizontal"] == "meters"
    assert final_conf.confidence_object["units"]["velocity_horizontal"] == "meters_per_second"


def test_replay_gnss_outage():
    rig = create_navigation_rig()
    dt = 0.1

    for step in range(20):
        t = step * dt
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        conf_healthy = step_rig(rig, dt, accel, gyro, gnss_data=gnss, ai_vel=ai_vel, t=t)

    healthy_pos_unc = conf_healthy.horizontal_accuracy_m
    outage_history = []

    for step in range(20, 50):
        t = step * dt
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        conf = step_rig(rig, dt, accel, gyro, gnss_data=None, ai_vel=ai_vel, t=t)
        outage_history.append(conf)

    assert rig["ekf"].mode == NavigationMode.DEAD_RECKONING
    assert outage_history[0].horizontal_accuracy_m > healthy_pos_unc
    assert outage_history[-1].horizontal_accuracy_m > outage_history[0].horizontal_accuracy_m
    for i in range(len(outage_history) - 1):
        assert outage_history[i + 1].horizontal_accuracy_m >= outage_history[i].horizontal_accuracy_m
    assert outage_history[-1].confidence_object["sensors"]["gnss"]["status"] == "UNAVAILABLE"
    assert outage_history[-1].confidence_object["sensors"]["ai_velocity"]["is_valid"] is True


def test_replay_longer_gnss_outage():
    rig = create_navigation_rig()
    dt = 0.1

    for step in range(20):
        t = step * dt
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), gnss_data=gnss, ai_vel=ai_vel, t=t)

    short_outage_conf = None
    for step in range(20, 50):
        t = step * dt
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        short_outage_conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                                     gnss_data=None, ai_vel=ai_vel, t=t)

    long_outage_conf = None
    for step in range(50, 150):
        t = step * dt
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        long_outage_conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                                    gnss_data=None, ai_vel=ai_vel, t=t)

    assert long_outage_conf.horizontal_accuracy_m > short_outage_conf.horizontal_accuracy_m
    assert long_outage_conf.heading_accuracy_deg >= short_outage_conf.heading_accuracy_deg
    assert long_outage_conf.overall_confidence < short_outage_conf.overall_confidence
    assert long_outage_conf.integrity_status in ("DEGRADED", "UNRELIABLE")


def test_replay_gnss_recovery():
    rig = create_navigation_rig()
    dt = 0.1
    initial_conf = None
    for step in range(20):
        t = step * dt
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        initial_conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), gnss_data=gnss, ai_vel=ai_vel, t=t)

    outage_conf = None
    for step in range(20, 60):
        t = step * dt
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        outage_conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                               gnss_data=None, ai_vel=ai_vel, t=t)

    peak_outage_pos_unc = outage_conf.horizontal_accuracy_m
    assert peak_outage_pos_unc > initial_conf.horizontal_accuracy_m

    recovery_history = []
    for step in range(60, 100):
        t = step * dt
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                        gnss_data=gnss, ai_vel=ai_vel, t=t)
        recovery_history.append(conf)

    recovered_conf = recovery_history[-1]
    assert recovered_conf.horizontal_accuracy_m < peak_outage_pos_unc
    assert recovered_conf.horizontal_accuracy_m <= 1.0
    assert recovered_conf.integrity_status == "NOMINAL"
    assert recovered_conf.overall_confidence > outage_conf.overall_confidence


def test_replay_gnss_anomaly():
    rig = create_navigation_rig()
    dt = 0.1

    for step in range(30):
        t = step * dt
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), gnss_data=gnss, ai_vel=ai_vel, t=t)

    t = 30 * dt
    jumped_gnss = {
        "east": 80.0,
        "north": 10.0 * t,
        "accuracy": 2.0,
        "speed": 10.0,
        "heading": 0.0,
        "timestamp": t,
    }
    ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
    conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                    gnss_data=jumped_gnss, ai_vel=ai_vel, t=t)

    gnss_diag = conf.confidence_object["sensors"]["gnss"]
    assert gnss_diag["status"] in ("SUSPICIOUS", "UNUSABLE")
    assert gnss_diag["is_trusted"] is False
    assert gnss_diag["variance_scale"] >= 50.0

    pos_east = rig["ekf"].get_position()[0]
    assert pos_east < 5.0


def test_replay_ai_failure():
    rig = create_navigation_rig()
    dt = 0.1

    for step in range(20):
        t = step * dt
        gnss = {"east": 0.0, "north": 10.0 * t, "accuracy": 2.0, "speed": 10.0, "heading": 0.0, "timestamp": t}
        ai_vel = AIVelocityMeasurement(10.0, 0.0, 0.16, 0.16, 2.0, True, "VALID")
        step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]), gnss_data=gnss, ai_vel=ai_vel, t=t)

    t = 20 * dt
    failed_ai = AIVelocityMeasurement(
        velocity_north=45.0,
        velocity_east=0.0,
        variance_north=4.0,
        variance_east=4.0,
        latency_ms=85.0,
        is_valid=False,
        status="HIGH_VARIANCE",
        reason="excessive variance",
    )
    conf = step_rig(rig, dt, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]),
                    gnss_data=None, ai_vel=failed_ai, t=t)

    ai_diag = conf.confidence_object["sensors"]["ai_velocity"]
    assert ai_diag["is_valid"] is False
    assert ai_diag["status"] == "HIGH_VARIANCE"
    assert "excessive variance" in ai_diag["reason"]
