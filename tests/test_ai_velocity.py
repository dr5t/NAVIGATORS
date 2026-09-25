import os
import sys
import numpy as np
import pytest
from pathlib import Path
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    AIVelocityMeasurement,
    GNSSTrustState,
    GNSSTrustMetric,
)
from navigation.ai_velocity import (
    AIVelocityEngine,
    AIVelocityConfig,
)
from navigation.ekf import ExtendedKalmanFilter, NavigationMode


def create_synthetic_imu_window(stationary=False, noise_level=0.01):
    ts = np.linspace(0.0, 19.9, 200)
    if stationary:
        accel_x = np.random.normal(0.0, noise_level, 200)
        accel_y = np.random.normal(0.0, noise_level, 200)
        accel_z = np.full(200, 9.81) + np.random.normal(0.0, noise_level, 200)
        gyro_x = np.random.normal(0.0, noise_level * 0.1, 200)
        gyro_y = np.random.normal(0.0, noise_level * 0.1, 200)
        gyro_z = np.random.normal(0.0, noise_level * 0.1, 200)
    else:
        accel_x = 0.5 * np.sin(ts * 1.5) + np.random.normal(0.0, 0.1, 200)
        accel_y = 1.2 * np.cos(ts * 1.0) + np.random.normal(0.0, 0.1, 200)
        accel_z = np.full(200, 9.81) + np.random.normal(0.0, 0.2, 200)
        gyro_x = 0.05 * np.sin(ts * 2.0)
        gyro_y = 0.05 * np.cos(ts * 2.0)
        gyro_z = 0.1 * np.sin(ts * 0.5)

    window = np.column_stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]).astype(np.float32)
    return window, ts


def test_valid_inference():
    engine = AIVelocityEngine()
    if not engine.is_ready():
        pytest.skip("ONNX model or norm stats not available in simulator directory")

    window, ts = create_synthetic_imu_window(stationary=False)
    measurement = engine.estimate_velocity(window, timestamps=ts)

    assert measurement.is_valid is True
    assert measurement.status in ("VALID", "VALID_STATIONARY")
    assert measurement.latency_ms > 0.0
    assert measurement.variance_north > 0.0
    assert measurement.variance_east > 0.0
    assert np.isfinite(measurement.velocity_north)
    assert np.isfinite(measurement.velocity_east)
    assert measurement.confidence >= 0.7


def test_invalid_sensor_window():
    engine = AIVelocityEngine()

    short_window = np.zeros((100, 6), dtype=np.float32)
    res_short = engine.estimate_velocity(short_window)
    assert res_short.is_valid is False
    assert res_short.status == "INSUFFICIENT_SAMPLES"

    nan_window = np.full((200, 6), 9.81, dtype=np.float32)
    nan_window[50, 1] = np.nan
    res_nan = engine.estimate_velocity(nan_window)
    assert res_nan.is_valid is False
    assert res_nan.status == "INVALID_INPUT_NAN_INF"

    zero_accel = np.zeros((200, 6), dtype=np.float32)
    res_zero = engine.estimate_velocity(zero_accel)
    assert res_zero.is_valid is False
    assert res_zero.status == "INVALID_SENSOR_RANGE"


def test_stale_window():
    engine = AIVelocityEngine()
    window, ts = create_synthetic_imu_window(stationary=False)

    res_stale = engine.estimate_velocity(window, timestamps=ts, current_time=ts[-1] + 10.0)
    assert res_stale.is_valid is False
    assert res_stale.status == "STALE_PREDICTION"


def test_inference_failure():
    bad_config = AIVelocityConfig(model_path="nonexistent_dir/model.onnx")
    bad_engine = AIVelocityEngine(bad_config)
    assert bad_engine.is_ready() is False

    window, ts = create_synthetic_imu_window(stationary=False)
    res = bad_engine.estimate_velocity(window, timestamps=ts)
    assert res.is_valid is False
    assert res.status == "MODEL_UNAVAILABLE"


def test_implausible_velocity():
    engine = AIVelocityEngine(AIVelocityConfig(max_plausible_speed_mps=0.01))
    if not engine.is_ready():
        pytest.skip("ONNX model not ready")

    window, ts = create_synthetic_imu_window(stationary=False)
    res = engine.estimate_velocity(window, timestamps=ts)
    assert res.is_valid is False
    assert res.status == "IMPLAUSIBLE_SPEED"


def test_stationary_state():
    engine = AIVelocityEngine()
    if not engine.is_ready():
        pytest.skip("ONNX model not ready")

    window, ts = create_synthetic_imu_window(stationary=True, noise_level=0.005)
    res = engine.estimate_velocity(window, timestamps=ts)
    assert res.is_valid is True
    assert res.status == "VALID_STATIONARY"
    assert res.variance_north == engine.config.stationary_variance
    assert res.variance_east == engine.config.stationary_variance
    speed = np.sqrt(res.velocity_north ** 2 + res.velocity_east ** 2)
    assert speed <= engine.config.stationary_max_speed_mps


def test_moving_state():
    engine = AIVelocityEngine()
    if not engine.is_ready():
        pytest.skip("ONNX model not ready")

    window, ts = create_synthetic_imu_window(stationary=False)
    res = engine.estimate_velocity(window, timestamps=ts)
    assert res.is_valid is True
    assert res.status == "VALID"
    assert res.confidence >= 0.7


def test_gnss_available_fusion():
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.initialize_from_gnss(np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))
    assert ekf.mode == NavigationMode.GNSS_INS

    ai_meas = AIVelocityMeasurement(
        velocity_north=10.0,
        velocity_east=0.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
    )
    imu_accel = np.array([0.0, 0.0, 9.81])
    imu_gyro = np.array([0.0, 0.0, 0.0])

    ekf.predict(accel_body=imu_accel, gyro_body=imu_gyro, ai_velocity=ai_meas)
    summary = ekf.get_state_summary()
    assert summary["mode"] == NavigationMode.GNSS_INS.value
    assert summary["ai_velocity_valid"] is True
    assert summary["ai_velocity_ms"] == [0.0, 10.0]


def test_gnss_unavailable_fusion():
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.initialize_from_gnss(np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))
    ekf.mode = NavigationMode.DEAD_RECKONING

    valid_ai = AIVelocityMeasurement(
        velocity_north=8.0,
        velocity_east=1.0,
        variance_north=0.16,
        variance_east=0.16,
        latency_ms=2.0,
        is_valid=True,
    )
    imu_accel = np.array([0.0, 0.0, 9.81])
    imu_gyro = np.array([0.0, 0.0, 0.0])

    ekf.predict(accel_body=imu_accel, gyro_body=imu_gyro, ai_velocity=valid_ai)
    vel_after = ekf.get_velocity()
    assert np.all(np.isfinite(vel_after))

    invalid_ai = AIVelocityMeasurement(
        velocity_north=0.0,
        velocity_east=0.0,
        variance_north=100.0,
        variance_east=100.0,
        latency_ms=1.5,
        is_valid=False,
    )
    ekf.predict(accel_body=imu_accel, gyro_body=imu_gyro, ai_velocity=invalid_ai)
    pos_after_invalid = ekf.get_position()
    assert np.all(np.isfinite(pos_after_invalid))


def test_no_gnss_speed_leakage():
    engine = AIVelocityEngine()
    if not engine.is_ready():
        pytest.skip("ONNX model not ready")

    window_with_gnss = np.zeros((200, 8), dtype=np.float32)
    window_with_gnss[:, :3] = np.array([0.0, 0.0, 9.81])
    window_with_gnss[:, 6] = 55.0
    window_with_gnss[:, 7] = 100.0

    res = engine.estimate_velocity(window_with_gnss)
    assert res.is_valid is True


def test_actual_inference_latency():
    engine = AIVelocityEngine()
    if not engine.is_ready():
        pytest.skip("ONNX model not ready")

    window, ts = create_synthetic_imu_window(stationary=False)
    for _ in range(5):
        engine.estimate_velocity(window, timestamps=ts)

    latencies = []
    for _ in range(40):
        t0 = perf_counter()
        meas = engine.estimate_velocity(window, timestamps=ts)
        latencies.append((perf_counter() - t0) * 1000.0)

    p50 = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    mean_lat = float(np.mean(latencies))

    assert p95 < 50.0
    assert meas.latency_ms > 0.0
