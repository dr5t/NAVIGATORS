"""Tests for the Extended Kalman Filter."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.ekf import ExtendedKalmanFilter, NavigationMode


class TestEKFInitialization:
    def test_initial_state_is_zero(self):
        ekf = ExtendedKalmanFilter()
        assert np.allclose(ekf.x, 0.0)
        assert ekf.x.shape == (15,)

    def test_initial_covariance_is_positive_definite(self):
        ekf = ExtendedKalmanFilter()
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0)

    def test_initial_mode_is_gnss_ins(self):
        ekf = ExtendedKalmanFilter()
        assert ekf.mode == NavigationMode.GNSS_INS

    def test_initialize_from_gnss(self):
        ekf = ExtendedKalmanFilter()
        ekf.initialize_from_gnss(
            position=np.array([100.0, 200.0]),
            velocity=np.array([5.0, 10.0]),
            heading=0.5,
        )
        assert ekf.x[0] == 100.0  # East
        assert ekf.x[1] == 200.0  # North
        assert ekf.x[3] == 5.0    # v_east
        assert ekf.x[4] == 10.0   # v_north
        assert ekf.x[8] == 0.5    # yaw


class TestEKFPrediction:
    def test_predict_maintains_covariance_positive_definite(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        for _ in range(100):
            ekf.predict(accel, gyro)

        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), "Covariance became non-positive-definite"

    def test_predict_grows_position_uncertainty(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        initial_unc = ekf.get_position_uncertainty()

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        for _ in range(50):
            ekf.predict(accel, gyro)

        final_unc = ekf.get_position_uncertainty()
        assert final_unc > initial_unc, "Uncertainty should grow during prediction"

    def test_stationary_vehicle_stays_near_origin(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        accel = np.array([0.0, 0.0, 9.81])  # Only gravity
        gyro = np.array([0.0, 0.0, 0.0])

        for _ in range(10):
            ekf.predict(accel, gyro)

        pos = ekf.get_position()
        # Position should stay close to zero for a stationary vehicle
        assert np.linalg.norm(pos[:2]) < 5.0


class TestEKFUpdate:
    def test_gnss_update_reduces_uncertainty(self):
        ekf = ExtendedKalmanFilter(dt=0.1)

        # Predict for a while to grow uncertainty
        for _ in range(20):
            ekf.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3))

        unc_before = ekf.get_position_uncertainty()

        # GNSS update
        ekf.update_gnss(np.array([0.0, 0.0, 0.0]))
        unc_after = ekf.get_position_uncertainty()

        assert unc_after < unc_before, "GNSS update should reduce uncertainty"

    def test_gnss_update_corrects_position(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.initialize_from_gnss(np.array([0.0, 0.0]))

        # Predict away from origin
        for _ in range(30):
            ekf.predict(np.array([1.0, 0.0, 9.81]), np.zeros(3))

        # GNSS says we're at (10, 20)
        ekf.update_gnss(np.array([10.0, 20.0, 0.0]))
        pos = ekf.get_position()

        # Position should move toward GNSS measurement
        assert abs(pos[0] - 10.0) < abs(pos[0])  # Closer to 10 than before

    def test_zupt_resets_velocity(self):
        ekf = ExtendedKalmanFilter(dt=0.1)
        ekf.x[3:6] = np.array([5.0, 3.0, 0.1])  # Non-zero velocity

        ekf.update_zupt(velocity_sigma=0.01)
        vel = ekf.get_velocity()

        assert np.linalg.norm(vel) < 1.0, "ZUPT should drive velocity toward zero"


class TestNavigationMode:
    def test_gnss_denied_switches_to_dr(self):
        ekf = ExtendedKalmanFilter()
        ekf.set_gnss_denied(timestamp=10.0)
        assert ekf.mode == NavigationMode.DEAD_RECKONING

    def test_gnss_restored_triggers_reacquisition(self):
        ekf = ExtendedKalmanFilter()
        ekf.set_gnss_denied()

        ekf.update_gnss(np.array([0.0, 0.0, 0.0]))
        assert ekf.mode == NavigationMode.REACQUISITION

    def test_state_summary_has_all_fields(self):
        ekf = ExtendedKalmanFilter()
        summary = ekf.get_state_summary()

        expected_keys = ["mode", "position_enu", "velocity_enu", "speed_ms",
                         "speed_kmh", "heading_deg", "position_uncertainty_m",
                         "accel_bias", "gyro_bias"]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"
