"""Tests for Dead Reckoning, NHC, ZUPT, and coordinate transforms."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.dead_reckoning import DeadReckoningEngine
from navigation.nhc import NonHolonomicConstraints
from navigation.zupt import ZUPTDetector
from utils.coordinates import (
    lla_to_ecef, ecef_to_lla, lla_to_enu, enu_to_lla,
    haversine_distance, compute_bearing, destination_point,
)
from data.synthetic_data import SyntheticDataGenerator, TrajectorySegment


class TestDeadReckoning:
    def test_stationary_vehicle_stays_put(self):
        dr = DeadReckoningEngine(dt=0.1)
        dr.start(np.array([0.0, 0.0]), heading=0.0, speed=0.0, timestamp=0.0)

        pos = np.array([0.0, 0.0])
        for i in range(100):
            pos = dr.update(ai_speed=0.0, timestamp=i * 0.1)

        assert np.linalg.norm(pos) < 0.1, "Stationary vehicle should not drift"

    def test_straight_line_forward(self):
        dr = DeadReckoningEngine(dt=0.1)
        dr.start(np.array([0.0, 0.0]), heading=0.0, speed=10.0, timestamp=0.0)

        pos = np.array([0.0, 0.0])
        for i in range(100):  # 10 seconds
            pos = dr.update(ai_speed=10.0, timestamp=i * 0.1)

        # After 10s at 10 m/s heading North: expect ~100m North
        assert abs(pos[1] - 100.0) < 5.0, f"Expected ~100m North, got {pos[1]:.1f}m"
        assert abs(pos[0]) < 5.0, f"Should not drift East: {pos[0]:.1f}m"

    def test_distance_tracking(self):
        dr = DeadReckoningEngine(dt=0.1)
        dr.start(np.array([0.0, 0.0]), heading=0.0, speed=10.0, timestamp=0.0)

        for i in range(100):
            dr.update(ai_speed=10.0, timestamp=i * 0.1)

        assert abs(dr.distance_traveled - 100.0) < 2.0

    def test_confidence_decays(self):
        dr = DeadReckoningEngine(dt=0.1)
        dr.start(np.array([0.0, 0.0]), heading=0.0, speed=10.0, timestamp=0.0)

        initial_conf = dr.get_confidence()
        for i in range(300):
            dr.update(ai_speed=10.0, timestamp=i * 0.1)

        final_conf = dr.get_confidence()
        assert final_conf < initial_conf, "Confidence should decay during DR"

    def test_stop_returns_summary(self):
        dr = DeadReckoningEngine(dt=0.1)
        dr.start(np.array([0.0, 0.0]), heading=0.0, speed=10.0, timestamp=0.0)

        for i in range(100):
            dr.update(ai_speed=10.0, timestamp=i * 0.1)

        summary = dr.stop()
        assert "outage_duration_s" in summary
        assert "distance_traveled_m" in summary
        assert "drift_percent" in summary
        assert not dr.is_active


class TestNHC:
    def test_zero_lateral_velocity(self):
        nhc = NonHolonomicConstraints()
        # Vehicle heading North, velocity has lateral component
        vel = np.array([3.0, 10.0])  # [East, North] - 3 m/s lateral
        heading = 0.0  # North

        constrained = nhc.apply_constraints(vel, heading)
        # After NHC, lateral velocity should be removed
        body_vel = nhc.compute_body_velocity(constrained, heading)
        assert abs(body_vel[1]) < 0.01, "Lateral velocity should be zero after NHC"

    def test_forward_velocity_preserved(self):
        nhc = NonHolonomicConstraints()
        vel = np.array([0.0, 10.0])  # Pure North velocity
        heading = 0.0

        constrained = nhc.apply_constraints(vel, heading)
        assert np.allclose(constrained, vel, atol=0.01)

    def test_heading_rate_clipping(self):
        nhc = NonHolonomicConstraints(max_heading_rate=0.5)
        assert nhc.check_heading_rate(1.0) == 0.5
        assert nhc.check_heading_rate(-1.0) == -0.5
        assert nhc.check_heading_rate(0.3) == 0.3

    def test_validation(self):
        nhc = NonHolonomicConstraints()
        result = nhc.validate_velocity(np.array([0.0, 10.0]), heading=0.0)
        assert result["valid"]
        assert abs(result["lateral_velocity"]) < 0.5


class TestZUPT:
    def test_detects_stationary(self):
        zupt = ZUPTDetector(
            accel_variance_threshold=0.1,
            gyro_variance_threshold=0.05,
            detection_window=5,
        )

        # Feed constant (gravity-only) readings
        for _ in range(10):
            accel = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, 0.01, 3)
            gyro = np.random.normal(0, 0.001, 3)
            zupt.update(accel, gyro)

        assert zupt.is_stationary, "Should detect stationary vehicle"

    def test_detects_moving(self):
        zupt = ZUPTDetector(detection_window=5)

        # Feed high-variance readings (moving vehicle)
        for _ in range(10):
            accel = np.array([2.0, 0.5, 9.81]) + np.random.normal(0, 1.0, 3)
            gyro = np.array([0.0, 0.0, 0.2]) + np.random.normal(0, 0.1, 3)
            zupt.update(accel, gyro)

        assert not zupt.is_stationary, "Should detect moving vehicle"

    def test_zupt_measurement(self):
        zupt = ZUPTDetector()
        z, R = zupt.get_zupt_measurement()
        assert np.allclose(z, 0.0), "ZUPT measurement should be zero velocity"
        assert R.shape == (3, 3)


class TestCoordinates:
    def test_lla_ecef_roundtrip(self):
        lat, lon, alt = 28.6139, 77.2090, 100.0
        ecef = lla_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_lla(ecef[0], ecef[1], ecef[2])

        assert abs(lat2 - lat) < 1e-8
        assert abs(lon2 - lon) < 1e-8
        assert abs(alt2 - alt) < 0.01

    def test_enu_roundtrip(self):
        ref_lat, ref_lon = 28.6139, 77.2090
        lat, lon, alt = 28.615, 77.210, 0.0

        enu = lla_to_enu(lat, lon, alt, ref_lat, ref_lon)
        lat2, lon2, alt2 = enu_to_lla(enu[0], enu[1], enu[2], ref_lat, ref_lon)

        assert abs(lat2 - lat) < 1e-6
        assert abs(lon2 - lon) < 1e-6

    def test_haversine_known_distance(self):
        # Delhi to Agra ≈ 178 km (great-circle)
        dist = haversine_distance(28.6139, 77.2090, 27.1767, 78.0081)
        assert 170_000 < dist < 190_000, f"Expected ~178km, got {dist/1000:.0f}km"

    def test_haversine_zero_distance(self):
        dist = haversine_distance(28.6139, 77.2090, 28.6139, 77.2090)
        assert dist < 0.01

    def test_bearing_north(self):
        # Point directly north
        bearing = compute_bearing(28.0, 77.0, 29.0, 77.0)
        assert abs(bearing) < 0.01 or abs(bearing - 2 * np.pi) < 0.01

    def test_destination_point(self):
        lat, lon = 28.6139, 77.2090
        # Go 1000m North
        dest_lat, dest_lon = destination_point(lat, lon, 0.0, 1000.0)
        dist = haversine_distance(lat, lon, dest_lat, dest_lon)
        assert abs(dist - 1000.0) < 1.0


class TestSyntheticData:
    def test_generates_valid_scenario(self):
        gen = SyntheticDataGenerator(sample_rate=10.0, seed=42)
        scenario = gen.generate_full_scenario()

        assert "trajectory" in scenario
        assert "imu" in scenario
        assert "gnss" in scenario
        assert "metadata" in scenario

        N = len(scenario["trajectory"]["timestamps"])
        assert N > 100
        assert scenario["imu"]["accel"].shape == (N, 3)
        assert scenario["imu"]["gyro"].shape == (N, 3)
        assert len(scenario["gnss"]["available"]) == N

    def test_gnss_outage_exists(self):
        gen = SyntheticDataGenerator(sample_rate=10.0, seed=42)
        scenario = gen.generate_full_scenario()

        available = scenario["gnss"]["available"]
        assert not np.all(available), "Should have GNSS outage"
        assert np.any(available), "Should have some GNSS availability"

    def test_reproducible_with_seed(self):
        gen1 = SyntheticDataGenerator(sample_rate=10.0, seed=123)
        gen2 = SyntheticDataGenerator(sample_rate=10.0, seed=123)

        s1 = gen1.generate_full_scenario()
        s2 = gen2.generate_full_scenario()

        assert np.allclose(s1["trajectory"]["positions"], s2["trajectory"]["positions"])
