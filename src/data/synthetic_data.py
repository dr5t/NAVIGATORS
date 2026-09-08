"""
Navigators IDR — Synthetic Data Generator
Generates realistic IMU + GNSS data from parameterized vehicle trajectories.

Enables development and testing of the full navigation pipeline without
requiring the real IO-VNBD dataset.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class TrajectorySegment:
    """Defines one segment of a vehicle trajectory."""
    duration: float          # seconds
    speed: float             # m/s (constant for this segment)
    heading_rate: float = 0.0  # rad/s (positive = turning left)
    acceleration: float = 0.0 # m/s² (longitudinal accel/decel)


@dataclass
class NoiseProfile:
    """Configurable noise parameters to simulate real smartphone IMU."""
    accel_noise_std: float = 0.5      # m/s² — accelerometer white noise
    gyro_noise_std: float = 0.02      # rad/s — gyroscope white noise
    accel_bias: np.ndarray = field(default_factory=lambda: np.array([0.1, -0.05, 0.2]))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.001, -0.002, 0.001]))
    accel_bias_drift: float = 0.001   # m/s²/s — bias random walk
    gyro_bias_drift: float = 0.0001   # rad/s/s — bias random walk
    vibration_amplitude: float = 0.3  # m/s² — engine/road vibration
    vibration_freq: float = 30.0      # Hz — primary vibration frequency
    gnss_noise_std: float = 2.5       # meters — GPS position noise


class SyntheticDataGenerator:
    """
    Generates synthetic IMU and GNSS data for testing the IDR pipeline.

    Creates a parameterized vehicle trajectory and simulates the
    corresponding sensor readings with configurable noise, vibration,
    and bias characteristics.
    """

    def __init__(
        self,
        sample_rate: float = 10.0,
        noise_profile: Optional[NoiseProfile] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            sample_rate: IMU sample rate in Hz.
            noise_profile: Noise characteristics. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.noise = noise_profile or NoiseProfile()
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(
        self, segments: List[TrajectorySegment]
    ) -> Dict[str, np.ndarray]:
        """
        Generate ground truth trajectory from a list of segments.

        Args:
            segments: List of TrajectorySegment defining the path.

        Returns:
            Dict with:
                'timestamps': (N,) time in seconds
                'positions': (N, 2) East-North positions in meters
                'velocities': (N, 2) velocity [v_east, v_north] in m/s
                'headings': (N,) heading in radians (0 = North, π/2 = East)
                'speeds': (N,) scalar speed in m/s
        """
        all_times = []
        all_positions = []
        all_velocities = []
        all_headings = []
        all_speeds = []

        t = 0.0
        pos = np.array([0.0, 0.0])  # East, North
        heading = 0.0  # Start facing North
        speed = segments[0].speed if segments else 0.0

        for seg in segments:
            n_samples = int(seg.duration * self.sample_rate)
            current_speed = speed

            for i in range(n_samples):
                # Update heading
                heading += seg.heading_rate * self.dt

                # Update speed (with acceleration)
                current_speed += seg.acceleration * self.dt
                current_speed = max(0.0, current_speed)  # No reverse

                # Velocity components
                v_east = current_speed * np.sin(heading)
                v_north = current_speed * np.cos(heading)

                # Update position
                pos = pos + np.array([v_east, v_north]) * self.dt

                all_times.append(t)
                all_positions.append(pos.copy())
                all_velocities.append([v_east, v_north])
                all_headings.append(heading)
                all_speeds.append(current_speed)

                t += self.dt

            speed = current_speed

        return {
            "timestamps": np.array(all_times),
            "positions": np.array(all_positions),
            "velocities": np.array(all_velocities),
            "headings": np.array(all_headings),
            "speeds": np.array(all_speeds),
        }

    def generate_imu_data(
        self, trajectory: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Simulate smartphone IMU readings from a ground truth trajectory.

        Generates accelerometer and gyroscope data with realistic noise,
        bias, and vibration characteristics.

        Args:
            trajectory: Output from generate_trajectory().

        Returns:
            Dict with:
                'accel': (N, 3) accelerometer readings [x, y, z] in m/s²
                'gyro': (N, 3) gyroscope readings [x, y, z] in rad/s
        """
        N = len(trajectory["timestamps"])
        velocities = trajectory["velocities"]
        headings = trajectory["headings"]
        speeds = trajectory["speeds"]
        timestamps = trajectory["timestamps"]

        accel = np.zeros((N, 3))
        gyro = np.zeros((N, 3))

        # Running bias state
        accel_bias = self.noise.accel_bias.copy()
        gyro_bias = self.noise.gyro_bias.copy()

        for i in range(N):
            # --- True acceleration (in vehicle frame) ---
            if i > 0:
                dv = velocities[i] - velocities[i - 1]
                # Longitudinal acceleration (forward)
                heading = headings[i]
                fwd = np.array([np.sin(heading), np.cos(heading)])
                a_fwd = np.dot(dv, fwd) / self.dt
                # Centripetal acceleration (lateral)
                right = np.array([np.cos(heading), -np.sin(heading)])
                a_lat = np.dot(dv, right) / self.dt
            else:
                a_fwd = 0.0
                a_lat = 0.0

            # Vehicle-frame accelerometer: [forward, right, down]
            # Add gravity in the down axis (phone Z when flat)
            accel[i, 0] = a_fwd   # Forward
            accel[i, 1] = a_lat   # Right (lateral/centripetal)
            accel[i, 2] = 9.81    # Down (gravity)

            # --- True gyroscope (heading rate around Z) ---
            if i > 0:
                dheading = headings[i] - headings[i - 1]
                # Wrap to [-π, π]
                dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
                gyro[i, 2] = dheading / self.dt  # Yaw rate
            else:
                gyro[i, 2] = 0.0

            # --- Add noise ---
            # White noise
            accel[i] += self.rng.normal(0, self.noise.accel_noise_std, 3)
            gyro[i] += self.rng.normal(0, self.noise.gyro_noise_std, 3)

            # Bias + drift
            accel_bias += self.rng.normal(0, self.noise.accel_bias_drift, 3) * self.dt
            gyro_bias += self.rng.normal(0, self.noise.gyro_bias_drift, 3) * self.dt
            accel[i] += accel_bias
            gyro[i] += gyro_bias

            # Engine/road vibration (sinusoidal)
            if speeds[i] > 1.0:  # Only when moving
                vib_scale = min(1.0, speeds[i] / 15.0)
                t = timestamps[i]
                vibration = self.noise.vibration_amplitude * vib_scale * np.array([
                    np.sin(2 * np.pi * self.noise.vibration_freq * t),
                    np.sin(2 * np.pi * self.noise.vibration_freq * t * 1.3),
                    np.sin(2 * np.pi * self.noise.vibration_freq * t * 0.7),
                ])
                accel[i] += vibration

        return {"accel": accel, "gyro": gyro}

    def generate_gnss_data(
        self,
        trajectory: Dict[str, np.ndarray],
        outage_ranges: Optional[List[Tuple[float, float]]] = None,
        ref_lat: float = 28.6139,
        ref_lon: float = 77.2090,
    ) -> Dict[str, np.ndarray]:
        """
        Generate simulated GNSS fixes with optional outage periods.

        Args:
            trajectory: Output from generate_trajectory().
            outage_ranges: List of (start_time, end_time) tuples where GNSS is denied.
            ref_lat, ref_lon: Reference point for converting ENU → LLA.

        Returns:
            Dict with:
                'positions': (N, 2) noisy GNSS positions in ENU [east, north]
                'available': (N,) boolean mask — True when GNSS fix available
                'lat': (N,) latitude in degrees (noisy)
                'lon': (N,) longitude in degrees (noisy)
        """
        N = len(trajectory["timestamps"])
        true_pos = trajectory["positions"]
        timestamps = trajectory["timestamps"]

        gnss_pos = np.zeros((N, 2))
        gnss_available = np.ones(N, dtype=bool)

        outage_ranges = outage_ranges or []

        for i in range(N):
            t = timestamps[i]

            # Check if in outage
            in_outage = any(start <= t <= end for start, end in outage_ranges)

            if in_outage:
                gnss_available[i] = False
                gnss_pos[i] = np.nan
            else:
                # Add noise to true position
                noise = self.rng.normal(0, self.noise.gnss_noise_std, 2)
                gnss_pos[i] = true_pos[i] + noise

        # Convert ENU to approximate lat/lon
        # Simple linear approximation (sufficient for local area)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))

        lat = ref_lat + true_pos[:, 1] / meters_per_deg_lat
        lon = ref_lon + true_pos[:, 0] / meters_per_deg_lon

        # Add noise to lat/lon where GNSS is available
        lat_noisy = lat.copy()
        lon_noisy = lon.copy()
        for i in range(N):
            if gnss_available[i]:
                lat_noisy[i] += self.rng.normal(0, self.noise.gnss_noise_std / meters_per_deg_lat)
                lon_noisy[i] += self.rng.normal(0, self.noise.gnss_noise_std / meters_per_deg_lon)
            else:
                lat_noisy[i] = np.nan
                lon_noisy[i] = np.nan

        gnss_vel = np.zeros((N, 2))
        true_vel = trajectory["velocities"]
        for i in range(N):
            if gnss_available[i]:
                gnss_vel[i] = true_vel[i] + self.rng.normal(0, 0.2, 2)
            else:
                gnss_vel[i] = np.nan

        return {
            "positions": gnss_pos,
            "velocities": gnss_vel,
            "available": gnss_available,
            "lat": lat_noisy,
            "lon": lon_noisy,
            "true_lat": lat,
            "true_lon": lon,
        }

    def generate_full_scenario(
        self,
        segments: Optional[List[TrajectorySegment]] = None,
        outage_ranges: Optional[List[Tuple[float, float]]] = None,
        ref_lat: float = 28.6139,
        ref_lon: float = 77.2090,
    ) -> Dict[str, Any]:
        """
        Generate a complete test scenario with trajectory, IMU, and GNSS data.

        If no segments provided, generates a default urban driving scenario
        with turns, stops, acceleration, and a GNSS outage (tunnel).

        Args:
            segments: Optional custom trajectory segments.
            outage_ranges: Optional GNSS outage periods.
            ref_lat, ref_lon: Reference coordinates.

        Returns:
            Complete scenario dict with all data streams and ground truth.
        """
        if segments is None:
            # Default scenario: Urban drive with tunnel
            segments = [
                TrajectorySegment(duration=5.0, speed=0.0, acceleration=3.0),    # Start from stop
                TrajectorySegment(duration=15.0, speed=15.0),                     # Cruise at 54 km/h
                TrajectorySegment(duration=5.0, speed=15.0, heading_rate=0.1),    # Gentle left turn
                TrajectorySegment(duration=20.0, speed=15.0),                     # Straight (enters tunnel)
                TrajectorySegment(duration=3.0, speed=15.0, heading_rate=-0.15),  # Right turn in tunnel
                TrajectorySegment(duration=10.0, speed=15.0),                     # Exit tunnel
                TrajectorySegment(duration=5.0, speed=15.0, acceleration=-2.0),   # Braking
                TrajectorySegment(duration=8.0, speed=0.0),                       # Stopped at light
                TrajectorySegment(duration=5.0, speed=0.0, acceleration=2.5),     # Accelerate
                TrajectorySegment(duration=15.0, speed=12.0),                     # Cruise
                TrajectorySegment(duration=5.0, speed=12.0, heading_rate=0.2),    # Left turn
                TrajectorySegment(duration=10.0, speed=12.0),                     # Final straight
            ]

        if outage_ranges is None:
            # GNSS outage during the "tunnel" portion (roughly 25s–55s)
            outage_ranges = [(25.0, 55.0)]

        trajectory = self.generate_trajectory(segments)
        imu_data = self.generate_imu_data(trajectory)
        gnss_data = self.generate_gnss_data(trajectory, outage_ranges, ref_lat, ref_lon)

        return {
            "trajectory": trajectory,
            "imu": imu_data,
            "gnss": gnss_data,
            "metadata": {
                "sample_rate": self.sample_rate,
                "ref_lat": ref_lat,
                "ref_lon": ref_lon,
                "outage_ranges": outage_ranges,
                "total_duration": float(trajectory["timestamps"][-1]),
                "total_distance": float(np.sum(np.linalg.norm(
                    np.diff(trajectory["positions"], axis=0), axis=1
                ))),
            },
        }
