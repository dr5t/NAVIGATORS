"""
Navigators IDR — Dead Reckoning Engine
Integrates AI velocity estimates with heading to maintain position during GNSS denial.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class DeadReckoningEngine:
    """
    Dead reckoning engine for GNSS-denied navigation.

    Integrates AI-predicted velocity and gyroscope-derived heading
    to maintain continuous position estimates when GNSS is lost.
    Applies Non-Holonomic Constraints to reduce drift.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_outage_duration: float = 120.0,
        drift_warning_threshold: float = 0.1,
    ):
        """
        Args:
            dt: Time step in seconds.
            max_outage_duration: Maximum DR duration before warning (seconds).
            drift_warning_threshold: Drift warning as fraction of distance.
        """
        self.dt = dt
        self.max_outage_duration = max_outage_duration
        self.drift_warning_threshold = drift_warning_threshold

        # State
        self.position = np.zeros(2)          # East, North (meters)
        self.heading = 0.0                   # radians (0 = North, π/2 = East)
        self.speed = 0.0                     # m/s
        self.velocity = np.zeros(2)          # v_east, v_north

        # Tracking
        self.is_active = False
        self.outage_start_time = None
        self.outage_duration = 0.0
        self.distance_traveled = 0.0
        self.last_gnss_position = np.zeros(2)

        # Trajectory history (for visualization and analysis)
        self.trajectory: List[Dict] = []

    def start(
        self,
        position: np.ndarray,
        heading: float,
        speed: float = 0.0,
        timestamp: float = 0.0,
    ):
        """
        Initialize dead reckoning from last known GNSS position.

        Args:
            position: (2,) last known position [East, North] in meters.
            heading: Last known heading in radians.
            speed: Last known speed in m/s.
            timestamp: Time of GNSS loss.
        """
        self.position = position.copy()
        self.heading = heading
        self.speed = speed
        self.velocity = np.array([
            speed * np.sin(heading),  # v_east
            speed * np.cos(heading),  # v_north
        ])
        self.last_gnss_position = position.copy()

        self.is_active = True
        self.outage_start_time = timestamp
        self.outage_duration = 0.0
        self.distance_traveled = 0.0
        self.trajectory = []

        self._record_state(timestamp)

    def update(
        self,
        ai_velocity: Optional[np.ndarray] = None,
        gyro_yaw_rate: float = 0.0,
        ai_speed: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """
        Advance dead reckoning by one time step.

        Args:
            ai_velocity: (2,) AI-predicted velocity [v_north, v_east] in m/s.
            gyro_yaw_rate: Gyroscope yaw rate in rad/s.
            ai_speed: Optional AI-predicted scalar speed (alternative to velocity).
            timestamp: Current timestamp.

        Returns:
            (2,) updated position [East, North].
        """
        if not self.is_active:
            return self.position.copy()

        # Update heading from gyroscope
        self.heading += gyro_yaw_rate * self.dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # Update velocity
        if ai_velocity is not None:
            # Use AI-predicted velocity directly
            self.velocity = np.array([ai_velocity[1], ai_velocity[0]])  # [v_east, v_north]
            self.speed = np.linalg.norm(ai_velocity)
        elif ai_speed is not None:
            # Use speed + heading to derive velocity
            self.speed = ai_speed
            self.velocity = np.array([
                self.speed * np.sin(self.heading),
                self.speed * np.cos(self.heading),
            ])
        else:
            # Pure heading integration (no velocity update — drift will grow)
            self.velocity = np.array([
                self.speed * np.sin(self.heading),
                self.speed * np.cos(self.heading),
            ])

        # Integrate position
        displacement = self.velocity * self.dt
        self.position += displacement
        self.distance_traveled += np.linalg.norm(displacement)

        # Update outage duration
        if timestamp is not None and self.outage_start_time is not None:
            self.outage_duration = timestamp - self.outage_start_time

        self._record_state(timestamp)

        return self.position.copy()

    def stop(self) -> Dict:
        """
        End dead reckoning (GNSS restored).

        Returns:
            Summary dict with DR performance statistics.
        """
        self.is_active = False

        drift = np.linalg.norm(self.position - self.last_gnss_position)
        drift_pct = (drift / max(self.distance_traveled, 1e-6)) * 100

        return {
            "outage_duration_s": self.outage_duration,
            "distance_traveled_m": self.distance_traveled,
            "final_position": self.position.copy(),
            "drift_from_start_m": drift,
            "drift_percent": drift_pct,
            "num_updates": len(self.trajectory),
            "warning": self.outage_duration > self.max_outage_duration,
        }

    def get_estimated_drift(self) -> float:
        """
        Estimate current positional drift from start of outage.

        Returns:
            Estimated drift in meters.
        """
        return float(np.linalg.norm(self.position - self.last_gnss_position))

    def get_drift_percentage(self) -> float:
        """
        Get drift as percentage of distance traveled.

        Returns:
            Drift percentage (target: < 10%).
        """
        if self.distance_traveled < 1e-6:
            return 0.0
        drift = self.get_estimated_drift()
        return (drift / self.distance_traveled) * 100.0

    def get_confidence(self) -> float:
        """
        Estimate confidence level (0-1) in the current DR position.

        Decays over time and distance traveled without GNSS corrections.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Time decay factor
        time_factor = np.exp(-self.outage_duration / 60.0)  # Halves every ~42s

        # Distance decay factor
        dist_factor = np.exp(-self.distance_traveled / 1000.0)  # Halves every ~693m

        return float(max(0.0, min(1.0, time_factor * dist_factor)))

    def _record_state(self, timestamp: Optional[float]):
        """Record current state to trajectory history."""
        self.trajectory.append({
            "timestamp": timestamp,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "heading": self.heading,
            "speed": self.speed,
            "distance_traveled": self.distance_traveled,
            "confidence": self.get_confidence(),
        })

    def get_trajectory_array(self) -> np.ndarray:
        """Return trajectory positions as (N, 2) numpy array."""
        if not self.trajectory:
            return np.empty((0, 2))
        return np.array([t["position"] for t in self.trajectory])
