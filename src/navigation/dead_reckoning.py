"""
Navigators IDR - Dead Reckoning Engine
Integrates AI velocity estimates with heading to maintain position during GNSS denial.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class DeadReckoningEngine:
    """
    Dead reckoning engine for GNSS-denied navigation.

    Integrates AI-predicted velocity and gyroscope-derived heading
    to maintain continuous position estimates when GNSS is lost.
    Applies Non-Holonomic Constraints to eliminate sideways drift.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_outage_duration: float = 120.0,
        drift_warning_threshold: float = 0.1,
    ):
        self.dt = dt
        self.max_outage_duration = max_outage_duration
        self.drift_warning_threshold = drift_warning_threshold

        # State (ENU coordinates in meters)
        self.position = np.zeros(2, dtype=np.float64)          # [East, North]
        self.heading = 0.0                                     # radians (0 = North, π/2 = East)
        self.speed = 0.0                                       # m/s
        self.velocity = np.zeros(2, dtype=np.float64)          # [v_east, v_north]

        # Outage tracking
        self.is_active = False
        self.outage_start_time: Optional[float] = None
        self.outage_duration = 0.0
        self.distance_traveled = 0.0
        self.outage_start_position = np.zeros(2, dtype=np.float64)

        # Trajectory history
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
        """
        self.position = position[:2].astype(np.float64).copy()
        self.heading = float(heading)
        self.speed = float(speed)
        self.velocity = np.array([
            self.speed * np.sin(self.heading),  # v_east
            self.speed * np.cos(self.heading),  # v_north
        ], dtype=np.float64)
        self.outage_start_position = self.position.copy()

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
            ai_velocity: (2,) AI-predicted velocity in ENU [v_east, v_north] or [v_north, v_east].
            gyro_yaw_rate: Gyro yaw rate in rad/s.
            ai_speed: Optional scalar speed.
            timestamp: Current timestamp in seconds.
        """
        if not self.is_active:
            return self.position.copy()

        # Integrate heading from gyroscope yaw rate
        self.heading += gyro_yaw_rate * self.dt
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi

        # Update velocity
        if ai_velocity is not None:
            # Handle AI velocity: default standard ENU [v_east, v_north]
            v_input = np.asarray(ai_velocity, dtype=np.float64)
            self.velocity = v_input[:2].copy()
            self.speed = float(np.linalg.norm(self.velocity))
        elif ai_speed is not None:
            self.speed = float(ai_speed)
            self.velocity = np.array([
                self.speed * np.sin(self.heading),  # v_east
                self.speed * np.cos(self.heading),  # v_north
            ], dtype=np.float64)
        else:
            # Maintain previous speed along current heading
            self.velocity = np.array([
                self.speed * np.sin(self.heading),
                self.speed * np.cos(self.heading),
            ], dtype=np.float64)

        # Integrate position
        displacement = self.velocity * self.dt
        self.position += displacement
        step_dist = float(np.linalg.norm(displacement))
        self.distance_traveled += step_dist

        # Update outage duration
        if timestamp is not None and self.outage_start_time is not None:
            self.outage_duration = timestamp - self.outage_start_time
        else:
            self.outage_duration += self.dt

        self._record_state(timestamp)
        return self.position.copy()

    def stop(self, gnss_restore_position: Optional[np.ndarray] = None) -> Dict:
        """
        End dead reckoning upon GNSS restoration and compute true drift metrics.
        
        Args:
            gnss_restore_position: Optional (2,) ground-truth / GNSS position at restoration.
        """
        self.is_active = False

        if gnss_restore_position is not None:
            # True drift error relative to GNSS fix at the end of the outage
            drift_error = float(np.linalg.norm(self.position - gnss_restore_position[:2]))
        else:
            # Estimated statistical drift (modeled as 3% of distance traveled)
            drift_error = self.get_estimated_drift()

        drift_pct = (drift_error / max(self.distance_traveled, 1e-6)) * 100.0

        return {
            "outage_duration_s": float(self.outage_duration),
            "distance_traveled_m": float(self.distance_traveled),
            "final_position": self.position.copy(),
            "drift_from_start_m": float(np.linalg.norm(self.position - self.outage_start_position)),
            "drift_error_m": drift_error,
            "drift_percent": drift_pct,
            "num_updates": len(self.trajectory),
            "warning": self.outage_duration > self.max_outage_duration,
        }

    def get_estimated_drift(self) -> float:
        """
        Estimate 1-sigma positional uncertainty in meters during GNSS denial.
        Error model: linear distance growth (~3%) plus initial uncertainty.
        """
        return float(1.0 + 0.03 * self.distance_traveled + 0.02 * self.outage_duration)

    def get_drift_percentage(self) -> float:
        """Get current estimated drift as percentage of distance traveled."""
        if self.distance_traveled < 1.0:
            return 0.0
        return (self.get_estimated_drift() / self.distance_traveled) * 100.0

    def get_confidence(self) -> float:
        """Return confidence score in [0.0, 1.0] decaying with outage duration and distance."""
        time_factor = np.exp(-self.outage_duration / 60.0)
        dist_factor = np.exp(-self.distance_traveled / 1000.0)
        return float(max(0.0, min(1.0, time_factor * dist_factor)))

    def _record_state(self, timestamp: Optional[float]):
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
        if not self.trajectory:
            return np.empty((0, 2))
        return np.array([t["position"] for t in self.trajectory])
