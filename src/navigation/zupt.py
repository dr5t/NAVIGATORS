"""
Navigators IDR - Zero Velocity Update (ZUPT) Detection
Multi-signal robust detector identifying when the vehicle is stationary.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque


class ZUPTDetector:
    """
    Detects stationary (zero-velocity) periods from IMU and motion data.

    Robust multi-criteria detection:
        1. Accelerometer magnitude variance
        2. Gyroscope axis variance and magnitude
        3. Gravity vector consistency (|mean_accel| ≈ 9.81 m/s²)
        4. Optional velocity/speed check
        5. Minimum dwell time to prevent false triggers on smooth motion
    """

    def __init__(
        self,
        accel_variance_threshold: float = 0.05,
        gyro_variance_threshold: float = 0.01,
        detection_window: int = 10,
        velocity_reset_sigma: float = 0.01,
        gravity: float = 9.81,
        min_stationary_samples: int = 3,
    ):
        self.accel_threshold = accel_variance_threshold
        self.gyro_threshold = gyro_variance_threshold
        self.window_size = detection_window
        self.velocity_sigma = velocity_reset_sigma
        self.gravity = gravity
        self.min_stationary_samples = min_stationary_samples


        self.accel_buffer = deque(maxlen=detection_window)
        self.gyro_buffer = deque(maxlen=detection_window)


        self.is_stationary = False
        self.consecutive_stationary = 0
        self.stationary_duration = 0.0
        self.last_detection_metrics: Dict[str, Any] = {}

    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float = 0.1,
        estimated_speed: Optional[float] = None,
    ) -> bool:
        """
        Evaluate if vehicle is stationary based on multi-signal criteria.
        """
        self.accel_buffer.append(accel.copy())
        self.gyro_buffer.append(gyro.copy())

        if len(self.accel_buffer) < self.window_size:
            self.is_stationary = False
            return False

        accel_array = np.array(list(self.accel_buffer))
        gyro_array = np.array(list(self.gyro_buffer))


        accel_magnitudes = np.linalg.norm(accel_array, axis=1)
        accel_var = float(np.var(accel_magnitudes))
        accel_mean_mag = float(np.mean(accel_magnitudes))


        gravity_consistent = abs(accel_mean_mag - self.gravity) < 1.0


        gyro_var = float(np.max(np.var(gyro_array, axis=0)))
        max_gyro_rate = float(np.max(np.linalg.norm(gyro_array, axis=1)))


        speed_ok = True
        if estimated_speed is not None:
            speed_ok = estimated_speed < 0.5


        condition_met = (
            accel_var < self.accel_threshold
            and gyro_var < self.gyro_threshold
            and max_gyro_rate < 0.15
            and gravity_consistent
            and speed_ok
        )

        if condition_met:
            self.consecutive_stationary += 1
        else:
            self.consecutive_stationary = 0

        was_stationary = self.is_stationary
        self.is_stationary = self.consecutive_stationary >= self.min_stationary_samples

        if self.is_stationary:
            self.stationary_duration += dt
        else:
            self.stationary_duration = 0.0

        self.last_detection_metrics = {
            "accel_variance": accel_var,
            "gyro_variance": gyro_var,
            "accel_mean_mag": accel_mean_mag,
            "is_stationary": self.is_stationary,
            "stationary_duration": self.stationary_duration,
            "consecutive_samples": self.consecutive_stationary,
        }

        return bool(self.is_stationary)

    def get_zupt_measurement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero velocity measurement vector and covariance matrix."""
        z = np.zeros(3, dtype=np.float64)
        R = np.eye(3, dtype=np.float64) * (self.velocity_sigma ** 2)

        if self.stationary_duration > 1.0:
            boost = min(10.0, self.stationary_duration)
            R /= boost

        return z, R

    def reset(self):
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.is_stationary = False
        self.consecutive_stationary = 0
        self.stationary_duration = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        return self.last_detection_metrics.copy()
