"""
Navigators IDR — Zero Velocity Update (ZUPT) Detection
Detects when the vehicle is stationary to reset accumulated velocity errors.

ZUPT is a critical error-bounding technique: when the vehicle stops
(e.g., at a traffic light), we know velocity = 0 with high certainty.
This resets accumulated drift in the velocity estimate.
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque


class ZUPTDetector:
    """
    Detects stationary (zero-velocity) periods from IMU data.

    Uses variance-based detection on accelerometer and gyroscope
    readings to identify when the vehicle has stopped.
    """

    def __init__(
        self,
        accel_variance_threshold: float = 0.05,
        gyro_variance_threshold: float = 0.01,
        detection_window: int = 10,
        velocity_reset_sigma: float = 0.01,
        gravity: float = 9.81,
    ):
        """
        Args:
            accel_variance_threshold: Max accelerometer variance (m²/s⁴) for stationary.
            gyro_variance_threshold: Max gyroscope variance (rad²/s²) for stationary.
            detection_window: Number of samples to evaluate for stationarity.
            velocity_reset_sigma: ZUPT measurement noise (m/s) — lower = more trust.
            gravity: Expected gravity magnitude for accelerometer normalization.
        """
        self.accel_threshold = accel_variance_threshold
        self.gyro_threshold = gyro_variance_threshold
        self.window_size = detection_window
        self.velocity_sigma = velocity_reset_sigma
        self.gravity = gravity

        # Buffers for sliding window
        self.accel_buffer = deque(maxlen=detection_window)
        self.gyro_buffer = deque(maxlen=detection_window)

        # State
        self.is_stationary = False
        self.stationary_duration = 0.0  # seconds
        self.last_detection_metrics = {}

    def update(
        self, accel: np.ndarray, gyro: np.ndarray, dt: float = 0.1
    ) -> bool:
        """
        Process a new IMU sample and determine if stationary.

        Args:
            accel: (3,) accelerometer reading in m/s².
            gyro: (3,) gyroscope reading in rad/s.
            dt: Time step in seconds.

        Returns:
            True if vehicle is currently stationary.
        """
        self.accel_buffer.append(accel.copy())
        self.gyro_buffer.append(gyro.copy())

        if len(self.accel_buffer) < self.window_size:
            self.is_stationary = False
            return False

        # Compute variances over the window
        accel_array = np.array(list(self.accel_buffer))
        gyro_array = np.array(list(self.gyro_buffer))

        # For accelerometer: compute variance of magnitude
        # (when stationary, magnitude ≈ gravity with low variance)
        accel_magnitudes = np.linalg.norm(accel_array, axis=1)
        accel_var = np.var(accel_magnitudes)

        # For gyroscope: compute variance of each axis and take max
        gyro_var = np.max(np.var(gyro_array, axis=0))

        # Detection decision
        was_stationary = self.is_stationary
        self.is_stationary = (accel_var < self.accel_threshold and
                              gyro_var < self.gyro_threshold)

        # Track duration
        if self.is_stationary:
            self.stationary_duration += dt
        else:
            self.stationary_duration = 0.0

        # Store metrics for diagnostics
        self.last_detection_metrics = {
            "accel_variance": float(accel_var),
            "gyro_variance": float(gyro_var),
            "accel_threshold": self.accel_threshold,
            "gyro_threshold": self.gyro_threshold,
            "is_stationary": self.is_stationary,
            "stationary_duration": self.stationary_duration,
            "transition": (
                "moving→stopped" if self.is_stationary and not was_stationary
                else "stopped→moving" if not self.is_stationary and was_stationary
                else "stationary" if self.is_stationary
                else "moving"
            ),
        }

        return self.is_stationary

    def get_zupt_measurement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the ZUPT measurement for EKF update.

        Returns:
            Tuple of:
                z: (3,) zero velocity measurement
                R: (3, 3) measurement noise covariance
        """
        z = np.zeros(3)
        R = np.eye(3) * self.velocity_sigma ** 2

        # If stationary for longer, increase confidence (reduce noise)
        if self.stationary_duration > 2.0:
            confidence_boost = min(10.0, self.stationary_duration / 2.0)
            R /= confidence_boost

        return z, R

    def reset(self):
        """Reset the ZUPT detector state."""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.is_stationary = False
        self.stationary_duration = 0.0

    def get_metrics(self) -> dict:
        """Return the latest detection metrics for diagnostics."""
        return self.last_detection_metrics.copy()
