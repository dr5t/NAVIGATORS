"""
Navigators IDR — IMU Data Preprocessor
Noise filtering, gravity removal, phone alignment, and windowing.

Handles the critical preprocessing step that converts raw smartphone IMU
readings into clean, vehicle-frame-aligned feature windows for the AI model.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional


class IMUPreprocessor:
    """
    Preprocesses raw IMU data for the velocity estimation model.

    Pipeline:
        1. Low-pass Butterworth filter (remove high-freq vibration)
        2. Gravity vector estimation and removal
        3. Phone-to-vehicle frame rotation
        4. Sliding window creation
    """

    def __init__(
        self,
        sample_rate: float = 10.0,
        lowpass_cutoff: float = 3.0,
        lowpass_order: int = 4,
        gravity: float = 9.81,
        alignment_method: str = "pca",
    ):
        """
        Args:
            sample_rate: IMU sampling frequency in Hz.
            lowpass_cutoff: Butterworth filter cutoff frequency in Hz.
            lowpass_order: Butterworth filter order.
            gravity: Expected gravity magnitude in m/s².
            alignment_method: Phone-to-vehicle alignment ('pca', 'static_gravity', 'manual').
        """
        self.sample_rate = sample_rate
        self.lowpass_cutoff = lowpass_cutoff
        self.lowpass_order = lowpass_order
        self.gravity = gravity
        self.alignment_method = alignment_method

        # Precompute Butterworth filter coefficients
        nyquist = sample_rate / 2.0
        if lowpass_cutoff < nyquist:
            self.b, self.a = butter(lowpass_order, lowpass_cutoff / nyquist, btype='low')
        else:
            self.b, self.a = None, None

    def lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase Butterworth low-pass filter to remove vibration noise.

        Args:
            data: (N, C) array of sensor readings (N samples, C channels).

        Returns:
            Filtered data of same shape.
        """
        if self.b is None or len(data) < 3 * max(len(self.a), len(self.b)):
            return data  # Too short to filter or cutoff >= Nyquist

        filtered = np.zeros_like(data)
        for c in range(data.shape[1]):
            filtered[:, c] = filtfilt(self.b, self.a, data[:, c])
        return filtered

    def estimate_gravity_vector(self, accel: np.ndarray) -> np.ndarray:
        """
        Estimate the gravity direction from a static or quasi-static accelerometer window.

        Uses the mean accelerometer reading (assumes vehicle is approximately
        stationary or moving at constant velocity for part of the sequence).

        Args:
            accel: (N, 3) accelerometer readings in m/s².

        Returns:
            Unit gravity vector (3,) in sensor frame.
        """
        mean_accel = np.mean(accel, axis=0)
        magnitude = np.linalg.norm(mean_accel)
        if magnitude < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        return mean_accel / magnitude

    def remove_gravity(self, accel: np.ndarray, gravity_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove gravity component from accelerometer data.

        Args:
            accel: (N, 3) raw accelerometer data in m/s².
            gravity_vec: Optional precomputed unit gravity vector.

        Returns:
            (N, 3) linear acceleration (gravity-free).
        """
        if gravity_vec is None:
            gravity_vec = self.estimate_gravity_vector(accel)

        gravity_component = self.gravity * gravity_vec
        return accel - gravity_component[np.newaxis, :]

    def estimate_rotation_matrix(self, accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """
        Estimate rotation matrix from phone frame to vehicle frame.

        Uses PCA on the accelerometer data to identify the dominant motion axis,
        combined with gravity for the vertical.

        Args:
            accel: (N, 3) accelerometer data.
            gyro: (N, 3) gyroscope data.

        Returns:
            (3, 3) rotation matrix R such that v_vehicle = R @ v_phone.
        """
        if self.alignment_method == "static_gravity":
            return self._align_by_gravity(accel)
        elif self.alignment_method == "pca":
            return self._align_by_pca(accel)
        else:
            return np.eye(3)  # Manual / identity

    def _align_by_gravity(self, accel: np.ndarray) -> np.ndarray:
        """Align based on gravity vector — assumes Z is vertical."""
        g = self.estimate_gravity_vector(accel)

        # Z-axis = gravity direction (down)
        z_axis = g / np.linalg.norm(g)

        # Choose an arbitrary perpendicular for X
        if abs(z_axis[0]) < 0.9:
            x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        else:
            x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        return np.stack([x_axis, y_axis, z_axis], axis=0)

    def _align_by_pca(self, accel: np.ndarray) -> np.ndarray:
        """
        Align using PCA on gravity-removed acceleration.
        Principal component ≈ forward direction of vehicle motion.
        """
        linear_accel = self.remove_gravity(accel)

        # PCA
        centered = linear_accel - np.mean(linear_accel, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Forward axis = first principal component
        forward = eigenvectors[:, 0]

        # Gravity axis
        g = self.estimate_gravity_vector(accel)
        up = g / np.linalg.norm(g)

        # Make forward perpendicular to up
        forward = forward - np.dot(forward, up) * up
        fwd_norm = np.linalg.norm(forward)
        if fwd_norm < 1e-10:
            forward = eigenvectors[:, 1]
            forward = forward - np.dot(forward, up) * up
            fwd_norm = np.linalg.norm(forward)
        forward /= fwd_norm

        # Right axis = cross product
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        return np.stack([forward, right, up], axis=0)

    def rotate_to_vehicle_frame(
        self, accel: np.ndarray, gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate IMU data from phone frame to vehicle frame.

        Args:
            accel: (N, 3) accelerometer data in phone frame.
            gyro: (N, 3) gyroscope data in phone frame.

        Returns:
            Tuple of (accel_vehicle, gyro_vehicle), both (N, 3).
        """
        R = self.estimate_rotation_matrix(accel, gyro)
        accel_rotated = (R @ accel.T).T
        gyro_rotated = (R @ gyro.T).T
        return accel_rotated, gyro_rotated

    def create_windows(
        self,
        data: np.ndarray,
        window_size: int = 200,
        stride: int = 50
    ) -> np.ndarray:
        """
        Create overlapping sliding windows from a time series.

        Args:
            data: (N, C) input data (N timesteps, C features).
            window_size: Number of samples per window.
            stride: Step size between consecutive windows.

        Returns:
            (num_windows, window_size, C) array of windowed data.
        """
        N, C = data.shape
        num_windows = max(0, (N - window_size) // stride + 1)

        if num_windows == 0:
            return np.empty((0, window_size, C))

        windows = np.zeros((num_windows, window_size, C))
        for i in range(num_windows):
            start = i * stride
            windows[i] = data[start : start + window_size]

        return windows

    def normalize(
        self, data: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-score normalize data per channel.

        Args:
            data: (N, C) or (W, N, C) input data.
            mean: Precomputed mean per channel.
            std: Precomputed std per channel.

        Returns:
            Tuple of (normalized_data, mean, std).
        """
        original_shape = data.shape
        if data.ndim == 3:
            W, N, C = data.shape
            data_flat = data.reshape(-1, C)
        else:
            data_flat = data

        if mean is None:
            mean = np.mean(data_flat, axis=0)
        if std is None:
            std = np.std(data_flat, axis=0)
            std[std < 1e-8] = 1.0  # Prevent division by zero

        normalized = (data_flat - mean) / std

        if len(original_shape) == 3:
            normalized = normalized.reshape(original_shape)

        return normalized, mean, std

    def preprocess(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        window_size: int = 200,
        stride: int = 50,
        do_normalize: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Full preprocessing pipeline: filter → align → gravity remove → window → normalize.

        Args:
            accel: (N, 3) raw accelerometer data in m/s².
            gyro: (N, 3) raw gyroscope data in rad/s.
            window_size: Samples per window.
            stride: Window stride.
            do_normalize: Whether to z-score normalize.

        Returns:
            Tuple of:
                - (num_windows, window_size, 6) processed IMU windows
                - dict with normalization stats and rotation matrix
        """
        # 1. Low-pass filter
        accel_filtered = self.lowpass_filter(accel)
        gyro_filtered = self.lowpass_filter(gyro)

        # 2. Rotate to vehicle frame
        accel_vehicle, gyro_vehicle = self.rotate_to_vehicle_frame(
            accel_filtered, gyro_filtered
        )

        # 3. Remove gravity
        accel_linear = self.remove_gravity(accel_vehicle)

        # 4. Combine into 6-channel IMU data
        imu_data = np.hstack([accel_linear, gyro_vehicle])  # (N, 6)

        # 5. Create sliding windows
        windows = self.create_windows(imu_data, window_size, stride)

        # 6. Normalize
        metadata = {"rotation_matrix": self.estimate_rotation_matrix(accel, gyro)}
        if do_normalize and windows.shape[0] > 0:
            windows, mean, std = self.normalize(windows)
            metadata["mean"] = mean
            metadata["std"] = std

        return windows, metadata
