"""
Navigators IDR — Phone-to-Vehicle Alignment & Coordinate Transformations
Implements robust attitude estimation, frame transformations, and calibration
between arbitrary phone mounting orientations and the vehicle frame.

Coordinate frames:
    - Phone Frame: Sensor coordinate frame (device dependent)
    - Vehicle Frame: Forward (X), Right (Y), Down (Z)
    - Navigation Frame: East (X), North (Y), Up (Z) [ENU]
"""

import numpy as np
from typing import Tuple, Optional, Dict


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion [w, x, y, z] using Shepperd's method.
    """
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Compute rotation matrix using standard Tait-Bryan 3-2-1 (yaw-pitch-roll) sequence.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                  cp * cr               ]
    ], dtype=np.float64)


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract roll, pitch, yaw from 3x3 rotation matrix.
    Returns (roll, pitch, yaw) in radians.
    """
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


class PhoneVehicleAligner:
    """
    Estimates and maintains the rotation matrix R_vehicle_phone between the
    smartphone sensor frame and the vehicle body frame:
        a_vehicle = R_vehicle_phone @ a_phone
        w_vehicle = R_vehicle_phone @ w_phone
    
    Vehicle frame conventions:
        X: Forward
        Y: Right
        Z: Down
    """

    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.R_vehicle_phone = np.eye(3, dtype=np.float64)
        self.is_calibrated = False
        self.confidence = 0.0

        # Accumulation buffers for leveling and heading determination
        self.static_accel_samples = []
        self.motion_samples = []

    def calibrate_gravity(self, accel_samples: np.ndarray) -> np.ndarray:
        """
        Estimate vehicle vertical axis (Z_down) from static or quasi-static accelerometer data.
        
        Args:
            accel_samples: (N, 3) accelerometer readings in phone frame.
            
        Returns:
            Unit gravity vector in phone frame pointing downwards.
        """
        if accel_samples.ndim == 1:
            mean_accel = accel_samples
        else:
            mean_accel = np.mean(accel_samples, axis=0)

        norm = np.linalg.norm(mean_accel)
        if norm < 1e-3:
            return np.array([0.0, 0.0, 1.0])

        # Phone accelerometer measures reaction force upwards when resting: a = +g * z_up = -g * z_down
        # So unit vector in direction of gravity (down) is mean_accel / norm
        z_down_phone = mean_accel / norm
        return z_down_phone

    def set_orientation(self, roll: float, pitch: float, yaw: float):
        """
        Manually set vehicle-to-phone rotation using known angles.
        """
        self.R_vehicle_phone = euler_to_rotation_matrix(roll, pitch, yaw)
        self.is_calibrated = True
        self.confidence = 1.0

    def compute_alignment_matrix(
        self,
        static_accel: np.ndarray,
        forward_motion_accel: Optional[np.ndarray] = None,
        forward_hint: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute complete 3D rotation matrix R_vehicle_phone mapping Phone frame to Vehicle frame.
        
        Args:
            static_accel: (N, 3) or (3,) static accelerometer readings to determine down axis (Z).
            forward_motion_accel: (N, 3) accelerometer readings during acceleration/braking to determine forward axis (X).
            forward_hint: Optional (3,) vector indicating known phone forward axis.
            
        Returns:
            (3, 3) orthogonal rotation matrix R_vehicle_phone.
        """
        z_down = self.calibrate_gravity(static_accel)

        # Determine forward axis (X_forward)
        if forward_motion_accel is not None and len(forward_motion_accel) > 5:
            # PCA on dynamic acceleration to find longitudinal motion line
            lin_accel = forward_motion_accel - (z_down * np.dot(forward_motion_accel, z_down)[:, np.newaxis])
            cov = np.cov(lin_accel.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_axis = eigenvecs[:, np.argmax(eigenvals)]

            # Project to horizontal plane
            x_fwd = principal_axis - np.dot(principal_axis, z_down) * z_down
            norm_x = np.linalg.norm(x_fwd)
            if norm_x > 1e-4:
                x_fwd = x_fwd / norm_x
            else:
                x_fwd = np.array([1.0, 0.0, 0.0])

            # Resolve 180° ambiguity using forward hint or acceleration sign
            if forward_hint is not None and np.dot(x_fwd, forward_hint) < 0:
                x_fwd = -x_fwd
        elif forward_hint is not None:
            # Project forward hint to horizontal plane
            x_fwd = forward_hint - np.dot(forward_hint, z_down) * z_down
            norm_x = np.linalg.norm(x_fwd)
            if norm_x > 1e-4:
                x_fwd = x_fwd / norm_x
            else:
                x_fwd = np.array([1.0, 0.0, 0.0])
        else:
            # Pick canonical perpendicular vector for X
            if abs(z_down[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            x_fwd = perp - np.dot(perp, z_down) * z_down
            x_fwd = x_fwd / np.linalg.norm(x_fwd)

        # Right axis = Z_down x X_fwd
        y_right = np.cross(z_down, x_fwd)
        norm_y = np.linalg.norm(y_right)
        if norm_y > 1e-4:
            y_right = y_right / norm_y
        else:
            y_right = np.array([0.0, 1.0, 0.0])

        # Recompute orthogonal X_fwd = Y_right x Z_down
        x_fwd = np.cross(y_right, z_down)
        x_fwd = x_fwd / np.linalg.norm(x_fwd)

        # R_vehicle_phone rows are the vehicle axes expressed in phone frame:
        # v_vehicle = [x_fwd^T; y_right^T; z_down^T] @ v_phone
        R = np.stack([x_fwd, y_right, z_down], axis=0)
        self.R_vehicle_phone = R
        self.is_calibrated = True
        self.confidence = 0.95
        return self.R_vehicle_phone

    def transform_accel(self, accel_phone: np.ndarray) -> np.ndarray:
        """
        Transform acceleration from phone frame to vehicle frame.
        a_vehicle = R_vehicle_phone @ a_phone
        """
        if accel_phone.ndim == 1:
            return self.R_vehicle_phone @ accel_phone
        return (self.R_vehicle_phone @ accel_phone.T).T

    def transform_gyro(self, gyro_phone: np.ndarray) -> np.ndarray:
        """
        Transform angular rate from phone frame to vehicle frame.
        w_vehicle = R_vehicle_phone @ w_phone
        """
        if gyro_phone.ndim == 1:
            return self.R_vehicle_phone @ gyro_phone
        return (self.R_vehicle_phone @ gyro_phone.T).T

    def get_rotation_matrix(self) -> np.ndarray:
        """Return the current 3x3 rotation matrix."""
        return self.R_vehicle_phone.copy()

    def get_quaternion(self) -> np.ndarray:
        """Return attitude as unit quaternion [w, x, y, z]."""
        return rotation_matrix_to_quaternion(self.R_vehicle_phone)

    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Return attitude as (roll, pitch, yaw) in radians."""
        return rotation_matrix_to_euler(self.R_vehicle_phone)
