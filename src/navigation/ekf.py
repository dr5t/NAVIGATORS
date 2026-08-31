"""
Navigators IDR — Extended Kalman Filter
15-state EKF for GNSS + INS sensor fusion.

State vector (15 elements):
    [0:3]  — Position (East, North, Up) in meters
    [3:6]  — Velocity (v_E, v_N, v_U) in m/s
    [6:9]  — Orientation (roll, pitch, yaw) in radians
    [9:12] — Accelerometer bias (b_ax, b_ay, b_az) in m/s²
    [12:15] — Gyroscope bias (b_gx, b_gy, b_gz) in rad/s

Supports three operating modes:
    1. GNSS + INS fusion (normal)
    2. Dead reckoning only (GNSS denied)
    3. Re-acquisition (GNSS restored — fast convergence)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum


class NavigationMode(Enum):
    """Current navigation operating mode."""
    GNSS_INS = "gnss_ins"      # Full fusion
    DEAD_RECKONING = "dr"      # GNSS denied
    REACQUISITION = "reacq"    # GNSS just restored


class ExtendedKalmanFilter:
    """
    15-state Extended Kalman Filter for vehicle navigation.

    Fuses AI-predicted velocity measurements with GNSS position/velocity
    updates, while estimating and compensating for sensor biases.
    """

    # State indices
    POS = slice(0, 3)
    VEL = slice(3, 6)
    ORI = slice(6, 9)
    ABIAS = slice(9, 12)
    GBIAS = slice(12, 15)

    STATE_DIM = 15

    def __init__(
        self,
        process_noise: Optional[Dict[str, float]] = None,
        gnss_noise: Optional[Dict[str, float]] = None,
        ai_velocity_noise: float = 0.3,
        dt: float = 0.1,
    ):
        """
        Args:
            process_noise: Dict with keys 'position', 'velocity', 'orientation',
                          'accel_bias', 'gyro_bias' specifying process noise variances.
            gnss_noise: Dict with keys 'position', 'velocity' for GNSS measurement noise.
            ai_velocity_noise: AI velocity prediction uncertainty (m/s).
            dt: Time step in seconds.
        """
        self.dt = dt

        # Default process noise
        pn = process_noise or {}
        self.q_pos = pn.get("position", 0.01)
        self.q_vel = pn.get("velocity", 0.1)
        self.q_ori = pn.get("orientation", 0.001)
        self.q_abias = pn.get("accel_bias", 0.0001)
        self.q_gbias = pn.get("gyro_bias", 0.00001)

        # GNSS measurement noise
        gn = gnss_noise or {}
        self.r_gnss_pos = gn.get("position", 2.5)
        self.r_gnss_vel = gn.get("velocity", 0.5)

        # AI velocity noise
        self.r_ai_vel = ai_velocity_noise

        # Initialize state
        self.x = np.zeros(self.STATE_DIM)  # State vector
        self.P = np.eye(self.STATE_DIM)    # State covariance

        # Set initial uncertainties
        self.P[self.POS, self.POS] *= 10.0      # 10m position uncertainty
        self.P[self.VEL, self.VEL] *= 5.0       # 5 m/s velocity uncertainty
        self.P[self.ORI, self.ORI] *= 0.1       # ~6° orientation uncertainty
        self.P[self.ABIAS, self.ABIAS] *= 0.5   # Bias uncertainty
        self.P[self.GBIAS, self.GBIAS] *= 0.01

        # Navigation mode
        self.mode = NavigationMode.GNSS_INS
        self.gnss_outage_start = None
        self.last_gnss_time = None

        # Diagnostics
        self.innovation_history = []

    def _build_process_noise(self) -> np.ndarray:
        """Construct the process noise covariance matrix Q."""
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q[self.POS, self.POS] = np.eye(3) * self.q_pos * self.dt ** 2
        Q[self.VEL, self.VEL] = np.eye(3) * self.q_vel * self.dt
        Q[self.ORI, self.ORI] = np.eye(3) * self.q_ori * self.dt
        Q[self.ABIAS, self.ABIAS] = np.eye(3) * self.q_abias * self.dt
        Q[self.GBIAS, self.GBIAS] = np.eye(3) * self.q_gbias * self.dt
        return Q

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Compute rotation matrix from body to navigation frame (3-2-1 Euler angles).

        Args:
            roll: Roll angle in radians.
            pitch: Pitch angle in radians.
            yaw: Yaw (heading) angle in radians.

        Returns:
            (3, 3) rotation matrix.
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
            [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
            [-sp,      cp * sr,                  cp * cr               ]
        ])
        return R

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        ai_velocity: Optional[np.ndarray] = None,
    ):
        """
        EKF prediction step using IMU measurements.

        Propagates the state forward using the motion model with
        IMU-derived acceleration and angular velocity.

        Args:
            accel_body: (3,) accelerometer reading in body frame [m/s²].
            gyro_body: (3,) gyroscope reading in body frame [rad/s].
            ai_velocity: Optional (2,) AI-predicted velocity [v_north, v_east].
                        Used as a pseudo-measurement if GNSS is denied.
        """
        # Extract current state
        roll, pitch, yaw = self.x[self.ORI]
        accel_bias = self.x[self.ABIAS]
        gyro_bias = self.x[self.GBIAS]

        # Compensate for biases
        accel_corrected = accel_body - accel_bias
        gyro_corrected = gyro_body - gyro_bias

        # Rotation from body to navigation frame
        R_b2n = self._rotation_matrix(roll, pitch, yaw)

        # Specific force in navigation frame (remove gravity)
        gravity = np.array([0.0, 0.0, 9.81])
        accel_nav = R_b2n @ accel_corrected - gravity

        # --- State transition ---
        # Position update: p_new = p + v * dt + 0.5 * a * dt²
        self.x[self.POS] += self.x[self.VEL] * self.dt + 0.5 * accel_nav * self.dt ** 2

        # Velocity update: v_new = v + a * dt
        self.x[self.VEL] += accel_nav * self.dt

        # Orientation update (simplified Euler integration)
        # For small angles: dθ ≈ ω * dt
        self.x[self.ORI] += gyro_corrected * self.dt

        # Wrap yaw to [-π, π]
        self.x[8] = (self.x[8] + np.pi) % (2 * np.pi) - np.pi

        # Biases modeled as random walks (no change in prediction)

        # --- Jacobian of state transition (F matrix) ---
        F = np.eye(self.STATE_DIM)

        # ∂position/∂velocity
        F[self.POS, self.VEL] = np.eye(3) * self.dt

        # ∂velocity/∂orientation (linearized rotation effect)
        # Simplified: cross-product matrix of accel_corrected
        ax, ay, az = accel_corrected
        skew_accel = np.array([
            [0, -az, ay],
            [az, 0, -ax],
            [-ay, ax, 0]
        ])
        F[3:6, 6:9] = -R_b2n @ skew_accel * self.dt

        # ∂velocity/∂accel_bias
        F[3:6, 9:12] = -R_b2n * self.dt

        # ∂orientation/∂gyro_bias
        F[6:9, 12:15] = -np.eye(3) * self.dt

        # --- Covariance propagation ---
        Q = self._build_process_noise()

        # Increase process noise during dead reckoning (uncertainty grows faster)
        if self.mode == NavigationMode.DEAD_RECKONING:
            Q[self.POS, self.POS] *= 5.0
            Q[self.VEL, self.VEL] *= 3.0

        self.P = F @ self.P @ F.T + Q

        # --- AI velocity as pseudo-measurement during DR ---
        if ai_velocity is not None and self.mode == NavigationMode.DEAD_RECKONING:
            self._update_ai_velocity(ai_velocity)

    def update_gnss(
        self,
        gnss_position: np.ndarray,
        gnss_velocity: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ):
        """
        EKF update step using GNSS measurements.

        Args:
            gnss_position: (3,) or (2,) GNSS position in ENU [meters].
            gnss_velocity: Optional (3,) or (2,) GNSS velocity [m/s].
            timestamp: Current time for outage tracking.
        """
        # Handle mode transitions
        if self.mode == NavigationMode.DEAD_RECKONING:
            self.mode = NavigationMode.REACQUISITION
            # Temporarily trust GNSS more during re-acquisition
            reacq_scale = 0.5
        else:
            self.mode = NavigationMode.GNSS_INS
            reacq_scale = 1.0

        if timestamp is not None:
            self.last_gnss_time = timestamp

        # --- Position update ---
        if len(gnss_position) == 2:
            gnss_position = np.array([gnss_position[0], gnss_position[1], 0.0])

        # Measurement matrix (position observation)
        H_pos = np.zeros((3, self.STATE_DIM))
        H_pos[:3, :3] = np.eye(3)

        # Measurement noise
        R_pos = np.eye(3) * (self.r_gnss_pos ** 2) * reacq_scale

        # Innovation
        z_pos = gnss_position
        y_pos = z_pos - H_pos @ self.x

        # Kalman gain
        S_pos = H_pos @ self.P @ H_pos.T + R_pos
        K_pos = self.P @ H_pos.T @ np.linalg.inv(S_pos)

        # State update
        self.x += K_pos @ y_pos
        I_KH = np.eye(self.STATE_DIM) - K_pos @ H_pos
        self.P = I_KH @ self.P @ I_KH.T + K_pos @ R_pos @ K_pos.T

        # --- Velocity update (if available) ---
        if gnss_velocity is not None:
            if len(gnss_velocity) == 2:
                gnss_velocity = np.array([gnss_velocity[0], gnss_velocity[1], 0.0])

            H_vel = np.zeros((3, self.STATE_DIM))
            H_vel[:3, 3:6] = np.eye(3)

            R_vel = np.eye(3) * (self.r_gnss_vel ** 2) * reacq_scale

            y_vel = gnss_velocity - H_vel @ self.x
            S_vel = H_vel @ self.P @ H_vel.T + R_vel
            K_vel = self.P @ H_vel.T @ np.linalg.inv(S_vel)

            self.x += K_vel @ y_vel
            I_KH_v = np.eye(self.STATE_DIM) - K_vel @ H_vel
            self.P = I_KH_v @ self.P @ I_KH_v.T + K_vel @ R_vel @ K_vel.T

        # Store innovation for diagnostics
        self.innovation_history.append(float(np.linalg.norm(y_pos)))

        # Return to normal fusion after a few updates in reacquisition
        if self.mode == NavigationMode.REACQUISITION:
            if len(self.innovation_history) >= 5:
                recent = self.innovation_history[-5:]
                if all(inn < self.r_gnss_pos * 3 for inn in recent):
                    self.mode = NavigationMode.GNSS_INS

    def _update_ai_velocity(self, ai_velocity: np.ndarray):
        """
        Update state using AI-predicted velocity as a measurement.

        Used during GNSS denial to bound velocity drift.

        Args:
            ai_velocity: (2,) predicted velocity [v_north, v_east] in m/s.
        """
        # Measurement: v_E, v_N from AI
        H = np.zeros((2, self.STATE_DIM))
        H[0, 3] = 1.0  # v_east
        H[1, 4] = 1.0  # v_north

        R = np.eye(2) * (self.r_ai_vel ** 2)

        z = np.array([ai_velocity[1], ai_velocity[0]])  # [v_east, v_north]
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def set_gnss_denied(self, timestamp: Optional[float] = None):
        """
        Signal that GNSS is no longer available.

        Args:
            timestamp: Time of GNSS loss for outage tracking.
        """
        if self.mode != NavigationMode.DEAD_RECKONING:
            self.mode = NavigationMode.DEAD_RECKONING
            self.gnss_outage_start = timestamp

    def update_zupt(self, velocity_sigma: float = 0.01):
        """
        Apply Zero Velocity Update — reset velocity to near-zero.

        Called when the vehicle is detected as stationary.

        Args:
            velocity_sigma: Measurement noise for the zero-velocity observation.
        """
        H = np.zeros((3, self.STATE_DIM))
        H[:3, 3:6] = np.eye(3)

        R = np.eye(3) * velocity_sigma ** 2

        z = np.zeros(3)  # Zero velocity
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_position(self) -> np.ndarray:
        """Return current estimated position (East, North, Up)."""
        return self.x[self.POS].copy()

    def get_velocity(self) -> np.ndarray:
        """Return current estimated velocity (v_E, v_N, v_U)."""
        return self.x[self.VEL].copy()

    def get_heading(self) -> float:
        """Return current estimated heading (yaw) in radians."""
        return float(self.x[8])

    def get_position_uncertainty(self) -> float:
        """Return 1-sigma position uncertainty in meters (horizontal)."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

    def get_state_summary(self) -> Dict:
        """Return a human-readable summary of the current state."""
        pos = self.get_position()
        vel = self.get_velocity()
        speed = np.linalg.norm(vel[:2])
        heading_deg = np.degrees(self.get_heading()) % 360

        return {
            "mode": self.mode.value,
            "position_enu": pos.tolist(),
            "velocity_enu": vel.tolist(),
            "speed_ms": float(speed),
            "speed_kmh": float(speed * 3.6),
            "heading_deg": float(heading_deg),
            "position_uncertainty_m": self.get_position_uncertainty(),
            "accel_bias": self.x[self.ABIAS].tolist(),
            "gyro_bias": self.x[self.GBIAS].tolist(),
        }

    def initialize_from_gnss(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        heading: Optional[float] = None,
    ):
        """
        Initialize EKF state from a GNSS fix.

        Args:
            position: (2,) or (3,) initial position in ENU.
            velocity: Optional (2,) or (3,) initial velocity.
            heading: Optional initial heading in radians.
        """
        if len(position) >= 2:
            self.x[0] = position[0]
            self.x[1] = position[1]
        if len(position) >= 3:
            self.x[2] = position[2]

        if velocity is not None:
            if len(velocity) >= 2:
                self.x[3] = velocity[0]
                self.x[4] = velocity[1]
            if len(velocity) >= 3:
                self.x[5] = velocity[2]

        if heading is not None:
            self.x[8] = heading

        # Reduce initial uncertainty
        self.P[self.POS, self.POS] = np.eye(3) * (self.r_gnss_pos ** 2)
        self.mode = NavigationMode.GNSS_INS
