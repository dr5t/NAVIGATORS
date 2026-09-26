"""
Navigators IDR - Extended Kalman Filter
15-state EKF for vehicle navigation with GNSS, INS, AI-velocity, NHC, and ZUPT fusion.

State vector (15 elements):
    [0:3]   - Position (East, North, Up) in meters [ENU frame]
    [3:6]   - Velocity (v_E, v_N, v_U) in m/s
    [6:9]   - Orientation (roll, pitch, yaw) in radians
    [9:12]  - Accelerometer bias (b_ax, b_ay, b_az) in m/s²
    [12:15] - Gyroscope bias (b_gx, b_gy, b_gz) in rad/s

Operating modes:
    1. GNSS_AIDED (normal high-accuracy fusion)
    2. GNSS_DEGRADED (poor satellite geometry/high noise)
    3. DEAD_RECKONING (GNSS outage: AI velocity + NHC + ZUPT)
    4. REACQUISITION (GNSS restored: smooth, non-jumping convergence)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum

try:
    from navigation.interfaces import AIVelocityMeasurement, MapConstraint
except ImportError:
    from src.navigation.interfaces import AIVelocityMeasurement, MapConstraint



class NavigationMode(Enum):
    """Current navigation operating mode."""
    GNSS_AIDED = "gnss_aided"
    GNSS_INS = "gnss_ins"
    GNSS_DEGRADED = "gnss_degraded"
    DEAD_RECKONING = "dr"
    REACQUISITION = "reacq"


class ExtendedKalmanFilter:
    """
    15-state Extended Kalman Filter for vehicle navigation.
    
    Features:
    - AI velocity pseudo-measurement integration
    - Kinematic Non-Holonomic Constraints (NHC) measurement updates
    - Zero Velocity Updates (ZUPT)
    - Anti-jump smooth GNSS reacquisition
    - Numerically stable Joseph-form covariance updates
    """


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
        self.dt = dt


        pn = process_noise or {}
        self.q_pos = pn.get("position", 0.5)
        self.q_vel = pn.get("velocity", 2.0)
        self.q_ori = pn.get("orientation", 0.05)
        self.q_abias = pn.get("accel_bias", 0.001)
        self.q_gbias = pn.get("gyro_bias", 0.0001)


        gn = gnss_noise or {}
        self.r_gnss_pos = gn.get("position", 2.5)
        self.r_gnss_vel = gn.get("velocity", 0.5)
        self.r_ai_vel = ai_velocity_noise
        self.r_nhc_lateral = 1.0
        self.r_nhc_vertical = 1.0


        self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.P = np.eye(self.STATE_DIM, dtype=np.float64)


        self.P[self.POS, self.POS] *= 10.0
        self.P[self.VEL, self.VEL] *= 5.0
        self.P[self.ORI, self.ORI] *= 0.1
        self.P[self.ABIAS, self.ABIAS] *= 0.5
        self.P[self.GBIAS, self.GBIAS] *= 0.01


        self.mode = NavigationMode.GNSS_INS
        self.gnss_outage_start: Optional[float] = None
        self.last_gnss_time: Optional[float] = None
        self.consecutive_good_gnss = 0
        self.reacquisition_steps = 0


        self.innovation_history: List[float] = []
        self.last_trust_metric = None

    def _build_process_noise(self) -> np.ndarray:
        """Construct the process noise covariance matrix Q."""
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=np.float64)
        Q[self.POS, self.POS] = np.eye(3) * (self.q_pos * (self.dt ** 2))
        Q[self.VEL, self.VEL] = np.eye(3) * (self.q_vel * self.dt)
        Q[self.ORI, self.ORI] = np.eye(3) * (self.q_ori * self.dt)
        Q[self.ABIAS, self.ABIAS] = np.eye(3) * (self.q_abias * self.dt)
        Q[self.GBIAS, self.GBIAS] = np.eye(3) * (self.q_gbias * self.dt)
        return Q

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Compute rotation matrix from Vehicle Body Frame (Forward, Right, Down)
        to Navigation Frame (East, North, Up) [ENU].
        Yaw is heading in radians clockwise from North (0 = North, pi/2 = East).
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.sin(yaw), np.cos(yaw)




        R = np.array([
            [cy * cp,   sy * cr + cy * sp * sr,  -sy * sr + cy * sp * cr],
            [sy * cp,  -cy * cr + sy * sp * sr,   cy * sr + sy * sp * cr],
            [sp,       -cp * sr,                 -cp * cr               ]
        ], dtype=np.float64)
        return R

    def _enforce_covariance_symmetry(self):
        """Guarantee positive-definiteness and symmetry of covariance matrix P."""
        self.P = 0.5 * (self.P + self.P.T)

        for i in range(self.STATE_DIM):
            if self.P[i, i] < 1e-9:
                self.P[i, i] = 1e-9

    def predict(
        self,
        accel_body: np.ndarray,
        gyro_body: np.ndarray,
        ai_velocity: Optional[Union[np.ndarray, AIVelocityMeasurement]] = None,
        apply_nhc: bool = True,
    ):
        """
        EKF propagation step using IMU measurements.
        
        Args:
            accel_body: (3,) accelerometer readings [m/s²].
            gyro_body: (3,) gyroscope readings [rad/s].
            ai_velocity: Optional (2,) predicted velocity in [v_east, v_north].
            apply_nhc: Whether to apply kinematic Non-Holonomic Constraints during prediction.
        """
        roll, pitch, yaw = self.x[self.ORI]
        accel_bias = self.x[self.ABIAS]
        gyro_bias = self.x[self.GBIAS]


        accel_corrected = accel_body - accel_bias
        gyro_corrected = gyro_body - gyro_bias

        R_b2n = self._rotation_matrix(roll, pitch, yaw)


        gravity_nav = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        accel_nav = R_b2n @ accel_corrected - gravity_nav




        self.x[self.POS] += self.x[self.VEL] * self.dt + 0.5 * accel_nav * (self.dt ** 2)
        self.x[self.VEL] += accel_nav * self.dt

        self.x[self.ORI] += gyro_corrected * self.dt


        self.x[8] = (self.x[8] + np.pi) % (2.0 * np.pi) - np.pi


        F = np.eye(self.STATE_DIM, dtype=np.float64)
        F[self.POS, self.VEL] = np.eye(3) * self.dt


        ax, ay, az = accel_corrected
        skew_accel = np.array([
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0]
        ], dtype=np.float64)
        F[3:6, 6:9] = -R_b2n @ skew_accel * self.dt
        F[3:6, 9:12] = -R_b2n * self.dt
        F[6:9, 12:15] = -np.eye(3) * self.dt


        Q = self._build_process_noise()
        if self.mode == NavigationMode.DEAD_RECKONING:
            Q[self.POS, self.POS] *= 4.0
            Q[self.VEL, self.VEL] *= 2.0

        self.P = F @ self.P @ F.T + Q
        self._enforce_covariance_symmetry()


        if ai_velocity is not None:
            if hasattr(ai_velocity, "is_valid"):
                self.last_ai_measurement = ai_velocity
            if self.mode in [NavigationMode.DEAD_RECKONING, NavigationMode.GNSS_DEGRADED]:
                self._update_ai_velocity(ai_velocity)


        if apply_nhc:
            self.update_nhc(yaw_rate=float(gyro_corrected[2]))

    def update_gnss(
        self,
        gnss_position: np.ndarray,
        gnss_velocity: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        trust_metric: Optional[Any] = None,
    ):
        if trust_metric is not None:
            self.last_trust_metric = trust_metric
            state_val = getattr(trust_metric, "state", None)
            if state_val == "UNUSABLE" or (getattr(state_val, "value", None) == "UNUSABLE"):
                return
        prior_x = self.x.copy()
        prior_p = self.P.copy()
        if self.mode == NavigationMode.DEAD_RECKONING:
            self.mode = NavigationMode.REACQUISITION
            self.reacquisition_steps = 0
            self.consecutive_good_gnss = 1
        elif self.mode == NavigationMode.REACQUISITION:
            self.reacquisition_steps += 1
            self.consecutive_good_gnss += 1
        else:
            self.mode = NavigationMode.GNSS_INS
            self.consecutive_good_gnss += 1

        if timestamp is not None:
            self.last_gnss_time = timestamp


        if len(gnss_position) == 2:
            gnss_pos_3d = np.array([gnss_position[0], gnss_position[1], 0.0], dtype=np.float64)
        else:
            gnss_pos_3d = np.array(gnss_position[:3], dtype=np.float64)

        H_pos = np.zeros((3, self.STATE_DIM), dtype=np.float64)
        H_pos[:3, :3] = np.eye(3)

        R_pos = np.eye(3, dtype=np.float64) * (self.r_gnss_pos ** 2)
        if trust_metric is not None and hasattr(trust_metric, "position_variance_scale"):
            R_pos = R_pos * float(trust_metric.position_variance_scale)



        if self.mode == NavigationMode.REACQUISITION:

            ramp = max(0.1, min(1.0, self.reacquisition_steps / 10.0))
            R_pos = R_pos * (1.0 / ramp)

        z_pos = gnss_pos_3d
        y_pos = z_pos - H_pos @ self.x


        self.innovation_history.append(float(np.linalg.norm(y_pos[:2])))


        if self.mode == NavigationMode.REACQUISITION:
            max_step = 10.0
            y_norm = np.linalg.norm(y_pos[:2])
            if y_norm > max_step:
                y_pos[:2] = y_pos[:2] * (max_step / y_norm)


        S_pos = H_pos @ self.P @ H_pos.T + R_pos
        K_pos = self.P @ H_pos.T @ np.linalg.inv(S_pos)


        self.x += K_pos @ y_pos
        I_KH = np.eye(self.STATE_DIM) - K_pos @ H_pos
        self.P = I_KH @ self.P @ I_KH.T + K_pos @ R_pos @ K_pos.T
        self._enforce_covariance_symmetry()


        if gnss_velocity is not None:
            if len(gnss_velocity) == 2:
                gnss_vel_3d = np.array([gnss_velocity[0], gnss_velocity[1], 0.0], dtype=np.float64)
            else:
                gnss_vel_3d = np.array(gnss_velocity[:3], dtype=np.float64)

            H_vel = np.zeros((3, self.STATE_DIM), dtype=np.float64)
            H_vel[:3, 3:6] = np.eye(3)
            R_vel = np.eye(3, dtype=np.float64) * (self.r_gnss_vel ** 2)
            if trust_metric is not None and hasattr(trust_metric, "velocity_variance_scale"):
                R_vel = R_vel * float(trust_metric.velocity_variance_scale)

            y_vel = gnss_vel_3d - H_vel @ self.x
            S_vel = H_vel @ self.P @ H_vel.T + R_vel
            K_vel = self.P @ H_vel.T @ np.linalg.inv(S_vel)

            self.x += K_vel @ y_vel
            I_KH_v = np.eye(self.STATE_DIM) - K_vel @ H_vel
            self.P = I_KH_v @ self.P @ I_KH_v.T + K_vel @ R_vel @ K_vel.T
            self._enforce_covariance_symmetry()


            speed_horiz = float(np.linalg.norm(gnss_vel_3d[:2]))
            if speed_horiz > 2.0:
                cog = float(np.arctan2(gnss_vel_3d[0], gnss_vel_3d[1]))
                yaw_err = (cog - self.x[8] + np.pi) % (2.0 * np.pi) - np.pi
                self.x[8] = (self.x[8] + 0.15 * yaw_err + np.pi) % (2.0 * np.pi) - np.pi
                cog_var = (self.r_gnss_vel / speed_horiz) ** 2
                self.P[8, 8] = float(max(0.005, (1.0 - 0.15) ** 2 * self.P[8, 8] + (0.15 ** 2) * cog_var))

        if self.mode == NavigationMode.REACQUISITION:


            correction = self.x - prior_x
            correction[8] = (correction[8] + np.pi) % (2 * np.pi) - np.pi
            distance = float(np.linalg.norm(correction[:2]))
            fraction = min(1.0, 2.0 / max(distance, 1e-12))
            self.x = prior_x + fraction * correction
            self.P = (1 - fraction) * prior_p + fraction * self.P
            self._enforce_covariance_symmetry()
            residual = np.linalg.norm(gnss_pos_3d[:2] - self.x[:2])
            if self.consecutive_good_gnss >= 5 and residual <= 3.0:
                self.mode = NavigationMode.GNSS_INS

    def _update_ai_velocity(self, ai_velocity: Any):
        if ai_velocity is None:
            return
        if hasattr(ai_velocity, "is_valid"):
            self.last_ai_measurement = ai_velocity
            if not ai_velocity.is_valid:
                return
            v_east = float(ai_velocity.velocity_east)
            v_north = float(ai_velocity.velocity_north)
            var_e = float(getattr(ai_velocity, "variance_east", self.r_ai_vel ** 2))
            var_n = float(getattr(ai_velocity, "variance_north", self.r_ai_vel ** 2))
            R = np.diag([var_e, var_n])
        else:
            v_east = float(ai_velocity[0])
            v_north = float(ai_velocity[1])
            R = np.eye(2, dtype=np.float64) * (self.r_ai_vel ** 2)

        H = np.zeros((2, self.STATE_DIM), dtype=np.float64)
        H[0, 3] = 1.0
        H[1, 4] = 1.0

        z = np.array([v_east, v_north], dtype=np.float64)
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._enforce_covariance_symmetry()

        speed_horiz = float(np.linalg.norm(z))
        if speed_horiz > 2.0:
            cog = float(np.arctan2(v_east, v_north))
            yaw_err = (cog - self.x[8] + np.pi) % (2.0 * np.pi) - np.pi
            self.x[8] = (self.x[8] + 0.15 * yaw_err + np.pi) % (2.0 * np.pi) - np.pi

    def apply_adaptive_noise(self, noise_params: Any):
        if noise_params is not None:
            self.adaptive_noise_params = noise_params

    def update_map_constraint(
        self,
        snapped_position: Optional[Union[np.ndarray, MapConstraint]] = None,
        confidence: float = 1.0,
        covariance_scale: float = 1.0,
        constraint: Optional[MapConstraint] = None,
    ):
        target_constraint = constraint
        if isinstance(snapped_position, MapConstraint):
            target_constraint = snapped_position
        elif target_constraint is None and snapped_position is None:
            return

        if target_constraint is not None:
            if not target_constraint.has_constraint:
                return
            if target_constraint.measurement_matrix is None or target_constraint.noise_covariance is None:
                if target_constraint.snapped_position is not None:
                    snapped_position = target_constraint.snapped_position
                    confidence = target_constraint.confidence
                else:
                    return
            else:
                H = np.asarray(target_constraint.measurement_matrix, dtype=np.float64)
                R = np.asarray(target_constraint.noise_covariance, dtype=np.float64)
                z = np.asarray(target_constraint.observation_vector, dtype=np.float64)
                if H.shape[1] != self.STATE_DIM:
                    H_padded = np.zeros((H.shape[0], self.STATE_DIM), dtype=np.float64)
                    H_padded[:, :min(H.shape[1], self.STATE_DIM)] = H[:, :min(H.shape[1], self.STATE_DIM)]
                    H = H_padded
                y = z - H @ self.x
                if target_constraint.heading_constraint_applied and len(z) >= 3:
                    y[2] = (z[2] - self.x[8] + np.pi) % (2.0 * np.pi) - np.pi
                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x += K @ y
                self.x[8] = (self.x[8] + np.pi) % (2.0 * np.pi) - np.pi
                I_KH = np.eye(self.STATE_DIM) - K @ H
                self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
                self._enforce_covariance_symmetry()
                return

        if snapped_position is None or confidence <= 0.2:
            return
        H = np.zeros((2, self.STATE_DIM), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        base_road_variance = 4.0
        r_val = (base_road_variance / max(confidence, 0.05)) * max(0.05, covariance_scale)
        R = np.eye(2, dtype=np.float64) * r_val

        pos_arr = np.asarray(snapped_position)
        z = np.asarray(pos_arr[:2], dtype=np.float64)
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._enforce_covariance_symmetry()


    def update_nhc(self, yaw_rate: float = 0.0):
        """
        Apply Non-Holonomic Constraints (NHC) as a pseudo-measurement.
        Relaxed during turns to avoid fighting vehicle steering dynamics.
        """
        speed = float(np.linalg.norm(self.x[self.VEL][:2]))
        if speed < 0.5:
            return


        turn_dilation = 1.0 + 30.0 * (abs(yaw_rate) ** 2)
        r_lat = self.r_nhc_lateral * np.sqrt(turn_dilation)
        r_vert = self.r_nhc_vertical

        heading = self.get_heading()
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)




        H = np.zeros((2, self.STATE_DIM), dtype=np.float64)
        H[0, 3] = cos_h
        H[0, 4] = -sin_h
        H[1, 5] = -1.0

        R = np.diag([r_lat ** 2, r_vert ** 2])

        z = np.zeros(2, dtype=np.float64)
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._enforce_covariance_symmetry()

    def update_zupt(self, velocity_sigma: float = 0.01):
        """
        Apply Zero Velocity Update (ZUPT) when vehicle is detected stationary.
        """
        H = np.zeros((3, self.STATE_DIM), dtype=np.float64)
        H[:3, 3:6] = np.eye(3)

        R = np.eye(3, dtype=np.float64) * (velocity_sigma ** 2)
        z = np.zeros(3, dtype=np.float64)
        y = z - H @ self.x

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._enforce_covariance_symmetry()

    def set_gnss_denied(self, timestamp: Optional[float] = None):
        """Signal that GNSS is lost and switch to Dead Reckoning."""
        if self.mode != NavigationMode.DEAD_RECKONING:
            self.mode = NavigationMode.DEAD_RECKONING
            self.gnss_outage_start = timestamp
            self.consecutive_good_gnss = 0

    def get_position(self) -> np.ndarray:
        return self.x[self.POS].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[self.VEL].copy()

    def get_heading(self) -> float:
        return float(self.x[8])

    def get_position_uncertainty(self) -> float:
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

    def get_velocity_uncertainty(self) -> float:
        return float(np.sqrt(self.P[3, 3] + self.P[4, 4]))

    def get_heading_uncertainty_rad(self) -> float:
        return float(np.sqrt(max(0.0, self.P[8, 8])))

    def get_heading_uncertainty_deg(self) -> float:
        return float(np.degrees(self.get_heading_uncertainty_rad()))

    def get_uncertainty_profile(self) -> Dict[str, float]:
        return {
            "position_horizontal_m": self.get_position_uncertainty(),
            "position_vertical_m": float(np.sqrt(max(0.0, self.P[2, 2]))),
            "velocity_horizontal_mps": self.get_velocity_uncertainty(),
            "velocity_vertical_mps": float(np.sqrt(max(0.0, self.P[5, 5]))),
            "heading_deg": self.get_heading_uncertainty_deg(),
            "heading_rad": self.get_heading_uncertainty_rad(),
        }

    def get_state_summary(self) -> Dict[str, Any]:
        pos = self.get_position()
        vel = self.get_velocity()
        speed = np.linalg.norm(vel[:2])
        heading_deg = np.degrees(self.get_heading()) % 360

        summary = {
            "mode": self.mode.value,
            "position_enu": pos.tolist(),
            "velocity_enu": vel.tolist(),
            "speed_ms": float(speed),
            "speed_kmh": float(speed * 3.6),
            "heading_deg": float(heading_deg),
            "position_uncertainty_m": self.get_position_uncertainty(),
            "velocity_uncertainty_mps": self.get_velocity_uncertainty(),
            "heading_uncertainty_deg": self.get_heading_uncertainty_deg(),
            "uncertainties": self.get_uncertainty_profile(),
            "accel_bias": self.x[self.ABIAS].tolist(),
            "gyro_bias": self.x[self.GBIAS].tolist(),
        }
        if hasattr(self, "last_trust_metric") and self.last_trust_metric is not None:
            summary["gnss_trust_state"] = getattr(self.last_trust_metric.state, "value", str(self.last_trust_metric.state))
            summary["gnss_trust_score"] = float(self.last_trust_metric.trust_score)
        if hasattr(self, "last_ai_measurement") and self.last_ai_measurement is not None:
            summary["ai_velocity_valid"] = bool(getattr(self.last_ai_measurement, "is_valid", False))
            summary["ai_velocity_ms"] = [
                float(getattr(self.last_ai_measurement, "velocity_east", 0.0)),
                float(getattr(self.last_ai_measurement, "velocity_north", 0.0)),
            ]
        if hasattr(self, "adaptive_noise_params") and self.adaptive_noise_params is not None:
            summary["adaptive_fusion"] = {
                "process_noise_scale_pos": float(self.adaptive_noise_params.process_noise_scale_pos),
                "process_noise_scale_vel": float(self.adaptive_noise_params.process_noise_scale_vel),
                "measurement_noise_scale_gnss_pos": float(self.adaptive_noise_params.measurement_noise_scale_gnss_pos),
                "measurement_noise_scale_ai_vel": float(self.adaptive_noise_params.measurement_noise_scale_ai_vel),
                "measurement_noise_scale_map": float(self.adaptive_noise_params.measurement_noise_scale_map),
                "downweighted_reasons": list(self.adaptive_noise_params.downweighted_reasons),
                "rejected_reasons": list(self.adaptive_noise_params.rejected_reasons),
                "telemetry": dict(self.adaptive_noise_params.telemetry),
            }
        return summary

    def initialize_from_gnss(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        heading: Optional[float] = None,
    ):
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

        self.P[self.POS, self.POS] = np.eye(3) * (self.r_gnss_pos ** 2)
        self.mode = NavigationMode.GNSS_INS
        self.consecutive_good_gnss = 5
