"""
Navigators IDR — Non-Holonomic Constraints (NHC)
Enforces vehicle kinematic constraints to reduce impossible trajectories.

Non-holonomic constraints for a ground vehicle:
    1. No lateral velocity (wheels don't slide sideways)
    2. No vertical velocity (vehicle stays on the ground)
    3. Heading rate bounded by vehicle dynamics

These constraints are applied as pseudo-measurements in the EKF.
"""

import numpy as np
from typing import Optional, Tuple


class NonHolonomicConstraints:
    """
    Applies non-holonomic constraints to vehicle navigation estimates.

    Ground vehicles cannot move sideways or vertically (relative to their
    body frame). These constraints significantly reduce DR drift by
    eliminating physically impossible motion estimates.
    """

    def __init__(
        self,
        lateral_velocity_sigma: float = 0.1,
        vertical_velocity_sigma: float = 0.1,
        max_heading_rate: float = 0.5,
    ):
        """
        Args:
            lateral_velocity_sigma: Expected lateral velocity noise (m/s).
                Lower values = stricter constraint.
            vertical_velocity_sigma: Expected vertical velocity noise (m/s).
            max_heading_rate: Maximum turning rate (rad/s) for constraint checking.
        """
        self.lateral_sigma = lateral_velocity_sigma
        self.vertical_sigma = vertical_velocity_sigma
        self.max_heading_rate = max_heading_rate

    def compute_body_velocity(
        self,
        velocity_nav: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """
        Convert navigation-frame velocity to body-frame velocity.

        Body frame:
            x = forward (along vehicle heading)
            y = right (lateral)
            z = down (vertical)

        Args:
            velocity_nav: (2,) or (3,) velocity in ENU [v_east, v_north, (v_up)].
            heading: Vehicle heading (yaw) in radians.

        Returns:
            (3,) velocity in body frame [v_forward, v_lateral, v_vertical].
        """
        v_e = velocity_nav[0]
        v_n = velocity_nav[1]
        v_u = velocity_nav[2] if len(velocity_nav) > 2 else 0.0

        # Rotation from navigation to body frame (2D for horizontal)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        v_forward = v_e * sin_h + v_n * cos_h
        v_lateral = v_e * cos_h - v_n * sin_h
        v_vertical = -v_u  # Down = -Up

        return np.array([v_forward, v_lateral, v_vertical])

    def apply_constraints(
        self,
        velocity_nav: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """
        Apply NHC by zeroing lateral and vertical velocity components.

        Args:
            velocity_nav: (2,) or (3,) velocity in ENU.
            heading: Vehicle heading in radians.

        Returns:
            (2,) or (3,) constrained velocity in ENU.
        """
        v_body = self.compute_body_velocity(velocity_nav, heading)

        # Zero out lateral and vertical velocity
        v_body[1] = 0.0  # No sideslip
        v_body[2] = 0.0  # No vertical motion

        # Convert back to navigation frame
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        v_e = v_body[0] * sin_h
        v_n = v_body[0] * cos_h

        if len(velocity_nav) > 2:
            return np.array([v_e, v_n, 0.0])
        return np.array([v_e, v_n])

    def get_nhc_measurement_matrix(
        self,
        heading: float,
        state_dim: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the NHC measurement matrix for EKF update.

        The NHC is formulated as a zero-velocity measurement in the
        lateral and vertical directions of the body frame.

        Args:
            heading: Current heading in radians.
            state_dim: EKF state dimension.

        Returns:
            Tuple of (H, z, R):
                H: (2, state_dim) measurement matrix
                z: (2,) measurement (zeros)
                R: (2, 2) measurement noise covariance
        """
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # H maps velocity states to lateral and vertical body velocities
        H = np.zeros((2, state_dim))

        # Lateral velocity = v_e * cos(h) - v_n * sin(h)
        H[0, 3] = cos_h    # ∂v_lateral/∂v_east
        H[0, 4] = -sin_h   # ∂v_lateral/∂v_north

        # Vertical velocity = -v_up
        H[1, 5] = -1.0     # ∂v_vertical/∂v_up

        # Zero measurement (lateral and vertical velocity should be zero)
        z = np.zeros(2)

        # Measurement noise
        R = np.diag([self.lateral_sigma ** 2, self.vertical_sigma ** 2])

        return H, z, R

    def check_heading_rate(self, heading_rate: float) -> float:
        """
        Constrain heading rate to physically plausible values.

        Args:
            heading_rate: Measured heading rate in rad/s.

        Returns:
            Constrained heading rate.
        """
        return np.clip(heading_rate, -self.max_heading_rate, self.max_heading_rate)

    def validate_velocity(
        self,
        velocity_nav: np.ndarray,
        heading: float,
        speed_limit: float = 50.0,
    ) -> dict:
        """
        Check if velocity satisfies non-holonomic constraints.

        Args:
            velocity_nav: (2+,) velocity in ENU frame.
            heading: Vehicle heading in radians.
            speed_limit: Maximum plausible speed (m/s).

        Returns:
            Dict with validation results and violation magnitudes.
        """
        v_body = self.compute_body_velocity(velocity_nav, heading)
        speed = abs(v_body[0])

        return {
            "valid": True if (
                abs(v_body[1]) < self.lateral_sigma * 3
                and abs(v_body[2]) < self.vertical_sigma * 3
                and speed < speed_limit
            ) else False,
            "forward_speed": float(v_body[0]),
            "lateral_velocity": float(v_body[1]),
            "vertical_velocity": float(v_body[2]),
            "lateral_violation": float(max(0, abs(v_body[1]) - self.lateral_sigma * 3)),
            "speed_violation": float(max(0, speed - speed_limit)),
        }
