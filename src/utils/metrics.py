"""
Navigators IDR - Evaluation Metrics
Positional drift, CEP, along/cross-track error, and convergence metrics.

All metrics assume inputs in ENU (meters) or LLA (degrees) coordinates.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any


def positional_drift_percent(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray
) -> float:
    """
    Compute positional drift as a percentage of total distance traveled.

    This is the primary metric from the project requirements:
    "Dead-reckoning positional drift: < 10% of the distance traveled."

    Args:
        estimated_positions: (N, 2) or (N, 3) array of estimated ENU positions.
        true_positions: (N, 2) or (N, 3) array of ground truth ENU positions.

    Returns:
        Drift percentage (e.g., 5.2 means 5.2% drift).
    """
    # Final position error
    final_error = np.linalg.norm(estimated_positions[-1, :2] - true_positions[-1, :2])

    # Total distance traveled (ground truth)
    deltas = np.diff(true_positions[:, :2], axis=0)
    total_distance = np.sum(np.linalg.norm(deltas, axis=1))

    if total_distance < 1e-6:
        return 0.0

    return (final_error / total_distance) * 100.0


def absolute_trajectory_error(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE) statistics.

    Args:
        estimated_positions: (N, 2) or (N, 3) estimated ENU positions.
        true_positions: (N, 2) or (N, 3) true ENU positions.

    Returns:
        Dict with 'mean', 'median', 'std', 'max', 'rmse' in meters.
    """
    errors = np.linalg.norm(
        estimated_positions[:, :2] - true_positions[:, :2], axis=1
    )

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "max": float(np.max(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def circular_error_probable(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray,
    percentiles: Tuple[float, ...] = (50.0, 95.0)
) -> Dict[str, float]:
    """
    Compute Circular Error Probable (CEP) at specified percentiles.

    CEP50 = radius enclosing 50% of position errors
    CEP95 = radius enclosing 95% of position errors

    Args:
        estimated_positions: (N, 2) or (N, 3) estimated positions.
        true_positions: (N, 2) or (N, 3) true positions.
        percentiles: Tuple of percentiles to compute.

    Returns:
        Dict mapping 'CEP{p}' to radius in meters.
    """
    errors = np.linalg.norm(
        estimated_positions[:, :2] - true_positions[:, :2], axis=1
    )

    result = {}
    for p in percentiles:
        result[f"CEP{int(p)}"] = float(np.percentile(errors, p))

    return result


def along_cross_track_error(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Decompose position error into along-track and cross-track components.

    Along-track error: error in the direction of travel (longitudinal).
    Cross-track error: error perpendicular to direction of travel (lateral).

    Args:
        estimated_positions: (N, 2) estimated ENU positions.
        true_positions: (N, 2) true ENU positions.

    Returns:
        Dict with 'along_track' and 'cross_track' arrays (N,).
    """
    N = len(true_positions)
    along_track = np.zeros(N)
    cross_track = np.zeros(N)

    for i in range(1, N):
        # Direction of travel (from ground truth)
        direction = true_positions[i, :2] - true_positions[i - 1, :2]
        dist = np.linalg.norm(direction)

        if dist < 1e-10:
            continue

        unit_along = direction / dist
        unit_cross = np.array([-unit_along[1], unit_along[0]])

        # Position error vector
        error = estimated_positions[i, :2] - true_positions[i, :2]

        along_track[i] = np.dot(error, unit_along)
        cross_track[i] = np.dot(error, unit_cross)

    return {
        "along_track": along_track,
        "cross_track": cross_track,
    }


def velocity_error(
    estimated_velocity: np.ndarray,
    true_velocity: np.ndarray
) -> Dict[str, float]:
    """
    Compute velocity estimation error statistics.

    Args:
        estimated_velocity: (N, 2) estimated velocity [v_north, v_east].
        true_velocity: (N, 2) true velocity.

    Returns:
        Dict with speed and heading error statistics.
    """
    # Speed errors
    est_speed = np.linalg.norm(estimated_velocity, axis=1)
    true_speed = np.linalg.norm(true_velocity, axis=1)
    speed_errors = np.abs(est_speed - true_speed)

    # Heading errors (angular)
    est_heading = np.arctan2(estimated_velocity[:, 1], estimated_velocity[:, 0])
    true_heading = np.arctan2(true_velocity[:, 1], true_velocity[:, 0])

    heading_errors = np.abs(est_heading - true_heading)
    # Wrap to [0, π]
    heading_errors = np.minimum(heading_errors, 2 * np.pi - heading_errors)

    return {
        "speed_rmse": float(np.sqrt(np.mean(speed_errors ** 2))),
        "speed_mean": float(np.mean(speed_errors)),
        "speed_max": float(np.max(speed_errors)),
        "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(heading_errors ** 2)))),
        "heading_mean_deg": float(np.degrees(np.mean(heading_errors))),
    }


def gnss_reacquisition_convergence(
    position_errors: np.ndarray,
    threshold: float = 5.0,
    update_rate: float = 10.0
) -> Optional[float]:
    """
    Compute time to converge after GNSS re-acquisition.

    Measures how quickly position error drops below a threshold after
    GNSS signals are restored.

    Args:
        position_errors: (N,) array of position errors in meters after re-acquisition.
        threshold: Convergence threshold in meters.
        update_rate: System update rate in Hz.

    Returns:
        Convergence time in seconds, or None if threshold never reached.
    """
    below = np.where(position_errors < threshold)[0]
    if len(below) == 0:
        return None
    return float(below[0]) / update_rate


def compute_all_metrics(
    estimated_positions: np.ndarray,
    true_positions: np.ndarray,
    estimated_velocity: Optional[np.ndarray] = None,
    true_velocity: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of all evaluation metrics.

    Args:
        estimated_positions: (N, 2+) estimated ENU positions.
        true_positions: (N, 2+) true ENU positions.
        estimated_velocity: Optional (N, 2) estimated velocity.
        true_velocity: Optional (N, 2) true velocity.

    Returns:
        Dict with all metric categories.
    """
    results = {
        "drift_percent": positional_drift_percent(estimated_positions, true_positions),
        "ate": absolute_trajectory_error(estimated_positions, true_positions),
        "cep": circular_error_probable(estimated_positions, true_positions),
    }

    act = along_cross_track_error(estimated_positions[:, :2], true_positions[:, :2])
    results["along_track_rmse"] = float(np.sqrt(np.mean(act["along_track"] ** 2)))
    results["cross_track_rmse"] = float(np.sqrt(np.mean(act["cross_track"] ** 2)))

    if estimated_velocity is not None and true_velocity is not None:
        results["velocity"] = velocity_error(estimated_velocity, true_velocity)

    return results
