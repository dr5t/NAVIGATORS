import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

from src.navigation.interfaces import (
    GNSSRecoveryState,
    GNSSRecoveryPlan,
    IGNSSRecoveryManager,
    GNSSTrustState,
    GNSSTrustMetric,
    IGNSSTrustEngine,
)
from src.navigation.map_matching import RoadNetwork, RoadSegment


@dataclass
class GNSSRecoveryConfig:
    max_step_correction_m: float = 2.0
    convergence_residual_m: float = 2.0
    convergence_step_norm_m: float = 0.35
    min_valid_fixes_to_recover: int = 3
    consecutive_converged_steps_required: int = 3
    max_jump_speed_mps: float = 40.0
    max_implied_accel_mps2: float = 20.0
    max_norm_innovation: float = 4.0
    min_trust_score: float = 0.50
    max_heading_diff_rad: float = 0.85
    max_parallel_road_divergence_m: float = 6.0


@dataclass
class GNSSRecoveryMetrics:
    recovery_times_s: List[float] = field(default_factory=list)
    max_position_discontinuities_m: List[float] = field(default_factory=list)
    final_position_errors_m: List[float] = field(default_factory=list)
    false_recovery_count: int = 0
    total_recovery_episodes: int = 0

    def add_episode(
        self,
        recovery_time_s: float,
        max_discontinuity_m: float,
        final_position_error_m: float,
        is_false_recovery: bool = False,
    ):
        self.total_recovery_episodes += 1
        self.recovery_times_s.append(float(recovery_time_s))
        self.max_position_discontinuities_m.append(float(max_discontinuity_m))
        self.final_position_errors_m.append(float(final_position_error_m))
        if is_false_recovery:
            self.false_recovery_count += 1

    def compute_summary(self) -> Dict[str, float]:
        if self.total_recovery_episodes == 0:
            return {
                "total_episodes": 0.0,
                "mean_recovery_time_s": 0.0,
                "max_recovery_time_s": 0.0,
                "max_position_discontinuity_m": 0.0,
                "mean_final_position_error_m": 0.0,
                "max_final_position_error_m": 0.0,
                "false_recovery_count": 0.0,
                "false_recovery_rate_pct": 0.0,
            }

        rec_times = np.array(self.recovery_times_s, dtype=np.float64)
        disconts = np.array(self.max_position_discontinuities_m, dtype=np.float64)
        final_errs = np.array(self.final_position_errors_m, dtype=np.float64)

        false_rate = (self.false_recovery_count / self.total_recovery_episodes) * 100.0

        return {
            "total_episodes": float(self.total_recovery_episodes),
            "mean_recovery_time_s": float(round(float(np.mean(rec_times)), 2)),
            "max_recovery_time_s": float(round(float(np.max(rec_times)), 2)),
            "max_position_discontinuity_m": float(round(float(np.max(disconts)), 3)),
            "mean_final_position_error_m": float(round(float(np.mean(final_errs)), 3)),
            "max_final_position_error_m": float(round(float(np.max(final_errs)), 3)),
            "false_recovery_count": float(self.false_recovery_count),
            "false_recovery_rate_pct": float(round(false_rate, 2)),
        }


class ValidatedGNSSRecoveryManager(IGNSSRecoveryManager):
    def __init__(
        self,
        config: Optional[GNSSRecoveryConfig] = None,
        trust_engine: Optional[IGNSSTrustEngine] = None,
        road_network: Optional[RoadNetwork] = None,
    ):
        self.config = config or GNSSRecoveryConfig()
        self.trust_engine = trust_engine
        self.road_network = road_network
        self.state: GNSSRecoveryState = GNSSRecoveryState.REACQUIRING
        self.valid_consecutive_fixes: int = 0
        self.consecutive_converged: int = 0
        self.recovery_step_index: int = 0
        self.first_gnss_time: Optional[float] = None
        self.recovery_completed_time: Optional[float] = None
        self.max_observed_discontinuity: float = 0.0
        self.last_valid_gnss_pos: Optional[np.ndarray] = None
        self.last_gnss_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.state = GNSSRecoveryState.REACQUIRING
        self.valid_consecutive_fixes = 0
        self.consecutive_converged = 0
        self.recovery_step_index = 0
        self.first_gnss_time = None
        self.recovery_completed_time = None
        self.max_observed_discontinuity = 0.0
        self.last_valid_gnss_pos = None
        self.last_gnss_timestamp = None
        if self.trust_engine is not None:
            self.trust_engine.reset()

    def is_converged(self) -> bool:
        return self.state == GNSSRecoveryState.NORMAL

    def compute_recovery_step(
        self,
        current_estimate: np.ndarray,
        fresh_gnss: np.ndarray,
        recovery_step_index: int,
        *args,
        **kwargs,
    ) -> GNSSRecoveryPlan:
        cur_est = np.asarray(current_estimate, dtype=np.float64)
        gnss_pos = np.asarray(fresh_gnss[:2], dtype=np.float64)
        diff = gnss_pos - cur_est[:2]
        dist = float(np.linalg.norm(diff))
        step = min(self.config.max_step_correction_m, dist)

        corrected = cur_est.copy()
        if dist > 1e-9:
            corrected[:2] += diff * (step / dist)

        discont = step
        if discont > self.max_observed_discontinuity:
            self.max_observed_discontinuity = discont

        is_conv = (dist <= self.config.convergence_residual_m and step <= self.config.convergence_step_norm_m)
        if is_conv:
            self.consecutive_converged += 1
            if self.consecutive_converged >= self.config.consecutive_converged_steps_required:
                self.state = GNSSRecoveryState.NORMAL
        else:
            self.consecutive_converged = 0

        cov_diag = np.ones(15 if len(cur_est) >= 15 else len(cur_est), dtype=np.float64)
        cov_scale = max(1.0, dist / 2.0)
        cov_diag[0:2] = cov_scale

        return GNSSRecoveryPlan(
            corrected_state=corrected,
            smoothed_covariance_diagonal=cov_diag,
            step_correction_norm_m=step,
            is_converged=self.is_converged(),
            remaining_steps=max(0, 5 - recovery_step_index),
            state=self.state,
            discontinuity_m=discont,
            trust_score=1.0,
            consistency_passed=True,
        )

    def process_gnss_return(
        self,
        gnss_data: Optional[Dict[str, Any]],
        current_estimate: np.ndarray,
        dt: float = 0.1,
        dr_position: Optional[np.ndarray] = None,
        dr_velocity: Optional[np.ndarray] = None,
        dr_heading: Optional[float] = None,
        dr_uncertainty: Optional[float] = None,
        ai_velocity: Optional[np.ndarray] = None,
        road_segment: Optional[RoadSegment] = None,
        timestamp: Optional[float] = None,
    ) -> GNSSRecoveryPlan:
        cur_est = np.asarray(current_estimate, dtype=np.float64)
        ts = float(timestamp if timestamp is not None else (self.last_gnss_timestamp + dt if self.last_gnss_timestamp is not None else 0.0))

        if not gnss_data or not isinstance(gnss_data, dict):
            return GNSSRecoveryPlan(
                corrected_state=cur_est.copy(),
                smoothed_covariance_diagonal=np.ones(len(cur_est)),
                step_correction_norm_m=0.0,
                is_converged=False,
                remaining_steps=10,
                state=self.state,
                discontinuity_m=0.0,
                trust_score=0.0,
                consistency_passed=False,
                diagnostics={"reason": "NO_GNSS_DATA"},
            )

        gnss_pos = None
        if "position_enu" in gnss_data:
            gnss_pos = np.asarray(gnss_data["position_enu"], dtype=np.float64)[:2]
        elif "position" in gnss_data:
            gnss_pos = np.asarray(gnss_data["position"], dtype=np.float64)[:2]
        elif "east" in gnss_data and "north" in gnss_data:
            gnss_pos = np.array([float(gnss_data["east"]), float(gnss_data["north"])], dtype=np.float64)

        if gnss_pos is None or not np.all(np.isfinite(gnss_pos)):
            return GNSSRecoveryPlan(
                corrected_state=cur_est.copy(),
                smoothed_covariance_diagonal=np.ones(len(cur_est)),
                step_correction_norm_m=0.0,
                is_converged=False,
                remaining_steps=10,
                state=self.state,
                discontinuity_m=0.0,
                trust_score=0.0,
                consistency_passed=False,
                diagnostics={"reason": "INVALID_COORDINATES"},
            )

        if self.first_gnss_time is None:
            self.first_gnss_time = ts

        acc_m = float(gnss_data.get("accuracy", 3.0))
        is_measurement_valid = True
        rejection_reasons = []

        if acc_m > 40.0:
            is_measurement_valid = False
            rejection_reasons.append("HIGH_ACCURACY_VARIANCE")

        if self.last_valid_gnss_pos is not None:
            dt_step = max(0.01, ts - (self.last_gnss_timestamp or (ts - dt)))
            jump_speed = float(np.linalg.norm(gnss_pos - self.last_valid_gnss_pos)) / dt_step
            if jump_speed > self.config.max_jump_speed_mps:
                is_measurement_valid = False
                rejection_reasons.append(f"POSITION_JUMP_SPEED_{jump_speed:.1f}")

        trust_score = 1.0
        if self.trust_engine is not None:
            t_metric = self.trust_engine.evaluate_trust(
                gnss_data=gnss_data,
                dt=dt,
                current_position=cur_est[:2],
                current_velocity=cur_est[3:5] if len(cur_est) >= 5 else None,
                current_heading=float(cur_est[8]) if len(cur_est) >= 9 else None,
                position_uncertainty=dr_uncertainty,
                ai_velocity=ai_velocity,
            )
            trust_score = float(t_metric.trust_score)
            if trust_score < self.config.min_trust_score or t_metric.state == GNSSTrustState.UNUSABLE:
                is_measurement_valid = False
                rejection_reasons.append(f"UNTRUSTED_GNSS_SCORE_{trust_score:.2f}")

        dr_pos = np.asarray(dr_position[:2], dtype=np.float64) if dr_position is not None else cur_est[:2]
        pos_residual = float(np.linalg.norm(gnss_pos - dr_pos))
        sigma_dr = float(dr_uncertainty if dr_uncertainty is not None else 3.0)
        sigma_tot = float(np.sqrt(sigma_dr ** 2 + acc_m ** 2))
        norm_innov = pos_residual / max(1.0, sigma_tot)

        if norm_innov > self.config.max_norm_innovation and pos_residual > 35.0:
            is_measurement_valid = False
            rejection_reasons.append(f"HIGH_NORM_INNOV_{norm_innov:.1f}")

        if road_segment is not None and self.road_network is not None:
            _, dist_to_active = self.road_network.nearest_point_on_segment(gnss_pos, road_segment)
            tangent = road_segment.direction[:2]
            normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
            lat_dev = abs(float(np.dot(gnss_pos - dr_pos, normal)))
            if lat_dev > self.config.max_parallel_road_divergence_m and dist_to_active > 8.0:
                is_measurement_valid = False
                rejection_reasons.append("PARALLEL_ROAD_MISMATCH")

        self.last_gnss_timestamp = ts

        if not is_measurement_valid:
            self.valid_consecutive_fixes = 0
            self.consecutive_converged = 0
            if self.state != GNSSRecoveryState.NORMAL:
                self.state = GNSSRecoveryState.REACQUIRING
            return GNSSRecoveryPlan(
                corrected_state=cur_est.copy(),
                smoothed_covariance_diagonal=np.ones(len(cur_est)) * 5.0,
                step_correction_norm_m=0.0,
                is_converged=False,
                remaining_steps=10,
                state=self.state,
                discontinuity_m=0.0,
                trust_score=trust_score,
                consistency_passed=False,
                diagnostics={"reasons": rejection_reasons, "residual_m": pos_residual},
            )

        self.last_valid_gnss_pos = gnss_pos.copy()
        self.valid_consecutive_fixes += 1

        if self.state == GNSSRecoveryState.REACQUIRING:
            self.state = GNSSRecoveryState.VALIDATING

        if self.state == GNSSRecoveryState.VALIDATING:
            if self.valid_consecutive_fixes >= self.config.min_valid_fixes_to_recover:
                self.state = GNSSRecoveryState.RECOVERING
                self.recovery_step_index = 0
            else:
                return GNSSRecoveryPlan(
                    corrected_state=cur_est.copy(),
                    smoothed_covariance_diagonal=np.ones(len(cur_est)) * 3.0,
                    step_correction_norm_m=0.0,
                    is_converged=False,
                    remaining_steps=max(1, self.config.min_valid_fixes_to_recover - self.valid_consecutive_fixes),
                    state=self.state,
                    discontinuity_m=0.0,
                    trust_score=trust_score,
                    consistency_passed=True,
                    diagnostics={"validating_fix": self.valid_consecutive_fixes},
                )

        self.recovery_step_index += 1
        diff = gnss_pos - cur_est[:2]
        dist = float(np.linalg.norm(diff))

        step_limit = self.config.max_step_correction_m
        step = min(step_limit, dist)

        corrected = cur_est.copy()
        if dist > 1e-9:
            corrected[:2] += diff * (step / dist)

        discont = step
        if discont > self.max_observed_discontinuity:
            self.max_observed_discontinuity = discont

        is_step_converged = (dist <= self.config.convergence_residual_m and step <= self.config.convergence_step_norm_m)
        if is_step_converged:
            self.consecutive_converged += 1
            if self.consecutive_converged >= self.config.consecutive_converged_steps_required:
                self.state = GNSSRecoveryState.NORMAL
                if self.recovery_completed_time is None:
                    self.recovery_completed_time = ts
        else:
            self.consecutive_converged = 0

        cov_diag = np.ones(len(cur_est), dtype=np.float64)
        cov_scale = max(1.0, dist / 2.0)
        cov_diag[0:2] = cov_scale

        return GNSSRecoveryPlan(
            corrected_state=corrected,
            smoothed_covariance_diagonal=cov_diag,
            step_correction_norm_m=step,
            is_converged=self.is_converged(),
            remaining_steps=0 if self.is_converged() else max(1, 8 - self.recovery_step_index),
            state=self.state,
            discontinuity_m=discont,
            trust_score=trust_score,
            consistency_passed=True,
            diagnostics={
                "residual_m": dist,
                "consecutive_valid": self.valid_consecutive_fixes,
                "step_index": self.recovery_step_index,
            },
        )
