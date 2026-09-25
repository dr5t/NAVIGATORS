import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from src.navigation.interfaces import (
    GNSSAnomalyStatus,
    GNSSAnomalyType,
    GNSSAnomalyReport,
    IGNSSAnomalyDetector,
    RoadHypothesis,
)

GNSS_POSITION_JUMP = "GNSS_POSITION_JUMP"
GNSS_IMPOSSIBLE_DISPLACEMENT = "GNSS_IMPOSSIBLE_DISPLACEMENT"
GNSS_SPEED_INCONSISTENCY = "GNSS_SPEED_INCONSISTENCY"
GNSS_VELOCITY_DISAGREEMENT = "GNSS_VELOCITY_DISAGREEMENT"
GNSS_HEADING_INCONSISTENCY = "GNSS_HEADING_INCONSISTENCY"
GNSS_NAVIGATION_RESIDUAL_HIGH = "GNSS_NAVIGATION_RESIDUAL_HIGH"
GNSS_MAP_INCONSISTENCY = "GNSS_MAP_INCONSISTENCY"


@dataclass
class GNSSAnomalyDetectorConfig:
    max_jump_displacement_m: float = 30.0
    max_jump_speed_mps: float = 40.0
    max_kinematic_accel_mps2: float = 12.0
    max_speed_mps: float = 65.0
    max_vel_discrepancy_mps: float = 8.0
    max_heading_divergence_rad: float = 0.8
    min_speed_for_heading_check_mps: float = 2.5
    max_normalized_innovation: float = 4.0
    max_raw_innovation_m: float = 35.0
    max_cross_track_m: float = 25.0
    max_map_heading_divergence_rad: float = 1.05
    min_consecutive_normal_fixes: int = 3
    stale_timeout_s: float = 3.0


class GNSSAnomalyDetector(IGNSSAnomalyDetector):
    def __init__(self, config: Optional[GNSSAnomalyDetectorConfig] = None):
        self.config = config or GNSSAnomalyDetectorConfig()
        self.last_gnss_pos: Optional[np.ndarray] = None
        self.last_gnss_vel: Optional[np.ndarray] = None
        self.last_gnss_time: Optional[float] = None
        self.last_speed: Optional[float] = None
        self.consecutive_normal_fixes: int = 0
        self.is_in_anomaly: bool = False
        self.last_report: Optional[GNSSAnomalyReport] = None

    def reset(self) -> None:
        self.last_gnss_pos = None
        self.last_gnss_vel = None
        self.last_gnss_time = None
        self.last_speed = None
        self.consecutive_normal_fixes = 0
        self.is_in_anomaly = False
        self.last_report = None

    def detect_anomalies(
        self,
        gnss_data: Optional[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> GNSSAnomalyReport:
        if gnss_data is None or not isinstance(gnss_data, dict):
            report = GNSSAnomalyReport(
                detected=False,
                anomaly_type=GNSSAnomalyType.OUTAGE,
                confidence=0.0,
                severity=1.0,
                status=GNSSAnomalyStatus.UNKNOWN,
                reason_codes=[],
                metrics={},
                details={"reason": "missing_gnss_data"},
            )
            self.last_report = report
            return report

        dt = float(kwargs.get("dt", 1.0))
        predicted_state = kwargs.get("predicted_state")
        innovation = kwargs.get("innovation")
        current_position = kwargs.get("current_position")
        current_velocity = kwargs.get("current_velocity")
        current_heading = kwargs.get("current_heading")
        position_uncertainty = kwargs.get("position_uncertainty")
        velocity_uncertainty = kwargs.get("velocity_uncertainty")
        inertial_velocity = kwargs.get("inertial_velocity")
        road_hypothesis = kwargs.get("road_hypothesis")

        if len(args) >= 1 and predicted_state is None:
            predicted_state = args[0]
        if len(args) >= 2 and innovation is None:
            innovation = args[1]

        if predicted_state is not None:
            pred_arr = np.asarray(predicted_state, dtype=np.float64)
            if current_position is None and len(pred_arr) >= 2:
                current_position = pred_arr[:2]
            if current_velocity is None and len(pred_arr) >= 5:
                current_velocity = pred_arr[3:5]
            if current_heading is None and len(pred_arr) >= 9:
                current_heading = float(pred_arr[8])

        gnss_pos = None
        if "position_enu" in gnss_data:
            gnss_pos = np.asarray(gnss_data["position_enu"], dtype=np.float64)[:2]
        elif "east" in gnss_data and "north" in gnss_data:
            gnss_pos = np.array([float(gnss_data["east"]), float(gnss_data["north"])], dtype=np.float64)

        if gnss_pos is None or not np.all(np.isfinite(gnss_pos)):
            report = GNSSAnomalyReport(
                detected=False,
                anomaly_type=GNSSAnomalyType.OUTAGE,
                confidence=0.0,
                severity=1.0,
                status=GNSSAnomalyStatus.UNKNOWN,
                reason_codes=[],
                metrics={},
                details={"reason": "non_finite_or_missing_position"},
            )
            self.last_report = report
            return report

        accuracy = float(gnss_data.get("accuracy", 5.0))
        if not np.isfinite(accuracy) or accuracy <= 0:
            accuracy = 5.0

        ts = gnss_data.get("timestamp")
        calc_dt = dt
        if ts is not None and np.isfinite(ts):
            ts = float(ts)
            if self.last_gnss_time is not None:
                diff_t = ts - self.last_gnss_time
                if diff_t > 1000.0:
                    diff_t /= 1000.0
                if diff_t > 0:
                    calc_dt = diff_t

        calc_dt = max(0.001, min(calc_dt, 10.0))

        speed_curr = None
        if "speed" in gnss_data and np.isfinite(gnss_data["speed"]):
            speed_curr = max(0.0, float(gnss_data["speed"]))

        gnss_vel = None
        if "velocity_enu" in gnss_data:
            gnss_vel = np.asarray(gnss_data["velocity_enu"], dtype=np.float64)[:2]
        elif speed_curr is not None and "heading" in gnss_data and np.isfinite(gnss_data["heading"]):
            h = float(gnss_data["heading"])
            gnss_vel = np.array([speed_curr * np.sin(h), speed_curr * np.cos(h)], dtype=np.float64)

        if speed_curr is None and gnss_vel is not None and np.all(np.isfinite(gnss_vel)):
            speed_curr = float(np.linalg.norm(gnss_vel))

        reason_codes: List[str] = []
        metrics: Dict[str, float] = {
            "accuracy_m": accuracy,
            "dt_s": calc_dt,
            "speed_mps": speed_curr if speed_curr is not None else 0.0,
        }

        pos_disp = 0.0
        jump_speed = 0.0
        if self.last_gnss_pos is not None:
            pos_disp = float(np.linalg.norm(gnss_pos - self.last_gnss_pos))
            jump_speed = pos_disp / calc_dt
            metrics["displacement_m"] = pos_disp
            metrics["jump_speed_mps"] = jump_speed

            if jump_speed > self.config.max_jump_speed_mps:
                reason_codes.append(GNSS_POSITION_JUMP)
            elif pos_disp > self.config.max_jump_displacement_m:
                expected_disp = (speed_curr or 10.0) * calc_dt + 15.0
                if pos_disp > 2.5 * expected_disp:
                    reason_codes.append(GNSS_POSITION_JUMP)

            prev_speed = self.last_speed if self.last_speed is not None else (speed_curr or 10.0)
            max_kinematic_disp = (
                prev_speed * calc_dt
                + 0.5 * self.config.max_kinematic_accel_mps2 * (calc_dt ** 2)
                + 3.0 * accuracy
                + 5.0
            )
            if pos_disp > max_kinematic_disp or (calc_dt <= 1.0 and pos_disp > 55.0):
                if GNSS_IMPOSSIBLE_DISPLACEMENT not in reason_codes:
                    reason_codes.append(GNSS_IMPOSSIBLE_DISPLACEMENT)

        if speed_curr is not None:
            if speed_curr > self.config.max_speed_mps:
                reason_codes.append(GNSS_SPEED_INCONSISTENCY)
            elif self.last_speed is not None and calc_dt > 0:
                implied_accel = abs(speed_curr - self.last_speed) / calc_dt
                metrics["implied_accel_mps2"] = implied_accel
                if implied_accel > self.config.max_kinematic_accel_mps2:
                    reason_codes.append(GNSS_SPEED_INCONSISTENCY)

        ref_vel = inertial_velocity if inertial_velocity is not None else current_velocity
        if gnss_vel is not None and ref_vel is not None:
            ref_v_arr = np.asarray(ref_vel, dtype=np.float64)[:2]
            if np.all(np.isfinite(ref_v_arr)):
                v_diff = float(np.linalg.norm(gnss_vel - ref_v_arr))
                metrics["vel_residual_mps"] = v_diff
                sigma_v = np.sqrt((velocity_uncertainty or 1.5) ** 2 + 1.0)
                if v_diff > max(self.config.max_vel_discrepancy_mps, 3.5 * sigma_v):
                    reason_codes.append(GNSS_VELOCITY_DISAGREEMENT)

        if speed_curr is not None and speed_curr >= self.config.min_speed_for_heading_check_mps:
            motion_h = None
            if current_heading is not None:
                motion_h = float(current_heading)
            elif ref_vel is not None:
                ref_v_arr = np.asarray(ref_vel, dtype=np.float64)[:2]
                if np.all(np.isfinite(ref_v_arr)) and np.linalg.norm(ref_v_arr) >= 2.0:
                    motion_h = float(np.arctan2(ref_v_arr[0], ref_v_arr[1]))

            gnss_h = None
            if "heading" in gnss_data and np.isfinite(gnss_data["heading"]):
                gnss_h = float(gnss_data["heading"])
            elif gnss_vel is not None and np.linalg.norm(gnss_vel) >= 2.0:
                gnss_h = float(np.arctan2(gnss_vel[0], gnss_vel[1]))

            if motion_h is not None and gnss_h is not None:
                h_diff = abs((gnss_h - motion_h + np.pi) % (2.0 * np.pi) - np.pi)
                metrics["heading_divergence_rad"] = h_diff
                if h_diff > self.config.max_heading_divergence_rad:
                    reason_codes.append(GNSS_HEADING_INCONSISTENCY)

        if innovation is not None:
            innov_arr = np.asarray(innovation, dtype=np.float64)[:2]
            if np.all(np.isfinite(innov_arr)):
                pos_resid = float(np.linalg.norm(innov_arr))
                metrics["position_residual_m"] = pos_resid
                p_unc = float(position_uncertainty if position_uncertainty is not None else 3.0)
                sigma_tot = np.sqrt(p_unc ** 2 + accuracy ** 2)
                norm_innov = pos_resid / max(sigma_tot, 1e-3)
                metrics["normalized_innovation"] = norm_innov
                if norm_innov > self.config.max_normalized_innovation or (pos_resid > self.config.max_raw_innovation_m and p_unc < 10.0):
                    reason_codes.append(GNSS_NAVIGATION_RESIDUAL_HIGH)
        elif current_position is not None:
            cur_p_arr = np.asarray(current_position, dtype=np.float64)[:2]
            if np.all(np.isfinite(cur_p_arr)):
                pos_resid = float(np.linalg.norm(gnss_pos - cur_p_arr))
                metrics["position_residual_m"] = pos_resid
                p_unc = float(position_uncertainty if position_uncertainty is not None else 3.0)
                sigma_tot = np.sqrt(p_unc ** 2 + accuracy ** 2)
                norm_innov = pos_resid / max(sigma_tot, 1e-3)
                metrics["normalized_innovation"] = norm_innov
                if norm_innov > self.config.max_normalized_innovation or (pos_resid > self.config.max_raw_innovation_m and p_unc < 10.0):
                    reason_codes.append(GNSS_NAVIGATION_RESIDUAL_HIGH)

        if road_hypothesis is not None:
            cross_track = None
            prob = 1.0
            h_diff_map = 0.0
            if hasattr(road_hypothesis, "cross_track_distance_m"):
                cross_track = abs(float(road_hypothesis.cross_track_distance_m))
                prob = getattr(road_hypothesis, "probability", 1.0)
                h_diff_map = getattr(road_hypothesis, "heading_difference_rad", 0.0)
            elif isinstance(road_hypothesis, dict):
                cross_track = abs(float(road_hypothesis.get("cross_track_distance_m", road_hypothesis.get("cross_track", 0.0))))
                prob = float(road_hypothesis.get("probability", 1.0))
                h_diff_map = float(road_hypothesis.get("heading_difference_rad", road_hypothesis.get("heading_diff", 0.0)))
            elif isinstance(road_hypothesis, (int, float)):
                cross_track = abs(float(road_hypothesis))

            if cross_track is not None:
                metrics["cross_track_m"] = cross_track
                if prob >= 0.5:
                    if cross_track > (self.config.max_cross_track_m + accuracy):
                        reason_codes.append(GNSS_MAP_INCONSISTENCY)
                    elif speed_curr is not None and speed_curr > 5.0 and abs(h_diff_map) > self.config.max_map_heading_divergence_rad:
                        reason_codes.append(GNSS_MAP_INCONSISTENCY)

        is_anomalous = len(reason_codes) > 0
        if is_anomalous:
            self.is_in_anomaly = True
            self.consecutive_normal_fixes = 0
            has_step = (GNSS_POSITION_JUMP in reason_codes or GNSS_IMPOSSIBLE_DISPLACEMENT in reason_codes)
            has_speed = GNSS_SPEED_INCONSISTENCY in reason_codes
            has_resid = GNSS_NAVIGATION_RESIDUAL_HIGH in reason_codes

            if has_step:
                anomaly_type = GNSSAnomalyType.STEP_DISCONTINUITY
                sev = 0.9
            elif has_speed:
                anomaly_type = GNSSAnomalyType.STEP_DISCONTINUITY
                sev = 0.8
            elif has_resid:
                anomaly_type = GNSSAnomalyType.MULTIPATH
                sev = 0.6
            else:
                anomaly_type = GNSSAnomalyType.MULTIPATH
                sev = 0.5

            sev = min(1.0, sev + 0.1 * (len(reason_codes) - 1))
            conf = min(0.99, 0.70 + 0.1 * len(reason_codes))
            status = GNSSAnomalyStatus.ANOMALOUS
        elif self.last_gnss_pos is None and current_position is None and current_velocity is None and inertial_velocity is None and current_heading is None:
            status = GNSSAnomalyStatus.UNKNOWN
            anomaly_type = GNSSAnomalyType.NONE
            sev = 0.0
            conf = 0.5
        else:
            self.consecutive_normal_fixes += 1
            if self.consecutive_normal_fixes >= self.config.min_consecutive_normal_fixes:
                self.is_in_anomaly = False
            status = GNSSAnomalyStatus.NORMAL
            anomaly_type = GNSSAnomalyType.NONE
            sev = 0.0
            conf = 0.95

        self._update_tracking(gnss_pos, gnss_vel, speed_curr, ts)

        report = GNSSAnomalyReport(
            detected=is_anomalous,
            anomaly_type=anomaly_type,
            confidence=float(conf),
            severity=float(sev),
            status=status,
            reason_codes=reason_codes,
            metrics=metrics,
            details={
                "reason_codes": reason_codes,
                "is_in_anomaly": self.is_in_anomaly,
                "consecutive_normal_fixes": self.consecutive_normal_fixes,
            },
        )
        self.last_report = report
        return report

    def _update_tracking(
        self,
        gnss_pos: np.ndarray,
        gnss_vel: Optional[np.ndarray],
        speed: Optional[float],
        ts: Optional[float],
    ) -> None:
        self.last_gnss_pos = gnss_pos.copy()
        if gnss_vel is not None:
            self.last_gnss_vel = gnss_vel.copy()
        self.last_speed = speed
        if ts is not None:
            self.last_gnss_time = ts
