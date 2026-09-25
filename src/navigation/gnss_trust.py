import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from navigation.interfaces import (
    GNSSTrustState,
    GNSSTrustMetric,
    IGNSSTrustEngine,
)
from navigation.gnss_anomaly import (
    GNSSAnomalyDetector,
    IGNSSAnomalyDetector,
    GNSSAnomalyStatus,
    GNSSAnomalyReport,
    GNSS_POSITION_JUMP,
    GNSS_IMPOSSIBLE_DISPLACEMENT,
    GNSS_SPEED_INCONSISTENCY,
    GNSS_VELOCITY_DISAGREEMENT,
    GNSS_HEADING_INCONSISTENCY,
    GNSS_NAVIGATION_RESIDUAL_HIGH,
    GNSS_MAP_INCONSISTENCY,
)


@dataclass
class GNSSTrustConfig:
    max_trusted_accuracy_m: float = 10.0
    max_degraded_accuracy_m: float = 25.0
    max_suspicious_accuracy_m: float = 50.0
    max_unusable_accuracy_m: float = 80.0
    max_jump_speed_trusted_mps: float = 30.0
    max_jump_speed_suspicious_mps: float = 50.0
    max_jump_speed_unusable_mps: float = 80.0
    max_physical_accel_mps2: float = 12.0
    max_unusable_accel_mps2: float = 20.0
    max_speed_unusable_mps: float = 65.0
    max_heading_diff_suspicious_rad: float = 1.2
    max_heading_diff_unusable_rad: float = 2.0
    min_speed_for_heading_check_mps: float = 2.5
    max_norm_innov_trusted: float = 2.5
    max_norm_innov_degraded: float = 4.0
    max_norm_innov_suspicious: float = 7.0
    max_vel_discrepancy_mps: float = 10.0
    stale_timeout_s: float = 3.0
    min_recovery_fixes: int = 3
    trusted_variance_scale: float = 1.0
    degraded_variance_scale: float = 5.0
    suspicious_variance_scale: float = 50.0
    unusable_variance_scale: float = 10000.0


@dataclass
class GNSSQualityFeatures:
    accuracy_m: float = 999.0
    dt_s: float = 0.0
    jump_speed_mps: float = 0.0
    implied_accel_mps2: float = 0.0
    vel_residual_mps: float = 0.0
    pos_innovation_m: float = 0.0
    normalized_innovation: float = 0.0
    heading_divergence_rad: float = 0.0
    map_distance_m: Optional[float] = None
    is_timestamp_valid: bool = False
    speed_mps: float = 0.0


class GNSSTrustEngine(IGNSSTrustEngine):
    def __init__(
        self,
        config: Optional[GNSSTrustConfig] = None,
        anomaly_detector: Optional[IGNSSAnomalyDetector] = None,
    ):
        self.config = config or GNSSTrustConfig()
        self.anomaly_detector = anomaly_detector or GNSSAnomalyDetector()
        self.last_gnss_pos: Optional[np.ndarray] = None
        self.last_gnss_vel: Optional[np.ndarray] = None
        self.last_gnss_timestamp: Optional[float] = None
        self.consecutive_valid_fixes: int = 0
        self.last_metric: Optional[GNSSTrustMetric] = None
        self.in_recovery: bool = False

    def reset(self) -> None:
        self.last_gnss_pos = None
        self.last_gnss_vel = None
        self.last_gnss_timestamp = None
        self.consecutive_valid_fixes = 0
        self.last_metric = None
        self.in_recovery = False
        if self.anomaly_detector is not None:
            self.anomaly_detector.reset()

    def extract_features(
        self,
        gnss_data: Optional[Dict[str, Any]],
        dt: float,
        current_position: Optional[np.ndarray],
        current_velocity: Optional[np.ndarray],
        current_heading: Optional[float],
        position_uncertainty: Optional[float],
        ai_velocity: Optional[np.ndarray],
        road_distance: Optional[float],
    ) -> GNSSQualityFeatures:
        features = GNSSQualityFeatures()
        if gnss_data is None or not isinstance(gnss_data, dict):
            return features

        accuracy = float(gnss_data.get("accuracy", 999.0))
        features.accuracy_m = accuracy if np.isfinite(accuracy) else 999.0

        ts = gnss_data.get("timestamp")
        if ts is not None and np.isfinite(ts):
            if self.last_gnss_timestamp is not None:
                calc_dt = ts - self.last_gnss_timestamp
                if calc_dt > 1000.0:
                    calc_dt /= 1000.0
                features.dt_s = calc_dt
                features.is_timestamp_valid = (0.0 < calc_dt <= self.config.stale_timeout_s)
            else:
                features.dt_s = dt
                features.is_timestamp_valid = (0.0 < dt <= self.config.stale_timeout_s)
        else:
            features.dt_s = dt
            features.is_timestamp_valid = (0.0 < dt <= self.config.stale_timeout_s)

        gnss_pos = None
        if "position_enu" in gnss_data:
            gnss_pos = np.asarray(gnss_data["position_enu"], dtype=np.float64)[:2]
        elif "lat" in gnss_data and "lon" in gnss_data and "east" in gnss_data and "north" in gnss_data:
            gnss_pos = np.array([float(gnss_data["east"]), float(gnss_data["north"])], dtype=np.float64)

        if gnss_pos is not None and np.all(np.isfinite(gnss_pos)):
            if self.last_gnss_pos is not None and features.dt_s > 0:
                pos_disp = float(np.linalg.norm(gnss_pos - self.last_gnss_pos))
                features.jump_speed_mps = pos_disp / features.dt_s

            if current_position is not None:
                cur_pos_2d = np.asarray(current_position, dtype=np.float64)[:2]
                if np.all(np.isfinite(cur_pos_2d)):
                    innov = float(np.linalg.norm(gnss_pos - cur_pos_2d))
                    features.pos_innovation_m = innov
                    sigma_tot = np.sqrt((position_uncertainty or 3.0) ** 2 + features.accuracy_m ** 2)
                    features.normalized_innovation = innov / max(sigma_tot, 1e-3)

        speed = gnss_data.get("speed")
        if speed is not None and np.isfinite(speed):
            features.speed_mps = max(0.0, float(speed))

        gnss_vel = None
        if "velocity_enu" in gnss_data:
            gnss_vel = np.asarray(gnss_data["velocity_enu"], dtype=np.float64)[:2]
        elif "speed" in gnss_data and "heading" in gnss_data:
            h = float(gnss_data["heading"])
            s = float(gnss_data["speed"])
            gnss_vel = np.array([s * np.sin(h), s * np.cos(h)], dtype=np.float64)

        if gnss_vel is not None and np.all(np.isfinite(gnss_vel)):
            if self.last_gnss_vel is not None and features.dt_s > 0:
                vel_disp = float(np.linalg.norm(gnss_vel - self.last_gnss_vel))
                features.implied_accel_mps2 = vel_disp / features.dt_s

            if current_velocity is not None:
                cur_v = np.asarray(current_velocity, dtype=np.float64)[:2]
                if np.all(np.isfinite(cur_v)):
                    features.vel_residual_mps = float(np.linalg.norm(gnss_vel - cur_v))
            elif ai_velocity is not None:
                ai_v = np.asarray(ai_velocity, dtype=np.float64)[:2]
                if np.all(np.isfinite(ai_v)):
                    features.vel_residual_mps = float(np.linalg.norm(gnss_vel - ai_v))

            if features.speed_mps >= self.config.min_speed_for_heading_check_mps and current_heading is not None:
                cog = float(np.arctan2(gnss_vel[0], gnss_vel[1]))
                diff = abs((cog - current_heading + np.pi) % (2.0 * np.pi) - np.pi)
                features.heading_divergence_rad = diff

        features.map_distance_m = road_distance
        return features

    def compute_reliability_score(
        self,
        features: GNSSQualityFeatures,
        state: GNSSTrustState,
    ) -> float:
        if state == GNSSTrustState.UNUSABLE:
            return 0.0

        cfg = self.config
        s_acc = np.exp(-0.5 * min(features.accuracy_m / cfg.max_trusted_accuracy_m, 5.0) ** 2)
        s_innov = np.exp(-0.5 * min(features.normalized_innovation / cfg.max_norm_innov_trusted, 5.0) ** 2)

        vel_res = min(features.vel_residual_mps, 20.0)
        s_vel = np.exp(-0.5 * (vel_res / 5.0) ** 2)

        if features.speed_mps >= cfg.min_speed_for_heading_check_mps:
            s_head = np.exp(-0.5 * (features.heading_divergence_rad / 0.5) ** 2)
        else:
            s_head = 1.0

        jump = min(features.jump_speed_mps, cfg.max_jump_speed_suspicious_mps)
        s_jump = np.exp(-0.5 * (jump / 15.0) ** 2)

        s_map = 1.0
        if features.map_distance_m is not None and features.map_distance_m > 30.0:
            s_map = np.exp(-0.5 * ((features.map_distance_m - 30.0) / 20.0) ** 2)

        raw_score = float(s_acc * 0.35 + s_innov * 0.25 + s_vel * 0.15 + s_head * 0.10 + s_jump * 0.15)
        raw_score *= s_map

        if state == GNSSTrustState.SUSPICIOUS:
            return float(np.clip(raw_score * 0.35, 0.05, 0.35))
        elif state == GNSSTrustState.DEGRADED:
            return float(np.clip(raw_score, 0.36, 0.74))
        else:
            return float(np.clip(raw_score, 0.75, 1.0))

    def evaluate_trust(
        self,
        gnss_data: Optional[Dict[str, Any]],
        imu_accel: Optional[np.ndarray] = None,
        dt: float = 0.1,
        current_position: Optional[np.ndarray] = None,
        current_velocity: Optional[np.ndarray] = None,
        current_heading: Optional[float] = None,
        position_uncertainty: Optional[float] = None,
        ai_velocity: Optional[np.ndarray] = None,
        road_distance: Optional[float] = None,
        *args,
        **kwargs,
    ) -> GNSSTrustMetric:
        cfg = self.config

        anomaly_rep = kwargs.get("anomaly_report")
        road_hyp = kwargs.get("road_hypothesis") or road_distance
        if anomaly_rep is None and self.anomaly_detector is not None:
            anomaly_rep = self.anomaly_detector.detect_anomalies(
                gnss_data=gnss_data,
                dt=dt,
                current_position=current_position,
                current_velocity=current_velocity,
                current_heading=current_heading,
                position_uncertainty=position_uncertainty,
                inertial_velocity=ai_velocity,
                road_hypothesis=road_hyp,
            )

        if gnss_data is None or not isinstance(gnss_data, dict) or len(gnss_data) == 0:
            self.consecutive_valid_fixes = 0
            self.in_recovery = True
            metric = GNSSTrustMetric(
                trust_score=0.0,
                is_trusted=False,
                position_variance_scale=cfg.unusable_variance_scale,
                velocity_variance_scale=cfg.unusable_variance_scale,
                state=GNSSTrustState.UNUSABLE,
                diagnostics={
                    "reason": "gnss_data_none_or_empty",
                    "anomaly_report": anomaly_rep,
                    "anomaly_reasons": anomaly_rep.reason_codes if anomaly_rep is not None else [],
                },
            )
            self.last_metric = metric
            return metric

        features = self.extract_features(
            gnss_data=gnss_data,
            dt=dt,
            current_position=current_position,
            current_velocity=current_velocity,
            current_heading=current_heading,
            position_uncertainty=position_uncertainty,
            ai_velocity=ai_velocity,
            road_distance=road_distance,
        )

        reasons = []

        if not features.is_timestamp_valid:
            reasons.append("invalid_or_stale_timestamp")
        if features.accuracy_m > cfg.max_unusable_accuracy_m:
            reasons.append("accuracy_exceeds_unusable_limit")
        if features.jump_speed_mps > cfg.max_jump_speed_unusable_mps:
            reasons.append("impossible_position_jump_speed")
        if features.speed_mps > cfg.max_speed_unusable_mps:
            reasons.append("impossible_ground_speed")
        if features.implied_accel_mps2 > cfg.max_unusable_accel_mps2:
            reasons.append("impossible_acceleration")
        if features.normalized_innovation > 8.0:
            reasons.append("extreme_normalized_innovation")

        if anomaly_rep is not None and anomaly_rep.status == GNSSAnomalyStatus.ANOMALOUS:
            for rc in anomaly_rep.reason_codes:
                if rc in (GNSS_POSITION_JUMP, GNSS_IMPOSSIBLE_DISPLACEMENT):
                    if rc not in reasons:
                        reasons.append(rc)
                elif rc == GNSS_SPEED_INCONSISTENCY and features.speed_mps > cfg.max_speed_unusable_mps:
                    if rc not in reasons:
                        reasons.append(rc)

        if len(reasons) > 0:
            self.consecutive_valid_fixes = 0
            self.in_recovery = True
            metric = GNSSTrustMetric(
                trust_score=0.0,
                is_trusted=False,
                position_variance_scale=cfg.unusable_variance_scale,
                velocity_variance_scale=cfg.unusable_variance_scale,
                state=GNSSTrustState.UNUSABLE,
                diagnostics={
                    "reasons": reasons,
                    "accuracy_m": features.accuracy_m,
                    "jump_speed_mps": features.jump_speed_mps,
                    "speed_mps": features.speed_mps,
                    "implied_accel_mps2": features.implied_accel_mps2,
                    "normalized_innovation": features.normalized_innovation,
                    "anomaly_report": anomaly_rep,
                    "anomaly_reasons": anomaly_rep.reason_codes if anomaly_rep is not None else [],
                },
            )
            self._update_tracking(gnss_data, features)
            self.last_metric = metric
            return metric

        suspicious_reasons = []
        if features.jump_speed_mps > cfg.max_jump_speed_suspicious_mps:
            suspicious_reasons.append("high_position_jump_speed")
        if features.normalized_innovation > cfg.max_norm_innov_degraded:
            suspicious_reasons.append("high_normalized_innovation")
        if features.accuracy_m > cfg.max_suspicious_accuracy_m:
            suspicious_reasons.append("poor_accuracy_suspicious")
        if (features.speed_mps >= cfg.min_speed_for_heading_check_mps and
                features.heading_divergence_rad > cfg.max_heading_diff_suspicious_rad):
            suspicious_reasons.append("heading_inconsistency")
        if features.vel_residual_mps > cfg.max_vel_discrepancy_mps:
            suspicious_reasons.append("velocity_residual_excessive")
        if features.implied_accel_mps2 > cfg.max_physical_accel_mps2:
            suspicious_reasons.append("high_implied_acceleration")

        if anomaly_rep is not None and anomaly_rep.status == GNSSAnomalyStatus.ANOMALOUS:
            for rc in anomaly_rep.reason_codes:
                if rc not in reasons and rc not in suspicious_reasons:
                    suspicious_reasons.append(rc)

        if len(suspicious_reasons) > 0:
            self.consecutive_valid_fixes = max(0, self.consecutive_valid_fixes - 1)
            score = self.compute_reliability_score(features, GNSSTrustState.SUSPICIOUS)
            metric = GNSSTrustMetric(
                trust_score=score,
                is_trusted=False,
                position_variance_scale=cfg.suspicious_variance_scale,
                velocity_variance_scale=cfg.suspicious_variance_scale,
                state=GNSSTrustState.SUSPICIOUS,
                diagnostics={
                    "reasons": suspicious_reasons,
                    "accuracy_m": features.accuracy_m,
                    "jump_speed_mps": features.jump_speed_mps,
                    "heading_divergence_rad": features.heading_divergence_rad,
                    "normalized_innovation": features.normalized_innovation,
                    "anomaly_report": anomaly_rep,
                    "anomaly_reasons": anomaly_rep.reason_codes if anomaly_rep is not None else [],
                },
            )
            self._update_tracking(gnss_data, features)
            self.last_metric = metric
            return metric

        degraded_reasons = []
        if features.accuracy_m > cfg.max_trusted_accuracy_m:
            degraded_reasons.append("degraded_accuracy")
        if features.normalized_innovation > cfg.max_norm_innov_trusted:
            degraded_reasons.append("moderate_normalized_innovation")
        if features.jump_speed_mps > cfg.max_jump_speed_trusted_mps:
            degraded_reasons.append("moderate_position_shift")
        if features.vel_residual_mps > 4.0:
            degraded_reasons.append("moderate_velocity_residual")

        tentative_valid = self.consecutive_valid_fixes + 1
        if self.in_recovery and tentative_valid < cfg.min_recovery_fixes:
            degraded_reasons.append("recovering_from_outage")

        if len(degraded_reasons) > 0:
            self.consecutive_valid_fixes += 1
            if self.consecutive_valid_fixes >= cfg.min_recovery_fixes:
                self.in_recovery = False

            score = self.compute_reliability_score(features, GNSSTrustState.DEGRADED)
            v_scale = cfg.degraded_variance_scale * max(1.0, features.accuracy_m / cfg.max_trusted_accuracy_m)
            metric = GNSSTrustMetric(
                trust_score=score,
                is_trusted=True,
                position_variance_scale=float(v_scale),
                velocity_variance_scale=float(v_scale * 0.5),
                state=GNSSTrustState.DEGRADED,
                diagnostics={
                    "reasons": degraded_reasons,
                    "accuracy_m": features.accuracy_m,
                    "consecutive_valid_fixes": self.consecutive_valid_fixes,
                    "anomaly_report": anomaly_rep,
                    "anomaly_reasons": anomaly_rep.reason_codes if anomaly_rep is not None else [],
                },
            )
            self._update_tracking(gnss_data, features)
            self.last_metric = metric
            return metric

        self.consecutive_valid_fixes += 1
        self.in_recovery = False
        score = self.compute_reliability_score(features, GNSSTrustState.TRUSTED)
        metric = GNSSTrustMetric(
            trust_score=score,
            is_trusted=True,
            position_variance_scale=cfg.trusted_variance_scale,
            velocity_variance_scale=cfg.trusted_variance_scale,
            state=GNSSTrustState.TRUSTED,
            diagnostics={
                "accuracy_m": features.accuracy_m,
                "consecutive_valid_fixes": self.consecutive_valid_fixes,
                "anomaly_report": anomaly_rep,
                "anomaly_reasons": anomaly_rep.reason_codes if anomaly_rep is not None else [],
            },
        )
        self._update_tracking(gnss_data, features)
        self.last_metric = metric
        return metric

    def _update_tracking(self, gnss_data: Dict[str, Any], features: GNSSQualityFeatures) -> None:
        if "position_enu" in gnss_data:
            self.last_gnss_pos = np.asarray(gnss_data["position_enu"], dtype=np.float64)[:2]
        elif "east" in gnss_data and "north" in gnss_data:
            self.last_gnss_pos = np.array([float(gnss_data["east"]), float(gnss_data["north"])], dtype=np.float64)

        if "velocity_enu" in gnss_data:
            self.last_gnss_vel = np.asarray(gnss_data["velocity_enu"], dtype=np.float64)[:2]
        elif "speed" in gnss_data and "heading" in gnss_data:
            h = float(gnss_data["heading"])
            s = float(gnss_data["speed"])
            self.last_gnss_vel = np.array([s * np.sin(h), s * np.cos(h)], dtype=np.float64)

        ts = gnss_data.get("timestamp")
        if ts is not None and np.isfinite(ts):
            self.last_gnss_timestamp = float(ts)
        elif self.last_gnss_timestamp is not None:
            self.last_gnss_timestamp += features.dt_s
        else:
            self.last_gnss_timestamp = 0.0
