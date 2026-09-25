import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

from navigation.interfaces import (
    GNSSDegradationRisk,
    GNSSDegradationPrediction,
    IGNSSDegradationPredictor,
)


@dataclass
class GNSSDegradationConfig:
    min_fixes_for_prediction: int = 3
    window_size: int = 25
    accuracy_degrading_threshold_m: float = 6.0
    accuracy_high_risk_threshold_m: float = 14.0
    accuracy_growth_rate_threshold_mps: float = 1.0
    hdop_degrading_threshold: float = 1.8
    hdop_high_risk_threshold: float = 3.2
    satellites_degrading_threshold: int = 6
    satellites_high_risk_threshold: int = 4
    cn0_degrading_threshold_dbhz: float = 30.0
    cn0_high_risk_threshold_dbhz: float = 24.0
    norm_innovation_degrading: float = 2.5
    norm_innovation_high_risk: float = 4.5
    vel_residual_degrading_mps: float = 1.8
    vel_residual_high_risk_mps: float = 4.0
    jump_speed_unpredictable_mps: float = 45.0
    implied_accel_unpredictable_mps2: float = 25.0
    risk_degrading_threshold: float = 0.35
    risk_high_risk_threshold: float = 0.65
    weight_quality: float = 0.30
    weight_residuals: float = 0.25
    weight_stability: float = 0.20
    weight_velocity: float = 0.15
    weight_environment: float = 0.10


@dataclass
class GNSSPredictionEvaluationMetrics:
    total_epochs: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    lead_times_s: List[float] = field(default_factory=list)
    detection_latencies_s: List[float] = field(default_factory=list)
    total_degradation_events: int = 0
    detected_degradation_events: int = 0
    missed_degradation_events: int = 0

    def compute_summary(self) -> Dict[str, float]:
        prec = 1.0
        if (self.true_positives + self.false_positives) > 0:
            prec = self.true_positives / (self.true_positives + self.false_positives)

        rec = 1.0
        if (self.true_positives + self.false_negatives) > 0:
            rec = self.true_positives / (self.true_positives + self.false_negatives)

        fpr = 0.0
        if (self.false_positives + self.true_negatives) > 0:
            fpr = self.false_positives / (self.false_positives + self.true_negatives)

        mean_lead_time = 0.0
        if self.lead_times_s:
            mean_lead_time = float(np.mean(self.lead_times_s))

        mean_latency = 0.0
        if self.detection_latencies_s:
            mean_latency = float(np.mean(self.detection_latencies_s))

        missed_rate = 0.0
        if self.total_degradation_events > 0:
            missed_rate = (self.missed_degradation_events / self.total_degradation_events) * 100.0

        return {
            "total_epochs": float(self.total_epochs),
            "true_positives": float(self.true_positives),
            "false_positives": float(self.false_positives),
            "true_negatives": float(self.true_negatives),
            "false_negatives": float(self.false_negatives),
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "false_positive_rate": float(round(fpr, 4)),
            "mean_detection_lead_time_s": float(round(mean_lead_time, 2)),
            "mean_event_detection_latency_s": float(round(mean_latency, 2)),
            "total_degradation_events": float(self.total_degradation_events),
            "detected_degradation_events": float(self.detected_degradation_events),
            "missed_degradation_events": float(self.missed_degradation_events),
            "missed_event_rate_pct": float(round(missed_rate, 2)),
        }



class BaselineGNSSDegradationPredictor(IGNSSDegradationPredictor):
    def __init__(self, config: Optional[GNSSDegradationConfig] = None):
        self.config = config or GNSSDegradationConfig()
        self.history: List[Dict[str, Any]] = []
        self.last_prediction: Optional[GNSSDegradationPrediction] = None

    def reset(self) -> None:
        self.history.clear()
        self.last_prediction = None

    def _extract_gnss_scalars(self, gnss_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not gnss_data or not isinstance(gnss_data, dict):
            return {}

        acc = float(gnss_data.get("accuracy", gnss_data.get("hacc", 3.0)))
        hdop = float(gnss_data.get("hdop", gnss_data.get("dop", 1.0)))
        sats = float(gnss_data.get("satellites", gnss_data.get("num_sats", 10.0)))
        cn0 = float(gnss_data.get("cn0", gnss_data.get("snr", 38.0)))
        speed = float(gnss_data.get("speed", 0.0))
        ts = float(gnss_data.get("timestamp", len(self.history) * 0.1))

        pos = None
        if "position_enu" in gnss_data:
            pos = np.asarray(gnss_data["position_enu"], dtype=np.float64)[:2]
        elif "position" in gnss_data:
            pos = np.asarray(gnss_data["position"], dtype=np.float64)[:2]
        elif "east" in gnss_data and "north" in gnss_data:
            pos = np.array([float(gnss_data["east"]), float(gnss_data["north"])], dtype=np.float64)

        vel = None
        if "velocity_enu" in gnss_data:
            vel = np.asarray(gnss_data["velocity_enu"], dtype=np.float64)[:2]
        elif "velocity" in gnss_data:
            vel = np.asarray(gnss_data["velocity"], dtype=np.float64)[:2]

        return {
            "accuracy": acc if np.isfinite(acc) else 999.0,
            "hdop": hdop if np.isfinite(hdop) else 5.0,
            "satellites": sats if np.isfinite(sats) else 0.0,
            "cn0": cn0 if np.isfinite(cn0) else 0.0,
            "speed": speed if np.isfinite(speed) else 0.0,
            "timestamp": ts if np.isfinite(ts) else 0.0,
            "pos": pos,
            "vel": vel,
        }

    def predict(
        self,
        gnss_data: Optional[Dict[str, Any]],
        navigation_state: Optional[np.ndarray] = None,
        ai_velocity: Optional[np.ndarray] = None,
        map_context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> GNSSDegradationPrediction:
        scalars = self._extract_gnss_scalars(gnss_data)
        if not scalars:
            pred = GNSSDegradationPrediction(
                risk_level=GNSSDegradationRisk.HIGH_RISK if len(self.history) >= self.config.min_fixes_for_prediction else GNSSDegradationRisk.UNKNOWN,
                confidence=0.90 if len(self.history) >= self.config.min_fixes_for_prediction else 0.0,
                predicted_outage_probability=1.0 if len(self.history) >= self.config.min_fixes_for_prediction else 0.0,
                evidence_score=1.0 if len(self.history) >= self.config.min_fixes_for_prediction else 0.0,
                reasons=["GNSS signal missing or null"],
                recommended_actions=["PREPARE_DR_STATE", "PREPARE_ADAPTIVE_FUSION"] if len(self.history) >= self.config.min_fixes_for_prediction else ["AWAIT_EVIDENCE"],
            )
            self.last_prediction = pred
            return pred

        self.history.append(scalars)
        if len(self.history) > self.config.window_size:
            self.history.pop(0)

        if len(self.history) < self.config.min_fixes_for_prediction:
            pred = GNSSDegradationPrediction(
                risk_level=GNSSDegradationRisk.UNKNOWN,
                confidence=0.0,
                predicted_outage_probability=0.0,
                evidence_score=0.0,
                signals={"accuracy": scalars["accuracy"], "satellites": scalars["satellites"]},
                reasons=["Insufficient historical fixes to establish baseline"],
                recommended_actions=["AWAIT_EVIDENCE"],
            )
            self.last_prediction = pred
            return pred

        cur_acc = scalars["accuracy"]
        cur_hdop = scalars["hdop"]
        cur_sats = scalars["satellites"]
        cur_cn0 = scalars["cn0"]

        score_acc = 0.0
        if cur_acc >= self.config.accuracy_high_risk_threshold_m:
            score_acc = 1.0
        elif cur_acc >= self.config.accuracy_degrading_threshold_m:
            score_acc = 0.50 + 0.50 * (cur_acc - self.config.accuracy_degrading_threshold_m) / (self.config.accuracy_high_risk_threshold_m - self.config.accuracy_degrading_threshold_m)
        else:
            score_acc = max(0.0, cur_acc / self.config.accuracy_degrading_threshold_m * 0.40)

        score_hdop = 0.0
        if cur_hdop >= self.config.hdop_high_risk_threshold:
            score_hdop = 1.0
        elif cur_hdop >= self.config.hdop_degrading_threshold:
            score_hdop = 0.50 + 0.50 * (cur_hdop - self.config.hdop_degrading_threshold) / (self.config.hdop_high_risk_threshold - self.config.hdop_degrading_threshold)

        score_sats = 0.0
        if cur_sats <= self.config.satellites_high_risk_threshold:
            score_sats = 1.0
        elif cur_sats <= self.config.satellites_degrading_threshold:
            score_sats = 0.50 + 0.50 * (self.config.satellites_degrading_threshold - cur_sats) / max(1.0, float(self.config.satellites_degrading_threshold - self.config.satellites_high_risk_threshold))

        score_cn0 = 0.0
        if cur_cn0 <= self.config.cn0_high_risk_threshold_dbhz:
            score_cn0 = 1.0
        elif cur_cn0 <= self.config.cn0_degrading_threshold_dbhz:
            score_cn0 = 0.50 + 0.50 * (self.config.cn0_degrading_threshold_dbhz - cur_cn0) / max(1.0, float(self.config.cn0_degrading_threshold_dbhz - self.config.cn0_high_risk_threshold_dbhz))

        score_growth = 0.0
        if len(self.history) >= 4:
            first_acc = self.history[-4]["accuracy"]
            dt_span = max(0.1, self.history[-1]["timestamp"] - self.history[-4]["timestamp"])
            rate = (cur_acc - first_acc) / dt_span
            if rate > self.config.accuracy_growth_rate_threshold_mps:
                score_growth = min(1.0, rate / (self.config.accuracy_growth_rate_threshold_mps * 2.5))

        quality_subscore = 0.35 * score_acc + 0.25 * score_hdop + 0.20 * score_sats + 0.10 * score_cn0 + 0.10 * score_growth

        residuals_subscore = 0.0
        pos_innov = 0.0
        norm_innov = 0.0
        if navigation_state is not None and scalars["pos"] is not None:
            nav_pos = np.asarray(navigation_state[:2], dtype=np.float64)
            pos_innov = float(np.linalg.norm(scalars["pos"] - nav_pos))
            nav_unc = float(kwargs.get("position_uncertainty", 3.0))
            tot_sigma = float(np.sqrt(nav_unc ** 2 + cur_acc ** 2))
            norm_innov = pos_innov / max(0.5, tot_sigma)
            if norm_innov >= self.config.norm_innovation_high_risk:
                residuals_subscore = 1.0
            elif norm_innov >= self.config.norm_innovation_degrading:
                residuals_subscore = 0.50 + 0.50 * (norm_innov - self.config.norm_innovation_degrading) / (self.config.norm_innovation_high_risk - self.config.norm_innovation_degrading)
            else:
                residuals_subscore = min(0.40, norm_innov / self.config.norm_innovation_degrading * 0.40)

        stability_subscore = 0.0
        jump_speed = 0.0
        implied_accel = 0.0
        if len(self.history) >= 2 and scalars["pos"] is not None and self.history[-2]["pos"] is not None:
            prev_pos = self.history[-2]["pos"]
            dt_step = max(0.01, scalars["timestamp"] - self.history[-2]["timestamp"])
            jump_speed = float(np.linalg.norm(scalars["pos"] - prev_pos)) / dt_step

            if len(self.history) >= 3 and self.history[-3]["pos"] is not None:
                prev_prev = self.history[-3]["pos"]
                dt_prev = max(0.01, self.history[-2]["timestamp"] - self.history[-3]["timestamp"])
                v_cur = (scalars["pos"] - prev_pos) / dt_step
                v_prev = (prev_pos - prev_prev) / dt_prev
                implied_accel = float(np.linalg.norm(v_cur - v_prev)) / dt_step

            if jump_speed > 25.0 or implied_accel > 15.0:
                stability_subscore = 1.0
            elif jump_speed > 12.0 or implied_accel > 8.0:
                stability_subscore = 0.60
            elif jump_speed > 6.0:
                stability_subscore = 0.30

        velocity_subscore = 0.0
        vel_residual = 0.0
        if ai_velocity is not None and scalars["vel"] is not None:
            ai_v = np.asarray(ai_velocity[:2], dtype=np.float64)
            vel_residual = float(np.linalg.norm(scalars["vel"] - ai_v))
            if vel_residual >= self.config.vel_residual_high_risk_mps:
                velocity_subscore = 1.0
            elif vel_residual >= self.config.vel_residual_degrading_mps:
                velocity_subscore = 0.50 + 0.50 * (vel_residual - self.config.vel_residual_degrading_mps) / (self.config.vel_residual_high_risk_mps - self.config.vel_residual_degrading_mps)
            else:
                velocity_subscore = min(0.35, vel_residual / self.config.vel_residual_degrading_mps * 0.35)

        environment_subscore = 0.0
        if map_context and isinstance(map_context, dict):
            if map_context.get("approaching_tunnel", False):
                dist_to_tunnel = float(map_context.get("distance_to_tunnel_m", 100.0))
                if dist_to_tunnel <= 30.0:
                    environment_subscore = 1.0
                elif dist_to_tunnel <= 80.0:
                    environment_subscore = 0.60
                elif dist_to_tunnel <= 150.0:
                    environment_subscore = 0.30
            elif map_context.get("in_urban_canyon", False):
                environment_subscore = 0.50

        evidence_score = (
            self.config.weight_quality * quality_subscore
            + self.config.weight_residuals * residuals_subscore
            + self.config.weight_stability * stability_subscore
            + self.config.weight_velocity * velocity_subscore
            + self.config.weight_environment * environment_subscore
        )

        signals = {
            "accuracy_m": cur_acc,
            "hdop": cur_hdop,
            "satellites": cur_sats,
            "cn0_dbhz": cur_cn0,
            "normalized_innovation": norm_innov,
            "jump_speed_mps": jump_speed,
            "implied_accel_mps2": implied_accel,
            "velocity_residual_mps": vel_residual,
            "quality_subscore": quality_subscore,
            "residuals_subscore": residuals_subscore,
            "stability_subscore": stability_subscore,
            "velocity_subscore": velocity_subscore,
            "environment_subscore": environment_subscore,
        }

        reasons = []
        if cur_acc >= self.config.accuracy_degrading_threshold_m:
            reasons.append(f"Accuracy degraded to {cur_acc:.1f}m")
        if cur_hdop >= self.config.hdop_degrading_threshold:
            reasons.append(f"Elevated HDOP {cur_hdop:.1f}")
        if cur_sats <= self.config.satellites_degrading_threshold:
            reasons.append(f"Low satellite count {int(cur_sats)}")
        if norm_innov >= self.config.norm_innovation_degrading:
            reasons.append(f"High normalized innovation {norm_innov:.1f}")
        if vel_residual >= self.config.vel_residual_degrading_mps:
            reasons.append(f"Velocity residual discrepancy {vel_residual:.1f} m/s")
        if jump_speed > 20.0:
            reasons.append(f"Position jump speed {jump_speed:.1f} m/s")
        if environment_subscore > 0.40:
            reasons.append("Environmental satellite obstruction hazard detected")

        if jump_speed >= self.config.jump_speed_unpredictable_mps or implied_accel >= self.config.implied_accel_unpredictable_mps2:
            risk = GNSSDegradationRisk.UNPREDICTABLE
            prob = 0.85
            actions = ["ISOLATE_GNSS_OUTLIERS", "PRESERVE_NAVIGATION_CONTEXT", "INCREASE_MONITORING_FREQUENCY"]
            lead_time = 0.0
        elif evidence_score >= self.config.risk_high_risk_threshold:
            risk = GNSSDegradationRisk.HIGH_RISK
            prob = float(min(1.0, 0.70 + 0.30 * evidence_score))
            actions = ["PREPARE_DR_STATE", "PREPARE_ADAPTIVE_FUSION", "PRESERVE_NAVIGATION_CONTEXT", "INCREASE_MONITORING_FREQUENCY"]
            lead_time = max(1.0, 4.0 * (1.0 - evidence_score))
        elif evidence_score >= self.config.risk_degrading_threshold:
            risk = GNSSDegradationRisk.DEGRADING
            prob = float(min(0.70, 0.30 + 0.40 * (evidence_score - self.config.risk_degrading_threshold) / (self.config.risk_high_risk_threshold - self.config.risk_degrading_threshold)))
            actions = ["PREPARE_DR_STATE", "PREPARE_ADAPTIVE_FUSION"]
            lead_time = max(3.0, 7.0 * (1.0 - evidence_score))
        else:
            risk = GNSSDegradationRisk.NORMAL
            prob = float(max(0.0, evidence_score * 0.30))
            actions = ["MONITOR_NOMINAL"]
            lead_time = 0.0

        conf = float(min(1.0, 0.50 + 0.50 * (len(self.history) / self.config.window_size)))

        prediction = GNSSDegradationPrediction(
            risk_level=risk,
            confidence=conf,
            predicted_outage_probability=prob,
            evidence_score=float(round(evidence_score, 4)),
            signals=signals,
            reasons=reasons,
            lead_time_estimate_s=float(round(lead_time, 2)),
            recommended_actions=actions,
        )
        self.last_prediction = prediction
        return prediction


class GNSSPredictionEvaluator:
    def __init__(self, predictor: IGNSSDegradationPredictor):
        self.predictor = predictor

    def evaluate_session(
        self,
        timeline: List[Dict[str, Any]],
        degradation_events: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[GNSSPredictionEvaluationMetrics, List[GNSSDegradationPrediction]]:
        self.predictor.reset()
        metrics = GNSSPredictionEvaluationMetrics()
        predictions: List[GNSSDegradationPrediction] = []

        events = degradation_events or []
        metrics.total_degradation_events = len(events)
        event_detected_flags = [False] * len(events)

        first_alarm_time_per_event = [None] * len(events)

        for step in timeline:
            metrics.total_epochs += 1
            ts = float(step.get("timestamp", 0.0))
            gnss_data = step.get("gnss_data")
            nav_state = step.get("navigation_state")
            ai_vel = step.get("ai_velocity")
            map_ctx = step.get("map_context")
            gt_label = str(step.get("ground_truth_label", "NORMAL"))

            pred = self.predictor.predict(
                gnss_data=gnss_data,
                navigation_state=nav_state,
                ai_velocity=ai_vel,
                map_context=map_ctx,
            )
            predictions.append(pred)

            is_alarm = pred.risk_level in [
                GNSSDegradationRisk.DEGRADING,
                GNSSDegradationRisk.HIGH_RISK,
                GNSSDegradationRisk.UNPREDICTABLE,
            ]

            is_gt_degraded = gt_label in ["DEGRADING", "HIGH_RISK", "OUTAGE"]

            if is_alarm and is_gt_degraded:
                metrics.true_positives += 1
            elif is_alarm and not is_gt_degraded:
                metrics.false_positives += 1
            elif not is_alarm and not is_gt_degraded:
                metrics.true_negatives += 1
            elif not is_alarm and is_gt_degraded:
                metrics.false_negatives += 1

            for idx, ev in enumerate(events):
                t_deg_start = ev.get("degradation_start", 0.0)
                t_outage_start = ev.get("outage_start", t_deg_start)
                t_outage_end = ev.get("outage_end", t_outage_start + 10.0)

                if t_deg_start <= ts <= t_outage_end:
                    if is_alarm and not event_detected_flags[idx]:
                        event_detected_flags[idx] = True
                        first_alarm_time_per_event[idx] = ts
                        latency = max(0.0, ts - t_deg_start)
                        lead_time = max(0.0, t_outage_start - ts)
                        metrics.detection_latencies_s.append(latency)
                        metrics.lead_times_s.append(lead_time)

        metrics.detected_degradation_events = sum(1 for f in event_detected_flags if f)
        metrics.missed_degradation_events = metrics.total_degradation_events - metrics.detected_degradation_events

        return metrics, predictions
