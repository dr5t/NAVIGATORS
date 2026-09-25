from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import numpy as np

from src.navigation.interfaces import (
    IConfidenceEstimator,
    ConfidenceEstimate,
    GNSSTrustMetric,
    GNSSTrustState,
    AIVelocityMeasurement,
    RoadHypothesis,
)


@dataclass
class ConfidenceConfig:
    position_alert_limit_m: float = 15.0
    velocity_alert_limit_mps: float = 2.5
    heading_alert_limit_deg: float = 15.0
    nominal_position_scale_m: float = 8.0
    nominal_velocity_scale_mps: float = 1.5


class ConfidenceEstimator(IConfidenceEstimator):
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

    def compute_confidence(
        self,
        covariance: np.ndarray,
        dr_duration: float = 0.0,
        map_confidence: float = 0.0,
        *args,
        **kwargs,
    ) -> ConfidenceEstimate:
        cov = np.asarray(covariance, dtype=np.float64)
        if cov.ndim == 1:
            h_acc = float(cov[0])
            v_acc = float(cov[1]) if len(cov) > 1 else 0.0
            vel_acc = float(cov[2]) if len(cov) > 2 else 0.0
            head_acc = float(cov[3]) if len(cov) > 3 else 0.0
        elif cov.ndim == 2:
            dim = cov.shape[0]
            h_acc = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1])))
            v_acc = float(np.sqrt(max(0.0, cov[2, 2]))) if dim > 2 else 0.0
            vel_acc = float(np.sqrt(max(0.0, cov[3, 3] + cov[4, 4]))) if dim > 4 else 0.0
            head_acc = float(np.degrees(np.sqrt(max(0.0, cov[8, 8])))) if dim > 8 else 0.0
        else:
            h_acc = 10.0
            v_acc = 5.0
            vel_acc = 1.0
            head_acc = 5.0

        gnss_trust: Optional[GNSSTrustMetric] = kwargs.get("gnss_trust")
        ai_velocity: Optional[AIVelocityMeasurement] = kwargs.get("ai_velocity")
        road_hypothesis: Optional[RoadHypothesis] = kwargs.get("road_hypothesis")
        ekf_mode: str = str(kwargs.get("ekf_mode", "GNSS_INS"))
        distance_traveled: float = float(kwargs.get("distance_traveled", 0.0))
        timestamp: float = float(kwargs.get("timestamp", 0.0))

        gnss_info: Dict[str, Any] = {
            "status": "UNAVAILABLE",
            "is_trusted": False,
            "trust_score": 0.0,
            "variance_scale": 10000.0,
            "reported_accuracy_m": None,
            "reasons": ["no_gnss_metric"],
        }
        if gnss_trust is not None:
            st = getattr(gnss_trust.state, "value", str(gnss_trust.state))
            gnss_info["status"] = st
            gnss_info["is_trusted"] = bool(gnss_trust.is_trusted)
            gnss_info["trust_score"] = float(gnss_trust.trust_score)
            gnss_info["variance_scale"] = float(gnss_trust.position_variance_scale)
            gnss_info["reasons"] = list(gnss_trust.diagnostics.get("reasons", []))
            if "accuracy_m" in gnss_trust.diagnostics:
                gnss_info["reported_accuracy_m"] = float(gnss_trust.diagnostics["accuracy_m"])

        ai_info: Dict[str, Any] = {
            "status": "UNAVAILABLE",
            "is_valid": False,
            "std_dev_mps": None,
            "latency_ms": None,
            "reason": "no_ai_measurement",
        }
        if ai_velocity is not None:
            ai_info["status"] = str(ai_velocity.status)
            ai_info["is_valid"] = bool(ai_velocity.is_valid)
            std_dev = float(np.sqrt(ai_velocity.variance_east + ai_velocity.variance_north))
            ai_info["std_dev_mps"] = float(round(std_dev, 4))
            ai_info["latency_ms"] = float(round(ai_velocity.latency_ms, 2))
            ai_info["reason"] = ai_velocity.reason

        map_info: Dict[str, Any] = {
            "is_matched": bool(map_confidence > 0.5),
            "confidence": float(round(map_confidence, 4)),
            "cross_track_m": None,
            "segment_id": None,
        }
        if road_hypothesis is not None:
            map_info["is_matched"] = bool(road_hypothesis.probability > 0.5)
            map_info["confidence"] = float(round(road_hypothesis.probability, 4))
            map_info["cross_track_m"] = float(round(road_hypothesis.cross_track_distance_m, 2))
            map_info["segment_id"] = str(road_hypothesis.segment_id)

        cfg = self.config
        pos_score = float(np.exp(-h_acc / max(1.0, cfg.nominal_position_scale_m)))
        vel_score = float(np.exp(-vel_acc / max(0.2, cfg.nominal_velocity_scale_mps)))
        head_score = float(np.exp(-head_acc / max(1.0, cfg.heading_alert_limit_deg)))

        integrity_score = float(np.clip(pos_score * vel_score * head_score, 0.0, 1.0))

        if h_acc > cfg.position_alert_limit_m or vel_acc > cfg.velocity_alert_limit_mps:
            integrity_status = "UNRELIABLE"
        elif h_acc > (cfg.position_alert_limit_m * 0.6) or not gnss_info["is_trusted"]:
            integrity_status = "DEGRADED"
        else:
            integrity_status = "NOMINAL"

        if distance_traveled > 2.0:
            dr_drift_percent = float((h_acc / distance_traveled) * 100.0)
        else:
            dr_drift_percent = 0.0

        time_decay = float(np.exp(-max(0.0, dr_duration) / 60.0))
        overall_confidence = float(np.clip(integrity_score * max(0.1, time_decay), 0.0, 1.0))

        confidence_object = {
            "timestamp": timestamp,
            "uncertainties": {
                "position_horizontal_m": float(round(h_acc, 3)),
                "position_vertical_m": float(round(v_acc, 3)),
                "velocity_horizontal_mps": float(round(vel_acc, 3)),
                "heading_deg": float(round(head_acc, 2)),
            },
            "units": {
                "position_horizontal": "meters",
                "position_vertical": "meters",
                "velocity_horizontal": "meters_per_second",
                "heading": "degrees",
            },
            "sensors": {
                "gnss": gnss_info,
                "ai_velocity": ai_info,
                "map_matching": map_info,
            },
            "estimator": {
                "mode": ekf_mode,
                "integrity_status": integrity_status,
                "integrity_score": float(round(integrity_score, 4)),
                "dr_drift_percent": float(round(dr_drift_percent, 2)),
                "overall_confidence": float(round(overall_confidence, 4)),
            },
        }

        return ConfidenceEstimate(
            horizontal_accuracy_m=float(round(h_acc, 3)),
            heading_accuracy_deg=float(round(head_acc, 2)),
            integrity_score=float(round(integrity_score, 4)),
            overall_confidence=float(round(overall_confidence, 4)),
            dr_drift_percent=float(round(dr_drift_percent, 2)),
            velocity_uncertainty_mps=float(round(vel_acc, 3)),
            vertical_accuracy_m=float(round(v_acc, 3)),
            gnss_reliability=gnss_info,
            ai_velocity_reliability=ai_info,
            map_match_confidence=map_info,
            confidence_object=confidence_object,
            integrity_status=integrity_status,
        )
