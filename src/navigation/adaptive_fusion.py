import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from navigation.interfaces import (
    AdaptiveNoiseParameters,
    IAdaptiveFusionEngine,
    GNSSTrustState,
    GNSSTrustMetric,
    AIVelocityMeasurement,
    RoadHypothesis,
    MapConstraint,
)


@dataclass
class AdaptiveFusionConfig:
    gnss_trusted_pos_scale: float = 1.0
    gnss_trusted_vel_scale: float = 1.0
    gnss_degraded_pos_scale: float = 5.0
    gnss_degraded_vel_scale: float = 2.5
    gnss_suspicious_pos_scale: float = 50.0
    gnss_suspicious_vel_scale: float = 50.0
    gnss_unusable_scale: float = 10000.0
    ai_nominal_scale: float = 1.0
    ai_gnss_healthy_scale: float = 1.5
    ai_gnss_lost_scale: float = 0.8
    ai_degraded_scale: float = 5.0
    ai_rejected_scale: float = 10000.0
    map_nominal_scale: float = 1.0
    map_gnss_healthy_scale: float = 2.0
    map_gnss_degraded_scale: float = 0.8
    map_gnss_lost_scale: float = 0.5
    map_rejected_scale: float = 10000.0
    process_noise_moving_scale: float = 1.0
    process_noise_stationary_scale: float = 0.1
    process_noise_dr_pos_scale: float = 2.0
    process_noise_dr_vel_scale: float = 1.5


class AdaptiveFusionEngine(IAdaptiveFusionEngine):
    def __init__(self, config: Optional[AdaptiveFusionConfig] = None):
        self.config = config or AdaptiveFusionConfig()
        self.last_params: Optional[AdaptiveNoiseParameters] = None
        self.counters: Dict[str, int] = {
            "gnss_updates": 0,
            "gnss_downweighted": 0,
            "gnss_rejected": 0,
            "ai_updates": 0,
            "ai_downweighted": 0,
            "ai_rejected": 0,
            "map_updates": 0,
            "map_downweighted": 0,
            "map_rejected": 0,
        }

    def reset(self) -> None:
        self.last_params = None
        for k in self.counters:
            self.counters[k] = 0

    def compute_adaptive_noise(
        self,
        current_mode: str = "GNSS_INS",
        innovation: Optional[np.ndarray] = None,
        motion_state: str = "MOVING",
        *args,
        **kwargs,
    ) -> AdaptiveNoiseParameters:
        cfg = self.config
        gnss_trust = kwargs.get("gnss_trust")
        ai_velocity = kwargs.get("ai_velocity")
        map_constraint = kwargs.get("map_constraint") or kwargs.get("road_hypothesis")

        downweighted_reasons: List[str] = []
        rejected_reasons: List[str] = []

        is_stationary = (motion_state.upper() == "STATIONARY")
        mode_str = current_mode.upper() if isinstance(current_mode, str) else str(current_mode)
        is_dr_mode = ("DEAD_RECKONING" in mode_str or "DR" in mode_str)

        q_pos = cfg.process_noise_dr_pos_scale if is_dr_mode else cfg.process_noise_moving_scale
        q_vel = cfg.process_noise_dr_vel_scale if is_dr_mode else cfg.process_noise_moving_scale
        q_att = 1.0

        if is_stationary:
            q_pos *= cfg.process_noise_stationary_scale
            q_vel *= cfg.process_noise_stationary_scale

        gnss_pos_scale = cfg.gnss_trusted_pos_scale
        gnss_vel_scale = cfg.gnss_trusted_vel_scale
        gnss_weight = 1.0
        gnss_status = "HEALTHY"

        if gnss_trust is None:
            gnss_pos_scale = cfg.gnss_unusable_scale
            gnss_vel_scale = cfg.gnss_unusable_scale
            gnss_weight = 0.0
            gnss_status = "LOST"
            rejected_reasons.append("GNSS rejected: no trust metric provided or signal lost")
            self.counters["gnss_rejected"] += 1
        elif gnss_trust.state == GNSSTrustState.UNUSABLE:
            gnss_pos_scale = cfg.gnss_unusable_scale
            gnss_vel_scale = cfg.gnss_unusable_scale
            gnss_weight = 0.0
            gnss_status = "UNUSABLE"
            reasons_str = ", ".join(gnss_trust.diagnostics.get("reasons", ["unusable_fix"]))
            rejected_reasons.append(f"GNSS rejected: {reasons_str}")
            self.counters["gnss_rejected"] += 1
        elif gnss_trust.state == GNSSTrustState.SUSPICIOUS:
            scale_val = max(cfg.gnss_suspicious_pos_scale, float(gnss_trust.position_variance_scale))
            gnss_pos_scale = scale_val
            gnss_vel_scale = scale_val
            gnss_weight = float(1.0 / scale_val)
            gnss_status = "SUSPICIOUS"
            downweighted_reasons.append("GNSS down-weighted: suspicious position jump, innovation residual or heading divergence")
            self.counters["gnss_downweighted"] += 1
        elif gnss_trust.state == GNSSTrustState.DEGRADED:
            scale_val = max(cfg.gnss_degraded_pos_scale, float(gnss_trust.position_variance_scale))
            gnss_pos_scale = scale_val
            gnss_vel_scale = float(scale_val * 0.5)
            gnss_weight = float(1.0 / scale_val)
            gnss_status = "DEGRADED"
            downweighted_reasons.append("GNSS down-weighted: degraded reported accuracy or recovery hysteresis")
            self.counters["gnss_downweighted"] += 1
        else:
            self.counters["gnss_updates"] += 1

        ai_scale = cfg.ai_nominal_scale
        ai_weight = 1.0
        ai_status = "VALID"

        if ai_velocity is None:
            ai_scale = cfg.ai_rejected_scale
            ai_weight = 0.0
            ai_status = "UNAVAILABLE"
            rejected_reasons.append("AI velocity rejected: no measurement provided")
            self.counters["ai_rejected"] += 1
        elif not ai_velocity.is_valid:
            ai_scale = cfg.ai_rejected_scale
            ai_weight = 0.0
            ai_status = ai_velocity.status
            reason_txt = ai_velocity.reason or ai_velocity.status
            rejected_reasons.append(f"AI velocity rejected: {reason_txt}")
            self.counters["ai_rejected"] += 1
        elif ai_velocity.status == "VALID_STATIONARY":
            ai_scale = 0.1
            ai_weight = 1.0
            ai_status = "STATIONARY"
            self.counters["ai_updates"] += 1
        else:
            if gnss_status == "HEALTHY":
                ai_scale = cfg.ai_gnss_healthy_scale
                ai_weight = float(1.0 / ai_scale)
            elif gnss_status in ("DEGRADED", "SUSPICIOUS"):
                ai_scale = cfg.ai_nominal_scale
                ai_weight = 1.0
            else:
                ai_scale = cfg.ai_gnss_lost_scale
                ai_weight = float(1.0 / ai_scale)

            if ai_velocity.confidence < 0.7:
                ai_scale *= cfg.ai_degraded_scale
                ai_weight = float(1.0 / ai_scale)
                ai_status = "DEGRADED"
                downweighted_reasons.append("AI velocity down-weighted: low model confidence")
                self.counters["ai_downweighted"] += 1
            else:
                self.counters["ai_updates"] += 1

        map_scale = cfg.map_nominal_scale
        map_weight = 1.0
        map_status = "UNAVAILABLE"

        if map_constraint is None:
            map_scale = cfg.map_rejected_scale
            map_weight = 0.0
            rejected_reasons.append("Map constraint rejected: no map matching hypothesis available")
            self.counters["map_rejected"] += 1
        else:
            conf = 1.0
            cross_track = 0.0
            if hasattr(map_constraint, "confidence"):
                conf = float(map_constraint.confidence)
                cross_track = float(getattr(map_constraint, "distance_to_road", 0.0))
            elif hasattr(map_constraint, "probability"):
                conf = float(map_constraint.probability)
                cross_track = float(getattr(map_constraint, "cross_track_distance_m", 0.0))
            elif isinstance(map_constraint, dict):
                conf = float(map_constraint.get("confidence", map_constraint.get("probability", 1.0)))
                cross_track = float(map_constraint.get("cross_track", map_constraint.get("distance_to_road", 0.0)))

            if conf < 0.4 or cross_track > 35.0:
                map_scale = cfg.map_rejected_scale
                map_weight = 0.0
                map_status = "REJECTED"
                rejected_reasons.append(f"Map constraint rejected: low confidence ({conf:.2f}) or large cross-track ({cross_track:.1f}m)")
                self.counters["map_rejected"] += 1
            else:
                if gnss_status == "HEALTHY":
                    map_scale = cfg.map_gnss_healthy_scale / max(conf, 0.1)
                    map_weight = float(1.0 / map_scale)
                    map_status = "NOMINAL"
                elif gnss_status in ("DEGRADED", "SUSPICIOUS"):
                    map_scale = cfg.map_gnss_degraded_scale / max(conf, 0.1)
                    map_weight = float(1.0 / map_scale)
                    map_status = "ELEVATED_INFLUENCE"
                else:
                    map_scale = cfg.map_gnss_lost_scale / max(conf, 0.1)
                    map_weight = float(1.0 / map_scale)
                    map_status = "PRIMARY_LATERAL_CONSTRAINT"

                self.counters["map_updates"] += 1

        telemetry: Dict[str, Any] = {
            "gnss_status": gnss_status,
            "gnss_weight": gnss_weight,
            "gnss_noise_scale": gnss_pos_scale,
            "ai_status": ai_status,
            "ai_weight": ai_weight,
            "ai_noise_scale": ai_scale,
            "map_status": map_status,
            "map_weight": map_weight,
            "map_noise_scale": map_scale,
            "is_stationary": is_stationary,
            "is_dr_mode": is_dr_mode,
            "counters": dict(self.counters),
        }

        params = AdaptiveNoiseParameters(
            process_noise_scale_pos=float(q_pos),
            process_noise_scale_vel=float(q_vel),
            process_noise_scale_att=float(q_att),
            measurement_noise_scale_gnss_pos=float(gnss_pos_scale),
            measurement_noise_scale_gnss_vel=float(gnss_vel_scale),
            measurement_noise_scale_ai_vel=float(ai_scale),
            measurement_noise_scale_map=float(map_scale),
            telemetry=telemetry,
            downweighted_reasons=downweighted_reasons,
            rejected_reasons=rejected_reasons,
        )
        self.last_params = params
        return params
