"""Explicit, serializable interventions. No fitted or hardcoded outcome values."""
from dataclasses import asdict, dataclass, replace
import math


@dataclass(frozen=True)
class Architecture:
    name: str
    ai_velocity: bool = False
    ekf: bool = False
    vehicle_constraints: bool = False
    gnss_trust: bool = False
    anomaly_detection: bool = False
    adaptive_fusion: bool = False
    confidence: bool = False
    road_hypotheses: bool = False
    map_constraints: bool = False
    recovery: bool = False
    outage_prediction: bool = False
    production: bool = False


FULL = Architecture("Full Navigators", True, True, True, True, True, True, True, True, True, True, True)
ARCHITECTURES = {
    "A": Architecture("GNSS + INS"),
    "B": Architecture("INS + vehicle constraints", vehicle_constraints=True),
    "C": Architecture("INS + AI velocity", ai_velocity=True),
    "D": Architecture("INS + AI velocity + EKF", ai_velocity=True, ekf=True),
    "E": Architecture("INS + AI velocity + EKF + map constraints", ai_velocity=True, ekf=True, map_constraints=True),
    "F": FULL,
    "production": Architecture("Existing Python replay G", ai_velocity=True, ekf=True,
                               vehicle_constraints=True, map_constraints=True, recovery=True, production=True),
}
REMOVALS = ("gnss_trust", "anomaly_detection", "ai_velocity", "adaptive_fusion",
            "confidence", "road_hypotheses", "map_constraints", "recovery")
ARCHITECTURES.update({"F_without_" + component: replace(FULL, name="Full without " + component,
                                                        **{component: False}) for component in REMOVALS})
ENVIRONMENTS = ("urban", "highway", "rural", "hilly", "parallel_roads", "service_roads",
                "gnss_degraded", "gnss_fully_unavailable")
DEFAULTS = dict(schema_version="1.0.0", seed=42, sessions=[],
                architectures=list(ARCHITECTURES), outage_durations_s=[10, 30, 60, 120, 300],
                outage_starts_s=[30], include_observed=True, calibration_seconds=5.,
                recovery_window_s=15., recovery_error_threshold_m=5., recovery_hold_s=3.,
                prediction_horizon_s=5., prediction_threshold=.65, bootstrap_samples=2000,
                repeats=1, timeout_s=600., material_improvement_m=1.,
                production_adapter="python_replay_G", production_model=None, candidate_model=None)


def configuration(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown benchmark configuration fields")
    result = {**DEFAULTS, **value}
    if result["schema_version"] != "1.0.0" or result["production_adapter"] != "python_replay_G":
        raise ValueError("Unsupported benchmark schema/production adapter")
    if not isinstance(result["architectures"], list) or not result["architectures"] or any(k not in ARCHITECTURES for k in result["architectures"]):
        raise ValueError("Select known architectures")
    if len(set(result["architectures"])) != len(result["architectures"]):
        raise ValueError("Duplicate architectures")
    if not isinstance(result["sessions"], list) or not isinstance(result["include_observed"], bool):
        raise ValueError("sessions must be a list and include_observed boolean")
    for key in ("outage_durations_s", "outage_starts_s"):
        values = result[key]
        if not isinstance(values, list) or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a list of unique numbers")
        for v in values:
            if not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"Invalid {key}")
    for key in ("calibration_seconds", "recovery_window_s", "recovery_error_threshold_m", "recovery_hold_s",
                "prediction_horizon_s", "prediction_threshold", "timeout_s", "material_improvement_m"):
        v = result[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if result["prediction_threshold"] > 1 or result["recovery_hold_s"] > result["recovery_window_s"]:
        raise ValueError("Invalid prediction/recovery thresholds")
    for key in ("seed", "bootstrap_samples", "repeats"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    return result


def interventions():
    return {key: asdict(value) for key, value in ARCHITECTURES.items()}
