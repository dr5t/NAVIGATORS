import os
import sys
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    GNSSDegradationRisk,
    GNSSDegradationPrediction,
    IGNSSDegradationPredictor,
)
from navigation.gnss_prediction import (
    GNSSDegradationConfig,
    BaselineGNSSDegradationPredictor,
    GNSSPredictionEvaluator,
    GNSSPredictionEvaluationMetrics,
)


def create_nominal_gnss_step(timestamp: float, east: float = 0.0, north: float = 0.0) -> dict:
    return {
        "timestamp": timestamp,
        "gnss_data": {
            "accuracy": 2.2,
            "hdop": 0.9,
            "satellites": 11,
            "cn0": 39.0,
            "speed": 12.0,
            "position_enu": [east, north],
            "velocity_enu": [0.0, 12.0],
            "timestamp": timestamp,
        },
        "navigation_state": np.array([east, north, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        "ai_velocity": np.array([0.0, 12.0]),
        "map_context": {"approaching_tunnel": False, "in_urban_canyon": False},
        "ground_truth_label": "NORMAL",
    }


def create_degrading_gnss_step(
    timestamp: float,
    degradation_factor: float,
    east: float = 0.0,
    north: float = 0.0,
    ground_truth_label: str = "DEGRADING",
) -> dict:
    factor = max(0.0, min(1.0, degradation_factor))
    acc = 2.2 + factor * 14.0
    hdop = 0.9 + factor * 3.0
    sats = max(3, int(11 - factor * 8))
    cn0 = 39.0 - factor * 18.0
    drift_e = factor * 5.0
    return {
        "timestamp": timestamp,
        "gnss_data": {
            "accuracy": acc,
            "hdop": hdop,
            "satellites": sats,
            "cn0": cn0,
            "speed": 12.0,
            "position_enu": [east + drift_e, north],
            "velocity_enu": [drift_e / 0.5, 12.0],
            "timestamp": timestamp,
        },
        "navigation_state": np.array([east, north, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
        "ai_velocity": np.array([0.0, 12.0]),
        "map_context": {"approaching_tunnel": factor > 0.5, "distance_to_tunnel_m": max(10.0, 100.0 * (1.0 - factor))},
        "ground_truth_label": ground_truth_label,
    }


def test_insufficient_evidence_returns_unknown():
    predictor = BaselineGNSSDegradationPredictor()
    step1 = create_nominal_gnss_step(0.0)
    pred1 = predictor.predict(step1["gnss_data"], step1["navigation_state"], step1["ai_velocity"])
    assert pred1.risk_level == GNSSDegradationRisk.UNKNOWN
    assert pred1.confidence == 0.0
    assert "AWAIT_EVIDENCE" in pred1.recommended_actions

    step2 = create_nominal_gnss_step(0.1)
    pred2 = predictor.predict(step2["gnss_data"], step2["navigation_state"], step2["ai_velocity"])
    assert pred2.risk_level == GNSSDegradationRisk.UNKNOWN

    step3 = create_nominal_gnss_step(0.2)
    pred3 = predictor.predict(step3["gnss_data"], step3["navigation_state"], step3["ai_velocity"])
    assert pred3.risk_level == GNSSDegradationRisk.NORMAL
    assert pred3.confidence > 0.0


def test_nominal_healthy_gnss():
    predictor = BaselineGNSSDegradationPredictor()
    evaluator = GNSSPredictionEvaluator(predictor)

    timeline = []
    for step in range(50):
        t = step * 0.1
        timeline.append(create_nominal_gnss_step(t, east=0.0, north=12.0 * t))

    metrics, predictions = evaluator.evaluate_session(timeline)
    summary = metrics.compute_summary()

    assert summary["false_positive_rate"] == 0.0
    assert summary["false_positives"] == 0
    assert summary["true_negatives"] > 40
    for pred in predictions[3:]:
        assert pred.risk_level == GNSSDegradationRisk.NORMAL
        assert pred.predicted_outage_probability < 0.25
        assert "MONITOR_NOMINAL" in pred.recommended_actions


def test_tunnel_approach_lead_time():
    predictor = BaselineGNSSDegradationPredictor()
    evaluator = GNSSPredictionEvaluator(predictor)

    timeline = []
    total_steps = 150
    dt = 0.1

    degradation_start_t = 6.0
    outage_start_t = 10.0
    outage_end_t = 15.0

    events = [{
        "degradation_start": degradation_start_t,
        "outage_start": outage_start_t,
        "outage_end": outage_end_t,
    }]

    for step in range(total_steps):
        t = step * dt
        y_pos = 12.0 * t
        if t < degradation_start_t:
            timeline.append(create_nominal_gnss_step(t, 0.0, y_pos))
        elif t < outage_start_t:
            deg_progress = (t - degradation_start_t) / (outage_start_t - degradation_start_t)
            timeline.append(create_degrading_gnss_step(t, deg_progress, 0.0, y_pos, ground_truth_label="DEGRADING"))
        else:
            timeline.append({
                "timestamp": t,
                "gnss_data": None,
                "navigation_state": np.array([0.0, y_pos, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]),
                "ai_velocity": np.array([0.0, 12.0]),
                "map_context": {"approaching_tunnel": True, "distance_to_tunnel_m": 0.0},
                "ground_truth_label": "OUTAGE",
            })

    metrics, predictions = evaluator.evaluate_session(timeline, events)
    summary = metrics.compute_summary()

    assert summary["total_degradation_events"] == 1
    assert summary["detected_degradation_events"] == 1
    assert summary["missed_degradation_events"] == 0
    assert summary["mean_detection_lead_time_s"] >= 1.5
    assert summary["mean_detection_lead_time_s"] <= 4.0
    assert summary["recall"] >= 0.85
    assert summary["precision"] >= 0.85
    assert summary["false_positive_rate"] == 0.0

    elevated = [p for p in predictions if p.risk_level in [GNSSDegradationRisk.DEGRADING, GNSSDegradationRisk.HIGH_RISK]]
    assert len(elevated) > 0
    for p in elevated:
        assert "PREPARE_DR_STATE" in p.recommended_actions
        assert "PREPARE_ADAPTIVE_FUSION" in p.recommended_actions


def test_urban_canyon_multipath_degradation():
    predictor = BaselineGNSSDegradationPredictor()
    timeline = []

    for step in range(10):
        timeline.append(create_nominal_gnss_step(step * 0.1, 0.0, 10.0 * step * 0.1))

    for step in range(10, 25):
        t = step * 0.1
        drift = 12.0
        step_data = {
            "timestamp": t,
            "gnss_data": {
                "accuracy": 15.0,
                "hdop": 3.4,
                "satellites": 5,
                "cn0": 26.0,
                "speed": 10.0,
                "position_enu": [drift, 10.0 * t],
                "velocity_enu": [4.5, 10.0],
                "timestamp": t,
            },
            "navigation_state": np.array([0.0, 10.0 * t, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
            "ai_velocity": np.array([0.0, 10.0]),
            "map_context": {"in_urban_canyon": True},
            "ground_truth_label": "DEGRADING",
        }
        timeline.append(step_data)

    evaluator = GNSSPredictionEvaluator(predictor)
    metrics, predictions = evaluator.evaluate_session(timeline)
    degraded_preds = [p for p in predictions[12:] if p.risk_level in [GNSSDegradationRisk.DEGRADING, GNSSDegradationRisk.HIGH_RISK]]
    assert len(degraded_preds) >= 10
    top_pred = degraded_preds[-1]
    assert top_pred.evidence_score >= 0.50
    assert any("Accuracy degraded" in r for r in top_pred.reasons) or any("High normalized innovation" in r for r in top_pred.reasons)



def test_sudden_jamming_unpredictable():
    predictor = BaselineGNSSDegradationPredictor()
    for step in range(5):
        s = create_nominal_gnss_step(step * 0.1, 0.0, 12.0 * step * 0.1)
        predictor.predict(s["gnss_data"], s["navigation_state"], s["ai_velocity"])

    jam_step = {
        "accuracy": 45.0,
        "hdop": 5.8,
        "satellites": 3,
        "cn0": 15.0,
        "speed": 65.0,
        "position_enu": [55.0, 20.0],
        "velocity_enu": [45.0, 50.0],
        "timestamp": 0.5,
    }
    nav_st = np.array([0.0, 6.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0])
    ai_v = np.array([0.0, 12.0])

    pred = predictor.predict(jam_step, nav_st, ai_v)
    assert pred.risk_level == GNSSDegradationRisk.UNPREDICTABLE
    assert "ISOLATE_GNSS_OUTLIERS" in pred.recommended_actions
    assert "PRESERVE_NAVIGATION_CONTEXT" in pred.recommended_actions


def test_held_out_validation_session_suburban():
    predictor = BaselineGNSSDegradationPredictor()
    evaluator = GNSSPredictionEvaluator(predictor)

    timeline = []
    events = [{"degradation_start": 8.0, "outage_start": 12.0, "outage_end": 16.0}]

    for step in range(200):
        t = step * 0.1
        y = 15.0 * t
        if t < 8.0:
            timeline.append(create_nominal_gnss_step(t, 0.0, y))
        elif t < 12.0:
            p = (t - 8.0) / 4.0
            timeline.append(create_degrading_gnss_step(t, p, 0.0, y, ground_truth_label="DEGRADING"))
        elif t < 16.0:
            timeline.append({
                "timestamp": t,
                "gnss_data": None,
                "navigation_state": np.array([0.0, y, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0]),
                "ai_velocity": np.array([0.0, 15.0]),
                "map_context": {},
                "ground_truth_label": "OUTAGE",
            })
        else:
            timeline.append(create_nominal_gnss_step(t, 0.0, y))

    metrics, predictions = evaluator.evaluate_session(timeline, events)
    summary = metrics.compute_summary()

    assert summary["total_degradation_events"] == 1
    assert summary["detected_degradation_events"] == 1
    assert summary["missed_degradation_events"] == 0
    assert summary["precision"] >= 0.85
    assert summary["recall"] >= 0.85
    assert summary["false_positive_rate"] <= 0.05
    assert summary["mean_detection_lead_time_s"] >= 1.5


def test_held_out_validation_session_overpass():
    predictor = BaselineGNSSDegradationPredictor()
    evaluator = GNSSPredictionEvaluator(predictor)

    timeline = []
    events = [
        {"degradation_start": 5.0, "outage_start": 8.0, "outage_end": 11.0},
        {"degradation_start": 16.0, "outage_start": 19.0, "outage_end": 22.0},
    ]

    for step in range(260):
        t = step * 0.1
        y = 10.0 * t
        if 5.0 <= t < 8.0:
            p = (t - 5.0) / 3.0
            timeline.append(create_degrading_gnss_step(t, p, 0.0, y, ground_truth_label="DEGRADING"))
        elif 8.0 <= t < 11.0:
            timeline.append({
                "timestamp": t,
                "gnss_data": None,
                "navigation_state": np.array([0.0, y, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
                "ai_velocity": np.array([0.0, 10.0]),
                "map_context": {},
                "ground_truth_label": "OUTAGE",
            })
        elif 16.0 <= t < 19.0:
            p = (t - 16.0) / 3.0
            timeline.append(create_degrading_gnss_step(t, p, 0.0, y, ground_truth_label="DEGRADING"))
        elif 19.0 <= t < 22.0:
            timeline.append({
                "timestamp": t,
                "gnss_data": None,
                "navigation_state": np.array([0.0, y, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
                "ai_velocity": np.array([0.0, 10.0]),
                "map_context": {},
                "ground_truth_label": "OUTAGE",
            })
        else:
            timeline.append(create_nominal_gnss_step(t, 0.0, y))

    metrics, predictions = evaluator.evaluate_session(timeline, events)
    summary = metrics.compute_summary()

    assert summary["total_degradation_events"] == 2
    assert summary["detected_degradation_events"] == 2
    assert summary["missed_degradation_events"] == 0
    assert summary["precision"] >= 0.85
    assert summary["recall"] >= 0.75
    assert summary["false_positive_rate"] <= 0.05
    assert summary["mean_detection_lead_time_s"] >= 1.5



def test_modular_predictor_substitution():
    class CustomLearnedPredictor(IGNSSDegradationPredictor):
        def __init__(self):
            self.count = 0

        def reset(self) -> None:
            self.count = 0

        def predict(
            self,
            gnss_data: Optional[dict],
            navigation_state: Optional[np.ndarray] = None,
            ai_velocity: Optional[np.ndarray] = None,
            map_context: Optional[dict] = None,
            *args,
            **kwargs,
        ) -> GNSSDegradationPrediction:
            self.count += 1
            if gnss_data is None:
                return GNSSDegradationPrediction(
                    risk_level=GNSSDegradationRisk.HIGH_RISK,
                    confidence=0.99,
                    predicted_outage_probability=1.0,
                    evidence_score=1.0,
                )
            acc = gnss_data.get("accuracy", 3.0)
            if acc > 10.0:
                return GNSSDegradationPrediction(
                    risk_level=GNSSDegradationRisk.HIGH_RISK,
                    confidence=0.90,
                    predicted_outage_probability=0.85,
                    evidence_score=0.80,
                )
            return GNSSDegradationPrediction(
                risk_level=GNSSDegradationRisk.NORMAL,
                confidence=0.80,
                predicted_outage_probability=0.05,
                evidence_score=0.10,
            )

    custom_predictor = CustomLearnedPredictor()
    evaluator = GNSSPredictionEvaluator(custom_predictor)
    timeline = [
        create_nominal_gnss_step(0.0),
        create_nominal_gnss_step(0.1),
        create_degrading_gnss_step(0.2, degradation_factor=0.9, ground_truth_label="HIGH_RISK"),
    ]
    metrics, preds = evaluator.evaluate_session(timeline)
    assert custom_predictor.count == 3
    assert preds[0].risk_level == GNSSDegradationRisk.NORMAL
    assert preds[2].risk_level == GNSSDegradationRisk.HIGH_RISK
