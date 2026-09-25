import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluation.research.config import ARCHITECTURES, ENVIRONMENTS, configuration, FULL, Architecture
from evaluation.research.metrics import score, DEFINITIONS
from evaluation.research.statistics import aggregate, paired_statistics
from evaluation.research.runner import provenance


def test_benchmark_architectures_and_environments():
    assert "A" in ARCHITECTURES
    assert "B" in ARCHITECTURES
    assert "C" in ARCHITECTURES
    assert "D" in ARCHITECTURES
    assert "E" in ARCHITECTURES
    assert "F" in ARCHITECTURES
    assert "production" in ARCHITECTURES

    assert "F_without_gnss_trust" in ARCHITECTURES
    assert "F_without_anomaly_detection" in ARCHITECTURES
    assert "F_without_ai_velocity" in ARCHITECTURES
    assert "F_without_adaptive_fusion" in ARCHITECTURES
    assert "F_without_confidence" in ARCHITECTURES
    assert "F_without_road_hypotheses" in ARCHITECTURES
    assert "F_without_map_constraints" in ARCHITECTURES
    assert "F_without_recovery" in ARCHITECTURES

    for env in ["urban", "highway", "rural", "hilly", "parallel_roads", "service_roads", "gnss_degraded", "gnss_fully_unavailable"]:
        assert env in ENVIRONMENTS


def test_benchmark_configuration_default_values():
    cfg = configuration({})
    assert cfg["schema_version"] == "1.0.0"
    assert cfg["outage_durations_s"] == [10, 30, 60, 120, 300]
    assert cfg["bootstrap_samples"] == 2000
    assert "A" in cfg["architectures"]
    assert "F" in cfg["architectures"]


def test_metric_scoring_computation():
    rows = []
    for i in range(50):
        ref_pos = np.array([float(i), float(i * 2)])
        est_pos = ref_pos + np.array([0.2, -0.1])
        ref_vel = np.array([1.0, 2.0])
        est_vel = np.array([1.05, 1.95])
        ref_heading = 0.5
        est_heading = 0.52

        rows.append({
            "sample_index": i,
            "reference_eligible": True,
            "receiver_epoch": True,
            "reference": {
                "position": ref_pos,
                "velocity": ref_vel,
                "heading": ref_heading,
                "anomaly": False,
                "road_segment_id": "seg_01",
            },
            "estimate": np.array([est_pos[0], est_pos[1], est_vel[0], est_vel[1], est_heading]),
            "future_outage": False,
            "anomaly_detected": False,
            "outage_probability": 0.05,
            "road_hypothesis_id": "seg_01",
            "map_match_id": "seg_01",
            "inference_latency_ms": 1.2,
            "loop_latency_ms": 2.5,
        })

    capabilities = FULL

    result = score(rows, capabilities)
    m = result["metrics"]

    assert m["position_rmse_m"] is not None
    assert m["position_rmse_m"] > 0.0
    assert m["absolute_trajectory_error_m"] is not None
    assert m["velocity_mae_mps"] is not None
    assert m["velocity_rmse_mps"] is not None
    assert m["heading_mae_deg"] is not None
    assert m["final_position_error_m"] is not None
    assert m["road_hypothesis_accuracy"] == 1.0
    assert m["map_matching_accuracy"] == 1.0


def test_statistical_aggregation():
    results = [
        {
            "status": "completed",
            "job_id": "job_01",
            "architecture": "F",
            "input_identity": {
                "source_dataset": "navigators_india",
                "source_kind": "navigators_collection",
                "manifest_sha256": "abc123hash",
                "reference_kind": "rtk",
                "reference_source_dataset": "navigators_india",
                "session_id": "sess_01",
            },
            "case": {"kind": "outage", "duration_s": 30},
            "regions": [
                {
                    "phase": "outage",
                    "environment": "urban",
                    "metrics": {"position_rmse_m": 2.1, "velocity_rmse_mps": 0.3},
                    "support": {
                        "position_rmse_m": {"sample_indices": [1, 2, 3]},
                        "velocity_rmse_mps": {"sample_indices": [1, 2, 3]},
                    },
                }
            ],
        },
        {
            "status": "completed",
            "job_id": "job_02",
            "architecture": "F",
            "input_identity": {
                "source_dataset": "navigators_india",
                "source_kind": "navigators_collection",
                "manifest_sha256": "abc123hash",
                "reference_kind": "rtk",
                "reference_source_dataset": "navigators_india",
                "session_id": "sess_02",
            },
            "case": {"kind": "outage", "duration_s": 30},
            "regions": [
                {
                    "phase": "outage",
                    "environment": "urban",
                    "metrics": {"position_rmse_m": 2.5, "velocity_rmse_mps": 0.4},
                    "support": {
                        "position_rmse_m": {"sample_indices": [1, 2, 3]},
                        "velocity_rmse_mps": {"sample_indices": [1, 2, 3]},
                    },
                }
            ],
        },
    ]

    agg = aggregate(results)
    assert len(agg) == 1
    assert agg[0]["metrics"]["position_rmse_m"]["mean"] == pytest.approx(2.3, abs=0.01)


def test_provenance_meta_extraction():
    prov = provenance()
    assert "benchmark_version" in prov
    assert "python" in prov
    assert "platform" in prov
    assert "code_sha256" in prov
