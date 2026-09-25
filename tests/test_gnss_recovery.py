import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    GNSSRecoveryState,
    GNSSRecoveryPlan,
    GNSSTrustState,
    GNSSTrustMetric,
    IGNSSTrustEngine,
)
from navigation.map_matching import RoadNetwork, RoadSegment
from navigation.gnss_recovery import (
    GNSSRecoveryConfig,
    GNSSRecoveryMetrics,
    ValidatedGNSSRecoveryManager,
)


class MockTrustEngine(IGNSSTrustEngine):
    def __init__(self, default_score: float = 0.95, default_state: GNSSTrustState = GNSSTrustState.TRUSTED):
        self.default_score = default_score
        self.default_state = default_state
        self.call_count = 0

    def evaluate_trust(
        self,
        gnss_data=None,
        imu_accel=None,
        dt=0.1,
        current_position=None,
        current_velocity=None,
        current_heading=None,
        position_uncertainty=None,
        ai_velocity=None,
        road_distance=None,
    ) -> GNSSTrustMetric:
        self.call_count += 1
        score = float(gnss_data.get("trust_score", self.default_score)) if gnss_data else self.default_score
        state = self.default_state if score >= 0.50 else GNSSTrustState.UNUSABLE
        return GNSSTrustMetric(
            trust_score=score,
            is_trusted=(score >= 0.50),
            position_variance_scale=1.0 if score >= 0.50 else 10.0,
            velocity_variance_scale=1.0 if score >= 0.50 else 10.0,
            state=state,
            diagnostics={"mock": score},
        )

    def reset(self) -> None:
        self.call_count = 0


def test_normal_gnss_return():
    trust_engine = MockTrustEngine(default_score=0.95)
    config = GNSSRecoveryConfig(
        max_step_correction_m=2.0,
        convergence_residual_m=2.0,
        convergence_step_norm_m=0.35,
        min_valid_fixes_to_recover=3,
        consecutive_converged_steps_required=3,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=trust_engine)

    current_estimate = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    true_gnss = np.array([16.0, 10.0], dtype=np.float64)

    observed_discontinuities = []
    t = 100.0

    for step in range(12):
        gnss_data = {
            "position": true_gnss.tolist(),
            "accuracy": 2.5,
            "speed": 0.0,
            "heading": 0.0,
        }
        plan = manager.process_gnss_return(
            gnss_data=gnss_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=4.0,
            timestamp=t,
        )
        observed_discontinuities.append(plan.step_correction_norm_m)
        current_estimate = plan.corrected_state
        t += 0.1

        if step < 2:
            assert plan.state in (GNSSRecoveryState.REACQUIRING, GNSSRecoveryState.VALIDATING)
            assert plan.step_correction_norm_m == 0.0
        elif step >= 3 and not plan.is_converged:
            assert plan.state == GNSSRecoveryState.RECOVERING
            assert plan.step_correction_norm_m <= config.max_step_correction_m

    assert max(observed_discontinuities) <= config.max_step_correction_m
    assert manager.is_converged()
    assert manager.state == GNSSRecoveryState.NORMAL
    final_err = float(np.linalg.norm(current_estimate[:2] - true_gnss))
    assert final_err < 0.5


def test_bad_first_fix():
    trust_engine = MockTrustEngine()
    config = GNSSRecoveryConfig(
        max_step_correction_m=2.0,
        min_valid_fixes_to_recover=3,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=trust_engine)

    current_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    bad_gnss = {
        "position": [500.0, 500.0],
        "accuracy": 85.0,
        "trust_score": 0.10,
    }
    plan_bad = manager.process_gnss_return(
        gnss_data=bad_gnss,
        current_estimate=current_estimate,
        dt=0.1,
        dr_position=current_estimate[:2],
        dr_uncertainty=3.0,
        timestamp=10.0,
    )

    assert not plan_bad.consistency_passed
    assert plan_bad.step_correction_norm_m == 0.0
    assert plan_bad.state == GNSSRecoveryState.REACQUIRING
    assert np.allclose(plan_bad.corrected_state[:2], current_estimate[:2])

    good_pos = np.array([3.0, 0.0], dtype=np.float64)
    t = 10.1
    for i in range(8):
        good_gnss = {
            "position": good_pos.tolist(),
            "accuracy": 2.0,
            "trust_score": 0.95,
        }
        plan_good = manager.process_gnss_return(
            gnss_data=good_gnss,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=3.0,
            timestamp=t,
        )
        current_estimate = plan_good.corrected_state
        t += 0.1

    assert manager.is_converged()
    assert np.linalg.norm(current_estimate[:2] - good_pos) < 0.5


def test_large_position_jump():
    trust_engine = MockTrustEngine(default_score=0.90)
    config = GNSSRecoveryConfig(
        max_step_correction_m=2.0,
        min_valid_fixes_to_recover=2,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=trust_engine)

    current_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    distant_gnss = np.array([24.0, 0.0], dtype=np.float64)

    step_norms = []
    t = 0.0
    for _ in range(18):
        gnss_data = {
            "position": distant_gnss.tolist(),
            "accuracy": 3.0,
            "trust_score": 0.90,
        }
        plan = manager.process_gnss_return(
            gnss_data=gnss_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=20.0,
            timestamp=t,
        )
        step_norms.append(plan.step_correction_norm_m)
        current_estimate = plan.corrected_state
        t += 0.1

    assert max(step_norms) <= 2.0001
    assert manager.is_converged()
    assert np.linalg.norm(current_estimate[:2] - distant_gnss) < 0.5


def test_delayed_stable_gnss():
    trust_engine = MockTrustEngine()
    config = GNSSRecoveryConfig(
        min_valid_fixes_to_recover=3,
        min_trust_score=0.60,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=trust_engine)
    current_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    t = 0.0
    for _ in range(5):
        unstable_data = {
            "position": [5.0, 2.0],
            "accuracy": 45.0,
            "trust_score": 0.35,
        }
        plan = manager.process_gnss_return(
            gnss_data=unstable_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=5.0,
            timestamp=t,
        )
        assert not plan.consistency_passed
        assert plan.step_correction_norm_m == 0.0
        assert manager.state == GNSSRecoveryState.REACQUIRING
        t += 0.1

    stable_pos = np.array([4.0, 1.0], dtype=np.float64)
    for _ in range(10):
        stable_data = {
            "position": stable_pos.tolist(),
            "accuracy": 2.2,
            "trust_score": 0.92,
        }
        plan = manager.process_gnss_return(
            gnss_data=stable_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=5.0,
            timestamp=t,
        )
        current_estimate = plan.corrected_state
        t += 0.1

    assert manager.is_converged()
    assert np.linalg.norm(current_estimate[:2] - stable_pos) < 0.5


def test_parallel_road_gnss_recovery():
    road_network = RoadNetwork()
    seg_main = RoadSegment(
        id="main_arterial",
        start=np.array([0.0, 0.0]),
        end=np.array([0.0, 1000.0]),
        name="Main Arterial",
        speed_limit=60.0,
        one_way=True,
    )
    seg_parallel = RoadSegment(
        id="parallel_service",
        start=np.array([15.0, 0.0]),
        end=np.array([15.0, 1000.0]),
        name="Service Road",
        speed_limit=30.0,
        one_way=True,
    )
    road_network.add_segment(seg_main)
    road_network.add_segment(seg_parallel)

    config = GNSSRecoveryConfig(
        max_parallel_road_divergence_m=6.0,
    )
    manager = ValidatedGNSSRecoveryManager(
        config=config,
        trust_engine=MockTrustEngine(),
        road_network=road_network,
    )

    current_estimate = np.array([0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    parallel_gnss_data = {
        "position": [15.2, 50.5],
        "accuracy": 3.0,
        "trust_score": 0.85,
    }
    plan = manager.process_gnss_return(
        gnss_data=parallel_gnss_data,
        current_estimate=current_estimate,
        dt=0.1,
        dr_position=current_estimate[:2],
        dr_uncertainty=2.0,
        road_segment=seg_main,
        timestamp=10.0,
    )

    assert not plan.consistency_passed
    assert plan.step_correction_norm_m == 0.0
    assert "PARALLEL_ROAD_MISMATCH" in plan.diagnostics.get("reasons", [])


def test_long_dr_outage():
    config = GNSSRecoveryConfig(
        max_step_correction_m=2.0,
        max_norm_innovation=4.0,
        min_valid_fixes_to_recover=2,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=MockTrustEngine())

    current_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    true_gnss = np.array([30.0, 0.0], dtype=np.float64)

    t = 60.0
    for _ in range(25):
        gnss_data = {
            "position": true_gnss.tolist(),
            "accuracy": 2.5,
            "trust_score": 0.95,
        }
        plan = manager.process_gnss_return(
            gnss_data=gnss_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=18.0,
            timestamp=t,
        )
        current_estimate = plan.corrected_state
        t += 0.1

    assert manager.is_converged()
    assert np.linalg.norm(current_estimate[:2] - true_gnss) < 0.5


def test_short_dr_outage():
    config = GNSSRecoveryConfig(
        max_step_correction_m=2.0,
        min_valid_fixes_to_recover=2,
    )
    manager = ValidatedGNSSRecoveryManager(config=config, trust_engine=MockTrustEngine())

    current_estimate = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    true_gnss = np.array([1.2, 0.5], dtype=np.float64)

    t = 3.0
    steps = 0
    for _ in range(10):
        gnss_data = {
            "position": true_gnss.tolist(),
            "accuracy": 1.5,
            "trust_score": 0.98,
        }
        plan = manager.process_gnss_return(
            gnss_data=gnss_data,
            current_estimate=current_estimate,
            dt=0.1,
            dr_position=current_estimate[:2],
            dr_uncertainty=1.5,
            timestamp=t,
        )
        current_estimate = plan.corrected_state
        steps += 1
        t += 0.1
        if manager.is_converged():
            break

    assert manager.is_converged()
    assert steps <= 7
    assert np.linalg.norm(current_estimate[:2] - true_gnss) < 0.3


def test_recovery_metrics_summary():
    metrics = GNSSRecoveryMetrics()
    metrics.add_episode(recovery_time_s=1.2, max_discontinuity_m=1.85, final_position_error_m=0.32, is_false_recovery=False)
    metrics.add_episode(recovery_time_s=1.5, max_discontinuity_m=1.92, final_position_error_m=0.41, is_false_recovery=False)
    metrics.add_episode(recovery_time_s=2.1, max_discontinuity_m=2.00, final_position_error_m=0.38, is_false_recovery=False)

    summary = metrics.compute_summary()
    assert summary["total_episodes"] == 3.0
    assert summary["mean_recovery_time_s"] == 1.6
    assert summary["max_position_discontinuity_m"] == 2.0
    assert summary["false_recovery_count"] == 0.0
    assert summary["false_recovery_rate_pct"] == 0.0
