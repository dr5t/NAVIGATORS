import numpy as np
import torch
import pytest

from src.navigation.confidence_aware_navigation import (
    ConfidenceAwareConfig,
    ConfidenceAwareStepResult,
    ConfidenceAwareNavigationEngine,
    NavigationStateMode,
    VehicleDomainResult,
    PedestrianDomainResult,
    DomainSeparatedEvaluationSuite,
)
from src.navigation.interfaces import GNSSTrustState
from src.models.motion_classifier import MotionClass


def create_synthetic_imu_window(motion_type: str = "stationary", n_samples: int = 200):
    t = np.linspace(0, 20, n_samples)
    if motion_type == "stationary":
        acc = np.random.randn(n_samples, 3).astype(np.float32) * 0.01
        acc[:, 2] += 9.81
        gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.005
    elif motion_type == "pedestrian":
        acc = np.random.randn(n_samples, 3).astype(np.float32) * 1.5
        acc[:, 0] += np.sin(t * 3.0).astype(np.float32) * 2.0
        acc[:, 2] += 9.81
        gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.4
    else:
        acc = np.random.randn(n_samples, 3).astype(np.float32) * 0.4
        acc[:, 1] += np.sin(t * 0.5).astype(np.float32) * 1.0
        acc[:, 2] += 9.81
        gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.08

    return np.column_stack([acc, gyro])


def test_confidence_aware_config():
    cfg = ConfidenceAwareConfig()
    assert cfg.max_position_jump_m == 30.0
    assert cfg.max_velocity_inconsistency_mps == 3.0
    assert cfg.max_heading_inconsistency_deg == 30.0
    assert cfg.recovery_min_good_fixes == 3
    d = cfg.to_dict()
    assert d["confidence_threshold"] == 0.65


def test_gnss_available_vs_trustworthy_distinction():
    engine = ConfidenceAwareNavigationEngine()
    current_vel = np.array([10.0, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, 0.0])

    good_fix = {"latitude": 28.6139, "longitude": 77.2090, "accuracy": 4.0, "velocity_north": 10.0, "velocity_east": 0.0}
    avail, trust, state, anomalies = engine.evaluate_gnss_trustworthiness(good_fix, current_vel, current_pos)
    assert avail is True
    assert trust is True
    assert state == GNSSTrustState.TRUSTED
    assert len(anomalies) == 0

    bad_fix = {"latitude": 28.6250, "longitude": 77.2200, "accuracy": 45.0, "velocity_north": 0.0, "velocity_east": 0.0}
    avail2, trust2, state2, anomalies2 = engine.evaluate_gnss_trustworthiness(bad_fix, current_vel, current_pos)
    assert avail2 is True
    assert trust2 is False
    assert state2 in [GNSSTrustState.SUSPICIOUS, GNSSTrustState.UNUSABLE]
    assert len(anomalies2) > 0


def test_anti_jump_recovery_behavior():
    engine = ConfidenceAwareNavigationEngine(config=ConfidenceAwareConfig(recovery_min_good_fixes=3))
    win = create_synthetic_imu_window("vehicle", 200)

    res_dr = engine.process_step(win, timestamp=1.0, gnss_fix=None)
    assert res_dr.navigation_mode == NavigationStateMode.DR

    good_fix = {"latitude": 28.6139, "longitude": 77.2090, "accuracy": 5.0, "velocity_north": 2.0, "velocity_east": 0.0}

    res_rec1 = engine.process_step(win, timestamp=2.0, gnss_fix=good_fix)
    assert res_rec1.navigation_mode == NavigationStateMode.REACQUIRING

    res_rec2 = engine.process_step(win, timestamp=3.0, gnss_fix=good_fix)
    assert res_rec2.navigation_mode == NavigationStateMode.REACQUIRING

    res_rec3 = engine.process_step(win, timestamp=4.0, gnss_fix=good_fix)
    assert res_rec3.navigation_mode == NavigationStateMode.RECOVERED


def test_domain_separated_evaluation_suite():
    preds_v = np.array([[10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    targets_v = np.array([[10.0, 0.2], [10.0, 0.2]], dtype=np.float32)
    veh_res = DomainSeparatedEvaluationSuite.evaluate_vehicle_domain(preds_v, targets_v, [1.5], 95.0)
    assert isinstance(veh_res, VehicleDomainResult)
    assert veh_res.domain == "VEHICLE"
    assert veh_res.velocity_mae_mps >= 0.0
    assert veh_res.map_matching_accuracy_pct == 95.0

    preds_p = np.array([[1.2, 0.0], [1.2, 0.0]], dtype=np.float32)
    targets_p = np.array([[1.2, 0.1], [1.2, 0.1]], dtype=np.float32)
    ped_res = DomainSeparatedEvaluationSuite.evaluate_pedestrian_domain(preds_p, targets_p)
    assert isinstance(ped_res, PedestrianDomainResult)
    assert ped_res.domain == "PEDESTRIAN"
    assert ped_res.velocity_mae_mps >= 0.0
    assert ped_res.ate_rmse_m >= 0.0
