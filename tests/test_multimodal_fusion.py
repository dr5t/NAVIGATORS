import numpy as np
import torch
import pytest

from src.navigation.multimodal_fusion import (
    MultiModalFusionConfig,
    MultiModalFusionState,
    MultiModalAdaptiveFusionEngine,
)
from src.models.motion_classifier import MotionClass
from src.models.tcn_model import TCNVelocityEstimator
from src.navigation.ekf import ExtendedKalmanFilter, NavigationMode


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


def test_multimodal_fusion_config():
    config = MultiModalFusionConfig()
    assert config.confidence_threshold == 0.65
    assert config.vehicle_base_std_mps == 0.8
    assert config.pedestrian_base_std_mps == 0.3
    d = config.to_dict()
    assert d["unknown_std_mps"] == 100.0


def test_gnss_independence_from_internet():
    veh_model = TCNVelocityEstimator(input_channels=6, output_dim=2)
    ped_model = TCNVelocityEstimator(input_channels=6, output_dim=2)
    engine = MultiModalAdaptiveFusionEngine(vehicle_model=veh_model, pedestrian_model=ped_model)

    win = create_synthetic_imu_window("vehicle", 200)
    gnss_fix = {"latitude": 28.6139, "longitude": 77.2090, "accuracy": 5.0, "velocity_north": 10.0, "velocity_east": 0.0}

    state_online = engine.process_sensor_frame(win, timestamp=1.0, gnss_fix=gnss_fix, internet_connected=True)
    assert state_online.gnss_status == "HEALTHY"
    assert state_online.is_dead_reckoning is False

    state_offline = engine.process_sensor_frame(win, timestamp=2.0, gnss_fix=gnss_fix, internet_connected=False)
    assert state_offline.gnss_status == "HEALTHY"
    assert state_offline.is_dead_reckoning is False
    assert state_offline.internet_connected is False


def test_internet_loss_never_triggers_dead_reckoning():
    engine = MultiModalAdaptiveFusionEngine()
    win = create_synthetic_imu_window("vehicle", 200)
    gnss_fix = {"latitude": 28.6139, "longitude": 77.2090, "accuracy": 4.0}

    state_1 = engine.process_sensor_frame(win, timestamp=1.0, gnss_fix=gnss_fix, internet_connected=True)
    assert state_1.is_dead_reckoning is False

    state_2 = engine.process_sensor_frame(win, timestamp=2.0, gnss_fix=gnss_fix, internet_connected=False)
    assert state_2.is_dead_reckoning is False

    state_outage = engine.process_sensor_frame(win, timestamp=3.0, gnss_fix=None, internet_connected=True)
    assert state_outage.is_dead_reckoning is True
    assert state_outage.gnss_status == "OUTAGE"


def test_covariance_and_innovation_gating():
    config = MultiModalFusionConfig(innovation_gate_sigma=1.0, vehicle_base_std_mps=0.1)
    engine = MultiModalAdaptiveFusionEngine(config=config)

    win = create_synthetic_imu_window("vehicle", 200)
    state = engine.process_sensor_frame(win, timestamp=1.0)
    assert state.R_motion_cov.shape == (2, 2)
    assert state.R_motion_cov[0, 0] > 0.0


def test_full_replay_multimodal_adaptive_fusion():
    veh_model = TCNVelocityEstimator(input_channels=6, output_dim=2)
    ped_model = TCNVelocityEstimator(input_channels=6, output_dim=2)
    engine = MultiModalAdaptiveFusionEngine(vehicle_model=veh_model, pedestrian_model=ped_model)

    timeline = [
        ("stationary", {"latitude": 28.6139, "longitude": 77.2090, "accuracy": 3.0}),
        ("vehicle", {"latitude": 28.6140, "longitude": 77.2091, "accuracy": 5.0}),
        ("vehicle", None),
        ("pedestrian", None),
        ("stationary", {"latitude": 28.6145, "longitude": 77.2095, "accuracy": 4.0}),
    ]

    t = 1.0
    for motion_type, gnss_fix in timeline:
        win = create_synthetic_imu_window(motion_type, 200)
        state = engine.process_sensor_frame(win, timestamp=t, gnss_fix=gnss_fix, internet_connected=False)
        assert isinstance(state, MultiModalFusionState)
        assert state.position_enu.shape == (3,)
        assert state.velocity_enu.shape == (3,)
        t += 1.0
