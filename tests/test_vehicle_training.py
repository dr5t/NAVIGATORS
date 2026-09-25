import os
import shutil
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

from src.models.vehicle_training import (
    VehicleTrainingConfig,
    VehicleEvaluationResult,
    VehicleIMUDataset,
    VehicleTrainingPipeline,
)
from src.models.tcn_model import TCNVelocityEstimator
from src.data.sensor_normalization import DatasetNormalizationStats


def create_dummy_session(session_id: str, vehicle_id: str, source_dataset: str, n_samples: int = 300):
    t = np.linspace(0, 30, n_samples)
    acc = np.random.randn(n_samples, 3).astype(np.float32)
    acc[:, 2] += 9.81
    gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.1
    vn = np.sin(t * 0.1).astype(np.float32) * 5.0
    ve = np.cos(t * 0.1).astype(np.float32) * 3.0
    return {
        "session_id": session_id,
        "vehicle_id": vehicle_id,
        "source_dataset": source_dataset,
        "vehicle_type": "car",
        "road_type": "urban",
        "device_id": "phone_1",
        "timestamp": t,
        "accelerometer_x": acc[:, 0],
        "accelerometer_y": acc[:, 1],
        "accelerometer_z": acc[:, 2],
        "gyroscope_x": gyro[:, 0],
        "gyroscope_y": gyro[:, 1],
        "gyroscope_z": gyro[:, 2],
        "velocity_north": vn,
        "velocity_east": ve,
    }


def test_vehicle_training_config():
    config = VehicleTrainingConfig()
    assert config.window_size == 200
    assert config.sample_rate_hz == 10.0
    assert config.input_channels == 6
    assert config.output_dim == 2
    d = config.to_dict()
    assert d["seed"] == 42
    assert d["checkpoint_dir"] == "checkpoints/vehicle_exp"


def test_vehicle_split_separation():
    pipeline = VehicleTrainingPipeline()
    sessions = [
        create_dummy_session("s1", "veh_A", "IO-VNBD"),
        create_dummy_session("s2", "veh_A", "IO-VNBD"),
        create_dummy_session("s3", "veh_B", "IO-VNBD"),
        create_dummy_session("s4", "veh_C", "Navigators_India_Vehicle"),
        create_dummy_session("s5", "veh_D", "Navigators_India_Vehicle"),
    ]
    train_s, val_s, test_s = pipeline.prepare_splits(sessions, seed=42)
    train_vehicles = {s["vehicle_id"] for s in train_s}
    val_vehicles = {s["vehicle_id"] for s in val_s}
    test_vehicles = {s["vehicle_id"] for s in test_s}

    assert len(train_vehicles.intersection(val_vehicles)) == 0
    assert len(train_vehicles.intersection(test_vehicles)) == 0
    assert len(val_vehicles.intersection(test_vehicles)) == 0


def test_dataset_window_extraction():
    pipeline = VehicleTrainingPipeline()
    sess = create_dummy_session("s1", "veh_A", "IO-VNBD", n_samples=300)
    stats = pipeline.normalizer.fit([sess])
    windows, targets, meta = pipeline.extract_windows_and_targets([sess], stats, window_size=200, stride=50)
    assert len(windows) > 0
    assert windows.shape[1] == 200
    assert windows.shape[2] == 6
    assert targets.shape[1] == 2
    assert len(meta) == len(windows)


def test_vehicle_imu_dataset_torch():
    win = np.random.randn(10, 200, 6).astype(np.float32)
    tgt = np.random.randn(10, 2).astype(np.float32)
    ds = VehicleIMUDataset(win, tgt)
    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape == (200, 6)
    assert y.shape == (2,)


def test_vehicle_training_and_evaluation():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = VehicleTrainingConfig(
            epochs=2,
            batch_size=8,
            checkpoint_dir=tmp_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        pipeline = VehicleTrainingPipeline(config=config)
        sessions = [
            create_dummy_session("s1", "veh_1", "IO-VNBD", n_samples=250),
            create_dummy_session("s2", "veh_2", "IO-VNBD", n_samples=250),
            create_dummy_session("s3", "veh_3", "Navigators_India_Vehicle", n_samples=250),
            create_dummy_session("s4", "veh_4", "Navigators_India_Vehicle", n_samples=250),
            create_dummy_session("s5", "veh_5", "Navigators_India_Vehicle", n_samples=250),
        ]
        model, stats, eval_res = pipeline.train_experimental_model(sessions)
        assert isinstance(model, TCNVelocityEstimator)
        assert isinstance(stats, DatasetNormalizationStats)
        assert isinstance(eval_res, VehicleEvaluationResult)
        assert eval_res.velocity_mae_mps >= 0.0
        assert eval_res.velocity_rmse_mps >= 0.0
        assert eval_res.speed_mae_mps >= 0.0
        assert eval_res.speed_rmse_mps >= 0.0
        assert eval_res.north_velocity_error_mps >= 0.0
        assert eval_res.east_velocity_error_mps >= 0.0
        assert eval_res.parameter_count > 0
        assert eval_res.model_size_bytes > 0
        ckpt_path = Path(tmp_dir) / "vehicle_tcn_exp.pt"
        assert ckpt_path.exists()
    finally:
        shutil.rmtree(tmp_dir)
