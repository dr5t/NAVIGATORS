import os
import shutil
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

from src.models.pedestrian_training import (
    PedestrianTrainingConfig,
    PedestrianEvaluationResult,
    PedestrianIMUDataset,
    PedestrianTrainingPipeline,
)
from src.models.tcn_model import TCNVelocityEstimator
from src.data.sensor_normalization import DatasetNormalizationStats


def create_dummy_pedestrian_session(
    session_id: str,
    subject_id: str,
    source_dataset: str,
    device_id: str = "pixel_6",
    phone_placement: str = "handheld",
    walking_speed: str = "normal",
    motion_type: str = "walking",
    n_samples: int = 300,
):
    t = np.linspace(0, 30, n_samples)
    acc = np.random.randn(n_samples, 3).astype(np.float32)
    acc[:, 2] += 9.81
    gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.2
    vn = np.sin(t * 0.2).astype(np.float32) * 1.4
    ve = np.cos(t * 0.2).astype(np.float32) * 1.1
    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "user_id": subject_id,
        "device_id": device_id,
        "source_dataset": source_dataset,
        "phone_placement": phone_placement,
        "walking_speed": walking_speed,
        "motion_type": motion_type,
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


def test_pedestrian_training_config():
    config = PedestrianTrainingConfig()
    assert config.window_size == 200
    assert config.sample_rate_hz == 10.0
    assert config.input_channels == 6
    assert config.output_dim == 2
    d = config.to_dict()
    assert d["seed"] == 42
    assert d["checkpoint_dir"] == "checkpoints/pedestrian_exp"


def test_pedestrian_subject_split_separation():
    pipeline = PedestrianTrainingPipeline()
    sessions = [
        create_dummy_pedestrian_session("s1", "subj_A", "RoNIN"),
        create_dummy_pedestrian_session("s2", "subj_A", "RoNIN"),
        create_dummy_pedestrian_session("s3", "subj_B", "OxIOD"),
        create_dummy_pedestrian_session("s4", "subj_C", "Navigators_India_Pedestrian"),
        create_dummy_pedestrian_session("s5", "subj_D", "Navigators_India_Pedestrian"),
    ]
    train_s, val_s, test_s = pipeline.prepare_splits(sessions, seed=42)
    train_subjects = {s["subject_id"] for s in train_s}
    val_subjects = {s["subject_id"] for s in val_s}
    test_subjects = {s["subject_id"] for s in test_s}

    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0


def test_pedestrian_target_formulations():
    pipeline = PedestrianTrainingPipeline()
    sess = create_dummy_pedestrian_session("s1", "subj_A", "RoNIN", n_samples=300)
    stats = pipeline.normalizer.fit([sess])

    win_v1, tgt_v1, meta_v1 = pipeline.extract_windows_and_targets([sess], stats, formulation="velocity_2d")
    assert len(win_v1) > 0
    assert win_v1.shape[1] == 200
    assert win_v1.shape[2] == 6
    assert tgt_v1.shape[1] == 2

    win_v2, tgt_v2, meta_v2 = pipeline.extract_windows_and_targets([sess], stats, formulation="displacement_local")
    assert len(win_v2) > 0
    assert tgt_v2.shape[1] == 2


def test_pedestrian_imu_dataset_torch():
    win = np.random.randn(8, 200, 6).astype(np.float32)
    tgt = np.random.randn(8, 2).astype(np.float32)
    ds = PedestrianIMUDataset(win, tgt)
    assert len(ds) == 8
    x, y = ds[0]
    assert x.shape == (200, 6)
    assert y.shape == (2,)


def test_pedestrian_training_and_formulation_selection():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = PedestrianTrainingConfig(
            epochs=2,
            batch_size=8,
            checkpoint_dir=tmp_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        pipeline = PedestrianTrainingPipeline(config=config)

        sessions = [
            create_dummy_pedestrian_session("s1", "subj_1", "RoNIN", n_samples=250),
            create_dummy_pedestrian_session("s2", "subj_2", "OxIOD", n_samples=250),
            create_dummy_pedestrian_session("s3", "subj_3", "Navigators_India_Pedestrian", n_samples=250),
            create_dummy_pedestrian_session("s4", "subj_4", "Navigators_India_Pedestrian", n_samples=250),
            create_dummy_pedestrian_session("s5", "subj_5", "Navigators_India_Pedestrian", n_samples=250),
        ]

        model, stats, result = pipeline.train_and_select_pedestrian_model(sessions)
        assert isinstance(model, TCNVelocityEstimator)
        assert isinstance(stats, DatasetNormalizationStats)
        assert isinstance(result, PedestrianEvaluationResult)

        assert result.selected_formulation in ["velocity_2d", "displacement_local"]
        assert result.velocity_mae_mps >= 0.0
        assert result.velocity_rmse_mps >= 0.0
        assert result.trajectory_error_m >= 0.0
        assert result.ate_rmse_m >= 0.0
        assert result.fde_m >= 0.0
        assert result.heading_error_deg >= 0.0
        assert "velocity_2d" in result.formulation_comparison
        assert "displacement_local" in result.formulation_comparison

        ckpt_path = Path(tmp_dir) / "pedestrian_tcn_exp.pt"
        assert ckpt_path.exists()
    finally:
        shutil.rmtree(tmp_dir)
