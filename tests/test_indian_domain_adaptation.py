import os
import shutil
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

from src.models.indian_domain_adaptation import (
    DomainAdaptationConfig,
    ConditionMetrics,
    AdaptationIMUDataset,
    IndianDomainAdaptationPipeline,
)


def create_dummy_session(
    session_id: str,
    vehicle_id: str,
    source_dataset: str,
    device_id: str = "pixel_6",
    route_id: str = "r1",
    road_type: str = "urban",
    traffic_density: str = "medium",
    phone_mount_position: str = "dashboard",
    vehicle_type: str = "car",
    n_samples: int = 300,
):
    t = np.linspace(0, 30, n_samples)
    acc = np.random.randn(n_samples, 3).astype(np.float32)
    acc[:, 2] += 9.81
    gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.1
    vn = np.sin(t * 0.1).astype(np.float32) * 5.0
    ve = np.cos(t * 0.1).astype(np.float32) * 3.0
    return {
        "session_id": session_id,
        "vehicle_id": vehicle_id,
        "device_id": device_id,
        "route_id": route_id,
        "source_dataset": source_dataset,
        "vehicle_type": vehicle_type,
        "road_type": road_type,
        "traffic_density": traffic_density,
        "phone_mount_position": phone_mount_position,
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


def test_domain_adaptation_config():
    cfg = DomainAdaptationConfig()
    assert cfg.window_size == 200
    assert cfg.sample_rate_hz == 10.0
    assert cfg.pretrain_lr == 0.001
    assert cfg.finetune_lr == 0.0001
    d = cfg.to_dict()
    assert d["checkpoint_dir"] == "checkpoints/domain_adaptation_exp"
    assert d["seed"] == 42


def test_dataset_partition_no_leakage():
    pipeline = IndianDomainAdaptationPipeline()
    sessions = [
        create_dummy_session("s1", "v1", "Navigators_India_Vehicle", "d1", "r1"),
        create_dummy_session("s2", "v1", "Navigators_India_Vehicle", "d1", "r1"),
        create_dummy_session("s3", "v2", "Navigators_India_Vehicle", "d2", "r2"),
        create_dummy_session("s4", "v3", "Navigators_India_Vehicle", "d3", "r3"),
        create_dummy_session("s5", "v4", "Navigators_India_Vehicle", "d4", "r4"),
    ]
    tr_s, va_s, te_s = pipeline.partition_dataset(sessions, seed=42)
    tr_keys = {f"{s['vehicle_id']}_{s['session_id']}_{s['device_id']}_{s['route_id']}" for s in tr_s}
    va_keys = {f"{s['vehicle_id']}_{s['session_id']}_{s['device_id']}_{s['route_id']}" for s in va_s}
    te_keys = {f"{s['vehicle_id']}_{s['session_id']}_{s['device_id']}_{s['route_id']}" for s in te_s}

    assert len(tr_keys.intersection(va_keys)) == 0
    assert len(tr_keys.intersection(te_keys)) == 0
    assert len(va_keys.intersection(te_keys)) == 0


def test_trajectory_and_heading_error_computation():
    pipeline = IndianDomainAdaptationPipeline()
    preds = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    targets = np.array([[5.0, 0.1], [5.0, 0.2], [5.0, 0.3]], dtype=np.float32)

    traj_err, ct_err, head_err = pipeline.compute_trajectory_and_heading_errors(preds, targets)
    assert traj_err >= 0.0
    assert ct_err >= 0.0
    assert head_err >= 0.0


def test_adaptation_imu_dataset_torch():
    win = np.random.randn(5, 200, 6).astype(np.float32)
    tgt = np.random.randn(5, 2).astype(np.float32)
    ds = AdaptationIMUDataset(win, tgt)
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (200, 6)
    assert y.shape == (2,)


def test_run_domain_adaptation_experiments():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = DomainAdaptationConfig(
            epochs_pretrain=1,
            epochs_finetune=1,
            batch_size=8,
            checkpoint_dir=tmp_dir,
        )
        pipeline = IndianDomainAdaptationPipeline(config=config)

        io_sessions = [
            create_dummy_session("io1", "veh_io1", "IO-VNBD", "d_io1", "r_io1", n_samples=250),
            create_dummy_session("io2", "veh_io2", "IO-VNBD", "d_io2", "r_io2", n_samples=250),
            create_dummy_session("io3", "veh_io3", "IO-VNBD", "d_io3", "r_io3", n_samples=250),
        ]

        ind_sessions = [
            create_dummy_session("ind1", "veh_ind1", "Navigators_India_Vehicle", "d_ind1", "r_ind1", n_samples=250),
            create_dummy_session("ind2", "veh_ind2", "Navigators_India_Vehicle", "d_ind2", "r_ind2", n_samples=250),
            create_dummy_session("ind3", "veh_ind3", "Navigators_India_Vehicle", "d_ind3", "r_ind3", n_samples=250),
            create_dummy_session("ind4", "veh_ind4", "Navigators_India_Vehicle", "d_ind4", "r_ind4", n_samples=250),
        ]

        results = pipeline.run_domain_adaptation_experiments(io_sessions, ind_sessions)

        assert "A" in results
        assert "B" in results
        assert "C" in results
        assert "D" in results

        for cond_key in ["A", "B", "C", "D"]:
            m = results[cond_key]
            assert isinstance(m, ConditionMetrics)
            assert m.velocity_mae_mps >= 0.0
            assert m.velocity_rmse_mps >= 0.0
            assert m.speed_mae_mps >= 0.0
            assert m.speed_rmse_mps >= 0.0
            assert m.trajectory_error_m >= 0.0
            assert m.cross_track_error_m >= 0.0
            assert m.heading_error_deg >= 0.0

        ckpt_path = Path(tmp_dir) / "domain_adaptation_checkpoints.pt"
        assert ckpt_path.exists()
    finally:
        shutil.rmtree(tmp_dir)
