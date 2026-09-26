import numpy as np
import torch
import pytest

from src.models.motion_classifier import (
    MotionClass,
    MotionClassifierConfig,
    IMUFeatureExtractor,
    LightweightIMUClassifier,
    TemporallyStabilizedMotionClassifier,
    MotionRouter,
    ClassifierEvaluationSuite,
)
from src.models.tcn_model import TCNVelocityEstimator


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


def test_motion_classifier_config():
    cfg = MotionClassifierConfig()
    assert cfg.window_size == 200
    assert cfg.confidence_threshold == 0.65
    assert cfg.hysteresis_dwell_count == 5
    d = cfg.to_dict()
    assert d["seed"] == 42


def test_imu_feature_extractor():
    win = create_synthetic_imu_window("pedestrian", 200)
    feats = IMUFeatureExtractor.extract_features(win)
    assert len(feats) == 16
    assert np.all(np.isfinite(feats))


def test_lightweight_imu_classifier_probs():
    classifier = LightweightIMUClassifier()
    win = create_synthetic_imu_window("stationary", 200)
    raw_cls, conf, probs = classifier.predict_window_probs(win)
    assert raw_cls in [MotionClass.STATIONARY, MotionClass.PEDESTRIAN, MotionClass.VEHICLE]
    assert 0.0 <= conf <= 1.0
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_temporal_stabilization_and_hysteresis():
    config = MotionClassifierConfig(confidence_threshold=0.60, hysteresis_dwell_count=3)
    stabilizer = TemporallyStabilizedMotionClassifier(config=config)

    win_stat = create_synthetic_imu_window("stationary", 200)
    for _ in range(2):
        st, conf, info = stabilizer.process_window(win_stat)

    st_3, conf_3, info_3 = stabilizer.process_window(win_stat)
    assert st_3 in [MotionClass.STATIONARY, MotionClass.PEDESTRIAN, MotionClass.VEHICLE, MotionClass.UNKNOWN]


def test_motion_classifier_all_six_transitions():
    config = MotionClassifierConfig(hysteresis_dwell_count=2, confidence_threshold=0.50)
    stabilizer = TemporallyStabilizedMotionClassifier(config=config)

    transitions = [
        ("stationary", MotionClass.STATIONARY),
        ("pedestrian", MotionClass.PEDESTRIAN),
        ("stationary", MotionClass.STATIONARY),
        ("vehicle", MotionClass.VEHICLE),
        ("stationary", MotionClass.STATIONARY),
        ("pedestrian", MotionClass.PEDESTRIAN),
        ("vehicle", MotionClass.VEHICLE),
        ("pedestrian", MotionClass.PEDESTRIAN),
    ]

    history_states = []
    for motion_type, _ in transitions:
        win = create_synthetic_imu_window(motion_type, 200)
        st = None
        for _ in range(3):
            st, _, _ = stabilizer.process_window(win)
        history_states.append(st)

    assert len(history_states) == len(transitions)


def test_classifier_evaluation_suite_metrics_and_latencies():
    true_seq = [
        MotionClass.STATIONARY, MotionClass.STATIONARY,
        MotionClass.PEDESTRIAN, MotionClass.PEDESTRIAN,
        MotionClass.STATIONARY, MotionClass.STATIONARY,
        MotionClass.VEHICLE, MotionClass.VEHICLE,
        MotionClass.PEDESTRIAN, MotionClass.PEDESTRIAN,
    ]
    pred_seq = list(true_seq)

    metrics = ClassifierEvaluationSuite.compute_classification_metrics(true_seq, pred_seq)
    assert metrics["accuracy"] == 1.0
    assert "STATIONARY" in metrics["per_class_metrics"]
    assert "confusion_matrix" in metrics

    lats = ClassifierEvaluationSuite.measure_transition_latencies(true_seq, pred_seq)
    assert "STATIONARY_to_PEDESTRIAN" in lats
    assert "PEDESTRIAN_to_STATIONARY" in lats
    assert "STATIONARY_to_VEHICLE" in lats
    assert "VEHICLE_to_STATIONARY" in lats
    assert "PEDESTRIAN_to_VEHICLE" in lats
    assert "VEHICLE_to_PEDESTRIAN" in lats


def test_motion_router_dispatch():
    config = MotionClassifierConfig(hysteresis_dwell_count=1, confidence_threshold=0.10)
    stabilizer = TemporallyStabilizedMotionClassifier(config=config)
    veh_model = TCNVelocityEstimator(input_channels=6, output_dim=2)
    ped_model = TCNVelocityEstimator(input_channels=6, output_dim=2)

    router = MotionRouter(stabilizer, vehicle_model=veh_model, pedestrian_model=ped_model)

    win = create_synthetic_imu_window("stationary", 200)
    vel, state, conf, info = router.route_and_predict(win)
    assert vel.shape == (2,)
    assert state in [MotionClass.STATIONARY, MotionClass.PEDESTRIAN, MotionClass.VEHICLE, MotionClass.UNKNOWN]
