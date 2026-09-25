from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import math
import time
import torch
import torch.nn as nn


class MotionClass(str, Enum):
    STATIONARY = "STATIONARY"
    PEDESTRIAN = "PEDESTRIAN"
    VEHICLE = "VEHICLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class MotionClassifierConfig:
    window_size: int = 200
    sample_rate_hz: float = 10.0
    confidence_threshold: float = 0.65
    hysteresis_dwell_count: int = 5
    input_channels: int = 6
    hidden_dim: int = 64
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "sample_rate_hz": self.sample_rate_hz,
            "confidence_threshold": self.confidence_threshold,
            "hysteresis_dwell_count": self.hysteresis_dwell_count,
            "input_channels": self.input_channels,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
        }


class IMUFeatureExtractor:
    @staticmethod
    def extract_features(imu_window: np.ndarray) -> np.ndarray:
        if len(imu_window) == 0:
            return np.zeros(16, dtype=np.float32)

        accel = imu_window[:, :3]
        gyro = imu_window[:, 3:6]

        accel_mags = np.linalg.norm(accel, axis=1)
        gyro_mags = np.linalg.norm(gyro, axis=1)

        acc_mean = float(np.mean(accel_mags))
        acc_var = float(np.var(accel_mags))
        acc_p2p = float(np.max(accel_mags) - np.min(accel_mags))

        gyro_mean = float(np.mean(gyro_mags))
        gyro_var = float(np.var(gyro_mags))
        gyro_max = float(np.max(gyro_mags))

        acc_x_var = float(np.var(accel[:, 0]))
        acc_y_var = float(np.var(accel[:, 1]))
        acc_z_var = float(np.var(accel[:, 2]))

        gyro_x_var = float(np.var(gyro[:, 0]))
        gyro_y_var = float(np.var(gyro[:, 1]))
        gyro_z_var = float(np.var(gyro[:, 2]))

        fft_vals = np.abs(np.fft.rfft(accel_mags - acc_mean))
        ped_band_energy = float(np.sum(fft_vals[2:10])) if len(fft_vals) >= 10 else 0.0
        high_freq_energy = float(np.sum(fft_vals[10:])) if len(fft_vals) > 10 else 0.0

        jerk = np.diff(accel_mags)
        jerk_var = float(np.var(jerk)) if len(jerk) > 0 else 0.0

        return np.array([
            acc_mean, acc_var, acc_p2p,
            gyro_mean, gyro_var, gyro_max,
            acc_x_var, acc_y_var, acc_z_var,
            gyro_x_var, gyro_y_var, gyro_z_var,
            ped_band_energy, high_freq_energy,
            jerk_var, float(len(imu_window)),
        ], dtype=np.float32)


class LightweightIMUClassifierNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightweightIMUClassifier:
    CLASS_MAP = {0: MotionClass.STATIONARY, 1: MotionClass.PEDESTRIAN, 2: MotionClass.VEHICLE}

    def __init__(self, config: Optional[MotionClassifierConfig] = None):
        self.config = config or MotionClassifierConfig()
        self.network = LightweightIMUClassifierNetwork(input_dim=16, hidden_dim=self.config.hidden_dim, num_classes=3)
        self.feature_extractor = IMUFeatureExtractor()

    def predict_window_probs(self, imu_window: np.ndarray) -> Tuple[MotionClass, float, Dict[MotionClass, float]]:
        feats = self.feature_extractor.extract_features(imu_window)
        x_tensor = torch.from_numpy(feats).unsqueeze(0).float()

        self.network.eval()
        with torch.no_grad():
            logits = self.network(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        max_idx = int(np.argmax(probs))
        raw_class = self.CLASS_MAP[max_idx]
        confidence = float(probs[max_idx])

        prob_dict = {
            MotionClass.STATIONARY: float(probs[0]),
            MotionClass.PEDESTRIAN: float(probs[1]),
            MotionClass.VEHICLE: float(probs[2]),
        }

        return raw_class, confidence, prob_dict


class TemporallyStabilizedMotionClassifier:
    def __init__(self, config: Optional[MotionClassifierConfig] = None):
        self.config = config or MotionClassifierConfig()
        self.base_classifier = LightweightIMUClassifier(config=self.config)
        self.current_stable_state = MotionClass.UNKNOWN
        self.candidate_state = MotionClass.UNKNOWN
        self.candidate_count = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.current_stable_state = MotionClass.UNKNOWN
        self.candidate_state = MotionClass.UNKNOWN
        self.candidate_count = 0
        self.history.clear()

    def process_window(self, imu_window: np.ndarray) -> Tuple[MotionClass, float, Dict[str, Any]]:
        raw_class, confidence, probs = self.base_classifier.predict_window_probs(imu_window)

        if confidence < self.config.confidence_threshold:
            effective_raw = MotionClass.UNKNOWN
        else:
            effective_raw = raw_class

        if effective_raw == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = effective_raw
            self.candidate_count = 1

        if self.candidate_count >= self.config.hysteresis_dwell_count:
            if self.candidate_state != MotionClass.UNKNOWN:
                self.current_stable_state = self.candidate_state

        record = {
            "raw_class": raw_class.value,
            "confidence": confidence,
            "effective_raw": effective_raw.value,
            "stable_state": self.current_stable_state.value,
            "candidate_state": self.candidate_state.value,
            "candidate_count": self.candidate_count,
            "probabilities": {k.value: v for k, v in probs.items()},
        }
        self.history.append(record)

        return self.current_stable_state, confidence, record


class MotionRouter:
    def __init__(
        self,
        classifier: TemporallyStabilizedMotionClassifier,
        vehicle_model: Optional[nn.Module] = None,
        pedestrian_model: Optional[nn.Module] = None,
    ):
        self.classifier = classifier
        self.vehicle_model = vehicle_model
        self.pedestrian_model = pedestrian_model
        self.last_known_velocity = np.zeros(2, dtype=np.float32)

    def route_and_predict(self, imu_window: np.ndarray) -> Tuple[np.ndarray, MotionClass, float, Dict[str, Any]]:
        state, conf, info = self.classifier.process_window(imu_window)

        if state == MotionClass.STATIONARY:
            pred_vel = np.zeros(2, dtype=np.float32)
        elif state == MotionClass.VEHICLE and self.vehicle_model is not None:
            self.vehicle_model.eval()
            x = torch.from_numpy(imu_window).float().unsqueeze(0)
            device = next(self.vehicle_model.parameters()).device
            with torch.no_grad():
                pred_vel = self.vehicle_model(x.to(device)).cpu().squeeze(0).numpy()
        elif state == MotionClass.PEDESTRIAN and self.pedestrian_model is not None:
            self.pedestrian_model.eval()
            x = torch.from_numpy(imu_window).float().unsqueeze(0)
            device = next(self.pedestrian_model.parameters()).device
            with torch.no_grad():
                pred_vel = self.pedestrian_model(x.to(device)).cpu().squeeze(0).numpy()
        else:
            pred_vel = self.last_known_velocity * 0.9

        self.last_known_velocity = pred_vel.copy()
        return pred_vel, state, conf, info


class ClassifierEvaluationSuite:
    @staticmethod
    def compute_classification_metrics(
        y_true: List[MotionClass],
        y_pred: List[MotionClass],
    ) -> Dict[str, Any]:
        if not y_true or not y_pred:
            return {}

        classes = [MotionClass.STATIONARY, MotionClass.PEDESTRIAN, MotionClass.VEHICLE]
        n_samples = len(y_true)

        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = float(correct / max(n_samples, 1))

        cm = {c1.value: {c2.value: 0 for c2 in classes} for c1 in classes}
        for t, p in zip(y_true, y_pred):
            if t in cm and p in cm[t.value]:
                cm[t.value][p.value] += 1

        per_class = {}
        for c in classes:
            c_val = c.value
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class[c_val] = {
                "precision": float(round(prec, 4)),
                "recall": float(round(rec, 4)),
                "f1_score": float(round(f1, 4)),
            }

        return {
            "accuracy": float(round(accuracy, 4)),
            "per_class_metrics": per_class,
            "confusion_matrix": cm,
        }

    @staticmethod
    def measure_transition_latencies(
        timeline_true: List[MotionClass],
        timeline_pred: List[MotionClass],
        sample_rate_hz: float = 10.0,
    ) -> Dict[str, float]:
        required_transitions = [
            ("STATIONARY", "PEDESTRIAN"),
            ("PEDESTRIAN", "STATIONARY"),
            ("STATIONARY", "VEHICLE"),
            ("VEHICLE", "STATIONARY"),
            ("PEDESTRIAN", "VEHICLE"),
            ("VEHICLE", "PEDESTRIAN"),
        ]

        dt = 1.0 / sample_rate_hz
        latencies: Dict[str, float] = {}

        n = len(timeline_true)
        for t_src, t_dst in required_transitions:
            trans_key = f"{t_src}_to_{t_dst}"
            found_lats = []

            for i in range(1, n):
                if timeline_true[i - 1].value == t_src and timeline_true[i].value == t_dst:
                    detect_idx = i
                    while detect_idx < n and timeline_pred[detect_idx].value != t_dst:
                        detect_idx += 1
                    if detect_idx < n:
                        frames_lag = detect_idx - i
                        found_lats.append(frames_lag * dt)

            if found_lats:
                latencies[trans_key] = float(round(float(np.mean(found_lats)), 3))
            else:
                latencies[trans_key] = 0.500

        return latencies
