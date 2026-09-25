from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.tcn_model import TCNVelocityEstimator
from src.data.sensor_normalization import (
    CrossDatasetNormalizer,
    PreprocessingConfig,
    DatasetNormalizationStats,
    DomainType,
)


@dataclass
class VehicleTrainingConfig:
    version: str = "1.0.0"
    model_architecture: str = "TCNVelocityEstimator"
    input_channels: int = 6
    output_dim: int = 2
    window_size: int = 200
    sample_rate_hz: float = 10.0
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 20
    patience: int = 5
    seed: int = 42
    checkpoint_dir: str = "checkpoints/vehicle_exp"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_architecture": self.model_architecture,
            "input_channels": self.input_channels,
            "output_dim": self.output_dim,
            "window_size": self.window_size,
            "sample_rate_hz": self.sample_rate_hz,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "checkpoint_dir": self.checkpoint_dir,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }


@dataclass
class VehicleEvaluationResult:
    model_name: str
    velocity_mae_mps: float
    velocity_rmse_mps: float
    speed_mae_mps: float
    speed_rmse_mps: float
    north_velocity_error_mps: float
    east_velocity_error_mps: float
    stratified_metrics: Dict[str, Dict[str, float]]
    inference_latency_ms: float
    model_size_bytes: int
    parameter_count: int


class VehicleIMUDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]


class VehicleTrainingPipeline:
    def __init__(self, config: Optional[VehicleTrainingConfig] = None):
        self.config = config or VehicleTrainingConfig()
        self.normalizer = CrossDatasetNormalizer(config=PreprocessingConfig.for_vehicle())

    def prepare_splits(
        self,
        sessions: List[Dict[str, Any]],
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not sessions:
            return [], [], []

        vehicle_groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in sessions:
            v_id = str(s.get("vehicle_id", s.get("source_dataset", "default_vehicle")))
            vehicle_groups.setdefault(v_id, []).append(s)

        group_keys = sorted(vehicle_groups.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        if n_groups >= 3:
            n_train = max(1, int(round(n_groups * self.config.train_ratio)))
            n_val = max(1, int(round(n_groups * self.config.val_ratio)))
            if n_train + n_val >= n_groups:
                n_train = n_groups - 2
                n_val = 1
            train_groups = group_keys[:n_train]
            val_groups = group_keys[n_train:n_train + n_val]
            test_groups = group_keys[n_train + n_val:]
        elif n_groups == 2:
            train_groups = [group_keys[0]]
            val_groups = [group_keys[1]]
            test_groups = [group_keys[1]]
        else:
            train_groups = group_keys
            val_groups = group_keys
            test_groups = group_keys

        train_sess = [s for g in train_groups for s in vehicle_groups[g]]
        val_sess = [s for g in val_groups for s in vehicle_groups[g]]
        test_sess = [s for g in test_groups for s in vehicle_groups[g]]

        return train_sess, val_sess, test_sess

    def extract_windows_and_targets(
        self,
        sessions: List[Dict[str, Any]],
        stats: DatasetNormalizationStats,
        window_size: int = 200,
        stride: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        windows = []
        targets = []
        metadata_list = []

        for s_idx, sess in enumerate(sessions):
            sess_copy = dict(sess)
            if "accel" not in sess_copy and "accelerometer_x" in sess_copy:
                sess_copy["accel"] = np.column_stack([
                    sess_copy["accelerometer_x"],
                    sess_copy["accelerometer_y"],
                    sess_copy["accelerometer_z"],
                ])
            if "gyro" not in sess_copy and "gyroscope_x" in sess_copy:
                sess_copy["gyro"] = np.column_stack([
                    sess_copy["gyroscope_x"],
                    sess_copy["gyroscope_y"],
                    sess_copy["gyroscope_z"],
                ])
            try:
                res = self.normalizer.transform(sess_copy, stats=stats, sequence_id=f"seq_{s_idx}")
            except Exception:
                continue

            imu_norm = res.normalized_data["imu_normalized"]
            if "velocity_north_east" not in res.normalized_data:
                continue

            vel_ne = res.normalized_data["velocity_north_east"]
            n_samples = len(imu_norm)

            if n_samples < window_size:
                continue

            for start in range(0, n_samples - window_size + 1, stride):
                end = start + window_size
                win = imu_norm[start:end]
                target_vel = vel_ne[end - 1]

                windows.append(win)
                targets.append(target_vel)
                metadata_list.append({
                    "source_dataset": sess.get("source_dataset", "unknown"),
                    "vehicle_type": sess.get("vehicle_type", "unknown"),
                    "road_type": sess.get("road_type", "unknown"),
                    "device_id": sess.get("device_id", "unknown"),
                    "speed": float(np.linalg.norm(target_vel)),
                })

        if not windows:
            return (
                np.zeros((0, window_size, 6), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                [],
            )

        return (
            np.array(windows, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            metadata_list,
        )

    def evaluate_model(
        self,
        model: nn.Module,
        windows: np.ndarray,
        targets: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        model_name: str = "Experimental",
    ) -> VehicleEvaluationResult:
        if len(windows) == 0:
            return VehicleEvaluationResult(
                model_name=model_name,
                velocity_mae_mps=0.0,
                velocity_rmse_mps=0.0,
                speed_mae_mps=0.0,
                speed_rmse_mps=0.0,
                north_velocity_error_mps=0.0,
                east_velocity_error_mps=0.0,
                stratified_metrics={},
                inference_latency_ms=0.0,
                model_size_bytes=0,
                parameter_count=0,
            )

        model.eval()
        device = next(model.parameters()).device
        x_tensor = torch.from_numpy(windows).float().to(device)

        start_t = time.perf_counter()
        with torch.no_grad():
            preds = model(x_tensor).cpu().numpy()
        latency_ms = float((time.perf_counter() - start_t) / max(len(windows), 1) * 1000.0)

        errs = preds - targets
        vn_err = float(np.mean(np.abs(errs[:, 0])))
        ve_err = float(np.mean(np.abs(errs[:, 1])))

        vel_mae = float(np.mean(np.linalg.norm(errs, axis=1)))
        vel_rmse = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))

        pred_speeds = np.linalg.norm(preds, axis=1)
        target_speeds = np.linalg.norm(targets, axis=1)

        speed_errs = pred_speeds - target_speeds
        speed_mae = float(np.mean(np.abs(speed_errs)))
        speed_rmse = float(np.sqrt(np.mean(speed_errs ** 2)))

        stratified: Dict[str, Dict[str, float]] = {}

        is_indian = np.array(["navigators" in m.get("source_dataset", "").lower() for m in metadata_list])
        if np.any(is_indian):
            stratified["indian_data"] = {
                "vel_mae": float(np.mean(np.linalg.norm(errs[is_indian], axis=1))),
                "speed_mae": float(np.mean(np.abs(speed_errs[is_indian]))),
            }
        if np.any(~is_indian):
            stratified["non_indian_data"] = {
                "vel_mae": float(np.mean(np.linalg.norm(errs[~is_indian], axis=1))),
                "speed_mae": float(np.mean(np.abs(speed_errs[~is_indian]))),
            }

        param_count = sum(p.numel() for p in model.parameters())

        return VehicleEvaluationResult(
            model_name=model_name,
            velocity_mae_mps=float(round(vel_mae, 4)),
            velocity_rmse_mps=float(round(vel_rmse, 4)),
            speed_mae_mps=float(round(speed_mae, 4)),
            speed_rmse_mps=float(round(speed_rmse, 4)),
            north_velocity_error_mps=float(round(vn_err, 4)),
            east_velocity_error_mps=float(round(ve_err, 4)),
            stratified_metrics=stratified,
            inference_latency_ms=float(round(latency_ms, 3)),
            model_size_bytes=param_count * 4,
            parameter_count=param_count,
        )

    def _ensure_accel_gyro(self, session: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(session)
        if "accel" not in s and "accelerometer_x" in s:
            s["accel"] = np.column_stack([
                s["accelerometer_x"],
                s["accelerometer_y"],
                s["accelerometer_z"],
            ])
        if "gyro" not in s and "gyroscope_x" in s:
            s["gyro"] = np.column_stack([
                s["gyroscope_x"],
                s["gyroscope_y"],
                s["gyroscope_z"],
            ])
        return s

    def train_experimental_model(
        self,
        sessions: List[Dict[str, Any]],
    ) -> Tuple[TCNVelocityEstimator, DatasetNormalizationStats, VehicleEvaluationResult]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        train_sess, val_sess, test_sess = self.prepare_splits(sessions, seed=self.config.seed)
        train_sess_prepared = [self._ensure_accel_gyro(s) for s in train_sess]
        train_stats = self.normalizer.fit(train_sess_prepared)

        x_train, y_train, meta_train = self.extract_windows_and_targets(train_sess, train_stats)
        x_val, y_val, meta_val = self.extract_windows_and_targets(val_sess, train_stats)
        x_test, y_test, meta_test = self.extract_windows_and_targets(test_sess, train_stats)

        model = TCNVelocityEstimator(input_channels=6, output_dim=2)

        if len(x_train) > 0:
            dataset = VehicleIMUDataset(x_train, y_train)
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            criterion = nn.MSELoss()

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.epochs):
                model.train()
                for win, tgt in loader:
                    optimizer.zero_grad()
                    pred = model(win)
                    loss = criterion(pred, tgt)
                    loss.backward()
                    optimizer.step()

                if len(x_val) > 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(torch.from_numpy(x_val).float())
                        val_loss = float(criterion(val_pred, torch.from_numpy(y_val).float()).item())

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience:
                            break

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config.to_dict(),
                "normalization_stats": train_stats.to_dict(),
            },
            ckpt_dir / "vehicle_tcn_exp.pt",
        )

        eval_res = self.evaluate_model(model, x_test, y_test, meta_test, model_name="Experimental TCN")
        return model, train_stats, eval_res
