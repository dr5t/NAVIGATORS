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
from src.utils.metrics import along_cross_track_error


@dataclass
class DomainAdaptationConfig:
    version: str = "1.0.0"
    model_architecture: str = "TCNVelocityEstimator"
    input_channels: int = 6
    output_dim: int = 2
    window_size: int = 200
    sample_rate_hz: float = 10.0
    batch_size: int = 32
    pretrain_lr: float = 0.001
    finetune_lr: float = 0.0001
    weight_decay: float = 0.0001
    epochs_pretrain: int = 15
    epochs_finetune: int = 10
    patience: int = 5
    seed: int = 42
    checkpoint_dir: str = "checkpoints/domain_adaptation_exp"
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
            "pretrain_lr": self.pretrain_lr,
            "finetune_lr": self.finetune_lr,
            "weight_decay": self.weight_decay,
            "epochs_pretrain": self.epochs_pretrain,
            "epochs_finetune": self.epochs_finetune,
            "patience": self.patience,
            "seed": self.seed,
            "checkpoint_dir": self.checkpoint_dir,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
        }


@dataclass
class ConditionMetrics:
    condition_id: str
    condition_name: str
    velocity_mae_mps: float
    velocity_rmse_mps: float
    speed_mae_mps: float
    speed_rmse_mps: float
    trajectory_error_m: float
    cross_track_error_m: float
    heading_error_deg: float
    stratified_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outage_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "condition_name": self.condition_name,
            "velocity_mae_mps": self.velocity_mae_mps,
            "velocity_rmse_mps": self.velocity_rmse_mps,
            "speed_mae_mps": self.speed_mae_mps,
            "speed_rmse_mps": self.speed_rmse_mps,
            "trajectory_error_m": self.trajectory_error_m,
            "cross_track_error_m": self.cross_track_error_m,
            "heading_error_deg": self.heading_error_deg,
            "stratified_metrics": self.stratified_metrics,
            "outage_breakdown": self.outage_breakdown,
        }


class AdaptationIMUDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]


class IndianDomainAdaptationPipeline:
    def __init__(self, config: Optional[DomainAdaptationConfig] = None):
        self.config = config or DomainAdaptationConfig()
        self.normalizer = CrossDatasetNormalizer(config=PreprocessingConfig.for_vehicle())

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

    def partition_dataset(
        self,
        sessions: List[Dict[str, Any]],
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not sessions:
            return [], [], []

        group_map: Dict[str, List[Dict[str, Any]]] = {}
        for s in sessions:
            key = f"{s.get('vehicle_id', 'v')}_{s.get('session_id', 's')}_{s.get('device_id', 'd')}_{s.get('route_id', 'r')}"
            group_map.setdefault(key, []).append(s)

        keys = sorted(group_map.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)

        n = len(keys)
        if n >= 3:
            n_tr = max(1, int(round(n * self.config.train_ratio)))
            n_va = max(1, int(round(n * self.config.val_ratio)))
            if n_tr + n_va >= n:
                n_tr = n - 2
                n_va = 1
            tr_keys = keys[:n_tr]
            va_keys = keys[n_tr:n_tr + n_va]
            te_keys = keys[n_tr + n_va:]
        elif n == 2:
            tr_keys = [keys[0]]
            va_keys = [keys[1]]
            te_keys = [keys[1]]
        else:
            tr_keys = keys
            va_keys = keys
            te_keys = keys

        train_s = [s for k in tr_keys for s in group_map[k]]
        val_s = [s for k in va_keys for s in group_map[k]]
        test_s = [s for k in te_keys for s in group_map[k]]

        return train_s, val_s, test_s

    def extract_windows_and_targets(
        self,
        sessions: List[Dict[str, Any]],
        stats: DatasetNormalizationStats,
        window_size: int = 200,
        stride: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        windows = []
        targets = []
        meta = []

        dt = 1.0 / self.config.sample_rate_hz

        for s_idx, sess in enumerate(sessions):
            prepared = self._ensure_accel_gyro(sess)
            try:
                res = self.normalizer.transform(prepared, stats=stats, sequence_id=f"adapt_{s_idx}")
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
                meta.append({
                    "source_dataset": sess.get("source_dataset", "unknown"),
                    "vehicle_type": sess.get("vehicle_type", "car"),
                    "road_type": sess.get("road_type", "urban"),
                    "traffic_density": sess.get("traffic_density", "medium"),
                    "device_id": sess.get("device_id", "pixel_6"),
                    "phone_mount_position": sess.get("phone_mount_position", "dashboard"),
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
            meta,
        )

    def compute_trajectory_and_heading_errors(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        sample_rate_hz: float = 10.0,
    ) -> Tuple[float, float, float]:
        if len(preds) == 0:
            return 0.0, 0.0, 0.0

        dt = 1.0 / sample_rate_hz
        est_pos = np.cumsum(preds * dt, axis=0)
        true_pos = np.cumsum(targets * dt, axis=0)

        pos_diff = est_pos - true_pos
        traj_error = float(np.mean(np.linalg.norm(pos_diff, axis=1)))

        act = along_cross_track_error(est_pos, true_pos)
        ct_error = float(np.sqrt(np.mean(act["cross_track"] ** 2)))

        pred_headings = np.degrees(np.arctan2(preds[:, 1], preds[:, 0]))
        true_headings = np.degrees(np.arctan2(targets[:, 1], targets[:, 0]))
        diff_deg = (pred_headings - true_headings + 180.0) % 360.0 - 180.0
        heading_error = float(np.mean(np.abs(diff_deg)))

        return (
            float(round(traj_error, 4)),
            float(round(ct_error, 4)),
            float(round(heading_error, 4)),
        )

    def evaluate_condition(
        self,
        model: nn.Module,
        windows: np.ndarray,
        targets: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        condition_id: str,
        condition_name: str,
    ) -> ConditionMetrics:
        if len(windows) == 0:
            return ConditionMetrics(
                condition_id=condition_id,
                condition_name=condition_name,
                velocity_mae_mps=0.0,
                velocity_rmse_mps=0.0,
                speed_mae_mps=0.0,
                speed_rmse_mps=0.0,
                trajectory_error_m=0.0,
                cross_track_error_m=0.0,
                heading_error_deg=0.0,
            )

        model.eval()
        device = next(model.parameters()).device
        x_tensor = torch.from_numpy(windows).float().to(device)

        with torch.no_grad():
            preds = model(x_tensor).cpu().numpy()

        errs = preds - targets
        vel_mae = float(np.mean(np.linalg.norm(errs, axis=1)))
        vel_rmse = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))

        pred_speeds = np.linalg.norm(preds, axis=1)
        target_speeds = np.linalg.norm(targets, axis=1)
        speed_errs = pred_speeds - target_speeds
        speed_mae = float(np.mean(np.abs(speed_errs)))
        speed_rmse = float(np.sqrt(np.mean(speed_errs ** 2)))

        traj_err, ct_err, head_err = self.compute_trajectory_and_heading_errors(
            preds, targets, sample_rate_hz=self.config.sample_rate_hz
        )

        stratified: Dict[str, Dict[str, float]] = {}

        for cat_key in ["road_type", "traffic_density", "vehicle_type", "device_id", "phone_mount_position"]:
            values = set(m.get(cat_key, "unknown") for m in metadata_list)
            for v in values:
                idx_mask = np.array([m.get(cat_key, "unknown") == v for m in metadata_list])
                if np.any(idx_mask):
                    sub_errs = errs[idx_mask]
                    stratified[f"{cat_key}_{v}"] = {
                        "vel_mae": float(np.mean(np.linalg.norm(sub_errs, axis=1))),
                        "speed_mae": float(np.mean(np.abs(speed_errs[idx_mask]))),
                    }

        dt = 1.0 / self.config.sample_rate_hz
        outage_breakdown: Dict[str, float] = {}
        for outage_sec in [10, 30, 60, 120, 300]:
            frames = int(outage_sec * self.config.sample_rate_hz)
            if len(preds) >= frames:
                sub_p = preds[:frames]
                sub_t = targets[:frames]
                p_pos = np.cumsum(sub_p * dt, axis=0)
                t_pos = np.cumsum(sub_t * dt, axis=0)
                outage_breakdown[f"{outage_sec}s"] = float(round(float(np.linalg.norm(p_pos[-1] - t_pos[-1])), 3))

        return ConditionMetrics(
            condition_id=condition_id,
            condition_name=condition_name,
            velocity_mae_mps=float(round(vel_mae, 4)),
            velocity_rmse_mps=float(round(vel_rmse, 4)),
            speed_mae_mps=float(round(speed_mae, 4)),
            speed_rmse_mps=float(round(speed_rmse, 4)),
            trajectory_error_m=traj_err,
            cross_track_error_m=ct_err,
            heading_error_deg=head_err,
            stratified_metrics=stratified,
            outage_breakdown=outage_breakdown,
        )

    def _train_model_on_data(
        self,
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        lr: float,
        epochs: int,
    ) -> nn.Module:
        if len(x_train) == 0:
            return model

        dataset = AdaptationIMUDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.config.weight_decay)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
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

        return model

    def run_domain_adaptation_experiments(
        self,
        iovnbd_sessions: List[Dict[str, Any]],
        indian_sessions: List[Dict[str, Any]],
    ) -> Dict[str, ConditionMetrics]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        io_train, io_val, io_test = self.partition_dataset(iovnbd_sessions, seed=self.config.seed)
        ind_train, ind_val, ind_final_test = self.partition_dataset(indian_sessions, seed=self.config.seed)

        train_for_fitting = [self._ensure_accel_gyro(s) for s in (io_train + ind_train)]
        stats = self.normalizer.fit(train_for_fitting if train_for_fitting else [self._ensure_accel_gyro(s) for s in indian_sessions])

        x_io_tr, y_io_tr, _ = self.extract_windows_and_targets(io_train, stats)
        x_io_va, y_io_va, _ = self.extract_windows_and_targets(io_val, stats)

        x_ind_tr, y_ind_tr, _ = self.extract_windows_and_targets(ind_train, stats)
        x_ind_va, y_ind_va, _ = self.extract_windows_and_targets(ind_val, stats)

        x_final_test, y_final_test, meta_final_test = self.extract_windows_and_targets(ind_final_test, stats)

        results: Dict[str, ConditionMetrics] = {}

        model_a = TCNVelocityEstimator(input_channels=6, output_dim=2)
        model_a = self._train_model_on_data(
            model_a, x_io_tr, y_io_tr, x_io_va, y_io_va, lr=self.config.pretrain_lr, epochs=self.config.epochs_pretrain
        )
        results["A"] = self.evaluate_condition(
            model_a, x_final_test, y_final_test, meta_final_test, "A", "IO-VNBD Only"
        )

        model_b = TCNVelocityEstimator(input_channels=6, output_dim=2)
        model_b = self._train_model_on_data(
            model_b, x_ind_tr, y_ind_tr, x_ind_va, y_ind_va, lr=self.config.pretrain_lr, epochs=self.config.epochs_pretrain
        )
        results["B"] = self.evaluate_condition(
            model_b, x_final_test, y_final_test, meta_final_test, "B", "Navigators India Only"
        )

        x_joint_tr = np.vstack([x_io_tr, x_ind_tr]) if len(x_io_tr) > 0 and len(x_ind_tr) > 0 else (x_io_tr if len(x_io_tr) > 0 else x_ind_tr)
        y_joint_tr = np.vstack([y_io_tr, y_ind_tr]) if len(y_io_tr) > 0 and len(y_ind_tr) > 0 else (y_io_tr if len(y_io_tr) > 0 else y_ind_tr)
        x_joint_va = np.vstack([x_io_va, x_ind_va]) if len(x_io_va) > 0 and len(x_ind_va) > 0 else (x_io_va if len(x_io_va) > 0 else x_ind_va)
        y_joint_va = np.vstack([y_io_va, y_ind_va]) if len(y_io_va) > 0 and len(y_ind_va) > 0 else (y_io_va if len(y_io_va) > 0 else y_ind_va)

        model_c = TCNVelocityEstimator(input_channels=6, output_dim=2)
        model_c = self._train_model_on_data(
            model_c, x_joint_tr, y_joint_tr, x_joint_va, y_joint_va, lr=self.config.pretrain_lr, epochs=self.config.epochs_pretrain
        )
        results["C"] = self.evaluate_condition(
            model_c, x_final_test, y_final_test, meta_final_test, "C", "IO-VNBD + Navigators India Joint"
        )

        model_d = TCNVelocityEstimator(input_channels=6, output_dim=2)
        model_d = self._train_model_on_data(
            model_d, x_io_tr, y_io_tr, x_io_va, y_io_va, lr=self.config.pretrain_lr, epochs=self.config.epochs_pretrain
        )
        model_d = self._train_model_on_data(
            model_d, x_ind_tr, y_ind_tr, x_ind_va, y_ind_va, lr=self.config.finetune_lr, epochs=self.config.epochs_finetune
        )
        results["D"] = self.evaluate_condition(
            model_d, x_final_test, y_final_test, meta_final_test, "D", "IO-VNBD Pretrained + Indian Fine-Tuned"
        )

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_a": model_a.state_dict(),
                "model_b": model_b.state_dict(),
                "model_c": model_c.state_dict(),
                "model_d": model_d.state_dict(),
                "config": self.config.to_dict(),
                "normalization_stats": stats.to_dict(),
            },
            ckpt_dir / "domain_adaptation_checkpoints.pt",
        )

        return results
