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
from src.utils.metrics import absolute_trajectory_error, along_cross_track_error


@dataclass
class PedestrianTrainingConfig:
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
    checkpoint_dir: str = "checkpoints/pedestrian_exp"
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
class PedestrianEvaluationResult:
    selected_formulation: str
    velocity_mae_mps: float
    velocity_rmse_mps: float
    trajectory_error_m: float
    ate_rmse_m: float
    fde_m: float
    heading_error_deg: float
    formulation_comparison: Dict[str, Dict[str, float]]
    stratified_metrics: Dict[str, Dict[str, float]]
    inference_latency_ms: float
    model_size_bytes: int
    parameter_count: int


class PedestrianIMUDataset(Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets[idx]


class PedestrianTrainingPipeline:
    def __init__(self, config: Optional[PedestrianTrainingConfig] = None):
        self.config = config or PedestrianTrainingConfig()
        self.normalizer = CrossDatasetNormalizer(config=PreprocessingConfig.for_pedestrian())

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

    def prepare_splits(
        self,
        sessions: List[Dict[str, Any]],
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not sessions:
            return [], [], []

        subject_groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in sessions:
            subj_id = str(s.get("subject_id", s.get("user_id", s.get("session_id", "subj_0"))))
            subject_groups.setdefault(subj_id, []).append(s)

        group_keys = sorted(subject_groups.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(group_keys)

        n = len(group_keys)
        if n >= 3:
            n_tr = max(1, int(round(n * self.config.train_ratio)))
            n_va = max(1, int(round(n * self.config.val_ratio)))
            if n_tr + n_va >= n:
                n_tr = n - 2
                n_va = 1
            tr_keys = group_keys[:n_tr]
            va_keys = group_keys[n_tr:n_tr + n_va]
            te_keys = group_keys[n_tr + n_va:]
        elif n == 2:
            tr_keys = [group_keys[0]]
            va_keys = [group_keys[1]]
            te_keys = [group_keys[1]]
        else:
            tr_keys = group_keys
            va_keys = group_keys
            te_keys = group_keys

        train_s = [s for k in tr_keys for s in subject_groups[k]]
        val_s = [s for k in va_keys for s in subject_groups[k]]
        test_s = [s for k in te_keys for s in subject_groups[k]]

        return train_s, val_s, test_s

    def extract_windows_and_targets(
        self,
        sessions: List[Dict[str, Any]],
        stats: DatasetNormalizationStats,
        formulation: str = "velocity_2d",
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
                res = self.normalizer.transform(prepared, stats=stats, sequence_id=f"ped_{s_idx}")
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

                if formulation == "velocity_2d":
                    tgt = vel_ne[end - 1]
                else:
                    win_vels = vel_ne[start:end]
                    tgt = np.sum(win_vels, axis=0) * dt

                windows.append(win)
                targets.append(tgt)
                meta.append({
                    "subject_id": sess.get("subject_id", sess.get("user_id", "subj_0")),
                    "device_id": sess.get("device_id", "pixel_6"),
                    "phone_placement": sess.get("phone_placement", sess.get("phone_mount_position", "handheld")),
                    "walking_speed": sess.get("walking_speed", "normal"),
                    "motion_type": sess.get("motion_type", sess.get("activity", "walking")),
                    "source_dataset": sess.get("source_dataset", "unknown"),
                    "speed": float(np.linalg.norm(vel_ne[end - 1])),
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

    def _train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TCNVelocityEstimator:
        model = TCNVelocityEstimator(input_channels=6, output_dim=2)
        if len(x_train) == 0:
            return model

        dataset = PedestrianIMUDataset(x_train, y_train)
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

        return model

    def evaluate_model(
        self,
        model: nn.Module,
        windows: np.ndarray,
        targets: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        formulation: str = "velocity_2d",
    ) -> Dict[str, Any]:
        if len(windows) == 0:
            return {
                "vel_mae": 0.0,
                "vel_rmse": 0.0,
                "traj_err": 0.0,
                "ate_rmse": 0.0,
                "fde": 0.0,
                "heading_err": 0.0,
                "stratified": {},
                "latency_ms": 0.0,
            }

        model.eval()
        device = next(model.parameters()).device
        x_tensor = torch.from_numpy(windows).float().to(device)

        start_t = time.perf_counter()
        with torch.no_grad():
            preds = model(x_tensor).cpu().numpy()
        latency_ms = float((time.perf_counter() - start_t) / max(len(windows), 1) * 1000.0)

        dt = 1.0 / self.config.sample_rate_hz

        if formulation == "displacement_local":
            pred_vels = preds / dt
            true_vels = targets / dt
            pred_disps = preds
            true_disps = targets
        else:
            pred_vels = preds
            true_vels = targets
            pred_disps = preds * dt
            true_disps = targets * dt

        vel_errs = pred_vels - true_vels
        vel_mae = float(np.mean(np.linalg.norm(vel_errs, axis=1)))
        vel_rmse = float(np.sqrt(np.mean(np.sum(vel_errs ** 2, axis=1))))

        est_pos = np.cumsum(pred_disps, axis=0)
        true_pos = np.cumsum(true_disps, axis=0)

        ate_dict = absolute_trajectory_error(est_pos, true_pos)
        ate_rmse = ate_dict["rmse"]

        fde = float(np.linalg.norm(est_pos[-1] - true_pos[-1]))
        traj_err = float(np.mean(np.linalg.norm(est_pos - true_pos, axis=1)))

        pred_headings = np.degrees(np.arctan2(pred_vels[:, 1], pred_vels[:, 0]))
        true_headings = np.degrees(np.arctan2(true_vels[:, 1], true_vels[:, 0]))
        heading_diff = (pred_headings - true_headings + 180.0) % 360.0 - 180.0
        heading_err = float(np.mean(np.abs(heading_diff)))

        stratified: Dict[str, Dict[str, float]] = {}
        for cat_key in ["phone_placement", "walking_speed", "motion_type", "device_id", "subject_id"]:
            values = set(m.get(cat_key, "unknown") for m in metadata_list)
            for v in values:
                idx_mask = np.array([m.get(cat_key, "unknown") == v for m in metadata_list])
                if np.any(idx_mask):
                    sub_v_errs = vel_errs[idx_mask]
                    sub_est = np.cumsum(pred_disps[idx_mask], axis=0)
                    sub_true = np.cumsum(true_disps[idx_mask], axis=0)
                    sub_ate = absolute_trajectory_error(sub_est, sub_true)
                    stratified[f"{cat_key}_{v}"] = {
                        "vel_mae": float(np.mean(np.linalg.norm(sub_v_errs, axis=1))),
                        "ate_rmse": float(sub_ate["rmse"]),
                    }

        return {
            "vel_mae": float(round(vel_mae, 4)),
            "vel_rmse": float(round(vel_rmse, 4)),
            "traj_err": float(round(traj_err, 4)),
            "ate_rmse": float(round(ate_rmse, 4)),
            "fde": float(round(fde, 4)),
            "heading_err": float(round(heading_err, 4)),
            "stratified": stratified,
            "latency_ms": float(round(latency_ms, 3)),
        }

    def train_and_select_pedestrian_model(
        self,
        sessions: List[Dict[str, Any]],
    ) -> Tuple[TCNVelocityEstimator, DatasetNormalizationStats, PedestrianEvaluationResult]:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        train_sess, val_sess, test_sess = self.prepare_splits(sessions, seed=self.config.seed)
        train_prepared = [self._ensure_accel_gyro(s) for s in train_sess]
        stats = self.normalizer.fit(train_prepared if train_prepared else [self._ensure_accel_gyro(s) for s in sessions])

        x_tr_v1, y_tr_v1, _ = self.extract_windows_and_targets(train_sess, stats, formulation="velocity_2d")
        x_va_v1, y_va_v1, _ = self.extract_windows_and_targets(val_sess, stats, formulation="velocity_2d")
        x_te_v1, y_te_v1, meta_te_v1 = self.extract_windows_and_targets(test_sess, stats, formulation="velocity_2d")

        x_tr_v2, y_tr_v2, _ = self.extract_windows_and_targets(train_sess, stats, formulation="displacement_local")
        x_va_v2, y_va_v2, _ = self.extract_windows_and_targets(val_sess, stats, formulation="displacement_local")
        x_te_v2, y_te_v2, meta_te_v2 = self.extract_windows_and_targets(test_sess, stats, formulation="displacement_local")

        model_v1 = self._train_model(x_tr_v1, y_tr_v1, x_va_v1, y_va_v1)
        val_eval_v1 = self.evaluate_model(model_v1, x_va_v1, y_va_v1, [], formulation="velocity_2d")

        model_v2 = self._train_model(x_tr_v2, y_tr_v2, x_va_v2, y_va_v2)
        val_eval_v2 = self.evaluate_model(model_v2, x_va_v2, y_va_v2, [], formulation="displacement_local")

        form_comp = {
            "velocity_2d": {
                "val_vel_mae": val_eval_v1["vel_mae"],
                "val_ate_rmse": val_eval_v1["ate_rmse"],
            },
            "displacement_local": {
                "val_vel_mae": val_eval_v2["vel_mae"],
                "val_ate_rmse": val_eval_v2["ate_rmse"],
            },
        }

        if val_eval_v1["ate_rmse"] <= val_eval_v2["ate_rmse"]:
            selected_formulation = "velocity_2d"
            winning_model = model_v1
            test_eval = self.evaluate_model(winning_model, x_te_v1, y_te_v1, meta_te_v1, formulation="velocity_2d")
        else:
            selected_formulation = "displacement_local"
            winning_model = model_v2
            test_eval = self.evaluate_model(winning_model, x_te_v2, y_te_v2, meta_te_v2, formulation="displacement_local")

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        param_count = sum(p.numel() for p in winning_model.parameters())

        torch.save(
            {
                "model_state_dict": winning_model.state_dict(),
                "config": self.config.to_dict(),
                "selected_formulation": selected_formulation,
                "normalization_stats": stats.to_dict(),
                "formulation_comparison": form_comp,
            },
            ckpt_dir / "pedestrian_tcn_exp.pt",
        )

        result = PedestrianEvaluationResult(
            selected_formulation=selected_formulation,
            velocity_mae_mps=test_eval["vel_mae"],
            velocity_rmse_mps=test_eval["vel_rmse"],
            trajectory_error_m=test_eval["traj_err"],
            ate_rmse_m=test_eval["ate_rmse"],
            fde_m=test_eval["fde"],
            heading_error_deg=test_eval["heading_err"],
            formulation_comparison=form_comp,
            stratified_metrics=test_eval["stratified"],
            inference_latency_ms=test_eval["latency_ms"],
            model_size_bytes=param_count * 4,
            parameter_count=param_count,
        )

        return winning_model, stats, result
