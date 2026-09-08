"""
Navigators IDR — Model Trainer
Training loop with validation, early stopping, checkpointing, and logging.

Supports:
    - TCN and LSTM model architectures
    - MSE + angular direction loss
    - Cosine/Step/Plateau learning rate scheduling
    - TensorBoard logging
    - Model checkpointing with best-model tracking
"""

import os
import time
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from models.tcn_model import TCNVelocityEstimator
from models.lstm_model import LSTMVelocityEstimator


class AngularLoss(nn.Module):
    """
    Combined MSE + angular direction loss.

    The angular component penalizes heading errors more heavily,
    which is critical for dead reckoning where heading errors
    compound much faster than speed errors.
    """

    def __init__(self, angular_weight: float = 0.3):
        super().__init__()
        self.angular_weight = angular_weight
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, 2) predicted [v_north, v_east].
            target: (batch, 2) ground truth [v_north, v_east].

        Returns:
            Scalar loss.
        """
        # MSE component
        mse_loss = self.mse(pred, target)

        # Angular component: cosine similarity between direction vectors
        pred_norm = torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-6)
        target_norm = torch.norm(target, dim=1, keepdim=True).clamp(min=1e-6)

        cos_sim = torch.sum(
            (pred / pred_norm) * (target / target_norm), dim=1
        )
        angular_loss = 1.0 - cos_sim.mean()  # 0 when perfectly aligned

        return mse_loss + self.angular_weight * angular_loss


class Trainer:
    """
    Model trainer for velocity estimation networks.

    Handles the full training loop including:
        - Training and validation epochs
        - Learning rate scheduling
        - Early stopping
        - Checkpointing
        - TensorBoard logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: TCNVelocityEstimator or LSTMVelocityEstimator.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration dict.
            device: Torch device (auto-detected if None).
        """
        self.config = config or {}

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"[Trainer] Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training hyperparameters
        self.epochs = self.config.get("epochs", 100)
        self.lr = self.config.get("learning_rate", 0.001)
        self.weight_decay = self.config.get("weight_decay", 0.0001)
        self.gradient_clip = self.config.get("gradient_clip", 1.0)
        self.patience = self.config.get("early_stopping_patience", 15)

        # Loss function
        loss_type = self.config.get("loss", "mse_angular")
        if loss_type == "mse_angular":
            angular_weight = self.config.get("angular_loss_weight", 0.3)
            self.criterion = AngularLoss(angular_weight)
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()

        # Optimizer
        opt_name = self.config.get("optimizer", "adamw")
        if opt_name == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # Learning rate scheduler
        scheduler_name = self.config.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_lr_step", 20),
                gamma=self.config.get("step_lr_gamma", 0.5),
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=7, factor=0.5
            )
        else:
            self.scheduler = None

        # Checkpointing
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.config.get("log_dir", "./runs")
            self.writer = SummaryWriter(log_dir)

        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = {"train_loss": [], "val_loss": [], "lr": []}

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Data Augmentation (Vibrations, speed variation, random orientation)
            if self.config.get("use_augmentation", True):
                # 1. Continuous sensor noise
                batch_x += torch.randn_like(batch_x) * 0.01

                # 2. Impulsive vibration noise (potholes/bumps) affecting accelerometer (first 3 channels)
                mask = torch.rand_like(batch_x[:, :, :3]) < 0.01
                batch_x[:, :, :3] += mask.float() * torch.randn_like(batch_x[:, :, :3]) * 1.5

                # 3. Random speed scaling (+/- 10%)
                scale = torch.empty(batch_x.shape[0], 1, 1, device=self.device).uniform_(0.9, 1.1)
                batch_x = batch_x * scale
                batch_y = batch_y * scale.squeeze(1)

                # 4. Small random yaw rotation (heading misalignment)
                angles = torch.empty(batch_x.shape[0], device=self.device).uniform_(-0.15, 0.15)
                cos_a = torch.cos(angles).unsqueeze(1)
                sin_a = torch.sin(angles).unsqueeze(1)

                # Rotate Accel (channels 0, 1) and Gyro (channels 3, 4)
                a_f = batch_x[:, :, 0].clone()
                a_r = batch_x[:, :, 1].clone()
                batch_x[:, :, 0] = a_f * cos_a - a_r * sin_a
                batch_x[:, :, 1] = a_f * sin_a + a_r * cos_a

                g_f = batch_x[:, :, 3].clone()
                g_r = batch_x[:, :, 4].clone()
                batch_x[:, :, 3] = g_f * cos_a - g_r * sin_a
                batch_x[:, :, 4] = g_f * sin_a + g_r * cos_a

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation. Returns (mean_loss, metrics_dict)."""
        if self.val_loader is None:
            return float("inf"), {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

        mean_loss = total_loss / max(num_batches, 1)

        # Compute additional metrics
        if all_preds:
            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)

            speed_pred = np.linalg.norm(preds, axis=1)
            speed_true = np.linalg.norm(targets, axis=1)
            speed_rmse = float(np.sqrt(np.mean((speed_pred - speed_true) ** 2)))

            metrics = {
                "speed_rmse": speed_rmse,
                "speed_mean_error": float(np.mean(np.abs(speed_pred - speed_true))),
            }
        else:
            metrics = {}

        return mean_loss, metrics

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dict with training history and best model info.
        """
        print(f"\n{'='*60}")
        print(f"  Training: {self.model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}, LR: {self.lr}")
        print(f"{'='*60}\n")

        start_time = time.time()
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["lr"].append(current_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"Metrics/{k}", v, epoch)

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
                marker = " ★ (best)"
            else:
                self.epochs_without_improvement += 1
                marker = ""

            # Print progress
            elapsed = time.time() - epoch_start
            speed_str = f", speed_rmse={val_metrics.get('speed_rmse', 0):.4f}" if val_metrics else ""
            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train={train_loss:.6f} | val={val_loss:.6f}{speed_str} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s{marker}"
            )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        total_time = time.time() - start_time
        print(f"\n[Trainer] Training complete in {total_time:.1f}s")
        print(f"[Trainer] Best validation loss: {self.best_val_loss:.6f}")

        if self.writer:
            self.writer.close()

        return {
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch,
            "total_time_s": total_time,
            "history": self.training_history,
        }

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "model_class": self.model.__class__.__name__,
        }

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Dict]:
        """
        Load a model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            device: Device to load to.

        Returns:
            Tuple of (model, checkpoint_info).
        """
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})

        # Reconstruct model
        model_class = checkpoint.get("model_class", "TCNVelocityEstimator")
        
        # Determine top-level config for model creation
        if "model" not in config:
            # Fallback if old format
            config["model"] = {
                "type": "lstm" if model_class == "LSTMVelocityEstimator" else "tcn",
                "input_channels": config.get("input_channels", 6),
                "output_dim": config.get("output_dim", 2),
                "lstm": {
                    "hidden_size": config.get("hidden_size", 128),
                    "num_layers": config.get("num_layers", 3),
                },
                "tcn": {
                    "num_channels": config.get("num_channels", [64, 64, 128, 128]),
                    "kernel_size": config.get("kernel_size", 7),
                }
            }
            
        model = create_model(config)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return model, checkpoint


def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create a model from config.

    Args:
        config: Full configuration dict (with 'model' section).

    Returns:
        Instantiated model.
    """
    model_config = config.get("model", {})
    model_type = model_config.get("type", "tcn")

    if model_type == "lstm":
        lstm_cfg = model_config.get("lstm", {})
        model = LSTMVelocityEstimator(
            input_channels=model_config.get("input_channels", 6),
            output_dim=model_config.get("output_dim", 2),
            hidden_size=lstm_cfg.get("hidden_size", 128),
            num_layers=lstm_cfg.get("num_layers", 3),
            dropout=lstm_cfg.get("dropout", 0.2),
            bidirectional=lstm_cfg.get("bidirectional", True),
            use_attention=lstm_cfg.get("use_attention", True),
        )
    else:
        tcn_cfg = model_config.get("tcn", {})
        model = TCNVelocityEstimator(
            input_channels=model_config.get("input_channels", 6),
            output_dim=model_config.get("output_dim", 2),
            num_channels=tcn_cfg.get("num_channels", [64, 64, 128, 128]),
            kernel_size=tcn_cfg.get("kernel_size", 7),
            dropout=tcn_cfg.get("dropout", 0.2),
            use_skip_connections=tcn_cfg.get("use_skip_connections", True),
        )

    print(f"[Model] Created {model.__class__.__name__} "
          f"with {model.count_parameters():,} parameters")

    return model
