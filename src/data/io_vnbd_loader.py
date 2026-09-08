"""
Navigators IDR — IO-VNBD Dataset Loader
PyTorch Dataset for the Inertial Odometry Vehicle Navigation Benchmark Dataset.

The IO-VNBD dataset contains smartphone IMU recordings collected across
the UK, Nigeria, and France (~58 hours, ~4400 km).

Expected directory structure:
    io_vnbd/
    ├── train/
    │   ├── sequence_001/
    │   │   ├── imu.csv         # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    │   │   ├── gnss.csv        # timestamp, lat, lon, alt, speed, bearing
    │   │   └── ground_truth.csv # timestamp, lat, lon, alt, v_north, v_east, heading
    │   ├── sequence_002/
    │   └── ...
    ├── val/
    └── test/
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessor import IMUPreprocessor


class IOVNBDDataset(Dataset):
    """
    PyTorch Dataset for the IO-VNBD benchmark.

    Loads windowed IMU sequences and corresponding ground-truth velocity
    labels for training the velocity estimation model.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        window_size: int = 200,
        window_stride: int = 50,
        sample_rate: float = 10.0,
        normalize: bool = True,
        preprocessor: Optional[IMUPreprocessor] = None,
        transform=None,
    ):
        """
        Args:
            root_dir: Path to the IO-VNBD dataset root.
            split: 'train', 'val', or 'test'.
            window_size: Number of IMU samples per window.
            window_stride: Stride between consecutive windows.
            sample_rate: IMU sample rate in Hz.
            normalize: Whether to z-score normalize IMU data.
            preprocessor: Optional custom preprocessor. Creates default if None.
            transform: Optional PyTorch transform to apply.
        """
        self.root_dir = root_dir
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.transform = transform

        self.preprocessor = preprocessor or IMUPreprocessor(sample_rate=sample_rate)

        # Load and preprocess all sequences
        self.windows = []      # List of (window_size, 6) IMU windows
        self.labels = []       # List of (2,) velocity vectors [v_north, v_east]
        self.seq_ids = []      # Which sequence each window came from

        self._load_dataset()

    def _load_dataset(self):
        """Load all sequences from the split directory."""
        split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.exists(split_dir):
            print(f"[IO-VNBD] Split directory not found: {split_dir}")
            print("[IO-VNBD] Use SyntheticDataGenerator for testing without dataset.")
            return

        sequence_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        print(f"[IO-VNBD] Loading {len(sequence_dirs)} sequences from '{self.split}'...")

        global_imu_data = []

        for seq_dir_name in sequence_dirs:
            seq_path = os.path.join(split_dir, seq_dir_name)

            try:
                imu_data, gt_velocity = self._load_sequence(seq_path)
                if imu_data is None:
                    continue

                # Preprocess: filter, align, remove gravity, window
                windows, metadata = self.preprocessor.preprocess(
                    accel=imu_data[:, :3],
                    gyro=imu_data[:, 3:6],
                    window_size=self.window_size,
                    stride=self.window_stride,
                    do_normalize=False,  # Normalize globally after
                )

                # Create velocity labels for each window (use mean velocity in window)
                for i in range(len(windows)):
                    start = i * self.window_stride
                    end = start + self.window_size
                    if gt_velocity is not None and end <= len(gt_velocity):
                        # Mean velocity over the window
                        label = np.mean(gt_velocity[start:end], axis=0)
                        self.windows.append(windows[i])
                        self.labels.append(label)
                        self.seq_ids.append(seq_dir_name)

                global_imu_data.append(imu_data)

            except Exception as e:
                print(f"[IO-VNBD] Error loading {seq_dir_name}: {e}")
                continue

        # Global normalization
        if self.normalize and len(self.windows) > 0:
            all_windows = np.stack(self.windows)
            flat = all_windows.reshape(-1, all_windows.shape[-1])
            self._mean = np.mean(flat, axis=0)
            self._std = np.std(flat, axis=0)
            self._std[self._std < 1e-8] = 1.0

            for i in range(len(self.windows)):
                self.windows[i] = (self.windows[i] - self._mean) / self._std

        print(f"[IO-VNBD] Loaded {len(self.windows)} windows from {len(sequence_dirs)} sequences.")

    def _load_sequence(self, seq_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load a single sequence's IMU and ground truth data.

        Args:
            seq_path: Path to the sequence directory.

        Returns:
            Tuple of (imu_data, gt_velocity) or (None, None) on failure.
        """
        imu_file = os.path.join(seq_path, "imu.csv")
        gt_file = os.path.join(seq_path, "ground_truth.csv")

        if not os.path.exists(imu_file):
            # Try alternative file names
            for alt in ["imu_data.csv", "sensor_data.csv", "accelerometer.csv"]:
                alt_path = os.path.join(seq_path, alt)
                if os.path.exists(alt_path):
                    imu_file = alt_path
                    break
            else:
                return None, None

        if not os.path.exists(gt_file):
            for alt in ["gt.csv", "reference.csv", "ground_truth_data.csv"]:
                alt_path = os.path.join(seq_path, alt)
                if os.path.exists(alt_path):
                    gt_file = alt_path
                    break
            else:
                return None, None

        # Load IMU data
        imu_df = pd.read_csv(imu_file)
        # Expect columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        imu_cols = [c for c in imu_df.columns if c.lower() != "timestamp"]
        if len(imu_cols) < 6:
            return None, None
        imu_data = imu_df[imu_cols[:6]].values.astype(np.float64)

        # Load ground truth velocity
        gt_df = pd.read_csv(gt_file)
        # Look for velocity columns
        vel_cols = []
        for pattern in [("v_north", "v_east"), ("vn", "ve"), ("vel_n", "vel_e")]:
            matches = [c for c in gt_df.columns if c.lower() in [p.lower() for p in pattern]]
            if len(matches) >= 2:
                vel_cols = matches[:2]
                break

        if len(vel_cols) < 2:
            # Fallback: compute velocity from lat/lon if available
            if "speed" in [c.lower() for c in gt_df.columns] and "bearing" in [c.lower() for c in gt_df.columns]:
                speed_col = [c for c in gt_df.columns if c.lower() == "speed"][0]
                bearing_col = [c for c in gt_df.columns if c.lower() == "bearing"][0]
                speed = gt_df[speed_col].values
                bearing = np.radians(gt_df[bearing_col].values)
                gt_velocity = np.column_stack([
                    speed * np.cos(bearing),  # v_north
                    speed * np.sin(bearing),  # v_east
                ])
            else:
                return None, None
        else:
            gt_velocity = gt_df[vel_cols].values.astype(np.float64)

        # Align lengths
        min_len = min(len(imu_data), len(gt_velocity))
        imu_data = imu_data[:min_len]
        gt_velocity = gt_velocity[:min_len]

        return imu_data, gt_velocity

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single (window, label) pair.

        Returns:
            Tuple of:
                - window: (window_size, 6) float32 tensor
                - label: (2,) float32 tensor [v_north, v_east]
        """
        window = torch.tensor(self.windows[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            window = self.transform(window)

        return window, label

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Return normalization mean and std for inference preprocessing."""
        return {
            "mean": getattr(self, "_mean", np.zeros(6)),
            "std": getattr(self, "_std", np.ones(6)),
        }


def create_dataloaders(
    root_dir: str,
    window_size: int = 200,
    window_stride: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    sample_rate: float = 10.0,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for IO-VNBD.

    Args:
        root_dir: Path to IO-VNBD dataset root.
        window_size: Samples per window.
        window_stride: Window stride.
        batch_size: Batch size.
        num_workers: DataLoader worker threads.
        sample_rate: IMU sample rate.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = IOVNBDDataset(
            root_dir=root_dir,
            split=split,
            window_size=window_size,
            window_stride=window_stride,
            sample_rate=sample_rate,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders
