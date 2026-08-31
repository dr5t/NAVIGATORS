#!/usr/bin/env python3
"""
Navigators IDR — Training Script
Train TCN or LSTM velocity estimation models on IO-VNBD or synthetic data.

Usage:
    python scripts/train.py                          # Train with synthetic data
    python scripts/train.py --dataset ./data/io_vnbd # Train with IO-VNBD
    python scripts/train.py --model lstm              # Use LSTM instead of TCN
    python scripts/train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.trainer import Trainer, create_model
from data.synthetic_data import SyntheticDataGenerator, TrajectorySegment
from data.preprocessor import IMUPreprocessor


def create_synthetic_dataloaders(config: dict) -> dict:
    """Create train/val DataLoaders from synthetic data."""
    print("[Data] Generating synthetic training data...")

    gen = SyntheticDataGenerator(sample_rate=10.0, seed=42)
    preprocessor = IMUPreprocessor(sample_rate=10.0)

    all_windows = []
    all_labels = []

    # Generate multiple diverse scenarios
    scenarios = [
        # Highway driving
        [
            TrajectorySegment(duration=5.0, speed=0.0, acceleration=3.0),
            TrajectorySegment(duration=30.0, speed=25.0),
            TrajectorySegment(duration=5.0, speed=25.0, heading_rate=0.05),
            TrajectorySegment(duration=20.0, speed=25.0),
            TrajectorySegment(duration=5.0, speed=25.0, acceleration=-3.0),
            TrajectorySegment(duration=5.0, speed=0.0),
        ],
        # City driving with stops
        [
            TrajectorySegment(duration=3.0, speed=0.0, acceleration=2.0),
            TrajectorySegment(duration=10.0, speed=10.0),
            TrajectorySegment(duration=3.0, speed=10.0, heading_rate=0.3),
            TrajectorySegment(duration=8.0, speed=10.0),
            TrajectorySegment(duration=3.0, speed=10.0, acceleration=-3.0),
            TrajectorySegment(duration=5.0, speed=0.0),
            TrajectorySegment(duration=3.0, speed=0.0, acceleration=2.5),
            TrajectorySegment(duration=10.0, speed=8.0),
            TrajectorySegment(duration=4.0, speed=8.0, heading_rate=-0.25),
            TrajectorySegment(duration=10.0, speed=8.0),
        ],
        # Winding road
        [
            TrajectorySegment(duration=3.0, speed=0.0, acceleration=2.0),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=0.15),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=-0.2),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=0.1),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=-0.15),
            TrajectorySegment(duration=10.0, speed=12.0),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=0.25),
            TrajectorySegment(duration=5.0, speed=12.0, heading_rate=-0.3),
            TrajectorySegment(duration=10.0, speed=12.0),
        ],
        # Slow city with many turns
        [
            TrajectorySegment(duration=3.0, speed=0.0, acceleration=1.5),
            TrajectorySegment(duration=5.0, speed=5.0),
            TrajectorySegment(duration=3.0, speed=5.0, heading_rate=0.4),
            TrajectorySegment(duration=5.0, speed=5.0),
            TrajectorySegment(duration=3.0, speed=5.0, heading_rate=-0.4),
            TrajectorySegment(duration=5.0, speed=5.0),
            TrajectorySegment(duration=2.0, speed=5.0, acceleration=-2.0),
            TrajectorySegment(duration=3.0, speed=0.0),
            TrajectorySegment(duration=3.0, speed=0.0, acceleration=2.0),
            TrajectorySegment(duration=8.0, speed=7.0, heading_rate=0.1),
        ],
    ]

    window_size = config.get("data", {}).get("window_size", 200)
    window_stride = config.get("data", {}).get("window_stride", 50)

    for i, segments in enumerate(scenarios):
        # Generate multiple random variations of each scenario
        for seed in range(10):
            gen_var = SyntheticDataGenerator(sample_rate=10.0, seed=i * 100 + seed)
            scenario = gen_var.generate_full_scenario(segments=segments)

            accel = scenario["imu"]["accel"]
            gyro = scenario["imu"]["gyro"]
            velocities = scenario["trajectory"]["velocities"]

            # Preprocess
            windows, _ = preprocessor.preprocess(
                accel, gyro,
                window_size=window_size,
                stride=window_stride,
                do_normalize=False,
            )

            # Create labels (mean velocity per window)
            for j in range(len(windows)):
                start = j * window_stride
                end = start + window_size
                if end <= len(velocities):
                    label = np.mean(velocities[start:end], axis=0)
                    all_windows.append(windows[j])
                    all_labels.append(label)

    # Convert to tensors
    X = torch.tensor(np.array(all_windows), dtype=torch.float32)
    Y = torch.tensor(np.array(all_labels), dtype=torch.float32)

    print(f"[Data] Generated {len(X)} training windows")

    # Split: 80% train, 20% val
    split = int(0.8 * len(X))
    indices = torch.randperm(len(X))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = TensorDataset(X[train_idx], Y[train_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx])

    batch_size = config.get("training", {}).get("batch_size", 64)

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Navigators IDR velocity model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to IO-VNBD dataset (uses synthetic if not provided)")
    parser.add_argument("--model", type=str, default=None, choices=["tcn", "lstm"],
                        help="Model architecture (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", args.config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Apply CLI overrides
    if args.model:
        config.setdefault("model", {})["type"] = args.model
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr

    # Create model
    model = create_model(config)

    # Create data loaders
    if args.dataset and os.path.exists(args.dataset):
        from data.io_vnbd_loader import create_dataloaders
        loaders = create_dataloaders(args.dataset)
    else:
        print("[Data] No dataset provided — using synthetic data")
        loaders = create_synthetic_dataloaders(config)

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
        config=config.get("training", {}),
    )

    results = trainer.train()

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best val loss: {results['best_val_loss']:.6f}")
    print(f"  Total epochs:  {results['total_epochs']}")
    print(f"  Total time:    {results['total_time_s']:.1f}s")
    print(f"  Checkpoint:    checkpoints/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
