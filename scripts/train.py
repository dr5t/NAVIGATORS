#!/usr/bin/env python3
"""
Navigators IDR — Model Training Script
Trains the TCN model on the generated dataset.
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data.data_loader import get_dataloaders, validate_training_splits
from evaluation.preprocessing import PREPROCESSING_ID
from models.tcn_model import TCNVelocityEstimator
from models.trainer import Trainer

def main():
    print("="*60)
    print("  NAVIGATORS IDR — MODEL TRAINING (PHASE 3)")
    print("="*60)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "real_dataset")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Loading data from {data_dir}...")
    train_loader, val_loader, test_loader, stats = get_dataloaders(
        data_dir, batch_size=256, window_size=200, step_size=10
    )
    
    # Save stats for inference
    dataset_contract = validate_training_splits(data_dir)
    stats_dict = {
        "preprocessing": PREPROCESSING_ID,
        "sample_rate_hz": dataset_contract['sample_rate_hz'],
        "source_hashes": dataset_contract['source_hashes'],
        "window_size": 200,
        "output_order": ["east", "north"],
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist()
    }
    with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w") as f:
        json.dump(stats_dict, f)
        
    print(f"Normalization Stats Saved.")
    
    config = {
        "epochs": 15, # Fast training for this recovery phase
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "early_stopping_patience": 5,
        "loss": "mse_angular",
        "angular_loss_weight": 0.3,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "checkpoint_dir": checkpoint_dir,
        "use_augmentation": True
    }
    
    model = TCNVelocityEstimator(
        input_channels=6,
        output_dim=2,
        num_channels=[32, 64, 128], # Lightweight
        kernel_size=5,
        dropout=0.2
    )
    
    # Pack model configuration into trainer config for checkpoint saving
    config["data_contract"] = stats_dict
    config["model"] = {
        "type": "tcn",
        "input_channels": 6,
        "output_dim": 2,
        "tcn": {
            "num_channels": [32, 64, 128],
            "kernel_size": 5,
            "dropout": 0.2
        }
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()
    
    # Final Test Set Evaluation
    print("\nEvaluating on Test Set...")
    trainer.model, _ = Trainer.load_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"), device=trainer.device)
    trainer.val_loader = test_loader
    test_loss, test_metrics = trainer.validate()
    print(f"Test Loss: {test_loss:.4f}")
    if test_metrics:
        for k, v in test_metrics.items():
            print(f"Test {k}: {v:.4f}")

if __name__ == "__main__":
    main()
