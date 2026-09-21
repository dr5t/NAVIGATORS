import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import random

# Import our new parser
from src.data_prep.iovnbd_parser import discover_synchronized_sessions, parse_synchronized_iovnbd

import json
import os

class IOVNBDDataset(Dataset):
    def __init__(self, session_pairs: List[Tuple[str, str]], window_size: int = 200, stride: int = 20,
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        """
        PyTorch Dataset for IO-VNBD synchronized sessions.
        
        Args:
            session_pairs: List of (s_csv_path, v_csv_path)
            window_size: Length of the sliding window in samples (default 200)
            stride: Step size between windows (default 20)
            mean: Precomputed mean for Z-score normalization (optional)
            std: Precomputed std for Z-score normalization (optional)
        """
        self.window_size = window_size
        self.stride = stride
        
        self.mean: Optional[np.ndarray] = mean
        self.std: Optional[np.ndarray] = std
        
        self.windows_x = []
        self.targets_y = []
        
        for s_csv, v_csv in session_pairs:
            # Parse the synchronized pair
            X, Y = parse_synchronized_iovnbd(s_csv, v_csv)
            n_samples = X.shape[0]
            
            if n_samples < window_size:
                continue
                
            # Create sliding windows
            for start in range(0, n_samples - window_size, stride):
                end = start + window_size
                window_x = X[start:end]
                
                # The target is the velocity at the end of the window (simulating real-time estimation)
                target_y = Y[end - 1]
                
                self.windows_x.append(window_x)
                self.targets_y.append(target_y)
                
        if self.windows_x:
            self.windows_x = np.array(self.windows_x, dtype=np.float32)
            self.targets_y = np.array(self.targets_y, dtype=np.float32)
            
            # Calculate mean and std from this dataset if not provided (i.e. Training set)
            if self.mean is None or self.std is None:
                # Shape is (N, window_size, 6), calculate per-channel across all samples and time steps
                self.mean = np.mean(self.windows_x, axis=(0, 1), keepdims=True)
                self.std = np.std(self.windows_x, axis=(0, 1), keepdims=True)
                self.std[self.std == 0] = 1.0  # Prevent division by zero
                
            # Apply Z-score normalization
            self.windows_x = (self.windows_x - self.mean) / self.std
            
        else:
            self.windows_x = np.zeros((0, window_size, 6), dtype=np.float32)
            self.targets_y = np.zeros((0, 2), dtype=np.float32)
            if self.mean is None:
                self.mean = np.zeros((1, 1, 6), dtype=np.float32)
                self.std = np.ones((1, 1, 6), dtype=np.float32)

    def __len__(self):
        return len(self.windows_x)

    def __getitem__(self, index):
        return torch.from_numpy(self.windows_x[index]), torch.from_numpy(self.targets_y[index])


def create_iovnbd_dataloaders(base_dir: str = "data/IO-VNBD", window_size: int = 200, batch_size: int = 64, stats_dir: str = "checkpoints") -> Optional[Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Discovers all IO-VNBD synchronized sessions, splits them securely (to prevent leakage),
    and creates DataLoaders.
    """
    pairs = discover_synchronized_sessions(base_dir)
    
    if not pairs:
        print("ERROR: No IO-VNBD synchronized sessions found.")
        return None
        
    # Sort for deterministic behavior, then shuffle with a fixed seed
    pairs = sorted(pairs)
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = max(1, int(0.7 * n))
    val_end = max(train_end + 1, int(0.85 * n))
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"Dataset split (Sessions): Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Create Train dataset and compute normalization stats
    train_ds = IOVNBDDataset(train_pairs, window_size=window_size)
    
    # Apply exactly the same stats to Val and Test
    val_ds = IOVNBDDataset(val_pairs, window_size=window_size, mean=train_ds.mean, std=train_ds.std)
    test_ds = IOVNBDDataset(test_pairs, window_size=window_size, mean=train_ds.mean, std=train_ds.std)
    
    # Save normalization stats for Android inference
    os.makedirs(stats_dir, exist_ok=True)
    assert train_ds.mean is not None and train_ds.std is not None
    stats_dict = {
        "mean": train_ds.mean.flatten().tolist(),
        "std": train_ds.std.flatten().tolist(),
        "features": ["ACCELEROMETER X", "ACCELEROMETER Y", "ACCELEROMETER Z", 
                     "GYROSCOPE X", "GYROSCOPE Y", "GYROSCOPE Z"],
        "method": "z-score"
    }
    with open(os.path.join(stats_dir, "norm_stats.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)
    
    print(f"Dataset split (Windows): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
