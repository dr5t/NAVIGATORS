import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryWindowDataset(Dataset):
    """
    Loads complete trajectories and provides sliding windows of IMU data
    to predict the velocity at the last timestep of the window.
    """
    def __init__(self, data_dir: str, window_size: int = 200, step_size: int = 10, stats: dict = None):
        """
        Args:
            data_dir: Directory containing .npy trajectory files.
            window_size: Number of timesteps per window.
            step_size: Stride for sliding window.
            stats: Dictionary with 'mean' and 'std' for normalization. Computed if None.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        
        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")
            
        self.samples = []
        
        # Load all data into memory (since synthetic dataset is small enough)
        # For huge datasets, we would map or lazily load
        all_imu = []
        for file in self.files:
            data = np.load(file) # Shape (N, 8) -> [accel(3), gyro(3), vel(2)]
            N = len(data)
            
            if N < self.window_size:
                continue
                
            all_imu.append(data[:, :6])
                
            # Create window indices
            for start in range(0, N - self.window_size + 1, self.step_size):
                self.samples.append({
                    "data": data[start:start + self.window_size],
                    "target_idx": self.window_size - 1 # We predict the velocity at the end of the window
                })
                
        # Compute normalization statistics if not provided
        if stats is None:
            all_imu_cat = np.concatenate(all_imu, axis=0)
            self.mean = np.mean(all_imu_cat, axis=0).astype(np.float32)
            self.std = np.std(all_imu_cat, axis=0).astype(np.float32)
            self.std[self.std < 1e-6] = 1.0 # prevent div by zero
            self.stats = {"mean": self.mean, "std": self.std}
        else:
            self.stats = stats
            self.mean = stats["mean"]
            self.std = stats["std"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = sample["data"]
        
        imu = data[:, :6].astype(np.float32)
        # Normalize IMU
        imu = (imu - self.mean) / self.std
        
        # Target is velocity at the end of the window [v_east, v_north]
        # In tcn_model.py, output is usually [v_north, v_east], but synthetic_data.py has [v_east, v_north].
        # Let's keep it [v_east, v_north]
        vel = data[sample["target_idx"], 6:].astype(np.float32)
        
        return torch.tensor(imu), torch.tensor(vel)

def get_dataloaders(base_dir: str, batch_size: int = 128, window_size: int = 200, step_size: int = 10):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    train_dataset = TrajectoryWindowDataset(train_dir, window_size, step_size)
    # Use train stats to normalize val and test
    val_dataset = TrajectoryWindowDataset(val_dir, window_size, step_size, stats=train_dataset.stats)
    test_dataset = TrajectoryWindowDataset(test_dir, window_size, step_size, stats=train_dataset.stats)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.stats
