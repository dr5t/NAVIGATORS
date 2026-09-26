"""
Navigators IDR - Real-World Dataset Loading
Converts raw phone recordings into PyTorch DataLoaders.
"""
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class SensorTrajectoryDataset(Dataset):
    def __init__(self, trips: List[dict], window_size: int = 200, step_size: int = 20):
        self.window_size = window_size
        self.windows = []
        self.targets = []
        
        for trip in trips:
            data = trip.get('data', {})
            accel = data.get('accel', [])
            gyro = data.get('gyro', [])
            gnss = data.get('gnss', [])
            
            n = len(accel)
            if n < window_size:
                continue
                

            a_arr = np.array(accel, dtype=np.float32)
            g_arr = np.array(gyro, dtype=np.float32)
            

            imu_arr = np.concatenate([a_arr, g_arr], axis=1)
            

            for start in range(0, n - window_size, step_size):
                end = start + window_size
                


                target_gnss = gnss[end - 1]
                if target_gnss is None or target_gnss[3] is None or target_gnss[4] is None:

                    continue
                    
                speed = float(target_gnss[3])
                heading_deg = float(target_gnss[4])
                

                heading_rad = math.radians(heading_deg)
                v_n = speed * math.cos(heading_rad)
                v_e = speed * math.sin(heading_rad)
                
                window_data = imu_arr[start:end]
                
                self.windows.append(window_data)
                self.targets.append([v_n, v_e])
                
        if self.windows:
            self.windows = np.array(self.windows, dtype=np.float32)

            self.windows = np.transpose(self.windows, (0, 2, 1))
            self.targets = np.array(self.targets, dtype=np.float32)
        else:
            self.windows = np.zeros((0, 6, window_size), dtype=np.float32)
            self.targets = np.zeros((0, 2), dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows[idx]), torch.from_numpy(self.targets[idx])

def create_dataloaders(recordings_dir: Path, window_size: int = 200, batch_size: int = 64) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Reads all trip_*.json files, splits them by trip (to avoid data leakage),
    and creates DataLoaders.
    """
    files = sorted(recordings_dir.glob("trip_*.json"))
    trips = []
    for f in files:
        try:
            with open(f, 'r') as file:
                trips.append(json.load(file))
        except Exception as e:
            print(f"Skipping {f}: {e}")
            

    n = len(trips)
    train_end = max(1, int(0.7 * n))
    val_end = max(train_end + 1, int(0.85 * n))
    
    train_trips = trips[:train_end]
    val_trips = trips[train_end:val_end]
    test_trips = trips[val_end:]
    
    train_ds = SensorTrajectoryDataset(train_trips, window_size=window_size)
    val_ds = SensorTrajectoryDataset(val_trips, window_size=window_size)
    test_ds = SensorTrajectoryDataset(test_trips, window_size=window_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True) if len(train_ds) else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if len(test_ds) else None
    
    return train_loader, val_loader, test_loader
