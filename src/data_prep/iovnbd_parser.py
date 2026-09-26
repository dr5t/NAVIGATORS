import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

def parse_synchronized_iovnbd(s_csv_path: str, v_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a synchronized S (Smartphone) and V (Vehicle) CSV pair from IO-VNBD.
    Extracts IMU inputs (Accel, Gyro) and Ground Truth targets (Velocity N/E).
    
    Returns:
        X: np.ndarray of shape (N, 6) -> [Accel X, Y, Z, Gyro X, Y, Z]
        Y: np.ndarray of shape (N, 2) -> [V_N, V_E] (m/s)
    """
    try:

        s_df = pd.read_csv(s_csv_path, encoding='latin1')
        v_df = pd.read_csv(v_csv_path, encoding='latin1')
    except Exception as e:
        print(f"Error loading {s_csv_path} or {v_csv_path}: {e}")
        return np.empty((0, 6)), np.empty((0, 2))
        

    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]
    

    accel_cols = [c for c in s_df.columns if 'ACCELEROMETER X' in c or 'ACCELEROMETER Y' in c or 'ACCELEROMETER Z' in c]
    gyro_cols = [c for c in s_df.columns if 'GYROSCOPE Yaw' in c or 'GYROSCOPE Pitch' in c or 'GYROSCOPE Roll' in c or 'GYROSCOPE X' in c or 'GYROSCOPE Y' in c or 'GYROSCOPE Z' in c]
    
    if len(accel_cols) < 3 or len(gyro_cols) < 3:
        print(f"Missing IMU columns in {s_csv_path}")
        return np.empty((0, 6)), np.empty((0, 2))
        
    acc_x = [c for c in accel_cols if 'X' in c][0]
    acc_y = [c for c in accel_cols if 'Y' in c][0]
    acc_z = [c for c in accel_cols if 'Z' in c][0]
    gyr_y = [c for c in gyro_cols if ('Yaw' in c or 'X' in c)][0]
    gyr_p = [c for c in gyro_cols if ('Pitch' in c or 'Y' in c)][0]
    gyr_r = [c for c in gyro_cols if ('Roll' in c or 'Z' in c)][0]
    
    imu_features = s_df[[acc_x, acc_y, acc_z, gyr_y, gyr_p, gyr_r]].values
    

    if 'Velocity (km/hr)' not in v_df.columns or 'Heading (degrees)' not in v_df.columns:
        print(f"Missing Velocity or Heading in {v_csv_path}")
        return np.empty((0, 6)), np.empty((0, 2))
        
    vel_kmh = v_df['Velocity (km/hr)'].values
    heading_deg = v_df['Heading (degrees)'].values
    

    min_len = min(len(imu_features), len(vel_kmh))
    imu_features = imu_features[:min_len]
    vel_kmh = vel_kmh[:min_len]
    heading_deg = heading_deg[:min_len]
    

    mask = ~np.isnan(imu_features).any(axis=1) & ~np.isnan(vel_kmh) & ~np.isnan(heading_deg)
    imu_features = imu_features[mask]
    vel_kmh = vel_kmh[mask]
    heading_deg = heading_deg[mask]
    

    vel_ms = np.asarray(vel_kmh, dtype=np.float64) / 3.6
    heading_rad = np.radians(heading_deg)
    
    v_n = vel_ms * np.cos(heading_rad)
    v_e = vel_ms * np.sin(heading_rad)
    
    Y = np.stack([v_n, v_e], axis=1)
    
    return imu_features.astype(np.float32), Y.astype(np.float32)

def discover_synchronized_sessions(base_dir: str) -> List[Tuple[str, str]]:
    """
    Discovers all V and S synchronized CSV pairs in the IO-VNBD dataset.
    Returns a list of tuples: (s_csv_path, v_csv_path)
    """
    pairs = []
    base_path = Path(base_dir)
    
    sync_dir = base_path / "Synchronised V abd S datasets"
    if not sync_dir.exists():
        print(f"Warning: {sync_dir} not found.")
        return pairs
        
    for v_file in sync_dir.rglob("V-*.csv"):
        session_id = v_file.stem.replace("V-", "")
        

        s_file = v_file.parent / f"S-{session_id}.csv"
        

        if not s_file.exists():
            s_file = v_file.parent.parent / "S-Dataset" / f"S-{session_id}.csv"
            
        if s_file.exists():
            pairs.append((str(s_file), str(v_file)))
            
    return sorted(pairs)

if __name__ == "__main__":

    pairs = discover_synchronized_sessions("data/IO-VNBD")
    print(f"Found {len(pairs)} synchronized sessions.")
    for s, v in pairs:
        print(f"Loading {os.path.basename(s)} and {os.path.basename(v)}...")
        X, Y = parse_synchronized_iovnbd(s, v)
        print(f"  Shape X: {X.shape}, Shape Y: {Y.shape}")
