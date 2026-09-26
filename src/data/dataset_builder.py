#!/usr/bin/env python3
"""
Navigators IDR - Synthetic Dataset Builder
Generates a massive dataset of synthetic trajectories for training the TCN model.
Splits data strictly at the trajectory level into train/val/test directories.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from data.synthetic_data import SyntheticDataGenerator, TrajectorySegment

def generate_random_segments(rng: np.random.Generator, num_segments: int = 15) -> list:
    segments = []
    speed = rng.uniform(0.0, 5.0) # start slow or stopped
    for _ in range(num_segments):
        duration = rng.uniform(2.0, 20.0)
        
        # 10% chance to stop
        if rng.random() < 0.1:
            accel = -speed / 2.0 if speed > 0 else 0.0
            segments.append(TrajectorySegment(duration=duration, speed=speed, acceleration=accel))
            speed = 0.0
            continue
            
        # Random acceleration
        target_speed = rng.uniform(5.0, 25.0) # up to 90 km/h
        accel = (target_speed - speed) / (duration / 2.0)
        accel = np.clip(accel, -3.0, 2.5) # realistic limits
        
        # Random heading change
        heading_rate = rng.normal(0, 0.1) if rng.random() < 0.5 else 0.0
        if rng.random() < 0.2: # sharp turn
            heading_rate = rng.uniform(-0.3, 0.3)
            
        segments.append(TrajectorySegment(
            duration=duration, 
            speed=speed, 
            heading_rate=heading_rate, 
            acceleration=accel
        ))
        speed = max(0.0, speed + accel * duration)
        
    return segments

def build_dataset(base_dir: str, num_train: int = 200, num_val: int = 40, num_test: int = 40):
    os.makedirs(base_dir, exist_ok=True)
    
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }
    
    rng = np.random.default_rng(42)
    sample_rate = 10.0
    
    for split_name, count in splits.items():
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Generating {count} trajectories for {split_name}...")
        
        for i in tqdm(range(count)):
            gen = SyntheticDataGenerator(sample_rate=sample_rate, seed=int(rng.integers(1000000)))
            segments = generate_random_segments(rng, num_segments=int(rng.integers(10, 30)))
            
            # We don't need GNSS outages for TCN training, just true IMU and true velocity
            scenario = gen.generate_full_scenario(segments=segments, outage_ranges=[])
            
            # We only need IMU (accel, gyro) and Ground Truth Velocity
            accel = scenario["imu"]["accel"]
            gyro = scenario["imu"]["gyro"]
            vel = scenario["trajectory"]["velocities"]
            
            # Combine into a single array: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, v_east, v_north]
            # Shape: (N, 8)
            data = np.hstack([accel, gyro, vel])
            
            file_path = os.path.join(split_dir, f"traj_{i:04d}.npy")
            np.save(file_path, data)

if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic_trajectories")
    build_dataset(dataset_dir)
    print("Dataset generation complete!")
