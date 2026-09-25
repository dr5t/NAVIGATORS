#!/usr/bin/env python3
"""
Navigators IDR - Real Dataset Processing Pipeline
Ingests raw smartphone trips, synchronizes sensors, removes gravity,
generates training targets (GNSS velocity), and applies trip-level splitting.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import json
from typing import Literal
import numpy as np
from scipy.signal import butter, filtfilt

# Directories
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw_trips")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "real_dataset")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "split_config.json")

def load_split_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "train": [],
        "val": [],
        "test": [],
        "default": "train"
    }

def butter_filter(data, cutoff, fs, btype: Literal['low', 'high', 'bandpass', 'bandstop'] = 'low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype=btype, analog=False, output='ba')
    return filtfilt(b, a, data, axis=0)

import scipy.signal
def remove_outliers(data, threshold=4.0):
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-6] = 1.0
    outliers = np.abs((data - median) / std) > threshold
    data_clean = np.copy(data)
    for i in range(data.shape[1]):
        col_outliers = outliers[:, i]
        if np.any(col_outliers):
            valid_indices = np.where(~col_outliers)[0]
            if len(valid_indices) > 0:
                outlier_indices = np.where(col_outliers)[0]
                data_clean[outlier_indices, i] = np.interp(
                    outlier_indices, valid_indices, data_clean[valid_indices, i]
                )
            else:
                data_clean[:, i] = median[i]
    return data_clean

def apply_median_filter(data, kernel_size=5):
    # Use scipy's medfilt. kernel_size must be odd.
    # We apply it along the time axis (axis 0), kernel for axis 1 is 1.
    return scipy.signal.medfilt(data, kernel_size=(kernel_size, 1))

def align_to_vehicle_frame(accel, gnss_speed, fs):
    """
    Estimates the rotation matrix from phone frame to vehicle frame.
    Returns: aligned_accel, rotation_matrix
    """
    # 1. Estimate Z (Gravity) - mean of accel over the entire trip (assuming mostly flat driving)
    Z = np.mean(accel, axis=0)
    Z = Z / np.linalg.norm(Z)
    
    # 2. Project acceleration onto horizontal plane
    # accel_horiz = accel - (accel dot Z) * Z
    accel_proj = np.dot(accel, Z)[:, np.newaxis] * Z
    accel_horiz = accel - accel_proj
    
    # 3. Estimate X (Forward)
    # Forward acceleration occurs when speed is increasing.
    # We find periods of positive acceleration from GNSS.
    dspeed = np.gradient(gnss_speed) * fs
    forward_mask = dspeed > 0.5 # accelerating at > 0.5 m/s^2
    
    if np.sum(forward_mask) > 10:
        # Mean horizontal acceleration during speedups gives forward direction
        X_raw = np.mean(accel_horiz[forward_mask], axis=0)
    else:
        # Fallback: PCA on horizontal acceleration (principal axis of motion)
        cov = np.cov(accel_horiz.T)
        evals, evecs = np.linalg.eig(cov)
        X_raw = evecs[:, np.argmax(evals)]
        # We can't easily resolve the sign without GNSS, assume it's positive.
        
    X = X_raw - np.dot(X_raw, Z) * Z # Ensure perfectly orthogonal to Z
    X = X / np.linalg.norm(X)
    
    # 4. Compute Y (Lateral)
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)
    
    # Rotation matrix (Phone to Vehicle)
    # Rows are the vehicle axes expressed in phone frame
    R = np.vstack((X, Y, Z))
    
    return R

def preprocess_imu(accel, gyro, speed, fs):
    """
    Applies the full preprocessing pipeline: Coordinate alignment, outlier removal,
    median filtering, low-pass filtering, and gravity removal.
    Returns: accel_clean, gyro_clean, R_p2v
    """
    # 1. Coordinate-Frame Transformation
    R_p2v = align_to_vehicle_frame(accel, speed, fs)
    
    # Rotate raw IMU data into vehicle frame
    accel_veh = (R_p2v @ accel.T).T
    gyro_veh = (R_p2v @ gyro.T).T
    
    # 2. Comprehensive IMU Preprocessing
    accel_no_outliers = remove_outliers(accel_veh, threshold=4.0)
    gyro_no_outliers = remove_outliers(gyro_veh, threshold=4.0)
    
    accel_med = apply_median_filter(accel_no_outliers, kernel_size=5)
    gyro_med = apply_median_filter(gyro_no_outliers, kernel_size=5)
    
    accel_low = butter_filter(accel_med, cutoff=20.0, fs=fs, btype='low')
    gyro_clean = butter_filter(gyro_med, cutoff=20.0, fs=fs, btype='low')
    
    accel_clean = butter_filter(accel_low, cutoff=0.1, fs=fs, btype='high')
    
    return accel_clean, gyro_clean, R_p2v

def process_trip(filepath, out_path, calibration_seconds=5, aligned=False):
    from evaluation.recording import load_recording, gnss_velocity
    from evaluation.preprocessing import prepare_features, PREPROCESSING_ID
    recording = load_recording(filepath)
    if recording.metadata.get('navigation_mode') == 'walking':
        raise ValueError('Walking recordings require a pedestrian training frontend; do not train the vehicle model on them.')
    features, first, rotation = prepare_features(recording, calibration_seconds, aligned)
    targets = np.full((len(features), 2), np.nan, dtype=np.float32)
    for i in range(first, len(recording.timestamps)):
        if recording.valid[i]:
            targets[i - first] = gnss_velocity(recording.gnss[i])
    if len(features) < 200:
        raise ValueError('Recording has fewer than 200 IMU samples after calibration')
    final_data = np.column_stack((features, targets))
    np.save(out_path, final_data)
    metadata = {
        'source_sha256': recording.digest, 'source': recording.path,
        'provenance': recording.metadata['provenance'],
        'source_dataset': recording.metadata.get('source_dataset'),
        'source_kind': recording.metadata.get('source_kind'),
        'preprocessing': PREPROCESSING_ID, 'sample_rate_hz': recording.sample_rate,
        'calibration_seconds': calibration_seconds, 'rotation': rotation.tolist(),
        'output_order': ['east', 'north'], 'first_sample': first,
    }
    with open(str(out_path) + '.json', 'w') as file:
        json.dump(metadata, file, indent=2)
    print(f'Processed {filepath}: {len(features)} causal samples -> {out_path}')
    return True

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "test"), exist_ok=True)
    
    config = load_split_config()
    assigned = [name for split in ('train', 'val', 'test') for name in config.get(split, [])]
    if len(assigned) != len(set(assigned)):
        raise ValueError('A trip is assigned to multiple train/validation/test splits')
    
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if not raw_files:
        print(f"No raw trip files found in {RAW_DIR}.")
        print("Use the 'Record Trip Data' tool in the simulator to generate trips.")
        return
        
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        split = str(config.get("default", "train"))
        for key in ["train", "val", "test"]:
            files_in_split = config.get(key, [])
            if isinstance(files_in_split, (list, tuple)) and filename in files_in_split:
                split = key
                break
        out_path = os.path.join(OUT_DIR, split, filename.replace(".json", ".npy"))
        process_trip(filepath, out_path)
        
if __name__ == "__main__":
    main()
