#!/usr/bin/env python3
"""
Navigators IDR - Preprocessing Visualization
Generates a visual comparison of Raw vs Filtered IMU data to demonstrate
outlier handling, median filtering, and Butterworth low-pass filtering.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.pipeline import remove_outliers, apply_median_filter, butter_filter

def generate_noisy_signal(n_samples=500, fs=100, seed=42):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, n_samples/fs, n_samples)
    base_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    noise = np.random.normal(0, 0.3, n_samples)
    spikes = np.zeros(n_samples)
    spike_indices = np.random.choice(n_samples, size=10, replace=False)
    spikes[spike_indices] = np.random.choice([4.0, -4.0, 5.0, -5.0], size=10)
    raw_signal = base_signal + noise + spikes
    return t, raw_signal.reshape(-1, 1), base_signal

def main():
    fs = 100
    t, raw_accel, true_base = generate_noisy_signal(n_samples=400, fs=fs, seed=42)

    no_outliers = remove_outliers(raw_accel, threshold=3.0)
    med_filtered = apply_median_filter(no_outliers, kernel_size=5)
    fully_filtered = butter_filter(med_filtered, cutoff=5.0, fs=fs, btype='low')

    raw_flat = raw_accel.flatten()
    filt_flat = fully_filtered.flatten()
    raw_rmse = float(np.sqrt(np.mean((raw_flat - true_base) ** 2)))
    filt_rmse = float(np.sqrt(np.mean((filt_flat - true_base) ** 2)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(t, raw_accel, color='red', alpha=0.6, label='Raw Sensor Data (Noisy + Spikes)')
    ax1.plot(t, true_base, color='black', linestyle='--', linewidth=2, label='True Vehicle Dynamics')
    ax1.set_title(f"Before Preprocessing: Raw Acceleration Data (RMSE: {raw_rmse:.2f} m/s²)")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, fully_filtered, color='blue', linewidth=2, label='Processed Data (Outliers Removed + Filtered)')
    ax2.plot(t, true_base, color='black', linestyle='--', linewidth=2, label='True Vehicle Dynamics')
    ax2.set_title(f"After Preprocessing: Pipeline Output (RMSE: {filt_rmse:.2f} m/s²)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    

    out_path = os.path.join(os.path.dirname(__file__), "..", "preprocessing_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved preprocessing visualization to {out_path}")

if __name__ == "__main__":
    main()
