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

# Import the actual pipeline functions to prove they work
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.pipeline import remove_outliers, apply_median_filter, butter_filter

def generate_noisy_signal(n_samples=500, fs=100):
    t = np.linspace(0, n_samples/fs, n_samples)
    
    # Base signal (e.g., true vehicle acceleration)
    base_signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    
    # 1. High frequency vibration noise (engine, road)
    noise = np.random.normal(0, 0.3, n_samples)
    
    # 2. Spikes / Outliers (bumps, sensor glitches)
    spikes = np.zeros(n_samples)
    spike_indices = np.random.choice(n_samples, size=10, replace=False)
    spikes[spike_indices] = np.random.choice([4.0, -4.0, 5.0, -5.0], size=10)
    
    raw_signal = base_signal + noise + spikes
    # Reshape for pipeline functions which expect (N, features)
    return t, raw_signal.reshape(-1, 1), base_signal

def main():
    fs = 100
    t, raw_accel, true_base = generate_noisy_signal(n_samples=400, fs=fs)
    
    # Apply Pipeline
    # 1. Outlier removal
    no_outliers = remove_outliers(raw_accel, threshold=3.0)
    
    # 2. Median Filter
    med_filtered = apply_median_filter(no_outliers, kernel_size=5)
    
    # 3. Butterworth Low-Pass
    fully_filtered = butter_filter(med_filtered, cutoff=5.0, fs=fs, btype='low')
    
    # Plotting
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Raw vs Base
    plt.subplot(2, 1, 1)
    plt.plot(t, raw_accel, color='red', alpha=0.6, label='Raw Sensor Data (Noisy + Spikes)')
    plt.plot(t, true_base, color='black', linestyle='--', linewidth=2, label='True Vehicle Dynamics')
    plt.title("Before Preprocessing: Raw Acceleration Data")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Filtered vs Base
    plt.subplot(2, 1, 2)
    plt.plot(t, fully_filtered, color='blue', linewidth=2, label='Processed Data (Outliers Removed + Filtered)')
    plt.plot(t, true_base, color='black', linestyle='--', linewidth=2, label='True Vehicle Dynamics')
    plt.title("After Preprocessing: Pipeline Output")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    out_path = os.path.join(os.path.dirname(__file__), "..", "preprocessing_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved preprocessing visualization to {out_path}")

if __name__ == "__main__":
    main()
