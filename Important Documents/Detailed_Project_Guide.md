# Navigators IDR — Detailed Project Guide

## 1. Introduction
This guide explains the exact components and data flow of the Navigators IDR system.

## 2. Component Walkthrough
### 2.1 Preprocessing and Alignment
Raw IMU data is noisy and arbitrary in orientation. The system first estimates gravity to establish the *Down* vector. Then, by observing acceleration during movement, it establishes the *Forward* vector. This allows the system to rotate all raw IMU data into the vehicle frame.

### 2.2 Deep Learning (TCN) Velocity Estimation
Instead of raw double integration (which drifts quadratically in seconds), the system windows 200 timesteps (2 seconds) of aligned IMU data and passes it to an ONNX-exported Temporal Convolutional Network. The network outputs a `[v_forward, v_lateral]` prediction.

### 2.3 The 15-State EKF
The EKF state vector includes:
- Position (3D)
- Velocity (3D)
- Attitude (3D Quaternions/Euler)
- Accel Bias (3D)
- Gyro Bias (3D)

The EKF prediction step uses raw IMU. The update step fuses the TCN velocity prediction (as a pseudo-measurement), NHC, ZUPT, and GNSS (when available).

## 3. Data Collection Protocol
When generating the training dataset, variety is critical to generalization.
- **Do not train on one person's driving only.**
- Collect multiple trips across different environments (Urban, Highway, Residential).
- Capture different traffic states (Free-flowing, Stop-and-Go).
- Ensure the phone is mounted in completely different physical orientations across trips to force the Phone-to-Vehicle Triad alignment algorithm to generalize.
- **Labels**: During training *only*, the GNSS speed and heading are mathematically converted into `vel_forward` and `vel_lateral` to serve as the ground truth labels for the TCN.

## 4. Running the Validation Suite
The `scripts/benchmark.py` script runs the entire system through real-world scenarios. It dynamically injects artificial GNSS outages (e.g. 100m, 250m) and strictly cuts off the EKF's access to the GNSS tracks. The system must natively dead-reckon using only the TCN predictions, and the resulting positional error is measured against the hidden GNSS ground truth.
