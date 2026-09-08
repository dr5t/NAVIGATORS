# Detailed Project Guide

**Project Name:** Navigators  
**SIH Problem Statement ID:** SIH26168  

---

## 1. Project Overview

The Navigators project is an advanced AI/ML-driven Intelligent Dead Reckoning (IDR) system designed to solve the critical problem of GNSS (GPS/NavIC) denial. When a vehicle enters a tunnel, dense urban environment, or faces signal spoofing, traditional navigation fails. Our solution uses deep learning on ubiquitous smartphone IMU sensors (accelerometer and gyroscope) to accurately estimate vehicle motion, fusing it with map data to provide uninterrupted, highly precise navigation.

---

## 2. Theoretical Foundation

The system relies on four core technical pillars:

### 2.1 Deep Learning for Inertial Odometry
Traditional physical integration of IMU data causes rapid, uncorrectable drift due to compounding sensor noise. Instead, we treat motion estimation as a sequence-to-sequence problem. Using a **Temporal Convolutional Network (TCN)**, the AI learns the complex implicit patterns of human driving and vehicle dynamics, accurately predicting 2D velocity from a rolling window of IMU readings.

### 2.2 Signal Conditioning (Median Filtering)
A major challenge is mechanical noise—specifically impulsive shocks from potholes or speed bumps that can confuse the AI. Our pipeline incorporates **non-linear median filters** to mathematically isolate and remove these shock artifacts while preserving the underlying acceleration envelope.

### 2.3 Optimal Sensor Fusion (EKF)
To merge GNSS coordinates (when available) with the AI's velocity predictions, we employ a **15-State Extended Kalman Filter (EKF)**. The EKF tracks position, velocity, orientation, and sensor biases simultaneously. When GNSS drops, the EKF optimally integrates the AI predictions with high confidence, falling back to Dead Reckoning.

### 2.4 Kinematic Constraints & Snapping
Vehicles cannot move sideways (sideslip) under normal conditions. We enforce **Non-Holonomic Constraints (NHC)** to suppress lateral drift. Furthermore, when the vehicle stops, **Zero Velocity Updates (ZUPT)** reset accumulated errors to zero. Finally, **Geometric Map Matching** algorithms snap the estimated trajectory back to the mathematical centerlines of known road networks.

---

## 3. Engineering Architecture

### 3.1 Python Backend (Training & Benchmarking)
- **Framework:** PyTorch
- **Data Loaders:** Processes the massive IO-VNBD dataset (58 hours of driving data).
- **Augmentation Pipeline:** Injects continuous Gaussian noise, impulsive shocks, speed variation, and orientation misalignment to force the model to generalize across devices.
- **Evaluation:** Python scripts (`benchmark.py`) run rigorous GNSS-outage simulations to score the model against strict targets (e.g., < 100m drift over 1km).

### 3.2 JavaScript Edge Engine (Deployment)
- **Platform:** 100% Offline Progressive Web App (PWA).
- **Inference:** PyTorch models are converted to ONNX and run via WebAssembly (ONNX Runtime Web) on the device CPU.
- **Native Logic:** The complete 15-state EKF, Median Filters, ZUPT, NHC, and Map Matching algorithms are written in pure JavaScript, eliminating all server dependencies.
- **UI:** Leaflet.js renders the live trajectory locally.

---

## 4. Setup and Operation Guide

### 4.1 Training the Model
1. Ensure the IO-VNBD dataset is placed in `data/raw/` and processed using `scripts/process_data.py`.
2. Run `python scripts/train.py` to train the TCN model with augmentations.
3. Use `python src/edge/onnx_export.py` to compile the trained weights to `simulator/model.onnx`.

### 4.2 Running the Offline Simulator
1. Start a local HTTP server: `python -m http.server 8000` in the `simulator/` directory.
2. Open `http://localhost:8000` in a browser.
3. The simulator loads the ONNX model, initializes the JS EKF, and plays back historical sensor runs locally, demonstrating GNSS outage recovery natively in the browser.

### 4.3 Running the Live Edge App
1. Host the `simulator/` directory on a secure HTTPS server (required for mobile sensor access).
2. Open the URL on a mobile device and "Install to Home Screen" as a PWA.
3. Turn on Airplane Mode to prove offline capabilities.
4. Mount the phone in a vehicle, press "Start Live Navigation", and drive into a GNSS-denied area to observe the Intelligent Dead Reckoning.
