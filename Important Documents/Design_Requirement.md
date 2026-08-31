# Design Requirements
**Project Name:** Navigators
**SIH Problem Statement ID:** SIH26168

## 1. System Architecture
The system employs an edge-to-cloud architecture focused heavily on edge inference to ensure real-time responsiveness when GNSS is denied.

### 1.1 Edge Engine (Smartphone / External IMU)
*   **Sensor Interface:** Access raw data from smartphone sensors (Accelerometer, Gyroscope) at 10 Hz. (Up to 200 Hz for external FOG-IMU).
*   **Preprocessing Module:** Noise filtering, vibration dampening, and gravity vector extraction. Calculates phone-to-vehicle alignment.
*   **AI Inference Engine:** A lightweight neural network (e.g., TCN, LSTM, or Transformer) designed to predict 2D velocity or displacement from windowed IMU data.
*   **Sensor Fusion (EKF):** An Extended Kalman Filter to optimally fuse AI-predicted velocities with GNSS readings (when available).
*   **Kinematic Constraints Module:** Enforces Non-Holonomic Constraints (NHC).
*   **UI/Visualizer:** Renders the estimated trajectory continuously without jumping or freezing.

### 1.2 Processing Flow
1.  **GNSS Available:** GNSS + IMU -> AI/ML filtering -> Fusion (EKF) -> Map matching -> Position
2.  **GNSS Lost:** IMU -> AI speed/motion estimation -> Dead reckoning -> NHC -> Map matching -> Position
3.  **GNSS Returns:** Dead reckoning -> Re-acquisition -> GNSS + INS fusion -> Corrected continuous position

## 2. Software Requirements
*   **AI/ML Framework:** PyTorch for training; ONNX or TensorFlow Lite for edge deployment.
*   **Data Processing:** Pandas, NumPy, SciPy for preprocessing.
*   **Navigation & Map Matching:** OSRM, Valhalla, or custom map matching heuristics.

## 3. Hardware Requirements
*   **Development:** Workstation or Cloud instance with GPU (NVIDIA) for training on the 1.5GB IO-VNBD dataset.
*   **Deployment:** Modern smartphone (Android/iOS) with capable IMU, or dedicated edge hardware for external IMU integration.
