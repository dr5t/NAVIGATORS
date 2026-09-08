# Design Requirements & Architecture

## 1. System Overview
The Navigators IDR system is composed of an advanced Python-based Navigation Engine executing an Extended Kalman Filter (EKF) and Temporal Convolutional Network (TCN) pipeline to maintain accurate GNSS-denied navigation.

## 2. Core Components

### 2.1 Sensor Preprocessing
- **Inputs:** 3-axis Accelerometer, 3-axis Gyroscope.
- **Filtering:** Applies a Median filter (window=5) to remove spike noise, followed by a 4th-order low-pass Butterworth filter (cutoff=2.5 Hz) to isolate vehicle dynamics from high-frequency vibrations.
- **Alignment:** Uses Principal Component Analysis (PCA) to align the smartphone's arbitrary coordinate frame to the vehicle's forward axis, alongside gravity estimation to determine Pitch and Roll.

### 2.2 Deep Learning Velocity Model (TCN)
- **Architecture:** Temporal Convolutional Network (TCN).
- **Input:** 200-sample sliding window of aligned 6-DOF IMU data (approx. 20 seconds at 10Hz).
- **Output:** 2D horizontal velocity vector ($v_{east}$, $v_{north}$).
- **Deployment:** Exported as an ONNX model for high-performance, edge-optimized inference.

### 2.3 Extended Kalman Filter (EKF)
The central data fusion hub.
- **State Vector ($x_{15 \times 1}$):** 
  - Position ($x, y, z$)
  - Velocity ($v_x, v_y, v_z$)
  - Orientation (Roll, Pitch, Yaw)
  - Accelerometer Bias ($b_{ax}, b_{ay}, b_{az}$)
  - Gyroscope Bias ($b_{gx}, b_{gy}, b_{gz}$)
- **Process Model:** Integrates raw IMU data using the strapdown inertial navigation equations.
- **Measurement Model:** Fuses absolute GNSS fixes when available. When GNSS is denied, it fuses the AI-predicted 2D velocity as a highly-weighted pseudo-measurement.

### 2.4 Constraints
- **NHC (Non-Holonomic Constraints):** Applies pseudo-measurements assuming lateral velocity ($v_y \approx 0$) and vertical velocity ($v_z \approx 0$) are zero in the vehicle frame.
- **ZUPT (Zero-Velocity Update):** Uses acceleration variance thresholds to detect stops. When triggered, forces velocity to zero and heavily constrains covariance growth.

### 2.5 Map Matching
- A geometric and topological engine that uses a local offline road network graph.
- Uses distance and heading thresholds to snap the EKF's mathematical output to the most likely logical road segment.

## 3. Data Flow
1. Device Sensors -> API Endpoint (`POST /sensor/batch` or `/ws`).
2. Data passes to `preprocessor.py`.
3. Filtered data generates a 200-frame window.
4. `tcn_model.py` evaluates the window, yielding $v_{AI}$.
5. `ekf.py` integrates the raw IMU data, then updates via GNSS, NHC, ZUPT, and $v_{AI}$.
6. `map_matching.py` snaps the EKF's position to the grid.
7. System returns JSON payload with snapped coordinates and confidence metrics.
