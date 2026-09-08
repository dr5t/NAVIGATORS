# Navigators IDR — Project Requirements Document (PRD)

## 1. Overview
The Navigators Intelligent Dead Reckoning (IDR) system provides continuous vehicle navigation using smartphone sensors during GNSS outages.

## 2. Functional Requirements
- **FR1**: Collect IMU data (Accel/Gyro) at minimum 50Hz (target 100Hz).
- **FR2**: Detect GNSS loss and seamlessly switch to dead reckoning.
- **FR3**: Estimate 2D vehicle velocity using a deep learning model.
- **FR4**: Fuse AI velocity with IMU using a 15-state EKF.
- **FR5**: Run completely offline without cellular or backend API dependencies in the final production app.

## 3. Non-Functional Requirements
- **NFR1**: End-to-end latency per window must be under 100ms (10Hz). Current ONNX benchmark: ~0.7ms.
- **NFR2**: System must not leak ground truth data into the estimator during validation.
- **NFR3**: The application must be lightweight and battery-efficient on mid-range smartphones.

## 4. Excluded Scope
- Connecting to vehicle OBD-II ports.
- Utilizing external CAN bus data or speedometer ticks.
- Cloud-based inference (strictly prohibited for the final runtime).
