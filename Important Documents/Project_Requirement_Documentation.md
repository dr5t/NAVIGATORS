# Project Requirement Documentation (PRD)

## 1. Project Title
Navigators IDR — Intelligent Dead-Reckoning Navigation System (ISRO SIH26168)

## 2. Objective
To develop a smartphone-based navigation system capable of maintaining accurate positioning during Global Navigation Satellite System (GNSS) outages. The system must operate without relying on external vehicle hardware (such as OBD-II, CAN bus, or dedicated external IMUs) and without internet connectivity.

## 3. Core Requirements
- **Hardware Profile:** Smartphone only. Must utilize internal 6-DOF IMU (Accelerometer, Gyroscope) and native GNSS chips.
- **Offline Functionality:** The entire system—AI inference, filtering, constraints, and map matching—must run completely offline. No backend processing via cloud.
- **Accuracy Targets:** 
  - < 5 meters of drift during a 50-meter GNSS outage.
  - < 100 meters of drift during a 1000-meter GNSS outage.
- **Output Interfaces:** Provide real-time location (Latitude/Longitude), heading, and velocity to a local web client or API.

## 4. Key Functional Constraints
- **15-State Extended Kalman Filter (EKF):** Must be used as the core data fusion engine, natively estimating 3D Position, 3D Velocity, 3D Attitude, Accelerometer Bias, and Gyroscope Bias.
- **AI Velocity Estimation:** Must embed an ONNX-format deep learning model (e.g., Temporal Convolutional Network) to map sliding windows of IMU data to 2D velocity vectors (East, North), bypassing classical double-integration drift.
- **Zero-Velocity Update (ZUPT):** The system must detect when the vehicle is stationary to clamp accumulated errors.
- **Non-Holonomic Constraints (NHC):** The system must assume wheeled vehicle dynamics (no lateral sliding, no vertical flight) to bound error growth.
- **Map Matching:** The system must snap the raw mathematical trajectory to a known offline geographic road network.

## 5. System Architecture
The application runs as a local FastAPI backend serving standard REST interfaces and WebSockets, taking in live or batched sensor data from the client, processing it through the Python navigation engine, and returning the map-matched position.

## 6. Success Criteria
The project is successful when the Python navigation backend can ingest GNSS/IMU data, simulate outages up to 1000m, and output map-matched coordinates falling within the required error bounds using only its built-in algorithmic constraints and ONNX AI model.
