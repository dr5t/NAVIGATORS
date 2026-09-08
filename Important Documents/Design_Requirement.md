# Navigators IDR — Design Requirements
*Smart India Hackathon 2026 (SIH26168) — ISRO Problem Statement*

## 1. System Objective
The system provides intelligent dead reckoning (IDR) using smartphone-only sensors to estimate vehicle trajectory during GNSS outages (e.g., tunnels, forests, urban canyons).

## 2. Hard Requirements
- **Smartphone-Only**: Must not rely on OBD-II, CAN bus, vehicle speedometers, or dedicated FOG-IMUs.
- **Offline Capable**: The core inference and EKF must run without an internet connection, allowing the frontend (JavaScript) to estimate position seamlessly. The Python backend is explicitly designated as a **development and testing tool**.
- **No Ground-Truth Leakage**: The estimator must never have access to ground-truth velocity or position during an outage.

## 3. Core Architecture Components
1. **Sensor Interface**: Captures linear acceleration and gyroscope data at 100 Hz.
2. **Phone-to-Vehicle Alignment**: Dynamically aligns the arbitrary smartphone frame to the vehicle's Forward/Lateral/Vertical frame using gravity and forward acceleration vectors.
3. **Temporal Convolutional Network (TCN)**: A lightweight, low-latency ONNX model that predicts a 2D planar velocity vector from a 200-timestep window of IMU data.
4. **15-State Extended Kalman Filter (EKF)**: Fuses the AI velocity predictions with raw IMU data, and incorporates GNSS when available.
5. **Non-Holonomic Constraints (NHC)**: Restricts lateral and vertical velocity.
6. **Zero Velocity Updates (ZUPT)**: Freezes drift when the vehicle is stationary.

## 4. Data Collection Strategy
To prevent the model from overfitting to a single vehicle or driver, the training dataset MUST be highly varied.
- **Road Types**: City, highway, residential, intersections, turns, roundabouts, traffic (stop-and-go), speed bumps, rough roads.
- **Driving Conditions**: Slow, medium, fast, acceleration, braking, stationary.
- **Phone Conditions**: Different mounting angles, different mounting locations (dashboard, cupholder, windshield), different phone models.
*Variety is significantly more valuable than total distance.*

## 5. Benchmark Validation Status
Initial invalid benchmarks were flagged during forensic validation due to ground-truth leakage. The new benchmark suite cleanly firewalls ground truth. The system now trains purely on real-world datasets and achieves robust tracking during multi-distance GNSS outages using an offline, on-device WebAssembly execution pipeline.
