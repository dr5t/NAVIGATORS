# Navigators IDR — Intelligent Dead-Reckoning Navigation System
**(Smart India Hackathon 2026 — ISRO Problem Statement SIH26168)**

Navigators IDR is a smartphone-based, software-only navigation engine capable of bridging long-duration GNSS (Global Navigation Satellite System) outages using only the smartphone's built-in sensors (Accelerometer, Gyroscope) and an embedded AI inference model.

The system uses a highly optimized 15-state Extended Kalman Filter (EKF) that natively fuses hardware IMU data with 2D velocity predictions from an embedded Temporal Convolutional Network (TCN). The entire pipeline runs natively in Python without requiring OBD-II ports, CAN bus connections, or internet connectivity.

## Capabilities

*   **Software-Only & Hardware Independent:** No requirement for external IMU, vehicle speedometers, or OBD-II connectivity. It works natively on standard smartphone sensor streams.
*   **15-State Extended Kalman Filter:** Estimates 3D position, 3D velocity, 3D attitude, and IMU biases dynamically.
*   **AI Velocity Prediction (TCN/LSTM):** Uses a Temporal Convolutional Network to predict 2D vehicle velocity from windowed raw IMU streams, bypassing the noise and drift issues of standard double-integration.
*   **Constraint Modeling:** Includes Zero-Velocity Updates (ZUPT) and Non-Holonomic Constraints (NHC) to prevent lateral drift and vertical bounding.
*   **Offline Map Matching:** Incorporates a geometric/HMM Map Matcher to snap trajectories to offline road networks without Google Maps or Mapbox API calls.

## Benchmark Results

The system has been rigorously tested against simulated multi-distance outages.

| GNSS Outage Distance | System Target | Actual Drift | Status |
| :--- | :--- | :--- | :--- |
| **50 meters** | < 5.0 meters | **0.08 meters** | PASS ✓ |
| **1000 meters** | < 100.0 meters | **2.12 meters** | PASS ✓ |

*AI Inference (ONNX): 4.64ms per 200-sample window. Max update rate: 215Hz.*

## Architecture

1.  **Preprocessor:** Synchronizes IMU streams, applies Butterworth/Median filtering, and auto-aligns the smartphone frame to the vehicle frame using PCA and Gravity estimation.
2.  **TCN Inference:** A pre-trained ONNX model estimates `v_north` and `v_east` from 200-frame sliding windows.
3.  **EKF Fusion:** The 15-state EKF fuses pseudo-measurements (AI velocity, NHC, ZUPT) alongside raw IMU data. When GNSS is available, it provides absolute corrections. When GNSS is lost, the system seamlessly transitions to Dead Reckoning.
4.  **Map Matching:** The offline engine snaps the raw EKF output to the nearest topological road segment.

## Local Development & API

The engine serves a local FastAPI backend to receive sensor data and dispatch map-matched coordinates.

```bash
# Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Server
python src/api/server.py
```

### Endpoints
*   `GET /health` - System health check.
*   `POST /session/start` - Initialize a new navigation session.
*   `POST /sensor/batch` - Upload IMU/GNSS measurements.
*   `GET /navigation/state` - Get the current fused navigation state and map-matched coordinates.
*   `GET /metrics` - Retrieve real-time error bounds and drift estimations.

## Project Documents
All comprehensive design docs, specifications, and walkthroughs can be found in the `/Important Documents/` directory.

- `Project_Requirement_Documentation.md`
- `Design_Requirement.md`
- `Research.md`
- `Implementation_Plans.md`
- `Summary.md`
