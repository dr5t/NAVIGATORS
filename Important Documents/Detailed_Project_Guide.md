# Detailed Project Guide

## Overview
This guide provides instructions on how to set up, test, and understand the Navigators IDR Python codebase.

## Repository Structure
- `/src/navigation/`: Core mathematical engine.
  - `ekf.py`: The 15-state Extended Kalman Filter.
  - `nhc.py`: Non-Holonomic Constraint processor.
  - `zupt.py`: Zero-Velocity Update detector.
  - `map_matching.py`: Offline geographic snapping.
  - `dead_reckoning.py`: State manager for GNSS-denied periods.
  - `alignment.py`: Smartphone-to-vehicle PCA alignment.
- `/src/models/`: AI/ML processing.
  - `tcn_model.py`: Temporal Convolutional Network logic.
- `/src/data/`: Data handling.
  - `preprocessor.py`: Butterworth and Median filters.
- `/src/api/`: Interfaces.
  - `server.py`: FastAPI implementation serving REST endpoints and WebSockets.
- `/scripts/`: Evaluation and Utilities.
  - `benchmark.py`: Systematic benchmark evaluation for ISRO targets.

## Local Setup

### 1. Environment Requirements
- Python 3.9+
- `pip`

### 2. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Running the Benchmarks
To prove the system works, run the automated benchmark suite. It tests ablation configurations and the 50m / 1000m GNSS outages.
```bash
python scripts/benchmark.py
```
*Expected output: All sections report `PASS ✓` and Drift Percentages < 10%.*

### 4. Running the Navigation Server
To run the local backend server that can accept real sensor data:
```bash
python src/api/server.py
```
This will start a FastAPI server at `http://localhost:8000`.

### 5. Interacting with the API
The server exposes several endpoints for external clients (like a React Native or Flutter mobile app) to communicate with:
- **`GET /health`**: Verifies the service is running.
- **`POST /session/start`**: Creates a new tracking session and returns a `session_id`.
- **`POST /sensor/batch`**: Uploads a batch of IMU and GNSS data to be processed by the EKF.
- **`GET /navigation/state`**: Retrieves the current map-matched position and mode (GNSS or DR).
