# Navigators - Intelligent Dead Reckoning (IDR) System

**SIH Problem Statement ID:** SIH26168  
**Organization:** Indian Space Research Organisation (ISRO)  

## Project Overview

The Navigators IDR system solves the critical problem of GNSS (GPS/NavIC) denial in vehicle navigation. When a vehicle enters a tunnel, dense urban environment, or faces signal spoofing, traditional navigation fails. Our solution uses AI/ML on ubiquitous smartphone IMU sensors (accelerometer and gyroscope) to accurately estimate vehicle motion, fusing it with GNSS data via a 15-state Extended Kalman Filter (EKF) to provide uninterrupted, highly precise navigation.

## Key Features

- **100% Offline Edge Deployment:** The entire engine runs as a Progressive Web App (PWA) in WebAssembly and Javascript, requiring zero backend or internet connectivity.
- **Deep Learning Velocity Estimation:** A Temporal Convolutional Network (TCN) trained on the IO-VNBD dataset predicts 2D velocity from noisy smartphone IMUs.
- **Robust Signal Processing:** Non-linear median filters isolate and remove impulsive mechanical shocks (e.g., potholes) before they confuse the AI model.
- **Advanced Sensor Fusion:** A 15-state EKF optimally fuses AI predictions with GNSS.
- **Kinematic Constraints:** Non-Holonomic Constraints (NHC) and Zero Velocity Updates (ZUPT) limit exponential drift.
- **Map Matching:** Geometric snapping corrects diverging inertial trajectories to known road networks.

## Documentation

All core project documentation has been fully updated and can be found in the `Important Documents/` folder:

- **[Summary Document](Important%20Documents/Summary.md):** High-level overview of the problem, innovations, and deliverables.
- **[Project Requirements Documentation](Important%20Documents/Project_Requirement_Documentation.md):** Official requirements, objectives, and extreme performance targets.
- **[Design Requirements](Important%20Documents/Design_Requirement.md):** System architecture, processing flows, and hardware/software limits.
- **[Detailed Project Guide](Important%20Documents/Detailed_Project_Guide.md):** Theoretical foundation and engineering architecture.
- **[Implementation Plans & Progress](Important%20Documents/Implementation_Plans.md):** Phase-by-phase tracker of all work completed.
- **[Research Document](Important%20Documents/Research.md):** Dataset analysis (IO-VNBD), state-of-the-art ML modeling, and filtering theory.

## Running the Project

### Edge Simulator (PWA)
1. Start a local server: `python -m http.server 8000` inside the `simulator/` directory.
2. Open `http://localhost:8000` to run the fully integrated offline JS engine (EKF + WebAssembly TCN).

### Python Benchmarking & Training
```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/benchmark.py
```
