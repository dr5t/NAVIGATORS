# Navigators IDR: Master Project Documentation

**Project Name:** Navigators  
**SIH Problem Statement ID:** SIH26168 (S. No. 168)  
**Organization:** Indian Space Research Organisation (ISRO)  

---

## 1. Executive Summary

Conventional vehicle navigation systems freeze or jump erratically when Global Navigation Satellite Systems (GNSS) like GPS or NavIC are blocked by tunnels, underpasses, or urban canyons. 

The **Navigators Intelligent Dead Reckoning (IDR)** system solves this by utilizing an AI/ML-powered edge engine to maintain continuous, highly accurate vehicle positioning using only commercial smartphone IMU sensors during GNSS outages. By eliminating the need for OBD-II vehicle data and executing 100% offline via WebAssembly, the solution is secure, cost-effective, and highly resilient.

---

## 2. Core Objectives & Performance Targets

**Objective:** Develop an AI/ML-powered IDR system that utilizes a smartphone's IMU to continue vehicle navigation during GNSS outages, combining AI/ML, robust signal processing, GNSS/INS fusion, and map matching in a lightweight edge deployment.

### Key Performance Targets:
- **Dead-Reckoning Positional Drift:** < 10% of the distance travelled.
- **50m GNSS-Denied Travel:** < 5m drift, in under 1 minute.
- **1km GNSS-Denied Travel:** < 100m drift at 60 km/h.
- **Update Rates:** 10 Hz for Smartphone IMU; 200+ Hz for external FOG-IMU edge engines.

---

## 3. System Architecture & Key Innovations

The system is built on an **Edge-to-Cloud architecture**, heavily optimized for edge inference to guarantee real-time responsiveness and absolute privacy. 

### 3.1 Signal Processing & Noise Filtering
Raw smartphone IMU data (accelerometer/gyroscope) is extremely noisy. Before hitting the AI model, the data passes through a **non-linear median filter** and a Butterworth low-pass filter. This mathematically isolates and strips out impulsive mechanical shocks (like hitting a pothole) without distorting the underlying kinematic acceleration data required for accurate velocity estimation.

### 3.2 AI-Based Velocity Estimation (TCN)
Traditional pure physics integration of noisy sensors leads to cubic error drift. Instead, we use a **Temporal Convolutional Network (TCN)**. This neural network analyzes a rolling 20-second window of filtered vibration and motion patterns to directly predict the vehicle's 2D velocity vector. The model was trained on the **IO-VNBD dataset** (58 hours, 4,400 km) using rigorous data augmentations (impulsive shock injection, speed variance, heading misalignment) to guarantee it generalizes across varied mounting positions and vehicle types.

### 3.3 Sensor Fusion (15-State EKF)
To prevent unbounded drift over long periods, the AI's relative predictions must be fused with absolute references. We built a **15-state Extended Kalman Filter (EKF)** that tracks position, velocity, orientation, and sensor biases. When GNSS is available, it fuses the signals. When GNSS is lost, it falls back seamlessly to the AI velocity predictions.

### 3.4 Kinematic Constraints & Map Matching
- **Zero Velocity Updates (ZUPT):** Detects when the vehicle has stopped (e.g., at a red light) to mathematically reset accumulated velocity drift to zero.
- **Non-Holonomic Constraints (NHC):** Restricts the vehicle's lateral motion (sideslip), significantly reducing sideways drift.
- **Map Matching:** A lightweight Geometric Map Matcher snaps the estimated inertial trajectories back to known road network centerlines, utilizing offline grid algorithms.

---

## 4. 100% Offline Edge Deployment (PWA)

To meet ISRO's strict requirements for operations in GNSS-denied (and network-denied) environments, the entire navigation stack was ported to run on the Edge:

1. **WebAssembly AI Inference:** The PyTorch TCN model is exported to ONNX format and executed locally on the smartphone's CPU/NPU using ONNX Runtime Web.
2. **Pure JavaScript Navigation Engine:** The 15-state EKF, ZUPT, NHC, and Map Matching algorithms were written natively in highly optimized JavaScript.
3. **Progressive Web App (PWA):** The simulator and live engine are hosted as a secure PWA. Once loaded, the application and AI model are cached via a Service Worker. The entire system (UI rendering, sensor capture, neural network inference, and EKF fusion) operates completely offline in Airplane Mode, transmitting zero data.

---

## 5. Implementation Roadmap & Current Status

**Current State: System Ready for Field Testing**

### Completed Phases:
- **Phase 1: Core Engine & AI Model.** Data pipelines built for IO-VNBD. PyTorch TCN model trained and evaluated. Python 15-state EKF implemented and unit-tested.
- **Phase 2 & 3: Web Simulator & Live Sensors.** Real-time Leaflet.js dashboard built. Live mobile `DeviceMotionEvent` sensors wired to the pipeline.
- **Phase 4 & 5: Edge Processing & Hardening.** The entire architecture was pivoted to a 100% offline WebAssembly PWA. Robust median filtering and intense ML augmentations were integrated to beat the SIH benchmark targets.

### Future Work:
- **Field Data Collection:** Gather real-world IMU data on specific ISRO/defense vehicles to fine-tune the ONNX model.
- **Native Android APK Migration:** Wrap the PWA in React Native or Capacitor to bypass browser-imposed sensor sampling caps for even higher accuracy.
- **External IMU Integration:** Connect the Edge Engine directly to external, high-frequency FOG-IMUs via Bluetooth or serial connections.
