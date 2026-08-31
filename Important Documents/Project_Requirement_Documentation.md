# Project Requirement Documentation
**Project Name:** Navigators
**SIH Problem Statement ID:** SIH26168 (S. No. 168)
**Organization:** Indian Space Research Organisation (ISRO)

## 1. Introduction
The objective of this project is to develop an AI/ML-based Intelligent Dead Reckoning (IDR) system for seamless navigation. GNSS/GPS-based navigation becomes unreliable in tunnels, underpasses, and urban environments. When GNSS is lost, conventional navigation often fails. This solution maintains continuous and accurate vehicle positioning using inertial sensing and intelligent estimation.

## 2. Core Objective
Develop an AI/ML-powered IDR system that utilizes a smartphone's IMU sensors to continue vehicle navigation during GNSS outages without relying on OBD-II or a vehicle speedometer. The solution combines AI/ML, IMU processing, GNSS/INS fusion, road constraints, and map matching in a lightweight edge/mobile deployment.

## 3. Key Technical Requirements
*   **AI-based speed estimation:** Estimate vehicle speed/velocity from smartphone IMU data without OBD-II.
*   **Vibration and noise filtering:** Handle engine vibration, road vibration, potholes, bumps, shocks, and phone movement.
*   **Phone alignment:** Automatically estimate phone pitch, roll, and yaw relative to vehicle motion.
*   **Map matching:** Constrain drifting inertial trajectories to plausible roads using offline maps and road-network information.
*   **Vehicle kinematics:** Use Non-Holonomic Constraints (NHC) and related motion constraints to reduce impossible trajectories.
*   **GNSS + INS fusion:** Fuse GNSS and inertial measurements while reducing drift and improving velocity/position estimates.
*   **Seamless outage handling:** Switch rapidly between GNSS-aided navigation and dead reckoning, then return to fused navigation when GNSS returns.
*   **External IMU support:** Core edge engine should accommodate external IMU data, not only smartphone sensors.
*   **100% Offline Edge Execution:** The entire AI model and navigation engine must run directly on the mobile device's CPU/NPU without requiring *any* backend servers, internet connection, or cellular network.
*   **WebAssembly/ONNX Integration:** Utilize ONNX Runtime Web to execute PyTorch neural networks locally within a Progressive Web App (PWA).
*   **Real-time navigation UI:** Display a continuous vehicle position without obvious freezing or location jumps.

## 4. Performance Targets
*   **Dead-reckoning positional drift:** < 10% of the distance travelled
*   **50 m GNSS-denied travel:** < 5 m drift, in under 1 minute
*   **1 km GNSS-denied travel:** < 100 m drift at 60 km/h
*   **Smartphone GNSS + INS update rate:** 10 Hz
*   **FOG-IMU edge engine capability:** Around 200 Hz or higher
