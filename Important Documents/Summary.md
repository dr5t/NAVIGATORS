# Summary Document
**Project Name:** Navigators (SIH26168)

## Overview
This project proposes an AI/ML-powered Intelligent Dead Reckoning (IDR) system to solve the problem of GNSS (GPS) denial in vehicle navigation. Utilizing the IO-VNBD benchmark dataset, the project aims to train deep learning models that can accurately estimate vehicle motion solely from smartphone IMU sensors.

## Problem Solved
Conventional navigation systems freeze or jump erratically when GPS signals are blocked (e.g., in tunnels or urban canyons). By predicting velocity and applying map matching, our solution ensures seamless and accurate continuous navigation.

## Key Innovations
1.  **AI-based Speed Estimation:** Eliminates the need for OBD-II vehicle data by extracting speed directly from noisy smartphone IMU patterns using a robustly trained Temporal Convolutional Network.
2.  **Robust Sensor Fusion:** Employs an Extended Kalman Filter (EKF) to fuse ML predictions with GNSS data, augmented by non-linear median filtering to remove physical shock/pothole artifacts from IMU data.
3.  **100% Offline Edge Deployment:** Designed to run entirely in the browser as a Progressive Web App (PWA). The entire navigation stack (15-state EKF, ZUPT, NHC, Map Matching, and ONNX AI inference) runs natively on the mobile CPU without any backend server.

## Deliverables (Completed)
*   A robustly trained AI model (PyTorch -> ONNX) capable of generalizing across varied mechanical mounting conditions via intense data augmentations.
*   A sensor fusion engine integrating a 15-state EKF, Non-Holonomic Constraints (NHC), and Map Matching implemented in both Python (for benchmarking) and native JavaScript (for edge execution).
*   A lightweight edge application/PWA simulator demonstrating flawless offline navigation in GNSS-denied environments.
