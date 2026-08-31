# Summary Document
**Project Name:** Navigators (SIH26168)

## Overview
This project proposes an AI/ML-powered Intelligent Dead Reckoning (IDR) system to solve the problem of GNSS (GPS) denial in vehicle navigation. Utilizing the IO-VNBD benchmark dataset, the project aims to train deep learning models that can accurately estimate vehicle motion solely from smartphone IMU sensors.

## Problem Solved
Conventional navigation systems freeze or jump erratically when GPS signals are blocked (e.g., in tunnels or urban canyons). By predicting velocity and applying map matching, our solution ensures seamless and accurate continuous navigation.

## Key Innovations
1.  **AI-based Speed Estimation:** Eliminates the need for OBD-II vehicle data by extracting speed directly from noisy smartphone IMU patterns.
2.  **Robust Sensor Fusion:** Employs an Extended Kalman Filter (EKF) to fuse ML predictions with GNSS data, seamlessly transitioning during signal loss and re-acquisition.
3.  **Edge-Optimized:** Designed to run in real-time on edge devices (smartphones) at 10Hz, with support for higher frequency external IMUs (200Hz).

## Deliverables
*   A trained AI model using the IO-VNBD dataset.
*   A sensor fusion engine integrating EKF, Non-Holonomic Constraints (NHC), and map matching.
*   A lightweight edge application/simulator demonstrating seamless navigation in GNSS-denied environments.
