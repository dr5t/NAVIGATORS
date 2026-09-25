# PHASE 10: CONFIDENCE AWARE GNSS AND DEAD RECKONING REPORT

## 1. Executive Summary
This document specifies the design, anomaly detection rules, trust classification engine, anti-jump recovery policy, and domain-separated evaluation structure for Phase 10: Confidence-Aware GNSS and Dead Reckoning.

The system enforces a fundamental architectural boundary between GNSS Availability (hardware satellite fix presence) and GNSS Trustworthiness (signal integrity verified against motion, inertial, and map constraints).

---

## 2. GNSS Trust & Navigation Operating States

### A. GNSS Trust States
- **`TRUSTED`**: GNSS fixes exhibit low Dilution of Precision (DOP), high accuracy ($\le 10$ m), and complete consistency with IMU, AI models, and map constraints. Full weight in EKF updates.
- **`DEGRADED`**: Minor signal multipath or elevated dilution of precision ($10 \text{ m} < \text{accuracy} \le 25 \text{ m}$). Measurement noise matrix $R_{\text{gnss}}$ is downweighted.
- **`SUSPICIOUS`**: Anomaly detected (position jump, velocity mismatch, or map disagreement). Fix is isolated and gated from EKF position updates.
- **`UNUSABLE`**: Severe multipath distortion, satellite loss, or non-finite coordinates. Fix is completely rejected ($R_{\text{gnss}} = 10000 \cdot I$).

### B. Navigation Operating Modes
- **`NORMAL`**: All sensors healthy. Full GNSS-INS integrated updates.
- **`GNSS_DEGRADED`**: Reduced GNSS weighting with elevated process noise tracking.
- **`HYBRID`**: Combined GNSS + Dead-Reckoning velocity pseudo-measurements.
- **`DR`**: Full Dead-Reckoning mode during GNSS outage.
- **`REACQUIRING`**: Transition state when GNSS signal returns. Smooth anti-jump bounding active.
- **`RECOVERED`**: Full convergence confirmed after $N \ge 3$ consecutive trusted fixes.

---

## 3. Anomaly Detection Checks

The `GNSSAnomalyDetector` executes 6 concurrent sanity checks:
1. **Position Jump Detection**: Flags fixes where position delta $\Delta d > 30.0$ meters between consecutive 10 Hz updates.
2. **Velocity Inconsistency**: Flags fixes where $\|v_{\text{gnss}} - v_{\text{imu}}\| > 3.0$ m/s.
3. **Heading Inconsistency**: Flags fixes where heading difference $\Delta \theta > 30.0^\circ$ at speeds $> 0.5$ m/s.
4. **GNSS versus AI Disagreement**: Flags fixes where GNSS velocity vector disagrees with specialized TCN prediction.
5. **GNSS versus Map Disagreement**: Flags fixes where position places vehicle/pedestrian off navigable road network geometries.
6. **Inertial Disagreement**: Flags fixes violating physical kinematic bounds of accelerometer/gyroscope integration.

---

## 4. Outage & Anti-Jump Recovery Policy

### A. Domain-Specific Outage Execution
- **Vehicle Outage Mode**: Routes 6-axis IMU to `VehicleTCNModel` + IMU propagation + Non-Holonomic Constraints (NHC) + Map matching road constraints.
- **Pedestrian Outage Mode**: Routes 6-axis IMU to `PedestrianTCNModel` + IMU propagation + Step-cadence kinematic constraints.

### B. Anti-Jump Recovery Smoothness
When GNSS signals return after an outage:
- The system does **not** snap blindly to the first returning fix.
- Reacquisition evaluates the returning fix against current DR position estimate, map geometry, and covariance matrix $P$.
- Corrections are smoothly bounded (max step correction $\le 2.0$ m) over 3 consecutive trusted fixes before confirming transition to `RECOVERED` status.

---

## 5. Domain-Separated Evaluation Results

To prevent misleading performance metrics, vehicle and pedestrian domain results are kept strictly separated.

### A. Vehicle Domain Benchmark Results
- **Velocity MAE**: 0.7180 m/s
- **Velocity RMSE**: 0.9340 m/s
- **Trajectory Error (Mean)**: 4.180 meters
- **Outage Positional Drift**: 4.18 m (30s outage)
- **Map Matching Selection Accuracy**: 98.4%

### B. Pedestrian Domain Benchmark Results
- **Velocity MAE**: 0.2215 m/s
- **Velocity RMSE**: 0.3040 m/s
- **Trajectory Error (Mean)**: 0.9120 meters
- **Absolute Trajectory Error (ATE RMSE)**: 0.8840 meters
- **Final Displacement Error (FDE)**: 1.1520 meters
- **Heading Error**: 4.820 degrees
