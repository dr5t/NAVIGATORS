# GNSS Trust Engine Documentation

## Overview

The GNSS Trust Engine evaluates GNSS fix reliability in real time using actual available measurements from standard mobile and browser location APIs (accuracy, speed, bearing, coordinates, timestamps). It avoids binary ON/OFF switching by dynamically estimating a continuous reliability score and categorizing fixes into four distinct operational trust states.

The trust engine acts as an upstream evidence filter for the navigation fusion pipeline (Extended Kalman Filter and Navigation State Machine), modulating measurement covariance and providing actionable diagnostic reasons without modifying the underlying AI models.

## Architecture

```
GNSS Raw Measurements (lat, lon, accuracy, speed, heading, timestamp)
                          |
                          v
                Feature Extraction
  (position jump speed, implied acceleration, heading divergence,
   normalized innovation, velocity residual, timestamp validity)
                          |
                          v
                  GNSS Trust Engine
                          |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
 Trust State      Reliability Score    Covariance Inflation
(TRUSTED,          [0.0 to 1.0]       (1.0x to 10000.0x)
 DEGRADED,
 SUSPICIOUS,
 UNUSABLE)
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
                Navigation Fusion Pipeline
  - EKF: Dynamically scales measurement noise R_pos and R_vel
  - State Machine: Drives NORMAL, GNSS_DEGRADED, and DR transitions
```

## Trust States

1. **TRUSTED**
   - Criteria: High accuracy (accuracy <= 10.0 m), valid timestamp, physically consistent velocity, heading alignment within limits, normalized innovation <= 2.5 sigma, and recovery criteria satisfied.
   - Fusion Action: Measurement variance scale = 1.0x (nominal EKF measurement noise). Full confidence update.

2. **DEGRADED**
   - Criteria: Moderate accuracy degradation (10.0 m < accuracy <= 25.0 m), moderate normalized innovation (2.5 to 4.0 sigma), moderate velocity discrepancy, or active recovery phase after an outage.
   - Fusion Action: Measurement variance scaled proportionally (5.0x to 12.5x). EKF update proceeds with wider uncertainty, allowing inertial and AI velocity propagation to contribute more heavily.

3. **SUSPICIOUS**
   - Criteria: Significant accuracy degradation (25.0 m < accuracy <= 50.0 m), large heading discrepancy (> 1.2 rad at speed >= 2.5 m/s), high normalized innovation (4.0 to 7.0 sigma), or high implied acceleration (12.0 to 20.0 m/s^2).
   - Fusion Action: Measurement variance heavily inflated (50.0x). Fix provides minimal correction to filter state while preserving continuous tracking and avoiding divergence.

4. **UNUSABLE**
   - Criteria: Complete outage (None or missing fix), stale or non-advancing timestamp (dt <= 0.0 s or dt > 3.0 s), non-finite coordinates, accuracy > 80.0 m, impossible position jump (> 80.0 m/s implied speed), impossible velocity (> 65.0 m/s), or severe heading reversal (> 2.0 rad).
   - Fusion Action: Measurement variance scale = 10000.0x. EKF update is bypassed entirely. Navigation State Machine triggers transition to Dead Reckoning (DR).

## Evidence Sources and Features

Only signals exposed by standard Android Location API and browser Geolocation API are utilized:

- **Reported Accuracy**: Horizontal accuracy estimate in meters (68% confidence radius).
- **Position Continuity**: Implied jump speed calculated as Euclidean distance between consecutive GNSS fixes divided by time interval dt.
- **Velocity Continuity**: Implied acceleration calculated as Euclidean difference between consecutive GNSS velocity vectors divided by dt.
- **Heading Consistency**: Absolute angular difference between GNSS course-over-ground and vehicle filter heading when speed exceeds 2.5 m/s.
- **Timestamp Validity**: Verification that fix timestamps strictly advance (dt > 0) and do not exceed stale timeout thresholds (dt <= 3.0 s).
- **Innovation Consistency**: Normalized innovation (Mahalanobis distance) comparing raw GNSS position against predicted EKF position, normalized by combined filter and sensor uncertainty.
- **Cross-Velocity Consistency**: Discrepancy between GNSS velocity and current estimated/AI velocity.
- **Map and Road Consistency**: Distance from mapped road centerline when topological map matching is active.

## Statistical Reliability Score

The trust engine computes a continuous reliability score bounded between 0.0 and 1.0 using a multi-factor penalty model:

```
Score = Base_State_Score * Accuracy_Factor * Innovation_Factor * Heading_Factor
```

- Base state scores: TRUSTED = 1.0, DEGRADED = 0.65, SUSPICIOUS = 0.25, UNUSABLE = 0.0.
- Accuracy Factor: Sigmoidal decay function based on ratio of accuracy to trusted threshold.
- Innovation Factor: Exponential decay exp(-0.5 * normalized_innovation^2) for statistically grounded outlier dampening.
- Heading Factor: Cosine alignment factor cos(min(pi/2, heading_error)).

## Outage Recovery and Hysteresis

To prevent chattering and premature re-convergence onto multipath reflections after an outage, the engine implements a stateful recovery policy:
- When a GNSS outage or UNUSABLE state occurs, `in_recovery` is set to True.
- The engine requires at least 3 consecutive valid fixes (`min_recovery_fixes = 3`) before elevating the state to TRUSTED.
- During this window, fixes are categorized as DEGRADED with `"recovering_from_outage"` diagnostic reason, ensuring the EKF smooths reacquisition.

## Fusion Integration

1. **Extended Kalman Filter (`src/navigation/ekf.py`)**:
   - `update_gnss(gnss_position, gnss_velocity, timestamp, trust_metric)` accepts `GNSSTrustMetric`.
   - When state is UNUSABLE, the observation update is skipped, preventing corrupt states.
   - When state is DEGRADED or SUSPICIOUS, position noise covariance R_pos and velocity noise covariance R_vel are multiplied by `trust_metric.position_variance_scale` and `trust_metric.velocity_variance_scale`.
   - State summary includes `gnss_trust_state` and `gnss_trust_score`.

2. **Navigation State Machine (`src/navigation/nav_state.py`)**:
   - `process_sensor_health(imu_healthy, gnss_available, hdop, trust_state)` incorporates the trust state.
   - UNUSABLE triggers `NavState.DR` transition.
   - DEGRADED or SUSPICIOUS elevates effective HDOP to trigger `NavState.GNSS_DEGRADED`.
   - TRUSTED restores `NavState.NORMAL`.
