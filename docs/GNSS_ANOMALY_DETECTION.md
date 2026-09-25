# GNSS Anomaly Detection Layer Documentation

## Overview

The GNSS Anomaly Detection layer acts as an upstream validation filter for the Navigators navigation architecture. It identifies incoming GNSS measurements that are present in the data stream but physically, kinematically, statistically, or cartographically inconsistent with the actual motion of the vehicle.

The anomaly detector operates directly before the GNSS Trust Engine and Adaptive Fusion stages without deleting any measurement silently. Every flagged or down-weighted fix is annotated with explicit reason codes, metrics, and severity estimates.

## Architectural Pipeline

```
GNSS Measurement (lat, lon, accuracy, speed, heading, timestamp)
                          |
                          v
                GNSS Anomaly Detector
  - Kinematic consistency checks
  - Inertial / AI velocity cross-checks
  - Filter innovation Mahalanobis checks
  - Map / road geometry consistency
                          |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
   NORMAL             ANOMALOUS            UNKNOWN
(Consistent)     (One or more checks      (Initial fix /
                  flagged with reasons)    missing history)
                          |
                          v
                  GNSS Trust Engine
  - Down-weights suspicious fixes (variance inflation)
  - Rejects severe anomalies (marks UNUSABLE)
  - Preserves anomaly reports in diagnostics
                          |
                          v
             Adaptive Fusion (EKF & NavState)
```

## Reason Codes and Detection Logic

1. **GNSS_POSITION_JUMP**
   - Condition: Implied jump velocity v_jump = ||p_curr - p_prev|| / dt > 40.0 m/s (~144 km/h), or displacement ||p_curr - p_prev|| > 30.0 m exceeding 2.5 times the expected displacement at current speed.
   - Physical Origin: Land vehicles on public roadways cannot exhibit instantaneous spatial jumps between consecutive epochs without continuous acceleration. Discrete multi-meter shifts represent constellation switching or multipath reflections.

2. **GNSS_IMPOSSIBLE_DISPLACEMENT**
   - Condition: Measured displacement ||p_curr - p_prev|| exceeds Newtonian kinematic bound d_max = v_prev * dt + 0.5 * a_max * dt^2 + 3 * sigma_pos + 5.0 m, or ||p_curr - p_prev|| > 55.0 m for dt <= 1.0 s.
   - Physical Origin: Governed by Newton's second law and tire-road friction limit (mu <= 1.0, maximum vehicle acceleration a_max = 12.0 m/s^2).

3. **GNSS_SPEED_INCONSISTENCY**
   - Condition: Implied acceleration |v_curr - v_prev| / dt > 12.0 m/s^2, or reported speed > 65.0 m/s (~234 km/h).
   - Physical Origin: Standard vehicle braking and acceleration limits under tire-road adhesion friction circle (mu * g approx 9.8 to 12.0 m/s^2).

4. **GNSS_VELOCITY_DISAGREEMENT**
   - Condition: Vector difference ||v_gnss - v_inertial|| > max(8.0 m/s, 3.5 * sqrt(sigma_v_gnss^2 + sigma_v_inertial^2)).
   - Statistical Origin: 3.5-sigma confidence ellipse of combined velocity uncertainty, bounding false-positive probability to p < 0.001 under Gaussian sensor noise.

5. **GNSS_HEADING_INCONSISTENCY**
   - Condition: When vehicle speed >= 2.5 m/s, angular divergence |psi_gnss - psi_motion| > 0.8 rad (~46 degrees).
   - Kinematic Origin: Wheeled land vehicles have bounded sideslip angle |beta| < 15 degrees during non-drifting transit. GNSS bearing divergence at speed >= 2.5 m/s indicates receiver multipath or antenna misalignment.

6. **GNSS_NAVIGATION_RESIDUAL_HIGH**
   - Condition: Normalized innovation (Mahalanobis distance) ||y|| / sqrt(sigma_filter^2 + sigma_gnss^2) > 4.0, or raw position residual > 35.0 m under low filter uncertainty.
   - Statistical Origin: Under nominal filter tracking, normalized innovation squared follows a Chi-square distribution with 2 degrees of freedom. A threshold of 4.0 standard deviations corresponds to p < 0.0001 rejection probability.

7. **GNSS_MAP_INCONSISTENCY**
   - Condition: When high-confidence road matching data exists (probability >= 0.5), cross-track distance > 25.0 m + accuracy, or heading divergence from road link orientation > 1.05 rad (~60 degrees) at speed > 5.0 m/s.
   - Cartographic Origin: Road corridors (lanes, median, shoulder) have fixed physical widths (< 25 m from centerline). Greater lateral separation indicates parallel-road jumping or false off-road snapping.

## False-Positive Considerations

1. **Low-Speed Heading Singularity**: At speeds below 2.5 m/s, GNSS course-over-ground becomes ill-conditioned due to velocity vector norm approaching zero. The heading check is explicitly bypassed when speed < 2.5 m/s.
2. **Aggressive Braking / Acceleration**: The kinematic acceleration threshold is set to 12.0 m/s^2 rather than nominal 3.0 m/s^2 to prevent false positives during emergency stopping or dynamic maneuvering.
3. **Sensor Jitter and Uncertainty Budgeting**: All spatial and innovation tests incorporate sensor reported accuracy (3 * sigma_pos) and EKF state covariance rather than applying raw static distance thresholds.
4. **Initial Fix Handling**: The first fix without prior history returns status UNKNOWN with nominal confidence (0.5) to avoid false anomaly triggers on startup.
5. **Hysteresis Recovery Window**: Returning from ANOMALOUS to NORMAL requires 3 consecutive anomaly-free fixes to prevent chattering during intermittent multipath.

## Integration with Trust Engine and Fusion

- Anomalous fixes are not silently dropped.
- Severe anomalies (GNSS_POSITION_JUMP, GNSS_IMPOSSIBLE_DISPLACEMENT, GNSS_SPEED_INCONSISTENCY with speed > 65 m/s) drive the trust state to UNUSABLE. The EKF bypasses the update while preserving internal state and recording all diagnostics.
- Moderate anomalies (GNSS_NAVIGATION_RESIDUAL_HIGH, GNSS_HEADING_INCONSISTENCY, GNSS_MAP_INCONSISTENCY, GNSS_VELOCITY_DISAGREEMENT) drive trust state to SUSPICIOUS with 50.0x covariance inflation, allowing inertial dead reckoning to steer the trajectory while maintaining filter continuity.
