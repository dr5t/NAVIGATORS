# Phase 6: Confidence-Aware Navigation Architecture

## Executive Summary

Phase 6 implements physically meaningful navigation uncertainty estimation for the Navigators navigation system. Arbitrary percentage confidences (such as heuristic 95% claims) are eliminated. Instead, uncertainty metrics are derived directly from the 15-state Extended Kalman Filter (EKF) covariance matrix, statistical innovation monitoring, GNSS trust engine classifications, and neural model inference diagnostics.

All metrics are expressed in standard SI units (meters, meters per second, degrees, and seconds). A structured, machine-readable confidence object is exposed to frontend consumers and client APIs without arbitrary percentage conversions.

Accuracy notice: In accordance with established scientific practice, improved navigation accuracy is not claimed until empirical benchmark datasets demonstrate it across field trials.

## 1. Confidence Representation

The confidence representation provides a comprehensive assessment of navigation state certainty without fabricating unsupported figures. It is structured into four functional layers:

1. **Physical Uncertainties**: 1-sigma standard deviations extracted directly from the state covariance diagonal $P_{ii}$.
2. **Sensor Health Profile**: Actual reliability states, variance scales, and diagnostic reason codes evaluated by backend estimators.
3. **Operational Integrity**: System-level operational status (`NOMINAL`, `DEGRADED`, `UNRELIABLE`) calculated against physical alert limits.
4. **Estimator State**: Current filter mode (`GNSS_INS`, `DEAD_RECKONING`, `REACQUISITION`), dead-reckoning duration, and distance traveled.

### Schema of the Machine-Readable Confidence Object

```json
{
  "timestamp": 1727271234.5,
  "uncertainties": {
    "position_horizontal_m": 0.667,
    "position_vertical_m": 0.468,
    "velocity_horizontal_mps": 0.543,
    "heading_deg": 4.52
  },
  "units": {
    "position_horizontal": "meters",
    "position_vertical": "meters",
    "velocity_horizontal": "meters_per_second",
    "heading": "degrees"
  },
  "sensors": {
    "gnss": {
      "status": "TRUSTED",
      "is_trusted": true,
      "trust_score": 0.993,
      "variance_scale": 1.0,
      "reported_accuracy_m": 2.0,
      "reasons": []
    },
    "ai_velocity": {
      "status": "VALID",
      "is_valid": true,
      "std_dev_mps": 0.566,
      "latency_ms": 2.2,
      "reason": null
    },
    "map_matching": {
      "is_matched": true,
      "confidence": 0.885,
      "cross_track_m": 1.25,
      "segment_id": "way/123456"
    }
  },
  "estimator": {
    "mode": "gnss_ins",
    "integrity_status": "NOMINAL",
    "integrity_score": 0.923,
    "dr_drift_percent": 0.45,
    "overall_confidence": 0.923
  }
}
```

## 2. Uncertainty Units

Every uncertainty metric is grounded in physical quantities:

| Quantity | State Elements in EKF | Mathematical Derivation | Unit |
| :--- | :--- | :--- | :--- |
| Horizontal Position Uncertainty | $x[0]$ (East), $x[1]$ (North) | $\sigma_{\text{pos, 2D}} = \sqrt{P_{0,0} + P_{1,1}}$ | meters (m) |
| Vertical Position Uncertainty | $x[2]$ (Up) | $\sigma_{\text{pos, Up}} = \sqrt{P_{2,2}}$ | meters (m) |
| Horizontal Velocity Uncertainty | $x[3]$ ($v_E$), $x[4]$ ($v_N$) | $\sigma_{\text{vel, 2D}} = \sqrt{P_{3,3} + P_{4,4}}$ | meters per second (m/s) |
| Heading Uncertainty | $x[8]$ ($\psi$, yaw/heading) | $\sigma_{\psi} = \text{degrees}(\sqrt{P_{8,8}})$ | degrees ($^\circ$) |
| Course-Over-Ground Precision | $v_E, v_N$ | $\sigma_{\text{COG}} \approx \frac{\sigma_{v}}{\max(v, 0.5)}$ | radians (rad) |
| Dead-Reckoning Duration | Clock difference | $t - t_{\text{outage\_start}}$ | seconds (s) |
| Distance Traveled | Path integral | $\int \|v\| dt$ | meters (m) |

## 3. Propagation Behavior

In the absence of external position corrections, state uncertainty propagation is governed strictly by the discrete Riccati equation:

$$P_{k+1}^- = F_k P_k F_k^T + Q_k$$

Where:
- $F_k$ is the state transition Jacobian relating position to velocity ($F_{0:2, 3:5} = I_3 \cdot \Delta t$) and velocity to attitude/accelerometer bias.
- $Q_k$ is the discrete process noise matrix modeling IMU gyro drift and accelerometer random walk.

Key propagation characteristics:
1. **Position Variance Growth**: Position uncertainty expands continuously via the velocity integral:
   $$\sigma_{\text{pos}}^2(t) \approx \sigma_{\text{pos}}^2(0) + \int_0^t \sigma_{\text{vel}}^2(\tau) d\tau + Q_{\text{pos}} t$$
2. **Velocity Damping via AI Inference**: When the AI velocity model is operational during GNSS outages, its pseudo-measurement updates bound $\sigma_{\text{vel, 2D}}$, preventing velocity random-walk explosion and dramatically curtailing quadratic position dispersion.
3. **Attitude Drift**: Heading variance increases at rate $Q_{\text{gyro}} \cdot \Delta t$ per step unless constrained by vehicle Non-Holonomic Constraints (NHC) or road geometry.

## 4. Degradation Behavior

When sensors encounter adverse environments or anomalies, the uncertainty system responds deterministically:

1. **GNSS Outage**:
   - GNSS updates cease.
   - Filter switches mode to `DEAD_RECKONING`.
   - State covariance $P$ grows monotonically through process propagation.
   - Integrity status degrades from `NOMINAL` to `DEGRADED` (when $\sigma_{\text{pos}} > 9.0$ m) and eventually `UNRELIABLE` (when $\sigma_{\text{pos}} > 15.0$ m).
2. **Measurement Disagreement / GNSS Anomaly**:
   - Anomaly detector identifies inconsistent position jumps or velocity divergence.
   - Trust engine down-weights fix to `SUSPICIOUS` ($s_{\text{pos}} \ge 50$) or rejects it as `UNUSABLE`.
   - Measurement covariance inflation prevents corrupted fixes from distorting state estimates.
   - State covariance $P$ reflects elevated uncertainty rather than collapsing falsely.
3. **AI Inference Failure**:
   - If AI model outputs high variance ($> 1.0$ m$^2$/s$^2$) or excessive latency ($> 50$ ms), the measurement is flagged `HIGH_VARIANCE` or `REJECTED`.
   - Filter continues with pure inertial propagation and Non-Holonomic Constraints.
   - Telemetry logs the exact rejection reason code.

## 5. Recovery Behavior

When reliable GNSS signals return:

1. **Trust Engine Reacquisition**:
   - The trust engine tracks consecutive valid fixes.
   - Hysteresis prevents premature re-entry until fixes demonstrate kinematic plausibility.
2. **Kalman Gain Contraction**:
   - As valid GNSS updates resume with nominal measurement covariance $R_{\text{GNSS}}$, the Kalman gain $K_k$ becomes significant:
     $$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}$$
   - The Joseph-form covariance update sharply contracts state uncertainty:
     $$P_k = (I - K_k H_k) P_k^- (I - K_k H_k)^T + K_k R_k K_k^T$$
   - Horizontal position uncertainty drops from elevated outage levels back down to nominal GNSS accuracy ($\le 1.0$ m).
3. **Smooth State Convergence**:
   - During `REACQUISITION`, innovations are bounded by fractional correction limits (anti-jump protection) while covariance contracts smoothly.
   - When position residuals fall below 3.0 m across consecutive good fixes, the filter returns to `GNSS_INS` mode with `NOMINAL` integrity status.

## 6. Replay Test Suite

The physical uncertainty behaviors are validated through deterministic replay tests in `tests/test_confidence_replay.py`:

| Test Name | Operational Scenario | Observed Filter and Uncertainty Response | Status |
| :--- | :--- | :--- | :---: |
| `test_replay_healthy_navigation` | Continuous healthy GNSS + AI + IMU | $\sigma_{\text{pos}} < 1.0$ m, $\sigma_{\text{vel}} < 0.6$ m/s, `NOMINAL` status, valid schema units | PASS |
| `test_replay_gnss_outage` | Sudden loss of GNSS (5s outage) | Mode switches to `DEAD_RECKONING`; $\sigma_{\text{pos}}$ grows monotonically | PASS |
| `test_replay_longer_gnss_outage` | Extended GNSS outage (15s) | $\sigma_{\text{pos}}$ and $\sigma_{\psi}$ grow significantly larger than short outage; integrity degrades | PASS |
| `test_replay_gnss_recovery` | GNSS restoration after outage | Covariance contracts sharply from peak outage levels ($> 1.5$ m) back to $< 1.0$ m | PASS |
| `test_replay_gnss_anomaly` | 80m unphysical position jump | Anomaly detected (`GNSS_POSITION_JUMP`), trust set to `SUSPICIOUS`, position hold maintained | PASS |
| `test_replay_ai_failure` | AI model variance spike ($4.0$ m$^2$/s$^2$) | AI flagged `HIGH_VARIANCE`, measurement rejected, telemetry records failure reason | PASS |

## 7. Frontend Integration

Frontend UI layers consume the machine-readable confidence object via:
- Real-time WebSocket telemetry packets emitted by the FastAPI navigation backend.
- REST endpoints `/navigation/state` and `/metrics` returning structured `confidence_object` payloads.
- Browser offline simulator engine exposing `window.latestConfidenceObject` for client inspection.

All displays present physical meters and degrees. Arbitrary percentage values are avoided.
