# AI Velocity Model Integration Documentation

## Overview

The AI Velocity Model Integration embeds the existing verified Temporal Convolutional Network (TCN) ONNX production model into the Navigators navigation pipeline. AI velocity is treated strictly as an indirect vector observation source for the Extended Kalman Filter (EKF), rather than direct position control.

The integration adheres to strict physical and operational constraints:
- Zero retraining and zero weight modifications.
- Existing six-channel formulation strictly preserved (accelerometer X/Y/Z, gyroscope X/Y/Z).
- The AI never receives GNSS speed as an input.
- Navigation never stops when AI inference is unavailable; the system smoothly falls back to pure inertial propagation and kinematic constraints.

## Model Specifications

- Architecture: Temporal Convolutional Network (TCN).
- Runtime: ONNX Runtime (CPUExecutionProvider, single-threaded deterministic inference).
- Input: Exactly 200 frames of 6-axis IMU data (accelerometer X/Y/Z, gyroscope X/Y/Z).
- Sampling Rate: 10 Hz nominal (0.1 s epoch delta).
- Temporal Window: 20.0 seconds receptive field.
- Normalization: Z-score standardization using verified production statistics (simulator/norm_stats.json).
- Output: 2D horizontal velocity vector [V_North, V_East] in m/s.
- Validation Baseline: Verified 4.20 m/s MAE on the benchmark IO-VNBD dataset.

## Integration Architecture

```
Raw IMU Data (accel X/Y/Z, gyro X/Y/Z, timestamps)
                          |
                          v
                Input & Window Validation
  - 6-channel isolation (zero GNSS leakage)
  - 200-frame window completeness
  - Monotonic timestamp validation
  - Physical sensor range checks
                          |
                          v
               Z-Score Normalization
  (features - mean) / std from verified contract
                          |
                          v
               TCN ONNX Inference
           Output: [V_North, V_East]
                          |
                          v
           AI Measurement Validation Layer
  - Plausibility checks (speed <= 65 m/s)
  - Stationary state gating (ZUPT coupling)
  - Stale prediction detection (age <= 2.0 s)
  - Dynamic covariance estimation
                          |
                          v
          Extended Kalman Filter (EKF) Fusion
  - DEAD_RECKONING & GNSS_DEGRADED: Fuses velocity update
  - GNSS_INS: Monitors and logs AI predictions
  - Invalid / unavailable: Fallback to inertial integration
```

## Validation and Fault-Tolerance Controls

1. **Input Validity Checking**:
   - Rejects non-finite values (NaN, Inf).
   - Validates that accelerometer magnitude is physically plausible (between 2.0 m/s^2 and 50.0 m/s^2, capturing Earth's 1g gravity field without sensor saturation).
   - Validates that gyroscope angular velocity magnitude is below 30.0 rad/s.
   - Enforces that input tensor strictly contains 6 IMU channels. GNSS speed or geographic coordinates are never fed to the neural network.

2. **Temporal Window Validation**:
   - Requires exactly 200 frames. Windows with fewer frames return status INSUFFICIENT_SAMPLES.
   - Checks that 200 samples span approximately 20 seconds (duration within 10.0 s to 35.0 s).

3. **Timestamp Validation**:
   - Enforces strictly monotonic timestamps (t_i > t_{i-1}).
   - Rejects duplicate, backwards, or zero-delta timestamps.
   - Verifies sampling intervals remain within acceptable limits (0.02 s to 0.5 s).

4. **Inference Failure Handling**:
   - All ONNX session executions are wrapped in guarded exception handlers.
   - If the model file is missing or corrupted, the engine initializes safely with is_ready() == False and returns status MODEL_UNAVAILABLE.
   - If session.run() raises a runtime exception, the engine increments failed_inferences and returns status INFERENCE_FAILED.
   - Under no circumstances does an AI failure interrupt filter propagation or crash navigation.

5. **Stale Prediction Detection**:
   - Compares the timestamp of the latest sample in the window against the current filter time.
   - If the data is older than 2.0 seconds, the prediction is rejected with status STALE_PREDICTION.

6. **Velocity Plausibility and Stationary Gating**:
   - Caps maximum plausible velocity at 65.0 m/s (~234 km/h).
   - Monitors variance of linear acceleration and gyroscope norm over the recent 50 frames.
   - When stationary conditions are detected (accel variance < 0.05 m/s^2 and gyro norm < 0.05 rad/s), any spurious non-zero AI predictions are gated to zero, eliminating standstill drift.

7. **Dynamic Measurement Covariance**:
   - Nominal base variance: R = 0.16 (m/s)^2 (sigma = 0.4 m/s).
   - Verified stationary state: R = 0.01 (m/s)^2 with high confidence (0.99).
   - High speed (> 30 m/s): Covariance scaled by 1.5x.
   - Timestamp jitter or elevated implied acceleration: Covariance scaled up to 3.0x to reduce filter gain.

8. **Navigation Fusion Integration**:
   - During GNSS outages (DEAD_RECKONING) or degraded GNSS (GNSS_DEGRADED), valid AI velocity is fused as a 2D measurement in the EKF state correction step.
   - In healthy GNSS mode (GNSS_INS), AI velocity is tracked in state telemetry without overriding direct GNSS updates.
   - If AI measurement is invalid or unavailable, the EKF continues with standard dead reckoning, Non-Holonomic Constraints (NHC), and Zero-Velocity Updates (ZUPT).

## Measured Inference Latency Benchmarks

Evaluated over 100 consecutive inferences on standard CPU execution:
- Mean Latency: 1.86 ms
- Median (P50) Latency: 1.83 ms
- 95th Percentile (P95) Latency: 2.26 ms
- Minimum Latency: 1.61 ms
- Maximum Latency: 2.37 ms

The 2.26 ms P95 latency consumes approximately 2.3% of the 100 ms (10 Hz) epoch budget, confirming real-time determinism.
