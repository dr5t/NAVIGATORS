# Navigators Navigation Foundation Technical Audit

```
Document Identifier: NAV-AUDIT-P1
Status: Production Reference
Version: 1.0.0
Date: 2026-09-25
Scope: Core Navigation Pipeline Foundation Audit & Research Interfaces
```

---

## 1. Executive Summary and Architecture

This technical audit provides a comprehensive evaluation of the complete navigation pipeline across both runtime environments:
1. Python Reference and Evaluation Engine (`src/navigation/`, `src/evaluation/`, `src/data/`, `src/models/`, `replay.py`)
2. On-Device Edge / Browser Simulator Engine (`simulator/offline_engine.js`, `simulator/engine/`)

The navigation engine operates as an autonomous kinematic estimator designed to provide continuous location, velocity, and attitude estimates during standard operating conditions, degraded satellite geometry, and total GNSS outages (such as road tunnels or urban canyons).

The existing working implementation serves as the authoritative source of truth. Clean, type-safe interfaces have been formulated in `src/navigation/interfaces.py` to prepare exact integration points for future research phases without modifying working navigation logic.

---

## 2. Sensor Ingestion Audit

### 2.1 Accelerometer Ingestion
- Physical Signal: Specific force vector in three orthogonal axes [ax, ay, az] in meters per second squared (m/s^2).
- Data Sources:
  * Smartphone MEMS via HTML5 `DeviceMotionEvent.accelerationIncludingGravity` (device coordinate frame).
  * External Fiber Optic Gyroscope (FOG) / tactical IMU via `ExternalFOGIMUAdapter` (`src/data/sensor_adapter.py`).
  * Recorded / benchmark trip datasets (IO-VNBD and phone recordings via `load_recording`).
- Coordinate Frame: Raw measurements arrive in device body frame {b}.
- Range and Precision: Smartphone accelerometer typical range +/- 16g; noise density ~0.005 m/s^2/sqrt(Hz) on tactical units.

### 2.2 Gyroscope Ingestion
- Physical Signal: Angular rate vector around three orthogonal axes [wx, wy, wz] in radians per second (rad/s).
- Data Sources:
  * Smartphone MEMS via HTML5 `DeviceMotionEvent.rotationRate` (beta, gamma, alpha in degrees per second, converted via pi/180).
  * External FOG adapter with calibrated bias drift (<0.01 ppm).
- Axis Conventions: Converted to vehicle body frame {v} (roll rate, pitch rate, yaw rate) through alignment rotation matrix.

### 2.3 Magnetometer / Compass Ingestion
- Physical Signal: Device orientation angles relative to magnetic / true North.
- Status: Optional / secondary in smartphone runtimes.
- Implementation:
  * Captured via `deviceorientationabsolute` or `deviceorientation` event (`webkitCompassHeading` on WebKit / iOS, or (360 - alpha) on standard absolute compass).
  * In `simulator/offline_engine.js`, the compass heading is down-weighted with time-decaying covariance: `compassSigma = min(1.0, 0.3 + drDuration * 0.01)`.
  * Not required for vehicle navigation when forward acceleration calibration and gyroscope integration are active, but utilized as a weak heading anchor during prolonged stationary or low-dynamic GNSS outages.

### 2.4 GNSS Ingestion
- Signals Captured:
  * Geodetic position: WGS-84 latitude and longitude in decimal degrees.
  * Altitude: Ellipsoidal / orthometric height in meters (where available).
  * Horizontal accuracy: Estimated 1-sigma uncertainty radius in meters.
  * Ground speed: Scalar velocity over ground in m/s.
  * Course over ground: Track angle / heading in degrees clockwise from North.
  * Timestamp: Millisecond epoch timestamp from satellite receiver.
- Geolocation API: Browser `navigator.geolocation.watchPosition` with `{ enableHighAccuracy: true, maximumAge: 0, timeout: 3000 }`.
- Stale Fix Rejection: Fixes older than 3.0 seconds are discarded (`Date.now() - position.timestamp > 3000`).
- Duplicate Fix Suppression: Freshness tracking prevents fusing identical GNSS epochs into the EKF multiple times.

### 2.5 Timestamps and Sampling Rates
- Clock Sources: Monotonic high-resolution clock (`performance.now()` in ms / seconds) for interval calculations; UTC timestamps for telemetry logging.
- Sampling Frequencies:
  * Smartphone IMU: Nominal 10 Hz to 50 Hz depending on browser/hardware capabilities.
  * Tactical / External FOG: 200 Hz to 1000 Hz.
  * GNSS updates: Nominal 1 Hz.
  * Navigation loop: Fixed 10 Hz cycle (100 ms tick interval).
- Dynamic Rate Tracking: Both Python replay and JavaScript offline engine compute effective IMU sampling rates dynamically. If effective rate diverges from model contract (10 Hz) by > 25%, AI inference is gated with a `RATE_MISMATCH` diagnostic to protect estimation integrity.
- Delta Time Protection: Delta t is strictly clamped (`0.0 < dt <= 1.0`, nominal 0.1 s) to prevent numerical divergence on frame drop or background throttling.

---

## 3. Sensor Preprocessing Audit

### 3.1 Denoising and Filtering
The repository implements two filtering paradigms:
1. Online Causal Pipeline (`src/evaluation/preprocessing.py`, `simulator/engine/preprocessing.js`):
   * Median filter over a sliding window of 5 samples to remove shock impulses, potholes, and mechanical spikes without phase delay.
   * Single-pole causal low-pass filter with 20 Hz cutoff: `filtered = filtered + (1 - exp(-2 * pi * 20 * dt)) * (median - filtered)`.
   * Preserves causality: output at time t depends exclusively on past samples t' <= t.
2. Offline Training Pipeline (`src/data/preprocessor.py`):
   * 4th-order zero-phase Butterworth low-pass filter with 3.0 Hz cutoff frequency.
   * Median filter with kernel size 5 for impulsive outlier rejection.

### 3.2 Attitude Alignment and Calibration
Phone orientation inside a vehicle is arbitrary. The system estimates the orthogonal rotation matrix R_vehicle_phone mapping Phone frame to Vehicle frame:
1. Step 1 (Gravity Extraction): While the vehicle is stationary (dwell phase), specific force averages to local gravity. The downward unit vector Z_down is extracted:
   Z_down = mean(accel) / norm(mean(accel))
2. Step 2 (Forward Axis Extraction): When the vehicle accelerates forward (GNSS acceleration rate > 0.5 m/s^2), the net horizontal acceleration vector is projected orthogonal to Z_down:
   X_fwd = accel_horizontal / norm(accel_horizontal)
3. Step 3 (Triad Orthogonalization):
   Y_right = cross(Z_down, X_fwd) / norm(cross(Z_down, X_fwd))
   X_fwd = cross(Y_right, Z_down)
   R_vehicle_phone = stack([X_fwd, Y_right, Z_down])
4. Implementation Status: Symmetrically implemented in Python (`src/navigation/alignment.py`) and JavaScript (`simulator/engine/alignment.js`).

### 3.3 Coordinate Transforms
The pipeline maps states across four distinct coordinate systems:
1. Device Body Frame {b}: Sensor casing axes.
2. Vehicle Body Frame {v}: Forward (X), Right (Y), Down (Z).
3. Local Navigation Frame {n} (ENU): East (X), North (Y), Up (Z).
4. Earth Frame {e} (WGS-84): Latitude, Longitude, Altitude.

Geodetic to ENU Conversion:
- Tangent plane approximation centered at initial GNSS anchor (lat_0, lon_0):
  East = (lon - lon_0) * lon_scale
  North = (lat - lat_0) * meters_per_degree_lat
  Up = alt - alt_0
  where meters_per_degree_lat = 111132.954 - 559.822 * cos(2 * lat_0) + 1.175 * cos(4 * lat_0)
  lon_scale = meters_per_degree_lat * cos(deg_to_rad(lat_0))

### 3.4 Gravity Handling
- Specific force measured by accelerometer includes the reaction to gravity: f = a - g.
- In Local ENU navigation frame, gravity is defined as g_nav = [0.0, 0.0, -9.81] m/s^2.
- The linear acceleration is obtained by rotating corrected body accelerometer readings into navigation frame and subtracting gravity:
  accel_nav = R_b2n @ (accel_body - accel_bias) - g_nav.
- In feature preprocessing for AI inference, gravity (9.81 m/s^2) is removed along the vehicle vertical axis (Z).

### 3.5 Normalization
- Feature Dimensions: 6 input channels [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
- Method: Z-score normalization per channel:
  x_norm = (x - mean) / std.
- Precomputed Contract (`simulator/model.contract.json`, `checkpoints/norm_stats.json`):
  * Mean: [-0.023565, -0.035606, 8.931498, -0.000460, -0.000460, -0.000024]
  * Std: [1.599169, 1.517184, 1.158331, 0.113666, 0.113666, 0.131972]
- Integrity Check: Replay and edge engine verify that normalization statistics exactly match the training contract before running inference.

---

## 4. AI Velocity Inference Audit

### 4.1 Temporal Convolutional Network (TCN) Architecture
- Model Class: `TCNVelocityEstimator` (`src/models/tcn_model.py`).
- Input Specification: Tensor of shape `(batch, 200, 6)` corresponding to 200 temporal frames of 6 normalized IMU channels.
- Temporal Window: 200 frames at 10 Hz sampling rate = 20.0 seconds temporal window.
- Block Topology:
  * 5 Residual TCN Blocks with channel widths [64, 128, 256, 256, 512].
  * Causal Dilated 1D Convolutions with kernel size 7 and exponential dilation: d in [1, 2, 4, 8, 16].
  * Left-padding `padding = (kernel_size - 1) * dilation` trimmed to input length ensures strict causality (no future leakage).
  * Layer composition per block: CausalConv1d -> BatchNorm1d -> ReLU -> Dropout(0.2) -> CausalConv1d -> BatchNorm1d -> ReLU -> Dropout(0.2) + 1x1 Residual Projection.
  * Skip connections project intermediate block features to 512 channels.
  * Head: AdaptiveAvgPool1d(1) -> Flatten -> Linear(512, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 2).
- Parameter Count: ~1.2M parameters (lightweight edge-grade footprint).
- Output: 2D velocity vector [v_north, v_east] in m/s.

### 4.2 ONNX Runtime Edge Path
- Model Artifacts:
  * `simulator/model.onnx`: ONNX execution graph (opset 13).
  * `simulator/model.onnx.data`: Externalized model weights (2.23 MB).
  * `simulator/model.contract.json`: Verification metadata, input shapes, mean/std, output order.
- Engine Loading:
  * Desktop/Python: `onnxruntime.InferenceSession` with `CPUExecutionProvider`.
  * Browser: ONNX Runtime Web (`ort.env.wasm`) using WebAssembly backend with single-thread optimization (`ort.env.wasm.numThreads = 1`).
- Verification Status: Validated with 4.20 m/s MAE on IO-VNBD dataset.

### 4.3 Inference Frequency, Gating, and Latency
- Execution Cycle: Invoked at 10 Hz during GNSS outages and degraded states when:
  1. IMU sensor buffer is fully populated (200 samples).
  2. Vehicle motion detector confirms the vehicle is in motion (ZUPT stationary gate inactive).
  3. Effective sensor sampling rate matches model contract within +/- 25%.
- Stationary Gating: When the vehicle is stopped (detected via IMU variance or speed < 0.4 m/s), AI inference is explicitly bypassed (`STATIONARY_GATED`) to prevent synthetic integration drift while stopped at signals.
- Measured Execution Latency:
  * Desktop Python (M-series / x86): 2.5 ms to 4.5 ms per inference.
  * Browser WebAssembly: 6.0 ms to 12.0 ms per inference.
  * Maximum Latency Budget: 100 ms (meets 10 Hz real-time deadlines with >88% timing margin).

---

## 5. Navigation State Machine Audit

### 5.1 Architecture and Contract
The navigation state machine is authoritatively managed in Python by `NavigationStateEngine` (`src/navigation/nav_state.py`) and in JavaScript by `GnssStateMachine` (`simulator/offline_engine.js`).
Architectural Rule: The UI client strictly consumes states emitted by the navigation engine; the UI never independently decides state transitions.

### 5.2 Operating States
1. `STANDBY`: Navigation engine initialized, awaiting initial GNSS fix or sensor start.
2. `NORMAL`: Continuous GNSS + INS sensor fusion active with high accuracy (HDOP <= 2.5, accuracy <= 20 m).
3. `GNSS_DEGRADED`: High HDOP (> 2.5-3.0) or reduced satellite geometry (accuracy 20 m to 50 m); GNSS measurement noise inflated; inertial fusion weighted higher.
4. `DR` (DEAD_RECKONING): GNSS lost or denied; TCN AI velocity + Non-Holonomic Constraints + Zero Velocity Updates active.
5. `REACQUIRING`: GNSS signal restored after outage; anti-jump trajectory smoothing active.
6. `STOPPED`: Navigation session terminated normally.
7. `ERROR`: Critical sensor fault (e.g., IMU hardware failure or numerical NaN).

### 5.3 Valid Transition Matrix
```
STANDBY       -> {NORMAL, GNSS_DEGRADED, DR, STOPPED, ERROR}
NORMAL        -> {GNSS_DEGRADED, DR, STOPPED, ERROR}
GNSS_DEGRADED -> {NORMAL, DR, STOPPED, ERROR}
DR            -> {REACQUIRING, STOPPED, ERROR}
REACQUIRING   -> {NORMAL, GNSS_DEGRADED, DR, STOPPED, ERROR}
STOPPED       -> {STANDBY, NORMAL}
ERROR         -> {STANDBY, STOPPED}
```
Direct jumps from DR to NORMAL or STANDBY to REACQUIRING are strictly forbidden and throw ValueError.

### 5.4 Anti-Jump Reacquisition Convergence
When GNSS returns after a tunnel or outage:
- State transitions from `DR` to `REACQUIRING`.
- Innovation covariance ramp: Measurement noise R_pos is scaled by `1.0 / ramp` where `ramp = max(0.1, min(1.0, steps / 10.0))`.
- Step Bounding: Positional innovation norm is clamped to a maximum step of 2.0 m per 100 ms cycle:
  correction_fraction = min(1.0, 2.0 / norm(diff))
  position_new = position_old + correction_fraction * diff
- Convergence Trigger: Once consecutive good fixes >= 5 and residual <= 3.0 m, state returns to `NORMAL`.

---

## 6. Sensor Fusion Audit (15-State EKF)

### 6.1 State Vector Formulation
The Extended Kalman Filter state vector x in R^15 is defined in the local East-North-Up (ENU) frame:
```
Indices 0:3   - Position (p_East, p_North, p_Up) in meters
Indices 3:6   - Velocity (v_East, v_North, v_Up) in m/s
Indices 6:9   - Orientation (roll, pitch, yaw) in radians (yaw: 0 = North, pi/2 = East)
Indices 9:12  - Accelerometer bias (b_ax, b_ay, b_az) in m/s^2
Indices 12:15 - Gyroscope bias (b_gx, b_gy, b_gz) in rad/s
```

### 6.2 Process Model and Propagation
At every IMU timestep dt:
1. Bias correction:
   accel_corrected = accel_body - b_a
   gyro_corrected = gyro_body - b_g
2. Direction Cosine Matrix R_b2n calculated from roll, pitch, yaw.
3. Specific force transformed to navigation frame and gravity compensated:
   accel_nav = R_b2n @ accel_corrected - [0, 0, -9.81]
4. State propagation:
   pos_k = pos_{k-1} + vel_{k-1} * dt + 0.5 * accel_nav * dt^2
   vel_k = vel_{k-1} + accel_nav * dt
   orient_k = orient_{k-1} + gyro_corrected * dt
5. Discrete Jacobian F_k:
   F[0:3, 3:6] = I_3 * dt
   F[3:6, 6:9] = -R_b2n @ [accel_corrected]_x * dt
   F[3:6, 9:12] = -R_b2n * dt
   F[6:9, 12:15] = -I_3 * dt
6. Covariance propagation with Joseph-form symmetry preservation:
   P = F @ P @ F.T + Q
   P = 0.5 * (P + P.T)
   In DR mode, position process noise is multiplied by 4.0 and velocity noise by 2.0 to account for unobserved drift.

### 6.3 Measurement Models
1. GNSS Position and Velocity Update:
   H_pos = [I_3, 0_3x12]
   H_vel = [0_3x3, I_3, 0_3x9]
   R_pos = diag(sigma_pos^2), R_vel = diag(sigma_vel^2)
2. AI Velocity Pseudo-Measurement Update:
   H_ai = [[0, 0, 0, 1, 0, 0, 0_1x9],
           [0, 0, 0, 0, 1, 0, 0_1x9]]
   z_ai = [v_east, v_north]^T
   R_ai = diag(sigma_ai^2, sigma_ai^2), sigma_ai = 0.3 m/s
3. Non-Holonomic Constraints (NHC) (`src/navigation/nhc.py`):
   Assumes land vehicle does not slip laterally or jump vertically.
   v_lateral = v_East * cos(heading) - v_North * sin(heading) approx 0
   v_vertical = v_Up approx 0
   H_nhc = [[0, 0, 0, cos(heading), -sin(heading), 0, 0_1x9],
            [0, 0, 0, 0, 0, -1, 0_1x9]]
   Turn Relaxation: Lateral noise variance is dynamically inflated during turns to prevent fighting steering dynamics:
   r_lat = r_nhc_lateral * sqrt(1.0 + 30.0 * (abs(yaw_rate)^2))
4. Zero Velocity Updates (ZUPT) (`src/navigation/zupt.py`):
   When vehicle is detected stationary:
   H_zupt = [0_3x3, I_3, 0_3x9]
   z_zupt = [0, 0, 0]^T
   R_zupt = diag(0.01^2, 0.01^2, 0.01^2)

### 6.4 Covariance Update Stability
All measurement updates utilize the Joseph-form covariance equation:
P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
followed by explicit symmetry enforcement and minimum diagonal variance clipping (1e-9) to guarantee positive definiteness.

---

## 7. Map Matching Audit

### 7.1 Current Implementations
Two algorithms exist in the codebase:
1. `GeometricMapMatcher` (`src/navigation/map_matching.py`, `simulator/engine/map_matcher.js`):
   * Snaps position to the nearest road segment within a search radius (typically 30 m to 50 m).
   * Calculates perpendicular projection: t = clip(dot(point - start, segment_vec) / length_sq, 0, 1).
   * Filters out segments with heading mismatch > 45 degrees.
   * Feedback logic: When confidence > 0.8 during DR, snapped coordinates blend into position:
     pos = 0.9 * pos + 0.1 * snapped_pos.
2. `HMMMapMatcher` (`src/navigation/map_matching.py`):
   * Hidden Markov Model using Viterbi sequence decoding.
   * Emission probability: Gaussian distribution on orthogonal distance to centerline:
     p(z|r) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * (dist / sigma)^2).
   * Transition probability: Route plausibility based on distance difference between road distance and traveled distance:
     p(r_j | r_k) = (1 / beta) * exp(-abs(route_dist - travel_dist) / beta).

### 7.2 Road Network Input Format
- Vector format: GeoJSON / JSON (`simulator/data/road_network.json`).
- Structure:
  * Reference origin: latitude and longitude.
  * Roads list: Array of roads containing segment points, road ID, road name, speed limit, and one-way flag.
  * Spatial indexing: `scipy.spatial.cKDTree` in Python, spatial grid hash in JavaScript.

### 7.3 Map Matching Limitations
1. Single-Hypothesis Overwrite: Position blending (0.9/0.1) modifies state outside the Kalman update equation, which can distort covariance P if an incorrect road segment is selected.
2. Overpass and Multi-Level Ambiguity: Altitude is not matched; 2D projection cannot resolve stacked flyovers or parallel service roads.
3. No Topological Turn Awareness: Does not model turning restrictions, lane geometry, or junction decision branching.

---

## 8. Navigation Output Audit

### 8.1 State Telemetry Payload
The navigation output provides:
- Coordinates: ENU position [p_E, p_N, p_U] in meters and WGS-84 [latitude, longitude, altitude].
- Velocity: ENU velocity [v_E, v_N, v_U] in m/s, scalar speed in m/s and km/h.
- Attitude: Euler angles (roll, pitch, yaw) in radians, heading in degrees clockwise from North.
- Navigation Mode: `gnss_aided`, `gnss_ins`, `gnss_degraded`, `dr`, `reacq`.
- Sensor Biases: Accelerometer bias [b_ax, b_ay, b_az] and gyroscope bias [b_gx, b_gy, b_gz].

### 8.2 Confidence and Uncertainty Metrics
- EKF Position Uncertainty: 1-sigma horizontal position standard deviation:
  sigma_pos = sqrt(P[0, 0] + P[1, 1]) in meters.
- Dead Reckoning Drift Percent:
  drift_pct = (estimated_drift_m / distance_traveled_m) * 100.
- Map Match Confidence: Scalar score in [0.0, 1.0] based on orthogonal distance and heading alignment.
- Motion Diagnostics: Acceleration variance, dynamic acceleration magnitude, gyroscope variance, motion state (STATIONARY / MOVING / UNKNOWN).

---

## 9. Recommended Integration Points for Research Phases

The following 9 research components have been formalized as abstract interfaces in `src/navigation/interfaces.py`. Below are their exact architectural integration points in the navigation pipeline.

### 9.1 GNSS Trust Engine (`IGNSSTrustEngine`)
- Interface: `IGNSSTrustEngine` -> `evaluate_trust(gnss_data, imu_accel, dt) -> GNSSTrustMetric`
- Integration Point:
  * File: `src/navigation/ekf.py` -> before `ExtendedKalmanFilter.update_gnss()` (Line 212).
  * File: `simulator/offline_engine.js` -> before `this.ekf.updateGnss()` (Line 814).
- Purpose: Continuously evaluates raw GNSS quality (C/N0, satellite geometry, carrier phase residuals, velocity Doppler consistency) to output a continuous trust score in [0, 1] and dynamic variance scaling factor, replacing coarse step thresholds.

### 9.2 GNSS Anomaly Detection (`IGNSSAnomalyDetector`)
- Interface: `IGNSSAnomalyDetector` -> `detect_anomalies(gnss_data, predicted_state, innovation) -> GNSSAnomalyReport`
- Integration Point:
  * File: `src/navigation/ekf.py` -> inside `update_gnss()` at innovation computation (Line 255: `y_pos = z_pos - H_pos @ self.x`).
  * File: `simulator/offline_engine.js` -> inside EKF measurement update.
- Purpose: Analyzes normalized innovation squared (NIS) and chi-square test on measurement residuals to detect multipath reflections, step jumps, GNSS jamming, or spoofing before measurement fusion.

### 9.3 AI Velocity Measurement Engine (`IAIVelocityMeasurement`)
- Interface: `IAIVelocityMeasurement` -> `estimate_velocity(imu_window) -> AIVelocityMeasurement`
- Integration Point:
  * File: `src/navigation/ekf.py` -> `ExtendedKalmanFilter._update_ai_velocity()` (Line 318).
  * File: `simulator/offline_engine.js` -> `processInferenceStep()` (Line 720).
- Purpose: Standardizes deep learning velocity inference with explicit North/East uncertainty variances, enabling heteroscedastic noise weighting in EKF measurement covariance R_ai.

### 9.4 Adaptive Fusion Engine (`IAdaptiveFusionEngine`)
- Interface: `IAdaptiveFusionEngine` -> `compute_adaptive_noise(current_mode, innovation, motion_state) -> AdaptiveNoiseParameters`
- Integration Point:
  * File: `src/navigation/ekf.py` -> inside `ExtendedKalmanFilter.predict()` (Line 195: `Q = self._build_process_noise()`) and `update_gnss()` (Line 245: `R_pos`).
- Purpose: Dynamically adjusts process noise Q and measurement noise R matrices based on vehicle dynamics, vibration levels, and innovation sequences using covariance matching or Sage-Husa adaptive filtering.

### 9.5 Confidence Estimation Engine (`IConfidenceEstimator`)
- Interface: `IConfidenceEstimator` -> `compute_confidence(covariance, dr_duration, map_confidence) -> ConfidenceEstimate`
- Integration Point:
  * File: `src/navigation/ekf.py` -> `get_state_summary()` (Line 426).
  * File: `src/navigation/dead_reckoning.py` -> `get_confidence()` (Line 183).
  * File: `simulator/offline_engine.js` -> UI payload builder (Line 859).
- Purpose: Combines EKF state covariance, duration of GNSS denial, map match quality, and IMU thermal bias accumulation into an integrity-monitored confidence score for user display.

### 9.6 Road Hypotheses Tracker (`IRoadHypothesisTracker`)
- Interface: `IRoadHypothesisTracker` -> `update_hypotheses(position, heading, speed) -> List[RoadHypothesis]`
- Integration Point:
  * File: `src/navigation/map_matching.py` -> inside `GeometricMapMatcher.match()` and `HMMMapMatcher.match()`.
  * File: `simulator/offline_engine.js` -> before position snapping (Line 841).
- Purpose: Maintains multiple candidate road hypotheses across parallel roads, intersections, and ramps, replacing single nearest-segment assignment with probabilistic track trees.

### 9.7 Map Constraints Engine (`IMapConstraintEngine`)
- Interface: `IMapConstraintEngine` -> `generate_constraints(hypothesis, state) -> Optional[MapConstraint]`
- Integration Point:
  * File: `src/navigation/ekf.py` -> after `predict()`, as a formal Kalman measurement update `update_map_constraint(H, z, R)`.
- Purpose: Formulates road centerline and lane direction constraints as formal EKF measurement updates rather than ad-hoc coordinate overwrites, bounding lateral drift while maintaining mathematical covariance consistency.

### 9.8 GNSS Prediction Engine (`IGNSSPredictor`)
- Interface: `IGNSSPredictor` -> `predict_gnss(last_known_state, outage_duration) -> GNSSPrediction`
- Integration Point:
  * File: `src/navigation/nav_state.py` -> on transition into `DR` state (Line 113).
  * File: `src/navigation/dead_reckoning.py` -> `start()` (Line 45).
- Purpose: Extrapolates expected satellite visibility, Doppler shifts, and spatial entry coordinates for exit portals (such as the far end of a tunnel) to accelerate receiver reacquisition.

### 9.9 GNSS Recovery Manager (`IGNSSRecoveryManager`)
- Interface: `IGNSSRecoveryManager` -> `compute_recovery_step(current_estimate, fresh_gnss, recovery_step_index) -> GNSSRecoveryPlan`
- Integration Point:
  * File: `src/navigation/ekf.py` -> inside `update_gnss()` during `NavigationMode.REACQUISITION` (Lines 249-275 and 304-317).
  * File: `simulator/offline_engine.js` -> reacquisition state branch (Line 50).
- Purpose: Manages non-linear, multi-step anti-jump trajectory reconciliation upon GNSS restoration, ensuring that position corrections converge smoothly without visual snapping or routing disruption.

---

## 10. System Limitations and Experimental Boundaries

1. Physical In-Vehicle Road Trials Unvalidated: While software modules and automated tests pass across synthetic and recorded benchmarks, physical in-vehicle road testing with a physical device moving through real-world tunnels has not been performed.
2. Sensor Thermal Drift: Smartphone MEMS sensors lack hardware thermal stabilization. Under heavy processor load, accelerometer and gyroscope bias drift can deviate from static calibration values.
3. Mount Rigidity Requirement: The alignment rotation matrix assumes rigid attachment to the vehicle chassis (e.g. firm dashboard mount). Handheld use introduces substantial unmodeled attitude jitter.
4. Background Throttling: Mobile operating systems throttle browser background timers and sensor events. The navigation runtime must remain foregrounded with active Screen Wake Lock.

---

Report Prepared by Navigators Core Engineering
Authored for Phase 1 Navigation Foundation Audit
