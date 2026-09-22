# Navigators — Experimental Methodology & Ablation Protocol

```
Document Identifier: EXP-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Experimental Design Philosophy

Navigators adheres to strict empirical guidelines for navigation benchmarking:
1. **Zero Ground-Truth Leakage**: During simulated or real GNSS outage intervals, GNSS measurements (including derived GNSS speed and ground-track heading) are completely withheld from the estimator. GNSS coordinates are sequestered strictly as an evaluation reference to compute positional error post-run.
2. **Identical Inputs Across Ablations**: To isolate the exact marginal utility of each algorithmic component, ablations A through G are evaluated on the exact same sensor stream, calibration prefix, and outage window.
3. **Deterministic Reproducibility**: Random seeds, filter tuning matrices, normalization parameters, and input dataset hashes are recorded in test logs.

---

## 2. Seven-Stage Ablation Hierarchy (A through G)

The evaluation suite (`replay.py` and `scripts/benchmark.py`) defines seven progressive system configurations:

| Mode | Identifier | Filtering | AI Velocity | 15-State EKF | NHC Constraints | ZUPT Detector | Map Matching |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Raw IMU Baseline | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **B** | Filtered IMU | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **C** | AI Velocity Only | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **D** | AI + EKF Fusion | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **E** | AI + EKF + NHC | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **F** | AI + EKF + NHC + ZUPT | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **G** | Full Production System | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Architectural Roles of Each Stage:
- **Mode A (Raw IMU)**: Direct double-integration of aligned raw accelerometer readings. Illustrates unconstrained MEMS drift where quadratic position error accumulates rapidly ($> 500\text{ m}$ in 60s).
- **Mode B (Filtered IMU)**: Incorporates running median and causal 20 Hz low-pass filtering. Suppresses engine acoustic noise spikes.
- **Mode C (AI Velocity Direct)**: Directly integrates TCN-predicted 2D velocity vectors without Kalman filtering: $\mathbf{p}_k = \mathbf{p}_{k-1} + \mathbf{v}_{tcn} \Delta t$.
- **Mode D (AI + EKF)**: Fuses TCN velocity as a pseudo-measurement into the 15-state EKF, actively estimating accelerometer and gyroscope biases.
- **Mode E (AI + EKF + NHC)**: Activates Non-Holonomic Constraints ($v_y^v \approx 0, v_z^v \approx 0$), eliminating lateral vehicle slide and vertical altitude drift.
- **Mode F (AI + EKF + NHC + ZUPT)**: Integrates zero-velocity updates during traffic stops, freezing position error accumulation during stationary intervals.
- **Mode G (Full System)**: Snaps the filtered state trajectory to OpenStreetMap road centerlines, eliminating cumulative cross-track angular drift.

---

## 3. Evaluation Metrics & Mathematical Definitions

All positioning metrics are computed in horizontal meters relative to the local East-North-Up (ENU) tangent origin:

### 3.1 Absolute Trajectory Error (ATE RMSE)
Root Mean Squared Error between estimated coordinates $[p_E, p_N]$ and ground-truth reference coordinates $[p_E^{ref}, p_N^{ref}]$ over scorable timesteps $N$:
$$\text{ATE RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left( (p_{E,k} - p_{E,k}^{ref})^2 + (p_{N,k} - p_{N,k}^{ref})^2 \right)}$$

### 3.2 Maximum & Final Position Drift
- **Maximum Drift**: $\max_k \|\mathbf{p}_k - \mathbf{p}_k^{ref}\|$ over the outage duration.
- **Final Position Drift**: Euclidean error at the concluding epoch of the outage window:
  $$e_{final} = \|\mathbf{p}_{outage\_end} - \mathbf{p}_{outage\_end}^{ref}\|$$

### 3.3 Velocity Vector RMSE
$$\text{Velocity RMSE} = \sqrt{\frac{1}{N} \sum_{k=1}^N \left( (v_{E,k} - v_{E,k}^{ref})^2 + (v_{N,k} - v_{N,k}^{ref})^2 \right)}$$

### 3.4 Heading Angular Error (RMSE)
Wrapped angular error in degrees, computed strictly when vehicle ground speed exceeds $0.5\text{ m/s}$ to avoid stationary gimbal noise:
$$\Delta\psi_k = \text{wrap}_{180}(\psi_k - \psi_k^{ref})$$
$$\text{Heading RMSE} = \sqrt{\frac{1}{M} \sum_{j=1}^M (\Delta\psi_j)^2}$$

---

## 4. Execution Protocols

### 4.1 Running the Benchmark Suite
To execute the automated ablation suite against an input dataset:

```bash
# Replay full seven-stage ablation on a trip recording:
python replay.py --dataset data/raw_trips/held_out_trip.json --gnss-outage 60 --outage-start 10 --ablations

# Equivalent benchmark script:
python scripts/benchmark.py --dataset data/raw_trips/held_out_trip.json --gnss-outage 60
```

### 4.2 Output File Artifacts
The evaluation writes structured audit artifacts to `results/replay/`:
- `trip_results.json`: Full configuration parameters, component call counts, execution timings, and numerical metrics across whole-trip, outage, and recovery intervals.
- `trip_[A-G]_trajectory.csv`: Detailed timestamped state trajectories ($t, p_E, p_N, v_E, v_N, \psi$, mode, GNSS allowed).
- `trip_[A-G]_trajectory.json`: Interactive playback files renderable directly in the browser dashboard.
- `trip_ablation.csv`: Consolidated side-by-side comparison table of all modes.

---

Developed by Navigators
