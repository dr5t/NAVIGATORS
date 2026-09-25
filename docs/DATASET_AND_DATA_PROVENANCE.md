# Navigators — Dataset & Data Provenance Specification

```
Document Identifier: DATA-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Data Classification Framework

The [Phase 11 India Dataset pipeline](NAVIGATORS_INDIA_DATASET.md) is the executable
contract for new research releases. Historical external benchmark figures below
are not counts of original Navigators India data. No original collection size or
road accuracy is established by the pipeline implementation or its synthetic tests.

To maintain scientific integrity and respect data licensing, Navigators establishes a transparent, three-tier classification of all datasets utilized in the repository:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. PUBLIC EXTERNAL BENCHMARK DATASET                                   │
│    • IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark)      │
│    • Authors: Onyekpeu et al. (Open Access / Research Use)             │
│    • Provenance: External public repository (Git LFS)                  │
├────────────────────────────────────────────────────────────────────────┤
│ 2. DERIVED & SYNTHETIC DATASETS                                        │
│    • Kinematic simulated vehicle trajectories                          │
│    • Preprocessed and windowed feature tensors (105,706 windows)       │
│    • Provenance: Formatted via `src/data/` data preparation pipeline   │
├────────────────────────────────────────────────────────────────────────┤
│ 3. NAVIGATORS INGESTED TELEMETRY (OWN COLLECTED)                       │
│    • Crowdsourced smartphone drives recorded via browser client        │
│    • Provenance: User-consented local recordings via `data_recorder.js`│
│    • Current Status: Prototype diagnostics (`data/raw_trips/`)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. External Dataset: IO-VNBD Benchmark

### 2.1 Attribution & Citation
- **Full Title**: Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD)
- **Creators / Authors**: Uche Onyekpeu et al.
- **Repository**: [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
- **Ownership Notice**: Navigators claims zero proprietary ownership over IO-VNBD. It is utilized strictly under open scientific benchmarking terms.

### 2.2 Dataset Structure & Sensor Payloads
The dataset consists of synchronized CSV pairs recorded across multiple driving sessions:
- **`S_*.csv` (Sensor Stream)**: High-rate vehicle sensor readings.
  - Accelerometer: $a_x, a_y, a_z$ in meters per second squared ($\text{m/s}^2$).
  - Gyroscope: $\omega_x, \omega_y, \omega_z$ in radians per second ($\text{rad/s}$).
  - Magnetometer: $\mu_x, \mu_y, \mu_z$ in microteslas ($\mu\text{T}$).
- **`V_*.csv` (Ground Truth Kinematics)**: High-precision GNSS/INS reference solution.
  - Linear Speed: Velocity magnitude in meters per second ($\text{m/s}$).
  - Ground Track: True heading in degrees ($^\circ$).
  - Cartesian Velocities: Converted to North ($v_N$) and East ($v_E$) coordinates:
    $$v_N = \text{Speed} \times \cos(\text{Heading}_{rad})$$
    $$v_E = \text{Speed} \times \sin(\text{Heading}_{rad})$$

### 2.3 Sampling Frequency & Resampling
Raw IO-VNBD sensor files sampled at $100\text{ Hz}$ are downsampled to a uniform $10\text{ Hz}$ rate using an anti-aliasing low-pass filter to match the nominal polling frequency of smartphone browser runtimes.

---

## 3. Session-Level Partitioning & Leakage Elimination

To ensure strict zero-leakage validation, the 144 independent driving sessions in IO-VNBD are partitioned at the file/session level:

| Split | Number of Sessions | Total Windows ($W=200$) | Fraction | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 100 sessions | 63,423 windows | $70.0\%$ | Model parameter optimization. |
| **Validation** | 22 sessions | 30,349 windows | $15.0\%$ | Early stopping and hyperparameter tuning. |
| **Test (Held-Out)**| 22 sessions | 11,934 windows | $15.0\%$ | Final generalization benchmark evaluation. |
| **Total** | **144 sessions** | **105,706 windows** | **100.0%** | Comprehensive vehicle dynamics. |

---

## 4. Derived Synthetic & Diagnostic Datasets

Located in `data/synthetic_trajectories/` and `data/real_dataset/`:
- **Synthetic Vehicle Trajectories**: Mathematically simulated vehicle paths (constant acceleration, circular turns, stop-and-go maneuvers) corrupted with Gaussian noise and simulated accelerometer/gyroscope bias drift. Used in automated unit testing (`tests/test_ekf.py`, `tests/test_dead_reckoning.py`) to verify estimator convergence under known ground truth.
- **Trip 4 Diagnostic Log (`data/raw_trips/trip_4.json`)**: Early phone recording used for file format validation and recorder pipeline testing. Note: As documented in `docs/replay-and-evaluation.md`, Trip 4 exhibits non-monotonic GPS timestamps and is restricted to diagnostic format checks, not scientific accuracy claims.

---

## 5. Ingested Smartphone Contributor Data Protocol

The mobile web client incorporates an integrated sensor logger (`simulator/data_recorder.js`):
1. **User Consent**: Prior to recording, users must explicitly grant location and motion permissions and toggle the dataset consent switch (`consent: true`).
2. **Recorded Fields**:
   - `timestamps`: Microsecond-resolution elapsed time.
   - `accel`: Tri-axial acceleration including Earth gravity ($\text{m/s}^2$).
   - `gyro`: Tri-axial rotation rate ($\text{rad/s}$).
   - `gnss`: Array of $[lat, lon, altitude, speed, heading, accuracy]$.
   - `metadata`: User-agent string, device platform, browser version, and SHA-256 integrity hash.
3. **Data Quality Verification**: Administrative approval in `DatasetRepository`
   is distinct from sensor validation. New India Dataset releases must pass the
   [Phase 11 validation pipeline](NAVIGATORS_INDIA_DATASET.md), which archives raw
   bytes, rejects invalid sessions, and records the actual sampling, unit, axis,
   GNSS, identity and duplicate checks and their configurable thresholds.

---

Developed by Navigators
