# Phase 5: Adaptive Sensor Fusion Architecture

## Executive Summary

Phase 5 implements mathematically justified adaptive measurement weighting within the Navigators navigation fusion engine. The system dynamically scales measurement covariances and process noise in response to real-time reliability metrics evaluated across GNSS, AI velocity inference, IMU propagation, and road geometry constraints.

Fixed arbitrary sensor weights are eliminated. Instead, measurement updates enter the 15-state Extended Kalman Filter (EKF) through dynamic measurement covariance matrices (R) and process noise matrices (Q) derived from statistical innovation residuals, sensor health states, and model inference diagnostics.

Accuracy notice: In accordance with rigorous engineering practice, improved navigational accuracy is not claimed until empirical benchmark datasets demonstrate it across field trials.

## Mathematical Uncertainty Formulation

### 1. Dynamic Measurement Covariance

For any observation vector $z_k$ with nominal covariance matrix $R_{0}$, the adaptive fusion engine determines an inflation scalar $s \ge 1.0$:

$$R_k = s \cdot R_0$$

Within the standard Kalman gain formulation:

$$K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}$$

As measurement uncertainty grows ($s \to \infty$), the Kalman gain approaches zero ($K_k \to 0$), naturally diminishing the update's influence on state vector $x_k$ and posterior error covariance $P_k$ without manual filter switching or arbitrary heuristic weighting.

### 2. Sensor Covariance Adaptation Rules

#### A. GNSS Position and Velocity Updates
Nominal GNSS position variance $\sigma_{\text{pos}}^2$ is scaled according to the GNSS Trust Engine state and anomaly detector outputs:
- **TRUSTED**: $s_{\text{pos}} = 1.0, s_{\text{vel}} = 1.0$. Kalman update proceeds with reported sensor accuracy.
- **DEGRADED**: $s_{\text{pos}} = \max(4.0, s_{\text{metric}}), s_{\text{vel}} = 0.5 \cdot s_{\text{pos}}$. GNSS variance is inflated by at least $4\times$, reducing innovation pull while retaining macroscopic fix guidance.
- **SUSPICIOUS**: $s_{\text{pos}} = \max(50.0, s_{\text{metric}}), s_{\text{vel}} = s_{\text{pos}}$. Severe innovation residuals, jump speeds, or heading divergence scale variance by $50\times$ or higher, effectively neutralizing the update.
- **UNUSABLE / LOST**: $s_{\text{pos}} \ge 10^4$. The measurement update is rejected entirely, avoiding state vector distortion.

#### B. AI Velocity Pseudo-Measurement
When enabled, the 1D temporal convolutional network (TCN) outputs body-to-world horizontal velocities $[v_{\text{east}}, v_{\text{north}}]$ alongside diagonal variance entries $[\sigma_{\text{east}}^2, \sigma_{\text{north}}^2]$:
- **Healthy GNSS**: $R_{\text{AI}} = s_{\text{gnss\_healthy}} \cdot \text{diag}(\sigma_{\text{east}}^2, \sigma_{\text{north}}^2)$ with $s = 4.0$. GNSS provides ground-truth velocity, while AI velocity acts as a secondary damping prior.
- **Degraded or Suspicious GNSS**: $s = 1.0$. AI velocity operates at full nominal confidence, preventing drift.
- **GNSS Lost (Dead Reckoning)**: $s = 0.5$. AI velocity variance is tightened, elevating its relative Kalman gain to curb accelerometer integration drift.
- **Stationary Gated**: When zero-velocity detection triggers, AI variance is scaled down ($s = 0.1$) to lock velocity to zero.
- **Degraded or Outlier AI**: If inference exceeds latency thresholds ($> 50$ ms), exhibits variance $> 1.0$ m$^2$/s$^2$, or shows kinematic acceleration $> 12$ m/s$^2$, the AI measurement is down-weighted or rejected.

#### C. Road Geometry Constraints
Road network hypotheses project the 2D position onto the highest-probability road center-line:
$$R_{\text{map}} = \left( \frac{\sigma_{\text{road}}^2}{\max(\text{confidence}, 0.05)} \right) \cdot s_{\text{map}} \cdot I_2$$
- When GNSS is degraded or lost, $s_{\text{map}} = 0.5$, increasing the relative authority of road geometry to bound cross-track error.
- When GNSS is healthy, $s_{\text{map}} = 2.0$, preventing map discrepancies or outdated map geometries from pulling accurate GNSS fixes off-target.

#### D. IMU Propagation and Process Noise (Q)
The process noise matrix $Q$ models acceleration and angular rate uncertainty:
- **Stationary**: Process noise for position and velocity is scaled down by $0.01\times$, suppressing random walk drift while stopped.
- **Dead Reckoning**: Process noise scales up ($2.0\times$ for position, $3.0\times$ for velocity) to accurately reflect growing dead-reckoning uncertainty in error covariance $P$.

## System Architecture

```
                       +-----------------------+
                       | Raw Sensor Ingestion  |
                       |  GNSS, IMU, Map, AI   |
                       +-----------+-----------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+-----------------+       +-----------------+       +-----------------+
|  GNSS Anomaly   |       |   AI Velocity   |       |  Road Matcher   |
|    Detector     |       |   Validation    |       |   Hypotheses    |
+--------+--------+       +--------+--------+       +--------+--------+
         |                         |                         |
         v                         |                         |
+-----------------+                |                         |
|   GNSS Trust    |                |                         |
|     Engine      |                |                         |
+--------+--------+                |                         |
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
                                   v
             +-------------------------------------------+
             |      Adaptive Fusion Engine               |
             |  - Calculates covariance scales           |
             |  - Generates down-weight / reject reasons |
             |  - Emits telemetry metrics                |
             +---------------------+---------------------+
                                   |
                                   v
             +-------------------------------------------+
             |    Authoritative Extended Kalman Filter   |
             |  - Dynamically scaled Q and R matrices    |
             |  - Authoritative navigation state output  |
             +-------------------------------------------+
```

## Backend Authoritativeness

A fundamental design requirement is that client applications and frontend layers must never determine sensor weights or fusion parameters. 

1. **State Isolation**: The navigation filter operates entirely inside the core engine.
2. **Authority**: Frontend consumers receive final navigation states along with diagnostic explanations. They cannot inject arbitrary covariance overrides.
3. **Auditability**: Every down-weighting or rejection decision includes explicit human-readable reasons in telemetry dictionaries.

## Telemetry and Diagnostic Reason Codes

The filter produces structured telemetry at every update cycle:

| Metric Key | Description | Example Values |
| :--- | :--- | :--- |
| `gnss_status` | Current GNSS health category | `HEALTHY`, `DEGRADED`, `SUSPICIOUS`, `LOST`, `UNUSABLE` |
| `gnss_weight` | Effective relative weight ($1 / s_{\text{pos}}$) | `1.0` (Full), `0.1` (Degraded), `0.02` (Suspicious), `0.0` (Lost) |
| `ai_status` | AI inference validation state | `VALID`, `STATIONARY`, `DEGRADED`, `HIGH_VARIANCE`, `REJECTED` |
| `ai_weight` | Effective relative weight ($1 / s_{\text{ai}}$) | `0.25` (GNSS Healthy), `1.0` (Nominal), `2.0` (GNSS Lost), `0.0` (Rejected) |
| `map_weight` | Effective map constraint weight ($1 / s_{\text{map}}$) | `0.5` (GNSS Healthy), `1.0` (Nominal), `2.0` (GNSS Lost), `0.0` (No Map) |
| `downweighted_reasons` | List of triggers that caused covariance inflation | `["GNSS down-weighted: degraded reported accuracy or recovery hysteresis"]` |
| `rejected_reasons` | List of triggers that caused measurement rejection | `["AI velocity rejected: latency 65.0ms exceeds threshold 50.0ms"]` |

## Validation Matrix Results

The adaptive fusion engine is validated across deterministic test suites covering all required scenarios:

| # | Validation Scenario | GNSS Status | AI Status | Map Status | Expected Behavior | Test Status |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| 1 | GNSS Healthy | TRUSTED ($s=1$) | VALID ($s=4$) | Present ($s=2$) | GNSS dominates position; AI and Map maintain background damping | PASS |
| 2 | GNSS Degraded | DEGRADED ($s=10$) | VALID ($s=1$) | Present ($s=1$) | GNSS variance inflated; AI velocity and Map gain relative influence | PASS |
| 3 | GNSS Suspicious | SUSPICIOUS ($s=50$) | VALID ($s=1$) | Present ($s=1$) | GNSS down-weighted to $\le 2\%$; filter relies on AI velocity | PASS |
| 4 | GNSS Lost | UNUSABLE / None ($s=10^4$) | VALID ($s=0.5$) | Present ($s=0.5$) | GNSS update omitted; AI velocity and Map constraint bound DR drift | PASS |
| 5 | AI Unavailable | TRUSTED / DEGRADED | UNAVAILABLE ($s=10^4$) | Optional | AI update rejected; filter relies on GNSS and pure IMU propagation | PASS |
| 6 | AI Degraded | TRUSTED / DEGRADED | HIGH_VARIANCE ($s=5$) | Optional | AI measurement down-weighted ($s \ge 5.0$); no corrupting velocity pulls | PASS |
| 7 | Map Unavailable | Any | Any | None / Invalid | Map constraint update skipped; telemetry notes absence | PASS |
| 8 | Simultaneous Degradations | DEGRADED / SUSPICIOUS | DEGRADED / HIGH_VARIANCE | None | Both GNSS and AI covariances inflated; filter falls back to IMU DR | PASS |

Additional integration tests confirm:
- **Stationary Damping**: Process noise covariance scales drop by $100\times$ during detected stationary dwell, preventing position and velocity creep.
- **EKF State Consistency**: Covariance symmetry and positive-definiteness are rigorously maintained through Joseph-form covariance updates under high variance scaling.
