# Road Hypothesis Engine: Ambiguous Map-Matching Architecture

## 1. Executive Summary

In urban canyons, multi-level interchanges, parallel frontage roads, and complex intersections, forcing an estimated navigation position onto a single road segment prematurely introduces false map-snapping artifacts. When two or more candidate roads are physically and kinematically plausible, premature snapping induces track oscillation, false turn alerts, and corrupted dead-reckoning corrections.

The Road Hypothesis Engine implements a multi-hypothesis tracking layer for the Navigators navigation system. Rather than deterministically snapping the trajectory to the nearest line segment, the engine discovers, tracks, evaluates, and prunes candidate road hypotheses across discrete time steps. Each candidate road hypothesis maintains an explicit mathematical representation of geometric, heading, velocity, and connectivity compatibility.

Candidate hypotheses are preserved during ambiguous conditions. Once accumulated evidence clearly favors a single road, the engine converges. When no candidate road is physically plausible, the engine refuses to force map matching, returning an unconstrained state.

---

## 2. Hypothesis Architecture and Data Model

The hypothesis tracking layer operates between the navigation state estimator (Extended Kalman Filter) and downstream guidance consumers.

```
Navigation State [p_k, v_k, psi_k, Sigma_k]
                  |
                  v
       Road Network Spatial Index
                  |
                  v
         Candidate Selection
                  |
                  v
       Compatibility Evaluation:
       - L_geom (Cross-track & position uncertainty)
       - L_heading (Angular difference & one-way rules)
       - L_vel (Speed limit compatibility)
       - L_conn (Topological adjacency & reachability)
                  |
                  v
       Combined Likelihood Calculation
                  |
                  v
       Bayesian Posterior Normalization
                  |
                  v
       Ambiguity & Convergence Analysis
         /        |         \
        v         v          v
   CONVERGED  AMBIGUOUS   NO_MATCH
```

### 2.1 Candidate Hypothesis Fields

Each tracked hypothesis contains the following attributes:

- `segment_id`: Unique identifier of the candidate road segment.
- `probability`: Normalized posterior probability mass assigned to this candidate: p_i in [0, 1], where sum(p_i) = 1.0.
- `cross_track_distance_m`: Perpendicular spatial offset from estimated position to segment centerline.
- `along_track_distance_m`: Projected longitudinal coordinate along segment direction vector.
- `heading_difference_rad`: Angular discrepancy between navigation heading and segment bearing.
- `snapped_point`: 2D Cartesian coordinate of the nearest projection point on the segment.
- `geometric_compatibility`: Gaussian distance likelihood score in [0, 1].
- `heading_compatibility`: Angular alignment score in [0, 1].
- `velocity_compatibility`: Kinematic speed compatibility score in [0, 1].
- `connectivity_compatibility`: Topological transition prior score in [0, 1].
- `combined_likelihood`: Product of component likelihoods representing joint measurement support.
- `speed_limit`: Segment operational speed limit in km/h.
- `one_way`: Boolean flag indicating if directional traffic rules apply.
- `name`: Human-readable identifier or road designation.
- `age_steps`: Cumulative number of discrete filter cycles this hypothesis has survived.

---

## 3. Mathematical Formulation

The engine adheres strictly to established probability and likelihood formulations. Scores represent calibrated likelihood functions derived from navigation uncertainties rather than heuristic percentages.

### 3.1 Geometric Compatibility (L_geom)

The geometric observation likelihood evaluates the spatial proximity between the estimated position and candidate segment centerline:

$$\sigma_{\text{geom}} = \sqrt{\sigma_{\text{pos}}^2 + \sigma_{\text{map}}^2}$$

$$L_{\text{geom}} = \exp\left( -0.5 \left( \frac{d_{\perp}}{\sigma_{\text{geom}}} \right)^2 \right)$$

Where:
- $d_{\perp}$ is the perpendicular cross-track distance in meters.
- $\sigma_{\text{pos}}$ is the 1-sigma positional uncertainty propagated from the EKF error covariance matrix.
- $\sigma_{\text{map}}$ is the intrinsic map discretization error (defaulting to 2.0 m).

When positional uncertainty expands (for example during GNSS denial or tunnel traversal), $\sigma_{\text{geom}}$ expands proportionally, preventing premature rejection of neighboring parallel roads.

### 3.2 Heading Compatibility (L_heading)

The heading likelihood evaluates the directional alignment between vehicle course and road orientation:

$$\Delta\psi = \min\left( |\psi_{\text{veh}} - \psi_{\text{seg}}|, 2\pi - |\psi_{\text{veh}} - \psi_{\text{seg}}| \right)$$

For two-way segments, the opposite traversal direction is equally valid:

$$\Delta\psi_{\text{undirected}} = \min\left( \Delta\psi, |\pi - \Delta\psi| \right)$$

The compatibility score is evaluated using a circular Gaussian distribution:

$$\sigma_{\psi} = \max(\sigma_{\psi,0}, \sigma_{\text{nav},\psi})$$

$$L_{\text{heading}} = \exp\left( -0.5 \left( \frac{\Delta\psi}{\sigma_{\psi}} \right)^2 \right)$$

At vehicle speeds below 0.8 m/s (stationary or creep motion where IMU or GNSS heading exhibits high noise), heading compatibility is set to 1.0 to avoid penalizing candidates on noise.

### 3.3 Velocity Compatibility (L_vel)

The velocity score verifies whether the observed motion speed is physically plausible for the candidate road class:

$$v_{\text{max}} = \frac{v_{\text{limit}}}{3.6} + v_{\text{tolerance}}$$

If $v_{\text{meas}} \le v_{\text{max}}$:

$$L_{\text{vel}} = 1.0$$

If $v_{\text{meas}} > v_{\text{max}}$:

$$L_{\text{vel}} = \exp\left( -0.5 \left( \frac{v_{\text{meas}} - v_{\text{max}}}{\sigma_{\text{vel}}} \right)^2 \right)$$

This metric distinguishes high-speed expressways from parallel service roads or frontage paths having low speed limits.

### 3.4 Connectivity Compatibility (L_conn)

Connectivity evaluates the topological transition probability from prior active hypotheses $H_{j}(t-1)$ to current candidate $S_i(t)$:

$$L_{\text{conn},i} = \sum_{j} T_{ji} \cdot p_j(t-1)$$

Where the transition matrix $T_{ji}$ is determined by road topology:
- $T_{ji} = 0.95$ when $S_i = S_j$ (same segment continuation).
- $T_{ji} = 0.85$ when $S_i$ shares a graph vertex with $S_j$ (valid topological junction or ramp merge).
- $T_{ji} = 0.40$ when $S_i$ is within reachable kinematic travel budget $v \cdot \Delta t + 3\sigma_{\text{pos}}$ of $S_j$.
- $T_{ji} = 0.05$ when $S_i$ is topologically disconnected from $S_j$.

When no prior hypotheses exist (engine initialization or recovery from outage), $L_{\text{conn}} = 1.0$.

### 3.5 Combined Likelihood and Posterior Normalization

The joint measurement likelihood for hypothesis $i$ is:

$$W_i = L_{\text{geom},i} \cdot L_{\text{heading},i} \cdot L_{\text{vel},i} \cdot L_{\text{conn},i}$$

Candidates with negligible likelihood ($L_{\text{geom}} < 10^{-4}$ or $W_i < 10^{-7}$) are pruned prior to normalization.

The posterior probability mass distribution across all surviving candidates is normalized:

$$p_i = \frac{W_i}{\sum_{k=1}^K W_k}$$

Candidates whose posterior probability falls below the pruning threshold ($p_i < 0.03$) are eliminated, and surviving candidates are re-normalized so that $\sum p_i = 1.0$.

---

## 4. Ambiguity, Convergence, and No-Match Logic

The engine categorizes the operating environment into three mutually exclusive ambiguity states:

### 4.1 CONVERGED State
A single road hypothesis satisfies both absolute and relative dominance criteria:

1. Top candidate probability: $p_1 \ge 0.70$
2. Dominance margin over runner-up: $p_1 - p_2 \ge 0.30$

Under CONVERGED state:
- `get_best_hypothesis()` returns the top hypothesis.
- Downstream map-matching returns a valid snap to this segment.
- Snapped position and road metadata are published.

### 4.2 AMBIGUOUS State
Multiple candidate segments exhibit physical plausibility:

1. At least two hypotheses remain active.
2. Runner-up probability: $p_2 \ge 0.20$ or margin $(p_1 - p_2) < 0.30$.

Under AMBIGUOUS state:
- `get_best_hypothesis()` returns `None`.
- Downstream map-matching yields `matched_segment = None` and `confidence = 0.0`.
- The engine preserves all active candidates and continues tracking their joint probabilities without forcing a premature snap.

### 4.3 NO_MATCH State
No candidate segments lie within the adaptive spatial search radius, or all candidates fail minimum geometric compatibility.

Under NO_MATCH state:
- Active hypotheses list is empty.
- Downstream map-matching yields `matched_segment = None`.
- The navigation filter continues unconstrained inertial/dead-reckoning propagation without map snapping.

---

## 5. Map Matcher Integration

The hypothesis tracker implements `IRoadHypothesisTracker` and integrates directly into the existing `create_map_matcher` factory:

```python
matcher = create_map_matcher("hypothesis", road_network)
result = matcher.match(position, heading=heading, speed=speed, position_uncertainty=uncertainty)
```

If the hypothesis state is CONVERGED, `result.matched_segment` contains the dominant road segment and `result.confidence` reflects its posterior probability. If the state is AMBIGUOUS or NO_MATCH, `result.matched_segment` is `None` with `confidence = 0.0`, safeguarding downstream path computation from false snaps.

---

## 6. Verification and Replay Test Suite

All 8 required replay test scenarios plus factory interface integration have been implemented and verified in [tests/test_road_hypothesis.py](file:///Users/shauryatiwari/Navigators-SIH/tests/test_road_hypothesis.py):

| Test Case | Scenario Description | Tested Behavior | Status |
|:---|:---|:---|:---:|
| `test_replay_normal_single_road` | Unambiguous single corridor | Converges to highway segment (p > 0.95, state CONVERGED) | PASSED |
| `test_replay_parallel_roads` | Parallel roads spaced 12 m apart | Preserves both candidates (p1 ~ p2 ~ 0.50) while in ambiguous zone; converges to main road upon offset reduction | PASSED |
| `test_replay_service_road_versus_highway` | 100 km/h highway vs 40 km/h service road | Vehicle at 72 km/h evaluates L_vel = 1.0 on highway vs L_vel < 0.25 on service road; converges to highway (p > 0.85) | PASSED |
| `test_replay_intersection` | Perpendicular crossing (NS vs EW) | Heading alignment isolates NS road (L_head > 0.90) and rejects EW road (L_head < 0.01) | PASSED |
| `test_replay_road_merge` | Two ramps merging into single trunk | Converges to Ramp A during approach; detects ambiguity at junction node; converges to Trunk (p = 1.0, L_conn = 0.85) | PASSED |
| `test_replay_road_split` | Trunk corridor bifurcating into left/right forks | Tracks trunk approach; expands to multi-hypothesis at fork; converges to right fork following vehicle maneuver | PASSED |
| `test_replay_temporary_map_ambiguity` | Vehicle traverses midpoint between parallel roads | Sustains AMBIGUOUS state for 8+ cycles without false snap; converges when vehicle maneuvers toward parallel road | PASSED |
| `test_replay_gnss_outage_during_ambiguity` | Position uncertainty expands from 2 m to 10 m | Transitions from CONVERGED to AMBIGUOUS as uncertainty expands; recovers cleanly to CONVERGED upon fix restoration | PASSED |
| `test_matcher_interface_integration` | Compatibility with `IMapMatcher` interface | Returns `matched_segment = None` during ambiguity; returns valid snapped segment upon convergence | PASSED |

Total regression test suite execution:
- 336 Python tests passed (0 failures)
- 14 Node.js offline navigation tests passed (0 failures)
