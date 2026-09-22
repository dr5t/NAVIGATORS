# Navigators — Map Matching & Cartographic Snapping

```
Document Identifier: MAP-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Role in the Dead-Reckoning Pipeline

In pure inertial dead reckoning, position error grows unbounded over time because sensor bias errors undergo double-integration. While Non-Holonomic Constraints (NHC) bound lateral and vertical drift relative to the vehicle heading, any angular heading error $\Delta\psi$ causes the integrated trajectory to diverge angularly from the physical road.

Map matching acts as an orthogonal bounding constraint: by leveraging prior topological knowledge that passenger vehicles travel along predefined road network centerlines, the map matcher eliminates transverse cross-track error and constrains the vehicle trajectory to real-world infrastructure.

```
       [EKF Estimated Position & Heading (p_E, p_N, psi)]
                              │
                              ▼
        [Spatial Radius Search in Local Road Graph (R <= 30m)]
                              │
                              ▼
         [Heading Compatibility Gate (|psi - theta_seg| <= 45 deg)]
                              │
                              ▼
          [Orthogonal Projection onto Candidate Segments]
                              │
                              ▼
           [Scoring Function: Distance + Heading Penalty]
                              │
                              ▼
              [Best Match Selection & Point Snapping]
                              │
                              ▼
             [Snapped Position Rendered on Leaflet Map]
```

---

## 2. Road Network Data Model & Local Representation

The road network is represented as an offline vector graph stored in GeoJSON format (`simulator/data/road_network.json`):
- **Nodes ($V$)**: Intersections, road dead-ends, and topological inflection points represented by WGS-84 coordinates $(\text{lat}, \text{lon})$ and projected ENU Cartesian coordinates $(x, y)$.
- **Edges / Ways ($E$)**: Directed or bidirectional road links connecting nodes. Each link contains:
  - `id`: Unique OpenStreetMap way identifier.
  - `name`: Human-readable street name (e.g., "Airport Expressway").
  - `highway`: Functional road classification (`motorway`, `primary`, `secondary`, `residential`).
  - `coordinates`: Polyline coordinate array defining the road centerline.
  - `one_way`: Boolean flag enforcing traffic directionality.
  - `speed_limit`: Nominal travel speed in $\text{km/h}$.

---

## 3. Projection & Candidate Scoring Mathematics

### 3.1 Candidate Link Retrieval
Given an unconstrained filter estimate $\mathbf{p} = [p_E, p_N]^T$, the algorithm queries road polylines having at least one vertex within Euclidean search radius $R_{max} = 30.0\text{ m}$.

### 3.2 Orthogonal Segment Projection
For each line segment connecting vertex $\mathbf{a} = [a_E, a_N]^T$ to $\mathbf{b} = [b_E, b_N]^T$:
1. Segment direction vector: $\mathbf{v}_{seg} = \mathbf{b} - \mathbf{a}$.
2. Vector to vehicle estimate: $\mathbf{v}_{veh} = \mathbf{p} - \mathbf{a}$.
3. Normalized projection parameter $t$:
   $$t = \frac{\mathbf{v}_{veh} \cdot \mathbf{v}_{seg}}{\|\mathbf{v}_{seg}\|^2}$$
4. Clamping to finite segment bounds ($t \in [0, 1]$):
   $$t^* = \max(0.0, \, \min(1.0, \, t))$$
5. Orthogonal projection point:
   $$\mathbf{p}_{proj} = \mathbf{a} + t^* \mathbf{v}_{seg}$$
6. Perpendicular cross-track distance:
   $$d_\perp = \|\mathbf{p} - \mathbf{p}_{proj}\|$$

### 3.3 Heading Alignment Gating & Scoring
Let $\theta_{seg} = \text{atan2}(b_N - a_N, \, b_E - a_E)$ denote the azimuth of the road segment, and $\psi$ denote vehicle heading from the EKF.
1. Heading disparity angle:
   $$\Delta\theta = \min(|\psi - \theta_{seg}|, \, 360^\circ - |\psi - \theta_{seg}|)$$
2. For bidirectional roads, reverse heading is also tested:
   $$\Delta\theta_{eff} = \min(\Delta\theta, \, |180^\circ - \Delta\theta|)$$
3. Gating Criterion: Any segment with $\Delta\theta_{eff} > 45.0^\circ$ is rejected from candidacy.
4. Composite Cost Metric:
   $$J_i = d_\perp + w_\theta \cdot \left(\frac{\Delta\theta_{eff}}{45^\circ}\right)$$
   where $w_\theta = 10.0\text{ m}$ penalizes angular misalignment.

---

## 4. Centerline Snapping & Continuity

1. **Thresholded Snapping**: The segment with minimum cost $J^* = \min(J_i)$ is selected. If $J^* \le 25.0\text{ m}$, the vehicle position displayed on the map is snapped to $\mathbf{p}_{proj}$.
2. **Hysteresis & Link Continuity**: When transitioning between adjacent segments, the matcher favors remaining on the active road segment unless cross-track distance exceeds $15\text{ m}$, preventing erratic oscillating switches at complex multi-lane junctions.
3. **Off-Road Detection**: If all candidates exceed the rejection threshold $J_{max} = 30\text{ m}$ (e.g., driving through an unmapped parking lot or off-road path), map matching disengages gracefully and displays raw EKF dead-reckoning coordinates.

---

Developed by Navigators
