# Map Matching, Centerline Snapping, and Road Hypothesis Engine

```
Modules: src/navigation/map_matching.py, src/navigation/map_constraint.py, src/navigation/road_hypothesis.py
Data Source: OpenStreetMap Vector Road Geometry (Polyline Segments)
```

## Overview

Topological Map Matching projects unconstrained inertial dead-reckoning position estimates onto known vector road geometry.

During extended GNSS outages, dead-reckoning error tends to drift laterally across parallel streets. By projecting position vectors onto candidate polylines and evaluating heading alignment, map matching bounds lateral cross-track error to $< 0.60\text{ m}$.

---

## Spatial Indexing and Segment Projection

Road networks are stored as polyline segments in a spatial index:

1. **Perpendicular Distance Calculation**: For an estimated ENU position $\mathbf{p} = [x, y]^T$ and line segment $[A, B]$, orthogonal projection point $\mathbf{p}_{proj}$ is computed:

$$\mathbf{p}_{proj} = A + u (B - A), \quad u = \text{clamp}\left( \frac{(\mathbf{p} - A) \cdot (B - A)}{\|B - A\|^2}, 0, 1 \right)$$

2. **Cross-Track Error**: Perpendicular offset distance $d_{perp} = \|\mathbf{p} - \mathbf{p}_{proj}\|$.
3. **Heading Misalignment**: Heading difference $\Delta\psi = |\psi_{vehicle} - \psi_{segment}|$.

---

## Multi-Road Hypothesis Engine

When navigating through complex road geometry (parallel service roads, highway splits, or dense urban grid intersections):

- **Hypothesis Candidates**: Maintains multiple road candidate hypotheses $H_i = (\text{Segment}_i, w_i)$ weighted by spatial distance, heading alignment, speed limit consistency, and topological connectivity.
- **Ambiguity Detection**: If top two candidate weights $w_1, w_2$ satisfy $|w_1 - w_2| < \epsilon$, the system enters `AMBIGUOUS` state, broadening spatial search bounds until trajectory disambiguates.
