# Non-Holonomic Kinematic Constraints (NHC)

```
Kinematic Principle: Land Vehicle Kinematic Motion Limits
Module: src/navigation/dead_reckoning.py (NHC Constraint Block)
```

## Physical Basis of Non-Holonomic Constraints

Wheeled land vehicles operate under non-holonomic kinematic constraints. Assuming no lateral sliding or vertical launch off the road surface:

1. **Lateral Velocity Constraint**: The vehicle cannot move sideways relative to its heading vector ($v_y^V \approx 0$).
2. **Vertical Velocity Constraint**: The vehicle cannot move vertically out of the road plane ($v_z^V \approx 0$).

These constraints provide powerful synthetic measurement updates to bound error growth in the Extended Kalman Filter during extended GNSS outages.

---

## Mathematical Formulation

Let $\mathbf{v}^E = [v_N, v_E, v_U]^T$ be the velocity vector in the Local ENU frame. Transforming velocity into the vehicle body frame $V$ using heading angle $\psi$ and pitch angle $\theta$:

$$\mathbf{v}^V = C_E^V (\psi, \theta) \mathbf{v}^E$$

The non-holonomic measurement model is expressed as:

$$\mathbf{z}_{NHC} = \begin{bmatrix} v_y^V \\ v_z^V \end{bmatrix} = \begin{bmatrix} -v_N \sin\psi + v_E \cos\psi \\ -v_N \sin\theta \cos\psi - v_E \sin\theta \sin\psi + v_U \cos\theta \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} + \mathbf{v}_{NHC}$$

where $\mathbf{v}_{NHC} \sim \mathcal{N}(0, R_{NHC})$ represents constraint noise variance ($R_{NHC} = \text{diag}(\sigma_{lat}^2, \sigma_{vert}^2)$).

---

## EKF Measurement Update for NHC

The measurement matrix $H_{NHC} = \frac{\partial \mathbf{z}_{NHC}}{\partial \mathbf{x}}$ maps the 15-dimensional EKF state vector to lateral/vertical body velocity errors:

$$H_{NHC} = \begin{bmatrix}
0_{1\times 3} & -\sin\psi & \cos\psi & 0 & 0_{1\times 3} & 0_{1\times 3} & 0_{1\times 3} \\
0_{1\times 3} & -\sin\theta \cos\psi & -\sin\theta \sin\psi & \cos\theta & 0_{1\times 3} & 0_{1\times 3} & 0_{1\times 3}
\end{bmatrix}$$

Applying this update at every timestep bounds lateral cross-track drift during GNSS outages, keeping dead-reckoning trajectories strictly aligned along the vehicle longitudinal axis.
