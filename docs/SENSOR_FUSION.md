# Adaptive Sensor Fusion and Error-State Extended Kalman Filter

```
Module: src/navigation/ekf.py, src/navigation/adaptive_fusion.py, src/navigation/multimodal_fusion.py
State Dimension: 15-State Error Formulated EKF
```

## 15-State Error-State EKF Formulation

Navigators tracks state uncertainty using a 15-dimensional error-state vector $\delta\mathbf{x}$:

$$\delta\mathbf{x} = \begin{bmatrix} \delta\mathbf{p}_{3\times 1} & \delta\mathbf{v}_{3\times 1} & \delta\boldsymbol{\theta}_{3\times 1} & \mathbf{b}_{a, 3\times 1} & \mathbf{b}_{g, 3\times 1} \end{bmatrix}^T$$

where:
- $\delta\mathbf{p}$: 3D position error $[E, N, U]$ in meters.
- $\delta\mathbf{v}$: 3D velocity error $[v_E, v_N, v_U]$ in m/s.
- $\delta\boldsymbol{\theta}$: 3D attitude orientation error $[\phi, \theta, \psi]$ in radians.
- $\mathbf{b}_a$: 3D accelerometer bias in $\text{m/s}^2$.
- $\mathbf{b}_g$: 3D gyroscope bias in $\text{rad/s}$.

---

## State Propagation and Covariance Update

During time steps where only IMU measurements are received, error state and covariance matrix $P_{15\times 15}$ propagate according to linear error dynamics $F(t)$ and process noise $Q$:

$$\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}, \mathbf{u}_k)$$

$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

The state transition matrix $F_k$ encodes gravity coupling, velocity kinematics, attitude error dynamics, and bias random walks.

---

## Measurement Updates and Adaptive Covariance Scaling

When pseudo-measurements (learned TCN velocity, NHC constraints, ZUPT, or GNSS fixes) arrive, measurement updates follow:

$$\mathbf{y}_k = \mathbf{z}_k - h(\mathbf{x}_{k|k-1})$$

$$S_k = H_k P_{k|k-1} H_k^T + R_k$$

$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k \mathbf{y}_k$$

$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

### Adaptive Covariance Scaling ($R_k$)
The measurement covariance matrix $R_k$ is scaled dynamically based on:
1. **Classifier Confidence**: Lower confidence scales $R_k$ up to reduce model weight.
2. **GNSS HDOP / Dilution of Precision**: Inflated satellite uncertainty scales $R_{gnss}$ up.
3. **Innovation Gating**: If innovation residual $\mathbf{y}_k^T S_k^{-1} \mathbf{y}_k > \gamma$, the measurement update is rejected to protect EKF stability.
