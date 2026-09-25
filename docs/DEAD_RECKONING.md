# Navigators Inertial Dead Reckoning Engine

```
Module: src/navigation/dead_reckoning.py
Core Components: 3D Strapdown Kinematics, ZUPT, Drift Estimation
```

## Overview

Dead Reckoning (DR) is the continuous estimation of current position by integrating past known position, heading, and velocity over time when absolute positioning signals (GNSS) are unavailable.

Navigators implements a 3D strapdown inertial dead-reckoning engine specifically tailored to consumer smartphone IMUs.

---

## Kinematic Equations of Motion

Given acceleration $\mathbf{a}_B(t)$ and angular velocity $\boldsymbol{\omega}_B(t)$ measured in the smartphone body frame $B$, position $\mathbf{p}_{ENU}$, velocity $\mathbf{v}_{ENU}$, and attitude rotation matrix $C_B^E$ (from Body to Local ENU frame) are propagated via differential equations:

$$\dot{\mathbf{p}}_{ENU}(t) = \mathbf{v}_{ENU}(t)$$

$$\dot{\mathbf{v}}_{ENU}(t) = C_B^E(t) (\mathbf{a}_B(t) - \mathbf{b}_a) - \mathbf{g}_{ENU}$$

$$\dot{C}_B^E(t) = C_B^E(t) [\boldsymbol{\omega}_B(t) - \mathbf{b}_g]_\times$$

where $[\cdot]_\times$ denotes the skew-symmetric cross-product matrix, $\mathbf{b}_a$ is accelerometer bias, $\mathbf{b}_g$ is gyroscope bias, and $\mathbf{g}_{ENU} = [0, 0, 9.81]^T \text{ m/s}^2$ is the local gravity vector.

---

## Domain-Specific Outage Modes

### 1. Vehicle Dead Reckoning Mode
During GNSS outage in vehicle mode:
- **Learned Velocity Pseudo-Measurements**: The Vehicle TCN model processes 200-sample causal IMU windows to predict 2D planar velocity $[v_N, v_E]$.
- **Non-Holonomic Kinematic Constraints**: Enforces zero lateral and vertical velocity in vehicle body coordinates ($v_y \approx 0, v_z \approx 0$).
- **Zero Velocity Updates (ZUPT)**: Automatically detects stationary dwell states (when acceleration variance and gyro magnitude fall below ZUPT thresholds) to reset velocity drift to zero.

### 2. Pedestrian Dead Reckoning Mode
During GNSS outage in pedestrian mode:
- **Learned Step Increment & Velocity**: The Pedestrian TCN model processes causal IMU windows to predict 2D velocity or step displacement increments $[\Delta N, \Delta E]$.
- **Orientation Decoupling**: Decouples phone body orientation from human walking direction using dynamic triad alignment.
- **Foot-Flat ZUPT**: Applies zero-velocity updates during foot-flat phase detection.
