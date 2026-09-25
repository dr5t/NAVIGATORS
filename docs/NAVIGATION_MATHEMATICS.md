# Navigators :  Navigation Mathematics & Theoretical Formulation

```
Document Identifier: MATH-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Mathematical Notation & Coordinate Frames

### 1.1 Notation Glossary
- Lowercase boldface ($\mathbf{x}, \mathbf{p}, \mathbf{v}$): Vectors in $\mathbb{R}^n$.
- Uppercase boldface ($\mathbf{R}, \mathbf{P}, \mathbf{F}, \mathbf{H}$): Matrices in $\mathbb{R}^{m \times n}$.
- Subscripts ($k, k-1$): Discrete temporal epochs.
- Superscripts ($\{b\}, \{v\}, \{n\}, \{e\}$): Coordinate frames of reference.
- $\mathbf{R}_a^b$: Direction Cosine Matrix (DCM) transforming a vector from frame $\{a\}$ to frame $\{b\}$.
- $[\mathbf{v}]_\times$: Skew-symmetric cross-product matrix for $\mathbf{v} = [v_1, v_2, v_3]^T$:
  $$[\mathbf{v}]_\times = \begin{bmatrix} 0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0 \end{bmatrix}$$

### 1.2 Coordinate Reference Frames
1. **Device Body Frame ($\{b\}$)**: Orthogonal sensor triad aligned with physical smartphone casing.
2. **Vehicle Body Frame ($\{v\}$)**: Orthogonal axes aligned with vehicle dynamics ($X_v$: transverse/right, $Y_v$: forward/longitudinal, $Z_v$: vertical/up).
3. **Local Tangent Frame ($\{n\}$ - ENU)**: Local Cartesian East-North-Up frame tangent to Earth reference ellipsoid at initial GNSS anchor $(\phi_0, \lambda_0, h_0)$.
4. **Earth Frame ($\{e\}$ - WGS-84)**: Global geocentric reference frame defined by semi-major axis $a = 6378137.0\text{ m}$ and flattening $f = 1 / 298.257223563$.

---

## 2. Phone-to-Vehicle Alignment Mathematics

Let $\mathbf{f}^b$ denote the specific force measured by the tri-axial accelerometer in the device frame, and $\boldsymbol{\omega}^b$ denote the angular velocity measured by the gyroscope.

### 2.1 Gravity Vector Extraction (Stationary Dwell)
During zero-velocity conditions prior to vehicle movement:
$$\mathbf{f}_{dwell}^b = -\mathbf{R}_n^b \mathbf{g}^n + \mathbf{b}_a + \mathbf{w}_a$$
Averaging across $N_{cal}$ dwell samples yields estimated downward gravity unit vector:
$$\mathbf{u}_{down}^b = \frac{\frac{1}{N_{cal}} \sum_{i=1}^{N_{cal}} \mathbf{f}_i^b}{\left\| \frac{1}{N_{cal}} \sum_{i=1}^{N_{cal}} \mathbf{f}_i^b \right\|}$$

### 2.2 Forward Acceleration Vector Extraction
When the vehicle accelerates forward along longitudinal axis $Y_v$:
$$\mathbf{a}_{net}^b = \mathbf{f}_{motion}^b - \|\mathbf{g}\|\mathbf{u}_{down}^b$$
Projecting orthogonal to the downward vector:
$$\mathbf{u}_{fwd}^b = \frac{\mathbf{a}_{net}^b - (\mathbf{a}_{net}^b \cdot \mathbf{u}_{down}^b)\mathbf{u}_{down}^b}{\|\mathbf{a}_{net}^b - (\mathbf{a}_{net}^b \cdot \mathbf{u}_{down}^b)\mathbf{u}_{down}^b\|}$$

### 2.3 Triad Construction of Rotation Matrix $\mathbf{R}_b^v$
$$\mathbf{r}_1 = \mathbf{u}_{fwd}^b \quad (\text{Longitudinal / Forward})$$
$$\mathbf{r}_2 = \frac{\mathbf{u}_{fwd}^b \times \mathbf{u}_{down}^b}{\|\mathbf{u}_{fwd}^b \times \mathbf{u}_{down}^b\|} \quad (\text{Transverse / Right})$$
$$\mathbf{r}_3 = \mathbf{r}_1 \times \mathbf{r}_2 \quad (\text{Vertical / Up})$$
The complete orthogonal transformation from phone body to vehicle body is:
$$\mathbf{R}_b^v = \begin{bmatrix} \mathbf{r}_2^T \\ \mathbf{r}_1^T \\ \mathbf{r}_3^T \end{bmatrix} \in SO(3)$$

---

## 3. Kinematic Navigation State Equations

### 3.1 State Vector Formulation
The continuous 15-state vector $\mathbf{x}(t)$ is formulated as:
$$\mathbf{x}(t) = \begin{bmatrix} \mathbf{p}^n(t) \\ \mathbf{v}^n(t) \\ \boldsymbol{\theta}(t) \\ \mathbf{b}_a(t) \\ \mathbf{b}_g(t) \end{bmatrix} = \begin{bmatrix} [p_E, p_N, p_U]^T \\ [v_E, v_N, v_U]^T \\ [\phi, \theta, \psi]^T \\ [b_{ax}, b_{ay}, b_{az}]^T \\ [b_{gx}, b_{gy}, b_{gz}]^T \end{bmatrix} \in \mathbb{R}^{15}$$

### 3.2 Continuous-Time Differential Equations
$$\dot{\mathbf{p}}^n(t) = \mathbf{v}^n(t)$$
$$\dot{\mathbf{v}}^n(t) = \mathbf{R}_b^n(t)(\mathbf{f}^b(t) - \mathbf{b}_a(t) - \mathbf{w}_a(t)) + \mathbf{g}^n$$
$$\dot{\boldsymbol{\theta}}(t) = \mathbf{E}(\boldsymbol{\theta}(t))(\boldsymbol{\omega}^b(t) - \mathbf{b}_g(t) - \mathbf{w}_g(t))$$
$$\dot{\mathbf{b}}_a(t) = \mathbf{w}_{ba}(t)$$
$$\dot{\mathbf{b}}_g(t) = \mathbf{w}_{bg}(t)$$

where $\mathbf{g}^n = [0, 0, -9.81]^T\text{ m/s}^2$, and $\mathbf{E}(\boldsymbol{\theta})$ maps body angular rates to Euler angle rates:
$$\mathbf{E}(\phi, \theta, \psi) = \begin{bmatrix} 1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\ 0 & \cos\phi & -\sin\phi \\ 0 & \frac{\sin\phi}{\cos\theta} & \frac{\cos\phi}{\cos\theta} \end{bmatrix}$$

The rotation matrix $\mathbf{R}_b^n$ as a function of Euler angles $(\phi, \theta, \psi)$ follows the standard $Z-Y-X$ convention:
$$\mathbf{R}_b^n = \begin{bmatrix} \cos\psi\cos\theta & \cos\psi\sin\theta\sin\phi - \sin\psi\cos\phi & \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\ \sin\psi\cos\theta & \sin\psi\sin\theta\sin\phi + \cos\psi\cos\phi & \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\ -\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi \end{bmatrix}$$

---

## 4. Discrete Extended Kalman Filter Formulation

### 4.1 State Transition Matrix $\mathbf{F}_k$
Linearizing the continuous error dynamics over timestep $\Delta t$:
$$\mathbf{F}_k = \mathbf{I}_{15} + \mathbf{A}_k \Delta t$$
where the non-zero sub-blocks of Jacobian $\mathbf{A}_k$ are:
$$\mathbf{A}_{0:3, 3:6} = \mathbf{I}_{3 \times 3}$$
$$\mathbf{A}_{3:6, 6:9} = -[\mathbf{R}_b^n \hat{\mathbf{f}}^b]_\times$$
$$\mathbf{A}_{3:6, 9:12} = -\mathbf{R}_b^n$$
$$\mathbf{A}_{6:9, 6:9} \approx \mathbf{0}_{3 \times 3} \quad (\text{small angle approximation})$$
$$\mathbf{A}_{6:9, 12:15} = -\mathbf{E}(\hat{\boldsymbol{\theta}})$$

### 4.2 Discrete Process Noise Covariance $\mathbf{Q}_k$
$$\mathbf{Q}_k = \text{diag}\left( q_{pos}\Delta t^2 \mathbf{I}_3, \, q_{vel}\Delta t \mathbf{I}_3, \, q_{ori}\Delta t \mathbf{I}_3, \, q_{ba}\Delta t \mathbf{I}_3, \, q_{bg}\Delta t \mathbf{I}_3 \right)$$
Default spectral densities in the repository:
- $q_{pos} = 0.5\text{ m}^2/\text{s}^4$
- $q_{vel} = 2.0\text{ m}^2/\text{s}^3$
- $q_{ori} = 0.05\text{ rad}^2/\text{s}$
- $q_{ba} = 10^{-3}\text{ m}^2/\text{s}^5$
- $q_{bg} = 10^{-4}\text{ rad}^2/\text{s}^3$

### 4.3 State & Covariance Time Propagation
$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{f}_k^b, \boldsymbol{\omega}_k^b, \Delta t)$$
$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$

---

## 5. Multi-Sensor Measurement Updates

When an observation $\mathbf{z}_k$ occurs with measurement model $\mathbf{h}(\mathbf{x}_k)$:
$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_{k|k-1}) \quad (\text{Innovation})$$
$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k \quad (\text{Innovation Covariance})$$
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1} \quad (\text{Kalman Gain})$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$$

### 5.1 Joseph-Form Covariance Update
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$
This symmetric formulation guarantees positive semi-definiteness regardless of rounding errors in single-precision floating point.

### 5.2 GNSS Fusion Update
When satellite reception is active:
$$\mathbf{z}_{gnss} = [p_E^{gps}, p_N^{gps}, p_U^{gps}, v_E^{gps}, v_N^{gps}, v_U^{gps}]^T \in \mathbb{R}^6$$
$$\mathbf{H}_{gnss} = \begin{bmatrix} \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} & \mathbf{0}_{3 \times 9} \\ \mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 9} \end{bmatrix} \in \mathbb{R}^{6 \times 15}$$
$$\mathbf{R}_{gnss} = \text{diag}(\sigma_{pos}^2 \mathbf{I}_3, \, \sigma_{vel}^2 \mathbf{I}_3)$$

### 5.3 AI Velocity Pseudo-Measurement Update
During GNSS denial, the TCN model outputs 2D planar velocity $[v_N^{tcn}, v_E^{tcn}]$:
$$\mathbf{z}_{ai} = [v_E^{tcn}, v_N^{tcn}]^T \in \mathbb{R}^2$$
$$\mathbf{H}_{ai} = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 & \mathbf{0}_{1 \times 9} \\ 0 & 0 & 0 & 0 & 1 & 0 & \mathbf{0}_{1 \times 9} \end{bmatrix} \in \mathbb{R}^{2 \times 15}$$
$$\mathbf{R}_{ai} = \text{diag}(\sigma_{ai}^2, \sigma_{ai}^2), \quad \sigma_{ai} = 0.3\text{ m/s}$$

### 5.4 Non-Holonomic Constraints (NHC)
Assuming land vehicle non-slip dynamics:
$$v_{lateral}^v = -v_E \sin\psi + v_N \cos\psi \approx 0$$
$$v_{vertical}^v = v_U \approx 0$$
$$\mathbf{z}_{nhc} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad \mathbf{H}_{nhc} = \begin{bmatrix} 0 & 0 & 0 & -\sin\psi & \cos\psi & 0 & \mathbf{0}_{1 \times 9} \\ 0 & 0 & 0 & 0 & 0 & 1 & \mathbf{0}_{1 \times 9} \end{bmatrix}$$
$$\mathbf{R}_{nhc} = \text{diag}(1.0, 1.0)\text{ m}^2/\text{s}^2$$

### 5.5 Zero-Velocity Updates (ZUPT)
During stationary dwell:
$$\mathbf{z}_{zupt} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{H}_{zupt} = [\mathbf{0}_{3 \times 3}, \mathbf{I}_{3 \times 3}, \mathbf{0}_{3 \times 9}]$$
$$\mathbf{R}_{zupt} = \text{diag}(0.01, 0.01, 0.01)\text{ m}^2/\text{s}^2$$

---

## 6. Geodetic Coordinate Transformations

### 6.1 WGS-84 to Local ENU
Given geodetic coordinate $(\phi, \lambda, h)$ and reference origin $(\phi_0, \lambda_0, h_0)$:
1. Difference in radians: $\Delta\phi = \phi - \phi_0, \, \Delta\lambda = \lambda - \lambda_0$
2. Meridional radius of curvature $M$ and prime vertical radius $N$:
   $$M = \frac{a(1 - e^2)}{(1 - e^2 \sin^2\phi_0)^{3/2}}, \quad N = \frac{a}{\sqrt{1 - e^2 \sin^2\phi_0}}$$
   where eccentricity squared $e^2 = 2f - f^2$.
3. Local Cartesian ENU coordinates:
   $$p_E = (N + h_0) \cos\phi_0 \cdot \Delta\lambda$$
   $$p_N = (M + h_0) \cdot \Delta\phi$$
   $$p_U = h - h_0$$

---

Developed by Navigators
