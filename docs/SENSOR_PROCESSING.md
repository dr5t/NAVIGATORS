# Navigators :  Sensor Processing & Signal Conditioning

```
Document Identifier: SENS-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Physical Sensor Characteristics & Imperfections

Consumer smartphone inertial sensors are based on low-cost Micro-Electro-Mechanical Systems (MEMS) fabricated using silicon etching processes. Compared to navigation-grade or tactical-grade IMUs, smartphone MEMS sensors exhibit severe stochastic and systematic errors:

| Sensor Attribute | Typical Smartphone MEMS Value | Tactical Grade Benchmark | Impact on Inertial Navigation |
| :--- | :--- | :--- | :--- |
| **Accelerometer Bias Instability** | $10\text{--}50\text{ mg}$ ($0.1\text{--}0.5\text{ m/s}^2$) | $< 10\text{ }\mu\text{g}$ | Generates quadratic position drift: $\Delta p = \frac{1}{2} b_a t^2$. |
| **Gyroscope Drift Rate** | $10\text{--}50^\circ/\text{hr}$ (up to $0.1^\circ/\text{s}$ short-term) | $< 0.01^\circ/\text{hr}$ | Induces heading error $\Delta\psi$, which misdirects acceleration integration. |
| **Sampling Clock Jitter** | Up to $\pm 25\%$ variation between callbacks | $< 1\text{ ppm}$ | Distorts numerical integration if assumed uniform. |
| **Engine / Road Vibrations** | High-energy noise spikes ($10\text{--}80\text{ Hz}$) | Isolated by dampeners | Saturates integrators; confuses ZUPT detectors. |
| **Thermal Bias Sensitivity** | Strong thermal gradient during CPU load | Temperature compensated | Bias shifts dynamically as phone heats up under AI compute. |

---

## 2. Ingestion & Temporal Synchronization

In the browser runtime (`simulator/app.js`), the `DeviceMotionEvent` listener registers acceleration with gravity and rotation rates. Because modern mobile operating systems throttle and bundle sensor callbacks to conserve battery, raw events arrive with variable $\Delta t$.

### 2.1 FIFO Time-Buffer & Resampling
The sensor adapter accumulates incoming measurements into an interpolation ring buffer:
- **Timestamp Tagging**: Each event records `performance.now()` high-resolution monotonic time alongside `Date.now()` epoch time.
- **Strict Monotonicity**: Replay and edge tools explicitly verify $t_k > t_{k-1}$. Any time step reversal or freeze is flagged as an invalid sequence.
- **Sensor Rate Monitoring**: `device_profiler.js` monitors actual received callback frequency against expected $10\text{ Hz}$. If sample rate drops below $5\text{ Hz}$ or exceeds $100\text{ Hz}$, the engine warns of operating system scheduling interference.

---

## 3. Real-Time Denoising Pipeline

To eliminate high-frequency acoustic and chassis resonance without future-time leakage, Navigators implements two causal filtering stages in `src/evaluation/preprocessing.py` and `simulator/engine/preprocessing.js`:

```
[Raw Smartphone Sensor Sample]
               │
               ▼
   [5-Point Running Median Filter]  ──> Removes Spikes & Sensor Glitches
               │
               ▼
 [1st-Order Causal Low-Pass Filter] ──> Suppresses Vehicle Chassis Vibrations
               │
               ▼
[Triad Phone-to-Vehicle Alignment] ──> Rotates Accelerometer/Gyro into Vehicle Axes
               │
               ▼
 [Dynamic Gravity Decomposition]   ──> Isolates Pure Vehicle Acceleration
```

### 3.1 Five-Point Running Median Filter
A causal sliding window of 5 past samples:
$$\bar{x}_k = \text{median}(x_{k-4}, x_{k-3}, x_{k-2}, x_{k-1}, x_k)$$
This non-linear filter strips isolated electrical outliers and sensor dropped-frame glitches without blurring physical acceleration transients.

### 3.2 First-Order Causal Low-Pass Filter ($f_c = 20\text{ Hz}$)
A causal single-pole filter implemented via continuous-to-discrete bilinear mapping:
$$y_k = \alpha x_k + (1 - \alpha) y_{k-1}$$
$$\alpha = \frac{2\pi f_c \Delta t}{1 + 2\pi f_c \Delta t}$$
With $f_c = 20\text{ Hz}$ and $\Delta t = 0.1\text{ s}$, $\alpha \approx 0.926$. This effectively attenuates engine block vibrations ($> 30\text{ Hz}$) while preserving vehicle acceleration maneuvers ($0\text{--}3\text{ Hz}$).

---

## 4. Stationary Detection & Dynamic Gravity Separation

Accurate dead reckoning requires separating the massive $9.81\text{ m/s}^2$ gravity acceleration vector from true vehicle acceleration. Any $1^\circ$ error in gravity orientation leaks $9.81 \times \sin(1^\circ) \approx 0.17\text{ m/s}^2$ of false horizontal acceleration, causing $300\text{ m}$ of drift within 60 seconds!

### 4.1 ZUPT Stationary Detector
The engine monitors dynamic signal energy over a sliding 1.0-second window ($N=10$ samples):
$$\sigma_a^2 = \frac{1}{N}\sum_{i=0}^{N-1} \|\mathbf{a}_i - \bar{\mathbf{a}}\|^2$$
$$\|\bar{\boldsymbol{\omega}}\| = \left\| \frac{1}{N}\sum_{i=0}^{N-1} \boldsymbol{\omega}_i \right\|$$
Stationary state is asserted when:
$$\sigma_a^2 < 0.05\text{ m}^2/\text{s}^4 \quad \text{AND} \quad \|\bar{\boldsymbol{\omega}}\| < 0.05\text{ rad/s}$$

### 4.2 Dynamic Gravity Removal
During motion, estimated attitude Euler angles $[\phi, \theta, \psi]$ from the 15-state EKF define the instantaneous gravity vector in the vehicle frame:
$$\mathbf{g}^v = \mathbf{R}_n^v [0, 0, -9.81]^T = \begin{bmatrix} 9.81 \sin\theta \\ -9.81 \cos\theta\sin\phi \\ -9.81 \cos\theta\cos\phi \end{bmatrix}$$
The net linear acceleration driving vehicle velocity integration is:
$$\mathbf{a}_{net}^v = \mathbf{f}^v - \mathbf{g}^v - \mathbf{b}_a$$

---

Developed by Navigators
