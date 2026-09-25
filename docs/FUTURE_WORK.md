# Navigators :  Future Work & Technical Roadmap

```
Document Identifier: FUT-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Roadmap Architecture

The Navigators development roadmap outlines targeted engineering enhancements designed to address known operational limitations and incorporate next-generation sensor fusion modalities:

```
[Phase A: Physical In-Vehicle Road Trials] ──> Calibrate MEMS noise under live highway tunnels
                   │
                   ▼
[Phase B: Smartphone Barometer Integration] ──> Vertical elevation & bridge/tunnel altitude tracking
                   │
                   ▼
[Phase C: Visual-Inertial Odometry (VIO)]   ──> Windshield camera optical flow fusion
                   │
                   ▼
[Phase D: Native Background Daemon Wrapper] ──> Background navigation persistence via Android Foreground Service
```

---

## 2. Targeted Engineering Capabilities

### 2.1 Physical In-Vehicle Road Trials (Priority 1)
- Deploy the PWA on physical Android smartphones secured in standard vehicle dashboard mounts.
- Conduct live test drives through vehicular tunnels with verified ground truth surveyed at tunnel portals.
- Quantify true empirical position drift, maximum error, and reacquisition convergence under real-world multipath transients.

### 2.2 Smartphone Barometer Fusion for Altitude Tracking
- Modern smartphones include barometric pressure sensors capable of detecting altitude changes of $< 0.5\text{ m}$.
- Integrate barometric pressure observations into the 15-state EKF to decouple vertical elevation from inertial integration drift, enabling multi-level highway interchange and stacked tunnel navigation.

### 2.3 Visual-Inertial Odometry (VIO) Auxiliary Fusion
- Leverage the smartphone's forward-facing windshield camera using WebCodecs and lightweight feature tracking (FAST/ORB).
- Provide optical flow velocity vectors to augment the TCN velocity estimator during daytime driving, bounding inertial drift during long straight highway outages.

### 2.4 Native Background Daemon Wrapper
- Package the core WebAssembly / JavaScript engine inside an Android Foreground Service using an embedded WebView or headless V8 runtime.
- Maintain continuous dead-reckoning state propagation even when the phone screen is locked or another application (e.g., incoming phone call) takes the foreground.

---

Developed by Navigators
