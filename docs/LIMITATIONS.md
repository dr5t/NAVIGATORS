# Navigators — System Limitations & Scientific Disclosures

```
Document Identifier: LIM-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Scientific Integrity Statement

To prevent misleading claims and maintain the highest standard of technical honesty, this document explicitly details the physical, algorithmic, computational, and cartographic boundaries of the Navigators platform.

---

## 2. Hardware & Empirical Validation Limitations

1. **Unvalidated Physical In-Vehicle Road Trials**: While the software implementation, mathematical formulation, and automated test suite (288/288 tests) are fully verified against benchmark datasets and synthetic traces, **physical field road trials with an active smartphone moving through a physical road tunnel remain unvalidated**.
2. **Thermal Sensor Drift**: Consumer smartphone MEMS sensors lack internal temperature compensation circuits. Under heavy processor load (such as concurrent WebAssembly AI inference and high-brightness screen rendering), internal device temperature rises significantly, causing non-linear accelerometer and gyroscope bias drift that may exceed standard static calibration bounds.
3. **Mount Rigidity**: The Triad leveling and heading alignment assumes the smartphone is rigidly secured to the vehicle chassis (e.g., in a firm dashboard mount). If the phone is held loosely in a driver's hand, sliding on a seat, or manipulated during driving, the coordinate transformation $\mathbf{R}_b^v$ degrades, introducing substantial velocity estimation error.

---

## 3. Algorithmic & Sensor Limitations

1. **Prolonged Outage Error Growth**: In the absence of periodic stationary stops (ZUPT) or map-matching constraints (such as on an unmapped winding rural road), pure inertial dead-reckoning position error accumulates monotonically over time. The system is designed for typical tunnel durations ($30\text{--}180\text{ s}$); multi-hour GNSS outages without external reference will exhibit noticeable spatial drift.
2. **Reverse Driving & Skid Dynamics**: Non-Holonomic Constraints (NHC) assume forward vehicle travel with zero lateral skid. If a vehicle undergoes severe hydroplaning, ice skidding, or extended reverse travel, the NHC measurement model introduces erroneous corrections into the EKF state.

---

## 4. Mobile Operating System & Browser Limitations

1. **Background Execution Throttling**: Mobile operating systems (Android and iOS) aggressively throttle or pause browser timers, WebAssembly threads, and sensor event listeners when the browser tab is minimized or the screen is powered down. Navigation must remain in the foreground with active Screen Wake Lock.
2. **Browser Sensor Sampling Jitter**: Smartphone operating systems do not guarantee deterministic hard real-time scheduling for HTML5 motion events. Minor sampling frequency jitter ($\pm 15\%$) must be absorbed by interpolation buffers.

---

## 5. Cartographic & Offline Limitations

1. **Unmapped Infrastructure**: In subterranean parking garages, private warehouse lots, or newly constructed roads absent from OpenStreetMap vector data, map matching cannot snap coordinates, falling back entirely to kinematic dead reckoning (Ablation Mode F).
2. **Local Storage Caps**: Caching large high-resolution satellite imagery or continental vector road networks exceeds standard mobile browser LocalStorage quotas ($5\text{--}50\text{ MB}$). Navigators restricts local vector caches to immediate metropolitan navigation corridors (~1.1 MB per city zone).

---

Developed by Navigators
