# Navigators GNSS Processing, Trust Engine, and Anomaly Detection

```
Engine Component: GNSS Signal Integrity & Processing Suite
States: TRUSTED, DEGRADED, SUSPICIOUS, UNUSABLE
```

## GNSS Availability vs. GNSS Trustworthiness

A fundamental principle of Navigators is the strict separation between signal availability and signal trustworthiness:

- **GNSS Available**: Satellite signals are received by smartphone hardware and raw location callbacks are emitted.
- **GNSS Trustworthy**: Satellite fix integrity has been verified against physical motion constraints, inertial propagation, AI model predictions, and topological map geometry.

Raw GNSS availability alone does NOT guarantee signal integrity. Satellite reflections in urban canyons (multipath) or sudden satellite drops can produce corrupt position fixes while claiming low dilution of precision.

---

## GNSS Trust State Machine

The system continuously evaluates incoming satellite fixes and classifies GNSS integrity into four discrete operational states:

1. **`TRUSTED`**: High accuracy, zero anomaly flags detected. GNSS fixes are directly fused into the EKF measurement update block with nominal measurement variance.
2. **`DEGRADED`**: Minor noise or HDOP elevation detected. Measurement noise covariance matrix $R$ is scaled upward to weight inertial propagation and learned velocity models higher.
3. **`SUSPICIOUS`**: Anomaly detected (e.g. speed jump, map contradiction). Direct position updates are temporarily gated while the navigation engine enters hybrid filtering mode.
4. **`UNUSABLE`**: Severe anomaly, satellite outage, or position jump ($> 30\text{ m}$). Direct GNSS updates are rejected entirely and the system transitions to `DR` (Dead Reckoning) mode.

---

## Six-Axis Anomaly Detection Pipeline

Navigators executes six concurrent anomaly detectors on every incoming GNSS fix:

1. **Position Jump Detection**: Flags instantaneous horizontal jumps exceeding $30.0\text{ m}$.
2. **Velocity Inconsistency Detection**: Flags speed discrepancies exceeding $3.0\text{ m/s}$ between GNSS speed and integrated IMU velocity.
3. **Heading Inconsistency Detection**: Flags directional disagreements exceeding $30.0^\circ$ between GNSS course over ground and gyro/magnetometer heading.
4. **GNSS vs. AI Model Disagreement**: Flags velocity discrepancies exceeding $3.5\text{ m/s}$ between GNSS speed and deep TCN velocity predictions.
5. **GNSS vs. Map Disagreement**: Flags fixes located further than $25.0\text{ m}$ off the nearest map segment or heading mismatch exceeding $45.0^\circ$.
6. **Inertial Innovation Disagreement**: Evaluates Extended Kalman Filter innovation vector residual magnitude against chi-squared threshold bounds.

---

## Validated Anti-Jump GNSS Reacquisition

When returning from a GNSS outage or `UNUSABLE` state, Navigators strictly avoids blind snapping to initial satellite fixes.

- **Multi-Fix Verification**: The system requires $N = 3$ consecutive valid GNSS updates consistent with dead-reckoning trajectory estimates, current map geometry, and covariance bounds before transitioning from `REACQUIRING` to `RECOVERED` state.
- **Bounded Corrections**: During the `REACQUIRING` window, position adjustments are capped to prevent sudden visual jumps on client user interfaces.
