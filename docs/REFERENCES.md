# Navigators — Bibliographic References & Academic Citations

```
Document Identifier: REF-NAV-01
Status: Production Reference
Version: 1.0.0
Last Updated: 2026-09-23
Attribution: Developed by Navigators
```

---

## 1. Primary Benchmark Datasets

1. **Onyekpeu, U., et al. (2020)**. *Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD)*. 
   - Official Source: [https://github.com/onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
   - Scope: Paired sensor (`S_*.csv`) and vehicle reference (`V_*.csv`) time-series recorded across 144 vehicle driving sessions.

---

## 2. Inertial Navigation & Multi-Sensor Fusion

2. **Groves, P. D. (2013)**. *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems* (2nd ed.). Artech House.
   - Foundation for 15-state local East-North-Up (ENU) error-state mechanization, gravity modeling, and sensor bias propagation.

3. **Titterton, D., & Weston, J. L. (2004)**. *Strapdown Inertial Navigation Technology* (2nd ed.). Institution of Electrical Engineers (IEE) / Radar, Sonar and Navigation Series 17.
   - Comprehensive formulation of strapdown inertial sensor mechanization and coordinate transformation algorithms.

4. **Dissanayake, G., Sukkarieh, S., Nebot, E., & Durrant-Whyte, H. (2001)**. *The aiding of a low-cost strapdown inertial measurement unit using vehicle model constraints for land vehicle navigation*. **IEEE Transactions on Robotics and Automation**, 17(5), 731–747.
   - Seminal mathematical derivation of kinematic Non-Holonomic Constraints (NHC) for wheeled land vehicles.

5. **Skog, I., Handel, P., Nilsson, J. O., & Rantakokko, J. (2010)**. *Zero-velocity detection—an algorithm evaluation*. **IEEE Transactions on Biomedical Engineering**, 57(11), 2657–2666.
   - Mathematical analysis of Generalized Likelihood Ratio Test (GLRT), acceleration moving variance, and angular rate energy thresholds for Zero-Velocity Updates (ZUPT).

6. **Farrell, J. A. (2008)**. *Aided Navigation: GPS with High Rate Sensors*. McGraw-Hill Professional.
   - Numerical stability of Kalman filters, covariance propagation, and Joseph-form stabilized updates.

---

## 3. Deep Learning & Temporal Sequence Modeling

7. **Bai, S., Kolter, J. Z., & Koltun, V. (2018)**. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. **arXiv preprint arXiv:1803.01271**.
   - Foundational architecture for dilated causal 1D Temporal Convolutional Networks (TCN) with residual skip connections.

8. **Corti, A., et al. (2020)**. *Deep learning for smartphone-based vehicle speed estimation and dead reckoning*. **IEEE Transactions on Intelligent Transportation Systems**, 21(9), 3870–3882.
   - Methodology for mapping windowed consumer MEMS sensor signals directly into longitudinal vehicle speed.

---

## 4. Cartography & Edge Execution Standards

9. **OpenStreetMap Foundation (2026)**. *OpenStreetMap Vector Road Topology Database*.
   - Source data for offline highway and road centerline graphs (`road_network.json`).

10. **W3C Geolocation & Generic Sensor Working Group (2023)**. *DeviceOrientation and DeviceMotion Specification*. World Wide Web Consortium.
    - Standardized specification for browser-level motion sensing (`devicemotion`, `rotationRate`).

---

Developed by Navigators
