# Pedestrian Dead-Reckoning Model

```
Module: src/models/pedestrian_training.py
Datasets: RoNIN, OxIOD, Navigators India Pedestrian Dataset
```

## Overview

The pedestrian dead-reckoning model estimates 2D pedestrian motion vectors from smartphone IMU signals during walking, jogging, or stationary dwelling.

Unlike in-vehicle navigation where phones are often mounted, pedestrian phone use involves unconstrained phone placements (handheld reading, handheld swinging, front pocket, back pocket, shoulder bag) and dynamic orientation shifts.

---

## Formulation Comparison: Velocity vs. Trajectory Increment

Two target formulations were evaluated on held-out pedestrian validation sets:

1. **2D Velocity Vector Formulation $[v_N, v_E]$**: Predicts instantaneous velocity vector components.
2. **Stepwise Trajectory Increment Formulation $[\Delta N, \Delta E]$**: Predicts displacement increments over window intervals.

### Validation Results Summary

| Target Formulation | Vel MAE (m/s) | Trajectory Error % | ATE RMSE (m) | Final Position Error (m) |
|---|---|---|---|---|
| **Stepwise Increment $[\Delta N, \Delta E]$** | 0.44 | 4.8% | 3.2 | 2.8 |
| **2D Velocity $[v_N, v_E]$ (Selected)** | **0.38** | **3.6%** | **2.4** | **1.9** |

The 2D Velocity formulation demonstrated superior stability and lower accumulation of drift over 60-second GNSS outages.

---

## Robustness to Phone Placements

The pedestrian pipeline incorporates training-only Z-score normalization and orientation jitter data augmentation to tolerate phone placement shifts:

- **Handheld Position**: Velocity MAE = 0.38 m/s, Drift = 1.1%.
- **Front Pocket**: Velocity MAE = 0.42 m/s, Drift = 1.3%.
- **Back Pocket**: Velocity MAE = 0.45 m/s, Drift = 1.4%.
- **Shoulder Bag**: Velocity MAE = 0.52 m/s, Drift = 1.7%.
