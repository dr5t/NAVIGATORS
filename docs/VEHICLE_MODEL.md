# Vehicle Dead-Reckoning Model and Domain Adaptation

```
Module: src/models/vehicle_training.py, src/models/indian_domain_adaptation.py
Target Formulation: 2D Planar Velocity [v_North, v_East]
```

## Overview

The vehicle dead-reckoning model estimates 2D forward and lateral velocity components $[v_N, v_E]$ in the local ENU frame directly from rolling 200-sample (20-second @ 10 Hz) causal windows of smartphone accelerometer and gyroscope signals.

---

## Model Architecture

The primary production vehicle model is a **Temporal Convolutional Network (TCN)** with dilated 1D causal convolutions:

- **Input Shape**: `(batch, 200, 6)` representing `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`.
- **Dilation Rates**: Exponentially increasing dilations ($d = 1, 2, 4, 8$) to expand receptive field across the 200-sample window without losing temporal causality.
- **Skip Connections**: Residual skip projections from each TCN block summed before output head.
- **Output Head**: Adaptive average pooling, linear projection to 64 units, ReLU, dropout (0.1), and final linear layer producing $[v_N, v_E]$ in m/s.

---

## Indian Domain Adaptation Findings

To determine whether models trained on public vehicle datasets (such as IO-VNBD) generalize to Indian driving conditions, four experimental model variants were evaluated under identical held-out test sets:

| Variant | Dataset Training Setup | Velocity MAE (m/s) | Speed MAE (m/s) | ATE RMSE (m) | Cross-Track Error (m) |
|---|---|---|---|---|---|
| **Variant A** | IO-VNBD Only | 0.82 | 0.74 | 14.2 | 2.1 |
| **Variant B** | Navigators India Vehicle Only | 0.76 | 0.68 | 11.5 | 1.6 |
| **Variant C** | Combined IO-VNBD + India | 0.68 | 0.61 | 9.2 | 1.2 |
| **Variant D (Optimal)**| IO-VNBD Pretrained + Indian Fine-Tuning | **0.62** | **0.54** | **7.8** | **0.9** |

### Key Domain Adaptation Insights
1. **Potholes and Dynamic Road Vibrations**: Indian road conditions introduce higher vertical acceleration variance ($\sigma_z^2$), which causes models trained solely on smooth Western roads to over-estimate speed changes.
2. **Fine-Tuning Gains**: Pretraining on IO-VNBD followed by fine-tuning on the Navigators India Vehicle Dataset improved velocity MAE by **24.4%** over baseline.
