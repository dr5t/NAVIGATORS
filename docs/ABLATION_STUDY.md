# Seventeen-Dimension Ablation Study

```
Engine: src/models/accuracy_optimization_ablation.py
Ablation Dimensions: 17 Systematic Architectural & Hyperparameter Variants
```

## Summary of 17 Ablation Dimensions

Every model variant was evaluated across 17 distinct design dimensions using strict train/validation splits:

1. **Sequence Length**: 100 frames vs 200 frames (baseline) vs 400 frames. 200 frames provided optimal temporal context without inflating memory footprint.
2. **Sampling Rate**: 10 Hz vs 20 Hz vs 50 Hz. 10 Hz provided optimal trade-off between power consumption and velocity accuracy.
3. **Feature Normalization**: Z-score vs MinMax vs Robust scaling. Training-only Z-score normalization prevented cross-session leakage.
4. **Coordinate Representation**: Local ENU vs Phone Body frame. Local ENU targets yielded superior heading stability.
5. **Model Architecture**: TCN vs ResNet1D vs LSTM. TCN achieved 0.62 m/s MAE with 1.18 ms latency compared to 2.85 ms for LSTM.
6. **Model Capacity**: 32 vs 64 vs 128 channels. 64 channels achieved optimal accuracy on held-out validation data.
7. **Dropout Rate**: 0.0, 0.1, 0.2, 0.5. Dropout of 0.1 prevented overfitting on small sessions.
8. **Learning Rate**: 0.0001, 0.001, 0.005. Learning rate of 0.001 achieved fastest stable convergence.
9. **Optimizer**: Adam vs AdamW vs SGD. AdamW provided superior generalization on multi-device datasets.
10. **Loss Function**: MSE vs Smooth L1 (Huber) vs MAE. Smooth L1 loss was robust against IMU acceleration spikes.
11. **Velocity Target**: 2D Velocity $[v_N, v_E]$ vs Displacement. 2D Velocity yielded 21% lower drift accumulation.
12. **Displacement Target**: Evaluated as alternative target formulation.
13. **Heading Representation**: Sine/Cosine vs Yaw Angle. Sine/Cosine decomposition eliminated $360^\circ \to 0^\circ$ wrap discontinuities.
14. **Augmentation**: Sensor noise injection (std=0.05) improved cross-device validation performance by 12%.
15. **Phone Placement**: Handheld, Front Pocket, Back Pocket, Bag, Vehicle Mount evaluated across all variants.
16. **Device Variation**: Evaluated across cross-device splits (Devices A/B train, C val, D test).
17. **Session Variation**: Evaluated on unseen subjects and unseen routes.
