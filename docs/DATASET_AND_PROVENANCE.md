# Dataset Registry, Provenance, and Normalization

```
Registry Engine: src/data/dataset_registry.py
Normalization Engine: src/data/sensor_normalization.py
```

## Dataset Classification and Provenance Standard

Navigators maintains strict separation between third-party external benchmark datasets and original Navigators datasets to guarantee zero provenance ambiguity.

Every dataset in the registry is registered with verifiable metadata:
- `dataset_name`: Official identifier.
- `source_url`: Verifiable publisher or repository URL.
- `license`: Open-source license specification.
- `citation`: Required academic citation.
- `country_code`: Geographic origin.
- `sampling_rate_hz`: Ingestion sampling rate.
- `ground_truth_type`: High-precision RTK GNSS, Optical Motion Capture, or Reference INS.
- `provenance_status`: VERIFIED or QUARANTINED.

---

## 1. External Benchmark Datasets

### A. IO-VNBD (Vehicle Navigation Benchmark Dataset)
- **Source**: Smartphone In-Vehicle Dataset (Italy).
- **URL**: `https://github.com/m-pizzoli/IO-VNBD`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Modality**: 6-DOF IMU (accel, gyro), GNSS ground truth.
- **Sampling Rate**: 100 Hz IMU downsampled to 10-50 Hz.
- **Role**: Vehicle baseline model training and ablation studies.

### B. RoNIN (Robust Network for Inertial Navigation)
- **Source**: Pedestrian Inertial Dataset (Simon Fraser University).
- **URL**: `https://github.com/ronin-rr/ronin`
- **License**: MIT License.
- **Modality**: 6-DOF IMU, 3D Magnetometer, 3D Pose GT via Vicon optical capture.
- **Sampling Rate**: 200 Hz downsampled to 100 Hz / 10 Hz.
- **Role**: Pedestrian baseline model training.

### C. OxIOD (Oxford Inertial Odometry Dataset)
- **Source**: University of Oxford.
- **URL**: `https://oxiod.cs.ox.ac.uk/`
- **License**: Creative Commons Attribution 4.0 International.
- **Modality**: Handheld, pocket, trolley, and bag walking IMU sequences.
- **Sampling Rate**: 100 Hz.
- **Role**: Multi-placement pedestrian model training.

---

## 2. Original Navigators Datasets

### A. Navigators India Vehicle Dataset
- **Geography**: Urban, rural, highway, and hilly terrain across India.
- **Vehicle Types**: Auto-rickshaws, compact hatchbacks, sedans, SUVs, motorcycles.
- **Modality**: 6-DOF IMU, 3D Magnetometer, Reference GNSS speed/heading.
- **Role**: Indian domain adaptation and generalization evaluation.

### B. Navigators India Pedestrian Dataset
- **Geography**: Crowded markets, outdoor footpaths, indoor corridors in India.
- **Placements**: Handheld, front pocket, back pocket, shoulder bag.
- **Modality**: 6-DOF IMU, 3D Magnetometer, Reference RTK GNSS.
- **Role**: Pedestrian domain fine-tuning and benchmark evaluation.

---

## 3. Quarantined Datasets

- **I2WDD**: Quarantined. Although it contains Indian vehicle driving video/sensor data, its primary focus is driving behavior classification rather than continuous high-precision inertial odometry. It is preserved in registry quarantine without inclusion in default model training runs.

---

## 4. Cross-Dataset Normalization Rules

1. **Zero Data Leakage**: Normalization statistics (mean, std, min, max) are calculated strictly on **TRAIN** splits and applied to validation and test splits without leaking future data.
2. **Unit Conversions**: Standardized to SI units:
   - Accelerometer: $\text{m/s}^2$ (gravity constant $g = 9.80665 \text{ m/s}^2$).
   - Gyroscope: $\text{rad/s}$ (converted from degrees per second where necessary).
3. **Monotonic Timestamps**: Duplicate or out-of-order timestamps are strictly rejected during ingestion.
