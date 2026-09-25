# Phase 11: Navigators India Dataset Pipeline

## Architecture Overview

The Navigators India Dataset pipeline provides an auditable, research-grade data collection, validation, unit normalization, versioning, leakage-free splitting, and evaluation pipeline.

```
Raw Field Collection (.json, .jsonl, .csv)
      |
      v
Schema & Provenance Gate (Strict Third-Party Tagging)
      |
      v
Validation Engine (Timestamp Ordering, Jitter, Units, Axes, GNSS Consistency)
      |
      +---> [If Invalid] ---> Rejection Report (Quarantined Raw Session)
      |
      v [If Valid]
Lossless SI Normalization (Preserves Raw Values in Metadata)
      |
      v
Leakage-Free Multi-Level Splitter (Session / Vehicle / Device Isolation)
      |
      v
Versioned Release Archive & Audit Summary
```

## Dataset Separation and Third-Party Provenance

To ensure scientific rigor, third-party benchmark datasets (such as Oxford RobotCar or IO-VNBD) are never repackaged as original Navigators collection data. Each input file requires explicit provenance metadata:

- **Navigators India Field Data**: Tagged with `source_dataset: "Navigators India Dataset"`, `source_kind: "navigators_collection"`.
- **Third-Party Data**: Tagged with original `source_dataset` name, `source_kind: "external_benchmark"`, preserving original copyright and licensing metadata.

## Required Raw Fields and Schema Contract

The pipeline enforces presence of the following 20 core fields:

1. `timestamp` (elapsed or UTC seconds)
2. `accelerometer_x`, `accelerometer_y`, `accelerometer_z` (m/s^2 or g)
3. `gyroscope_x`, `gyroscope_y`, `gyroscope_z` (rad/s or deg/s)
4. `latitude`, `longitude` (WGS-84 degrees)
5. `GNSS_speed` (m/s or km/h)
6. `GNSS_heading` (degrees or radians)
7. `GNSS_accuracy` (meters)
8. `device_id` (string)
9. `vehicle_id` (string)
10. `session_id` (string)
11. `phone_mount_position` (dashboard, windshield, console, handheld)
12. `vehicle_type` (car, autorickshaw, two_wheeler, truck, bus)
13. `road_type` (urban_arterial, highway, rural, service_road, flyover, underpass)
14. `environment` (urban, highway, rural, hilly, underpass_tunnel)
15. `activity` (normal_gnss, degraded_gnss, gnss_outage, gnss_recovery, dense_traffic, stop_and_go, parallel_roads, service_roads)

## Lossless Normalization Strategy

Original raw field values, raw units, and raw sensor axis orientations are preserved verbatim in raw archives. The pipeline computes standardized SI representations alongside:

- Acceleration: Converted to m/s^2.
- Gyroscope: Converted to rad/s.
- Speed: Converted to m/s.
- Heading: Converted to radians [0, 2*pi).
- Coordinates: WGS-84 coordinates converted to local East-North-Up (ENU) position meters relative to initial session fix.

## Strict Validation (No Silent Repairs)

Data corruption, missing fields, or timing anomalies trigger immediate session rejection. The pipeline never performs silent interpolation or artificial data repair. Rejection categories include:

1. `TIMESTAMP_ORDERING_VIOLATION`: Non-monotonic or duplicate timestamps.
2. `SAMPLING_JITTER_EXCEEDED`: Time gaps exceeding 3x expected sampling period.
3. `INVALID_ACCEL_UNIT_OR_SCALE`: Unplausible acceleration magnitude (outside 0.1 to 200 m/s^2).
4. `INVALID_GYRO_AXIS_OR_SCALE`: Angular rate exceeding physical bounds (> 40 rad/s).
5. `GNSS_POSITION_JUMP_DETECTED`: Coordinate jump speed exceeding 100 m/s between consecutive fixes.
6. `INSUFFICIENT_SAMPLES`: Total samples below minimum session threshold.

## Multi-Level Group Splitting without Window Leakage

To prevent data leakage between train, validation, and test splits, windowing and group assignment strictly isolate data boundaries:

- **Session-Level Split**: Entire sessions are allocated exclusively to one split.
- **Vehicle-Level Split**: All sessions from a given `vehicle_id` are placed in the same split.
- **Device-Level Split**: All sessions recorded on a given `device_id` are grouped together.
- **Temporal Guard**: A minimum 5-second temporal buffer is maintained between adjacent windows within a session.

## Verification Instructions

Run the dataset pipeline test suite using Pytest:

```bash
./venv/bin/pytest tests/test_india_dataset_pipeline.py -v
```
