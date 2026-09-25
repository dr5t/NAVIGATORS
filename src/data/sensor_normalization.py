from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import math


class DomainType(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"


@dataclass
class PreprocessingConfig:
    domain: DomainType
    target_sample_rate_hz: float = 10.0
    accel_unit: str = "m/s^2"
    gyro_unit: str = "rad/s"
    speed_unit: str = "m/s"
    heading_unit: str = "rad"
    coordinate_convention: str = "ENU"
    lowpass_cutoff_hz: Optional[float] = 4.0
    gravity_handling: str = "retain_raw"
    max_allowed_jitter_s: float = 1.0
    max_allowed_gap_s: float = 2.0
    max_accel_magnitude: float = 150.0
    max_gyro_magnitude: float = 20.0

    @staticmethod
    def for_vehicle() -> "PreprocessingConfig":
        return PreprocessingConfig(
            domain=DomainType.VEHICLE,
            target_sample_rate_hz=10.0,
            lowpass_cutoff_hz=4.0,
            max_accel_magnitude=150.0,
            max_gyro_magnitude=20.0,
        )

    @staticmethod
    def for_pedestrian() -> "PreprocessingConfig":
        return PreprocessingConfig(
            domain=DomainType.PEDESTRIAN,
            target_sample_rate_hz=50.0,
            lowpass_cutoff_hz=12.0,
            max_accel_magnitude=80.0,
            max_gyro_magnitude=40.0,
        )


@dataclass
class TransformationRecord:
    sequence_id: str
    operations_applied: List[str]
    unit_conversions: Dict[str, str]
    axis_mapping: Dict[str, str]
    raw_sampling_rate_hz: float
    target_sampling_rate_hz: float
    resampled_sample_count: int
    normalization_stats_used: Dict[str, Any]


@dataclass
class DatasetNormalizationStats:
    domain: DomainType
    mean: np.ndarray
    std: np.ndarray
    feature_ordering: List[str]
    coordinate_convention: str
    target_sampling_rate_hz: float
    sample_count_train: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_ordering": self.feature_ordering,
            "coordinate_convention": self.coordinate_convention,
            "target_sampling_rate_hz": self.target_sampling_rate_hz,
            "sample_count_train": self.sample_count_train,
        }


@dataclass
class CrossDatasetSequenceResult:
    raw_data: Dict[str, np.ndarray]
    normalized_data: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    transformation_record: TransformationRecord


class CrossDatasetNormalizer:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig.for_vehicle()

    def validate_sequence(self, raw_data: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        errors = []
        if "timestamp" not in raw_data or len(raw_data["timestamp"]) == 0:
            errors.append("MISSING_TIMESTAMP")
            return False, errors

        ts = raw_data["timestamp"]
        if not np.all(np.isfinite(ts)):
            errors.append("NON_FINITE_TIMESTAMPS")

        dt = np.diff(ts)
        if len(dt) > 0 and np.any(dt <= 0):
            errors.append("NON_MONOTONIC_TIMESTAMPS")

        if len(dt) > 0 and np.max(dt) > self.config.max_allowed_gap_s:
            errors.append("SAMPLING_GAP_EXCEEDED")

        if len(ts) != len(set(ts)):
            errors.append("DUPLICATE_TIMESTAMPS")

        for key in ["accel", "gyro"]:
            if key in raw_data:
                arr = raw_data[key]
                if not np.all(np.isfinite(arr)):
                    errors.append(f"NON_FINITE_{key.upper()}")
                if key == "accel" and arr.ndim == 2:
                    mags = np.linalg.norm(arr, axis=1)
                    if np.max(mags) > self.config.max_accel_magnitude:
                        errors.append("ACCEL_MAGNITUDE_EXCEEDED")
                if key == "gyro" and arr.ndim == 2:
                    mags = np.linalg.norm(arr, axis=1)
                    if np.max(mags) > self.config.max_gyro_magnitude:
                        errors.append("GYRO_MAGNITUDE_EXCEEDED")
            else:
                errors.append(f"MISSING_{key.upper()}")

        return len(errors) == 0, errors

    def fit(self, training_sequences: List[Dict[str, np.ndarray]]) -> DatasetNormalizationStats:
        features_list = []
        feature_names = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

        for seq in training_sequences:
            is_valid, errors = self.validate_sequence(seq)
            if not is_valid:
                continue

            accel = seq["accel"]
            gyro = seq["gyro"]
            if "accel_unit" in seq and seq["accel_unit"] == "g":
                accel = accel * 9.80665
            if "gyro_unit" in seq and seq["gyro_unit"] == "deg/s":
                gyro = np.radians(gyro)

            feats = np.column_stack([accel, gyro])
            features_list.append(feats)

        if not features_list:
            mean = np.zeros(6, dtype=np.float64)
            std = np.ones(6, dtype=np.float64)
            total_samples = 0
        else:
            all_feats = np.vstack(features_list)
            mean = np.mean(all_feats, axis=0)
            std = np.std(all_feats, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            total_samples = len(all_feats)

        return DatasetNormalizationStats(
            domain=self.config.domain,
            mean=mean,
            std=std,
            feature_ordering=feature_names,
            coordinate_convention=self.config.coordinate_convention,
            target_sampling_rate_hz=self.config.target_sample_rate_hz,
            sample_count_train=total_samples,
        )

    def transform(
        self,
        raw_dict: Dict[str, Any],
        stats: DatasetNormalizationStats,
        sequence_id: str = "seq_01",
    ) -> CrossDatasetSequenceResult:
        is_valid, errors = self.validate_sequence(raw_dict)
        if not is_valid:
            raise ValueError(f"Sequence validation failed for '{sequence_id}': {', '.join(errors)}")

        ops = []
        unit_convs = {}
        axis_map = {"x": "right", "y": "forward_up", "z": "out_of_screen"}

        ts_raw = np.asarray(raw_dict["timestamp"], dtype=np.float64)
        ts_zero = ts_raw - ts_raw[0]
        ops.append("timestamp_zero_start")

        duration_s = ts_zero[-1] - ts_zero[0]
        raw_hz = float(len(ts_raw) / max(duration_s, 1e-6))

        target_hz = self.config.target_sample_rate_hz
        target_count = max(2, int(round(duration_s * target_hz)))
        ts_grid = np.linspace(0.0, duration_s, target_count, dtype=np.float64)
        ops.append(f"resample_uniform_{target_hz}hz")

        accel_raw = np.asarray(raw_dict["accel"], dtype=np.float64)
        gyro_raw = np.asarray(raw_dict["gyro"], dtype=np.float64)

        accel_unit = raw_dict.get("accel_unit", "m/s^2")
        if accel_unit == "g":
            accel_m = accel_raw * 9.80665
            unit_convs["accel"] = "g_to_mps2"
        else:
            accel_m = accel_raw.copy()
            unit_convs["accel"] = "mps2_identity"

        gyro_unit = raw_dict.get("gyro_unit", "rad/s")
        if gyro_unit == "deg/s":
            gyro_rad = np.radians(gyro_raw)
            unit_convs["gyro"] = "degs_to_rads"
        else:
            gyro_rad = gyro_raw.copy()
            unit_convs["gyro"] = "rads_identity"

        accel_grid = np.zeros((target_count, 3), dtype=np.float64)
        gyro_grid = np.zeros((target_count, 3), dtype=np.float64)

        for col in range(3):
            accel_grid[:, col] = np.interp(ts_grid, ts_zero, accel_m[:, col])
            gyro_grid[:, col] = np.interp(ts_grid, ts_zero, gyro_rad[:, col])

        raw_features = np.column_stack([accel_grid, gyro_grid])
        normalized_imu = (raw_features - stats.mean) / stats.std
        ops.append("zscore_normalize_train_stats")

        normalized_data = {
            "timestamps": ts_grid,
            "imu_unnormalized": raw_features,
            "imu_normalized": normalized_imu,
        }

        if "latitude" in raw_dict and "longitude" in raw_dict:
            lat = np.asarray(raw_dict["latitude"], dtype=np.float64)
            lon = np.asarray(raw_dict["longitude"], dtype=np.float64)
            lat0, lon0 = math.radians(lat[0]), math.radians(lon[0])
            dlat = np.radians(lat - lat[0])
            dlon = np.radians(lon - lon[0])
            east_raw = dlon * math.cos(lat0) * 6371000.0
            north_raw = dlat * 6371000.0

            east_grid = np.interp(ts_grid, ts_zero, east_raw)
            north_grid = np.interp(ts_grid, ts_zero, north_raw)
            normalized_data["enu_position"] = np.column_stack([east_grid, north_grid])
            ops.append("wgs84_to_local_enu")

        if "velocity_north" in raw_dict and "velocity_east" in raw_dict:
            vn_raw = np.asarray(raw_dict["velocity_north"], dtype=np.float64)
            ve_raw = np.asarray(raw_dict["velocity_east"], dtype=np.float64)
            vn_grid = np.interp(ts_grid, ts_zero, vn_raw)
            ve_grid = np.interp(ts_grid, ts_zero, ve_raw)
            normalized_data["velocity_north_east"] = np.column_stack([vn_grid, ve_grid])
            ops.append("ground_truth_velocity_alignment")

        record = TransformationRecord(
            sequence_id=sequence_id,
            operations_applied=ops,
            unit_conversions=unit_convs,
            axis_mapping=axis_map,
            raw_sampling_rate_hz=float(round(raw_hz, 2)),
            target_sampling_rate_hz=target_hz,
            resampled_sample_count=target_count,
            normalization_stats_used=stats.to_dict(),
        )

        return CrossDatasetSequenceResult(
            raw_data=raw_dict,
            normalized_data=normalized_data,
            metadata={
                "domain": self.config.domain.value,
                "sequence_id": sequence_id,
                "source_dataset": raw_dict.get("source_dataset", "unknown"),
            },
            transformation_record=record,
        )
