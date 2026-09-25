import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.sensor_normalization import (
    CrossDatasetNormalizer,
    PreprocessingConfig,
    DatasetNormalizationStats,
    DomainType,
    CrossDatasetSequenceResult,
)


def create_synthetic_sequence(
    num_samples: int = 100,
    sample_rate_hz: float = 10.0,
    accel_unit: str = "m/s^2",
    gyro_unit: str = "rad/s",
    accel_scale: float = 1.0,
    gyro_scale: float = 0.1,
    source_dataset: str = "IO-VNBD",
) -> dict:
    dt = 1.0 / sample_rate_hz
    ts = np.array([1000.0 + i * dt for i in range(num_samples)], dtype=np.float64)

    accel = np.random.normal(loc=[0.1, 0.0, 9.81], scale=accel_scale, size=(num_samples, 3))
    gyro = np.random.normal(loc=[0.01, -0.01, 0.005], scale=gyro_scale, size=(num_samples, 3))

    lat = np.array([28.6139 + i * dt * 0.0001 for i in range(num_samples)], dtype=np.float64)
    lon = np.array([77.2090 for _ in range(num_samples)], dtype=np.float64)
    vn = np.array([10.0 for _ in range(num_samples)], dtype=np.float64)
    ve = np.array([0.0 for _ in range(num_samples)], dtype=np.float64)

    return {
        "timestamp": ts,
        "accel": accel,
        "gyro": gyro,
        "accel_unit": accel_unit,
        "gyro_unit": gyro_unit,
        "latitude": lat,
        "longitude": lon,
        "velocity_north": vn,
        "velocity_east": ve,
        "source_dataset": source_dataset,
    }


def test_unit_conversions():
    config = PreprocessingConfig.for_vehicle()
    normalizer = CrossDatasetNormalizer(config=config)

    seq_g = create_synthetic_sequence(num_samples=50, accel_unit="g", gyro_unit="deg/s")
    stats = DatasetNormalizationStats(
        domain=DomainType.VEHICLE,
        mean=np.zeros(6),
        std=np.ones(6),
        feature_ordering=["a_x", "a_y", "a_z", "g_x", "g_y", "g_z"],
        coordinate_convention="ENU",
        target_sampling_rate_hz=10.0,
        sample_count_train=100,
    )

    res = normalizer.transform(seq_g, stats=stats, sequence_id="unit_test_01")
    assert "g_to_mps2" in res.transformation_record.unit_conversions.get("accel", "")
    assert "degs_to_rads" in res.transformation_record.unit_conversions.get("gyro", "")
    assert res.normalized_data["imu_unnormalized"].shape[1] == 6


def test_train_only_statistics_no_leakage():
    config = PreprocessingConfig.for_vehicle()
    normalizer = CrossDatasetNormalizer(config=config)

    train_seqs = [create_synthetic_sequence(num_samples=100, accel_scale=0.5) for _ in range(3)]
    test_seq = create_synthetic_sequence(num_samples=100, accel_scale=5.0)

    train_stats = normalizer.fit(train_seqs)

    assert train_stats.sample_count_train == 300
    assert np.all(train_stats.std > 0)

    res_test = normalizer.transform(test_seq, stats=train_stats, sequence_id="test_seq_01")
    used_stats = res_test.transformation_record.normalization_stats_used

    assert used_stats["sample_count_train"] == 300
    assert np.allclose(used_stats["mean"], train_stats.mean.tolist())


def test_validation_rejects_non_monotonic_timestamps():
    normalizer = CrossDatasetNormalizer()
    seq_bad = create_synthetic_sequence(num_samples=50)
    seq_bad["timestamp"][20] = seq_bad["timestamp"][15]

    is_valid, errors = normalizer.validate_sequence(seq_bad)
    assert not is_valid
    assert "NON_MONOTONIC_TIMESTAMPS" in errors

    with pytest.raises(ValueError, match="NON_MONOTONIC_TIMESTAMPS"):
        stats = DatasetNormalizationStats(
            domain=DomainType.VEHICLE,
            mean=np.zeros(6),
            std=np.ones(6),
            feature_ordering=[],
            coordinate_convention="ENU",
            target_sampling_rate_hz=10.0,
            sample_count_train=100,
        )
        normalizer.transform(seq_bad, stats=stats)


def test_validation_rejects_duplicate_timestamps():
    normalizer = CrossDatasetNormalizer()
    seq_dup = create_synthetic_sequence(num_samples=50)
    seq_dup["timestamp"][10] = seq_dup["timestamp"][9]

    is_valid, errors = normalizer.validate_sequence(seq_dup)
    assert not is_valid
    assert "DUPLICATE_TIMESTAMPS" in errors


def test_validation_rejects_non_finite_values():
    normalizer = CrossDatasetNormalizer()
    seq_nan = create_synthetic_sequence(num_samples=50)
    seq_nan["accel"][5, 0] = np.nan

    is_valid, errors = normalizer.validate_sequence(seq_nan)
    assert not is_valid
    assert "NON_FINITE_ACCEL" in errors


def test_vehicle_vs_pedestrian_config():
    veh_config = PreprocessingConfig.for_vehicle()
    ped_config = PreprocessingConfig.for_pedestrian()

    assert veh_config.target_sample_rate_hz == 10.0
    assert ped_config.target_sample_rate_hz == 50.0
    assert veh_config.lowpass_cutoff_hz == 4.0
    assert ped_config.lowpass_cutoff_hz == 12.0
    assert veh_config.domain == DomainType.VEHICLE
    assert ped_config.domain == DomainType.PEDESTRIAN


def test_ground_truth_and_coordinate_alignment():
    normalizer = CrossDatasetNormalizer(config=PreprocessingConfig.for_vehicle())
    seq = create_synthetic_sequence(num_samples=100)
    stats = normalizer.fit([seq])

    res = normalizer.transform(seq, stats=stats, sequence_id="align_seq")

    assert "enu_position" in res.normalized_data
    assert "velocity_north_east" in res.normalized_data
    assert len(res.normalized_data["enu_position"]) == res.transformation_record.resampled_sample_count
    assert np.allclose(res.normalized_data["enu_position"][0], [0.0, 0.0])
