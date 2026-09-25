import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.dataset_registry import (
    DatasetRegistry,
    DatasetDomain,
    DatasetProvenanceCategory,
    DatasetMetadata,
    UnifiedSensorRecord,
)


def test_registry_preset_datasets():
    registry = DatasetRegistry()

    v_datasets = registry.list_datasets(domain=DatasetDomain.VEHICLE)
    assert any(d.dataset_name == "IO-VNBD" for d in v_datasets)
    assert any(d.dataset_name == "Navigators India Dataset - Vehicle" for d in v_datasets)

    p_datasets = registry.list_datasets(domain=DatasetDomain.PEDESTRIAN)
    assert any(d.dataset_name == "RoNIN" for d in p_datasets)
    assert any(d.dataset_name == "OxIOD" for d in p_datasets)
    assert any(d.dataset_name == "Navigators India Dataset - Pedestrian" for d in p_datasets)


def test_i2wdd_is_quarantined_not_verified():
    registry = DatasetRegistry()
    i2wdd = registry.get_dataset("I2WDD")

    assert i2wdd is not None
    assert not i2wdd.is_verified
    assert "QUARANTINED" in i2wdd.known_limitations[0]

    verified_vehicle_datasets = registry.list_datasets(domain=DatasetDomain.VEHICLE, is_verified_only=True)
    assert not any(d.dataset_name == "I2WDD" for d in verified_vehicle_datasets)


def test_third_party_provenance_separation():
    registry = DatasetRegistry()
    external = registry.list_datasets(category=DatasetProvenanceCategory.EXTERNAL)
    original = registry.list_datasets(category=DatasetProvenanceCategory.ORIGINAL)

    assert all(d.category == DatasetProvenanceCategory.EXTERNAL for d in external)
    assert all(d.category == DatasetProvenanceCategory.ORIGINAL for d in original)

    iovnbd = registry.get_dataset("IO-VNBD")
    assert iovnbd.license == "CC-BY-4.0"
    assert "Oxford" in iovnbd.source


def test_unified_sensor_record_mapping_no_fabrication():
    raw_dict = {
        "timestamp": 1700000000.123,
        "accel_x": 0.05,
        "accel_y": 0.10,
        "accel_z": 9.81,
        "gyro_x": 0.01,
        "gyro_y": -0.02,
        "gyro_z": 0.005,
        "latitude": 28.6139,
        "longitude": 77.2090,
    }

    record = DatasetRegistry.map_to_unified_record(raw_dict, source_dataset="IO-VNBD")

    assert record.timestamp == 1700000000.123
    assert record.accel_z == 9.81
    assert record.gyro_x == 0.01
    assert record.latitude == 28.6139
    assert record.mag_x is None
    assert record.velocity_north is None
    assert record.vehicle_type is None
    assert record.source_dataset == "IO-VNBD"

    is_valid, errors = DatasetRegistry.validate_record_schema(record)
    assert is_valid
    assert len(errors) == 0


def test_record_validation_detects_missing_imu():
    raw_incomplete = {
        "timestamp": 100.0,
        "latitude": 13.0,
        "longitude": 77.0,
    }
    record = DatasetRegistry.map_to_unified_record(raw_incomplete, source_dataset="Test")
    is_valid, errors = DatasetRegistry.validate_record_schema(record)

    assert not is_valid
    assert "INCOMPLETE_6AXIS_IMU" in errors
