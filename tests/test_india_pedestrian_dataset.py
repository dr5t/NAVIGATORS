import os
import shutil
import tempfile
import json
import numpy as np
import pytest
from pathlib import Path

from src.data.india_pedestrian_dataset import (
    PedestrianSessionMetadata,
    PedestrianQualityReport,
    NavigatorsIndiaPedestrianPipeline,
)


def create_sample_raw_pedestrian_session(
    session_id: str = "ped_s1",
    participant_id: str = "p01",
    gt_type: str = "rtk_gnss",
    placement: str = "handheld",
    activity: str = "normal_walking",
    n_samples: int = 300,
    dur_s: float = 30.0,
):
    t = np.linspace(0, dur_s, n_samples)
    acc = np.random.randn(n_samples, 3).astype(np.float32)
    acc[:, 2] += 9.81
    gyro = np.random.randn(n_samples, 3).astype(np.float32) * 0.15
    lat = 28.6139 + np.linspace(0, 0.001, n_samples)
    lon = 77.2090 + np.linspace(0, 0.001, n_samples)
    vn = np.ones(n_samples, dtype=np.float32) * 1.2
    ve = np.ones(n_samples, dtype=np.float32) * 0.8

    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "user_id": participant_id,
        "device_model": "Pixel 6",
        "android_version": "14.0",
        "phone_placement": placement,
        "phone_orientation": "portrait_upright",
        "environment": "urban",
        "activity": activity,
        "ground_truth_type": gt_type,
        "timestamp": t,
        "accelerometer_x": acc[:, 0],
        "accelerometer_y": acc[:, 1],
        "accelerometer_z": acc[:, 2],
        "gyroscope_x": gyro[:, 0],
        "gyroscope_y": gyro[:, 1],
        "gyroscope_z": gyro[:, 2],
        "latitude": lat,
        "longitude": lon,
        "velocity_north": vn,
        "velocity_east": ve,
    }


def test_pedestrian_session_metadata_extraction():
    pipeline = NavigatorsIndiaPedestrianPipeline()
    sess = create_sample_raw_pedestrian_session()
    meta = pipeline.extract_metadata(sess)
    assert isinstance(meta, PedestrianSessionMetadata)
    assert meta.participant_id == "p01"
    assert meta.session_id == "ped_s1"
    assert meta.phone_placement == "handheld"
    assert meta.ground_truth_type == "rtk_gnss"
    assert meta.sample_count == 300
    assert meta.duration_s > 0.0
    d = meta.to_dict()
    assert d["source_dataset"] == "Navigators India Pedestrian Dataset"


def test_pedestrian_quality_report_and_gt_verification():
    pipeline = NavigatorsIndiaPedestrianPipeline()
    high_gt_sess = create_sample_raw_pedestrian_session(gt_type="rtk_gnss")
    report_high = pipeline.validate_session(high_gt_sess)
    assert report_high.validation_status in ["PASSED", "WARNING"]
    assert report_high.ground_truth_quality_score == 1.0

    low_gt_sess = create_sample_raw_pedestrian_session(gt_type="standard_smartphone_gnss")
    report_low = pipeline.validate_session(low_gt_sess)
    assert report_low.ground_truth_quality_score == 0.4
    assert any("UNRELIABLE_SINGLE_FREQ_GNSS" in issue for issue in report_low.issues_detected)


def test_controlled_outage_subset_generation():
    pipeline = NavigatorsIndiaPedestrianPipeline()
    sess = create_sample_raw_pedestrian_session(n_samples=900, dur_s=100.0)
    outages = pipeline.create_controlled_outage_subsets(sess, outage_durations_s=[10, 30])
    assert len(outages) == 2
    for out_sub in outages:
        assert out_sub["contains_outages"] is True
        assert "latitude_masked" in out_sub
        assert np.any(np.isnan(out_sub["latitude_masked"]))


def test_process_and_normalize_session():
    pipeline = NavigatorsIndiaPedestrianPipeline()
    sess = create_sample_raw_pedestrian_session()
    proc = pipeline.process_and_normalize_session(sess)
    assert "normalized_imu" in proc
    assert "timestamps_grid" in proc
    assert proc["normalized_imu"].shape[1] == 6


def test_export_session_artifact():
    tmp_dir = tempfile.mkdtemp()
    try:
        pipeline = NavigatorsIndiaPedestrianPipeline(base_output_dir=tmp_dir)
        sess = create_sample_raw_pedestrian_session()
        paths = pipeline.export_session_artifact(sess)

        assert paths["metadata"].exists()
        assert paths["quality_report"].exists()
        assert paths["ground_truth"].exists()
        assert paths["processed"].exists()

        with open(paths["metadata"], "r") as f:
            meta_json = json.load(f)
            assert meta_json["session_id"] == "ped_s1"
    finally:
        shutil.rmtree(tmp_dir)
