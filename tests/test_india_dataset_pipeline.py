"""Synthetic contract fixtures only: these tests are not collected road data."""
import copy
import csv
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.india_dataset import (DATASET_NAME, Policy, SessionCollector, build_release,
                                   validate_session, verify_release, window_index)
from src.data.india_dataset.evaluation import evaluate_release
from src.data.india_dataset.schema import GNSS, IMU
from src.data.india_dataset.__main__ import main


from typing import Dict, Any, List


def fixture(index=0, **metadata_changes) -> Dict[str, Any]:
    metadata = dict(schema_version="1.0.0", session_id=f"session-{index}", recording_id=f"drive-{index}",
                    source_kind="navigators_collection", source_dataset=DATASET_NAME, country="IN", consent=True,
                    provenance=dict(original_collection=True, collector="test-only fictional collector",
                                    collection_method="synthetic contract fixture, never publish as collected data", license="test-only"),
                    device_id=f"device-{index}", vehicle_id=f"vehicle-{index}", vehicle_type="car",
                    phone_mount_position="dashboard", activity="driving", road_type="urban", environment="urban",
                    timestamp_clock="elapsed", start_time_utc_s=1700000000 + index * 100,
                    sample_rate_hz=10, acceleration_includes_gravity=True,
                    axes=dict(frame="phone", description="x right, y up, z out of screen",
                              accelerometer=["x", "y", "z"], gyroscope=["x", "y", "z"]),
                    units=dict(timestamp="s", accelerometer="m/s^2", gyroscope="rad/s", coordinates="deg",
                               GNSS_speed="m/s", GNSS_heading="deg", GNSS_accuracy="m"))
    metadata.update(metadata_changes)
    records = [dict(timestamp=i / 10, accelerometer_x=i / 100 + index, accelerometer_y=0, accelerometer_z=9.80665,
                    gyroscope_x=0, gyroscope_y=0, gyroscope_z=0, latitude=13 + i / 1000000, longitude=77,
                    GNSS_speed=1.11, GNSS_heading=0, GNSS_accuracy=3, gnss_status="normal_gnss", gnss_timestamp=i / 10)
               for i in range(12)]
    return dict(metadata=metadata, records=records)


def save(tmp_path, value, name="session.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def release(tmp_path, count=1, **options):
    inputs = [save(tmp_path, fixture(i), f"s{i}.json") for i in range(count)]
    return build_release(inputs, tmp_path / "releases", "v1", window_size=4, stride=2, **options)


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_units_axes_and_raw_are_preserved(tmp_path):
    payload = fixture()
    payload["metadata"]["units"].update(timestamp="ms", accelerometer="g", gyroscope="deg/s", GNSS_speed="km/h")
    payload["metadata"]["axes"]["accelerometer"] = ["y", "-x", "z"]
    for row in payload["records"]:
        row.update(timestamp=row["timestamp"] * 1000, gnss_timestamp=row["gnss_timestamp"] * 1000,
                   accelerometer_x=1, accelerometer_y=2, accelerometer_z=3, gyroscope_z=180, GNSS_speed=36)
    path = save(tmp_path, payload)
    before = path.read_bytes()
    result = build_release([path], tmp_path / "release", "v1")
    manifest = verify_release(result)
    session = manifest["sessions"][0]
    assert (result / session["raw_artifacts"][0]).read_bytes() == before == path.read_bytes()
    rows = [json.loads(line) for line in (result / session["normalized_path"]).read_text().splitlines()]
    assert rows[1]["timestamp"] == pytest.approx(.1)
    assert rows[1]["accelerometer_x"] == pytest.approx(19.6133)
    assert rows[1]["accelerometer_y"] == pytest.approx(-9.80665)
    assert rows[1]["gyroscope_z"] == pytest.approx(3.141592653589793)
    assert rows[1]["GNSS_speed"] == 10


@pytest.mark.parametrize("change,code", [
    (lambda p: p["records"][2].update(timestamp=0), "timestamp_order"),
    (lambda p: p["records"][2].update(accelerometer_x=None), "missing_data"),
    (lambda p: p["records"][2].update(accelerometer_x=float("nan")), "nonfinite_or_type"),
    (lambda p: p["records"][2].update(GNSS_speed=-1), "gnss_range"),
    (lambda p: p["records"][2].update(latitude=95), "gnss_range"),
    (lambda p: p["records"][2].update(latitude=20), "gnss_jump"),
    (lambda p: p["records"][2].update(device_id="wrong"), "session_integrity"),
    (lambda p: p["records"][2].update(source_dataset="IO-VNBD"), "source_provenance"),
    (lambda p: p["metadata"].update(sample_rate_hz=100), "sampling_rate"),
    (lambda p: p["metadata"]["units"].update(accelerometer="unknown"), "units"),
    (lambda p: p["metadata"]["axes"].update(gyroscope=["x", "x", "z"]), "axes"),
    (lambda p: p["records"][2].update(gnss_timestamp=.4), "gnss_timestamp"),
    (lambda p: p["records"][2].update(categories=["tunnel"]), "collection_context"),
    (lambda p: p["metadata"].update(consent=False), "consent"),
])
def test_invalid_sessions_are_rejected_not_repaired(change, code):
    payload = fixture()
    change(payload)
    before = copy.deepcopy(payload)
    quality, rows = validate_session(**payload)
    assert not quality["accepted"] and rows == []
    assert code in codes(quality)
    assert json.dumps(before) == json.dumps(payload)


def test_sampling_gap_duplicate_row_and_missing_context():
    payload = fixture()
    payload["records"][-1]["timestamp"] = 4
    quality, _ = validate_session(**payload)
    assert "sampling_gap" in codes(quality)
    payload = fixture()
    payload["records"][2] = copy.deepcopy(payload["records"][1])
    quality, _ = validate_session(**payload)
    assert {"duplicate_sample", "timestamp_order"} <= codes(quality)
    payload = fixture(device_id=None, vehicle_id=None)
    quality, _ = validate_session(**payload)
    assert quality["accepted"] and "missing_identity" in codes(quality)


def test_outage_and_degraded_data_are_kept_without_fake_labels(tmp_path):
    payload = fixture()
    for row in payload["records"][2:5]:
        row.update({k: None for k in GNSS})
        row["gnss_status"] = "gnss_outage"
    payload["records"][5].update(gnss_status="gnss_recovery", latitude=14)
    payload["records"][6].update(gnss_status="degraded_gnss", latitude=14, GNSS_accuracy=400)
    quality, rows = validate_session(**payload)
    assert quality["accepted"]
    assert "gnss_jump" in codes(quality)
    assert all(rows[i]["latitude"] is None for i in range(2, 5))
    assert rows[5]["latitude"] == 14
    assert quality["statistics"]["category_samples"]["gnss_outage"] == 3
    for row in payload["records"]:
        row.update({k: None for k in GNSS})
        row["gnss_status"] = "gnss_outage"
    assert validate_session(**payload)[0]["accepted"]  # Complete outage sessions are valid.


def test_external_source_separation_and_quarantine(tmp_path):
    payload = fixture(source_kind="external_benchmark", source_dataset="IO-VNBD")
    payload["metadata"]["provenance"] = dict(source_uri="test://external", source_version="fixture-v1", license="test-only", attribution="external test authors")
    path = save(tmp_path, payload)
    own = build_release([path], tmp_path / "out", "v1")
    own_manifest = verify_release(own)
    assert own_manifest["sessions"] == []
    assert own_manifest["statistics"]["rejected_sessions"] == 1
    external = build_release([path], tmp_path / "out", "v1", source_kind="external_benchmark", source_dataset="IO-VNBD")
    manifest = verify_release(external)
    assert manifest["sessions"][0]["source_dataset"] == "IO-VNBD"
    assert own.parent != external.parent
    payload["metadata"]["source_dataset"] = DATASET_NAME
    assert "source_provenance" in codes(validate_session(**payload)[0])


def test_duplicates_and_overlapping_segments_reject_every_conflict(tmp_path):
    a, b = fixture(), fixture()
    b["metadata"]["session_id"] = "renamed"
    paths = [save(tmp_path, a, "a.json"), save(tmp_path, b, "b.json")]
    result = build_release(paths, tmp_path / "out", "duplicate")
    assert verify_release(result)["statistics"]["accepted_sessions"] == 0
    reports = json.loads((result / "quality_report.json").read_text())
    assert all("duplicate_content" in codes(r["quality"]) for r in reports)
    b["records"] = b["records"][6:]
    save(tmp_path, b, "b.json")
    result = build_release(paths, tmp_path / "out", "overlap")
    assert verify_release(result)["statistics"]["accepted_sessions"] == 0
    reports = json.loads((result / "quality_report.json").read_text())
    assert all("duplicate_segment" in codes(r["quality"]) for r in reports)


@pytest.mark.parametrize("strategy,key", [("session", "recording_id"), ("vehicle", "vehicle_id"), ("device", "device_id")])
def test_group_splits_and_windows_never_leak(tmp_path, strategy, key):
    inputs = [save(tmp_path, fixture(i, **{key: f"group-{i // 2}"}), f"{i}.json") for i in range(12)]
    path = build_release(inputs, tmp_path / "out", "one", strategy=strategy, window_size=4, stride=2)
    manifest = verify_release(path)
    membership = {}
    for s in manifest["sessions"]:
        group = s["metadata"][key]
        assert membership.setdefault(group, s["split"]) == s["split"]
    assert set(s["split"] for s in manifest["sessions"]) == {"train", "validation", "test"}
    windows = [json.loads(line) for line in (path / "windows.jsonl").read_text().splitlines()]
    for window in windows:
        session = next(s for s in manifest["sessions"] if s["session_id"] == window["session_id"])
        assert window["split"] == session["split"]
        assert 0 <= window["start"] < window["stop"] <= session["samples"]
    second = build_release(reversed(inputs), tmp_path / "out", "two", strategy=strategy, window_size=4, stride=2)
    assert verify_release(second)["sessions"] == manifest["sessions"]


def test_joint_transitive_groups_and_temporal_guard(tmp_path):
    a = fixture(0, vehicle_id="v0", device_id="d0")
    b = fixture(1, vehicle_id="v1", device_id="d0")
    c = fixture(2, vehicle_id="v1", device_id="d1")
    paths = [save(tmp_path, p, f"{i}.json") for i, p in enumerate([a, b, c])]
    result = build_release(paths, tmp_path / "out", "joint")
    assert verify_release(result)["statistics"]["independent_groups"] == 1
    b["metadata"]["start_time_utc_s"] = a["metadata"]["start_time_utc_s"] + 2
    save(tmp_path, b, "1.json")
    result = build_release(paths, tmp_path / "out", "session", strategy="session")
    sessions = verify_release(result)["sessions"]
    assert sessions[0]["split_group"] == sessions[1]["split_group"]


def test_empty_release_immutable_versions_integrity_and_malformed_inputs(tmp_path):
    path = build_release([], tmp_path, "empty")
    assert verify_release(path)["statistics"]["samples"] == 0
    assert len(verify_release(path)["statistics"]["missing_categories"]) == 14
    with pytest.raises(FileExistsError):
        build_release([], tmp_path, "empty")
    malformed = tmp_path / "broken.json"
    malformed.write_bytes(b'{bad')
    path = build_release([malformed], tmp_path, "bad")
    assert verify_release(path)["statistics"]["rejected_sessions"] == 1
    (path / "statistics.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        verify_release(path)


def test_collector_and_csv_ingestion(tmp_path):
    payload = fixture()
    path = tmp_path / "collected.jsonl"
    with SessionCollector(path, payload["metadata"]) as collector:
        for row in payload["records"]:
            collector.append(row)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        SessionCollector(path, payload["metadata"])
    result = build_release([path], tmp_path / "out", "jsonl")
    assert verify_release(result)["statistics"]["accepted_sessions"] == 1
    assert path.read_bytes() == original
    csv_path = tmp_path / "collected.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=payload["records"][0])
    writer.writeheader()
    writer.writerows(payload["records"])
    csv_path.write_text(buffer.getvalue())
    Path(str(csv_path) + ".metadata.json").write_text(json.dumps(payload["metadata"]))
    result = build_release([csv_path], tmp_path / "out", "csv")
    assert verify_release(result)["statistics"]["accepted_sessions"] == 1


def evaluation_files(tmp_path, path, reference_kind="independent"):
    manifest = verify_release(path)
    session = manifest["sessions"][0]
    normalized = [json.loads(line) for line in (path / session["normalized_path"]).read_text().splitlines()]
    rows = [dict(session_id=session["session_id"], sample_index=i, timestamp_utc_s=r["timestamp_utc_s"], latitude=r["latitude"], longitude=r["longitude"]) for i, r in enumerate(normalized)]
    predictions = dict(metadata=dict(model_id="synthetic-test-only", training_session_ids=[]), records=rows)
    reference = dict(metadata=dict(reference_kind=reference_kind, source_dataset="test reference", provenance="synthetic unit-test coordinates"), records=rows)
    return save(tmp_path, predictions, "predictions.json"), save(tmp_path, reference, "reference.json")


def test_evaluation_reference_coverage_metrics_and_leakage(tmp_path):
    path = release(tmp_path, fractions=dict(train=0, validation=0, test=1))
    predictions, reference = evaluation_files(tmp_path, path)
    report = evaluate_release(path, predictions, reference)
    assert report["metrics"]["position_rmse_m"] == 0
    assert report["by_category"]["gnss_outage"]["position_rmse_m"] is None
    assert report["coverage"]["scored_fraction"] == 1
    content = json.loads(predictions.read_text())
    content["records"] = content["records"][:3]
    predictions.write_text(json.dumps(content))
    assert evaluate_release(path, predictions, reference)["coverage"]["scored_fraction"] == .25
    content["records"][0]["timestamp_utc_s"] += 1
    predictions.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="timestamp"):
        evaluate_release(path, predictions, reference)
    content["metadata"]["training_session_ids"] = ["session-0"]
    predictions.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="leakage"):
        evaluate_release(path, predictions, reference)


def test_receiver_proxy_cannot_score_outage_and_empty_scores_are_null(tmp_path):
    payload = fixture()
    for row in payload["records"]:
        row["gnss_status"] = "gnss_outage"  # Retained receiver values are explicitly untrusted.
    path = build_release([save(tmp_path, payload)], tmp_path / "out", "v1", fractions=dict(train=0, validation=0, test=1))
    predictions, reference = evaluation_files(tmp_path, path, "receiver_proxy")
    report = evaluate_release(path, predictions, reference)
    assert report["metrics"]["samples"] == 0
    assert report["metrics"]["position_rmse_m"] is None
    assert report["coverage"]["excluded_proxy_samples"] == 12
    assert "not ground-truth accuracy" in report["metric_basis"]


def test_evaluation_nonzero_distance_and_duplicate_rejection(tmp_path):
    path = release(tmp_path, fractions=dict(train=0, validation=0, test=1))
    predictions, reference = evaluation_files(tmp_path, path)
    content = json.loads(predictions.read_text())
    for row in content["records"]:
        row["latitude"] += .001
    predictions.write_text(json.dumps(content))
    report = evaluate_release(path, predictions, reference)
    assert report["metrics"]["mean_position_error_m"] == pytest.approx(111.1949266)
    assert report["metrics"]["position_rmse_m"] == pytest.approx(111.1949266)
    assert report["metrics"]["p95_position_error_m"] == pytest.approx(111.1949266)
    content["records"].append(content["records"][0])
    predictions.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_release(path, predictions, reference)


def test_unrepresentable_rate_is_reported_without_nonfinite_json(tmp_path):
    payload = fixture()
    for i, row in enumerate(payload["records"]):
        row["timestamp"] = row["gnss_timestamp"] = i * 1e-310
    path = build_release([save(tmp_path, payload)], tmp_path / "out", "tiny-timestamps")
    assert verify_release(path)["statistics"]["rejected_sessions"] == 1
    assert "Infinity" not in (path / "quality_report.json").read_text()


def test_cli_reports_rejection_with_nonzero_status(tmp_path):
    payload = fixture()
    payload["records"][0]["timestamp"] = None
    path = save(tmp_path, payload)
    assert main(["build", str(path), "--output", str(tmp_path / "out"), "--version", "cli"]) == 2


def test_ambiguous_json_is_rejected_and_raw_is_quarantined(tmp_path):
    payload = fixture()
    path = save(tmp_path, payload)
    raw = path.read_text().replace('"timestamp": 0.0', '"timestamp": 9, "timestamp": 0.0', 1)
    path.write_text(raw)
    result = build_release([path], tmp_path / "out", "ambiguous")
    report = json.loads((result / "quality_report.json").read_text())[0]
    assert "parse_error" in codes(report["quality"])
    assert all(name.startswith("quarantine/raw/") for name in report["raw_artifacts"])
    assert (result / report["raw_artifacts"][0]).read_bytes() == path.read_bytes()
    assert verify_release(result)["statistics"]["accepted_sessions"] == 0


@pytest.mark.parametrize("key,value", [("timestamp", -1), ("timestamp", 1e308), ("accelerometer_x", 10**1000)])
def test_extreme_numbers_are_rejections_not_crashes(key, value):
    payload = fixture()
    rec: dict = payload["records"][0]
    rec[key] = value
    assert not validate_session(**payload)[0]["accepted"]


@pytest.mark.parametrize("kwargs", [dict(version="../escape"), dict(strategy="window"),
    dict(fractions=dict(train=1, validation=1, test=1)), dict(window_size=0)])
def test_invalid_configuration_fails_before_publishing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_release([], tmp_path, **{**dict(version="v1"), **kwargs})
