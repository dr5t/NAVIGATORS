from __future__ import annotations
"""Release verification and strict sample-identity joins, isolated from navigation."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from data.india_dataset.pipeline import verify_release, _strict_json
from data.india_dataset.schema import IMU, GNSS, digest
from evaluation.recording import Recording, enu, sha256


@dataclass
class Session:
    recording: Recording
    rows: list
    entry: dict
    manifest: dict
    references: dict
    reference_metadata: dict
    identity: dict
    map_path: str | None


def read_session(spec):
    if set(spec) - {"release", "session_id", "reference", "map", "manifest_sha256"}:
        raise ValueError("Unknown session specification fields")
    release = Path(spec["release"])
    manifest = verify_release(release)
    if spec.get("manifest_sha256") and spec["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Pinned release hash changed")
    entry = next((s for s in manifest["sessions"] if s["session_id"] == spec["session_id"]), None)
    if entry is None or entry["split"] != "test":
        raise ValueError("Benchmark sessions must belong to the verified test split")
    metadata = entry["metadata"]
    if metadata.get("acceleration_includes_gravity") is not True:
        raise ValueError("Current navigation frontend requires acceleration including gravity")
    rows = [_strict_json(line) for line in (release / entry["normalized_path"]).read_text().splitlines()]
    ts = np.array([r["elapsed_s"] for r in rows])
    accel = np.array([[r[k] for k in IMU[:3]] for r in rows], dtype=float)
    gyro = np.array([[r[k] for k in IMU[3:]] for r in rows], dtype=float)
    fixes = np.array([[r["latitude"], r["longitude"], 0., r["GNSS_speed"], r["GNSS_heading"], r["GNSS_accuracy"]] for r in rows], dtype=float)
    # Availability is a numeric/receiver gate, not a trust/anomaly classifier.
    available = np.isfinite(fixes).all(axis=1) & np.array([r["gnss_status"] != "gnss_outage" for r in rows])
    fix_ts = [r.get("gnss_timestamp") for r in rows]
    fresh = np.array([available[i] and (i == 0 or fix_ts[i] is None or fix_ts[i] != fix_ts[i-1]) for i in range(len(rows))])
    for i, row in enumerate(rows):
        if fix_ts[i] is not None:
            available[i] &= 0 <= row["timestamp"] - fix_ts[i] <= manifest["policy"]["max_fix_age_s"]
    recording = Recording(ts, accel, gyro, fixes, available, fresh & available, metadata,
                          str(release / entry["normalized_path"]), sha256(release / entry["normalized_path"]))
    refs, ref_meta = {}, {"reference_kind": "unavailable", "source_dataset": None}
    ref_hash = None
    if spec.get("reference"):
        ref_path = Path(spec["reference"])
        payload = _strict_json(ref_path.read_bytes())
        ref_meta = payload["metadata"]
        if ref_meta.get("reference_kind") not in ("independent", "receiver_proxy", "synthetic_truth"):
            raise ValueError("Explicit reference kind required")
        if any(not isinstance(ref_meta.get(k), str) or not ref_meta[k].strip() for k in ("source_dataset", "provenance")):
            raise ValueError("Reference source_dataset and provenance required")
        if ref_meta["reference_kind"] == "synthetic_truth" and manifest["source_kind"] != "synthetic":
            raise ValueError("Synthetic truth cannot score collected observations")
        for row in payload["records"]:
            if row.get("session_id") != entry["session_id"]:
                raise ValueError("Reference file must describe exactly this session")
            i = row.get("sample_index")
            if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(rows) or i in refs:
                raise ValueError("Invalid or duplicate reference sample identity")
            t = row.get("timestamp_utc_s")
            if not isinstance(t, (int, float)) or not np.isfinite(t) or abs(t - rows[i]["timestamp_utc_s"]) > 1e-6:
                raise ValueError("Reference timestamp does not match sample identity")
            for key in ("latitude", "longitude", "velocity_east_mps", "velocity_north_mps", "heading_rad"):
                v = row.get(key)
                if v is not None and (not isinstance(v, (float, int)) or isinstance(v, bool) or not np.isfinite(v)):
                    raise ValueError(f"Invalid reference {key}")
            if (row.get("latitude") is None) != (row.get("longitude") is None) or (row.get("latitude") is not None and (abs(row["latitude"]) > 90 or abs(row["longitude"]) > 180)):
                raise ValueError("Invalid reference coordinates")
            if (row.get("velocity_east_mps") is None) != (row.get("velocity_north_mps") is None):
                raise ValueError("Reference velocity requires both ENU components")
            if row.get("anomaly") is not None and not isinstance(row["anomaly"], bool):
                raise ValueError("Anomaly labels must be boolean or null")
            if row.get("road_segment_id") is not None and not isinstance(row["road_segment_id"], str):
                raise ValueError("Road labels must be exact segment identifiers")
            refs[i] = row
        ref_hash = sha256(ref_path)
    identity = dict(source_dataset=manifest["source_dataset"], source_kind=manifest["source_kind"],
                    dataset_version=manifest["dataset_version"], manifest_sha256=manifest["manifest_sha256"],
                    session_id=entry["session_id"], normalized_sha256=recording.digest,
                    reference_sha256=ref_hash, reference_kind=ref_meta["reference_kind"],
                    reference_source_dataset=ref_meta["source_dataset"],
                    split_group=entry["split_group"],
                    map_sha256=sha256(spec["map"]) if spec.get("map") else None)
    return Session(recording, rows, entry, manifest, refs, ref_meta, identity, spec.get("map"))


def cases(session, config):
    if config["include_observed"]:
        yield dict(kind="observed", start_s=None, duration_s=None)
    for start in config["outage_starts_s"]:
        for duration in config["outage_durations_s"]:
            yield dict(kind="masked_outage", start_s=start, duration_s=duration)


def prepare_case(session, case, config, model_window=0, model_rate=None):
    from evaluation.preprocessing import calibrate
    rec = session.recording
    first = int(np.searchsorted(rec.timestamps, config["calibration_seconds"]))
    if first >= len(rec.timestamps) - 1 or first < 3 or not rec.valid[first]:
        raise ValueError("Session lacks a usable initialization fix/calibration prefix")
    prefix = np.flatnonzero(rec.valid[:first])
    if len(prefix) < 3:
        raise ValueError("Insufficient calibration fixes")
    aligned = session.entry["metadata"]["axes"]["frame"] == "vehicle"
    rotation = np.eye(3) if aligned else calibrate(rec.accel[prefix], rec.gnss[prefix, 3], rec.timestamps[prefix])
    if model_rate and abs(rec.sample_rate / model_rate - 1) > .1:
        raise ValueError("Session sampling rate differs from model contract")
    mask = np.zeros(len(rec.timestamps), bool)
    if case["kind"] == "masked_outage":
        start, end = case["start_s"], case["start_s"] + case["duration_s"]
        if start <= rec.timestamps[first] or end + config["recovery_window_s"] > rec.timestamps[-1]:
            raise ValueError("Duration unsupported: require pre-outage initialization and the complete recovery window")
        if (start - rec.timestamps[first]) * rec.sample_rate < model_window:
            raise ValueError("Outage starts before the model history is ready")
        mask = (rec.timestamps >= start) & (rec.timestamps < end)
        if not mask.any():
            raise ValueError("Outage contains no samples")
    allowed = rec.valid & ~mask
    return first, rotation, rec.gnss[first, :2].copy(), mask, allowed


def reference_at(session, index, origin):
    ref = session.references.get(index, {})
    position = enu([ref["latitude"], ref["longitude"]], origin).tolist() if ref.get("latitude") is not None else None
    velocity = [ref["velocity_east_mps"], ref["velocity_north_mps"]] if ref.get("velocity_east_mps") is not None else None
    return dict(position=position, velocity=velocity, heading=ref.get("heading_rad"),
                anomaly=ref.get("anomaly"), road_segment_id=ref.get("road_segment_id"))


def generalization(session, model_specs):
    """Unknown development exposure stays unknown; never infer unseen from filenames."""
    inventories = []
    for model in model_specs:
        if not model or not model.get("training_inventory"):
            return {"unseen_" + k: None for k in ("session", "route", "vehicle", "device")}
        value = _strict_json(Path(model["training_inventory"]).read_bytes())
        if not isinstance(value, list) or any(not isinstance(v, dict) or not isinstance(v.get("source_dataset"), str) or not isinstance(v.get("session_id"), str) for v in value):
            raise ValueError("Training inventory must list development session identities and sources")
        inventories.extend(value)
    result = {}
    meta = session.entry["metadata"]
    for kind in ("session", "route", "vehicle", "device"):
        key = kind + "_id"
        value = meta.get(key)
        if value is None or any(v.get(key) is None for v in inventories):
            result["unseen_" + kind] = None
        else:
            result["unseen_" + kind] = not any(v["source_dataset"] == session.identity["source_dataset"] and v[key] == value for v in inventories)
    if result["unseen_session"] is False or any(v.get("recording_id") == meta["recording_id"] and v["source_dataset"] == session.identity["source_dataset"] for v in inventories):
        raise ValueError("Development/test leakage: this session or physical recording was used in development")
    return result
