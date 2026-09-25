"""Versioned raw contract and explicit, lossless unit normalization.

Validation never sorts, fills, filters, resamples, estimates alignment, or repairs.
Sensor frame declarations are checked, not inferred from the measured motion.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math

DATASET_NAME = "Navigators India Dataset"
SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"
IMU = tuple(f"{sensor}_{axis}" for sensor in ("accelerometer", "gyroscope") for axis in "xyz")
GNSS = ("latitude", "longitude", "GNSS_speed", "GNSS_heading", "GNSS_accuracy")
CONTEXT = ("device_id", "vehicle_id", "session_id", "phone_mount_position",
           "vehicle_type", "road_type", "environment", "activity")
RAW_FIELDS = ("timestamp",) + IMU + GNSS + CONTEXT
STATES = ("normal_gnss", "degraded_gnss", "gnss_outage", "gnss_recovery")
CATEGORIES = STATES + ("urban", "highway", "rural", "hilly", "dense_traffic",
                       "stop_and_go", "parallel_roads", "service_roads", "underpass", "tunnel")
UNIT_FACTORS = {
    "timestamp": {"s": 1., "ms": .001, "us": .000001, "ns": .000000001},
    "accelerometer": {"m/s^2": 1., "g": 9.80665},
    "gyroscope": {"rad/s": 1., "deg/s": math.pi / 180},
    "coordinates": {"deg": 1., "rad": 180 / math.pi},
    "GNSS_speed": {"m/s": 1., "km/h": 1 / 3.6},
    "GNSS_heading": {"deg": 1., "rad": 180 / math.pi},
    "GNSS_accuracy": {"m": 1.},
}
NORMALIZED_UNITS = {key: next(iter(values)) for key, values in UNIT_FACTORS.items()}


@dataclass(frozen=True)
class Policy:
    sample_rate_tolerance: float = .15
    max_gap_periods: float = 3.
    max_jitter_fraction: float = .5
    max_acceleration_mps2: float = 200.
    max_angular_rate_rad_s: float = 40.
    max_speed_mps: float = 100.
    normal_accuracy_m: float = 20.
    max_fix_age_s: float = 3.
    temporal_guard_s: float = 5.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not number(value) or value < 0:
                raise ValueError(f"Policy {key} must be finite and nonnegative")
        if self.max_gap_periods < 1 or self.sample_rate_tolerance >= 1:
            raise ValueError("Invalid sampling policy")


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def distance_m(a, b):
    lat1, lat2 = map(math.radians, (a[0], b[0]))
    dlat, dlon = lat2 - lat1, math.radians(b[1] - a[1])
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(min(1., math.sqrt(max(0., h))))


def validate_session(metadata, records, policy=None):
    """Return normalized rows plus machine-readable errors, warnings and statistics.

    GNSS loss must be explicitly annotated. In-range but implausible fixes during
    degraded/recovery periods are retained and flagged; normal fixes must agree.
    """
    policy = policy or Policy()
    issues, rows = [], []

    def issue(code, message, row=None, severity="error"):
        issues.append(dict(code=code, message=message, row=row, severity=severity))

    if not isinstance(metadata, dict) or not isinstance(records, list):
        issue("schema", "metadata must be an object and records must be a list")
        return {"accepted": False, "issues": issues, "statistics": {}}, []
    try:
        canonical(metadata)
    except (ValueError, TypeError):
        issue("metadata", "Metadata must be finite JSON values")
    for key in ("session_id", "recording_id", "source_dataset", "source_kind", "country",
                "timestamp_clock", "vehicle_type", "phone_mount_position", "activity"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            issue("metadata", f"Missing/non-string metadata.{key}")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        issue("schema_version", f"Expected schema_version {SCHEMA_VERSION}")
    kind, source = metadata.get("source_kind"), metadata.get("source_dataset")
    if kind not in ("navigators_collection", "external_benchmark", "synthetic"):
        issue("source_provenance", "Unknown source_kind")
    if (kind == "navigators_collection") != (source == DATASET_NAME):
        issue("source_provenance", "Only Navigators original collections may use the Navigators dataset name")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        issue("source_provenance", "provenance must be an object")
        provenance = {}
    required = ("collector", "collection_method", "license") if kind == "navigators_collection" else (
        "source_uri", "source_version", "license", "attribution")
    for key in required:
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            issue("source_provenance", f"Missing provenance.{key}")
    if kind == "navigators_collection":
        if metadata.get("country") != "IN" or provenance.get("original_collection") is not True:
            issue("source_provenance", "Original India collections require country IN and original_collection=true")
        if metadata.get("consent") is not True:
            issue("consent", "Original collections require consent=true")
        if any(provenance.get(k) for k in ("source_dataset", "derived_from_dataset")):
            issue("source_provenance", "Externally derived data cannot be an original collection")
    for key in ("device_id", "vehicle_id"):
        if metadata.get(key) is None:
            issue("missing_identity", f"{key} unavailable; isolation cannot be guaranteed", severity="warning")
        elif not isinstance(metadata[key], str) or not metadata[key].strip():
            issue("session_integrity", f"{key} must be a nonempty pseudonym or null")
    if metadata.get("timestamp_clock") not in ("unix", "elapsed"):
        issue("timestamp_clock", "timestamp_clock must be unix or elapsed")
    if metadata.get("timestamp_clock") == "elapsed":
        start_time = metadata.get("start_time_utc_s")
        if not number(start_time) or not 0 <= start_time <= 253402300799:
            issue("timestamp_clock", "Elapsed clocks require a UTC epoch start_time_utc_s within years 1970–9999")
    rate = metadata.get("sample_rate_hz")
    if not number(rate) or rate <= 0:
        issue("sampling_rate", "sample_rate_hz must be finite and positive")
    units = metadata.get("units", {})
    if not isinstance(units, dict):
        units = {}
    factors = {}
    for key, allowed in UNIT_FACTORS.items():
        value = units.get(key)
        if not isinstance(value, str) or value not in allowed:
            issue("units", f"Declare {key} unit from {list(allowed)}")
        else:
            factors[key] = allowed[value]
    axes = metadata.get("axes", {})
    if not isinstance(axes, dict):
        axes = {}
    for sensor in ("accelerometer", "gyroscope"):
        mapping = axes.get(sensor)
        if (not isinstance(mapping, list) or len(mapping) != 3
                or any(not isinstance(x, str) or x not in ("x", "y", "z", "-x", "-y", "-z") for x in mapping)
                or len({x.lstrip("-") for x in mapping}) != 3):
            issue("axes", f"axes.{sensor} must be a signed permutation of x,y,z")
    if axes.get("frame") not in ("phone", "vehicle"):
        issue("axes", "Declare axes.frame as phone or vehicle")
    if not isinstance(axes.get("description"), str) or not axes["description"].strip():
        issue("axes", "Describe the physical positive x,y,z directions")
    if metadata.get("acceleration_includes_gravity") not in (True, False) or not isinstance(metadata.get("acceleration_includes_gravity"), bool):
        issue("axes", "Declare acceleration_includes_gravity as boolean")
    if len(records) < 3:
        issue("session_integrity", "At least three samples are required")
    # Do not guess conversions when the contract is invalid.
    if any(i["severity"] == "error" for i in issues):
        return {"accepted": False, "issues": issues, "statistics": {"samples": len(records)}}, []
    missing, category_samples, seen = Counter(), Counter(), set()
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            issue("schema", "Sample must be an object", index)
            continue
        row = {}
        for key in ("source_dataset", "source_kind"):
            if key in raw and raw[key] != metadata.get(key):
                issue("source_provenance", f"Sample {key} conflicts with session provenance", index)
        for key in RAW_FIELDS:
            value = raw.get(key, metadata.get(key) if key in CONTEXT else None)
            row[key] = value
            if value is None:
                missing[key] += 1
        for key in CONTEXT:
            if row[key] is not None and (not isinstance(row[key], str) or not row[key].strip()):
                issue("context", f"{key} must be a nonempty string or null", index)
        for key in ("session_id", "device_id", "vehicle_id", "vehicle_type", "phone_mount_position"):
            if row[key] != metadata.get(key):
                issue("session_integrity", f"Sample {key} differs from session metadata", index)
        for key in ("timestamp",) + IMU + GNSS:
            if row[key] is None:
                if key == "timestamp" or key in IMU:
                    issue("missing_data", f"Missing required {key}", index)
                continue
            if not number(row[key]):
                issue("nonfinite_or_type", f"{key} must be a finite number or null", index)
                row[key] = None
                continue
            group = (key.split("_")[0] if key in IMU else
                     "coordinates" if key in ("latitude", "longitude") else key)
            row[key] *= factors[group]
            if not number(row[key]):
                issue("nonfinite_or_type", f"{key} conversion overflow", index)
                row[key] = None
        for sensor in ("accelerometer", "gyroscope"):
            original = {a: row[f"{sensor}_{a}"] for a in "xyz"}
            for target, axis in zip("xyz", axes[sensor]):
                value = original[axis.lstrip("-")]
                row[f"{sensor}_{target}"] = None if value is None else value * (-1 if axis.startswith("-") else 1)
        timestamp = row["timestamp"]
        if timestamp is not None:
            offset = metadata.get("start_time_utc_s", 0) if metadata["timestamp_clock"] == "elapsed" else 0
            if timestamp < 0 or not 0 <= timestamp + offset <= 253402300799:
                issue("timestamp_range", "Timestamp is negative or outside the supported UTC range", index)
                row["timestamp"] = None
        for key in IMU:
            limit = policy.max_acceleration_mps2 if key.startswith("accelerometer") else policy.max_angular_rate_rad_s
            if row[key] is not None and abs(row[key]) > limit:
                issue("imu_range", f"{key} exceeds declared policy limit {limit}", index)
        bounds = {"latitude": (-90, 90), "longitude": (-180, 180), "GNSS_speed": (0, policy.max_speed_mps),
                  "GNSS_heading": (0, 360), "GNSS_accuracy": (0, float("inf"))}
        for key, (low, high) in bounds.items():
            value = row[key]
            if value is not None and (value < low or value > high or
                    (key == "GNSS_heading" and value == 360) or (key == "GNSS_accuracy" and value == 0)):
                issue("gnss_range", f"{key} outside valid range", index)
        state = raw.get("gnss_status")
        if state not in STATES:
            issue("gnss_status", f"gnss_status must be one of {STATES}", index)
            state = None
        row["gnss_status"] = state
        row["gnss_status_basis"] = "collector_annotation"
        if (row["latitude"] is None) != (row["longitude"] is None):
            issue("gnss_consistency", "Latitude and longitude must be present/missing together", index)
        if state in ("normal_gnss", "gnss_recovery") and row["latitude"] is None:
            issue("gnss_consistency", "Normal/recovery samples require coordinates", index)
        if state == "gnss_outage" and any(row[k] is not None for k in GNSS):
            issue("outage_values_retained", "Outage contains receiver values; retained but ineligible as labels", index, "warning")
        if state == "normal_gnss" and row["GNSS_accuracy"] is not None and row["GNSS_accuracy"] > policy.normal_accuracy_m:
            issue("gnss_consistency", "Normal annotation conflicts with accuracy policy", index)
        fix = raw.get("gnss_timestamp")
        if fix is not None:
            if not number(fix):
                issue("gnss_timestamp", "gnss_timestamp must be finite in the session clock/units", index)
            elif row["timestamp"] is not None:
                row["gnss_timestamp"] = fix * factors["timestamp"]
                age = row["timestamp"] - row["gnss_timestamp"]
                if age < 0 or (state == "normal_gnss" and age > policy.max_fix_age_s):
                    issue("gnss_timestamp", "Future fix or stale fix annotated normal", index)
        tags = raw.get("categories", [])
        if not isinstance(tags, list) or any(not isinstance(t, str) or t not in CATEGORIES[4:] for t in tags):
            issue("categories", "categories must be a list of supported environment/road tags", index)
            tags = []
        tags = set(tags)
        for key in ("road_type", "environment", "activity"):
            if isinstance(row[key], str) and row[key] in CATEGORIES[4:]:
                tags.add(row[key])
        if state:
            tags.add(state)
        if tags & {"tunnel", "underpass"} and metadata.get("legal_collection_confirmed") is not True:
            issue("collection_context", "Tunnel/underpass collection requires legal_collection_confirmed=true", index)
        row["categories"] = sorted(tags)
        category_samples.update(tags)
        signature = digest({k: row[k] for k in ("timestamp",) + IMU + GNSS})
        if signature in seen:
            issue("duplicate_sample", "Duplicate sensor sample", index)
        seen.add(signature)
        rows.append(row)
    times = [r["timestamp"] for r in rows]
    sampling = {}
    if len(times) == len(records) and all(t is not None for t in times) and len(times) >= 3:
        delta = [b - a for a, b in zip(times, times[1:])]
        if any(d <= 0 for d in delta):
            issue("timestamp_order", "Timestamps must be strictly increasing; samples were not sorted")
        else:
            import statistics
            observed = 1 / statistics.median(delta)
            jitter = sum(abs(d * rate - 1) > policy.max_jitter_fraction for d in delta)
            sampling = dict(observed_rate_hz=observed if number(observed) else None, declared_rate_hz=rate,
                            max_gap_s=max(delta), irregular_intervals=jitter)
            if not number(observed) or abs(observed / rate - 1) > policy.sample_rate_tolerance:
                issue("sampling_rate", "Observed median sampling rate differs from declared rate")
            if max(delta) > policy.max_gap_periods / rate:
                issue("sampling_gap", "Sensor gap exceeds policy; no interpolation was applied")
            if jitter:
                issue("sampling_jitter", f"{jitter} intervals exceed jitter tolerance", severity="warning")
            offset = metadata.get("start_time_utc_s", 0) if metadata["timestamp_clock"] == "elapsed" else 0
            for row in rows:
                row["timestamp_utc_s"] = row["timestamp"] + offset
                row["elapsed_s"] = row["timestamp"] - times[0]
            previous = None
            previous_fix_time = None
            for index, row in enumerate(rows):
                fix_time = row.get("gnss_timestamp")
                if fix_time is not None:
                    if previous_fix_time is not None and fix_time < previous_fix_time:
                        issue("gnss_timestamp", "Receiver timestamps move backwards", index)
                    previous_fix_time = fix_time
                if row["latitude"] is None or row["longitude"] is None or row["gnss_status"] == "gnss_outage":
                    continue
                if previous is not None:
                    dt = row["timestamp"] - previous["timestamp"]
                    travel = distance_m([previous[k] for k in GNSS[:2]], [row[k] for k in GNSS[:2]])
                    allowance = (row["GNSS_accuracy"] or 0) + (previous["GNSS_accuracy"] or 0)
                    if travel > policy.max_speed_mps * dt + allowance:
                        severity = "error" if row["gnss_status"] == previous["gnss_status"] == "normal_gnss" else "warning"
                        issue("gnss_jump", "Coordinate jump exceeds speed/accuracy policy", index, severity)
                previous = row
    if any(raw.get("gnss_timestamp") is None for raw in records if isinstance(raw, dict)):
        issue("fix_freshness_unknown", "Some receiver timestamps unavailable; freshness cannot be verified", severity="warning")
    for key, count in sorted(missing.items()):
        if key not in IMU and key != "timestamp":
            issue("missing_optional_data", f"{key} unavailable in {count} samples", severity="warning")
    stats = dict(samples=len(records), missing_fields=dict(missing), category_samples=dict(category_samples), **sampling)
    if rows and "elapsed_s" in rows[-1]:
        stats["duration_s"] = rows[-1]["elapsed_s"]
    accepted = not any(i["severity"] == "error" for i in issues)
    return {"accepted": accepted, "issues": issues, "statistics": stats}, rows if accepted else []
