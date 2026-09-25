from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import json
import math
import time

from src.data.sensor_normalization import CrossDatasetNormalizer, PreprocessingConfig, DomainType


@dataclass
class PedestrianSessionMetadata:
    participant_id: str
    session_id: str
    device_model: str
    android_version: str
    phone_placement: str
    phone_orientation: str
    environment: str
    activity: str
    ground_truth_type: str
    sample_rate_hz: float
    duration_s: float
    sample_count: int
    contains_outages: bool = False
    source_dataset: str = "Navigators India Pedestrian Dataset"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "device_model": self.device_model,
            "android_version": self.android_version,
            "phone_placement": self.phone_placement,
            "phone_orientation": self.phone_orientation,
            "environment": self.environment,
            "activity": self.activity,
            "ground_truth_type": self.ground_truth_type,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_s": self.duration_s,
            "sample_count": self.sample_count,
            "contains_outages": self.contains_outages,
            "source_dataset": self.source_dataset,
        }


@dataclass
class PedestrianQualityReport:
    session_id: str
    total_samples: int
    valid_samples: int
    corrupted_samples: int
    missing_imu_count: int
    max_gap_s: float
    avg_sample_rate_hz: float
    ground_truth_quality_score: float
    validation_status: str
    issues_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "corrupted_samples": self.corrupted_samples,
            "missing_imu_count": self.missing_imu_count,
            "max_gap_s": self.max_gap_s,
            "avg_sample_rate_hz": self.avg_sample_rate_hz,
            "ground_truth_quality_score": self.ground_truth_quality_score,
            "validation_status": self.validation_status,
            "issues_detected": self.issues_detected,
        }


class NavigatorsIndiaPedestrianPipeline:
    VALID_PLACEMENTS = {"handheld", "front_pocket", "back_pocket", "bag", "fixed"}
    VALID_ACTIVITIES = {
        "slow_walking", "normal_walking", "fast_walking", "stopping",
        "starting", "turning", "straight_walking", "stairs"
    }
    VALID_ENVIRONMENTS = {"outdoor", "urban", "open_area", "crowded_area", "indoor"}
    HIGH_QUALITY_GT_TYPES = {"optical_mocap_vicon", "rtk_gnss", "vi_slam_reference", "dual_freq_gnss"}

    def __init__(self, base_output_dir: str = "data/navigators_india_pedestrian"):
        self.base_dir = Path(base_output_dir)
        self.normalizer = CrossDatasetNormalizer(config=PreprocessingConfig.for_pedestrian())

    def validate_session(self, raw_session: Dict[str, Any]) -> PedestrianQualityReport:
        sess_id = raw_session.get("session_id", "unknown_ped_session")
        issues = []

        if "timestamp" not in raw_session or len(raw_session["timestamp"]) == 0:
            return PedestrianQualityReport(
                session_id=sess_id,
                total_samples=0,
                valid_samples=0,
                corrupted_samples=0,
                missing_imu_count=0,
                max_gap_s=0.0,
                avg_sample_rate_hz=0.0,
                ground_truth_quality_score=0.0,
                validation_status="REJECTED",
                issues_detected=["MISSING_TIMESTAMPS"],
            )

        ts = np.asarray(raw_session["timestamp"], dtype=np.float64)
        total_n = len(ts)

        acc_present = "accelerometer_x" in raw_session or "accel" in raw_session
        gyro_present = "gyroscope_x" in raw_session or "gyro" in raw_session

        missing_imu = 0
        if not acc_present:
            issues.append("MISSING_ACCELEROMETER")
            missing_imu += total_n
        if not gyro_present:
            issues.append("MISSING_GYROSCOPE")
            missing_imu += total_n

        dt = np.diff(ts)
        max_gap = float(np.max(dt)) if len(dt) > 0 else 0.0
        if max_gap > 2.0:
            issues.append(f"LARGE_TIMESTAMP_GAP_{max_gap:.2f}s")

        if len(dt) > 0 and np.any(dt <= 0):
            issues.append("NON_MONOTONIC_TIMESTAMPS")

        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        avg_hz = float(total_n / max(dur, 1e-6))

        gt_type = raw_session.get("ground_truth_type", "standard_smartphone_gnss")
        gt_score = 0.0
        if gt_type in self.HIGH_QUALITY_GT_TYPES:
            gt_score = 1.0
        elif gt_type == "standard_smartphone_gnss":
            issues.append("UNRELIABLE_SINGLE_FREQ_GNSS_GROUND_TRUTH")
            gt_score = 0.4
        else:
            gt_score = 0.5

        placement = raw_session.get("phone_placement", "unknown")
        if placement not in self.VALID_PLACEMENTS:
            issues.append(f"UNKNOWN_PLACEMENT_{placement}")

        corrupted = 0
        for k in ["accelerometer_x", "gyroscope_x", "latitude", "longitude"]:
            if k in raw_session:
                arr = np.asarray(raw_session[k])
                if not np.all(np.isfinite(arr)):
                    corrupted += int(np.sum(~np.isfinite(arr)))
                    issues.append(f"NON_FINITE_VALUES_IN_{k.upper()}")

        valid_count = max(0, total_n - corrupted)

        if missing_imu > 0 or "NON_MONOTONIC_TIMESTAMPS" in issues or corrupted > total_n * 0.1:
            status = "REJECTED"
        elif len(issues) > 0:
            status = "WARNING"
        else:
            status = "PASSED"

        return PedestrianQualityReport(
            session_id=sess_id,
            total_samples=total_n,
            valid_samples=valid_count,
            corrupted_samples=corrupted,
            missing_imu_count=missing_imu,
            max_gap_s=float(round(max_gap, 4)),
            avg_sample_rate_hz=float(round(avg_hz, 2)),
            ground_truth_quality_score=float(gt_score),
            validation_status=status,
            issues_detected=issues,
        )

    def extract_metadata(self, raw_session: Dict[str, Any]) -> PedestrianSessionMetadata:
        ts = np.asarray(raw_session.get("timestamp", [0.0]), dtype=np.float64)
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        hz = float(len(ts) / max(dur, 1e-6))

        return PedestrianSessionMetadata(
            participant_id=str(raw_session.get("participant_id", raw_session.get("user_id", "p01"))),
            session_id=str(raw_session.get("session_id", "ped_sess_01")),
            device_model=str(raw_session.get("device_model", raw_session.get("device_id", "pixel_6"))),
            android_version=str(raw_session.get("android_version", "14.0")),
            phone_placement=str(raw_session.get("phone_placement", "handheld")),
            phone_orientation=str(raw_session.get("phone_orientation", "portrait_upright")),
            environment=str(raw_session.get("environment", "urban")),
            activity=str(raw_session.get("activity", "normal_walking")),
            ground_truth_type=str(raw_session.get("ground_truth_type", "rtk_gnss")),
            sample_rate_hz=float(round(hz, 2)),
            duration_s=float(round(dur, 2)),
            sample_count=len(ts),
            contains_outages=bool(raw_session.get("contains_outages", False)),
        )

    def create_controlled_outage_subsets(
        self,
        raw_session: Dict[str, Any],
        outage_durations_s: List[int] = [10, 30, 60, 120, 300],
    ) -> List[Dict[str, Any]]:
        outage_subsets = []
        if "timestamp" not in raw_session or "latitude" not in raw_session:
            return outage_subsets

        ts = np.asarray(raw_session["timestamp"], dtype=np.float64)
        dur = ts[-1] - ts[0]

        for outage_len in outage_durations_s:
            if dur < outage_len + 10.0:
                continue

            mid_t = ts[0] + dur / 2.0
            outage_start = mid_t - outage_len / 2.0
            outage_end = mid_t + outage_len / 2.0

            sub = dict(raw_session)
            lat = np.array(raw_session["latitude"], dtype=np.float64)
            lon = np.array(raw_session["longitude"], dtype=np.float64)

            mask_outage = (ts >= outage_start) & (ts <= outage_end)
            lat_masked = lat.copy()
            lon_masked = lon.copy()
            lat_masked[mask_outage] = np.nan
            lon_masked[mask_outage] = np.nan

            sub["latitude_masked"] = lat_masked
            sub["longitude_masked"] = lon_masked
            sub["outage_duration_s"] = outage_len
            sub["outage_start_s"] = float(outage_start - ts[0])
            sub["outage_end_s"] = float(outage_end - ts[0])
            sub["session_id"] = f"{raw_session.get('session_id', 'sess')}_outage_{outage_len}s"
            sub["contains_outages"] = True

            outage_subsets.append(sub)

        return outage_subsets

    def process_and_normalize_session(
        self,
        raw_session: Dict[str, Any],
        stats: Optional[Any] = None,
    ) -> Dict[str, Any]:
        s = dict(raw_session)
        if "accel" not in s and "accelerometer_x" in s:
            s["accel"] = np.column_stack([
                s["accelerometer_x"],
                s["accelerometer_y"],
                s["accelerometer_z"],
            ])
        if "gyro" not in s and "gyroscope_x" in s:
            s["gyro"] = np.column_stack([
                s["gyroscope_x"],
                s["gyroscope_y"],
                s["gyroscope_z"],
            ])

        if stats is None:
            stats = self.normalizer.fit([s])

        res = self.normalizer.transform(s, stats=stats, sequence_id=str(s.get("session_id", "seq_01")))

        return {
            "raw_session": raw_session,
            "normalized_imu": res.normalized_data["imu_normalized"],
            "timestamps_grid": res.normalized_data["timestamps"],
            "transformation_record": res.transformation_record,
        }

    def export_session_artifact(
        self,
        raw_session: Dict[str, Any],
    ) -> Dict[str, Path]:
        sess_id = str(raw_session.get("session_id", "ped_sess_01"))

        report = self.validate_session(raw_session)
        metadata = self.extract_metadata(raw_session)
        processed = self.process_and_normalize_session(raw_session)

        paths = {
            "raw": self.base_dir / "raw_data" / f"{sess_id}_raw.json",
            "processed": self.base_dir / "processed_data" / f"{sess_id}_proc.npz",
            "metadata": self.base_dir / "metadata" / f"{sess_id}_meta.json",
            "ground_truth": self.base_dir / "ground_truth" / f"{sess_id}_gt.json",
            "quality_report": self.base_dir / "quality_report" / f"{sess_id}_qr.json",
        }

        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)

        with open(paths["metadata"], "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        with open(paths["quality_report"], "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        gt_dict = {
            "session_id": sess_id,
            "ground_truth_type": metadata.ground_truth_type,
            "quality_score": report.ground_truth_quality_score,
            "latitude": np.asarray(raw_session.get("latitude", [])).astype(float).tolist(),
            "longitude": np.asarray(raw_session.get("longitude", [])).astype(float).tolist(),
            "velocity_north": np.asarray(raw_session.get("velocity_north", [])).astype(float).tolist(),
            "velocity_east": np.asarray(raw_session.get("velocity_east", [])).astype(float).tolist(),
        }
        with open(paths["ground_truth"], "w") as f:
            json.dump(gt_dict, f, indent=2)

        np.savez_compressed(
            paths["processed"],
            normalized_imu=processed["normalized_imu"],
            timestamps_grid=processed["timestamps_grid"],
        )

        return paths
