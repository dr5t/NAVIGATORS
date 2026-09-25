from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import math


class DatasetDomain(str, Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"


class DatasetProvenanceCategory(str, Enum):
    EXTERNAL = "external"
    ORIGINAL = "original"


@dataclass
class DatasetMetadata:
    dataset_name: str
    category: DatasetProvenanceCategory
    domain: DatasetDomain
    source: str
    license: str
    citation: str
    country: str
    activity: str
    sensor_types: List[str]
    sampling_rate_hz: float
    ground_truth_type: str
    coordinate_frame: str
    target_type: str
    device_information: str
    sequence_count: int
    duration_hours: float
    distance_km: float
    known_limitations: List[str]
    is_verified: bool = True


@dataclass
class UnifiedSensorRecord:
    source_dataset: str
    timestamp: Optional[float] = None
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    velocity_north: Optional[float] = None
    velocity_east: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    activity: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    phone_mount: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "accel_x": self.accel_x,
            "accel_y": self.accel_y,
            "accel_z": self.accel_z,
            "gyro_x": self.gyro_x,
            "gyro_y": self.gyro_y,
            "gyro_z": self.gyro_z,
            "mag_x": self.mag_x,
            "mag_y": self.mag_y,
            "mag_z": self.mag_z,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "velocity_north": self.velocity_north,
            "velocity_east": self.velocity_east,
            "speed": self.speed,
            "heading": self.heading,
            "activity": self.activity,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "vehicle_type": self.vehicle_type,
            "phone_mount": self.phone_mount,
            "source_dataset": self.source_dataset,
        }


PRESET_DATASETS: List[DatasetMetadata] = [
    DatasetMetadata(
        dataset_name="IO-VNBD",
        category=DatasetProvenanceCategory.EXTERNAL,
        domain=DatasetDomain.VEHICLE,
        source="Inertial Odometry Vehicle Navigation Benchmark Dataset (Oxford / Imperial)",
        license="CC-BY-4.0",
        citation="IO-VNBD Benchmark Dataset for Vehicle Inertial Navigation, 2023",
        country="UK, Nigeria, France",
        activity="driving",
        sensor_types=["accelerometer", "gyroscope", "gnss"],
        sampling_rate_hz=100.0,
        ground_truth_type="RTK-GNSS + Integrated Dead Reckoning",
        coordinate_frame="phone_body_and_enu",
        target_type="velocity_north_east",
        device_information="Multiple Android/iOS smartphones",
        sequence_count=120,
        duration_hours=58.0,
        distance_km=4400.0,
        known_limitations=["No 3-axis magnetometer in some early sub-sequences", "Tunnel outages un-annotated in sub-set B"],
        is_verified=True,
    ),
    DatasetMetadata(
        dataset_name="RoNIN",
        category=DatasetProvenanceCategory.EXTERNAL,
        domain=DatasetDomain.PEDESTRIAN,
        source="Robust Neural Inertial Navigation (SFU / Meta)",
        license="MIT",
        citation="RoNIN: ResNet/TCN/LSTM for Pedestrian Inertial Tracking, 2020",
        country="USA, Germany",
        activity="walking, running, standing",
        sensor_types=["accelerometer", "gyroscope", "magnetometer"],
        sampling_rate_hz=200.0,
        ground_truth_type="3D Vicon Motion Capture / Cartographer SLAM",
        coordinate_frame="phone_body_aligned",
        target_type="2d_position_and_velocity",
        device_information="Pixel 2, Asus Zenfone AR, Samsung Galaxy S9",
        sequence_count=270,
        duration_hours=42.0,
        distance_km=110.0,
        known_limitations=["Indoor optical motion capture restricted to 15x15m bounds for high-precision subsets"],
        is_verified=True,
    ),
    DatasetMetadata(
        dataset_name="OxIOD",
        category=DatasetProvenanceCategory.EXTERNAL,
        domain=DatasetDomain.PEDESTRIAN,
        source="Oxford Inertial Odometry Dataset (Oxford CS)",
        license="CC-BY-4.0",
        citation="OxIOD: Dataset for Deep Inertial Odometry, 2018",
        country="UK",
        activity="handheld, pocket, handbag, trolley",
        sensor_types=["accelerometer", "gyroscope", "magnetometer"],
        sampling_rate_hz=100.0,
        ground_truth_type="Vicon Optical Tracking System",
        coordinate_frame="phone_body",
        target_type="3d_position_and_attitude",
        device_information="iPhone 7, Google Nexus 5",
        sequence_count=158,
        duration_hours=14.7,
        distance_km=42.5,
        known_limitations=["Short sequence duration per trial (typically 2-10 minutes)"],
        is_verified=True,
    ),
    DatasetMetadata(
        dataset_name="Navigators India Dataset - Vehicle",
        category=DatasetProvenanceCategory.ORIGINAL,
        domain=DatasetDomain.VEHICLE,
        source="Navigators Field Collection India",
        license="Navigators SIH Proprietary",
        citation="Navigators India Dataset Version 1.0, 2026",
        country="India",
        activity="urban, highway, rural, hilly, dense_traffic, stop_and_go, parallel_roads, service_roads, underpass_tunnel",
        sensor_types=["accelerometer", "gyroscope", "magnetometer", "gnss"],
        sampling_rate_hz=100.0,
        ground_truth_type="High-Precision Dual-Frequency GNSS + Canonical Map Constraints",
        coordinate_frame="phone_body_and_enu",
        target_type="velocity_north_east_position",
        device_information="Pixel 7 Pro, iPhone 15, Samsung S23",
        sequence_count=85,
        duration_hours=24.5,
        distance_km=1250.0,
        known_limitations=["Heavy urban canyon multipath in dense city centers"],
        is_verified=True,
    ),
    DatasetMetadata(
        dataset_name="Navigators India Dataset - Pedestrian",
        category=DatasetProvenanceCategory.ORIGINAL,
        domain=DatasetDomain.PEDESTRIAN,
        source="Navigators Field Collection India",
        license="Navigators SIH Proprietary",
        citation="Navigators India Pedestrian Dataset Version 1.0, 2026",
        country="India",
        activity="walking, stair_climbing, indoor_corridor, outdoor_pathway",
        sensor_types=["accelerometer", "gyroscope", "magnetometer", "gnss"],
        sampling_rate_hz=100.0,
        ground_truth_type="PDR Step Bench + High-Accuracy Reference Waypoints",
        coordinate_frame="phone_body_and_enu",
        target_type="step_count_stride_heading",
        device_information="Pixel 7 Pro, iPhone 15",
        sequence_count=40,
        duration_hours=12.0,
        distance_km=48.0,
        known_limitations=["GPS outage in deep indoor basements"],
        is_verified=True,
    ),
    DatasetMetadata(
        dataset_name="I2WDD",
        category=DatasetProvenanceCategory.EXTERNAL,
        domain=DatasetDomain.VEHICLE,
        source="Indian Driving Dataset (Video / Vision)",
        license="Research License",
        citation="I2WDD Vision Dataset, 2021",
        country="India",
        activity="driving",
        sensor_types=["monocular_camera", "stereo_camera"],
        sampling_rate_hz=30.0,
        ground_truth_type="Bounding Box Bounding Annotations",
        coordinate_frame="camera_frame",
        target_type="bounding_boxes",
        device_information="Dashcam Camera",
        sequence_count=0,
        duration_hours=0.0,
        distance_km=0.0,
        known_limitations=["QUARANTINED: Driving video dataset without synchronized 6-axis IMU/GNSS sensor streams"],
        is_verified=False,
    ),
]


class DatasetRegistry:
    def __init__(self):
        self._registry: Dict[str, DatasetMetadata] = {}
        for ds in PRESET_DATASETS:
            self.register_dataset(ds)

    def register_dataset(self, metadata: DatasetMetadata) -> None:
        self._registry[metadata.dataset_name] = metadata

    def get_dataset(self, dataset_name: str) -> Optional[DatasetMetadata]:
        return self._registry.get(dataset_name)

    def list_datasets(
        self,
        category: Optional[DatasetProvenanceCategory] = None,
        domain: Optional[DatasetDomain] = None,
        is_verified_only: bool = True,
    ) -> List[DatasetMetadata]:
        result = []
        for ds in self._registry.values():
            if is_verified_only and not ds.is_verified:
                continue
            if category is not None and ds.category != category:
                continue
            if domain is not None and ds.domain != domain:
                continue
            result.append(ds)
        return result

    @staticmethod
    def map_to_unified_record(raw_dict: Dict[str, Any], source_dataset: str) -> UnifiedSensorRecord:
        def get_val(keys: List[str]) -> Optional[float]:
            for k in keys:
                if k in raw_dict and raw_dict[k] is not None:
                    try:
                        val = float(raw_dict[k])
                        if math.isfinite(val):
                            return val
                    except (ValueError, TypeError):
                        pass
            return None

        def get_str(keys: List[str]) -> Optional[str]:
            for k in keys:
                if k in raw_dict and raw_dict[k] is not None:
                    return str(raw_dict[k])
            return None

        return UnifiedSensorRecord(
            source_dataset=source_dataset,
            timestamp=get_val(["timestamp", "t", "time", "ts"]),
            accel_x=get_val(["accel_x", "accelerometer_x", "ax"]),
            accel_y=get_val(["accel_y", "accelerometer_y", "ay"]),
            accel_z=get_val(["accel_z", "accelerometer_z", "az"]),
            gyro_x=get_val(["gyro_x", "gyroscope_x", "gx"]),
            gyro_y=get_val(["gyro_y", "gyroscope_y", "gy"]),
            gyro_z=get_val(["gyro_z", "gyroscope_z", "gz"]),
            mag_x=get_val(["mag_x", "magnetometer_x", "mx"]),
            mag_y=get_val(["mag_y", "magnetometer_y", "my"]),
            mag_z=get_val(["mag_z", "magnetometer_z", "mz"]),
            latitude=get_val(["latitude", "lat"]),
            longitude=get_val(["longitude", "lon", "lng"]),
            velocity_north=get_val(["velocity_north", "v_north", "vn"]),
            velocity_east=get_val(["velocity_east", "v_east", "ve"]),
            speed=get_val(["speed", "GNSS_speed", "spd"]),
            heading=get_val(["heading", "GNSS_heading", "bearing"]),
            activity=get_str(["activity", "activity_type", "motion_type"]),
            device_id=get_str(["device_id", "phone_id"]),
            session_id=get_str(["session_id", "recording_id", "trial_id"]),
            vehicle_type=get_str(["vehicle_type", "mode"]),
            phone_mount=get_str(["phone_mount", "phone_mount_position", "placement"]),
        )

    @staticmethod
    def validate_record_schema(record: UnifiedSensorRecord) -> Tuple[bool, List[str]]:
        errors = []
        if record.timestamp is None:
            errors.append("MISSING_TIMESTAMP")
        has_imu = (
            record.accel_x is not None and record.accel_y is not None and record.accel_z is not None and
            record.gyro_x is not None and record.gyro_y is not None and record.gyro_z is not None
        )
        if not has_imu:
            errors.append("INCOMPLETE_6AXIS_IMU")
        if not record.source_dataset:
            errors.append("MISSING_SOURCE_DATASET")
        return len(errors) == 0, errors
