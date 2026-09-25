from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import math
import time

from src.models.motion_classifier import (
    MotionClass,
    MotionClassifierConfig,
    TemporallyStabilizedMotionClassifier,
    MotionRouter,
)
from src.models.motion_classifier import (
    MotionClass,
    MotionClassifierConfig,
    TemporallyStabilizedMotionClassifier,
    MotionRouter,
)
from src.navigation.ekf import ExtendedKalmanFilter, NavigationMode
from src.navigation.adaptive_fusion import AdaptiveFusionEngine, AdaptiveFusionConfig
from src.navigation.map_constraint import MapConstraintEngine, MapConstraintConfig


@dataclass
class MultiModalFusionConfig:
    confidence_threshold: float = 0.65
    hysteresis_dwell_count: int = 5
    vehicle_base_std_mps: float = 0.8
    pedestrian_base_std_mps: float = 0.3
    stationary_std_mps: float = 0.05
    unknown_std_mps: float = 100.0
    innovation_gate_sigma: float = 4.0
    max_allowed_gnss_accuracy_m: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "hysteresis_dwell_count": self.hysteresis_dwell_count,
            "vehicle_base_std_mps": self.vehicle_base_std_mps,
            "pedestrian_base_std_mps": self.pedestrian_base_std_mps,
            "stationary_std_mps": self.stationary_std_mps,
            "unknown_std_mps": self.unknown_std_mps,
            "innovation_gate_sigma": self.innovation_gate_sigma,
            "max_allowed_gnss_accuracy_m": self.max_allowed_gnss_accuracy_m,
        }


@dataclass
class MultiModalFusionState:
    timestamp: float
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    motion_state: MotionClass
    classifier_confidence: float
    active_model_name: str
    gnss_status: str
    is_dead_reckoning: bool
    internet_connected: bool
    R_motion_cov: np.ndarray
    R_gnss_cov: Optional[np.ndarray] = None
    rejected_reasons: List[str] = field(default_factory=list)


class MultiModalAdaptiveFusionEngine:
    def __init__(
        self,
        config: Optional[MultiModalFusionConfig] = None,
        vehicle_model: Optional[Any] = None,
        pedestrian_model: Optional[Any] = None,
        ekf: Optional[ExtendedKalmanFilter] = None,
        map_engine: Optional[Any] = None,
        road_network: Optional[RoadNetwork] = None,
    ):
        self.config = config or MultiModalFusionConfig()

        classifier_cfg = MotionClassifierConfig(
            confidence_threshold=self.config.confidence_threshold,
            hysteresis_dwell_count=self.config.hysteresis_dwell_count,
        )
        self.classifier = TemporallyStabilizedMotionClassifier(config=classifier_cfg)
        self.router = MotionRouter(
            classifier=self.classifier,
            vehicle_model=vehicle_model,
            pedestrian_model=pedestrian_model,
        )

        self.ekf = ekf or ExtendedKalmanFilter()
        self.adaptive_fusion = AdaptiveFusionEngine(config=AdaptiveFusionConfig())
        
        if map_engine is not None:
            self.map_engine = map_engine
        elif road_network is not None:
            self.map_engine = MapConstraintEngine(road_network=road_network, config=MapConstraintConfig())
        else:
            self.map_engine = None

        self.last_timestamp: float = 0.0
        self.is_dead_reckoning: bool = False

    def reset(self):
        self.classifier.reset()
        self.ekf = ExtendedKalmanFilter()
        self.adaptive_fusion.reset()
        if self.map_engine is not None and hasattr(self.map_engine, "reset"):
            self.map_engine.reset()
        self.last_timestamp = 0.0
        self.is_dead_reckoning = False

    def process_sensor_frame(
        self,
        imu_window: np.ndarray,
        timestamp: float,
        gnss_fix: Optional[Dict[str, Any]] = None,
        map_constraint: Optional[Dict[str, Any]] = None,
        internet_connected: bool = True,
    ) -> MultiModalFusionState:
        dt = float(timestamp - self.last_timestamp) if self.last_timestamp > 0.0 else 0.1
        dt = max(0.001, min(dt, 2.0))
        self.last_timestamp = timestamp
        self.ekf.dt = dt

        pred_vel, motion_state, conf, info = self.router.route_and_predict(imu_window)

        if motion_state == MotionClass.VEHICLE:
            active_model = "VehicleTCNModel"
            base_std = self.config.vehicle_base_std_mps
        elif motion_state == MotionClass.PEDESTRIAN:
            active_model = "PedestrianTCNModel"
            base_std = self.config.pedestrian_base_std_mps
        elif motion_state == MotionClass.STATIONARY:
            active_model = "StationaryZUPT"
            base_std = self.config.stationary_std_mps
        else:
            active_model = "FallbackNeutralState"
            base_std = self.config.unknown_std_mps

        if conf < self.config.confidence_threshold:
            base_std *= 10.0

        cov_diag = float(base_std ** 2)
        R_motion = np.array([[cov_diag, 0.0], [0.0, cov_diag]], dtype=np.float64)

        current_ekf_vel = self.ekf.x[3:5]
        vel_innov = np.linalg.norm(pred_vel - current_ekf_vel)
        max_allowed_innov = self.config.innovation_gate_sigma * base_std

        rejected_reasons = []

        if vel_innov > max_allowed_innov and motion_state != MotionClass.STATIONARY:
            R_motion *= 100.0
            rejected_reasons.append(f"AI Velocity innovation gated ({vel_innov:.2f} > {max_allowed_innov:.2f} m/s)")

        accel_body = imu_window[-1, :3] if len(imu_window) > 0 else np.array([0.0, 0.0, 9.81], dtype=np.float64)
        gyro_body = imu_window[-1, 3:6] if len(imu_window) > 0 else np.zeros(3, dtype=np.float64)

        ai_pass = pred_vel if (motion_state != MotionClass.UNKNOWN and conf >= self.config.confidence_threshold) else None
        self.ekf.predict(accel_body=accel_body, gyro_body=gyro_body, ai_velocity=ai_pass)

        gnss_valid = False
        gnss_status = "UNAVAILABLE"
        R_gnss = None

        if gnss_fix is not None and "latitude" in gnss_fix and "longitude" in gnss_fix:
            lat = float(gnss_fix["latitude"])
            lon = float(gnss_fix["longitude"])
            acc_m = float(gnss_fix.get("accuracy", gnss_fix.get("GNSS_accuracy", 10.0)))

            if np.isfinite(lat) and np.isfinite(lon) and acc_m <= self.config.max_allowed_gnss_accuracy_m:
                gnss_valid = True
                gnss_status = "HEALTHY"
                self.is_dead_reckoning = False

                r_gnss_var = float(max(1.0, acc_m) ** 2)
                R_gnss = np.array([[r_gnss_var, 0.0], [0.0, r_gnss_var]], dtype=np.float64)

                lat0 = math.radians(lat)
                east = (math.radians(lon) - math.radians(77.2090)) * math.cos(lat0) * 6371000.0
                north = math.radians(lat - 28.6139) * 6371000.0
                pos_arr = np.array([east, north], dtype=np.float64)
                vel_arr = np.array([float(gnss_fix.get("velocity_east", 0.0)), float(gnss_fix.get("velocity_north", 0.0))], dtype=np.float64)

                self.ekf.update_gnss(gnss_position=pos_arr, gnss_velocity=vel_arr, timestamp=timestamp)
            else:
                gnss_status = "DEGRADED"
                rejected_reasons.append(f"GNSS fix rejected: accuracy {acc_m:.1f}m exceeds limit")

        if not gnss_valid:
            self.is_dead_reckoning = True
            gnss_status = "OUTAGE"

        if map_constraint is not None and self.map_engine is not None:
            if hasattr(self.map_engine, "update"):
                self.map_engine.update(map_constraint)

        pos_enu = self.ekf.x[0:3].copy()
        vel_enu = self.ekf.x[3:6].copy()

        return MultiModalFusionState(
            timestamp=timestamp,
            position_enu=pos_enu,
            velocity_enu=vel_enu,
            motion_state=motion_state,
            classifier_confidence=conf,
            active_model_name=active_model,
            gnss_status=gnss_status,
            is_dead_reckoning=self.is_dead_reckoning,
            internet_connected=internet_connected,
            R_motion_cov=R_motion,
            R_gnss_cov=R_gnss,
            rejected_reasons=rejected_reasons,
        )
