from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import numpy as np

from navigation.interfaces import (
    AIVelocityMeasurement,
    IAIVelocityMeasurement,
)


class MotionCategory(str, Enum):
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    UNKNOWN = "UNKNOWN"


@dataclass
class MotionClassificationResult:
    category: MotionCategory
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)


class IMotionClassifier(ABC):
    @abstractmethod
    def classify_motion(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ) -> MotionClassificationResult:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class RuleBasedMotionClassifier(IMotionClassifier):
    def __init__(self, step_freq_min_hz: float = 1.0, step_freq_max_hz: float = 3.0):
        self.step_freq_min_hz = step_freq_min_hz
        self.step_freq_max_hz = step_freq_max_hz

    def classify_motion(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ) -> MotionClassificationResult:
        buf = np.asarray(imu_buffer, dtype=np.float64)
        if len(buf) < 20:
            return MotionClassificationResult(
                category=MotionCategory.UNKNOWN,
                confidence=0.0,
                probabilities={"VEHICLE": 0.33, "PEDESTRIAN": 0.33, "UNKNOWN": 0.34},
            )

        accel = buf[:, :3] if buf.shape[1] >= 3 else buf
        accel_mags = np.linalg.norm(accel, axis=1)
        var_accel = float(np.var(accel_mags))

        if var_accel > 1.5:
            cat = MotionCategory.PEDESTRIAN
            conf = min(1.0, var_accel / 5.0)
            probs = {"VEHICLE": 0.20, "PEDESTRIAN": 0.80, "UNKNOWN": 0.0}
        else:
            cat = MotionCategory.VEHICLE
            conf = 0.85
            probs = {"VEHICLE": 0.85, "PEDESTRIAN": 0.10, "UNKNOWN": 0.05}

        return MotionClassificationResult(
            category=cat,
            confidence=conf,
            probabilities=probs,
            features={"var_accel": var_accel},
        )

    def reset(self) -> None:
        pass


class IVelocityModel(ABC):
    @abstractmethod
    def estimate_velocity(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ) -> Optional[AIVelocityMeasurement]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class VehicleModel(IVelocityModel):
    def __init__(self, underlying_engine: Optional[IAIVelocityMeasurement] = None):
        self.engine = underlying_engine

    def estimate_velocity(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ) -> Optional[AIVelocityMeasurement]:
        if self.engine is None:
            return None
        return self.engine.predict(imu_buffer, dt=dt, *args, **kwargs)

    def reset(self) -> None:
        if self.engine is not None and hasattr(self.engine, "reset"):
            self.engine.reset()


class PedestrianModel(IVelocityModel):
    def __init__(self, step_length_m: float = 0.75):
        self.step_length_m = step_length_m
        self.step_count = 0

    def estimate_velocity(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ) -> Optional[AIVelocityMeasurement]:
        buf = np.asarray(imu_buffer, dtype=np.float64)
        if len(buf) < 10:
            return None

        accel = buf[:, :3] if buf.shape[1] >= 3 else buf
        accel_mags = np.linalg.norm(accel, axis=1)
        mean_speed = float(np.std(accel_mags) * 0.8)
        mean_speed = min(3.0, max(0.0, mean_speed))

        heading = float(kwargs.get("heading", 0.0))
        v_e = mean_speed * np.sin(heading)
        v_n = mean_speed * np.cos(heading)

        return AIVelocityMeasurement(
            velocity_north=float(v_n),
            velocity_east=float(v_e),
            variance_north=0.25,
            variance_east=0.25,
            latency_ms=0.5,
            is_valid=True,
        )

    def reset(self) -> None:
        self.step_count = 0


class SharedNavigationFusion:
    def __init__(
        self,
        classifier: Optional[IMotionClassifier] = None,
        vehicle_model: Optional[IVelocityModel] = None,
        pedestrian_model: Optional[IVelocityModel] = None,
    ):
        self.classifier = classifier or RuleBasedMotionClassifier()
        self.vehicle_model = vehicle_model or VehicleModel()
        self.pedestrian_model = pedestrian_model or PedestrianModel()
        self.active_mode: MotionCategory = MotionCategory.UNKNOWN

    def process_imu_window(
        self,
        imu_buffer: np.ndarray,
        dt: float = 0.1,
        forced_mode: Optional[MotionCategory] = None,
        *args,
        **kwargs,
    ) -> Tuple[MotionClassificationResult, Optional[AIVelocityMeasurement]]:
        if forced_mode is not None:
            class_res = MotionClassificationResult(
                category=forced_mode,
                confidence=1.0,
                probabilities={forced_mode.value: 1.0},
            )
        else:
            class_res = self.classifier.classify_motion(imu_buffer, dt=dt, *args, **kwargs)

        self.active_mode = class_res.category

        if self.active_mode == MotionCategory.PEDESTRIAN:
            vel_meas = self.pedestrian_model.estimate_velocity(imu_buffer, dt=dt, *args, **kwargs)
        else:
            vel_meas = self.vehicle_model.estimate_velocity(imu_buffer, dt=dt, *args, **kwargs)

        return class_res, vel_meas

    def reset(self) -> None:
        self.active_mode = MotionCategory.UNKNOWN
        self.classifier.reset()
        self.vehicle_model.reset()
        self.pedestrian_model.reset()
