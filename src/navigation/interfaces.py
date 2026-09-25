from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


class GNSSAnomalyType(str, Enum):
    NONE = "NONE"
    MULTIPATH = "MULTIPATH"
    SPOOFING = "SPOOFING"
    JAMMING = "JAMMING"
    STEP_DISCONTINUITY = "STEP_DISCONTINUITY"
    HIGH_DOP = "HIGH_DOP"
    OUTAGE = "OUTAGE"


class GNSSAnomalyStatus(str, Enum):
    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"
    UNKNOWN = "UNKNOWN"


class GNSSTrustState(str, Enum):
    TRUSTED = "TRUSTED"
    DEGRADED = "DEGRADED"
    SUSPICIOUS = "SUSPICIOUS"
    UNUSABLE = "UNUSABLE"


@dataclass
class GNSSTrustMetric:
    trust_score: float
    is_trusted: bool
    position_variance_scale: float
    velocity_variance_scale: float
    state: GNSSTrustState = GNSSTrustState.TRUSTED
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GNSSAnomalyReport:
    detected: bool = False
    anomaly_type: GNSSAnomalyType = GNSSAnomalyType.NONE
    confidence: float = 0.0
    severity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    status: GNSSAnomalyStatus = GNSSAnomalyStatus.NORMAL
    reason_codes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AIVelocityMeasurement:
    velocity_north: float
    velocity_east: float
    variance_north: float
    variance_east: float
    latency_ms: float
    is_valid: bool
    receptive_field_samples: int = 200
    timestamp: float = 0.0
    status: str = "VALID"
    confidence: float = 1.0
    reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveNoiseParameters:
    process_noise_scale_pos: float = 1.0
    process_noise_scale_vel: float = 1.0
    process_noise_scale_att: float = 1.0
    measurement_noise_scale_gnss_pos: float = 1.0
    measurement_noise_scale_gnss_vel: float = 1.0
    measurement_noise_scale_ai_vel: float = 1.0
    measurement_noise_scale_map: float = 1.0
    telemetry: Dict[str, Any] = field(default_factory=dict)
    downweighted_reasons: List[str] = field(default_factory=list)
    rejected_reasons: List[str] = field(default_factory=list)


@dataclass
class ConfidenceEstimate:
    horizontal_accuracy_m: float
    heading_accuracy_deg: float
    integrity_score: float
    overall_confidence: float
    dr_drift_percent: float


@dataclass
class RoadHypothesis:
    segment_id: str
    probability: float
    cross_track_distance_m: float
    along_track_distance_m: float
    heading_difference_rad: float
    snapped_point: np.ndarray


@dataclass
class MapConstraint:
    has_constraint: bool
    measurement_matrix: Optional[np.ndarray] = None
    observation_vector: Optional[np.ndarray] = None
    noise_covariance: Optional[np.ndarray] = None


@dataclass
class GNSSPrediction:
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    uncertainty_radius_m: float
    prediction_time_horizon_s: float


@dataclass
class GNSSRecoveryPlan:
    corrected_state: np.ndarray
    smoothed_covariance_diagonal: np.ndarray
    step_correction_norm_m: float
    is_converged: bool
    remaining_steps: int


class IGNSSTrustEngine(ABC):
    @abstractmethod
    def evaluate_trust(
        self,
        gnss_data: Optional[Dict[str, Any]],
        imu_accel: Optional[np.ndarray] = None,
        dt: float = 0.1,
        current_position: Optional[np.ndarray] = None,
        current_velocity: Optional[np.ndarray] = None,
        current_heading: Optional[float] = None,
        position_uncertainty: Optional[float] = None,
        ai_velocity: Optional[np.ndarray] = None,
        road_distance: Optional[float] = None,
    ) -> GNSSTrustMetric:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class IGNSSAnomalyDetector(ABC):
    @abstractmethod
    def detect_anomalies(
        self,
        gnss_data: Optional[Dict[str, Any]],
        *args,
        **kwargs,
    ) -> GNSSAnomalyReport:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class IAIVelocityMeasurement(ABC):
    @abstractmethod
    def estimate_velocity(
        self,
        imu_window: np.ndarray,
        *args,
        **kwargs,
    ) -> AIVelocityMeasurement:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass


class IAdaptiveFusionEngine(ABC):
    @abstractmethod
    def compute_adaptive_noise(
        self,
        current_mode: str = "GNSS_INS",
        innovation: Optional[np.ndarray] = None,
        motion_state: str = "MOVING",
        *args,
        **kwargs,
    ) -> AdaptiveNoiseParameters:
        pass


class IConfidenceEstimator(ABC):
    @abstractmethod
    def compute_confidence(
        self,
        covariance: np.ndarray,
        dr_duration: float,
        map_confidence: float,
    ) -> ConfidenceEstimate:
        pass


class IRoadHypothesisTracker(ABC):
    @abstractmethod
    def update_hypotheses(
        self,
        position: np.ndarray,
        heading: float,
        speed: float,
    ) -> List[RoadHypothesis]:
        pass

    @abstractmethod
    def get_best_hypothesis(self) -> Optional[RoadHypothesis]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class IMapConstraintEngine(ABC):
    @abstractmethod
    def generate_constraints(
        self,
        hypothesis: RoadHypothesis,
        state: np.ndarray,
    ) -> Optional[MapConstraint]:
        pass


class IGNSSPredictor(ABC):
    @abstractmethod
    def predict_gnss(
        self,
        last_known_state: np.ndarray,
        outage_duration: float,
    ) -> GNSSPrediction:
        pass


class IGNSSRecoveryManager(ABC):
    @abstractmethod
    def compute_recovery_step(
        self,
        current_estimate: np.ndarray,
        fresh_gnss: np.ndarray,
        recovery_step_index: int,
    ) -> GNSSRecoveryPlan:
        pass

    @abstractmethod
    def is_converged(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
