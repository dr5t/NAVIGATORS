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
    velocity_uncertainty_mps: float = 0.0
    vertical_accuracy_m: float = 0.0
    gnss_reliability: Dict[str, Any] = field(default_factory=dict)
    ai_velocity_reliability: Dict[str, Any] = field(default_factory=dict)
    map_match_confidence: Dict[str, Any] = field(default_factory=dict)
    confidence_object: Dict[str, Any] = field(default_factory=dict)
    integrity_status: str = "NOMINAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainties": {
                "position_horizontal_m": float(self.horizontal_accuracy_m),
                "position_vertical_m": float(self.vertical_accuracy_m),
                "velocity_horizontal_mps": float(self.velocity_uncertainty_mps),
                "heading_deg": float(self.heading_accuracy_deg),
            },
            "sensors": {
                "gnss": self.gnss_reliability,
                "ai_velocity": self.ai_velocity_reliability,
                "map_matching": self.map_match_confidence,
            },
            "integrity": {
                "status": self.integrity_status,
                "score": float(self.integrity_score),
                "overall_confidence": float(self.overall_confidence),
                "dr_drift_percent": float(self.dr_drift_percent),
            },
            "raw": self.confidence_object,
        }


class RoadAmbiguityState(str, Enum):
    CONVERGED = "CONVERGED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"


@dataclass
class RoadHypothesis:
    segment_id: str
    probability: float
    cross_track_distance_m: float
    along_track_distance_m: float
    heading_difference_rad: float
    snapped_point: np.ndarray
    geometric_compatibility: float = 1.0
    heading_compatibility: float = 1.0
    velocity_compatibility: float = 1.0
    connectivity_compatibility: float = 1.0
    combined_likelihood: float = 1.0
    speed_limit: float = 50.0
    one_way: bool = False
    name: str = ""
    age_steps: int = 1


@dataclass
class MapConstraint:
    has_constraint: bool
    measurement_matrix: Optional[np.ndarray] = None
    observation_vector: Optional[np.ndarray] = None
    noise_covariance: Optional[np.ndarray] = None
    snapped_position: Optional[np.ndarray] = None
    road_heading: Optional[float] = None
    cross_track_error_m: float = 0.0
    along_track_error_m: float = 0.0
    confidence: float = 0.0
    segment_id: Optional[str] = None
    heading_constraint_applied: bool = False
    position_constraint_applied: bool = False
    applied_variance_scale: float = 1.0
    normal_vector: Optional[np.ndarray] = None
    tangent_vector: Optional[np.ndarray] = None



@dataclass
class GNSSPrediction:
    predicted_position: np.ndarray
    predicted_velocity: np.ndarray
    uncertainty_radius_m: float
    prediction_time_horizon_s: float


class GNSSRecoveryState(str, Enum):
    REACQUIRING = "REACQUIRING"
    VALIDATING = "VALIDATING"
    RECOVERING = "RECOVERING"
    NORMAL = "NORMAL"


@dataclass
class GNSSRecoveryPlan:
    corrected_state: np.ndarray
    smoothed_covariance_diagonal: np.ndarray
    step_correction_norm_m: float
    is_converged: bool
    remaining_steps: int
    state: GNSSRecoveryState = GNSSRecoveryState.RECOVERING
    discontinuity_m: float = 0.0
    trust_score: float = 1.0
    consistency_passed: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)



class GNSSDegradationRisk(str, Enum):
    NORMAL = "NORMAL"
    DEGRADING = "DEGRADING"
    HIGH_RISK = "HIGH_RISK"
    UNPREDICTABLE = "UNPREDICTABLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class GNSSDegradationPrediction:
    risk_level: GNSSDegradationRisk
    confidence: float
    predicted_outage_probability: float
    evidence_score: float
    signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    lead_time_estimate_s: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)



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
        *args,
        **kwargs,
    ) -> ConfidenceEstimate:
        pass


class IRoadHypothesisTracker(ABC):
    @abstractmethod
    def update_hypotheses(
        self,
        position: np.ndarray,
        heading: float,
        speed: float,
        *args,
        **kwargs,
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
        hypothesis: Optional[RoadHypothesis],
        state: np.ndarray,
        *args,
        **kwargs,
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
        *args,
        **kwargs,
    ) -> GNSSRecoveryPlan:
        pass


    @abstractmethod
    def is_converged(self) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class IGNSSDegradationPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        gnss_data: Optional[Dict[str, Any]],
        navigation_state: Optional[np.ndarray] = None,
        ai_velocity: Optional[np.ndarray] = None,
        map_context: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> GNSSDegradationPrediction:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

