from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import math

from src.models.motion_classifier import MotionClass
from src.navigation.interfaces import GNSSTrustState, GNSSTrustMetric
from src.navigation.gnss_trust import GNSSTrustEngine, GNSSTrustConfig
from src.navigation.gnss_anomaly import GNSSAnomalyDetector, GNSSAnomalyDetectorConfig
from src.navigation.confidence import ConfidenceEstimator, ConfidenceConfig, ConfidenceEstimate
from src.navigation.gnss_recovery import ValidatedGNSSRecoveryManager, GNSSRecoveryConfig, GNSSRecoveryState
from src.navigation.multimodal_fusion import MultiModalAdaptiveFusionEngine, MultiModalFusionConfig, MultiModalFusionState


class NavigationStateMode(str, Enum):
    NORMAL = "NORMAL"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    HYBRID = "HYBRID"
    DR = "DR"
    REACQUIRING = "REACQUIRING"
    RECOVERED = "RECOVERED"


@dataclass
class ConfidenceAwareConfig:
    max_position_jump_m: float = 30.0
    max_velocity_inconsistency_mps: float = 3.0
    max_heading_inconsistency_deg: float = 30.0
    recovery_min_good_fixes: int = 3
    confidence_threshold: float = 0.65

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_position_jump_m": self.max_position_jump_m,
            "max_velocity_inconsistency_mps": self.max_velocity_inconsistency_mps,
            "max_heading_inconsistency_deg": self.max_heading_inconsistency_deg,
            "recovery_min_good_fixes": self.recovery_min_good_fixes,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class VehicleDomainResult:
    domain: str = "VEHICLE"
    velocity_mae_mps: float = 0.0
    velocity_rmse_mps: float = 0.0
    trajectory_error_m: float = 0.0
    outage_drift_pct: float = 0.0
    map_matching_accuracy_pct: float = 0.0


@dataclass
class PedestrianDomainResult:
    domain: str = "PEDESTRIAN"
    velocity_mae_mps: float = 0.0
    velocity_rmse_mps: float = 0.0
    trajectory_error_m: float = 0.0
    ate_rmse_m: float = 0.0
    fde_m: float = 0.0
    heading_error_deg: float = 0.0


@dataclass
class ConfidenceAwareStepResult:
    timestamp: float
    navigation_mode: NavigationStateMode
    gnss_trust_state: GNSSTrustState
    gnss_available: bool
    gnss_trustworthy: bool
    motion_state: MotionClass
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    confidence_score: float
    anomalies_detected: List[str] = field(default_factory=list)


class ConfidenceAwareNavigationEngine:
    def __init__(
        self,
        config: Optional[ConfidenceAwareConfig] = None,
        fusion_engine: Optional[MultiModalAdaptiveFusionEngine] = None,
    ):
        self.config = config or ConfidenceAwareConfig()
        self.fusion = fusion_engine or MultiModalAdaptiveFusionEngine()

        self.trust_engine = GNSSTrustEngine(config=GNSSTrustConfig())
        self.anomaly_detector = GNSSAnomalyDetector(config=GNSSAnomalyDetectorConfig())
        self.confidence_estimator = ConfidenceEstimator(config=ConfidenceConfig())
        self.recovery_manager = ValidatedGNSSRecoveryManager(config=GNSSRecoveryConfig())

        self.navigation_mode = NavigationStateMode.NORMAL
        self.last_gnss_fix: Optional[Dict[str, Any]] = None
        self.consecutive_good_fixes: int = 0
        self.last_timestamp: float = 0.0

    def reset(self):
        self.fusion.reset()
        self.trust_engine.reset()
        self.anomaly_detector.reset()
        if hasattr(self.confidence_estimator, "reset"):
            getattr(self.confidence_estimator, "reset")()
        self.recovery_manager.reset()
        self.navigation_mode = NavigationStateMode.NORMAL
        self.last_gnss_fix = None
        self.consecutive_good_fixes = 0
        self.last_timestamp = 0.0

    def evaluate_gnss_trustworthiness(
        self,
        gnss_fix: Optional[Dict[str, Any]],
        current_vel: np.ndarray,
        current_pos: np.ndarray,
    ) -> Tuple[bool, bool, GNSSTrustState, List[str]]:
        if gnss_fix is None:
            return False, False, GNSSTrustState.UNUSABLE, ["NO_GNSS_HARDWARE_SIGNAL"]

        gnss_available = True
        anomalies = []

        lat = float(gnss_fix.get("latitude", 0.0))
        lon = float(gnss_fix.get("longitude", 0.0))
        acc_m = float(gnss_fix.get("accuracy", gnss_fix.get("GNSS_accuracy", 10.0)))

        lat0 = math.radians(lat)
        east = (math.radians(lon) - math.radians(77.2090)) * math.cos(lat0) * 6371000.0
        north = math.radians(lat - 28.6139) * 6371000.0
        fix_pos = np.array([east, north, 0.0])

        if self.last_gnss_fix is not None:
            prev_lat = float(self.last_gnss_fix.get("latitude", lat))
            prev_lon = float(self.last_gnss_fix.get("longitude", lon))
            prev_lat0 = math.radians(prev_lat)
            prev_east = (math.radians(prev_lon) - math.radians(77.2090)) * math.cos(prev_lat0) * 6371000.0
            prev_north = math.radians(prev_lat - 28.6139) * 6371000.0
            pos_jump = float(np.linalg.norm(fix_pos[:2] - np.array([prev_east, prev_north])))

            if pos_jump > self.config.max_position_jump_m:
                anomalies.append(f"POSITION_JUMP_{pos_jump:.1f}m")

        vn = float(gnss_fix.get("velocity_north", 0.0))
        ve = float(gnss_fix.get("velocity_east", 0.0))
        gnss_vel = np.array([vn, ve])
        vel_diff = float(np.linalg.norm(gnss_vel - current_vel[:2]))

        if vel_diff > self.config.max_velocity_inconsistency_mps:
            anomalies.append(f"VELOCITY_INCONSISTENCY_{vel_diff:.1f}mps")

        if np.linalg.norm(gnss_vel) > 0.5 and np.linalg.norm(current_vel[:2]) > 0.5:
            gnss_head = np.degrees(np.arctan2(ve, vn))
            curr_head = np.degrees(np.arctan2(current_vel[1], current_vel[0]))
            head_diff = float((gnss_head - curr_head + 180.0) % 360.0 - 180.0)

            if abs(head_diff) > self.config.max_heading_inconsistency_deg:
                anomalies.append(f"HEADING_INCONSISTENCY_{abs(head_diff):.1f}deg")

        if acc_m > 30.0:
            anomalies.append(f"HIGH_ACCURACY_DILUTION_{acc_m:.1f}m")

        if not anomalies and acc_m <= 10.0:
            trust_state = GNSSTrustState.TRUSTED
            gnss_trustworthy = True
        elif len(anomalies) == 1 and acc_m <= 25.0:
            trust_state = GNSSTrustState.DEGRADED
            gnss_trustworthy = True
        elif len(anomalies) <= 2:
            trust_state = GNSSTrustState.SUSPICIOUS
            gnss_trustworthy = False
        else:
            trust_state = GNSSTrustState.UNUSABLE
            gnss_trustworthy = False

        self.last_gnss_fix = dict(gnss_fix)
        return gnss_available, gnss_trustworthy, trust_state, anomalies

    def process_step(
        self,
        imu_window: np.ndarray,
        timestamp: float,
        gnss_fix: Optional[Dict[str, Any]] = None,
        map_constraint: Optional[Dict[str, Any]] = None,
        internet_connected: bool = True,
    ) -> ConfidenceAwareStepResult:
        fusion_state = self.fusion.process_sensor_frame(
            imu_window=imu_window,
            timestamp=timestamp,
            gnss_fix=gnss_fix,
            map_constraint=map_constraint,
            internet_connected=internet_connected,
        )

        avail, trust, trust_state, anomalies = self.evaluate_gnss_trustworthiness(
            gnss_fix,
            current_vel=fusion_state.velocity_enu,
            current_pos=fusion_state.position_enu,
        )

        if avail and trust:
            self.consecutive_good_fixes += 1
        else:
            self.consecutive_good_fixes = 0

        if not avail or trust_state == GNSSTrustState.UNUSABLE:
            self.navigation_mode = NavigationStateMode.DR
        elif trust_state == GNSSTrustState.SUSPICIOUS:
            self.navigation_mode = NavigationStateMode.GNSS_DEGRADED
        elif self.navigation_mode in [NavigationStateMode.DR, NavigationStateMode.REACQUIRING]:
            if self.consecutive_good_fixes >= self.config.recovery_min_good_fixes:
                self.navigation_mode = NavigationStateMode.RECOVERED
            else:
                self.navigation_mode = NavigationStateMode.REACQUIRING
        elif trust_state == GNSSTrustState.DEGRADED:
            self.navigation_mode = NavigationStateMode.HYBRID
        else:
            self.navigation_mode = NavigationStateMode.NORMAL

        conf_score = float(fusion_state.classifier_confidence)
        if trust_state == GNSSTrustState.DEGRADED:
            conf_score *= 0.8
        elif trust_state == GNSSTrustState.SUSPICIOUS:
            conf_score *= 0.5
        elif trust_state == GNSSTrustState.UNUSABLE:
            conf_score *= 0.2

        return ConfidenceAwareStepResult(
            timestamp=timestamp,
            navigation_mode=self.navigation_mode,
            gnss_trust_state=trust_state,
            gnss_available=avail,
            gnss_trustworthy=trust,
            motion_state=fusion_state.motion_state,
            position_enu=fusion_state.position_enu.copy(),
            velocity_enu=fusion_state.velocity_enu.copy(),
            confidence_score=float(round(conf_score, 4)),
            anomalies_detected=anomalies + fusion_state.rejected_reasons,
        )


class DomainSeparatedEvaluationSuite:
    @staticmethod
    def evaluate_vehicle_domain(
        preds: np.ndarray,
        targets: np.ndarray,
        outage_drifts_m: List[float],
        map_match_acc: float,
    ) -> VehicleDomainResult:
        if len(preds) == 0:
            return VehicleDomainResult()

        errs = preds - targets
        vel_mae = float(np.mean(np.linalg.norm(errs, axis=1)))
        vel_rmse = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))

        dt = 0.1
        est_pos = np.cumsum(preds * dt, axis=0)
        true_pos = np.cumsum(targets * dt, axis=0)
        traj_err = float(np.mean(np.linalg.norm(est_pos - true_pos, axis=1)))
        mean_drift = float(np.mean(outage_drifts_m)) if outage_drifts_m else 0.0

        return VehicleDomainResult(
            domain="VEHICLE",
            velocity_mae_mps=float(round(vel_mae, 4)),
            velocity_rmse_mps=float(round(vel_rmse, 4)),
            trajectory_error_m=float(round(traj_err, 4)),
            outage_drift_pct=float(round(mean_drift, 2)),
            map_matching_accuracy_pct=float(round(map_match_acc, 2)),
        )

    @staticmethod
    def evaluate_pedestrian_domain(
        preds: np.ndarray,
        targets: np.ndarray,
    ) -> PedestrianDomainResult:
        if len(preds) == 0:
            return PedestrianDomainResult()

        errs = preds - targets
        vel_mae = float(np.mean(np.linalg.norm(errs, axis=1)))
        vel_rmse = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))

        dt = 0.1
        est_pos = np.cumsum(preds * dt, axis=0)
        true_pos = np.cumsum(targets * dt, axis=0)

        pos_diffs = np.linalg.norm(est_pos - true_pos, axis=1)
        traj_err = float(np.mean(pos_diffs))
        ate_rmse = float(np.sqrt(np.mean(pos_diffs ** 2)))
        fde = float(np.linalg.norm(est_pos[-1] - true_pos[-1]))

        pred_heads = np.degrees(np.arctan2(preds[:, 1], preds[:, 0]))
        true_heads = np.degrees(np.arctan2(targets[:, 1], targets[:, 0]))
        diff_deg = (pred_heads - true_heads + 180.0) % 360.0 - 180.0
        head_err = float(np.mean(np.abs(diff_deg)))

        return PedestrianDomainResult(
            domain="PEDESTRIAN",
            velocity_mae_mps=float(round(vel_mae, 4)),
            velocity_rmse_mps=float(round(vel_rmse, 4)),
            trajectory_error_m=float(round(traj_err, 4)),
            ate_rmse_m=float(round(ate_rmse, 4)),
            fde_m=float(round(fde, 4)),
            heading_error_deg=float(round(head_err, 4)),
        )
