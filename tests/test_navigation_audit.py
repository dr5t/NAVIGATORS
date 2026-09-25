import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.interfaces import (
    GNSSAnomalyType,
    GNSSTrustMetric,
    GNSSAnomalyReport,
    AIVelocityMeasurement,
    AdaptiveNoiseParameters,
    ConfidenceEstimate,
    RoadHypothesis,
    MapConstraint,
    GNSSPrediction,
    GNSSRecoveryPlan,
    IGNSSTrustEngine,
    IGNSSAnomalyDetector,
    IAIVelocityMeasurement,
    IAdaptiveFusionEngine,
    IConfidenceEstimator,
    IRoadHypothesisTracker,
    IMapConstraintEngine,
    IGNSSPredictor,
    IGNSSRecoveryManager,
)
from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from navigation.nav_state import NavigationStateEngine, NavState
from navigation.dead_reckoning import DeadReckoningEngine
from navigation.alignment import PhoneVehicleAligner
from navigation.nhc import NonHolonomicConstraints
from navigation.zupt import ZUPTDetector
from navigation.map_matching import RoadNetwork, RoadSegment, GeometricMapMatcher, HMMMapMatcher
from data.sensor_adapter import SmartphoneIMUAdapter, ExternalFOGIMUAdapter, SensorPacket
from evaluation.preprocessing import CausalFilter


class ConcreteTrustEngine(IGNSSTrustEngine):
    def evaluate_trust(self, gnss_data, imu_accel, dt):
        return GNSSTrustMetric(
            trust_score=0.95,
            is_trusted=True,
            position_variance_scale=1.0,
            velocity_variance_scale=1.0,
            diagnostics={"sats": 8},
        )

    def reset(self):
        pass


class ConcreteAnomalyDetector(IGNSSAnomalyDetector):
    def detect_anomalies(self, gnss_data, predicted_state, innovation):
        return GNSSAnomalyReport(
            detected=False,
            anomaly_type=GNSSAnomalyType.NONE,
            confidence=0.99,
            severity=0.0,
            details={},
        )

    def reset(self):
        pass


class ConcreteAIVelocity(IAIVelocityMeasurement):
    def estimate_velocity(self, imu_window):
        return AIVelocityMeasurement(
            velocity_north=5.0,
            velocity_east=0.0,
            variance_north=0.09,
            variance_east=0.09,
            latency_ms=4.2,
            is_valid=True,
        )

    def is_ready(self):
        return True


class ConcreteAdaptiveFusion(IAdaptiveFusionEngine):
    def compute_adaptive_noise(self, current_mode, innovation, motion_state):
        return AdaptiveNoiseParameters(
            process_noise_scale_pos=1.0,
            process_noise_scale_vel=1.0,
            process_noise_scale_att=1.0,
            measurement_noise_scale_gnss_pos=1.0,
            measurement_noise_scale_gnss_vel=1.0,
            measurement_noise_scale_ai_vel=1.0,
        )


class ConcreteConfidenceEstimator(IConfidenceEstimator):
    def compute_confidence(self, covariance, dr_duration, map_confidence):
        return ConfidenceEstimate(
            horizontal_accuracy_m=2.5,
            heading_accuracy_deg=1.2,
            integrity_score=0.98,
            overall_confidence=0.95,
            dr_drift_percent=0.5,
        )


class ConcreteRoadHypothesisTracker(IRoadHypothesisTracker):
    def update_hypotheses(self, position, heading, speed):
        return [
            RoadHypothesis(
                segment_id="road_1_0",
                probability=0.85,
                cross_track_distance_m=1.2,
                along_track_distance_m=50.0,
                heading_difference_rad=0.05,
                snapped_point=np.array([position[0], 0.0]),
            )
        ]

    def get_best_hypothesis(self):
        return RoadHypothesis(
            segment_id="road_1_0",
            probability=0.85,
            cross_track_distance_m=1.2,
            along_track_distance_m=50.0,
            heading_difference_rad=0.05,
            snapped_point=np.array([10.0, 0.0]),
        )

    def reset(self):
        pass


class ConcreteMapConstraintEngine(IMapConstraintEngine):
    def generate_constraints(self, hypothesis, state):
        return MapConstraint(
            has_constraint=True,
            measurement_matrix=np.zeros((1, 15)),
            observation_vector=np.zeros(1),
            noise_covariance=np.eye(1),
        )


class ConcreteGNSSPredictor(IGNSSPredictor):
    def predict_gnss(self, last_known_state, outage_duration):
        return GNSSPrediction(
            predicted_position=last_known_state[:3] + last_known_state[3:6] * outage_duration,
            predicted_velocity=last_known_state[3:6],
            uncertainty_radius_m=2.5 + 0.1 * outage_duration,
            prediction_time_horizon_s=outage_duration,
        )


class ConcreteGNSSRecoveryManager(IGNSSRecoveryManager):
    def __init__(self):
        self._converged = False

    def compute_recovery_step(self, current_estimate, fresh_gnss, recovery_step_index):
        diff = fresh_gnss[:2] - current_estimate[:2]
        norm = float(np.linalg.norm(diff[:2]))
        step = min(2.0, norm)
        corrected = current_estimate.copy()
        if norm > 1e-6:
            corrected[:2] += diff[:2] * (step / norm)
        self._converged = (norm <= 2.0)
        return GNSSRecoveryPlan(
            corrected_state=corrected,
            smoothed_covariance_diagonal=np.ones(15),
            step_correction_norm_m=step,
            is_converged=self._converged,
            remaining_steps=max(0, 5 - recovery_step_index),
        )

    def is_converged(self):
        return self._converged

    def reset(self):
        self._converged = False


def test_interface_contracts_instantiation():
    trust_engine = ConcreteTrustEngine()
    trust = trust_engine.evaluate_trust({}, np.array([0, 0, 9.81]), 0.1)
    assert trust.is_trusted is True
    assert trust.trust_score == 0.95

    anomaly_detector = ConcreteAnomalyDetector()
    report = anomaly_detector.detect_anomalies({}, np.zeros(15), np.zeros(3))
    assert report.detected is False
    assert report.anomaly_type == GNSSAnomalyType.NONE

    ai_vel = ConcreteAIVelocity()
    meas = ai_vel.estimate_velocity(np.zeros((200, 6)))
    assert meas.is_valid is True
    assert meas.velocity_north == 5.0

    fusion = ConcreteAdaptiveFusion()
    params = fusion.compute_adaptive_noise("dr", np.zeros(2), "MOVING")
    assert params.process_noise_scale_pos == 1.0

    conf_est = ConcreteConfidenceEstimator()
    conf = conf_est.compute_confidence(np.eye(15), 5.0, 0.9)
    assert conf.horizontal_accuracy_m == 2.5

    tracker = ConcreteRoadHypothesisTracker()
    hyps = tracker.update_hypotheses(np.array([10.0, 2.0]), 0.0, 5.0)
    assert len(hyps) == 1
    assert hyps[0].segment_id == "road_1_0"

    constraint_eng = ConcreteMapConstraintEngine()
    c = constraint_eng.generate_constraints(hyps[0], np.zeros(15))
    assert c.has_constraint is True

    predictor = ConcreteGNSSPredictor()
    pred = predictor.predict_gnss(np.zeros(15), 10.0)
    assert pred.uncertainty_radius_m == 3.5

    recovery = ConcreteGNSSRecoveryManager()
    plan = recovery.compute_recovery_step(np.zeros(15), np.array([10.0, 0.0, 0.0]), 1)
    assert plan.step_correction_norm_m <= 2.0


def test_sensor_ingestion_adapters():
    smart_adapter = SmartphoneIMUAdapter(sample_rate=10.0)
    packet1 = smart_adapter.process_raw(
        timestamp=100.0,
        raw_accel=np.array([0.1, 0.2, 9.81]),
        raw_gyro=np.array([0.01, 0.02, 0.03]),
    )
    assert isinstance(packet1, SensorPacket)
    assert packet1.source == "smartphone"
    assert packet1.sample_rate == 10.0
    assert packet1.sequence_id == 1

    fog_adapter = ExternalFOGIMUAdapter(sample_rate=200.0)
    packet2 = fog_adapter.process_raw(
        timestamp=100.005,
        raw_accel=np.array([0.0, 0.0, 9.806]),
        raw_gyro=np.array([0.0, 0.0, 0.001]),
    )
    assert packet2.source == "external_fog"
    assert packet2.sample_rate == 200.0
    assert packet2.is_calibrated is True


def test_causal_preprocessing():
    rot = np.eye(3)
    filt = CausalFilter(rot)
    aligned, linear = filt.step(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.1)
    assert aligned.shape == (6,)
    assert linear.shape == (6,)
    assert abs(linear[2]) < 0.5


def test_ekf_fusion_pipeline_invariants():
    ekf = ExtendedKalmanFilter(dt=0.1)
    assert ekf.STATE_DIM == 15
    assert ekf.x.shape == (15,)
    assert ekf.P.shape == (15, 15)

    ekf.initialize_from_gnss(
        position=np.array([10.0, 20.0, 5.0]),
        velocity=np.array([1.0, 2.0, 0.0]),
        heading=0.25,
    )
    assert ekf.x[0] == 10.0
    assert ekf.x[1] == 20.0
    assert ekf.x[8] == 0.25

    ekf.predict(
        accel_body=np.array([0.0, 0.0, 9.81]),
        gyro_body=np.zeros(3),
        apply_nhc=True,
    )
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)
    assert np.allclose(ekf.P, ekf.P.T)

    ekf.set_gnss_denied(timestamp=10.0)
    assert ekf.mode == NavigationMode.DEAD_RECKONING

    ekf._update_ai_velocity(np.array([1.5, 3.0]))
    vel = ekf.get_velocity()
    assert np.linalg.norm(vel[:2]) > 0.0

    ekf.update_zupt(velocity_sigma=0.01)
    vel_zupt = ekf.get_velocity()
    assert np.linalg.norm(vel_zupt) < np.linalg.norm(vel)

    ekf.update_gnss(np.array([10.0, 20.0, 5.0]), timestamp=15.0)
    assert ekf.mode == NavigationMode.REACQUISITION


def test_navigation_state_machine_strictness():
    engine = NavigationStateEngine(NavState.STANDBY)
    assert engine.current_state == NavState.STANDBY

    with pytest.raises(ValueError):
        engine.transition(NavState.REACQUIRING)

    engine.transition(NavState.NORMAL)
    assert engine.current_state == NavState.NORMAL

    with pytest.raises(ValueError):
        engine.transition(NavState.REACQUIRING)

    engine.transition(NavState.DR)
    assert engine.current_state == NavState.DR

    engine.transition(NavState.REACQUIRING)
    assert engine.current_state == NavState.REACQUIRING

    engine.transition(NavState.NORMAL)
    assert engine.current_state == NavState.NORMAL


def test_map_matching_invariants():
    roads = RoadNetwork()
    roads.add_road(
        points=[np.array([0.0, 0.0]), np.array([100.0, 0.0])],
        road_id="east_road",
        name="East Road",
    )
    roads.build_spatial_index()

    matcher = GeometricMapMatcher(roads, search_radius=30.0)
    result = matcher.match(np.array([50.0, 5.0]), heading=np.pi / 2)
    assert result.matched_segment is not None
    assert abs(result.snapped_position[1]) < 1e-6
    assert abs(result.snapped_position[0] - 50.0) < 1e-6
    assert result.confidence > 0.8

    hmm = HMMMapMatcher(roads, search_radius=30.0)
    hmm_result = hmm.match(np.array([50.0, 5.0]), heading=np.pi / 2, travel_distance=10.0)
    assert hmm_result.matched_segment is not None
    assert hmm_result.confidence > 0.0
