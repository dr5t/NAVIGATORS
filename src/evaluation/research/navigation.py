"""Causal benchmark adapters. This module never receives scoring references.

Existing components are composed here without changing production behavior.
Confidence is telemetry only in the current architecture: its ablation is allowed
to have no trajectory effect. Adaptive scales are explicitly applied to Q and R;
the existing EKF's apply_adaptive_noise method alone only stores telemetry.
"""
from collections import Counter, deque
from dataclasses import replace
from time import perf_counter
import numpy as np

from evaluation.preprocessing import CausalFilter
from evaluation.recording import enu, gnss_velocity
from evaluation.replay import ABLATIONS, Navigation, load_map
from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from navigation.zupt import ZUPTDetector
from navigation.gnss_trust import GNSSTrustEngine
from navigation.gnss_anomaly import GNSSAnomalyDetector
from navigation.adaptive_fusion import AdaptiveFusionEngine
from navigation.confidence import ConfidenceEstimator
from navigation.road_hypothesis import RoadHypothesisEngine
from navigation.map_constraint import MapConstraintEngine
from navigation.gnss_recovery import ValidatedGNSSRecoveryManager
from navigation.gnss_prediction import BaselineGNSSDegradationPredictor
from navigation.interfaces import (GNSSTrustMetric, GNSSTrustState, AIVelocityMeasurement,
                                   RoadHypothesis, RoadAmbiguityState, MapConstraint)


class NearestRoadTracker:
    """A stateless geometric control when temporal road hypotheses are disabled."""
    def __init__(self, matcher):
        self.matcher = matcher
        self.state = RoadAmbiguityState.NO_MATCH
        self.best = None

    def update_hypotheses(self, position, heading, speed, **kwargs):
        match = self.matcher.match(position, heading)
        segment = match.matched_segment
        if segment is None:
            self.best = None
            self.state = RoadAmbiguityState.NO_MATCH
            return []
        self.best = RoadHypothesis(segment.id, match.confidence, match.distance_to_road,
                                   float(np.dot(position - segment.start, segment.direction)),
                                   float((heading - segment.heading + np.pi) % (2*np.pi) - np.pi),
                                   match.snapped_position)
        self.state = RoadAmbiguityState.CONVERGED
        return [self.best]

    def get_best_hypothesis(self):
        return self.best


class ResearchNavigation:
    def __init__(self, architecture, rotation, initial_fix, origin, model=None, map_path=None):
        self.cfg, self.model, self.origin = architecture, model, origin
        self.frontend = CausalFilter(rotation)
        self.buffer = deque(maxlen=model.window_size if model else 1)
        self.ekf = ExtendedKalmanFilter()
        self.ekf.initialize_from_gnss(enu(initial_fix[:2], origin), gnss_velocity(initial_fix), np.deg2rad(initial_fix[4]))
        self.position = self.ekf.get_position()[:2].copy()
        self.velocity = self.ekf.get_velocity()[:2].copy()
        self.heading = self.ekf.get_heading()
        self.counts = Counter()
        self.anomaly = GNSSAnomalyDetector() if architecture.anomaly_detection else None
        self.trust = GNSSTrustEngine() if architecture.gnss_trust else None
        if self.trust:
            self.trust.anomaly_detector = None  # Anomaly detection is exclusively controlled above.
        self.adaptive = AdaptiveFusionEngine() if architecture.adaptive_fusion else None
        self.confidence = ConfidenceEstimator() if architecture.confidence else None
        self.recovery = ValidatedGNSSRecoveryManager() if architecture.recovery else None
        self.predictor = BaselineGNSSDegradationPredictor() if architecture.outage_prediction else None
        self.zupt = ZUPTDetector() if architecture.vehicle_constraints else None
        self.matcher = load_map(map_path, origin) if architecture.map_constraints or architecture.road_hypotheses else None
        self.tracker = None
        if self.matcher:
            self.tracker = RoadHypothesisEngine(self.matcher.roads) if architecture.road_hypotheses else NearestRoadTracker(self.matcher)
        self.map_engine = MapConstraintEngine(self.matcher.roads, hypothesis_tracker=self.tracker) if architecture.map_constraints else None
        self.recovering = False
        self.denied_since = None
        self.distance = 0.
        self.last_trust = None
        self.noise_base = {k: getattr(self.ekf, k) for k in ("q_pos", "q_vel", "q_ori", "r_gnss_pos", "r_gnss_vel")}

    def step(self, accel, gyro, dt, timestamp, gnss=None, fresh=False, receiver_time=None):
        started = perf_counter()
        cfg = self.cfg
        aligned, linear = self.frontend.step(accel, gyro, dt)
        self.buffer.append(linear)
        ai, measurement, inference_ms = None, None, None
        if cfg.ai_velocity and len(self.buffer) == self.buffer.maxlen:
            clock = perf_counter()
            ai = np.asarray(self.model.predict(self.buffer), dtype=float)
            inference_ms = (perf_counter() - clock) * 1000
            if ai.shape != (2,) or not np.isfinite(ai).all():
                raise ValueError("Invalid AI ENU velocity")
            measurement = AIVelocityMeasurement(ai[1], ai[0], self.ekf.r_ai_vel**2,
                                                self.ekf.r_ai_vel**2, inference_ms, True,
                                                receptive_field_samples=self.buffer.maxlen, timestamp=timestamp)
            self.counts["ai_velocity"] += 1
        data = None if gnss is None else dict(position_enu=enu(gnss[:2], self.origin),
                 velocity_enu=gnss_velocity(gnss), accuracy=float(gnss[5]), speed=float(gnss[3]),
                 heading=float(np.deg2rad(gnss[4])), timestamp=timestamp if receiver_time is None else receiver_time)
        context = dict(dt=max(dt, 1e-6), current_position=self.position, current_velocity=self.velocity,
                       current_heading=self.heading, position_uncertainty=self.ekf.get_position_uncertainty())
        anomaly = None
        if self.anomaly and (fresh or data is None):
            anomaly = self.anomaly.detect_anomalies(data, inertial_velocity=ai, **context)
            self.counts["anomaly_detection"] += 1
        if self.trust:
            if fresh or data is None:
                self.last_trust = self.trust.evaluate_trust(data, ai_velocity=ai, anomaly_report=anomaly, **context)
                self.counts["gnss_trust"] += 1
            trust = self.last_trust
        else:
            # The ablation uses receiver availability alone, not a hidden trust engine.
            trust = GNSSTrustMetric(1. if data else 0., data is not None, 1., 1.,
                                   GNSSTrustState.TRUSTED if data else GNSSTrustState.UNUSABLE)
        admitted = data is not None and trust is not None and trust.state != GNSSTrustState.UNUSABLE
        if anomaly is not None and anomaly.detected:
            admitted = False
        prediction = None
        if self.predictor and (fresh or data is None):
            output = self.predictor.predict(data, navigation_state=self.ekf.x.copy(), ai_velocity=ai)
            if output.risk_level.value != "UNKNOWN":
                prediction = float(output.predicted_outage_probability)
            self.counts["outage_prediction"] += 1
        noise = None
        if self.adaptive:
            noise = self.adaptive.compute_adaptive_noise("GNSS_INS" if admitted else "DR", gnss_trust=trust,
                                                         ai_velocity=measurement)
            self.ekf.apply_adaptive_noise(noise)
            for key, scale in (("q_pos", noise.process_noise_scale_pos), ("q_vel", noise.process_noise_scale_vel),
                               ("q_ori", noise.process_noise_scale_att)):
                setattr(self.ekf, key, self.noise_base[key] * scale)
            if measurement:
                measurement = replace(measurement, variance_east=measurement.variance_east * noise.measurement_noise_scale_ai_vel,
                                      variance_north=measurement.variance_north * noise.measurement_noise_scale_ai_vel)
            self.counts["adaptive_fusion"] += 1
        previous_position = self.position.copy()
        if not admitted:
            if self.denied_since is None:
                self.denied_since = timestamp
                if self.recovery:
                    self.recovery.reset()
            self.recovering = True
        gnss_correction = 0.
        if cfg.ekf:
            self.ekf.dt = dt
            if not admitted:
                self.ekf.set_gnss_denied(timestamp)
            elif trust is not None and trust.state in (GNSSTrustState.DEGRADED, GNSSTrustState.SUSPICIOUS):
                self.ekf.mode = NavigationMode.GNSS_DEGRADED
            self.ekf.predict(np.r_[linear[:2], linear[2] + 9.81], linear[3:], measurement,
                             apply_nhc=cfg.vehicle_constraints)
            self.counts["ekf"] += 1
            self.counts["vehicle_constraints"] += int(cfg.vehicle_constraints)
            if admitted and fresh:
                prior, covariance = self.ekf.x.copy(), self.ekf.P.copy()
                plan = None
                if self.recovery and self.recovering:
                    plan = self.recovery.process_gnss_return(data, prior, dt=max(dt, 1e-6),
                              dr_uncertainty=self.ekf.get_position_uncertainty(), timestamp=timestamp)
                    self.counts["recovery"] += 1
                if plan is None or (plan.consistency_passed and plan.step_correction_norm_m > 0):
                    effective_trust = trust if cfg.gnss_trust else None
                    if noise:
                        effective_trust = replace(trust, position_variance_scale=noise.measurement_noise_scale_gnss_pos,
                                                  velocity_variance_scale=noise.measurement_noise_scale_gnss_vel)
                    # Disable the EKF's implicit reacquisition path: the intervention
                    # is wholly owned by the explicitly enabled recovery manager.
                    self.ekf.mode = NavigationMode.GNSS_INS
                    self.ekf.update_gnss(data["position_enu"], data["velocity_enu"], timestamp, effective_trust)
                    if plan is not None:
                        correction = self.ekf.x - prior
                        correction[8] = (correction[8] + np.pi) % (2*np.pi) - np.pi
                        fraction = min(1., plan.step_correction_norm_m / max(np.linalg.norm(correction[:2]), 1e-12))
                        self.ekf.x = prior + fraction * correction
                        self.ekf.P = (1-fraction) * covariance + fraction * self.ekf.P
                    self.counts["gnss_updates"] += 1
                gnss_correction = float(np.linalg.norm(self.ekf.x[:2] - prior[:2]))
                if plan is None or plan.is_converged:
                    self.recovering = False
                    self.denied_since = None
            if self.zupt and self.zupt.update(aligned[:3], aligned[3:], dt,
                                              estimated_speed=float(np.linalg.norm(self.ekf.x[3:5]))):
                self.ekf.update_zupt()
                self.counts["zupt"] += 1
            self.position, self.velocity, self.heading = self.ekf.x[:2].copy(), self.ekf.x[3:5].copy(), float(self.ekf.x[8])
        else:
            self.heading = float((self.heading + linear[5]*dt + np.pi) % (2*np.pi) - np.pi)
            forward = np.array([np.sin(self.heading), np.cos(self.heading)])
            lateral = np.array([np.cos(self.heading), -np.sin(self.heading)])
            acceleration = linear[0]*forward + linear[1]*lateral
            if ai is not None:
                self.velocity = ai.copy()
                self.position += self.velocity * dt
                if np.linalg.norm(ai) > .5:
                    self.heading = float(np.arctan2(ai[0], ai[1]))
            else:
                self.position += self.velocity*dt + .5*acceleration*dt**2
                self.velocity += acceleration*dt
            if cfg.vehicle_constraints:
                self.velocity = np.dot(self.velocity, forward) * forward
                if self.zupt.update(aligned[:3], aligned[3:], dt, estimated_speed=float(np.linalg.norm(self.velocity))):
                    self.velocity[:] = 0
                    self.counts["zupt"] += 1
                self.counts["vehicle_constraints"] += 1
            if admitted and fresh:
                gnss_correction = float(np.linalg.norm(self.position - data["position_enu"]))
                self.position, self.velocity = data["position_enu"].copy(), data["velocity_enu"].copy()
                self.heading = data["heading"]
                self.counts["gnss_updates"] += 1
                self.recovering, self.denied_since = False, None
        hypothesis, constraint = None, MapConstraint(False)
        if self.map_engine:
            constraint = self.map_engine.evaluate_from_state(self.position, self.velocity, self.heading,
                             position_uncertainty=self.ekf.get_position_uncertainty(), dt=max(dt, 1e-6))
            self.counts["map_constraints"] += 1
        elif self.tracker:
            self.tracker.update_hypotheses(self.position, self.heading, float(np.linalg.norm(self.velocity)),
                             position_uncertainty=self.ekf.get_position_uncertainty(), dt=max(dt, 1e-6))
        if self.tracker:
            hypothesis = self.tracker.get_best_hypothesis()
            self.counts["road_hypotheses"] += int(cfg.road_hypotheses)
        if constraint.has_constraint:
            if self.adaptive:
                map_noise = self.adaptive.compute_adaptive_noise("GNSS_INS" if admitted else "DR",
                                            gnss_trust=trust, ai_velocity=measurement, map_constraint=constraint)
                if constraint.noise_covariance is not None:
                    constraint = replace(constraint, noise_covariance=constraint.noise_covariance * map_noise.measurement_noise_scale_map)
            self.ekf.update_map_constraint(constraint=constraint)
            self.position, self.velocity, self.heading = self.ekf.x[:2].copy(), self.ekf.x[3:5].copy(), float(self.ekf.x[8])
            self.counts["map_updates"] += 1
        self.distance += float(np.linalg.norm(self.position - previous_position))
        confidence = None
        if self.confidence:
            confidence = self.confidence.compute_confidence(self.ekf.P, timestamp=timestamp,
                         dr_duration=timestamp-self.denied_since if self.denied_since is not None else 0.,
                         gnss_trust=trust, ai_velocity=measurement, road_hypothesis=hypothesis,
                         distance_traveled=self.distance, ekf_mode=self.ekf.mode.value).to_dict()
            self.counts["confidence"] += 1
        estimate = np.r_[self.position, self.velocity, self.heading]
        if not np.isfinite(estimate).all():
            raise ValueError("Navigation diverged to nonfinite values")
        return dict(estimate=estimate.tolist(), inference_latency_ms=inference_ms,
                    loop_latency_ms=(perf_counter()-started)*1000, gnss_correction_m=gnss_correction,
                    gnss_admitted=bool(admitted), anomaly_detected=anomaly.detected if anomaly else None,
                    outage_probability=prediction, confidence=confidence,
                    road_hypothesis_id=hypothesis.segment_id if hypothesis and cfg.road_hypotheses else None,
                    map_match_id=constraint.segment_id if constraint.has_constraint else None,
                    recovery_active=self.recovering, component_counts=dict(self.counts))


class ProductionNavigation:
    """Instrumentation around the unchanged existing replay.Navigation G path."""
    def __init__(self, rotation, initial_fix, origin, model, map_path):
        self.matcher = load_map(map_path, origin)
        self.nav = Navigation(ABLATIONS["G"], rotation, initial_fix, origin, model, self.matcher)
        self.correction, self.match = 0., None
        update = self.nav.ekf.update_gnss

        def measured_update(*args, **kwargs):
            before = self.nav.ekf.x[:2].copy()
            result = update(*args, **kwargs)
            self.correction += float(np.linalg.norm(self.nav.ekf.x[:2]-before))
            return result

        self.nav.ekf.update_gnss = measured_update
        match = self.matcher.match

        def measured_match(*args, **kwargs):
            result = match(*args, **kwargs)
            self.match = result.matched_segment.id if result.matched_segment and result.confidence > .8 else None
            return result

        self.matcher.match = measured_match

    def step(self, accel, gyro, dt, timestamp, gnss=None, fresh=False, receiver_time=None):
        self.correction, self.match = 0., None
        previous_inferences = len(self.nav.timings["tcn"])
        estimate, mode = self.nav.step(accel, gyro, dt, gnss, fresh)
        return dict(estimate=estimate.tolist(), inference_latency_ms=self.nav.timings["tcn"][-1] if len(self.nav.timings["tcn"]) > previous_inferences else None,
                    loop_latency_ms=self.nav.timings["total_loop"][-1], gnss_correction_m=self.correction,
                    gnss_admitted=gnss is not None, anomaly_detected=None, outage_probability=None,
                    confidence=None, road_hypothesis_id=None, map_match_id=self.match,
                    recovery_active=mode == "reacq", component_counts=dict(self.nav.counters))
