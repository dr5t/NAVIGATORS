import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union

from navigation.interfaces import (
    IMapConstraintEngine,
    IRoadHypothesisTracker,
    MapConstraint,
    RoadHypothesis,
    RoadAmbiguityState,
)
from navigation.map_matching import RoadNetwork, RoadSegment
from navigation.road_hypothesis import RoadHypothesisEngine, RoadHypothesisConfig


@dataclass
class MapConstraintConfig:
    base_lateral_variance: float = 2.0
    along_track_variance: float = 10000.0
    base_heading_variance: float = 0.04
    min_confidence_threshold: float = 0.25
    convergence_probability_threshold: float = 0.70
    convergence_margin_threshold: float = 0.25
    max_innovation_sigma: float = 3.5
    min_speed_for_heading_mps: float = 1.0
    max_heading_diff_rad: float = 0.785398
    allow_ambiguous_soft_constraint: bool = False
    max_lateral_correction_per_step_m: float = 2.0
    max_heading_correction_per_step_rad: float = 0.20


@dataclass
class MapConstraintMetrics:
    cross_track_errors: List[float] = field(default_factory=list)
    along_track_errors: List[float] = field(default_factory=list)
    position_errors: List[float] = field(default_factory=list)
    road_selection_correct: int = 0
    road_selection_total: int = 0

    def add_step(
        self,
        estimated_pos: np.ndarray,
        ground_truth_pos: np.ndarray,
        road_segment: Optional[RoadSegment] = None,
        selected_segment_id: Optional[str] = None,
        true_segment_id: Optional[str] = None,
    ):
        est = np.asarray(estimated_pos, dtype=np.float64)[:2]
        gt = np.asarray(ground_truth_pos, dtype=np.float64)[:2]
        pos_err = float(np.linalg.norm(est - gt))
        self.position_errors.append(pos_err)

        if road_segment is not None:
            tangent = road_segment.direction[:2]
            normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
            diff = est - gt
            ct_err = float(abs(np.dot(diff, normal)))
            at_err = float(abs(np.dot(diff, tangent)))
            self.cross_track_errors.append(ct_err)
            self.along_track_errors.append(at_err)
        else:
            self.cross_track_errors.append(pos_err)
            self.along_track_errors.append(0.0)

        if true_segment_id is not None:
            self.road_selection_total += 1
            if selected_segment_id == true_segment_id:
                self.road_selection_correct += 1

    def compute_summary(self) -> Dict[str, float]:
        if not self.position_errors:
            return {
                "mean_pos_error_m": 0.0,
                "rms_pos_error_m": 0.0,
                "max_pos_error_m": 0.0,
                "p95_pos_error_m": 0.0,
                "mean_cross_track_error_m": 0.0,
                "rms_cross_track_error_m": 0.0,
                "max_cross_track_error_m": 0.0,
                "mean_along_track_error_m": 0.0,
                "rms_along_track_error_m": 0.0,
                "road_selection_accuracy_pct": 100.0,
            }

        pos_arr = np.array(self.position_errors, dtype=np.float64)
        ct_arr = np.array(self.cross_track_errors, dtype=np.float64)
        at_arr = np.array(self.along_track_errors, dtype=np.float64)

        accuracy = 100.0
        if self.road_selection_total > 0:
            accuracy = (self.road_selection_correct / self.road_selection_total) * 100.0

        return {
            "mean_pos_error_m": float(np.mean(pos_arr)),
            "rms_pos_error_m": float(np.sqrt(np.mean(pos_arr ** 2))),
            "max_pos_error_m": float(np.max(pos_arr)),
            "p95_pos_error_m": float(np.percentile(pos_arr, 95)),
            "mean_cross_track_error_m": float(np.mean(ct_arr)),
            "rms_cross_track_error_m": float(np.sqrt(np.mean(ct_arr ** 2))),
            "max_cross_track_error_m": float(np.max(ct_arr)),
            "mean_along_track_error_m": float(np.mean(at_arr)),
            "rms_along_track_error_m": float(np.sqrt(np.mean(at_arr ** 2))),
            "road_selection_accuracy_pct": float(round(accuracy, 2)),
        }


class MapConstraintEngine(IMapConstraintEngine):
    def __init__(
        self,
        road_network: RoadNetwork,
        config: Optional[MapConstraintConfig] = None,
        hypothesis_tracker: Optional[IRoadHypothesisTracker] = None,
    ):
        self.road_network = road_network
        self.config = config or MapConstraintConfig()
        self.tracker = hypothesis_tracker or RoadHypothesisEngine(
            road_network=road_network,
            config=RoadHypothesisConfig(
                convergence_probability=self.config.convergence_probability_threshold,
                convergence_margin=self.config.convergence_margin_threshold,
            ),
        )
        self.last_constraint: Optional[MapConstraint] = None

    def _get_segment_by_id(self, segment_id: str) -> Optional[RoadSegment]:
        for seg in self.road_network.segments:
            if seg.id == segment_id:
                return seg
        return None

    def generate_constraints(
        self,
        hypothesis: Optional[RoadHypothesis],
        state: np.ndarray,
        *args,
        **kwargs,
    ) -> Optional[MapConstraint]:
        if hypothesis is None:
            return MapConstraint(has_constraint=False)

        seg = self._get_segment_by_id(hypothesis.segment_id)
        if seg is None:
            return MapConstraint(has_constraint=False)

        pos_est = np.asarray(state[:2], dtype=np.float64)
        pos_unc = float(kwargs.get("position_uncertainty", 3.0))
        speed = float(kwargs.get("speed", 0.0))
        heading_est = float(kwargs.get("heading", 0.0))

        tangent = seg.direction[:2]
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)

        nearest, dist = self.road_network.nearest_point_on_segment(pos_est, seg)
        cross_track_err = dist

        start_to_pos = pos_est - seg.start[:2]
        along_track_dist = float(np.dot(start_to_pos, tangent))
        along_track_err = 0.0
        if along_track_dist < 0.0:
            along_track_err = abs(along_track_dist)
        elif along_track_dist > seg.length:
            along_track_err = along_track_dist - seg.length

        sigma_eff = np.sqrt(pos_unc ** 2 + self.config.base_lateral_variance)
        if cross_track_err > (self.config.max_innovation_sigma * sigma_eff):
            return MapConstraint(
                has_constraint=False,
                snapped_position=nearest.copy(),
                road_heading=seg.heading,
                cross_track_error_m=float(cross_track_err),
                along_track_error_m=float(along_track_err),
                confidence=0.0,
                segment_id=seg.id,
            )

        prob = max(0.01, min(1.0, hypothesis.probability))
        if prob < self.config.min_confidence_threshold:
            return MapConstraint(
                has_constraint=False,
                snapped_position=nearest.copy(),
                road_heading=seg.heading,
                cross_track_error_m=float(cross_track_err),
                along_track_error_m=float(along_track_err),
                confidence=prob,
                segment_id=seg.id,
            )

        var_scale = float(kwargs.get("covariance_scale", 1.0))
        sigma_lat_sq = (self.config.base_lateral_variance / (prob ** 2)) * max(0.05, var_scale)
        sigma_along_sq = self.config.along_track_variance

        n_outer = np.outer(normal, normal)
        t_outer = np.outer(tangent, tangent)
        r_pos = sigma_lat_sq * n_outer + sigma_along_sq * t_outer

        road_head = seg.heading
        if not seg.one_way:
            head_diff = abs(heading_est - road_head)
            head_diff = min(head_diff, 2.0 * np.pi - head_diff)
            if head_diff > (np.pi / 2.0):
                road_head = (road_head + np.pi) % (2.0 * np.pi)

        eff_diff = abs(heading_est - road_head)
        eff_diff = min(eff_diff, 2.0 * np.pi - eff_diff)

        apply_heading = False
        if speed >= self.config.min_speed_for_heading_mps and eff_diff <= self.config.max_heading_diff_rad:
            apply_heading = True

        state_dim = 15 if len(state) >= 15 else (9 if len(state) >= 9 else len(state))

        if apply_heading and state_dim >= 9:
            H = np.zeros((3, state_dim), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 8] = 1.0

            z = np.zeros(3, dtype=np.float64)
            z[0] = nearest[0]
            z[1] = nearest[1]
            yaw_diff = (road_head - heading_est + np.pi) % (2.0 * np.pi) - np.pi
            z[2] = heading_est + yaw_diff

            sigma_head_sq = (self.config.base_heading_variance / (prob ** 2)) * max(0.05, var_scale)
            R = np.zeros((3, 3), dtype=np.float64)
            R[0:2, 0:2] = r_pos
            R[2, 2] = sigma_head_sq
        else:
            H = np.zeros((2, state_dim), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0

            z = np.zeros(2, dtype=np.float64)
            z[0] = nearest[0]
            z[1] = nearest[1]
            R = r_pos

        constraint = MapConstraint(
            has_constraint=True,
            measurement_matrix=H,
            observation_vector=z,
            noise_covariance=R,
            snapped_position=nearest.copy(),
            road_heading=road_head,
            cross_track_error_m=float(cross_track_err),
            along_track_error_m=float(along_track_err),
            confidence=prob,
            segment_id=seg.id,
            heading_constraint_applied=apply_heading,
            position_constraint_applied=True,
            applied_variance_scale=var_scale,
            normal_vector=normal.copy(),
            tangent_vector=tangent.copy(),
        )
        self.last_constraint = constraint
        return constraint

    def evaluate_from_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        heading: float,
        position_uncertainty: float = 3.0,
        heading_uncertainty: float = 0.20,
        dt: float = 0.1,
        covariance_scale: float = 1.0,
    ) -> MapConstraint:
        pos = np.asarray(position, dtype=np.float64)[:2]
        vel = np.asarray(velocity, dtype=np.float64)[:2]
        speed = float(np.linalg.norm(vel))

        hypotheses = self.tracker.update_hypotheses(
            pos,
            float(heading),
            speed,
            position_uncertainty=position_uncertainty,
            heading_uncertainty=heading_uncertainty,
            dt=dt,
        )

        state_status = getattr(self.tracker, "state", RoadAmbiguityState.NO_MATCH)

        if not hypotheses or state_status == RoadAmbiguityState.NO_MATCH:
            return MapConstraint(has_constraint=False)

        if state_status == RoadAmbiguityState.AMBIGUOUS:
            seg1 = self._get_segment_by_id(hypotheses[0].segment_id)
            seg2 = self._get_segment_by_id(hypotheses[1].segment_id) if len(hypotheses) > 1 else None
            is_connected_corridor = False
            if seg1 is not None and seg2 is not None:
                pts1 = [seg1.start[:2], seg1.end[:2]]
                pts2 = [seg2.start[:2], seg2.end[:2]]
                endpoints_match = False
                for p1 in pts1:
                    for p2 in pts2:
                        if float(np.linalg.norm(p1 - p2)) <= 4.0:
                            endpoints_match = True
                            break
                    if endpoints_match:
                        break
                head_diff = abs(seg1.heading - seg2.heading)
                head_diff = min(head_diff, 2.0 * np.pi - head_diff)
                if not seg1.one_way or not seg2.one_way:
                    head_diff = min(head_diff, abs(np.pi - head_diff))
                if endpoints_match and head_diff <= self.config.max_heading_diff_rad:
                    is_connected_corridor = True

            if is_connected_corridor:
                comb_prob = min(1.0, hypotheses[0].probability + hypotheses[1].probability)
                full_state = np.zeros(15, dtype=np.float64)
                full_state[0:2] = pos
                full_state[3:5] = vel
                full_state[8] = heading
                top_hyp = RoadHypothesis(**{**hypotheses[0].__dict__, "probability": comb_prob})
                return self.generate_constraints(
                    top_hyp,
                    full_state,
                    position_uncertainty=position_uncertainty,
                    speed=speed,
                    heading=heading,
                    covariance_scale=covariance_scale,
                ) or MapConstraint(has_constraint=False)

            if not self.config.allow_ambiguous_soft_constraint:
                return MapConstraint(
                    has_constraint=False,
                    confidence=0.0,
                    cross_track_error_m=hypotheses[0].cross_track_distance_m,
                    segment_id=hypotheses[0].segment_id,
                )


            p1 = hypotheses[0].probability
            p2 = hypotheses[1].probability if len(hypotheses) > 1 else 0.0
            margin = p1 - p2
            if margin < 0.15:
                return MapConstraint(
                    has_constraint=False,
                    confidence=0.0,
                    cross_track_error_m=hypotheses[0].cross_track_distance_m,
                    segment_id=hypotheses[0].segment_id,
                )
            scaled_prob = p1 * (margin / max(0.01, p1))
            top_hyp = RoadHypothesis(**{**hypotheses[0].__dict__, "probability": scaled_prob})
            return self.generate_constraints(
                top_hyp,
                np.concatenate([pos, vel]),
                position_uncertainty=position_uncertainty,
                speed=speed,
                heading=heading,
                covariance_scale=covariance_scale / max(0.1, margin),
            ) or MapConstraint(has_constraint=False)

        best_hyp = hypotheses[0]
        full_state = np.zeros(15, dtype=np.float64)

        full_state[0:2] = pos
        full_state[3:5] = vel
        full_state[8] = heading

        return self.generate_constraints(
            best_hyp,
            full_state,
            position_uncertainty=position_uncertainty,
            speed=speed,
            heading=heading,
            covariance_scale=covariance_scale,
        ) or MapConstraint(has_constraint=False)

    def apply_to_dr(
        self,
        dr_engine: Any,
        constraint: Optional[MapConstraint] = None,
    ) -> np.ndarray:
        if constraint is None or not constraint.has_constraint:
            return dr_engine.position.copy()

        if constraint.snapped_position is None or constraint.normal_vector is None:
            return dr_engine.position.copy()

        est_pos = dr_engine.position.copy()
        snapped = constraint.snapped_position[:2]
        normal = constraint.normal_vector[:2]

        ct_vector = (est_pos - snapped)
        ct_deviation = float(np.dot(ct_vector, normal))

        estimated_drift = 1.0
        if hasattr(dr_engine, "get_estimated_drift"):
            estimated_drift = dr_engine.get_estimated_drift()

        lane_var = self.config.base_lateral_variance / max(0.05, constraint.confidence ** 2)
        drift_var = estimated_drift ** 2
        lat_gain = drift_var / (drift_var + lane_var)
        lat_gain = max(0.0, min(0.85, lat_gain * constraint.confidence))

        correction = lat_gain * ct_deviation * normal
        step_norm = float(np.linalg.norm(correction))
        if step_norm > self.config.max_lateral_correction_per_step_m:
            correction = correction * (self.config.max_lateral_correction_per_step_m / step_norm)

        dr_engine.position -= correction

        if constraint.heading_constraint_applied and constraint.road_heading is not None:
            road_h = constraint.road_heading
            cur_h = dr_engine.heading
            yaw_diff = (road_h - cur_h + np.pi) % (2.0 * np.pi) - np.pi
            if abs(yaw_diff) <= self.config.max_heading_diff_rad:
                head_gain = 0.25 * constraint.confidence
                head_step = head_gain * yaw_diff
                if abs(head_step) > self.config.max_heading_correction_per_step_rad:
                    head_step = np.sign(head_step) * self.config.max_heading_correction_per_step_rad
                new_h = (cur_h + head_step + np.pi) % (2.0 * np.pi) - np.pi
                dr_engine.heading = new_h
                dr_engine.velocity = np.array([
                    dr_engine.speed * np.sin(new_h),
                    dr_engine.speed * np.cos(new_h),
                ], dtype=np.float64)

        return dr_engine.position.copy()
