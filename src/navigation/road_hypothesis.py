import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from navigation.interfaces import (
    IRoadHypothesisTracker,
    RoadHypothesis,
    RoadAmbiguityState,
)
from navigation.map_matching import RoadNetwork, RoadSegment, MapMatchResult


@dataclass
class RoadHypothesisConfig:
    search_radius_m: float = 40.0
    convergence_probability: float = 0.70
    convergence_margin: float = 0.30
    ambiguity_threshold: float = 0.20
    max_active_hypotheses: int = 6
    min_prune_probability: float = 0.03
    default_pos_uncertainty_m: float = 3.0
    default_heading_uncertainty_rad: float = 0.35
    connectivity_same_segment: float = 0.85
    connectivity_junction: float = 0.80
    connectivity_reachable: float = 0.30
    connectivity_disconnected: float = 0.01


class RoadHypothesisEngine(IRoadHypothesisTracker):
    def __init__(
        self,
        road_network: RoadNetwork,
        config: Optional[RoadHypothesisConfig] = None,
        **kwargs,
    ):
        self.road_network = road_network
        self.config = config or RoadHypothesisConfig()
        self.active_hypotheses: List[RoadHypothesis] = []
        self.best_hypothesis: Optional[RoadHypothesis] = None
        self.state: RoadAmbiguityState = RoadAmbiguityState.NO_MATCH
        self.last_position: Optional[np.ndarray] = None
        self.consecutive_converged_steps: int = 0
        self.search_radius: float = kwargs.get("search_radius", self.config.search_radius_m)

    def _are_segments_connected(self, seg1: RoadSegment, seg2: RoadSegment, tolerance_m: float = 4.0) -> bool:
        if seg1.id == seg2.id:
            return True
        pts1 = [seg1.start, seg1.end]
        pts2 = [seg2.start, seg2.end]
        for p1 in pts1:
            for p2 in pts2:
                if float(np.linalg.norm(p1 - p2)) <= tolerance_m:
                    return True
        return False

    def update_hypotheses(
        self,
        position: np.ndarray,
        heading: float,
        speed: float,
        *args,
        **kwargs,
    ) -> List[RoadHypothesis]:
        pos = np.asarray(position, dtype=np.float64)[:2]
        pos_unc = float(kwargs.get("position_uncertainty", self.config.default_pos_uncertainty_m))
        heading_unc = float(kwargs.get("heading_uncertainty", self.config.default_heading_uncertainty_rad))
        dt = float(kwargs.get("dt", 0.1))

        eff_radius = max(self.config.search_radius_m, pos_unc * 3.0)
        candidate_segments = self.road_network.get_candidate_segments(pos, radius=eff_radius)

        if not candidate_segments:
            self.active_hypotheses = []
            self.best_hypothesis = None
            self.state = RoadAmbiguityState.NO_MATCH
            self.last_position = pos.copy()
            return []

        raw_candidates = []
        seg_map = {seg.id: seg for seg in candidate_segments}

        for seg in candidate_segments:
            nearest, dist = self.road_network.nearest_point_on_segment(pos, seg)
            along_track = float(np.dot(nearest - seg.start, seg.direction))

            sigma_geom = float(np.sqrt(pos_unc ** 2 + 4.0))
            l_geom = float(np.exp(-0.5 * (dist / sigma_geom) ** 2))

            if speed < 0.8:
                l_heading = 1.0
                heading_diff = 0.0
            else:
                diff = abs(heading - seg.heading)
                diff = min(diff, 2.0 * np.pi - diff)
                if not seg.one_way:
                    diff = min(diff, abs(np.pi - diff))
                heading_diff = float(diff)
                sigma_head = max(0.15, heading_unc)
                l_heading = float(np.exp(-0.5 * (heading_diff / sigma_head) ** 2))

            limit_mps = float(seg.speed_limit / 3.6)
            if speed > (limit_mps + 2.0):
                excess = speed - (limit_mps + 2.0)
                l_vel = float(np.exp(-0.5 * (excess / 4.0) ** 2))
            else:
                l_vel = 1.0

            if not self.active_hypotheses:
                l_conn = 1.0
            else:
                conn_sum = 0.0
                for prior_h in self.active_hypotheses:
                    prior_seg = self.road_network.segments
                    prior_obj = next((s for s in prior_seg if s.id == prior_h.segment_id), None)
                    if prior_obj is not None:
                        if prior_obj.id == seg.id:
                            t_prob = self.config.connectivity_same_segment
                        elif self._are_segments_connected(prior_obj, seg):
                            t_prob = self.config.connectivity_junction
                        else:
                            travel_budget = max(25.0, speed * dt + 3.0 * pos_unc)
                            _, d_seg = self.road_network.nearest_point_on_segment(seg.start, prior_obj)
                            if d_seg <= travel_budget:
                                t_prob = self.config.connectivity_reachable
                            else:
                                t_prob = self.config.connectivity_disconnected
                    else:
                        t_prob = self.config.connectivity_disconnected
                    conn_sum += t_prob * prior_h.probability
                l_conn = float(conn_sum)

            combined_l = l_geom * l_heading * l_vel * l_conn

            if l_geom > 1e-4 and combined_l > 1e-7:
                raw_candidates.append({
                    "segment": seg,
                    "nearest": nearest,
                    "dist": dist,
                    "along_track": along_track,
                    "heading_diff": heading_diff,
                    "l_geom": l_geom,
                    "l_heading": l_heading,
                    "l_vel": l_vel,
                    "l_conn": l_conn,
                    "combined_l": combined_l,
                })

        if not raw_candidates:
            self.active_hypotheses = []
            self.best_hypothesis = None
            self.state = RoadAmbiguityState.NO_MATCH
            self.last_position = pos.copy()
            return []

        total_l = sum(c["combined_l"] for c in raw_candidates)
        if total_l <= 0.0:
            self.active_hypotheses = []
            self.best_hypothesis = None
            self.state = RoadAmbiguityState.NO_MATCH
            self.last_position = pos.copy()
            return []

        hypotheses = []
        for c in raw_candidates:
            prob = float(c["combined_l"] / total_l)
            prev_age = 0
            for ph in self.active_hypotheses:
                if ph.segment_id == c["segment"].id:
                    prev_age = ph.age_steps
                    break
            hyp = RoadHypothesis(
                segment_id=str(c["segment"].id),
                probability=float(round(prob, 4)),
                cross_track_distance_m=float(round(c["dist"], 2)),
                along_track_distance_m=float(round(c["along_track"], 2)),
                heading_difference_rad=float(round(c["heading_diff"], 3)),
                snapped_point=c["nearest"].copy(),
                geometric_compatibility=float(round(c["l_geom"], 4)),
                heading_compatibility=float(round(c["l_heading"], 4)),
                velocity_compatibility=float(round(c["l_vel"], 4)),
                connectivity_compatibility=float(round(c["l_conn"], 4)),
                combined_likelihood=float(round(c["combined_l"], 6)),
                speed_limit=float(c["segment"].speed_limit),
                one_way=bool(c["segment"].one_way),
                name=str(c["segment"].name),
                age_steps=prev_age + 1,
            )
            hypotheses.append(hyp)

        hypotheses.sort(key=lambda h: h.probability, reverse=True)

        survivors = [h for h in hypotheses if h.probability >= self.config.min_prune_probability]
        if survivors:
            hypotheses = survivors[:self.config.max_active_hypotheses]
            renorm = sum(h.probability for h in hypotheses)
            if renorm > 0.0:
                for h in hypotheses:
                    h.probability = float(round(h.probability / renorm, 4))

        p_top = hypotheses[0].probability
        p_second = hypotheses[1].probability if len(hypotheses) > 1 else 0.0

        if p_top >= self.config.convergence_probability and (p_top - p_second) >= self.config.convergence_margin:
            self.state = RoadAmbiguityState.CONVERGED
            self.best_hypothesis = hypotheses[0]
            self.consecutive_converged_steps += 1
        elif len(hypotheses) > 1 and p_second >= self.config.ambiguity_threshold:
            self.state = RoadAmbiguityState.AMBIGUOUS
            self.best_hypothesis = None
            self.consecutive_converged_steps = 0
        else:
            self.state = RoadAmbiguityState.CONVERGED
            self.best_hypothesis = hypotheses[0]
            self.consecutive_converged_steps += 1

        self.active_hypotheses = hypotheses
        self.last_position = pos.copy()
        return list(self.active_hypotheses)

    def get_best_hypothesis(self) -> Optional[RoadHypothesis]:
        if self.state == RoadAmbiguityState.CONVERGED:
            return self.best_hypothesis
        return None

    def match(
        self,
        position: np.ndarray,
        heading: Optional[float] = None,
        speed: float = 0.0,
        **kwargs,
    ) -> MapMatchResult:
        pos = np.asarray(position, dtype=np.float64)[:2]
        head_val = float(heading) if heading is not None else 0.0
        self.update_hypotheses(pos, head_val, speed, **kwargs)

        if self.state == RoadAmbiguityState.CONVERGED and self.best_hypothesis is not None:
            best = self.best_hypothesis
            seg = next((s for s in self.road_network.segments if s.id == best.segment_id), None)
            return MapMatchResult(
                snapped_position=best.snapped_point.copy(),
                matched_segment=seg,
                distance_to_road=best.cross_track_distance_m,
                confidence=best.probability,
                heading_correction=seg.heading if seg else None,
            )
        elif self.state == RoadAmbiguityState.AMBIGUOUS and self.active_hypotheses:
            top = self.active_hypotheses[0]
            return MapMatchResult(
                snapped_position=pos.copy(),
                matched_segment=None,
                distance_to_road=top.cross_track_distance_m,
                confidence=0.0,
                heading_correction=None,
            )
        else:
            return MapMatchResult(
                snapped_position=pos.copy(),
                matched_segment=None,
                distance_to_road=float("inf"),
                confidence=0.0,
                heading_correction=None,
            )

    def reset(self) -> None:
        self.active_hypotheses = []
        self.best_hypothesis = None
        self.state = RoadAmbiguityState.NO_MATCH
        self.last_position = None
        self.consecutive_converged_steps = 0
