"""
Navigators IDR — Map Matching
Snaps estimated positions to the road network to bound drift.

Supports two methods:
    1. Geometric: Nearest-point snapping to road segments
    2. HMM: Hidden Markov Model for probabilistic road selection

Uses offline road network data (no external API required).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class RoadSegment:
    """Represents a single road segment."""
    id: str
    start: np.ndarray    # (2,) ENU position [East, North]
    end: np.ndarray      # (2,) ENU position [East, North]
    name: str = ""
    speed_limit: float = 50.0  # km/h
    one_way: bool = False
    heading: float = 0.0      # radians

    def __post_init__(self):
        """Compute segment properties."""
        direction = self.end - self.start
        self.length = float(np.linalg.norm(direction))
        if self.length > 0:
            self.heading = float(np.arctan2(direction[0], direction[1]))
        self.direction = direction / max(self.length, 1e-10)


@dataclass
class MapMatchResult:
    """Result of a map matching operation."""
    snapped_position: np.ndarray     # Matched position on road
    matched_segment: Optional[RoadSegment]  # Which road segment
    distance_to_road: float          # Perpendicular distance to nearest road
    confidence: float                # Match confidence [0, 1]
    heading_correction: Optional[float] = None  # Suggested heading from road


class RoadNetwork:
    """
    Simple offline road network for map matching.

    Can be initialized with synthetic road data or loaded from
    OpenStreetMap exports.
    """

    def __init__(self):
        self.segments: List[RoadSegment] = []
        self._spatial_index = None  # Simple grid index

    def add_segment(self, segment: RoadSegment):
        """Add a road segment to the network."""
        self.segments.append(segment)
        self._spatial_index = None  # Invalidate index

    def add_road(
        self,
        points: List[np.ndarray],
        road_id: str = "",
        name: str = "",
        speed_limit: float = 50.0,
        one_way: bool = False,
    ):
        """
        Add a road defined by a polyline of points.

        Args:
            points: List of (2,) ENU coordinates defining the road centerline.
            road_id: Unique identifier prefix for segments.
            name: Road name.
            speed_limit: Speed limit in km/h.
            one_way: Whether the road is one-way.
        """
        for i in range(len(points) - 1):
            seg = RoadSegment(
                id=f"{road_id}_{i}",
                start=np.array(points[i], dtype=float),
                end=np.array(points[i + 1], dtype=float),
                name=name,
                speed_limit=speed_limit,
                one_way=one_way,
            )
            self.add_segment(seg)

    def generate_grid_network(
        self,
        center: np.ndarray = np.zeros(2),
        grid_size: float = 100.0,
        num_blocks: int = 5,
    ):
        """
        Generate a simple grid road network for testing.

        Args:
            center: (2,) center position in ENU.
            grid_size: Distance between roads in meters.
            num_blocks: Number of blocks in each direction.
        """
        half = num_blocks * grid_size / 2

        # Horizontal roads (East-West)
        for i in range(num_blocks + 1):
            y = center[1] - half + i * grid_size
            points = [
                np.array([center[0] - half, y]),
                np.array([center[0] + half, y]),
            ]
            self.add_road(points, f"ew_{i}", f"East-West Road {i}")

        # Vertical roads (North-South)
        for i in range(num_blocks + 1):
            x = center[0] - half + i * grid_size
            points = [
                np.array([x, center[1] - half]),
                np.array([x, center[1] + half]),
            ]
            self.add_road(points, f"ns_{i}", f"North-South Road {i}")

    def nearest_point_on_segment(
        self, point: np.ndarray, segment: RoadSegment
    ) -> Tuple[np.ndarray, float]:
        """
        Find the nearest point on a road segment to a given point.

        Args:
            point: (2,) query position.
            segment: Road segment.

        Returns:
            Tuple of (nearest_point, distance).
        """
        # Vector from segment start to point
        v = point - segment.start
        # Segment direction vector
        u = segment.end - segment.start
        length_sq = np.dot(u, u)

        if length_sq < 1e-10:
            return segment.start.copy(), float(np.linalg.norm(v))

        # Project point onto segment line
        t = np.clip(np.dot(v, u) / length_sq, 0.0, 1.0)
        nearest = segment.start + t * u

        distance = float(np.linalg.norm(point - nearest))
        return nearest, distance


class GeometricMapMatcher:
    """
    Simple geometric map matching — snaps to nearest road segment.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        search_radius: float = 50.0,
    ):
        """
        Args:
            road_network: Road network to match against.
            search_radius: Maximum snap distance in meters.
        """
        self.roads = road_network
        self.search_radius = search_radius

    def match(
        self, position: np.ndarray, heading: Optional[float] = None
    ) -> MapMatchResult:
        """
        Match a position to the nearest road.

        Args:
            position: (2,) position in ENU.
            heading: Optional vehicle heading for disambiguation.

        Returns:
            MapMatchResult with snapped position and metadata.
        """
        best_point = position.copy()
        best_distance = float('inf')
        best_segment = None

        for seg in self.roads.segments:
            nearest, dist = self.roads.nearest_point_on_segment(position, seg)

            if dist < best_distance and dist <= self.search_radius:
                # If heading is provided, prefer segments aligned with travel direction
                if heading is not None:
                    heading_diff = abs(heading - seg.heading)
                    heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
                    # Allow opposite direction (heading_diff ≈ π) for two-way roads
                    if not seg.one_way:
                        heading_diff = min(heading_diff, abs(heading_diff - np.pi))

                    # Penalize segments with very different heading
                    if heading_diff > np.pi / 3:  # > 60°
                        dist *= (1 + heading_diff)

                if dist < best_distance:
                    best_distance = dist
                    best_point = nearest
                    best_segment = seg

        confidence = max(0.0, 1.0 - best_distance / self.search_radius)

        return MapMatchResult(
            snapped_position=best_point,
            matched_segment=best_segment,
            distance_to_road=best_distance,
            confidence=confidence,
            heading_correction=best_segment.heading if best_segment else None,
        )


class HMMMapMatcher:
    """
    Hidden Markov Model-based map matching.

    Uses emission probabilities (distance to road) and transition
    probabilities (route plausibility) for more robust matching.
    """

    def __init__(
        self,
        road_network: RoadNetwork,
        sigma: float = 5.0,
        beta: float = 3.0,
        search_radius: float = 50.0,
    ):
        """
        Args:
            road_network: Road network.
            sigma: Emission probability std (meters).
            beta: Transition probability scale (meters).
            search_radius: Maximum candidate search radius.
        """
        self.roads = road_network
        self.sigma = sigma
        self.beta = beta
        self.search_radius = search_radius

        # Viterbi state
        self.prev_candidates = None
        self.prev_probabilities = None

    def _emission_prob(self, distance: float) -> float:
        """Gaussian emission probability based on distance to road."""
        return np.exp(-0.5 * (distance / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))

    def _transition_prob(self, seg1: RoadSegment, seg2: RoadSegment, travel_dist: float) -> float:
        """Transition probability between two road segments."""
        # Route distance approximation (Euclidean between segment midpoints)
        mid1 = (seg1.start + seg1.end) / 2
        mid2 = (seg2.start + seg2.end) / 2
        route_dist = np.linalg.norm(mid2 - mid1)

        # Difference between route distance and travel distance
        diff = abs(route_dist - travel_dist)
        return np.exp(-diff / self.beta) / self.beta

    def match(
        self,
        position: np.ndarray,
        heading: Optional[float] = None,
        travel_distance: float = 0.0,
    ) -> MapMatchResult:
        """
        Match position using HMM Viterbi algorithm.

        Args:
            position: (2,) position in ENU.
            heading: Optional vehicle heading.
            travel_distance: Distance traveled since last match.

        Returns:
            MapMatchResult.
        """
        # Find candidate segments within search radius
        candidates = []
        for seg in self.roads.segments:
            nearest, dist = self.roads.nearest_point_on_segment(position, seg)
            if dist <= self.search_radius:
                candidates.append((seg, nearest, dist))

        if not candidates:
            # No candidates — return raw position
            return MapMatchResult(
                snapped_position=position.copy(),
                matched_segment=None,
                distance_to_road=float('inf'),
                confidence=0.0,
            )

        # Compute emission probabilities
        emissions = np.array([self._emission_prob(dist) for _, _, dist in candidates])

        if self.prev_candidates is not None and travel_distance > 0:
            # Viterbi step: emission × max(prev_prob × transition)
            probs = np.zeros(len(candidates))
            for j, (seg_j, _, _) in enumerate(candidates):
                max_prev = 0.0
                for k, (seg_k, _, _) in enumerate(self.prev_candidates):
                    trans = self._transition_prob(seg_k, seg_j, travel_distance)
                    prev_prob = self.prev_probabilities[k] if self.prev_probabilities is not None else 0.0
                    val = prev_prob * trans
                    max_prev = max(max_prev, val)
                probs[j] = emissions[j] * max_prev
        else:
            probs = emissions

        # Normalize
        total = np.sum(probs)
        if total > 0:
            probs /= total

        # Select best candidate
        best_idx = np.argmax(probs)
        best_seg, best_point, best_dist = candidates[best_idx]

        # Store for next step
        self.prev_candidates = candidates
        self.prev_probabilities = probs

        return MapMatchResult(
            snapped_position=best_point,
            matched_segment=best_seg,
            distance_to_road=best_dist,
            confidence=float(probs[best_idx]),
            heading_correction=best_seg.heading,
        )

    def reset(self):
        """Reset HMM state (e.g., after long GNSS outage)."""
        self.prev_candidates = None
        self.prev_probabilities = None


def create_map_matcher(
    method: str = "geometric",
    road_network: Optional[RoadNetwork] = None,
    **kwargs
) -> object:
    """
    Factory function to create a map matcher.

    Args:
        method: 'geometric' or 'hmm'.
        road_network: Road network. Creates a test grid if None.
        **kwargs: Additional arguments for the matcher.

    Returns:
        GeometricMapMatcher or HMMMapMatcher instance.
    """
    if road_network is None:
        road_network = RoadNetwork()
        road_network.generate_grid_network()

    if method == "hmm":
        return HMMMapMatcher(road_network, **kwargs)
    else:
        return GeometricMapMatcher(road_network, **kwargs)
