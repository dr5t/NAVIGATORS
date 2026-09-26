import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.map_matching import RoadNetwork, RoadSegment, create_map_matcher
from navigation.road_hypothesis import RoadHypothesisEngine, RoadHypothesisConfig
from navigation.interfaces import RoadAmbiguityState, RoadHypothesis


def _best_id(engine) -> str:
    best = engine.get_best_hypothesis()
    assert best is not None
    return best.segment_id


def build_parallel_network():
    network = RoadNetwork()
    network.add_road([np.array([0.0, -100.0]), np.array([0.0, 100.0])], "main", "Main Road", 60.0)
    network.add_road([np.array([12.0, -100.0]), np.array([12.0, 100.0])], "parallel", "Parallel Road", 60.0)
    network.build_spatial_index()
    return network


def test_replay_normal_single_road():
    network = RoadNetwork()
    network.add_road([np.array([0.0, -50.0]), np.array([0.0, 150.0])], "highway", "Highway 1", 80.0)
    network.build_spatial_index()

    engine = RoadHypothesisEngine(network)
    dt = 0.1

    history = []
    for step in range(30):
        t = step * dt
        pos = np.array([0.5, 10.0 * t])
        heading = 0.0
        speed = 10.0
        hyps = engine.update_hypotheses(pos, heading, speed, position_uncertainty=2.0)
        history.append(hyps)

    assert len(history[-1]) == 1
    top = history[-1][0]
    assert top.segment_id == "highway_0"
    assert top.probability > 0.95
    assert top.geometric_compatibility > 0.90
    assert top.heading_compatibility > 0.95
    assert top.velocity_compatibility == 1.0
    assert top.connectivity_compatibility > 0.80
    assert top.combined_likelihood > 0.0
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == "highway_0"


def test_replay_parallel_roads():
    network = build_parallel_network()
    engine = RoadHypothesisEngine(network)
    dt = 0.1

    ambiguous_history = []
    for step in range(15):
        t = step * dt
        pos = np.array([6.0, 10.0 * t])
        heading = 0.0
        speed = 10.0
        hyps = engine.update_hypotheses(pos, heading, speed, position_uncertainty=8.0)
        ambiguous_history.append(hyps)

    final_ambig = ambiguous_history[-1]
    assert len(final_ambig) == 2
    p0 = final_ambig[0].probability
    p1 = final_ambig[1].probability
    assert abs(p0 - p1) < 0.25
    assert p0 > 0.35 and p1 > 0.35
    assert engine.state == RoadAmbiguityState.AMBIGUOUS
    assert engine.get_best_hypothesis() is None

    resolved_history = []
    for step in range(15, 30):
        t = step * dt
        pos = np.array([0.5, 10.0 * t])
        heading = 0.0
        speed = 10.0
        hyps = engine.update_hypotheses(pos, heading, speed, position_uncertainty=2.5)
        resolved_history.append(hyps)

    final_resolved = resolved_history[-1]
    assert final_resolved[0].segment_id == "main_0"
    assert final_resolved[0].probability > 0.75
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == "main_0"


def test_replay_service_road_versus_highway():
    network = RoadNetwork()
    network.add_road([np.array([0.0, -100.0]), np.array([0.0, 200.0])], "highway", "Highway", 100.0)
    network.add_road([np.array([14.0, -100.0]), np.array([14.0, 200.0])], "service", "Service Road", 40.0)
    network.build_spatial_index()

    engine = RoadHypothesisEngine(network)
    dt = 0.1

    history = []
    for step in range(25):
        t = step * dt
        pos = np.array([6.0, 20.0 * t])
        heading = 0.0
        speed = 20.0
        hyps = engine.update_hypotheses(pos, heading, speed, position_uncertainty=7.0)
        history.append(hyps)

    initial_hyps = history[0]
    highway_hyp_init = next(h for h in initial_hyps if "highway" in h.segment_id)
    service_hyp_init = next(h for h in initial_hyps if "service" in h.segment_id)

    assert highway_hyp_init.velocity_compatibility == 1.0
    assert service_hyp_init.velocity_compatibility < 0.25

    final_hyps = history[-1]
    highway_hyp = next(h for h in final_hyps if "highway" in h.segment_id)
    assert highway_hyp.velocity_compatibility == 1.0
    assert highway_hyp.probability > 0.85
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == highway_hyp.segment_id


def test_replay_intersection():
    network = RoadNetwork()
    network.add_road([np.array([0.0, -100.0]), np.array([0.0, 100.0])], "ns", "North-South Road", 50.0)
    network.add_road([np.array([-100.0, 0.0]), np.array([100.0, 0.0])], "ew", "East-West Road", 50.0)
    network.build_spatial_index()

    engine = RoadHypothesisEngine(network)
    dt = 0.1

    history = []
    for step in range(20):
        t = -10.0 + step * 1.0
        pos = np.array([0.2, t])
        heading = 0.0
        speed = 10.0
        hyps = engine.update_hypotheses(pos, heading, speed, position_uncertainty=2.0)
        history.append(hyps)

    final_hyps = history[-1]
    ns_hyp = next((h for h in final_hyps if "ns" in h.segment_id), None)
    ew_hyp = next((h for h in final_hyps if "ew" in h.segment_id), None)

    assert ns_hyp is not None
    assert ns_hyp.heading_compatibility > 0.90
    if ew_hyp is not None:
        assert ew_hyp.heading_compatibility < 0.01
        assert ew_hyp.probability < 0.05
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == ns_hyp.segment_id


def test_replay_road_merge():
    network = RoadNetwork()
    network.add_road([np.array([-15.0, -50.0]), np.array([0.0, 0.0])], "ramp_a", "Ramp A", 60.0)
    network.add_road([np.array([15.0, -50.0]), np.array([0.0, 0.0])], "ramp_b", "Ramp B", 60.0)
    network.add_road([np.array([0.0, 0.0]), np.array([0.0, 80.0])], "trunk", "Trunk Road", 80.0)
    network.build_spatial_index()

    engine = RoadHypothesisEngine(network)
    dt = 0.1

    ramp_hyps = []
    h = 0.0
    for step in range(12):
        alpha = step / 14.0
        pos = (1.0 - alpha) * np.array([-15.0, -50.0]) + alpha * np.array([0.0, 0.0])
        seg_dir = np.array([0.0, 0.0]) - np.array([-15.0, -50.0])
        h = float(np.arctan2(seg_dir[0], seg_dir[1]))
        ramp_hyps = engine.update_hypotheses(pos, h, speed=10.0, position_uncertainty=2.5)

    assert _best_id(engine) == "ramp_a_0"

    junction_hyps = engine.update_hypotheses(np.array([0.0, 0.0]), h, speed=10.0, position_uncertainty=2.5)
    assert len(junction_hyps) >= 2

    merge_history = []
    for step in range(15):
        y = 5.0 + step * 4.0
        pos = np.array([0.2, y])
        h = 0.0
        hyps = engine.update_hypotheses(pos, h, speed=10.0, position_uncertainty=2.0)
        merge_history.append(hyps)

    trunk_hyp = next(h for h in merge_history[-1] if "trunk" in h.segment_id)
    assert trunk_hyp.probability > 0.80
    assert trunk_hyp.connectivity_compatibility > 0.70
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == trunk_hyp.segment_id


def test_replay_road_split():
    network = RoadNetwork()
    network.add_road([np.array([0.0, -60.0]), np.array([0.0, 0.0])], "trunk", "Trunk", 60.0)
    network.add_road([np.array([0.0, 0.0]), np.array([-20.0, 60.0])], "fork_left", "Left Fork", 50.0)
    network.add_road([np.array([0.0, 0.0]), np.array([20.0, 60.0])], "fork_right", "Right Fork", 50.0)
    network.build_spatial_index()

    engine = RoadHypothesisEngine(network)
    dt = 0.1

    for step in range(15):
        y = -60.0 + step * 3.8
        pos = np.array([0.0, y])
        engine.update_hypotheses(pos, 0.0, speed=10.0, position_uncertainty=2.0)

    assert _best_id(engine) == "trunk_0"

    fork_hyps = engine.update_hypotheses(np.array([0.0, 2.0]), 0.0, speed=10.0, position_uncertainty=4.0)
    assert len(fork_hyps) >= 2

    right_history = []
    for step in range(1, 15):
        alpha = step / 15.0
        pos = alpha * np.array([20.0, 60.0]) + np.array([0.2, 0.0])
        seg_dir = np.array([20.0, 60.0])
        h = float(np.arctan2(seg_dir[0], seg_dir[1]))
        hyps = engine.update_hypotheses(pos, h, speed=10.0, position_uncertainty=2.5)
        right_history.append(hyps)

    final_hyps = right_history[-1]
    right_hyp = next(h for h in final_hyps if "fork_right" in h.segment_id)
    assert right_hyp.probability > 0.85
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == right_hyp.segment_id


def test_replay_temporary_map_ambiguity():
    network = build_parallel_network()
    engine = RoadHypothesisEngine(network)
    dt = 0.1

    ambig_count = 0
    for step in range(10):
        t = step * dt
        pos = np.array([6.0, 10.0 * t])
        hyps = engine.update_hypotheses(pos, 0.0, speed=10.0, position_uncertainty=6.0)
        if engine.state == RoadAmbiguityState.AMBIGUOUS:
            ambig_count += 1

    assert ambig_count >= 8
    assert engine.get_best_hypothesis() is None

    for step in range(10, 20):
        t = step * dt
        pos = np.array([12.0, 10.0 * t])
        hyps = engine.update_hypotheses(pos, 0.0, speed=10.0, position_uncertainty=2.0)

    assert engine.state == RoadAmbiguityState.CONVERGED
    assert _best_id(engine) == "parallel_0"


def test_replay_gnss_outage_during_ambiguity():
    network = build_parallel_network()
    engine = RoadHypothesisEngine(network)
    dt = 0.1

    hyps_healthy = engine.update_hypotheses(np.array([5.0, 0.0]), 0.0, speed=10.0, position_uncertainty=2.0)
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert hyps_healthy[0].segment_id == "main_0"

    outage_hyps = None
    for step in range(1, 15):
        unc = 2.0 + step * 0.6
        pos = np.array([5.0, step * 1.0])
        outage_hyps = engine.update_hypotheses(pos, 0.0, speed=10.0, position_uncertainty=unc)

    assert engine.state == RoadAmbiguityState.AMBIGUOUS
    assert outage_hyps is not None
    assert len(outage_hyps) == 2
    assert engine.get_best_hypothesis() is None

    recovered_hyps = engine.update_hypotheses(np.array([0.5, 20.0]), 0.0, speed=10.0, position_uncertainty=1.8)
    assert engine.state == RoadAmbiguityState.CONVERGED
    assert recovered_hyps[0].segment_id == "main_0"
    assert recovered_hyps[0].probability > 0.80


def test_matcher_interface_integration():
    network = build_parallel_network()
    matcher = create_map_matcher("hypothesis", network)
    result_ambig = matcher.match(np.array([6.0, 10.0]), heading=0.0, speed=10.0, position_uncertainty=8.0)
    assert result_ambig.matched_segment is None
    assert result_ambig.confidence == 0.0

    result_converged = matcher.match(np.array([0.2, 10.0]), heading=0.0, speed=10.0, position_uncertainty=2.0)
    assert result_converged.matched_segment is not None
    assert result_converged.matched_segment.id == "main_0"
    assert result_converged.confidence > 0.70
