import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from navigation.map_matching import RoadNetwork, RoadSegment
from navigation.road_hypothesis import RoadHypothesisEngine, RoadHypothesisConfig
from navigation.map_constraint import MapConstraintEngine, MapConstraintConfig, MapConstraintMetrics
from navigation.dead_reckoning import DeadReckoningEngine
from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from navigation.interfaces import RoadAmbiguityState, MapConstraint


def build_straight_network(length_m: float = 300.0) -> RoadNetwork:
    network = RoadNetwork()
    network.add_road(
        [np.array([0.0, 0.0]), np.array([0.0, length_m])],
        road_id="straight",
        name="Straight Road",
        speed_limit=60.0,
    )
    network.build_spatial_index()
    return network


def build_curve_network(radius_m: float = 60.0, num_points: int = 20) -> RoadNetwork:
    network = RoadNetwork()
    angles = np.linspace(np.pi, np.pi / 2.0, num_points)
    pts = []
    for ang in angles:
        x = radius_m + radius_m * np.cos(ang)
        y = radius_m * np.sin(ang)
        pts.append(np.array([x, y]))
    network.add_road(
        pts,
        road_id="curve",
        name="Curved Road",
        speed_limit=50.0,
    )
    network.build_spatial_index()
    return network


def build_intersection_network() -> RoadNetwork:
    network = RoadNetwork()
    network.add_road(
        [np.array([0.0, -50.0]), np.array([0.0, 100.0])],
        road_id="main",
        name="Main Road",
        speed_limit=50.0,
    )
    network.add_road(
        [np.array([-50.0, 25.0]), np.array([50.0, 25.0])],
        road_id="cross",
        name="Cross Street",
        speed_limit=40.0,
    )
    network.build_spatial_index()
    return network


def build_parallel_network_p8() -> RoadNetwork:
    network = RoadNetwork()
    network.add_road(
        [np.array([0.0, 0.0]), np.array([0.0, 250.0])],
        road_id="road_a",
        name="Road A Arterial",
        speed_limit=60.0,
    )
    network.add_road(
        [np.array([12.0, 0.0]), np.array([12.0, 250.0])],
        road_id="road_b",
        name="Road B Parallel",
        speed_limit=60.0,
    )
    network.build_spatial_index()
    return network


def build_service_road_network_p8() -> RoadNetwork:
    network = RoadNetwork()
    network.add_road(
        [np.array([0.0, 0.0]), np.array([0.0, 300.0])],
        road_id="highway",
        name="Highway",
        speed_limit=100.0,
    )
    network.add_road(
        [np.array([10.0, 0.0]), np.array([10.0, 300.0])],
        road_id="service",
        name="Service Road",
        speed_limit=35.0,
    )
    network.build_spatial_index()
    return network


def build_gap_network() -> RoadNetwork:
    network = RoadNetwork()
    network.add_road(
        [np.array([0.0, 0.0]), np.array([0.0, 50.0])],
        road_id="seg1",
        name="Section 1",
        speed_limit=50.0,
    )
    network.add_road(
        [np.array([0.0, 100.0]), np.array([0.0, 200.0])],
        road_id="seg2",
        name="Section 2",
        speed_limit=50.0,
    )
    network.build_spatial_index()
    return network


def test_replay_straight_road():
    network = build_straight_network(200.0)
    engine = MapConstraintEngine(network)
    dr_unconstrained = DeadReckoningEngine(dt=0.1)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([0.0, 0.0])
    dr_unconstrained.start(init_pos, heading=0.0, speed=15.0)
    dr_constrained.start(init_pos, heading=0.0, speed=15.0)

    unconstrained_metrics = MapConstraintMetrics()
    constrained_metrics = MapConstraintMetrics()

    road_seg = network.segments[0]
    dt = 0.1
    steps = 40

    for step in range(steps):
        t = step * dt
        gt_pos = np.array([0.0, 15.0 * t])

        dr_unconstrained.update(ai_velocity=np.array([0.6, 15.0]), gyro_yaw_rate=0.005)
        dr_constrained.update(ai_velocity=np.array([0.6, 15.0]), gyro_yaw_rate=0.005)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=dr_constrained.get_estimated_drift(),
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        selected_id = constraint.segment_id if constraint.has_constraint else None
        unconstrained_metrics.add_step(
            dr_unconstrained.position,
            gt_pos,
            road_segment=road_seg,
            true_segment_id="straight_0",
        )
        constrained_metrics.add_step(
            dr_constrained.position,
            gt_pos,
            road_segment=road_seg,
            selected_segment_id=selected_id,
            true_segment_id="straight_0",
        )

    summary_unconstrained = unconstrained_metrics.compute_summary()
    summary_constrained = constrained_metrics.compute_summary()

    assert summary_unconstrained["max_cross_track_error_m"] > 2.0
    assert summary_constrained["max_cross_track_error_m"] < 0.60
    assert summary_constrained["mean_cross_track_error_m"] < summary_unconstrained["mean_cross_track_error_m"] * 0.25
    assert summary_constrained["road_selection_accuracy_pct"] == 100.0


def test_replay_curve():
    network = build_curve_network(radius_m=60.0, num_points=20)
    engine = MapConstraintEngine(network)
    dr_unconstrained = DeadReckoningEngine(dt=0.1)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    start_pos = network.segments[0].start[:2]
    start_head = network.segments[0].heading
    dr_unconstrained.start(start_pos, heading=start_head, speed=10.0)
    dr_constrained.start(start_pos, heading=start_head, speed=10.0)

    unconstrained_metrics = MapConstraintMetrics()
    constrained_metrics = MapConstraintMetrics()

    steps = 70
    dt = 0.1
    speed = 10.0
    radius = 60.0
    true_yaw_rate = speed / radius

    for step in range(steps):
        t = (step + 1) * dt
        phi = np.pi - (speed * t / radius)
        gt_x = radius + radius * np.cos(phi)
        gt_y = radius * np.sin(phi)
        gt_pos = np.array([gt_x, gt_y])


        biased_yaw_rate = true_yaw_rate * 0.85
        dr_unconstrained.update(ai_speed=speed, gyro_yaw_rate=biased_yaw_rate)
        dr_constrained.update(ai_speed=speed, gyro_yaw_rate=biased_yaw_rate)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=dr_constrained.get_estimated_drift(),
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        selected_id = constraint.segment_id if constraint.has_constraint else None
        unconstrained_metrics.add_step(dr_unconstrained.position, gt_pos)
        constrained_metrics.add_step(
            dr_constrained.position,
            gt_pos,
            selected_segment_id=selected_id,
            true_segment_id=selected_id,
        )

    summary_unconstrained = unconstrained_metrics.compute_summary()
    summary_constrained = constrained_metrics.compute_summary()

    assert summary_unconstrained["max_pos_error_m"] > 2.5
    assert summary_constrained["max_pos_error_m"] < 1.20
    assert summary_constrained["mean_pos_error_m"] < summary_unconstrained["mean_pos_error_m"] * 0.40



def test_replay_intersection():
    network = build_intersection_network()
    engine = MapConstraintEngine(network)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([0.0, -10.0])
    dr_constrained.start(init_pos, heading=0.0, speed=10.0)

    ambiguous_seen = False
    road_selection_correct = 0
    total_steps = 35
    dt = 0.1

    for step in range(total_steps):
        t = step * dt
        gt_pos = np.array([0.0, -10.0 + 10.0 * t])

        dr_constrained.update(ai_speed=10.0, gyro_yaw_rate=0.0)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=2.0,
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        tracker_state = engine.tracker.state
        if tracker_state == RoadAmbiguityState.AMBIGUOUS:
            ambiguous_seen = True

        if constraint.has_constraint:
            assert constraint.segment_id is not None
            assert "main" in constraint.segment_id
            road_selection_correct += 1

        cross_track = abs(dr_constrained.position[0])
        assert cross_track < 1.0

    assert road_selection_correct > 0


def test_replay_parallel_road():
    network = build_parallel_network_p8()
    engine = MapConstraintEngine(network)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([0.8, 0.0])
    dr_constrained.start(init_pos, heading=0.0, speed=12.0)
    metrics = MapConstraintMetrics()
    road_a_seg = network.segments[0]

    steps = 40
    dt = 0.1
    for step in range(steps):
        t = step * dt
        gt_pos = np.array([0.0, 12.0 * t])

        dr_constrained.update(ai_velocity=np.array([0.15, 12.0]), gyro_yaw_rate=0.0)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=2.5,
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        selected_id = constraint.segment_id if constraint.has_constraint else None
        metrics.add_step(
            dr_constrained.position,
            gt_pos,
            road_segment=road_a_seg,
            selected_segment_id=selected_id,
            true_segment_id="road_a_0",
        )

        assert constraint.segment_id != "road_b_0"

    summary = metrics.compute_summary()
    assert summary["road_selection_accuracy_pct"] == 100.0
    assert summary["max_cross_track_error_m"] < 0.60


def test_replay_service_road():
    network = build_service_road_network_p8()
    engine = MapConstraintEngine(network)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([1.0, 0.0])
    dr_constrained.start(init_pos, heading=0.0, speed=24.0)
    metrics = MapConstraintMetrics()
    highway_seg = network.segments[0]

    steps = 30
    dt = 0.1
    for step in range(steps):
        t = step * dt
        gt_pos = np.array([0.0, 24.0 * t])

        dr_constrained.update(ai_velocity=np.array([0.1, 24.0]), gyro_yaw_rate=0.0)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=3.0,
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        selected_id = constraint.segment_id if constraint.has_constraint else None
        metrics.add_step(
            dr_constrained.position,
            gt_pos,
            road_segment=highway_seg,
            selected_segment_id=selected_id,
            true_segment_id="highway_0",
        )

        assert constraint.segment_id != "service_0"

    summary = metrics.compute_summary()
    assert summary["road_selection_accuracy_pct"] == 100.0
    assert summary["mean_cross_track_error_m"] < 0.35
    assert summary["max_cross_track_error_m"] <= 1.0


def test_replay_road_gap():
    network = build_gap_network()
    engine = MapConstraintEngine(network)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([0.2, 0.0])
    dr_constrained.start(init_pos, heading=0.0, speed=10.0)

    steps = 150
    dt = 0.1
    unconstrained_gap_steps = 0
    constrained_section1_steps = 0
    constrained_section2_steps = 0

    for step in range(steps):
        t = step * dt
        y_pos = 10.0 * t

        dr_constrained.update(ai_speed=10.0, gyro_yaw_rate=0.0)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=2.0,
            dt=dt,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        assert not np.isnan(dr_constrained.position[0])
        assert not np.isnan(dr_constrained.position[1])

        if 55.0 <= y_pos <= 95.0:
            if not constraint.has_constraint:
                unconstrained_gap_steps += 1
        elif y_pos < 45.0:
            if constraint.has_constraint:
                constrained_section1_steps += 1
        elif y_pos > 105.0:
            if constraint.has_constraint:
                constrained_section2_steps += 1

    assert unconstrained_gap_steps > 0
    assert constrained_section1_steps > 0
    assert constrained_section2_steps > 0


def test_replay_wrong_map_candidate():
    network = build_straight_network(200.0)
    engine = MapConstraintEngine(network)
    dr_constrained = DeadReckoningEngine(dt=0.1)

    init_pos = np.array([35.0, 0.0])
    dr_constrained.start(init_pos, heading=0.0, speed=10.0)

    for step in range(25):
        dr_constrained.update(ai_speed=10.0, gyro_yaw_rate=0.0)

        constraint = engine.evaluate_from_state(
            position=dr_constrained.position,
            velocity=dr_constrained.velocity,
            heading=dr_constrained.heading,
            position_uncertainty=3.0,
            dt=0.1,
        )
        engine.apply_to_dr(dr_constrained, constraint)

        assert constraint.has_constraint is False
        assert dr_constrained.position[0] > 30.0


def test_replay_gnss_outage_ekf():
    network = build_straight_network(300.0)
    engine = MapConstraintEngine(network)

    ekf_unconstrained = ExtendedKalmanFilter()
    ekf_constrained = ExtendedKalmanFilter()

    init_state = np.zeros(15, dtype=np.float64)
    init_state[1] = 0.0
    init_state[4] = 12.0
    ekf_unconstrained.x = init_state.copy()
    ekf_constrained.x = init_state.copy()


    accel_nom = np.array([0.0, 0.0, 9.81])
    gyro_zero = np.array([0.0, 0.0, 0.0])

    for i in range(10):
        t = i * 0.1
        pos_meas = np.array([0.0, 12.0 * t])
        vel_meas = np.array([0.0, 12.0])
        ekf_unconstrained.predict(accel_nom, gyro_zero)
        ekf_constrained.predict(accel_nom, gyro_zero)
        ekf_unconstrained.update_gnss(pos_meas, vel_meas)
        ekf_constrained.update_gnss(pos_meas, vel_meas)

    ekf_unconstrained.mode = NavigationMode.DEAD_RECKONING
    ekf_constrained.mode = NavigationMode.DEAD_RECKONING

    unconstrained_metrics = MapConstraintMetrics()
    constrained_metrics = MapConstraintMetrics()
    road_seg = network.segments[0]

    outage_steps = 100
    dt = 0.1

    for step in range(outage_steps):
        t = 1.0 + step * dt
        gt_pos = np.array([0.0, 12.0 * t])

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.008])

        ekf_unconstrained.predict(accel, gyro)
        ekf_constrained.predict(accel, gyro)

        cur_pos = ekf_constrained.get_position()[:2]
        cur_vel = ekf_constrained.get_velocity()[:2]
        cur_head = ekf_constrained.get_heading()
        pos_unc = ekf_constrained.get_position_uncertainty()

        constraint = engine.evaluate_from_state(
            position=cur_pos,
            velocity=cur_vel,
            heading=cur_head,
            position_uncertainty=pos_unc,
            dt=dt,
        )
        ekf_constrained.update_map_constraint(constraint=constraint)

        selected_id = constraint.segment_id if constraint.has_constraint else None
        unconstrained_metrics.add_step(
            ekf_unconstrained.get_position()[:2],
            gt_pos,
            road_segment=road_seg,
            true_segment_id="straight_0",
        )
        constrained_metrics.add_step(
            ekf_constrained.get_position()[:2],
            gt_pos,
            road_segment=road_seg,
            selected_segment_id=selected_id,
            true_segment_id="straight_0",
        )

    summary_unconstrained = unconstrained_metrics.compute_summary()
    summary_constrained = constrained_metrics.compute_summary()

    assert summary_unconstrained["max_cross_track_error_m"] > 3.0
    assert summary_constrained["max_cross_track_error_m"] < 0.60
    assert summary_constrained["mean_cross_track_error_m"] < summary_unconstrained["mean_cross_track_error_m"] * 0.20
    assert summary_constrained["road_selection_accuracy_pct"] == 100.0
