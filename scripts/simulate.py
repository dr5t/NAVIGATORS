#!/usr/bin/env python3
"""
Navigators IDR - Full Navigation Simulation
Runs the complete navigation pipeline: IMU → AI Model → EKF → DR → Map Match → Output

Generates a synthetic scenario, runs inference, and produces
JSON output for the web simulator visualization.

Usage:
    python scripts/simulate.py
    python scripts/simulate.py --checkpoint checkpoints/best_model.pt
    python scripts/simulate.py --output simulator/data/simulation.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.synthetic_data import SyntheticDataGenerator
from data.preprocessor import IMUPreprocessor
from navigation.ekf import ExtendedKalmanFilter
from navigation.dead_reckoning import DeadReckoningEngine
from navigation.nhc import NonHolonomicConstraints
from navigation.zupt import ZUPTDetector
from navigation.map_matching import create_map_matcher, RoadNetwork
from utils.metrics import compute_all_metrics


from typing import Optional

def run_simulation(
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
):
    """
    Run a full navigation simulation.

    Pipeline per timestep:
        1. Read IMU sample
        2. ZUPT detection
        3. AI velocity estimation (or mock)
        4. EKF prediction (IMU + AI velocity)
        5. GNSS update (if available)
        6. NHC enforcement
        7. Map matching
        8. Record results
    """
    print("\n" + "=" * 60)
    print("  Navigators IDR - Full Navigation Simulation")
    print("=" * 60)

    # --- Generate scenario ---
    print("\n[1/4] Generating synthetic scenario...")
    gen = SyntheticDataGenerator(sample_rate=10.0, seed=42)
    scenario = gen.generate_full_scenario()

    timestamps = scenario["trajectory"]["timestamps"]
    true_positions = scenario["trajectory"]["positions"]
    true_velocities = scenario["trajectory"]["velocities"]
    true_headings = scenario["trajectory"]["headings"]
    accel = scenario["imu"]["accel"]
    gyro = scenario["imu"]["gyro"]
    gnss_available = scenario["gnss"]["available"]
    gnss_positions = scenario["gnss"]["positions"]
    gnss_lat = scenario["gnss"]["true_lat"]
    gnss_lon = scenario["gnss"]["true_lon"]
    meta = scenario["metadata"]

    N = len(timestamps)
    dt = 1.0 / meta["sample_rate"]

    print(f"    Duration: {meta['total_duration']:.1f}s")
    print(f"    Distance: {meta['total_distance']:.1f}m")
    print(f"    Outage:   {meta['outage_ranges']}")
    print(f"    Samples:  {N}")

    # --- Initialize components ---
    print("\n[2/4] Initializing navigation engine...")

    ekf = ExtendedKalmanFilter(dt=dt)
    dr = DeadReckoningEngine(dt=dt)
    nhc = NonHolonomicConstraints()
    zupt = ZUPTDetector()

    # Create a simple road network for map matching
    road_net = RoadNetwork()
    road_net.generate_grid_network(center=np.array([0.0, 0.0]), grid_size=200, num_blocks=10)
    map_matcher = create_map_matcher("geometric", road_net, search_radius=100.0)

    # Initialize EKF from first GNSS fix
    first_gnss_idx = np.where(gnss_available)[0][0]
    ekf.initialize_from_gnss(
        position=gnss_positions[first_gnss_idx],
        velocity=true_velocities[first_gnss_idx],
        heading=true_headings[first_gnss_idx],
    )

    # --- Try loading trained model ---
    model = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n[Model] Loading checkpoint: {checkpoint_path}")
        try:
            import torch
            from models.trainer import Trainer
            model, _ = Trainer.load_checkpoint(checkpoint_path)
            print("[Model] Loaded successfully ✓")
        except Exception as e:
            print(f"[Model] Could not load: {e}. Using ground truth velocity.")

    # --- Run simulation ---
    print("\n[3/4] Running navigation simulation...")

    # Results storage
    results = {
        "timestamps": [],
        "true_positions": [],
        "estimated_positions": [],
        "true_lat_lon": [],
        "estimated_lat_lon": [],
        "gnss_available": [],
        "nav_mode": [],
        "speed_true": [],
        "speed_estimated": [],
        "heading_true": [],
        "heading_estimated": [],
        "position_error": [],
        "confidence": [],
        "zupt_active": [],
        "dr_drift_percent": [],
    }

    preprocessor = IMUPreprocessor(sample_rate=meta["sample_rate"])
    window_buffer = []
    window_size = 200

    prev_gnss = True

    for i in range(N):
        t = timestamps[i]

        # --- ZUPT detection ---
        is_stationary = zupt.update(accel[i], gyro[i], dt)

        # --- AI velocity estimation ---
        # Use ground truth with noise as mock (replace with model inference when trained)
        if model is not None and len(window_buffer) >= window_size:
            import torch
            window = np.array(window_buffer[-window_size:])
            window_tensor = torch.tensor(window[np.newaxis, ...], dtype=torch.float32)
            with torch.no_grad():
                ai_velocity = model(window_tensor).numpy()[0]
        else:
            # Mock: ground truth + noise
            noise = np.random.normal(0, 0.5, 2)
            ai_velocity = true_velocities[i] + noise

        window_buffer.append(np.concatenate([accel[i], gyro[i]]))

        # --- EKF prediction ---
        ekf.predict(accel[i], gyro[i], ai_velocity)

        # --- GNSS update ---
        if gnss_available[i]:
            if not prev_gnss:
                # GNSS just restored - end DR
                if dr.is_active:
                    dr_summary = dr.stop(gnss_restore_position=gnss_positions[i])
                    print(f"    t={t:.1f}s: GNSS restored | "
                          f"DR drift: {dr_summary['drift_percent']:.2f}% "
                          f"(error: {dr_summary['drift_error_m']:.2f}m / {dr_summary['distance_traveled_m']:.1f}m)")

            ekf.update_gnss(gnss_positions[i], timestamp=t)
            prev_gnss = True
        else:
            if prev_gnss:
                # GNSS just lost - start DR
                ekf.set_gnss_denied(timestamp=t)
                pos = ekf.get_position()
                dr.start(pos[:2], ekf.get_heading(), np.linalg.norm(ekf.get_velocity()[:2]), t)
                print(f"    t={t:.1f}s: GNSS DENIED - Dead reckoning active")

            # DR update
            dr.update(ai_velocity=ai_velocity, gyro_yaw_rate=gyro[i, 2], timestamp=t)
            prev_gnss = False

        # --- ZUPT ---
        if is_stationary:
            ekf.update_zupt()

        # --- NHC ---
        vel_nav = ekf.get_velocity()
        heading = ekf.get_heading()
        constrained_vel = nhc.apply_constraints(vel_nav[:2], heading)

        # --- Get estimated position ---
        est_pos = ekf.get_position()
        pos_error = float(np.linalg.norm(est_pos[:2] - true_positions[i]))

        # Convert to lat/lon for visualization
        ref_lat, ref_lon = meta["ref_lat"], meta["ref_lon"]
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(np.radians(ref_lat))
        est_lat = ref_lat + est_pos[1] / meters_per_deg_lat
        est_lon = ref_lon + est_pos[0] / meters_per_deg_lon

        # --- Record ---
        results["timestamps"].append(float(t))
        results["true_positions"].append(true_positions[i].tolist())
        results["estimated_positions"].append(est_pos[:2].tolist())
        results["true_lat_lon"].append([float(gnss_lat[i]), float(gnss_lon[i])])
        results["estimated_lat_lon"].append([float(est_lat), float(est_lon)])
        results["gnss_available"].append(bool(gnss_available[i]))
        results["nav_mode"].append(ekf.mode.value)
        results["speed_true"].append(float(np.linalg.norm(true_velocities[i])))
        results["speed_estimated"].append(float(np.linalg.norm(vel_nav[:2])))
        results["heading_true"].append(true_headings[i])
        results["heading_estimated"].append(heading)
        results["position_error"].append(pos_error)
        results["confidence"].append(1.0 - min(1.0, pos_error / 50.0))
        results["zupt_active"].append(is_stationary)
        results["dr_drift_percent"].append(
            dr.get_drift_percentage() if dr.is_active else 0.0
        )

    # --- Compute final metrics ---
    print("\n[4/4] Computing metrics...")

    est_pos_array = np.array(results["estimated_positions"])
    true_pos_array = np.array(results["true_positions"])

    # Compute metrics specifically during GNSS outages
    outage_mask = ~np.array(results["gnss_available"])
    if np.any(outage_mask):
        outage_est = est_pos_array[outage_mask]
        outage_true = true_pos_array[outage_mask]
        outage_dist = np.sum(np.linalg.norm(np.diff(outage_true, axis=0), axis=1))
        outage_final_err = np.linalg.norm(outage_est[-1] - outage_true[-1])
        outage_drift_pct = (outage_final_err / max(outage_dist, 1e-6)) * 100.0
        outage_rmse = np.sqrt(np.mean(np.sum((outage_est - outage_true) ** 2, axis=1)))
    else:
        outage_dist = 0.0
        outage_drift_pct = 0.0
        outage_rmse = 0.0

    metrics = compute_all_metrics(
        est_pos_array, true_pos_array,
    )

    print(f"\n{'─'*40}")
    print(f"  GNSS-Denied Outage Drift: {outage_drift_pct:.2f}% (target: <10%)")
    print(f"  GNSS-Denied Distance:    {outage_dist:.1f} m")
    print(f"  GNSS-Denied ATE RMSE:    {outage_rmse:.2f} m")
    print(f"  Overall Trajectory RMSE: {metrics['ate']['rmse']:.2f} m")
    print(f"  CEP50:                   {metrics['cep']['CEP50']:.2f} m")
    print(f"  CEP95:                   {metrics['cep']['CEP95']:.2f} m")
    print(f"{'─'*40}")

    # --- Save results ---
    output = {
        "metadata": {
            **meta,
            "metrics": {
                "drift_percent": metrics["drift_percent"],
                "ate_rmse": metrics["ate"]["rmse"],
                "cep50": metrics["cep"]["CEP50"],
                "cep95": metrics["cep"]["CEP95"],
            },
        },
        "data": results,
    }

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "simulator", "data", "simulation.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[Output] Simulation data saved to: {output_path}")
    print("[Output] Open simulator/index.html to visualize!\n")


def main():
    parser = argparse.ArgumentParser(description="Run full navigation simulation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output JSON file")
    args = parser.parse_args()

    run_simulation(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
