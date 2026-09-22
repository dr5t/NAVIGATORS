#!/usr/bin/env python3
"""Replay a saved phone recording through navigation, with a reproducible GPS outage."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.recording import load_recording, sha256, enu, METERS_PER_DEGREE
from evaluation.replay import ABLATIONS, VelocityModel, run_replay


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', required=True, type=Path)
    cli.add_argument('--gnss-outage', type=float, default=60, help='Duration in seconds')
    cli.add_argument('--outage-start', type=float, default=10, help='Elapsed recording time in seconds')
    cli.add_argument('--calibration-seconds', type=float, default=5)
    cli.add_argument('--aligned', action='store_true', help='Input IMU is already in the vehicle frame (including gravity)')
    cli.add_argument('--gyro-order', choices=['xyz', 'alpha-beta-gamma'])
    cli.add_argument('--mode', choices=ABLATIONS, default='G')
    cli.add_argument('--ablations', action='store_true', help='Run all seven A–G configurations')
    cli.add_argument('--backend', choices=['onnx', 'pytorch'], default='onnx')
    cli.add_argument('--model', type=Path, help='ONNX file or PyTorch checkpoint')
    cli.add_argument('--stats', type=Path, default=ROOT / 'checkpoints/norm_stats.json')
    cli.add_argument('--allow-legacy-model', action='store_true', help='Diagnostic only: allow an unverified preprocessing contract')
    cli.add_argument('--map', type=Path, default=ROOT / 'simulator/data/road_network.json')
    cli.add_argument('--output', type=Path, default=ROOT / 'results/replay')
    return cli


def save_trajectory(path, recording, estimates, modes, allowed):
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'east_m', 'north_m', 'velocity_east_mps', 'velocity_north_mps',
                         'heading_rad', 'mode', 'gnss_allowed'])
        for t, row, mode, gps in zip(recording.timestamps, estimates, modes, allowed):
            writer.writerow([t, *[float(v) if np.isfinite(v) else '' for v in row], mode, int(gps)])


def save_browser_trajectory(path, recording, estimates, modes, allowed, report):
    indices = np.flatnonzero(np.isfinite(estimates).all(axis=1))
    rows = estimates[indices]
    origin = report['origin']
    positions = np.column_stack((origin[0] + rows[:, 1] / METERS_PER_DEGREE,
        origin[1] + rows[:, 0] / (METERS_PER_DEGREE * np.cos(np.deg2rad(origin[0])))))
    scored = recording.valid[indices] & recording.fresh[indices]
    reference = recording.gnss[indices, :2]
    errors = np.linalg.norm(rows[:, :2] - enu(reference, origin), axis=1)
    payload = {
        'metadata': {**report, 'source': 'replay.py', 'evaluation_metrics': report['metrics'],
            'metrics': {'ate_rmse': report['metrics']['whole_trip']['position_rmse_m']}},
        'data': {
            'timestamps': recording.timestamps[indices].tolist(),
            'estimated_lat_lon': positions.tolist(),
            'true_lat_lon': [point.tolist() if valid else None for point, valid in zip(reference, scored)],
            'speed_estimated': np.linalg.norm(rows[:, 2:4], axis=1).tolist(),
            'heading_estimated': rows[:, 4].tolist(),
            'gnss_available': allowed[indices].tolist(),
            'nav_mode': [modes[i] for i in indices],
            'position_error': [float(error) if valid else None for error, valid in zip(errors, scored)],

            'confidence': [None] * len(indices),
            'dr_drift_percent': [None] * len(indices),
            'zupt_active': [None] * len(indices),
        },
    }
    path.write_text(json.dumps(payload, allow_nan=False) + '\n')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        recording = load_recording(args.dataset, args.gyro_order)
    except (ValueError, OSError, KeyError) as error:
        print(f'Replay blocked: {error}', file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    keys = list(ABLATIONS) if args.ablations else [args.mode]
    model, model_error = None, None
    if any(ABLATIONS[key].ai for key in keys):
        try:
            model_path = args.model or ROOT / ('simulator/model.onnx' if args.backend == 'onnx' else 'checkpoints/best_model.pt')
            model = VelocityModel(model_path, args.stats, args.backend, args.allow_legacy_model)
        except Exception as error:
            model_error = str(error)
    results = []
    print(f'{args.dataset.stem}\nGNSS outage: {args.gnss_outage:g} sec at {args.outage_start:g} sec')
    print('Development-machine benchmark · comparison against recorded GNSS')
    for key in keys:
        config = ABLATIONS[key]
        try:
            if config.ai and model_error:
                raise ValueError(model_error)
            report, estimates, modes, allowed = run_replay(
                recording, config, args.outage_start, args.gnss_outage,
                args.calibration_seconds, args.aligned, model if config.ai else None, args.map)
            report['status'] = 'completed'
            report['mode'] = key
            report['source_hashes'] = {str(p.relative_to(ROOT)): sha256(p) for p in [
                ROOT / 'replay.py', ROOT / 'src/evaluation/replay.py', ROOT / 'src/evaluation/recording.py',
                ROOT / 'src/evaluation/preprocessing.py', ROOT / 'src/navigation/ekf.py',
                ROOT / 'src/navigation/map_matching.py', ROOT / 'src/navigation/zupt.py']}
            save_trajectory(args.output / f'{args.dataset.stem}_{key}_trajectory.csv', recording, estimates, modes, allowed)
            save_browser_trajectory(args.output / f'{args.dataset.stem}_{key}_trajectory.json', recording, estimates, modes, allowed, report)
            score = report['metrics']['outage']
            print(f'\n{key} - {config.name} (outage reference samples: {score["reference_samples"]})')
            for label, metric in [('Final position error', 'final_position_error_m'), ('Mean position error', 'mean_position_error_m'),
                                  ('Max position error', 'max_position_error_m'), ('Velocity RMSE', 'velocity_rmse_mps'),
                                  ('Heading RMSE', 'heading_rmse_deg')]:
                value = score[metric]
                unit = 'deg' if metric.endswith('_deg') else 'm/s' if metric.endswith('_mps') else 'm'
                print(f'  {label}: {value:.6g} {unit}' if value is not None else f'  {label}: unavailable')
        except Exception as error:
            report = {'mode': key, 'status': 'blocked', 'error': str(error)}
            print(f'{key} - BLOCKED: {error}')
        results.append(report)
    output = args.output / f'{args.dataset.stem}_results.json'
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')

    with (args.output / f'{args.dataset.stem}_ablation.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mode', 'status', 'final_error_m', 'mean_error_m', 'max_error_m', 'velocity_rmse_mps', 'heading_rmse_deg', 'reason'])
        for report in results:
            score = report.get('metrics', {}).get('outage', {})
            writer.writerow([report['mode'], report['status'], *[score.get(k) for k in (
                'final_position_error_m', 'mean_position_error_m', 'max_position_error_m', 'velocity_rmse_mps', 'heading_rmse_deg')], report.get('error', '')])
    print(f'\nResults: {output}')
    return 0 if all(r['status'] == 'completed' for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
