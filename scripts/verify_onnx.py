#!/usr/bin/env python3
"""Measure PyTorch/ONNX parity on identical windows from a supplied phone recording."""
import argparse
import json
import re
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.recording import load_recording, sha256
from evaluation.preprocessing import prepare_features, PREPROCESSING_ID


def compare_outputs(pytorch, onnx, atol=1e-5, rtol=1e-4):
    pytorch, onnx = np.asarray(pytorch), np.asarray(onnx)
    if pytorch.shape != onnx.shape or not np.isfinite(pytorch).all() or not np.isfinite(onnx).all():
        return {'passed': False, 'reason': 'Output shape mismatch or non-finite output'}
    difference = np.abs(pytorch - onnx)
    return {'passed': bool(np.all(difference <= atol + rtol * np.abs(pytorch))),
            'atol': atol, 'rtol': rtol, 'max_absolute_difference': float(difference.max()),
            'mean_absolute_difference': float(difference.mean()),
            'max_relative_difference': float(np.max(difference / np.maximum(np.abs(pytorch), 1e-12))),
            'output_scalars': pytorch.size}


def verify(args):
    import torch
    import onnx
    import onnxruntime as ort
    from models.trainer import Trainer
    recording = load_recording(args.dataset, args.gyro_order)
    features, first, _ = prepare_features(recording, args.calibration_seconds, args.aligned)
    stats = json.loads(args.stats.read_text())
    window_size = int(stats.get('window_size', 200))
    mean, std = np.asarray(stats['mean'], np.float32), np.asarray(stats['std'], np.float32)
    if mean.shape != (6,) or std.shape != (6,) or not np.isfinite(np.r_[mean, std]).all() or np.any(std <= 0):
        raise ValueError('Invalid training normalization statistics')
    if len(features) < window_size or args.windows < 1:
        raise ValueError('Need at least one complete real IMU window after calibration and --windows >= 1')
    indices = np.unique(np.linspace(0, len(features) - window_size, min(args.windows, len(features) - window_size + 1), dtype=int))
    windows = np.stack([(features[i:i + window_size] - mean) / std for i in indices]).astype(np.float32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model, checkpoint = Trainer.load_checkpoint(str(args.checkpoint), device=torch.device('cpu'))
    trained_contract = checkpoint.get('config', {}).get('data_contract')
    if trained_contract is not None and trained_contract != stats:
        raise ValueError('Provided normalization statistics differ from the checkpoint training contract')
    with torch.inference_mode():
        pt_outputs = np.stack([model(torch.from_numpy(window[None])).numpy()[0] for window in windows])
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='onnx-parity-', dir=args.onnx.parent) as directory:
        candidate = Path(directory) / args.onnx.name if args.export else args.onnx
        if args.export:
            # Export the checkpoint's recorded architecture on a real input window.
            torch.onnx.export(model, (torch.from_numpy(windows[:1]),), str(candidate),
                              input_names=['imu_window'], output_names=['velocity'],
                              opset_version=18, dynamo=False, external_data=False,
                              dynamic_axes={'imu_window': {0: 'batch'}, 'velocity': {0: 'batch'}})
        onnx.checker.check_model(str(candidate))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        timer = perf_counter()
        session = ort.InferenceSession(str(candidate), options, providers=['CPUExecutionProvider'])
        startup_ms = (perf_counter() - timer) * 1000
        ort_outputs = np.stack([np.asarray(session.run(None, {session.get_inputs()[0].name: w[None]})[0])[0] for w in windows])
        report = {**compare_outputs(pt_outputs, ort_outputs, args.atol, args.rtol),
                  'dataset': str(args.dataset.resolve()), 'dataset_sha256': recording.digest,
                  'provenance': recording.metadata['provenance'],
                  'checkpoint_sha256': sha256(args.checkpoint), 'normalization_sha256': sha256(args.stats),
                  'preprocessing': PREPROCESSING_ID, 'checkpoint_preprocessing': stats.get('preprocessing', 'legacy-unverified'),
                  'window_start_indices': (indices + first).tolist(), 'window_count': len(windows),
                  'window_size': window_size, 'onnx_startup_ms': startup_ms,
                  'pytorch_version': torch.__version__, 'onnxruntime_version': ort.__version__,
                  'pytorch_outputs': pt_outputs.tolist(), 'onnx_outputs': ort_outputs.tolist(),
                  'claim_scope': 'Numerical export parity only; does not establish navigation accuracy or real recording provenance'}
        if report['passed'] and args.export:
            shutil.copyfile(candidate, args.onnx)
            onnx_hash = sha256(args.onnx)
            artifact = {**stats, 'checkpoint_sha256': report['checkpoint_sha256'],
                        'onnx_sha256': onnx_hash, 'output_order': ['east', 'north'],
                        'external_data': False}
            args.onnx.with_suffix('.contract.json').write_text(json.dumps(artifact, indent=2) + '\n')
            if args.onnx.resolve() == ROOT / 'simulator/model.onnx':
                worker = ROOT / 'simulator/sw.js'
                contract_hash = sha256(args.onnx.with_suffix('.contract.json'))
                version = onnx_hash[:12] + '-' + contract_hash[:8]
                worker.write_text(re.sub(r"const CACHE_NAME = '[^']+';", f"const CACHE_NAME = 'navigators-idr-offline-{version}';", worker.read_text(), count=1))
        report['onnx_sha256'] = sha256(args.onnx if report['passed'] else candidate)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in report.items() if 'outputs' not in k}, indent=2))
        return 0 if report['passed'] else 1


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', required=True, type=Path)
    cli.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/best_model.pt')
    cli.add_argument('--onnx', type=Path, default=ROOT / 'simulator/model.onnx')
    cli.add_argument('--stats', type=Path, default=ROOT / 'checkpoints/norm_stats.json')
    cli.add_argument('--export', action='store_true', help='Export to staging; replace ONNX only after measured parity passes')
    cli.add_argument('--windows', type=int, default=32)
    cli.add_argument('--calibration-seconds', type=float, default=5)
    cli.add_argument('--aligned', action='store_true')
    cli.add_argument('--gyro-order', choices=['xyz', 'alpha-beta-gamma'])
    cli.add_argument('--atol', type=float, default=1e-5)
    cli.add_argument('--rtol', type=float, default=1e-4)
    cli.add_argument('--report', type=Path, default=ROOT / 'results/onnx_parity.json')
    args = cli.parse_args(argv)
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        cli.error('Tolerances must be finite and nonnegative')
    try:
        return verify(args)
    except Exception as error:
        print(f'ONNX verification blocked: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
