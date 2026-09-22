"""Replay navigation with a strict GPS gate, identical A–G inputs, and separate scoring."""
from collections import deque
from dataclasses import dataclass, asdict
import json
import platform
from pathlib import Path
import resource
import sys
from time import perf_counter
import numpy as np
from navigation.ekf import ExtendedKalmanFilter
from navigation.zupt import ZUPTDetector
from navigation.map_matching import RoadNetwork, create_map_matcher
from evaluation.recording import enu, gnss_velocity, outage_mask, sha256, METERS_PER_DEGREE
from evaluation.preprocessing import CausalFilter, calibrate, PREPROCESSING_ID


@dataclass(frozen=True)
class Configuration:
    name: str
    filtered: bool = False
    ai: bool = False
    ekf: bool = False
    nhc: bool = False
    zupt: bool = False
    map_matching: bool = False


ABLATIONS = {
    'A': Configuration('Raw IMU'),
    'B': Configuration('Filtered IMU', True),
    'C': Configuration('AI velocity', True, True),
    'D': Configuration('AI + EKF', True, True, True),
    'E': Configuration('AI + EKF + NHC', True, True, True, True),
    'F': Configuration('AI + EKF + NHC + ZUPT', True, True, True, True, True),
    'G': Configuration('Full system', True, True, True, True, True, True),
}


def timing_summary(values):
    if not values:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'max_ms': None}
    return {'count': len(values), 'mean_ms': float(np.mean(values)),
            'p50_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'max_ms': float(np.max(values))}


def machine_info():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'label': 'Development-machine benchmark', 'system': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(),
            'python': platform.python_version(),
            'process_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss * 1024),
            'ram_scope': 'Whole Python process high-water RSS, includes imports and prior ablations'}


class VelocityModel:
    def __init__(self, model_path, stats_path, backend='onnx', allow_legacy=False):
        started = perf_counter()
        self.path = Path(model_path)
        self.stats_path = Path(stats_path)
        stats = json.loads(self.stats_path.read_text())
        self.mean = np.asarray(stats['mean'], dtype=np.float32)
        self.std = np.asarray(stats['std'], dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,) or not np.isfinite(np.r_[self.mean, self.std]).all() or np.any(self.std <= 0):
            raise ValueError('Normalization statistics must contain six finite means and positive standard deviations')
        self.preprocessing = stats.get('preprocessing', 'legacy-unverified')
        if self.preprocessing != PREPROCESSING_ID and not allow_legacy:
            raise ValueError('Checkpoint preprocessing is unverified/noncausal. Train on the causal pipeline first; '
                             '--allow-legacy-model is available for diagnostic runs only')
        self.window_size = int(stats.get('window_size', 200))
        self.sample_rate = stats.get('sample_rate_hz')
        self.backend = backend
        if not allow_legacy and stats.get('output_order') != ['east', 'north']:
            raise ValueError('Model contract must specify ENU output order [east, north]')
        if backend == 'onnx':
            if not allow_legacy:
                artifact = json.loads(self.path.with_suffix('.contract.json').read_text())
                if artifact.get('onnx_sha256') != sha256(self.path):
                    raise ValueError('ONNX artifact hash differs from its verified export contract')
                if any(artifact.get(key) != stats.get(key) for key in ('mean', 'std', 'preprocessing', 'window_size', 'sample_rate_hz', 'output_order')):
                    raise ValueError('Normalization/preprocessing differs from the verified ONNX export')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
            shape = self.session.get_inputs()[0].shape
            if len(shape) != 3 or shape[2] != 6 or (isinstance(shape[1], int) and shape[1] != self.window_size):
                raise ValueError(f'Model shape {shape} does not match normalization/window contract')
        else:
            import torch
            from models.trainer import Trainer
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            self.model, checkpoint = Trainer.load_checkpoint(str(model_path), device=torch.device('cpu'))
            if not allow_legacy and checkpoint.get('config', {}).get('data_contract') != stats:
                raise ValueError('Normalization differs from the checkpoint training contract')
        self.startup_ms = (perf_counter() - started) * 1000

    def predict(self, window):
        window = ((np.asarray(window, dtype=np.float32) - self.mean) / self.std)[None]
        if self.backend == 'onnx':
            output = self.session.run(None, {self.session.get_inputs()[0].name: window})[0][0]
        else:
            import torch
            with torch.inference_mode():
                output = self.model(torch.from_numpy(window)).numpy()[0]
        if output.shape != (2,) or not np.isfinite(output).all():
            raise ValueError('AI produced invalid ENU velocity')
        return output.astype(float)

    def metadata(self):
        files = [self.path]
        if self.backend == 'onnx':
            import onnx
            model = onnx.load(str(self.path), load_external_data=False)
            locations = {entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == 'location'}
            files += [self.path.parent / location for location in sorted(locations)]
        return {'backend': self.backend, 'preprocessing': self.preprocessing,
                'files': {str(p): sha256(p) for p in files},
                'normalization_sha256': sha256(self.stats_path),
                'model_size_bytes': sum(p.stat().st_size for p in files),
                'startup_ms': self.startup_ms, 'sample_rate_hz': self.sample_rate,
                'threads': 1}


def prepare(recording, start, duration, calibration_seconds=5, aligned=False):
    mask = outage_mask(recording, start, duration)
    if calibration_seconds <= 0 or calibration_seconds >= start:
        raise ValueError('Calibration duration must be positive and end before the outage starts')
    end = int(np.searchsorted(recording.timestamps, calibration_seconds))
    prefix = np.flatnonzero(recording.valid[:end])
    if len(prefix) < 3:
        raise ValueError('Need valid GPS throughout an initial calibration prefix')
    rotation = np.eye(3) if aligned else calibrate(recording.accel[prefix], recording.gnss[prefix, 3], recording.timestamps[prefix])

    first = end
    if not recording.valid[first]:
        raise ValueError('No current initial GPS fix at calibration completion')
    origin = recording.gnss[first, :2].copy()
    return mask, rotation, first, origin


def load_map(path, origin):
    data = json.loads(Path(path).read_text())
    if not data.get('roads') or not all(k in data.get('origin', {}) for k in ('lat', 'lon')):
        raise ValueError('Full system requires a valid downloaded OSM road network with an origin')
    map_origin = np.array([data['origin']['lat'], data['origin']['lon']])
    offset = enu(map_origin, origin)
    scale = np.cos(np.deg2rad(origin[0])) / np.cos(np.deg2rad(map_origin[0]))
    roads = RoadNetwork()
    for road in data['roads']:
        points = np.asarray(road['points'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Invalid OSM road coordinates')
        points = points * [scale, 1] + offset
        roads.add_road(points, road['id'], road.get('name', ''), one_way=road.get('one_way', False))
    points = np.array([p for seg in roads.segments for p in (seg.start, seg.end)])
    if not ((points.min(axis=0) <= 0).all() and (points.max(axis=0) >= 0).all()):
        raise ValueError('Downloaded map does not cover the initial fix; download the recording area')
    roads.build_spatial_index()
    return create_map_matcher('geometric', roads, search_radius=30)


class Navigation:
    """Consumes only current IMU + an optional permitted GPS fix; no scoring references."""
    def __init__(self, config, rotation, initial_fix, origin, model=None, matcher=None):
        self.config, self.origin, self.model, self.matcher = config, origin, model, matcher
        self.filter = CausalFilter(rotation)
        self.buffer = deque(maxlen=model.window_size if model else 200)
        self.ekf = ExtendedKalmanFilter()
        self.ekf.initialize_from_gnss(enu(initial_fix[:2], origin), gnss_velocity(initial_fix), np.deg2rad(initial_fix[4]))
        self.zupt = ZUPTDetector()
        self.position = self.ekf.get_position()[:2]
        self.velocity = self.ekf.get_velocity()[:2]
        self.heading = self.ekf.get_heading()
        self.last_fix = None
        self.timings = {k: [] for k in ('tcn', 'ekf', 'map_matching', 'total_loop')}
        self.counters = {'ai_updates': 0, 'gnss_updates': 0, 'nhc_calls': 0, 'zupt_updates': 0, 'map_matches': 0}

    def step(self, accel, gyro, dt, gnss=None, fresh=False):
        started = perf_counter()
        cfg = self.config
        aligned, linear = self.filter.step(accel, gyro, dt)
        imu = np.r_[linear[:3] + [0, 0, 9.81], linear[3:]] if cfg.filtered else aligned
        self.buffer.append(linear)
        ai = None
        if cfg.ai and len(self.buffer) == self.buffer.maxlen:
            timer = perf_counter()
            ai = self.model.predict(self.buffer)
            self.timings['tcn'].append((perf_counter() - timer) * 1000)
            self.counters['ai_updates'] += 1
        timer = perf_counter()
        if cfg.ekf:
            self.ekf.dt = dt
            if gnss is None:
                self.ekf.set_gnss_denied()
            self.ekf.predict(imu[:3], imu[3:], ai, apply_nhc=cfg.nhc)
            self.counters['nhc_calls'] += int(cfg.nhc)
            if gnss is not None and fresh:
                self.ekf.update_gnss(enu(gnss[:2], self.origin), gnss_velocity(gnss))
                self.counters['gnss_updates'] += 1
            if cfg.zupt and self.zupt.update(aligned[:3], aligned[3:], dt,
                    estimated_speed=float(np.linalg.norm(self.ekf.get_velocity()[:2]))):
                self.ekf.update_zupt()
                self.counters['zupt_updates'] += 1
            self.position = self.ekf.get_position()[:2]
            self.velocity = self.ekf.get_velocity()[:2]
            self.heading = self.ekf.get_heading()
            mode = self.ekf.mode.value
            self.timings['ekf'].append((perf_counter() - timer) * 1000)
        else:
            self.heading = (self.heading + imu[5] * dt + np.pi) % (2 * np.pi) - np.pi
            a = np.array([imu[0] * np.sin(self.heading) + imu[1] * np.cos(self.heading),
                          imu[0] * np.cos(self.heading) - imu[1] * np.sin(self.heading)])
            if ai is not None:
                self.velocity = ai.copy()
                self.position += self.velocity * dt
                if np.linalg.norm(ai) > 0.5:
                    self.heading = float(np.arctan2(ai[0], ai[1]))
            else:
                self.position += self.velocity * dt + 0.5 * a * dt ** 2
                self.velocity += a * dt
            if gnss is not None and fresh:
                self.position = enu(gnss[:2], self.origin)
                self.velocity = gnss_velocity(gnss)
                self.heading = np.deg2rad(gnss[4])
                self.counters['gnss_updates'] += 1
            mode = 'dr' if gnss is None else 'gnss_ins'
        if cfg.map_matching and gnss is None:
            timer = perf_counter()
            result = self.matcher.match(self.position, heading=self.heading)
            if result.confidence > 0.8:
                self.position = 0.9 * self.position + 0.1 * result.snapped_position
                self.ekf.x[:2] = self.position
                self.counters['map_matches'] += 1
            self.timings['map_matching'].append((perf_counter() - timer) * 1000)
        self.timings['total_loop'].append((perf_counter() - started) * 1000)
        if not np.isfinite(np.r_[self.position, self.velocity, self.heading]).all():
            raise ValueError('Navigation diverged to non-finite values')
        return np.r_[self.position, self.velocity, self.heading], mode


def metrics(estimates, reference_position, reference_velocity, reference_heading, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {'reference_samples': 0, 'final_position_error_m': None, 'mean_position_error_m': None,
                'max_position_error_m': None, 'position_rmse_m': None, 'velocity_rmse_mps': None,
                'heading_rmse_deg': None, 'heading_samples': 0}
    error = np.linalg.norm(estimates[indices, :2] - reference_position[indices], axis=1)
    velocity_error = estimates[indices, 2:4] - reference_velocity[indices]
    moving = indices[np.linalg.norm(reference_velocity[indices], axis=1) >= 0.5]
    angle = (estimates[moving, 4] - reference_heading[moving] + np.pi) % (2*np.pi) - np.pi
    return {'reference_samples': len(indices), 'final_position_error_m': float(error[-1]),
            'mean_position_error_m': float(np.mean(error)), 'max_position_error_m': float(np.max(error)),
            'position_rmse_m': float(np.sqrt(np.mean(error ** 2))),
            'velocity_rmse_mps': float(np.sqrt(np.mean(np.sum(velocity_error ** 2, axis=1)))),
            'heading_rmse_deg': float(np.rad2deg(np.sqrt(np.mean(angle ** 2)))) if len(moving) else None,
            'heading_samples': len(moving)}


def run_replay(recording, config, start, duration, calibration_seconds=5, aligned=False, model=None, map_path=None):
    mask, rotation, first, origin = prepare(recording, start, duration, calibration_seconds, aligned)
    if config.ai and model is None:
        raise ValueError('This ablation requires a trained model; no synthetic AI fallback is allowed')
    if model and model.sample_rate and abs(recording.sample_rate / model.sample_rate - 1) > 0.1:
        raise ValueError('Recording sample rate differs from the model contract by more than 10%; resample or retrain')
    if config.ai and (start - recording.timestamps[first]) * recording.sample_rate < model.window_size:
        raise ValueError('Outage starts before the AI window is full; increase --outage-start')
    matcher = load_map(map_path, origin) if config.map_matching else None
    nav = Navigation(config, rotation, recording.gnss[first], origin, model, matcher)
    estimates = np.full((len(recording.timestamps), 5), np.nan)
    modes = ['calibrating'] * len(estimates)
    allowed = recording.valid & ~mask
    transitions = []
    previous_mode = None
    for i in range(first, len(estimates)):

        fix = recording.gnss[i].copy() if allowed[i] else None
        dt = 0.0 if i == first else recording.timestamps[i] - recording.timestamps[i - 1]
        estimates[i], modes[i] = nav.step(recording.accel[i], recording.gyro[i], dt, fix,
                                        bool(allowed[i] and (recording.fresh[i] or i == first)))
        if modes[i] != previous_mode:
            transitions.append({'time_s': float(recording.timestamps[i]), 'mode': modes[i]})
            previous_mode = modes[i]

    reference_position = enu(recording.gnss[:, :2], origin)
    heading = np.deg2rad(recording.gnss[:, 4])
    reference_velocity = recording.gnss[:, 3, None] * np.column_stack((np.sin(heading), np.cos(heading)))
    scored = recording.valid & recording.fresh & (np.arange(len(estimates)) >= first)
    scores = {name: metrics(estimates, reference_position, reference_velocity, heading, scored & selection)
              for name, selection in [('whole_trip', np.ones(len(mask), bool)), ('outage', mask),
                                       ('recovery', recording.timestamps >= start + duration)]}
    report = {'configuration': asdict(config), 'dataset': {'path': recording.path, 'sha256': recording.digest,
               'duration_s': float(recording.timestamps[-1]), 'sample_rate_hz': recording.sample_rate,
               'metadata': recording.metadata},
              'outage': {'start_s': start, 'requested_duration_s': duration, 'end_s': start + duration,
                         'samples': int(mask.sum()), 'interval': '[start, end)'},
              'calibration': {'seconds': calibration_seconds, 'aligned_input': aligned, 'rotation': rotation.tolist()},
              'reference': 'Recorded GNSS reference; not independent ground truth. Metrics use fresh valid fixes only.',
              'preprocessing': PREPROCESSING_ID, 'origin': origin.tolist(), 'metrics': scores,
              'transitions': transitions, 'component_counts': nav.counters,
              'timings': {key: timing_summary(values) for key, values in nav.timings.items()},
              'timing_scope': 'Actual calls, including first inference; navigation loop excludes scoring and output I/O',
              'machine': machine_info(), 'model': model.metadata() if model else None,
              'map_sha256': sha256(map_path) if config.map_matching else None}
    return report, estimates, modes, allowed
