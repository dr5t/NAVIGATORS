"""Strict phone recording ingestion. Units: seconds, m/s², rad/s, WGS84 degrees."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

METERS_PER_DEGREE = 6371000.0 * np.pi / 180
CSV_COLUMNS = ('timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
               'lat', 'lon', 'alt', 'speed', 'heading', 'accuracy')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Recording:
    timestamps: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray
    gnss: np.ndarray  # lat, lon, alt, speed m/s, heading degrees, accuracy meters
    valid: np.ndarray
    fresh: np.ndarray
    metadata: dict
    path: str = ''
    digest: str = ''

    @property
    def sample_rate(self):
        return float(1 / np.median(np.diff(self.timestamps)))


def load_recording(path, gyro_order=None):
    path = Path(path)
    metadata = {}
    fix_time = None
    if path.suffix.lower() == '.csv':
        rows = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding='utf-8')
        if not set(CSV_COLUMNS) <= set(rows.dtype.names or ()):
            raise ValueError('CSV requires columns: ' + ','.join(CSV_COLUMNS))
        rows = np.atleast_1d(rows)
        ts = rows['timestamp']
        accel = np.column_stack([rows[k] for k in ('ax', 'ay', 'az')])
        gyro = np.column_stack([rows[k] for k in ('gx', 'gy', 'gz')])
        gnss = np.column_stack([rows[k] for k in CSV_COLUMNS[7:]])
        if 'gnss_timestamp' in rows.dtype.names:
            fix_time = rows['gnss_timestamp']
    elif path.suffix.lower() == '.json':
        payload = json.loads(path.read_text())
        metadata = payload.get('metadata', {})
        data = payload.get('data', {})
        try:
            ts, accel, gyro, gnss = [np.asarray(data[k], dtype=float)
                                     for k in ('timestamps', 'accel', 'gyro', 'gnss')]
        except (KeyError, TypeError) as error:
            raise ValueError('JSON requires data.timestamps, accel, gyro, gnss arrays') from error
        if 'gnss_timestamps' in data:
            fix_time = np.asarray(data['gnss_timestamps'], dtype=float)
    else:
        raise ValueError('Use a raw phone .json or .csv recording, not processed windows')
    if ts.ndim != 1 or len(ts) < 3 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Recording timestamps must be finite, strictly increasing seconds (at least 3 samples)')
    if accel.shape != (len(ts), 3) or gyro.shape != accel.shape or gnss.shape != (len(ts), 6):
        raise ValueError('Sensor arrays must have equal lengths: accel/gyro Nx3, GNSS Nx6')
    if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
        raise ValueError('IMU contains missing or non-finite samples; repair the recording first')
    if np.max(np.diff(ts)) > 1:
        raise ValueError('IMU gap exceeds 1 second; split the recording at gaps before replay')
    order = gyro_order or metadata.get('gyro_order')
    if order is None:
        order = 'alpha-beta-gamma' if 'alpha' in metadata.get('columns', {}).get('gyro', '') else 'xyz'
    if order not in ('xyz', 'alpha-beta-gamma'):
        raise ValueError('gyro_order must be xyz or alpha-beta-gamma')
    if order == 'alpha-beta-gamma':
        gyro = gyro[:, [1, 2, 0]]  # DeviceMotion beta=x, gamma=y, alpha=z
    valid = (np.isfinite(gnss).all(axis=1) & (np.abs(gnss[:, 0]) < 85)
             & (np.abs(gnss[:, 1]) <= 180) & (gnss[:, 3] >= 0)
             & (gnss[:, 3] <= 100) & (gnss[:, 4] >= 0) & (gnss[:, 4] < 360)
             & (gnss[:, 5] > 0) & (gnss[:, 5] <= 50))
    fresh = np.r_[True, np.any(np.diff(gnss, axis=0) != 0, axis=1)]
    notes = []
    if fix_time is not None:
        if fix_time.shape != ts.shape:
            raise ValueError('GNSS timestamps must match the IMU timestamps length')
        valid &= np.isfinite(fix_time) & (ts - fix_time >= -0.1) & (ts - fix_time <= 3)
        fresh = np.r_[True, np.diff(fix_time) != 0]
    else:
        # Older recorder versions repeat the last fix; expire unchanged values after 3s.
        last_change = np.maximum.accumulate(np.where(fresh, ts, ts[0]))
        valid &= ts - last_change <= 3
        notes.append('No GNSS timestamps: freshness inferred from changed GNSS values')
    indices = np.flatnonzero(valid & fresh)
    if len(indices) < 2:
        raise ValueError('Need at least two valid, fresh GNSS fixes with nonnegative speed and positive accuracy')
    fixes = gnss[indices]
    dt = np.diff(ts[indices])
    delta = np.diff(fixes[:, :2], axis=0) * METERS_PER_DEGREE
    delta[:, 1] *= np.cos(np.deg2rad(fixes[:-1, 0]))
    # Allow 100 m/s plus both fixes' uncertainty. Reject corrupt geodetic coordinates.
    if np.any(np.linalg.norm(delta, axis=1) > 100 * dt + fixes[:-1, 5] + fixes[1:, 5]):
        raise ValueError('Implausible GNSS coordinate jumps: recording is not suitable for navigation evaluation')
    return Recording(ts - ts[0], accel, gyro, gnss, valid, fresh,
                     {**metadata, 'gyro_order_used': order, 'ingestion_notes': notes,
                      'provenance': metadata.get('provenance', 'unverified recording')},
                     str(path.resolve()), sha256(path))


def outage_mask(recording, start, duration):
    if not np.isfinite([start, duration]).all() or start < 0 or duration <= 0:
        raise ValueError('Outage start must be nonnegative and duration must be positive seconds')
    if start + duration >= recording.timestamps[-1]:
        raise ValueError(f'Requested outage [{start:g}, {start + duration:g}) s exceeds '
                         f'{recording.timestamps[-1]:g}s trip or leaves no recovery samples')
    mask = (recording.timestamps >= start) & (recording.timestamps < start + duration)
    if not mask.any():
        raise ValueError('Outage contains no recording samples')
    return mask


def enu(lat_lon, origin):
    delta = (np.asarray(lat_lon) - origin) * METERS_PER_DEGREE
    return np.stack((delta[..., 1] * np.cos(np.deg2rad(origin[0])), delta[..., 0]), axis=-1)


def gnss_velocity(fix):
    heading = np.deg2rad(fix[4])
    return np.array([fix[3] * np.sin(heading), fix[3] * np.cos(heading)])
