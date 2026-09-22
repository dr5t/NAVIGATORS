"""Causal IMU frontend; alignment is fitted only on the pre-outage calibration prefix."""
from collections import deque
import numpy as np

PREPROCESSING_ID = 'causal-median5-lowpass20-gravity-v1'


def calibrate(accel, speeds, timestamps):
    gravity = np.mean(accel, axis=0)
    magnitude = np.linalg.norm(gravity)
    if not 7 < magnitude < 12:
        raise ValueError('Calibration needs accelerometer readings including gravity (~9.81 m/s²)')
    z = gravity / magnitude
    horizontal = accel - np.outer(accel @ z, z)
    rate = np.diff(speeds) / np.diff(timestamps)
    accelerating = np.flatnonzero(rate > 0.5) + 1
    if len(accelerating) < 3:
        raise ValueError('Calibration needs forward acceleration with valid GPS; use --aligned only for known vehicle-frame data')
    x = np.mean(horizontal[accelerating], axis=0)
    x -= (x @ z) * z
    if np.linalg.norm(x) < 0.05:
        raise ValueError('Cannot resolve vehicle forward axis from calibration prefix')
    x /= np.linalg.norm(x)
    return np.stack((x, np.cross(z, x), z))


class CausalFilter:
    def __init__(self, rotation):
        self.rotation = np.asarray(rotation)
        self.history = deque(maxlen=5)
        self.filtered = None

    def step(self, accel, gyro, dt):
        aligned = np.r_[self.rotation @ accel, self.rotation @ gyro]
        self.history.append(aligned)
        median = np.median(self.history, axis=0)
        gain = 1 - np.exp(-2 * np.pi * 20 * dt)
        self.filtered = median.copy() if self.filtered is None else self.filtered + gain * (median - self.filtered)
        linear = self.filtered.copy()
        linear[2] -= 9.81
        return aligned, linear


def prepare_features(recording, calibration_seconds=5, aligned=False):
    """Same causal features as replay; never process the entire trip with filtfilt."""
    end = int(np.searchsorted(recording.timestamps, calibration_seconds))
    if calibration_seconds <= 0 or end >= len(recording.timestamps) - 1:
        raise ValueError('Recording must extend beyond a positive calibration prefix')
    valid = np.flatnonzero(recording.valid[:end])
    if len(valid) < 3:
        raise ValueError('Not enough valid GPS samples in calibration prefix')
    rotation = np.eye(3) if aligned else calibrate(recording.accel[valid], recording.gnss[valid, 3], recording.timestamps[valid])
    frontend = CausalFilter(rotation)
    features = []
    for i in range(end, len(recording.timestamps)):
        dt = 0 if i == end else recording.timestamps[i] - recording.timestamps[i - 1]
        _, feature = frontend.step(recording.accel[i], recording.gyro[i], dt)
        features.append(feature)
    return np.asarray(features, dtype=np.float32), end, rotation
