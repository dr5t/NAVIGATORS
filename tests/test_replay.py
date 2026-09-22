"""Algorithm tests use explicit synthetic fixtures; these are not road accuracy benchmarks."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from evaluation.recording import Recording, load_recording, outage_mask, enu, METERS_PER_DEGREE
from evaluation.replay import ABLATIONS, run_replay, metrics
from evaluation.preprocessing import CausalFilter, prepare_features
from data.data_loader import validate_training_splits
from navigation.ekf import ExtendedKalmanFilter, NavigationMode
from scripts.verify_onnx import compare_outputs
from replay import save_browser_trajectory


@pytest.fixture
def recording():
    ts = np.arange(0, 10, 0.02)

    accel = np.column_stack((0.5 * np.sin(ts * 13), 0.2 * np.cos(ts * 9), np.full(len(ts), 9.81)))
    gyro = np.zeros((len(ts), 3))
    gnss = np.column_stack((13.0326 + 2 * ts / METERS_PER_DEGREE,
        np.full(len(ts), 77.5582), np.zeros(len(ts)), np.full(len(ts), 2), np.zeros(len(ts)), np.full(len(ts), 2)))
    return Recording(ts, accel, gyro, gnss, np.ones(len(ts), bool), np.ones(len(ts), bool),
                     {'provenance': 'synthetic test fixture'}, 'fixture', 'fixture')


@pytest.fixture
def road_map(tmp_path):
    file = tmp_path / 'roads.json'
    file.write_text(json.dumps({'origin': {'lat': 13.0326, 'lon': 77.5582}, 'roads': [
        {'id': 'fixture', 'points': [[0, -100], [0, 100]], 'name': 'Synthetic test road'}]}))
    return file


class TestVelocityModel:
    __test__ = False
    window_size = 20
    sample_rate = 50
    def predict(self, window):

        return np.array([np.mean(np.asarray(window)[:, 1]) * 0.1, 2])
    def metadata(self):
        return {'provenance': 'synthetic test model'}


def run(recording, mode, road_map):
    return run_replay(recording, ABLATIONS[mode], start=3, duration=3, calibration_seconds=1,
                      aligned=True, model=TestVelocityModel() if ABLATIONS[mode].ai else None, map_path=road_map)


@pytest.mark.parametrize('mode', list(ABLATIONS))
def test_hidden_gps_never_changes_navigation(recording, road_map, mode):
    changed = replace(recording, gnss=recording.gnss.copy())
    hidden = (recording.timestamps >= 3) & (recording.timestamps < 6)
    changed.gnss[hidden] += 500
    original = run(recording, mode, road_map)
    altered = run(changed, mode, road_map)
    np.testing.assert_array_equal(original[1], altered[1])
    assert original[2] == altered[2]
    assert not original[3][hidden].any()
    first_denied = np.flatnonzero(hidden)[0]
    assert original[2][first_denied] == 'dr'


def test_replay_is_deterministic_and_uses_only_past_imu(recording, road_map):
    report, estimate, _, _ = run(recording, 'F', road_map)
    repeated = run(recording, 'F', road_map)
    np.testing.assert_array_equal(estimate, repeated[1])
    assert report['metrics'] == repeated[0]['metrics']
    changed = replace(recording, accel=recording.accel.copy())
    changed.accel[recording.timestamps >= 7, :2] += 100
    altered = run(changed, 'F', road_map)[1]
    np.testing.assert_array_equal(estimate[recording.timestamps < 7], altered[recording.timestamps < 7])


def test_ablations_switch_actual_components(recording, road_map):
    outputs = {key: run(recording, key, road_map) for key in ABLATIONS}
    for key, (report, *_rest) in outputs.items():
        config, counts = ABLATIONS[key], report['component_counts']
        assert bool(counts['ai_updates']) == config.ai
        assert bool(counts['nhc_calls']) == config.nhc
        assert bool(report['timings']['ekf']['count']) == config.ekf
        assert bool(report['timings']['map_matching']['count']) == config.map_matching
        assert report['machine']['label'] == 'Development-machine benchmark'
    assert not np.allclose(outputs['A'][1][150:300, :2], outputs['B'][1][150:300, :2])


@pytest.mark.parametrize('mode', ['A', 'G'])
def test_browser_export_preserves_estimates_and_missing_measurements(recording, road_map, tmp_path, mode):
    missing = (recording.timestamps >= 4) & (recording.timestamps < 5)
    recording.valid[missing] = False
    recording.gnss[missing] = np.nan
    recording.fresh[325:330] = False
    report, estimates, modes, allowed = run(recording, mode, road_map)
    path = tmp_path / 'trajectory.json'
    save_browser_trajectory(path, recording, estimates, modes, allowed, report)
    payload = json.loads(path.read_text())
    data, metadata = payload['data'], payload['metadata']
    indices = np.flatnonzero(np.isfinite(estimates).all(axis=1))
    np.testing.assert_allclose(enu(data['estimated_lat_lon'], report['origin']), estimates[indices, :2], atol=1e-8)
    np.testing.assert_array_equal(data['timestamps'], recording.timestamps[indices])
    np.testing.assert_array_equal(data['gnss_available'], allowed[indices])
    np.testing.assert_allclose(data['speed_estimated'], np.linalg.norm(estimates[indices, 2:4], axis=1))
    assert data['nav_mode'] == [modes[i] for i in indices]
    assert data['timestamps'][0] > 0
    for j, i in enumerate(indices):
        scored = bool(recording.valid[i] and recording.fresh[i])
        assert (data['true_lat_lon'][j] is not None) == scored
        assert (data['position_error'][j] is not None) == scored
        if scored:
            expected = np.linalg.norm(estimates[i, :2] - enu(recording.gnss[i, :2], report['origin']))
            assert data['position_error'][j] == pytest.approx(expected)
    for key in ['confidence', 'dr_drift_percent', 'zupt_active']:
        assert data[key] == [None] * len(indices)
    assert metadata['configuration'] == report['configuration']
    assert metadata['evaluation_metrics'] == report['metrics']
    assert metadata['dataset']['metadata']['provenance'] == 'synthetic test fixture'


def test_outage_bounds_are_seconds_and_never_silently_truncated(recording):
    mask = outage_mask(recording, 3, 3)
    assert recording.timestamps[mask][0] == 3
    assert recording.timestamps[mask][-1] < 6
    with pytest.raises(ValueError, match='exceeds'):
        outage_mask(recording, 3, 60)
    with pytest.raises(ValueError):
        outage_mask(recording, 3, 0)


def test_metrics_mean_rmse_and_wrapped_heading_are_distinct():
    estimate = np.array([[0, 0, 0, 1, np.deg2rad(359)], [4, 0, 0, 1, np.deg2rad(359)]])
    report = metrics(estimate, np.zeros((2, 2)), np.array([[0, 1], [0, 1]]),
                     np.full(2, np.deg2rad(1)), np.ones(2, bool))
    assert report['mean_position_error_m'] == 2
    assert report['position_rmse_m'] == pytest.approx(np.sqrt(8))
    assert report['heading_rmse_deg'] == pytest.approx(2)
    empty = metrics(estimate, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), np.zeros(2, bool))
    assert empty['mean_position_error_m'] is None


def test_csv_epoch_timestamps_and_stale_fixes(tmp_path, recording):
    file = tmp_path / 'trip.csv'
    data = np.column_stack((recording.timestamps + 1700000000, recording.accel, recording.gyro,
                            recording.gnss, recording.timestamps + 1700000000))
    np.savetxt(file, data, delimiter=',', header='timestamp,ax,ay,az,gx,gy,gz,lat,lon,alt,speed,heading,accuracy,gnss_timestamp', comments='')
    loaded = load_recording(file)
    assert loaded.timestamps[0] == 0
    np.testing.assert_allclose(loaded.timestamps, recording.timestamps, atol=1e-6)
    assert loaded.valid.all()
    data[200:, -1] = 1700000000
    np.savetxt(file, data, delimiter=',', header='timestamp,ax,ay,az,gx,gy,gz,lat,lon,alt,speed,heading,accuracy,gnss_timestamp', comments='')
    assert not load_recording(file).valid[200:].any()


def test_invalid_recording_and_duplicate_training_splits_fail(tmp_path):

    invalid = tmp_path / 'invalid.json'
    invalid.write_text(json.dumps({'data': {'timestamps': [0, 0.1, 0.2],
        'accel': [[0, 0, 9.81]] * 3, 'gyro': [[0, 0, 0]] * 3,
        'gnss': [[0, 0, 0, -1, 0, -1]] * 3}}))
    with pytest.raises(ValueError):
        load_recording(invalid)
    for split in ('train', 'val', 'test'):
        directory = tmp_path / split
        directory.mkdir()
        np.save(directory / 'same.npy', np.zeros((300, 8)))
    with pytest.raises(ValueError, match='leakage'):
        validate_training_splits(tmp_path)


def test_recovery_caps_every_fix_until_converged_including_velocity_coupling():
    ekf = ExtendedKalmanFilter()
    ekf.x[0] = 80
    ekf.P = np.eye(15) * 1000
    ekf.P[0, 3] = ekf.P[3, 0] = 200
    ekf.set_gnss_denied()
    for i in range(200):
        before = ekf.x.copy()
        ekf.update_gnss(np.zeros(2), np.array([0, 1]))
        if i < 5:
            assert ekf.mode == NavigationMode.REACQUISITION
        assert np.linalg.norm(ekf.x[:2] - before[:2]) <= 2.000001
        assert np.linalg.eigvalsh(ekf.P).min() > 0
        if ekf.mode == NavigationMode.GNSS_INS:
            assert np.linalg.norm(ekf.x[:2]) <= 3
            break
    else:
        pytest.fail('Recovery did not converge')


def test_parity_rejects_nonfinite_or_measurable_mismatch():
    assert compare_outputs([[1, 2]], [[1 + 1e-7, 2]])['passed']
    assert not compare_outputs([[1, 2]], [[1, 3]])['passed']
    assert not compare_outputs([[1, 2]], [[np.nan, 2]])['passed']


def test_training_features_match_replay_filter(recording):
    features, first, rotation = prepare_features(recording, 1, True)
    frontend = CausalFilter(rotation)
    for j, i in enumerate(range(first, len(recording.timestamps))):
        _, value = frontend.step(recording.accel[i], recording.gyro[i], 0 if i == first else 0.02)
        np.testing.assert_allclose(features[j], value, atol=1e-6)
