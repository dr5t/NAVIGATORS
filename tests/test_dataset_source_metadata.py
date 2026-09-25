"""Attribution survives legacy benchmark windowing and phone feature export."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_external_benchmark_window_metadata(monkeypatch):
    import src.models.iovnbd_dataset as module

    def parse(sensor, reference):
        return np.arange(120, dtype=float).reshape(20, 6), np.zeros((20, 2))

    monkeypatch.setattr(module, "parse_synchronized_iovnbd", parse)
    dataset = module.IOVNBDDataset([("external/S_test.csv", "external/V_test.csv")], window_size=4, stride=2)
    x, y = dataset[1]
    metadata = dataset.get_sample_metadata(1)
    assert x.shape == (4, 6) and y.shape == (2,)
    assert metadata["source_dataset"] == "IO-VNBD"
    assert metadata["source_kind"] == "external_benchmark"
    assert metadata["sensor_file"] == "external/S_test.csv"
    assert (metadata["start"], metadata["stop"]) == (2, 6)


def test_legacy_processed_features_preserve_source(tmp_path):
    from data.pipeline import process_trip

    ts = np.arange(0, 10, .02)
    source = tmp_path / "external.json"
    source.write_text(json.dumps({
        "metadata": {"source_dataset": "External fixture", "source_kind": "external_benchmark",
                     "provenance": "Synthetic test of external attribution"},
        "data": {"timestamps": ts.tolist(), "accel": np.tile([0, 0, 9.81], (len(ts), 1)).tolist(),
                 "gyro": np.zeros((len(ts), 3)).tolist(),
                 "gnss": np.column_stack((13 + ts / 111195, np.full(len(ts), 77), np.zeros(len(ts)),
                                          np.ones(len(ts)), np.zeros(len(ts)), np.full(len(ts), 3))).tolist()}}))
    output = tmp_path / "features.npy"
    assert process_trip(source, output, calibration_seconds=1, aligned=True)
    metadata = json.loads(Path(str(output) + ".json").read_text())
    assert metadata["source_dataset"] == "External fixture"
    assert metadata["source_kind"] == "external_benchmark"
    assert metadata["source_sha256"]
