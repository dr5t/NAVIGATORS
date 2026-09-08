"""Tests for Sensor Adapters and External IMU interface."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.sensor_adapter import SmartphoneIMUAdapter, ExternalFOGIMUAdapter, SensorPacket
from navigation.alignment import PhoneVehicleAligner


def test_smartphone_adapter_with_aligner():
    aligner = PhoneVehicleAligner()
    aligner.set_orientation(0.0, 0.0, 0.0)

    adapter = SmartphoneIMUAdapter(sample_rate=10.0, aligner=aligner)
    acc = np.array([1.0, 0.5, 9.81])
    gyr = np.array([0.01, -0.02, 0.1])

    packet = adapter.process_raw(1.5, acc, gyr)
    assert isinstance(packet, SensorPacket)
    assert packet.source == "smartphone"
    assert packet.sample_rate == 10.0
    assert np.allclose(packet.accel, acc)
    assert np.allclose(packet.gyro, gyr)


def test_external_fog_throughput():
    fog = ExternalFOGIMUAdapter(sample_rate=200.0)
    res = fog.benchmark_throughput(iterations=5000)
    assert res["achievable_hz"] > 1000.0, f"Expected > 1 kHz throughput, got {res['achievable_hz']:.1f} Hz"
    assert res["avg_latency_us"] < 100.0
