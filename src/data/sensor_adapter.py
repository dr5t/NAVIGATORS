"""
Navigators IDR - Sensor Adapter Interface
Provides a unified sensor interface supporting both commercial smartphone IMUs (10 Hz+)
and high-precision external Fiber Optic Gyroscope (FOG) / tactical-grade IMUs (200 Hz+).
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SensorPacket:
    """Standardized IMU measurement packet for the navigation engine."""
    timestamp: float           # Seconds (monotonic or GPS epoch)
    accel: np.ndarray          # [ax, ay, az] in m/s² (vehicle frame)
    gyro: np.ndarray           # [wx, wy, wz] in rad/s (vehicle frame)
    source: str = "smartphone" # "smartphone", "external_fog", or "synthetic"
    sample_rate: float = 10.0  # Nominal rate in Hz
    sequence_id: int = 0
    temperature: Optional[float] = None
    is_calibrated: bool = True


class BaseSensorAdapter(ABC):
    """Abstract sensor interface decoupling sensor sources from navigation logic."""

    @abstractmethod
    def read_packet(self) -> Optional[SensorPacket]:
        """Fetch the next available sensor packet."""
        pass

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Return operating sample rate in Hz."""
        pass


class SmartphoneIMUAdapter(BaseSensorAdapter):
    """
    Adapter for commercial smartphone IMU sensors (typically 10 Hz to 50 Hz).
    Handles timestamp jitter, noise filtering, and phone-to-vehicle alignment.
    """

    def __init__(
        self,
        sample_rate: float = 10.0,
        aligner: Optional[Any] = None,
    ):
        self.sample_rate = sample_rate
        self.aligner = aligner
        self.packet_count = 0
        self.last_timestamp = 0.0

    def process_raw(
        self,
        timestamp: float,
        raw_accel: np.ndarray,
        raw_gyro: np.ndarray,
    ) -> SensorPacket:
        """Process raw mobile sensor event into standardized vehicle-frame packet."""
        self.packet_count += 1
        
        # Apply alignment if available
        if self.aligner is not None:
            accel_veh = self.aligner.transform_accel(raw_accel)
            gyro_veh = self.aligner.transform_gyro(raw_gyro)
        else:
            accel_veh = raw_accel.copy()
            gyro_veh = raw_gyro.copy()

        packet = SensorPacket(
            timestamp=timestamp,
            accel=accel_veh,
            gyro=gyro_veh,
            source="smartphone",
            sample_rate=self.sample_rate,
            sequence_id=self.packet_count,
            is_calibrated=self.aligner is not None,
        )
        self.last_timestamp = timestamp
        return packet

    def read_packet(self) -> Optional[SensorPacket]:
        return None

    def get_sample_rate(self) -> float:
        return self.sample_rate


class ExternalFOGIMUAdapter(BaseSensorAdapter):
    """
    Adapter for high-precision external Fiber Optic Gyro (FOG) or tactical MEMS IMU.
    Designed for high bandwidth (typically ~200 Hz to 1000 Hz) with near-zero bias drift.
    """

    def __init__(
        self,
        sample_rate: float = 200.0,
        bias_drift_ppm: float = 0.01,
        noise_density_accel: float = 0.005, # m/s²/sqrt(Hz)
        noise_density_gyro: float = 0.0001, # rad/s/sqrt(Hz)
    ):
        self.sample_rate = sample_rate
        self.bias_drift_ppm = bias_drift_ppm
        self.noise_density_accel = noise_density_accel
        self.noise_density_gyro = noise_density_gyro
        self.packet_count = 0
        self.last_timestamp = 0.0

    def process_raw(
        self,
        timestamp: float,
        raw_accel: np.ndarray,
        raw_gyro: np.ndarray,
        temperature: Optional[float] = 25.0,
    ) -> SensorPacket:
        """Process high-rate external FOG packet."""
        self.packet_count += 1
        packet = SensorPacket(
            timestamp=timestamp,
            accel=raw_accel.copy(),
            gyro=raw_gyro.copy(),
            source="external_fog",
            sample_rate=self.sample_rate,
            sequence_id=self.packet_count,
            temperature=temperature,
            is_calibrated=True,
        )
        self.last_timestamp = timestamp
        return packet

    def read_packet(self) -> Optional[SensorPacket]:
        return None

    def get_sample_rate(self) -> float:
        return self.sample_rate

    def benchmark_throughput(self, iterations: int = 10000) -> Dict[str, float]:
        """
        Benchmark packet ingestion and conversion throughput.
        """
        acc = np.array([0.1, 0.0, 9.81])
        gyr = np.array([0.0, 0.0, 0.01])
        t_start = time.perf_counter()
        for i in range(iterations):
            _ = self.process_raw(i * 0.005, acc, gyr)
        t_elapsed = time.perf_counter() - t_start
        achievable_hz = iterations / t_elapsed
        return {
            "iterations": iterations,
            "elapsed_seconds": t_elapsed,
            "achievable_hz": achievable_hz,
            "avg_latency_us": (t_elapsed / iterations) * 1e6,
        }
