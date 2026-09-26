import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

from navigation.interfaces import (
    AIVelocityMeasurement,
    IAIVelocityMeasurement,
)


@dataclass
class AIVelocityConfig:
    model_path: str = "simulator/model.onnx"
    stats_path: str = "simulator/norm_stats.json"
    window_size: int = 200
    sample_rate_hz: float = 10.0
    expected_dt_s: float = 0.1
    min_dt_s: float = 0.02
    max_dt_s: float = 0.5
    max_stale_age_s: float = 2.0
    base_variance: float = 0.16
    max_plausible_speed_mps: float = 65.0
    max_plausible_acceleration_mps2: float = 15.0
    stationary_accel_var_threshold: float = 0.05
    stationary_gyro_norm_threshold: float = 0.05
    stationary_variance: float = 0.01
    stationary_max_speed_mps: float = 1.0
    min_accel_norm_mps2: float = 2.0
    max_accel_norm_mps2: float = 50.0
    max_gyro_norm_radps: float = 30.0


class AIVelocityEngine(IAIVelocityMeasurement):
    def __init__(self, config: Optional[AIVelocityConfig] = None):
        self.config = config or AIVelocityConfig()
        self.session = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.last_valid_prediction: Optional[AIVelocityMeasurement] = None
        self.last_prediction_time: Optional[float] = None
        self.last_inference_latency_ms: float = 0.0
        self.total_inferences: int = 0
        self.failed_inferences: int = 0
        self.buffer = []
        self._initialize_model()

    def _initialize_model(self) -> None:
        model_path = Path(self.config.model_path)
        stats_path = Path(self.config.stats_path)

        if not model_path.is_file() or not stats_path.is_file():
            return

        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
            if self.mean.shape != (6,) or self.std.shape != (6,):
                self.mean = None
                self.std = None
                return

            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception:
            self.session = None
            self.mean = None
            self.std = None

    def is_ready(self) -> bool:
        return self.session is not None and self.mean is not None and self.std is not None

    def reset(self) -> None:
        self.buffer.clear()
        self.last_valid_prediction = None
        self.last_prediction_time = None
        self.last_inference_latency_ms = 0.0
        self.total_inferences = 0
        self.failed_inferences = 0

    def add_sample(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Optional[AIVelocityMeasurement]:
        accel_arr = np.asarray(accel, dtype=np.float64)[:3]
        gyro_arr = np.asarray(gyro, dtype=np.float64)[:3]
        ts = float(timestamp) if timestamp is not None else float(perf_counter())

        sample = np.concatenate([accel_arr, gyro_arr, [ts]])
        self.buffer.append(sample)
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)

        if len(self.buffer) == self.config.window_size:
            window_arr = np.array(self.buffer, dtype=np.float32)
            return self.estimate_velocity(window_arr[:, :6], timestamps=window_arr[:, 6])
        return None

    def estimate_velocity(
        self,
        imu_window: np.ndarray,
        *args,
        **kwargs,
    ) -> AIVelocityMeasurement:
        t_start = perf_counter()
        current_time = kwargs.get("current_time")
        timestamps = kwargs.get("timestamps")

        if len(args) >= 1 and timestamps is None:
            timestamps = args[0]
        if len(args) >= 2 and current_time is None:
            current_time = args[1]

        if imu_window is None:
            return self._create_invalid_measurement(
                reason="imu_window_is_none",
                status="INVALID_INPUT",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        window_arr = np.asarray(imu_window, dtype=np.float32)

        if window_arr.ndim != 2:
            return self._create_invalid_measurement(
                reason="invalid_dimensions",
                status="INVALID_INPUT",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        if window_arr.shape[1] == 7 and timestamps is None:
            timestamps = window_arr[:, 0].copy()
            window_arr = window_arr[:, 1:7]
        elif window_arr.shape[1] > 6:
            window_arr = window_arr[:, :6]

        if window_arr.shape[0] != self.config.window_size:
            return self._create_invalid_measurement(
                reason=f"expected_200_frames_got_{window_arr.shape[0]}",
                status="INSUFFICIENT_SAMPLES",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        if window_arr.shape[1] != 6:
            return self._create_invalid_measurement(
                reason=f"expected_6_channels_got_{window_arr.shape[1]}",
                status="INVALID_INPUT",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        if not np.all(np.isfinite(window_arr)):
            return self._create_invalid_measurement(
                reason="contains_nan_or_inf",
                status="INVALID_INPUT_NAN_INF",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        accel_mags = np.linalg.norm(window_arr[:, :3], axis=1)
        if np.any(accel_mags < self.config.min_accel_norm_mps2) or np.any(accel_mags > self.config.max_accel_norm_mps2):
            return self._create_invalid_measurement(
                reason="accel_magnitude_out_of_physical_range",
                status="INVALID_SENSOR_RANGE",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        gyro_mags = np.linalg.norm(window_arr[:, 3:6], axis=1)
        if np.any(gyro_mags > self.config.max_gyro_norm_radps):
            return self._create_invalid_measurement(
                reason="gyro_magnitude_out_of_physical_range",
                status="INVALID_SENSOR_RANGE",
                latency_ms=(perf_counter() - t_start) * 1000.0,
            )

        ts_valid = True
        ts_jitter = 1.0
        latest_ts = 0.0

        if timestamps is not None:
            ts_arr = np.asarray(timestamps, dtype=np.float64)
            if len(ts_arr) == self.config.window_size and np.all(np.isfinite(ts_arr)):
                diffs = np.diff(ts_arr)
                if np.any(diffs <= 0.0):
                    return self._create_invalid_measurement(
                        reason="non_monotonic_or_duplicate_timestamps",
                        status="INVALID_TIMESTAMPS",
                        latency_ms=(perf_counter() - t_start) * 1000.0,
                    )
                if np.any(diffs < self.config.min_dt_s) or np.any(diffs > self.config.max_dt_s):
                    return self._create_invalid_measurement(
                        reason="sampling_interval_out_of_bounds",
                        status="INVALID_SAMPLING_RATE",
                        latency_ms=(perf_counter() - t_start) * 1000.0,
                    )
                duration = ts_arr[-1] - ts_arr[0]
                if duration < 10.0 or duration > 35.0:
                    return self._create_invalid_measurement(
                        reason=f"temporal_window_duration_anomalous_{duration:.2f}s",
                        status="INVALID_WINDOW_DURATION",
                        latency_ms=(perf_counter() - t_start) * 1000.0,
                    )
                std_dt = float(np.std(diffs))
                ts_jitter = 1.0 + min(2.0, std_dt / self.config.expected_dt_s)
                latest_ts = float(ts_arr[-1])

                if current_time is not None:
                    c_time = float(current_time)
                    if c_time - latest_ts > self.config.max_stale_age_s:
                        return self._create_invalid_measurement(
                            reason=f"prediction_stale_age_{(c_time - latest_ts):.2f}s",
                            status="STALE_PREDICTION",
                            latency_ms=(perf_counter() - t_start) * 1000.0,
                            timestamp=latest_ts,
                        )

        if not self.is_ready() or self.mean is None or self.std is None or self.session is None or self.input_name is None:
            return self._create_invalid_measurement(
                reason="onnx_model_or_statistics_not_loaded",
                status="MODEL_UNAVAILABLE",
                latency_ms=(perf_counter() - t_start) * 1000.0,
                timestamp=latest_ts,
            )

        norm_window = ((window_arr - self.mean) / self.std)[None]

        try:
            self.total_inferences += 1
            outputs = self.session.run(None, {self.input_name: norm_window})
            out_arr = np.asarray(outputs[0])
            raw_vel = np.asarray(out_arr[0], dtype=np.float64)
        except Exception as e:
            self.failed_inferences += 1
            return self._create_invalid_measurement(
                reason=f"inference_exception: {str(e)}",
                status="INFERENCE_FAILED",
                latency_ms=(perf_counter() - t_start) * 1000.0,
                timestamp=latest_ts,
            )

        latency_ms = (perf_counter() - t_start) * 1000.0
        self.last_inference_latency_ms = latency_ms

        if raw_vel.shape != (2,) or not np.all(np.isfinite(raw_vel)):
            return self._create_invalid_measurement(
                reason="model_output_non_finite_or_malformed",
                status="INVALID_OUTPUT",
                latency_ms=latency_ms,
                timestamp=latest_ts,
            )

        v_north = float(raw_vel[0])
        v_east = float(raw_vel[1])
        predicted_speed = float(np.sqrt(v_north ** 2 + v_east ** 2))

        if predicted_speed > self.config.max_plausible_speed_mps:
            return self._create_invalid_measurement(
                reason=f"predicted_speed_excessive_{predicted_speed:.2f}mps",
                status="IMPLAUSIBLE_SPEED",
                latency_ms=latency_ms,
                timestamp=latest_ts,
            )

        accel_variance = float(np.mean(np.var(window_arr[-50:, :3], axis=0)))
        gyro_norm_recent = float(np.mean(np.linalg.norm(window_arr[-50:, 3:6], axis=1)))
        is_stationary = (
            accel_variance < self.config.stationary_accel_var_threshold
            and gyro_norm_recent < self.config.stationary_gyro_norm_threshold
        )

        var_scale = ts_jitter
        conf = 0.95

        if is_stationary:
            if predicted_speed > self.config.stationary_max_speed_mps:
                v_north = 0.0
                v_east = 0.0
                predicted_speed = 0.0
            variance_n = float(self.config.stationary_variance)
            variance_e = float(self.config.stationary_variance)
            conf = 0.99
            status = "VALID_STATIONARY"
        else:
            if predicted_speed > 30.0:
                var_scale *= 1.5
            if self.last_valid_prediction is not None and latest_ts > 0 and self.last_prediction_time is not None:
                dt_pred = latest_ts - self.last_prediction_time
                if dt_pred > 0:
                    prev_vn = self.last_valid_prediction.velocity_north
                    prev_ve = self.last_valid_prediction.velocity_east
                    dv = np.sqrt((v_north - prev_vn) ** 2 + (v_east - prev_ve) ** 2)
                    implied_accel = dv / dt_pred
                    if implied_accel > self.config.max_plausible_acceleration_mps2:
                        var_scale *= 3.0
                        conf = max(0.5, conf - 0.3)
            variance_n = float(self.config.base_variance * var_scale)
            variance_e = float(self.config.base_variance * var_scale)
            status = "VALID"

        measurement = AIVelocityMeasurement(
            velocity_north=v_north,
            velocity_east=v_east,
            variance_north=variance_n,
            variance_east=variance_e,
            latency_ms=latency_ms,
            is_valid=True,
            receptive_field_samples=self.config.window_size,
            timestamp=latest_ts,
            status=status,
            confidence=conf,
            reason=None,
            diagnostics={
                "predicted_speed": predicted_speed,
                "is_stationary": is_stationary,
                "accel_variance": accel_variance,
                "gyro_norm_recent": gyro_norm_recent,
                "variance_scale": var_scale,
                "latency_ms": latency_ms,
            },
        )
        self.last_valid_prediction = measurement
        self.last_prediction_time = latest_ts
        return measurement

    def _create_invalid_measurement(
        self,
        reason: str,
        status: str,
        latency_ms: float,
        timestamp: float = 0.0,
    ) -> AIVelocityMeasurement:
        return AIVelocityMeasurement(
            velocity_north=0.0,
            velocity_east=0.0,
            variance_north=100.0,
            variance_east=100.0,
            latency_ms=latency_ms,
            is_valid=False,
            receptive_field_samples=self.config.window_size,
            timestamp=timestamp,
            status=status,
            confidence=0.0,
            reason=reason,
            diagnostics={"error": reason, "status": status},
        )
