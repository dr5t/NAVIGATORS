"""
Navigators IDR - ONNX Export
Export trained PyTorch models to ONNX format for edge deployment.

ONNX models can run on:
    - Android via ONNX Runtime Mobile
    - iOS via Core ML (via onnx-coreml converter)
    - Edge devices via TensorRT or ONNX Runtime
"""

import os
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    window_size: int = 200,
    input_channels: int = 6,
    batch_size: int = 1,
    opset_version: int = 13,
    optimize: bool = True,
) -> str:
    """
    Export a PyTorch velocity estimation model to ONNX.

    Args:
        model: Trained TCN or LSTM model.
        output_path: Path for the output .onnx file.
        window_size: Input sequence length.
        input_channels: Number of input channels (6 for IMU).
        batch_size: Export batch size (1 for real-time inference).
        opset_version: ONNX opset version.
        optimize: Whether to run ONNX optimization passes.

    Returns:
        Path to the exported ONNX model.
    """
    if not HAS_ONNX:
        raise ImportError("onnx and onnxruntime required. Install with: pip install onnx onnxruntime")

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    dummy_input = torch.randn(batch_size, window_size, input_channels).to(device)

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["velocity"],
        dynamic_axes={
            "imu_window": {0: "batch_size"},
            "velocity": {0: "batch_size"},
        },
    )

    print(f"[ONNX] Exported model to: {output_path}")

    import onnx
    import onnx.checker
    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX] Model validation passed ✓")

    # Optimize
    if optimize:
        from onnx import optimizer  # type: ignore
        try:
            passes = ["eliminate_identity", "fuse_bn_into_conv", "fuse_consecutive_transposes"]
            optimized = optimizer.optimize(onnx_model, passes)
            onnx.save(optimized, output_path)
            print("[ONNX] Optimization passes applied ✓")
        except Exception:
            print("[ONNX] Optimization skipped (optional passes unavailable)")

    # Report size
    file_size = os.path.getsize(output_path)
    print(f"[ONNX] Model size: {file_size / 1024:.1f} KB")

    return output_path


def verify_onnx_model(
    onnx_path: str,
    pytorch_model: nn.Module,
    window_size: int = 200,
    input_channels: int = 6,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify that ONNX model produces identical outputs to PyTorch model.

    Args:
        onnx_path: Path to ONNX model.
        pytorch_model: Original PyTorch model.
        window_size: Input sequence length.
        input_channels: Number of input channels.
        tolerance: Maximum acceptable difference.

    Returns:
        True if outputs match within tolerance.
    """
    if not HAS_ONNX:
        raise ImportError("onnxruntime required for verification")

    pytorch_model.eval()

    # Generate test input
    test_input = np.random.randn(1, window_size, input_channels).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_output = pytorch_model(pt_input).numpy()

    import onnxruntime as ort
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(None, {"imu_window": test_input})[0]

    # Compare
    max_diff = np.max(np.abs(pt_output - ort_output))
    match = max_diff < tolerance

    print(f"[ONNX] Verification: max_diff={max_diff:.2e}, "
          f"tolerance={tolerance:.2e}, {'PASS ✓' if match else 'FAIL ✗'}")

    return match


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for edge deployment.

    Wraps the ONNX model for easy real-time inference on edge devices.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to the ONNX model file.
        """
        if not HAS_ONNX:
            raise ImportError("onnxruntime required. Install with: pip install onnxruntime")

        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        print(f"[ONNX Engine] Loaded model: input_shape={input_shape}")

    def predict(self, imu_window: np.ndarray) -> np.ndarray:
        """
        Run inference on a single IMU window.

        Args:
            imu_window: (window_size, 6) or (1, window_size, 6) IMU data.

        Returns:
            (2,) predicted velocity [v_north, v_east].
        """
        if imu_window.ndim == 2:
            imu_window = imu_window[np.newaxis, ...]  # Add batch dim

        imu_window = imu_window.astype(np.float32)
        result = self.session.run(None, {self.input_name: imu_window})  # type: ignore
        return result[0][0]  # type: ignore
