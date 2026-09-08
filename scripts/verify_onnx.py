#!/usr/bin/env python3
"""
Navigators IDR — ONNX Export and Validation Script
Exports the trained PyTorch model to ONNX and verifies inference parity.
"""

import os
import sys
import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.tcn_model import TCNVelocityEstimator

def verify_and_export():
    print("="*60)
    print("  NAVIGATORS IDR — ONNX EXPORT & VALIDATION (PHASE 4)")
    print("="*60)
    
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "simulator", "model.onnx")
    
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Instantiate the model with the same config used in train.py
    model = TCNVelocityEstimator(
        input_channels=6, 
        num_channels=[32, 64, 128], 
        kernel_size=5, 
        dropout=0.2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Export to ONNX
    print(f"Exporting model to ONNX -> {onnx_path}...")
    dummy_input = torch.randn(1, 200, 6) # (batch, window_size, channels)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print("Export successful.")
    
    # Verify Parity
    print("Verifying PyTorch vs ONNX parity...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    # Generate random test input
    x = torch.randn(32, 200, 6)
    
    # PyTorch inference
    with torch.no_grad():
        pt_out = model(x).numpy()
        
    # ONNX inference
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: x.numpy()})[0]
    
    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    
    print(f"Max Absolute Difference:  {max_diff:.8f}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    
    tolerance = 1e-5
    passed = max_diff < tolerance
    print(f"Tolerance: {tolerance}")
    print(f"Pass: {'YES ✓' if passed else 'NO ✗'}")
    
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    verify_and_export()
