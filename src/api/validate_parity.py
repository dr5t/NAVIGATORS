import os
import sys
import torch
import onnxruntime as ort
import numpy as np

# Add src to path so we can import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.trainer import Trainer, create_model

def validate_parity():
    print("=" * 50)
    print("ONNX vs PyTorch Parity Validation")
    print("=" * 50)

    # 1. Load PyTorch model
    pt_path = os.path.join("checkpoints", "best_model.pt")
    if not os.path.exists(pt_path):
        print(f"Error: Could not find PyTorch model at {pt_path}")
        return
        
    print("Loading PyTorch model...")
    model, checkpoint = Trainer.load_checkpoint(pt_path, device=torch.device("cpu"))
    model.eval()
    
    # 2. Load ONNX model
    onnx_path = os.path.join("simulator", "model.onnx")
    if not os.path.exists(onnx_path):
        print(f"Error: Could not find ONNX model at {onnx_path}")
        return
        
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # 3. Generate dummy input matching window size
    window_size = 200
    # The ONNX export used dynamic axes for batch size, but we'll test batch=1
    dummy_input = torch.randn(1, window_size, 6, dtype=torch.float32)
    numpy_input = dummy_input.numpy()
    
    # 4. PyTorch Inference
    print("Running PyTorch Inference...")
    with torch.no_grad():
        pt_out = model(dummy_input)
    
    # 5. ONNX Inference
    print("Running ONNX Inference...")
    # Get input name from the session
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: numpy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # ONNX runtime returns a list of outputs; our model returns 1 tensor
    onnx_out = ort_outs[0]
    
    pt_out_np = pt_out.numpy()
    
    print("-" * 50)
    print("PyTorch Output:")
    print(pt_out_np)
    print("ONNX Output:")
    print(onnx_out)
    
    # 6. Compare
    abs_diff = np.abs(pt_out_np - onnx_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print("-" * 50)
    print(f"Max Absolute Difference:  {max_diff:.8f}")
    print(f"Mean Absolute Difference: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        print("\n✅ PARITY VALIDATION PASSED!")
    else:
        print("\n❌ PARITY VALIDATION FAILED! Differences exceed 1e-4 tolerance.")
        
if __name__ == "__main__":
    validate_parity()
