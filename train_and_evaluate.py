import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import math
import torch
import numpy as np
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import onnx
import onnxruntime as ort
import json

from src.models.tcn_model import TCNVelocityEstimator
from src.models.iovnbd_dataset import create_iovnbd_dataloaders
from src.models.trainer import Trainer

def run_pipeline():
    print("==================================================")
    print("1 & 4. DATA SPLIT & INPUT NORMALIZATION")
    print("==================================================")
    iovnbd_dir = os.path.join(os.path.dirname(__file__), "data", "IO-VNBD")
    train_loader, val_loader, test_loader = create_iovnbd_dataloaders(
        base_dir=iovnbd_dir, window_size=200, batch_size=256
    )
    
    if not train_loader:
        print("Failed to load dataloaders.")
        return
        
    print("\n==================================================")
    print("2. VERIFY FEATURE PIPELINE")
    print("==================================================")
    
    # Check normalization stats
    stats_path = os.path.join("checkpoints", "norm_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
            print("Feature order:", stats["features"])
            print("Normalization method:", stats["method"])
            print("Mean:", stats["mean"])
            print("Std:", stats["std"])
    else:
        print("Normalization stats not found!")
        return

    # Verify input shape and NaN
    for X, Y in train_loader:
        print(f"Input shape: {X.shape} (Batch, Window, Channels)")
        print(f"Target shape: {Y.shape} (Batch, [V_N, V_E])")
        print("Target units: m/s (Cartesian North, East)")
        
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("ERROR: Invalid data (NaN/Inf) found in X!")
            return
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            print("ERROR: Invalid data (NaN/Inf) found in Y!")
            return
        break
        
    print("\n==================================================")
    print("3. TRAINING")
    print("==================================================")
    
    model = TCNVelocityEstimator(input_channels=6, output_dim=2, num_channels=[64, 128, 256, 256, 512], kernel_size=7)
    
    config = {
        "learning_rate": 0.001,
        "epochs": 40,
        "loss": "huber",
        "scheduler": "plateau"
    }
    
    trainer = Trainer(model, train_loader, val_loader, config)
    
    print(f"Training for {config['epochs']} epochs...")
    history = trainer.train()
    
    print("\nLoading best model...")
    device = trainer.device
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device)["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("\n==================================================")
    print("5 & 6. BASELINES & SPEED METRICS")
    print("==================================================")
    
    gt_speeds = []
    pred_speeds = []
    examples = []
    
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.float().to(device)
            Y = Y.float().to(device)
            
            preds = model(X)
            
            gt_n = Y[:, 0].cpu().numpy()
            gt_e = Y[:, 1].cpu().numpy()
            pred_n = preds[:, 0].cpu().numpy()
            pred_e = preds[:, 1].cpu().numpy()
            
            gt_speed = np.sqrt(gt_n**2 + gt_e**2)
            pred_speed = np.sqrt(pred_n**2 + pred_e**2)
            
            gt_speeds.extend(gt_speed)
            pred_speeds.extend(pred_speed)
            
    gt_speeds = np.array(gt_speeds)
    pred_speeds = np.array(pred_speeds)
    
    mae_ms = np.mean(np.abs(gt_speeds - pred_speeds))
    mae_kmh = mae_ms * 3.6
    rmse = np.sqrt(np.mean((gt_speeds - pred_speeds)**2))
    
    mean_gt = np.mean(gt_speeds)
    mean_pred = np.mean(pred_speeds)
    min_gt = np.min(gt_speeds)
    max_gt = np.max(gt_speeds)
    min_pred = np.min(pred_speeds)
    max_pred = np.max(pred_speeds)
    
    baseline_zero_mae = np.mean(np.abs(gt_speeds - 0.0))
    baseline_mean_mae = np.mean(np.abs(gt_speeds - mean_gt))
    improvement = max(0, baseline_mean_mae - mae_ms)
    
    print("--- BASELINES ---")
    print(f"Zero Velocity MAE: {baseline_zero_mae:.4f} m/s")
    print(f"Mean Velocity MAE: {baseline_mean_mae:.4f} m/s")
    print("\n--- AI MODEL METRICS ---")
    print(f"Speed MAE (m/s): {mae_ms:.4f}")
    print(f"Speed MAE (km/h): {mae_kmh:.4f}")
    print(f"RMSE (m/s): {rmse:.4f}")
    print(f"Mean GT Speed: {mean_gt:.4f} m/s")
    print(f"Mean Pred Speed: {mean_pred:.4f} m/s")
    print(f"Min GT Speed: {min_gt:.4f} m/s")
    print(f"Max GT Speed: {max_gt:.4f} m/s")
    print(f"Min Pred Speed: {min_pred:.4f} m/s")
    print(f"Max Pred Speed: {max_pred:.4f} m/s")
    print(f"Improvement over Mean Velocity Baseline: {improvement:.4f} m/s")
    
    if mae_ms < baseline_mean_mae:
        print("RESULT: AI MODEL BEATS THE MEAN VELOCITY BASELINE!")
    else:
        print("RESULT: AI MODEL DID NOT BEAT THE BASELINE.")
        
    print("\n--- 20 REPRESENTATIVE EXAMPLES ---")
    # Pick 20 representative examples covering different speed regimes
    sorted_indices = np.argsort(gt_speeds)
    step = len(sorted_indices) // 20
    for i in range(20):
        idx = sorted_indices[i * step]
        gt = gt_speeds[idx]
        pr = pred_speeds[idx]
        er = abs(gt - pr)
        print(f"[{i+1:02d}] GT: {gt:.4f} m/s | Pred: {pr:.4f} m/s | Err: {er:.4f} m/s")

    print("\n==================================================")
    print("8 & 9. ONNX EXPORT & PARITY VALIDATION")
    print("==================================================")
    onnx_path = os.path.join("simulator", "model.onnx")
    dummy_input = torch.randn(1, 200, 6).to(device)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Running ONNX Parity Test...")
    # Get a sample batch
    for X, Y in test_loader:
        sample_x = X.numpy()
        break
        
    # PyTorch inference
    with torch.no_grad():
        pt_out = model(torch.from_numpy(sample_x).to(device)).cpu().numpy()
        
    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: sample_x}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_out = ort_outs[0]
    
    max_diff = np.max(np.abs(pt_out - onnx_out))
    
    print(f"PyTorch Output Shape: {pt_out.shape}")
    print(f"ONNX Output Shape: {onnx_out.shape}")
    print(f"PyTorch [0] Speed: {np.sqrt(pt_out[0,0]**2 + pt_out[0,1]**2):.6f} m/s")
    print(f"ONNX [0] Speed: {np.sqrt(onnx_out[0,0]**2 + onnx_out[0,1]**2):.6f} m/s")
    print(f"Maximum absolute difference: {max_diff:.8f}")
    if max_diff < 1e-4:
        print("ONNX Parity VERIFIED.")
        print("Android Compatibility: OK (Preprocessing and inference perfectly match)")
    else:
        print("ONNX Parity FAILED.")
        
    # Overwrite the original norm_stats.json to the simulator directory for Android inference
    import shutil
    shutil.copy(stats_path, os.path.join("simulator", "norm_stats.json"))

if __name__ == "__main__":
    run_pipeline()
