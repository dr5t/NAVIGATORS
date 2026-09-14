import os
import torch
import numpy as np
from src.models.tcn_model import TCNVelocityEstimator
from src.models.iovnbd_dataset import create_iovnbd_dataloaders
from src.data_prep.iovnbd_parser import parse_synchronized_iovnbd

def evaluate():
    print("Loading test data...")
    train_loader, val_loader, test_loader = create_iovnbd_dataloaders("data/IO-VNBD", window_size=200, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNVelocityEstimator(input_channels=6, output_dim=2, num_channels=[64, 64, 128, 128], kernel_size=7)
    
    best_model_path = os.path.join("checkpoints", "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading checkpoint {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("NO CHECKPOINT FOUND. The model will be untrained.")
        
    model.to(device)
    model.eval()
    
    gt_speeds = []
    pred_speeds = []
    examples = []
    
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
            
            preds = model(X)
            
            gt_n = Y[:, 0].cpu().numpy()
            gt_e = Y[:, 1].cpu().numpy()
            pred_n = preds[:, 0].cpu().numpy()
            pred_e = preds[:, 1].cpu().numpy()
            
            gt_speed = np.sqrt(gt_n**2 + gt_e**2)
            pred_speed = np.sqrt(pred_n**2 + pred_e**2)
            
            gt_speeds.extend(gt_speed)
            pred_speeds.extend(pred_speed)
            
            if len(examples) < 20:
                for i in range(len(gt_speed)):
                    if len(examples) < 20:
                        examples.append((gt_speed[i], pred_speed[i], abs(gt_speed[i] - pred_speed[i])))
                        
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
    
    print("\n--- METRICS ---")
    print(f"MAE (m/s): {mae_ms:.4f}")
    print(f"MAE (km/h): {mae_kmh:.4f}")
    print(f"RMSE (m/s): {rmse:.4f}")
    print(f"Mean GT Speed: {mean_gt:.4f} m/s")
    print(f"Mean Pred Speed: {mean_pred:.4f} m/s")
    print(f"Min/Max GT Speed: {min_gt:.4f} / {max_gt:.4f} m/s")
    print(f"Min/Max Pred Speed: {min_pred:.4f} / {max_pred:.4f} m/s")
    
    print("\n--- BASELINES ---")
    print(f"Zero Velocity MAE: {baseline_zero_mae:.4f} m/s")
    print(f"Mean Velocity MAE: {baseline_mean_mae:.4f} m/s")
    
    print("\n--- EXAMPLES ---")
    for i, (gt, pr, er) in enumerate(examples):
        print(f"[{i+1:02d}] GT: {gt:.4f} m/s | Pred: {pr:.4f} m/s | Err: {er:.4f} m/s")

if __name__ == "__main__":
    evaluate()
