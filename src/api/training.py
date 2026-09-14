from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import os
import threading

import math
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.models.iovnbd_dataset import create_iovnbd_dataloaders, create_iovnbd_dataloaders as create_dataloaders
    from src.models.trainer import Trainer
    from src.models.tcn_model import TCNVelocityEstimator
except ImportError:
    from models.iovnbd_dataset import create_iovnbd_dataloaders, create_iovnbd_dataloaders as create_dataloaders
    from models.trainer import Trainer
    from models.tcn_model import TCNVelocityEstimator

import shutil
import numpy as np

router = APIRouter(prefix="/training", tags=["training"])

# Global training state
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 40,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "current_lr": 0.001,
    "eta": "—",
    "train_samples": 0,
    "val_samples": 0,
    "test_samples": 0,
    "status": "Idle",
    "epoch_history": [],
    "metrics": {}
}

class TrainingRequest(BaseModel):
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 0.001
    window_size: int = 200

def run_training(epochs: int, batch_size: int, learning_rate: float, window_size: int):
    global training_state
    
    candidate_dir = "checkpoints_candidate"
    if os.path.exists(candidate_dir):
        for f in os.listdir(candidate_dir):
            p = os.path.join(candidate_dir, f)
            if os.path.isfile(p):
                os.remove(p)
    os.makedirs(candidate_dir, exist_ok=True)
    
    try:
        training_state["is_training"] = True
        training_state["total_epochs"] = epochs
        training_state["current_epoch"] = 0
        training_state["epoch_history"] = []
        training_state["metrics"] = {}
        training_state["status"] = "Loading IO-VNBD Data..."
        
        # Load the official IO-VNBD dataset (144 synchronized sessions)
        iovnbd_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "IO-VNBD")
        train_loader, val_loader, test_loader = create_iovnbd_dataloaders(
            base_dir=iovnbd_dir,
            window_size=window_size,
            batch_size=batch_size,
            stats_dir=candidate_dir
        )
        
        if not train_loader:
            training_state["status"] = "Error: IO-VNBD dataset not found."
            training_state["is_training"] = False
            return
            
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)
        test_count = len(test_loader.dataset)
        
        training_state["train_samples"] = train_count
        training_state["val_samples"] = val_count
        training_state["test_samples"] = test_count
        training_state["status"] = f"Initializing TCN... (Train: {train_count}, Val: {val_count}, Test: {test_count})"
        
        # Instantiate model (TCN with 559,234 parameters)
        model = TCNVelocityEstimator(
            input_channels=6,
            output_dim=2,
            num_channels=[64, 64, 128, 128],
            kernel_size=7,
            dropout=0.2
        )
        
        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss": "mse_angular",
            "checkpoint_dir": candidate_dir,
            "early_stopping_patience": epochs + 10,  # Run all configured epochs
            "optimizer": "adamw",
            "scheduler": "cosine",
            "use_augmentation": True
        }
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        training_state["status"] = f"Training in progress (Epoch 1/{epochs})..."
        
        def epoch_callback(epoch, total_epochs, train_loss, val_loss, elapsed, current_lr=None):
            lr_val = float(current_lr) if current_lr is not None else learning_rate
            training_state["current_epoch"] = epoch
            training_state["train_loss"] = float(train_loss)
            training_state["val_loss"] = float(val_loss)
            training_state["current_lr"] = lr_val
            remaining_epochs = max(0, total_epochs - epoch)
            training_state["eta"] = f"{remaining_epochs * elapsed:.1f}s"
            training_state["status"] = f"Training epoch {epoch}/{total_epochs}"
            training_state["epoch_history"].append({
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "duration_s": round(float(elapsed), 2),
                "learning_rate": lr_val
            })
            print(f"[Training API] Epoch {epoch:2d}/{total_epochs:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_val:.2e} | Duration: {elapsed:.1f}s", flush=True)
        
        res = trainer.train(epoch_callback=epoch_callback)
        
        training_state["status"] = "Evaluating candidate on held-out test set..."
        
        # Load best candidate checkpoint for test set evaluation
        import torch
        best_candidate_path = os.path.join(candidate_dir, "best_model.pt")
        if os.path.exists(best_candidate_path):
            checkpoint = torch.load(best_candidate_path, map_location=trainer.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            
        model.eval()
        gt_speeds = []
        pred_speeds = []
        
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(trainer.device)
                preds = model(X)
                
                gt_n = Y[:, 0].cpu().numpy()
                gt_e = Y[:, 1].cpu().numpy()
                pr_n = preds[:, 0].cpu().numpy()
                pr_e = preds[:, 1].cpu().numpy()
                
                gt_s = np.sqrt(gt_n**2 + gt_e**2)
                pr_s = np.sqrt(pr_n**2 + pr_e**2)
                
                gt_speeds.extend(gt_s)
                pred_speeds.extend(pr_s)
                
        gt_speeds = np.array(gt_speeds, dtype=np.float64)
        pred_speeds = np.array(pred_speeds, dtype=np.float64)
        
        candidate_test_mae = float(np.mean(np.abs(pred_speeds - gt_speeds)))
        candidate_test_rmse = float(np.sqrt(np.mean((pred_speeds - gt_speeds)**2)))
        zero_velocity_mae = float(np.mean(np.abs(gt_speeds - 0.0)))
        mean_velocity_mae = float(np.mean(np.abs(gt_speeds - np.mean(gt_speeds))))
        error_reduction_pct = float(((mean_velocity_mae - candidate_test_mae) / mean_velocity_mae) * 100.0)
        
        print(f"\n--- CANDIDATE TEST EVALUATION ---")
        print(f"Candidate Test MAE:  {candidate_test_mae:.4f} m/s")
        print(f"Candidate Test RMSE: {candidate_test_rmse:.4f} m/s")
        print(f"Zero Velocity Baseline MAE: {zero_velocity_mae:.4f} m/s")
        print(f"Mean Velocity Baseline MAE: {mean_velocity_mae:.4f} m/s")
        print(f"Error Reduction vs Mean:    {error_reduction_pct:.2f}%")
        
        # ONNX Export for Candidate
        candidate_onnx_path = os.path.join(candidate_dir, "model.onnx")
        model_cpu = model.to("cpu")
        dummy_input = torch.randn(1, window_size, 6, device="cpu")
        torch.onnx.export(
            model_cpu, dummy_input, candidate_onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=18, dynamo=False,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Verify ONNX Parity
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(candidate_onnx_path)
        dummy_np = np.random.randn(1, window_size, 6).astype(np.float32)
        with torch.no_grad():
            pt_out = model_cpu(torch.from_numpy(dummy_np)).numpy()
        ort_out = ort_sess.run(None, {"input": dummy_np})[0]
        onnx_parity_max_diff = float(np.max(np.abs(pt_out - ort_out)))
        print(f"ONNX / PyTorch Numerical Parity Max Diff: {onnx_parity_max_diff:.8f}")
        
        # Model Promotion Rule
        # Frozen Production Benchmark: 4.2039 m/s
        production_mae = 4.2039
        production_rmse = 6.0735
        is_strictly_better = candidate_test_mae < (production_mae - 1e-4)
        parity_passed = onnx_parity_max_diff < 1e-4
        
        if is_strictly_better and parity_passed:
            shutil.copy2(best_candidate_path, os.path.join("checkpoints", "best_model.pt"))
            shutil.copy2(candidate_onnx_path, os.path.join("simulator", "model.onnx"))
            shutil.copy2(os.path.join(candidate_dir, "norm_stats.json"), os.path.join("checkpoints", "norm_stats.json"))
            candidate_promoted = True
            promotion_status = f"PROMOTED: New model ({candidate_test_mae:.4f} m/s) outperformed production ({production_mae:.4f} m/s)"
        else:
            candidate_promoted = False
            if not is_strictly_better:
                promotion_status = f"RETAINED PRODUCTION: Candidate ({candidate_test_mae:.4f} m/s) did not outperform frozen production model ({production_mae:.4f} m/s)"
            else:
                promotion_status = f"RETAINED PRODUCTION: Candidate failed parity check ({onnx_parity_max_diff})"
                
        print(f"[Promotion Status] {promotion_status}")
        
        training_state["status"] = "Training Complete"
        training_state["metrics"] = {
            "total_epochs": epochs,
            "best_val_loss": float(res.get("best_val_loss", 0)),
            "total_time_s": float(res.get("total_time_s", 0)),
            "candidate_test_mae": candidate_test_mae,
            "candidate_test_rmse": candidate_test_rmse,
            "zero_velocity_mae": zero_velocity_mae,
            "mean_velocity_mae": mean_velocity_mae,
            "error_reduction_pct": error_reduction_pct,
            "production_mae": production_mae,
            "production_rmse": production_rmse,
            "onnx_parity_max_diff": onnx_parity_max_diff,
            "candidate_promoted": candidate_promoted,
            "promotion_status": promotion_status
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        training_state["status"] = f"Error: {e}"
    finally:
        training_state["is_training"] = False

@router.post("/start")
def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    from fastapi import HTTPException
    global training_state
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
        
    # Synchronously check for IO-VNBD data existence
    iovnbd_dir = Path(os.path.join(os.path.dirname(__file__), "..", "..", "data", "IO-VNBD"))
    sync_dir = iovnbd_dir / "Synchronised V abd S datasets"
    if not sync_dir.exists():
        raise HTTPException(status_code=400, detail="IO-VNBD dataset not found. Please download it first.")
        
    # Run in background thread to avoid blocking the event loop
    thread = threading.Thread(
        target=run_training,
        args=(req.epochs, req.batch_size, req.learning_rate, req.window_size)
    )
    thread.start()
    
    return {"status": "Training started", "epochs": req.epochs, "batch_size": req.batch_size}

@router.get("/status")
def get_training_status():
    global training_state
    return training_state
