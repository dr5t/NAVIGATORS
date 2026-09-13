from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import os
import threading

from models.dataset import create_dataloaders
from models.iovnbd_dataset import create_iovnbd_dataloaders
from models.trainer import Trainer
from models.tcn_model import TCNVelocityEstimator

router = APIRouter(prefix="/training", tags=["training"])

# Global training state
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "status": "Idle",
    "metrics": {}
}

class TrainingRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    window_size: int = 200

def run_training(epochs: int, batch_size: int, learning_rate: float, window_size: int):
    global training_state
    
    try:
        training_state["is_training"] = True
        training_state["status"] = "Loading IO-VNBD Data..."
        
        # Load the official IO-VNBD dataset instead of local recordings
        iovnbd_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "IO-VNBD")
        train_loader, val_loader, test_loader = create_iovnbd_dataloaders(
            base_dir=iovnbd_dir,
            window_size=window_size,
            batch_size=batch_size
        )
        
        if not train_loader:
            training_state["status"] = "Error: IO-VNBD dataset not found. Please run setup instructions."
            training_state["is_training"] = False
            return
            
        training_state["status"] = "Initializing Model..."
        
        # Instantiate model (e.g., TCN)
        model = TCNVelocityEstimator(
            input_channels=6,
            output_dim=2,
            num_channels=[64, 64, 128, 128],
            kernel_size=7
        )
        
        config = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss": "mse_angular",
            "checkpoint_dir": "checkpoints"
        }
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        training_state["total_epochs"] = epochs
        training_state["status"] = "Training in progress..."
        
        def epoch_callback(epoch, total_epochs, train_loss, val_loss, elapsed):
            training_state["current_epoch"] = epoch
            training_state["train_loss"] = train_loss
            training_state["val_loss"] = val_loss
            training_state["eta"] = f"{(total_epochs - epoch) * elapsed:.1f}s"
            
            # Optionally broadcast via websocket if we imported ws_manager
            # but for now state polling from the client is fine.
        
        res = trainer.train(epoch_callback=epoch_callback)
        
        training_state["status"] = "Evaluating on Test Set..."
        
        # 1. Baseline: Zero Velocity (or mean velocity)
        # We will compute the MAE/RMSE for predicting [0, 0] vs True
        baseline_se = 0.0
        baseline_ae = 0.0
        
        # 2. Model Test Metrics
        import torch
        best_model_path = os.path.join("checkpoints", "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            
        model.eval()
        test_se = 0.0
        test_ae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(trainer.device)
                Y = Y.to(trainer.device)
                
                preds = model(X)
                
                diff = preds - Y
                test_se += torch.sum(diff ** 2).item()
                test_ae += torch.sum(torch.abs(diff)).item()
                
                baseline_diff = 0.0 - Y # Zero velocity baseline
                baseline_se += torch.sum(baseline_diff ** 2).item()
                baseline_ae += torch.sum(torch.abs(baseline_diff)).item()
                
                num_samples += Y.size(0) * Y.size(1)
                
        test_mae = test_ae / max(1, num_samples)
        test_rmse = (test_se / max(1, num_samples)) ** 0.5
        
        base_mae = baseline_ae / max(1, num_samples)
        base_rmse = (baseline_se / max(1, num_samples)) ** 0.5
        
        training_state["status"] = "Training Complete"
        training_state["metrics"] = {
            "best_val_loss": res.get("best_val_loss", 0),
            "total_time_s": res.get("total_time_s", 0),
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse
        }
        
        print(f"Test MAE: {test_mae:.4f} (Baseline: {base_mae:.4f})")
        print(f"Test RMSE: {test_rmse:.4f} (Baseline: {base_rmse:.4f})")
        
        # ONNX Export
        try:
            onnx_path = os.path.join("simulator", "model.onnx")
            # Dummy input for TCN: (batch, length, channels)
            dummy_input = torch.randn(1, window_size, 6)
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except Exception as ex:
            training_state["status"] = f"Error: ONNX export failed: {ex}"
            training_state["is_training"] = False
            return
        
    except Exception as e:
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
    
    return {"status": "Training started"}

@router.get("/status")
def get_training_status():
    global training_state
    return training_state
