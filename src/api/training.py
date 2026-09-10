from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import os
import threading

from models.dataset import create_dataloaders
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
        training_state["status"] = "Loading Data..."
        
        recordings_dir = Path(os.path.join(os.path.dirname(__file__), "..", "..", "data", "phone_recordings"))
        train_loader, val_loader, test_loader = create_dataloaders(
            recordings_dir=recordings_dir,
            window_size=window_size,
            batch_size=batch_size
        )
        
        if not train_loader:
            training_state["status"] = "Error: Not enough data for training"
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
        
        training_state["status"] = "Training Complete"
        training_state["metrics"] = {
            "best_val_loss": res.get("best_val_loss", 0),
            "total_time_s": res.get("total_time_s", 0)
        }
        
        # ONNX Export
        try:
            import torch
            best_model_path = os.path.join("checkpoints", "best_model.pth")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                
            onnx_path = os.path.join("simulator", "model.onnx")
            # Dummy input for TCN: (batch, channels, length)
            dummy_input = torch.randn(1, 6, window_size)
            torch.onnx.export(
                model, dummy_input, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except Exception as ex:
            print(f"ONNX export failed: {ex}")
        
    except Exception as e:
        training_state["status"] = f"Error: {e}"
    finally:
        training_state["is_training"] = False

@router.post("/start")
def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    global training_state
    if training_state["is_training"]:
        return {"error": "Training already in progress"}
        
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
