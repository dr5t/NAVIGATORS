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
        
        recordings_dir = Path("recordings")
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
        
        # We need to hook into the trainer's epoch loop to update state.
        # But for simplicity, we'll wrap the train loop or just run it. 
        # The trainer class currently runs the whole loop. Let's patch it or just run it.
        # Since we can't easily hook into Trainer without modifying it, let's just let it run
        # and we won't get live epoch updates unless we monkey-patch.
        
        # Monkey patch print to catch epoch updates? No, let's just run it.
        # Ideally we would modify trainer.py to accept a callback.
        
        res = trainer.train()
        
        training_state["status"] = "Training Complete"
        training_state["metrics"] = {
            "best_val_loss": res["best_val_loss"],
            "total_time_s": res["total_time_s"]
        }
        
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
