import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.trainer import Trainer
from models.tcn_model import TCNVelocityEstimator

best_model_path = os.path.join("checkpoints", "best_model.pt")
model, checkpoint = Trainer.load_checkpoint(best_model_path, device=torch.device("cpu"))

onnx_path = os.path.join("simulator", "model.onnx")
dummy_input = torch.randn(1, 200, 6)
torch.onnx.export(
    model, (dummy_input,), onnx_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("ONNX export succeeded!")
