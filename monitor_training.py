#!/usr/bin/env python3
import time
import urllib.request
import json
import sys

def main():
    last_epoch = 0
    print("Monitoring 40-epoch training...", flush=True)
    
    while True:
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/training/status")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                curr_epoch = data.get("current_epoch", 0)
                total_epochs = data.get("total_epochs", 40)
                is_training = data.get("is_training", False)
                status = data.get("status", "")
                
                # Check for new epochs
                history = data.get("epoch_history", [])
                while last_epoch < len(history):
                    ep = history[last_epoch]
                    print(f"[EPOCH {ep['epoch']:02d}/{total_epochs:02d}] "
                          f"Train Loss: {ep['train_loss']:.6f} | "
                          f"Val Loss: {ep['val_loss']:.6f} | "
                          f"LR: {ep['learning_rate']:.6f} | "
                          f"Duration: {ep['duration_s']:.1f}s", flush=True)
                    last_epoch += 1
                    
                if (not is_training and status in ["Training Complete", "Error"]) or status.startswith("Error"):
                    print(f"\nTraining stopped with status: {status}", flush=True)
                    if "metrics" in data and data["metrics"]:
                        print("\nFinal Metrics:\n" + json.dumps(data["metrics"], indent=2), flush=True)
                    break
        except Exception as e:
            print(f"Error querying status: {e}", flush=True)
            
        time.sleep(15)

if __name__ == "__main__":
    main()
