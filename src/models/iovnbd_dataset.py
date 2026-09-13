import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import random

# Import our new parser
from data_prep.iovnbd_parser import discover_synchronized_sessions, parse_synchronized_iovnbd

class IOVNBDDataset(Dataset):
    def __init__(self, session_pairs: List[Tuple[str, str]], window_size: int = 200, stride: int = 20):
        """
        PyTorch Dataset for IO-VNBD synchronized sessions.
        
        Args:
            session_pairs: List of (s_csv_path, v_csv_path)
            window_size: Length of the sliding window in samples (default 200)
            stride: Step size between windows (default 20)
        """
        self.window_size = window_size
        self.stride = stride
        
        self.windows_x = []
        self.targets_y = []
        
        for s_csv, v_csv in session_pairs:
            # Parse the synchronized pair
            X, Y = parse_synchronized_iovnbd(s_csv, v_csv)
            n_samples = X.shape[0]
            
            if n_samples < window_size:
                continue
                
            # Create sliding windows
            for start in range(0, n_samples - window_size, stride):
                end = start + window_size
                window_x = X[start:end]
                
                # The target is the velocity at the end of the window (simulating real-time estimation)
                target_y = Y[end - 1]
                
                self.windows_x.append(window_x)
                self.targets_y.append(target_y)
                
        if self.windows_x:
            self.windows_x = np.array(self.windows_x, dtype=np.float32)
            self.targets_y = np.array(self.targets_y, dtype=np.float32)
        else:
            self.windows_x = np.zeros((0, window_size, 6), dtype=np.float32)
            self.targets_y = np.zeros((0, 2), dtype=np.float32)

    def __len__(self):
        return len(self.windows_x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.windows_x[idx]), torch.from_numpy(self.targets_y[idx])


def create_iovnbd_dataloaders(base_dir: str = "data/IO-VNBD", window_size: int = 200, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Discovers all IO-VNBD synchronized sessions, splits them securely (to prevent leakage),
    and creates DataLoaders.
    """
    pairs = discover_synchronized_sessions(base_dir)
    
    if not pairs:
        print("ERROR: No IO-VNBD synchronized sessions found.")
        return None, None, None
        
    # Sort for deterministic behavior, then shuffle with a fixed seed
    pairs = sorted(pairs)
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = max(1, int(0.7 * n))
    val_end = max(train_end + 1, int(0.85 * n))
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"Dataset split (Sessions): Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    train_ds = IOVNBDDataset(train_pairs, window_size=window_size)
    val_ds = IOVNBDDataset(val_pairs, window_size=window_size)
    test_ds = IOVNBDDataset(test_pairs, window_size=window_size)
    
    print(f"Dataset split (Windows): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
