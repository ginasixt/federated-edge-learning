"""federated-learning: A Flower / PyTorch app."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

def load_client_data(
    parquet_path: str,
    stats_path: str,
    train_row_ids: List[int],
    val_row_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Lädt NUR die für diesen Client relevanten Daten.
    
    ✅ Parquet ist bereits normalisiert!
    ✅ Class-Weights sind in norm_stats.json gespeichert!
    """
    # 1. Lade Metadaten (enthält Class-Weights!)
    meta = json.loads(Path(stats_path).read_text())
    
    # 2. Lade NUR Client-Rows (PyArrow Filter - sehr schnell!)
    all_row_ids = set(train_row_ids + val_row_ids)
    
    df = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", list(all_row_ids))]
    )
    
    # 3. Extrahiere Features & Labels (KEINE Normalisierung mehr!)
    target_col = meta["target"]
    y_all = df[target_col].astype(int).values
    X_all = df.drop(columns=[target_col, "__row_id__"]).values.astype("float32")
    
    # Binary labels (0=healthy, 1=prediabetic+diabetic)
    y_all = (y_all >= 1).astype("int64")
    
    # 4. Row-ID Mapping
    row_id_to_idx = {
        int(row_id): idx 
        for idx, row_id in enumerate(df["__row_id__"])
    }
    
    # 5. Train/Val Split
    train_indices = [row_id_to_idx[int(rid)] for rid in train_row_ids]
    val_indices = [row_id_to_idx[int(rid)] for rid in val_row_ids]
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    
    # 6. ✅ Class-Weights aus norm_stats.json (vorberechnet, weight bosst für pos schon drin 1,5)
    # // 0.86066732366 pos_weight ohne boost
    # 2.0 weigt boost: 1.7213346473321132
    class_weights = torch.tensor(
        [meta["neg_weight"], meta["pos_weight"]], 
        dtype=torch.float32
    )
    
    return X_train, y_train, X_val, y_val, class_weights


def make_loaders_from_arrays(X_train, y_train, X_val, y_val, batch_size=128):
    """Erstellt PyTorch DataLoader aus NumPy Arrays."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), 
        torch.from_numpy(y_val)
    )
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False)
    )
