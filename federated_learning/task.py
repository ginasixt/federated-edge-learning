"""federated-learning: A Flower / PyTorch app."""

# federated_learning/task.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Optional

# GLOBALER DATA CACHE
_FULL_DATA_CACHE: Optional[tuple] = None  # (X_all, y_all, row_id_to_idx, meta)
_STATS_CACHE = {}


def load_full_dataset_cached(
    parquet_path: str,
    stats_path: str,
    use_cache: bool = True
) -> tuple:
    """
    Lädt KOMPLETTES Dataset einmal und cached es im RAM.
    Wenn schon einmal gecached, dann wird es einfach nur zurückgeben
    
    Returns:
        (X_all, y_all, row_id_to_idx, meta)
        - X_all: (253680, 21) normalized features
        - y_all: (253680,) binary labels
        - row_id_to_idx: {row_id → array index} mapping
        - meta: Normalization stats
    """
    global _FULL_DATA_CACHE
    
    # Cache-Hit: Daten sind bereits im RAM
    if use_cache and _FULL_DATA_CACHE is not None:
        return _FULL_DATA_CACHE
    
    # Cache-Miss: Lade einmalig von Disk
    print(f"   Loading FULL dataset into RAM (will be cached)...")
    print(f"   Source: {parquet_path}")
    
    # 1. Lade Metadaten
    meta = json.loads(Path(stats_path).read_text())
    
    # 2. Lade KOMPLETTES Parquet (OHNE Filter!)
    df = pd.read_parquet(parquet_path)
    print(f"   Loaded: {len(df):,} samples × {len(df.columns)} columns")
    
    # 3. Extrahiere Features & Target
    target_col = meta["target"]
    y_all = df[target_col].astype(int).values
    X_all = df.drop(columns=[target_col, "__row_id__"]).astype(float)
    

    # 4. Normalisierung (mit globalen Stats)
    # Normalisierung der Features (Mittelwert 0, Standardabweichung 1)
        # Features (z.B. Alter, BMI, Blutdruck) unterschieldiche Wertebereiche.
        # --> würde bedeuten, Features mit großen Werten dominieren Training .
        # Mit Normalisierung ( Mittelwert 0, Standardabweichung 1) werden die Features vergleichbar skaliert.
    # Also berechen wir mean und std auf Trainingsdaten (sonst Data Leakage) und normalisiwren Train und Test mit diesen Werten.
    mean = pd.Series(meta["mean"])
    std = pd.Series(meta["std"])
    X_all = (X_all - mean) / std
    X_all = X_all.values.astype("float32")

    # 5. For binary classification, map labels:
    # Screening: 0=gesund (neg), 1=prä+2=diabetes (pos)
    y_all = (y_all >= 1).astype("int64")

    # We could also do diabetes-only classification here
    #   Diagnosis: 0=gesund+1=prä (neg), 2=diabetes (pos)
    # or a multiclass classification (0,1,2), but we will use our AI for screening.

    # 6. Erstelle Row-ID → Array-Index Mapping
    row_id_to_idx = {
        int(row_id): idx 
        for idx, row_id in enumerate(df["__row_id__"])
    }
    
    # Speichere im Cache
    _FULL_DATA_CACHE = (X_all, y_all, row_id_to_idx, meta)
    
    print(f" Dataset cached in RAM:")
    print(f"   Shape: X={X_all.shape}, y={y_all.shape}")
    print(f"   Memory: ~{X_all.nbytes / (1024**2):.1f} MB")
    print(f"   Row-ID mapping: {len(row_id_to_idx):,} entries\n")
    
    return _FULL_DATA_CACHE


def load_client_data(
    parquet_path: str,
    stats_path: str,
    train_row_ids: List[int],
    val_row_ids: List[int],
    boost_factor: float = 2.0,  # Parameter für Class-Weights
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]: 
    """
    Lädt Client-spezifische Daten + berechnet globale Class-Weights.
    
    Args:
        parquet_path: Pfad zum Parquet-File
        stats_path: Pfad zu Normalisierungs-Stats
        train_row_ids: Client-spezifische Train-Row-IDs
        val_row_ids: Client-spezifische Val-Row-IDs
        boost_factor: Multiplikator für positive Klasse (default: 2.0)
        use_cache: Nutze gecachtes Dataset
    
    Returns:
        X_train, y_train, X_val, y_val, class_weights
        - class_weights: torch.Tensor([w_neg, w_pos]) als CPU-Tensor
    """
    # 1. Lade gecachtes Dataset
    X_all, y_all, row_id_to_idx, meta = load_full_dataset_cached(
        parquet_path, 
        stats_path, 
        use_cache
    )
    
    # 2. Client-Daten extrahieren
    try:
        train_indices = [row_id_to_idx[int(rid)] for rid in train_row_ids]
        val_indices = [row_id_to_idx[int(rid)] for rid in val_row_ids]
    except KeyError as e:
        raise ValueError(f"Invalid row_id in client data: {e}")
    
    # 3. Direkte NumPy-Indexierung (RAM-Speed!)
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    
    # 4. Berechne Class-Weights aus globalem Train-Set (cached!)
    train_idx = np.array(meta["train_idx"])
    train_positions = [row_id_to_idx[int(rid)] for rid in train_idx]
    y_train_global = y_all[train_positions]
    
    pos = int((y_train_global >= 1).sum())
    neg = int((y_train_global < 1).sum())
    tot = max(1, pos + neg)
    
    w_pos = (neg / tot) * boost_factor
    w_neg = pos / tot
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
    
    return X_train, y_train, X_val, y_val, class_weights


def clear_data_cache():
    """Leert den globalen Cache (z.B. zwischen Runs)."""
    global _FULL_DATA_CACHE, _STATS_CACHE
    _FULL_DATA_CACHE = None
    _STATS_CACHE.clear()
    print("🧹 Data cache cleared")


def make_loaders_from_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Erstellt DataLoader aus vorbereiteten Arrays.
    """
    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xval = torch.from_numpy(X_val)
    yval = torch.from_numpy(y_val)
    
    tr = TensorDataset(Xtr, ytr)
    val = TensorDataset(Xval, yval)
    
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True),
    )
