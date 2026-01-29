"""federated-learning: A Flower / PyTorch app."""

# federated_learning/task.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Optional

# ✅ NEU: Ray für Shared Memory
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False
    print("⚠️  Ray not available - falling back to process-local cache")

# GLOBALER DATA CACHE (nur für Fallback ohne Ray)
_FULL_DATA_CACHE: Optional[tuple] = None
_STATS_CACHE = {}


# ✅ NEU: Dataset-Holder Actor (Singleton für Shared Memory)
@ray.remote
class DatasetHolder:
    """
    Ray Actor der das Dataset EINMAL lädt und für alle Clients bereitstellt.
    Wird als Singleton instanziiert (nur 1× existiert im Cluster).
    """
    def __init__(self, parquet_path: str, stats_path: str):
        print(f"🔄 DatasetHolder: Loading dataset ONCE into memory...")
        print(f"   Source: {parquet_path}")
        
        # Lade Metadaten
        meta = json.loads(Path(stats_path).read_text())
        
        # Lade KOMPLETTES Parquet
        df = pd.read_parquet(parquet_path)
        print(f"   Loaded: {len(df):,} samples × {len(df.columns)} columns")
        
        # Extrahiere Features & Target
        target_col = meta["target"]
        y_all = df[target_col].astype(int).values
        X_all = df.drop(columns=[target_col, "__row_id__"]).astype(float)
        
        # Normalisierung
        mean = pd.Series(meta["mean"])
        std = pd.Series(meta["std"])
        X_all = (X_all - mean) / std
        X_all = X_all.values.astype("float32")
        
        # Binary labels
        y_all = (y_all >= 1).astype("int64")
        
        # Row-ID Mapping
        row_id_to_idx = {
            int(row_id): idx 
            for idx, row_id in enumerate(df["__row_id__"])
        }
        
        # Speichere als Class-Member (bleibt im Actor-RAM!)
        self.data = (X_all, y_all, row_id_to_idx, meta)
        
        print(f"✅ DatasetHolder: Dataset cached in Actor memory")
        print(f"   Shape: X={X_all.shape}, y={y_all.shape}")
        print(f"   Memory: ~{X_all.nbytes / (1024**2):.1f} MB")
        print(f"   Row-ID mapping: {len(row_id_to_idx):,} entries\n")
    
    def get_data(self):
        """Gibt Dataset-Referenz zurück (zero-copy via Ray Object Store)."""
        return self.data


def load_full_dataset_cached(
    parquet_path: str,
    stats_path: str,
    use_cache: bool = True
) -> tuple:
    """
    Lädt Dataset aus Ray DatasetHolder (Singleton Actor).
    
    Returns:
        (X_all, y_all, row_id_to_idx, meta)
    """
    global _FULL_DATA_CACHE
    
    # ✅ RAY SHARED MEMORY PATH
    if RAY_AVAILABLE and use_cache and ray is not None:
        # Hole oder erstelle Singleton-Actor
        try:
            holder = ray.get_actor("dataset_holder")
        except ValueError:
            # Actor existiert noch nicht → Erstelle ihn
            holder = DatasetHolder.options(
                name="dataset_holder",
                lifetime="detached",  # Überlebt Worker-Neustarts
                max_concurrency=1000   # Viele Clients können gleichzeitig zugreifen
            ).remote(parquet_path, stats_path)
        
        # Hole Daten aus Actor (zero-copy via Object Store!)
        data_ref = holder.get_data.remote()
        return ray.get(data_ref)
    
    # ❌ FALLBACK: Process-local Cache (für Tests ohne Ray)
    if use_cache and _FULL_DATA_CACHE is not None:
        return _FULL_DATA_CACHE
    
    print(f"   Loading FULL dataset into RAM (process-local, NOT shared)...")
    print(f"   Source: {parquet_path}")
    
    # Lade Daten (wie vorher)
    meta = json.loads(Path(stats_path).read_text())
    df = pd.read_parquet(parquet_path)
    print(f"   Loaded: {len(df):,} samples × {len(df.columns)} columns")
    
    target_col = meta["target"]
    y_all = df[target_col].astype(int).values
    X_all = df.drop(columns=[target_col, "__row_id__"]).astype(float)
    
    mean = pd.Series(meta["mean"])
    std = pd.Series(meta["std"])
    X_all = (X_all - mean) / std
    X_all = X_all.values.astype("float32")
    
    y_all = (y_all >= 1).astype("int64")
    
    row_id_to_idx = {
        int(row_id): idx 
        for idx, row_id in enumerate(df["__row_id__"])
    }
    
    _FULL_DATA_CACHE = (X_all, y_all, row_id_to_idx, meta)
    
    print(f" Dataset cached in RAM (process-local):")
    print(f"   Shape: X={X_all.shape}, y={y_all.shape}")
    print(f"   Memory: ~{X_all.nbytes / (1024**2):.1f} MB")
    print(f"   Row-ID mapping: {len(row_id_to_idx):,} entries\n")
    
    return _FULL_DATA_CACHE


def load_client_data(
    parquet_path: str,
    stats_path: str,
    train_row_ids: List[int],
    val_row_ids: List[int],
    boost_factor: float = 2.0,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Lädt Client-spezifische Daten aus dem gecachten Dataset."""
    # Lade gecachtes Dataset (automatisch Ray Shared Memory wenn verfügbar)
    X_all, y_all, row_id_to_idx, meta = load_full_dataset_cached(
        parquet_path, 
        stats_path, 
        use_cache
    )
    
    # Client-Daten extrahieren
    try:
        train_indices = [row_id_to_idx[int(rid)] for rid in train_row_ids]
        val_indices = [row_id_to_idx[int(rid)] for rid in val_row_ids]
    except KeyError as e:
        raise ValueError(f"Invalid row_id in client data: {e}")
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[val_indices]
    y_val = y_all[val_indices]
    
    # Berechne Class-Weights aus globalem Train-Set
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


def clear_data_cache():
    """Leert den globalen Cache (z.B. zwischen Runs)."""
    global _FULL_DATA_CACHE, _STATS_CACHE
    _FULL_DATA_CACHE = None
    _STATS_CACHE.clear()
    
    # Entferne Ray Actor (wenn vorhanden)
    if RAY_AVAILABLE and ray is not None:
        try:
            holder = ray.get_actor("dataset_holder")
            ray.kill(holder)
            print("🧹 Ray DatasetHolder Actor killed")
        except ValueError:
            pass  # Actor existiert nicht
    
    print("🧹 Data cache cleared")
