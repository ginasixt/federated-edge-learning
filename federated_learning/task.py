"""federated-learning: A Flower / PyTorch app."""

# federated_learning/task.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Lädt die vorbereiteten Daten und Normalisierungs-Statistiken
def load_prepared(parquet_path: str, stats_path: str):
    """
    Load prepared parquet + stats and map labels for binary classification.
        label_mode:
        - "prepos": 0→neg, 1+2→pos (Screening: pre-diabetic + diabetic = risk)
        - "diabetes_only": 0+1→neg, 2→pos (Diagnosis: only diabetic = pos)
        - "multiclass": keep original {0,1,2} (requires out_dim=3)
    """
    
    df = pd.read_parquet(parquet_path)
    meta = json.loads(Path(stats_path).read_text())

    # Features und Target extrahieren und Normalierungparameter laden
    tgt = meta["target"]
    tr_idx = np.array(meta["train_idx"])
    val_idx = np.array(meta["val_idx"])
    mean = meta["mean"]
    std = meta["std"]


    y_all = df[tgt].astype(int).values 
    X_all = df.drop(columns=[tgt, "__row_id__"]).astype(float)

    # Normalisierung der Features (Mittelwert 0, Standardabweichung 1)
        # Features (z.B. Alter, BMI, Blutdruck) unterschieldiche Wertebereiche.
        # --> würde bedeuten, Features mit großen Werten dominieren Training .
        # Mit Normalisierung ( Mittelwert 0, Standardabweichung 1) werden die Features vergleichbar skaliert.
    # Also berechen wir mean und std auf Trainingsdaten (sonst Data Leakage) und normalisiwren Train und Test mit diesen Werten.
    X_all = (X_all - pd.Series(mean)) / pd.Series(std)
    X_all = X_all.values.astype("float32")

    # For binary classification, map labels:
    #   Screening: 0=gesund (neg), 1=prä+2=diabetes (pos)
    y_all = (y_all >= 1).astype("int64")

    # We could also do diabetes-only classification here
    #   Diagnosis: 0=gesund+1=prä (neg), 2=diabetes (pos)
    # or a multiclass classification (0,1,2), but we will use our AI for screening.

    return X_all, y_all, tr_idx, val_idx

def make_loaders_for_indices(X, y, train_indices, val_indices, batch_size=128):
    """
    Erstellt DataLoader für Train, Validation und Test.
    Args:
        X: Feature-Matrix (numpy array)
        y: Zielvariable (numpy array)
        train_indices: Indizes für Trainingsdaten
        val_indices: Indizes für Validierungsdaten
        batch_size: Batch-Größe für Trainings-DataLoader
    
    Returns:
        Tuple mit zwei DataLoadern: (train_loader, val_loader)
    """
    Xtr, ytr = torch.tensor(X[train_indices]), torch.tensor(y[train_indices])
    Xval, yval = torch.tensor(X[val_indices]), torch.tensor(y[val_indices])

    tr = TensorDataset(Xtr, ytr)
    val = TensorDataset(Xval, yval)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=1024, shuffle=False),
    )

    # Xte, yte = torch.tensor(X[test_indices]), torch.tensor(y[test_indices])
    # te = TensorDataset(Xte, yte)

    # return (
    #     DataLoader(te, batch_size=1024, shuffle=False),
    # )
