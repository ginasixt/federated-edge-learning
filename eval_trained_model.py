#!/usr/bin/env python3
"""
Evaluierungsscript für trainierte Modelle - EVAL ONLY, KEIN TRAINING!

Lädt ein trainiertes Modell und evaluiert es auf Validierungsdaten (ALLE Val-Clients).

Verwendung:
    python eval_trained_model.py \
        --model-path <path_to_model.pt> \
        --split-path <path_to_split_json> \
        --data-path <path_to_parquet> \
        --stats-path <path_to_norm_stats.json> \
        --round <round_number> \
        --output-dir <output_directory> \
        --eval-all-clients True

Beispiel:
    python eval_trained_model.py \
        --model-path result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/model_round_80.pt \
        --split-path splits_iid_scaling/splits_iid_16384_clients.json \
        --data-path data/diabetes_normalized.parquet \
        --stats-path data/norm_stats.json \
        --round 80 \
        --output-dir result/splits_iid_scaling/splits_iid_16384_clients.json/eval_results/ \
        --eval-all-clients True
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG & DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# MLP Model (identisch zu client_app.py)
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [256, 128], out_dim: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        p = torch.exp(-ce)
        focal = (1 - p) ** self.gamma * ce
        return focal.mean()


def load_validation_data(
    split_path: str,
    parquet_path: str,
    stats_path: str,
    eval_all_clients: bool = True,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Lädt Validierungsdaten aus ALLEN Val-Clients (oder nur einem).
    
    Args:
        split_path: Path zur splits JSON
        parquet_path: Path zum Parquet-File mit Daten
        stats_path: Path zu norm_stats.json
        eval_all_clients: Wenn True, lädt ALLE Val-Clients, sonst nur den ersten
        
    Returns:
        X_val, y_val, class_weights
    """
    import pandas as pd
    
    # Lade Split-Metadaten
    with open(split_path) as f:
        split_data = json.load(f)
    
    # Lade Norm Stats (mit Class Weights)
    with open(stats_path) as f:
        stats = json.load(f)
    
    # Lade Normalisierungsstats
    class_weights = torch.tensor(
        [stats["neg_weight"], stats["pos_weight"]], 
        dtype=torch.float32
    )
    
    # Sammle ALLE Val Row IDs
    all_val_row_ids = []
    
    if "train" in split_data and "val" in split_data:
        # Format 1: IID Split Format
        print("✅ IID Split Format erkannt (train/val)")
        val_data_dict = split_data["val"]
        
        # Nutze val_client_range aus Meta falls vorhanden
        if "meta" in split_data and "val_client_range" in split_data["meta"]:
            val_range = split_data["meta"]["val_client_range"]
            min_cid = val_range.get("min", 0)
            max_cid = val_range.get("max", len(val_data_dict) - 1)
            print(f"   Val-Client Range aus Meta: {min_cid} - {max_cid}")
            
            if eval_all_clients:
                # Lade ALLE Val-Clients in diesem Range
                for cid in range(min_cid, max_cid + 1):
                    cid_str = str(cid)
                    if cid_str in val_data_dict and len(val_data_dict[cid_str]) > 0:
                        all_val_row_ids.extend(val_data_dict[cid_str])
            else:
                # Lade nur ersten Client
                all_val_row_ids.extend(val_data_dict[str(min_cid)])
        else:
            # Fallback: Alle Clients mit Val-Daten
            if eval_all_clients:
                for cid_str in val_data_dict.keys():
                    if len(val_data_dict[cid_str]) > 0:
                        all_val_row_ids.extend(val_data_dict[cid_str])
            else:
                all_val_row_ids.extend(list(val_data_dict.values())[0])
    
    elif "data" in split_data:
        # Format 2: Dirichlet Split Format
        print("✅ Dirichlet Split Format erkannt (data)")
        
        if eval_all_clients:
            for cid_str, client_data in split_data["data"].items():
                if client_data.get("val") and len(client_data["val"]) > 0:
                    all_val_row_ids.extend(client_data["val"])
        else:
            for cid_str, client_data in split_data["data"].items():
                if client_data.get("val") and len(client_data["val"]) > 0:
                    all_val_row_ids.extend(client_data["val"])
                    break
    else:
        raise ValueError("Unbekanntes Split-Format! Erwartet 'train'/'val' oder 'data'.")
    
    if not all_val_row_ids:
        raise ValueError("Keine Validierungsdaten im Split gefunden!")
    
    print(f"   ✅ Lade {len(all_val_row_ids)} Val-Samples")
    
    # Lade Parquet mit Filter (sehr schnell!)
    df = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", all_val_row_ids)]
    )
    
    # Extrahiere Features & Labels
    target_col = stats["target"]
    y_val = df[target_col].astype(int).values
    X_val = df.drop(columns=[target_col, "__row_id__"]).values.astype("float32")
    
    # Binary labels
    y_val = (y_val >= 1).astype("int64")
    
    print(f"   Features: {X_val.shape[1]}, Samples: {X_val.shape[0]}")
    print(f"   Class distribution: {(y_val == 0).sum()} neg, {(y_val == 1).sum()} pos")
    
    return X_val, y_val, class_weights


def evaluate_multi_threshold(
    model: nn.Module, 
    loader: DataLoader, 
    crit: nn.Module,
    threshold_grid: List[float] = None
) -> Tuple[float, int, dict]:
    """
    Evaluiert Modell auf Validierungsdaten mit Multi-Threshold Metriken.
    
    Returns:
        avg_loss, n_samples, metrics_dict
    """
    if threshold_grid is None:
        threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    probs_all = []
    y_all = []
    
    print("🔄 Evaluating...")
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs.cpu())
            y_all.append(yb.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}, Samples processed: {n_samples}")
    
    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    
    # ============================================================
    # 1) MULTI-THRESHOLD METRIKEN
    # ============================================================
    all_threshold_results = []
    
    for thr in threshold_grid:
        preds = (probs >= thr).astype(int)
        
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        
        # Metriken
        n_pos = tp + fn
        n_neg = tn + fp
        
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        spec = 1.0 - fpr
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0.0
        balanced_acc = (tpr + spec) / 2.0
        youden = tpr + spec - 1.0
        
        prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
        n_alerts = tp + fp
        alerts_per_1000 = (n_alerts / n_samples * 1000) if n_samples > 0 else 0.0
        
        result = {
            "threshold": float(thr),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": float(tpr),
            "recall": float(tpr),
            "fpr": float(fpr),
            "spec": float(spec),
            "ppv": float(ppv),
            "precision": float(ppv),
            "npv": float(npv),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_acc),
            "youden": float(youden),
            "prevalence": float(prevalence),
            "alerts_per_1000": float(alerts_per_1000),
        }
        all_threshold_results.append(result)
    
    # ============================================================
    # 2) CALIBRATION PLOT METRIKEN (10 Bins)
    # ============================================================
    num_calib_bins = 10
    calib_edges = np.linspace(0.0, 1.0, num_calib_bins + 1)
    calib_bin_n = []
    calib_bin_sum_pred = []
    calib_bin_sum_true = []
    
    for i in range(num_calib_bins):
        lower = calib_edges[i]
        upper = calib_edges[i + 1]
        
        if i == num_calib_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        
        n_in_bin = int(mask.sum())
        sum_pred = float(probs[mask].sum()) if n_in_bin > 0 else 0.0
        sum_true = int(y[mask].sum()) if n_in_bin > 0 else 0
        
        calib_bin_n.append(n_in_bin)
        calib_bin_sum_pred.append(sum_pred)
        calib_bin_sum_true.append(sum_true)
    
    # ============================================================
    # 3) RISK DISTRIBUTION PLOT (20 Bins)
    # ============================================================
    num_risk_bins = 20
    risk_edges = np.linspace(0.0, 1.0, num_risk_bins + 1)
    
    hist_pred_y0 = []
    for i in range(num_risk_bins):
        lower = risk_edges[i]
        upper = risk_edges[i + 1]
        
        if i == num_risk_bins - 1:
            mask = (probs >= lower) & (probs <= upper) & (y == 0)
        else:
            mask = (probs >= lower) & (probs < upper) & (y == 0)
        
        count = int(mask.sum())
        hist_pred_y0.append(count)
    
    hist_pred_y1 = []
    for i in range(num_risk_bins):
        lower = risk_edges[i]
        upper = risk_edges[i + 1]
        
        if i == num_risk_bins - 1:
            mask = (probs >= lower) & (probs <= upper) & (y == 1)
        else:
            mask = (probs >= lower) & (probs < upper) & (y == 1)
        
        count = int(mask.sum())
        hist_pred_y1.append(count)
    
    # ============================================================
    # Zusammenfassen
    # ============================================================
    metrics = {
        "auc": 0.0,
        "n_samples": n_samples,
        "all_thresholds": all_threshold_results,
        # Calibration
        "calib_edges_json": json.dumps(calib_edges.tolist()),
        "calib_bin_n_json": json.dumps(calib_bin_n),
        "calib_bin_sum_pred_json": json.dumps(calib_bin_sum_pred),
        "calib_bin_sum_true_json": json.dumps(calib_bin_sum_true),
        # Risk Distribution
        "risk_edges_json": json.dumps(risk_edges.tolist()),
        "hist_pred_y0_json": json.dumps(hist_pred_y0),
        "hist_pred_y1_json": json.dumps(hist_pred_y1),
    }
    
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, n_samples, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiere trainiertes Modell (Eval-Only, kein Training)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path zur trainierten Modeldatei (.pt)"
    )
    parser.add_argument(
        "--split-path",
        type=str,
        required=True,
        help="Path zur Split-JSON"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path zur Parquet-Datadatei"
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="data/norm_stats.json",
        help="Path zu norm_stats.json (Default: data/norm_stats.json)"
    )
    parser.add_argument(
        "--round",
        type=int,
        default=80,
        help="Round-Nummer (für Metadaten, Default: 80)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result/eval_results",
        help="Ausgabe-Verzeichnis für Evaluierungsergebnisse"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size für Evaluation (Default: 128)"
    )
    parser.add_argument(
        "--eval-all-clients",
        type=bool,
        default=True,
        help="Evaluiere ALLE Val-Clients (True) oder nur den ersten (False) (Default: True)"
    )
    parser.add_argument(
        "--in-dim",
        type=int,
        default=21,
        help="Input Feature-Dimension (Default: 21)"
    )
    
    args = parser.parse_args()
    
    # ============================================================
    # 1) SETUP
    # ============================================================
    print("=" * 70)
    print("🔬 EVALUATION SCRIPT - Trainiertes Modell evaluieren")
    print("=" * 70)
    print(f"Model:        {args.model_path}")
    print(f"Split:        {args.split_path}")
    print(f"Data:         {args.data_path}")
    print(f"Output:       {args.output_dir}")
    print(f"Round:        {args.round}")
    print()
    
    # 2) Validierungsdaten laden
    print("📊 Loading validation data...")
    X_val, y_val, class_weights = load_validation_data(
        args.split_path,
        args.data_path,
        args.stats_path,
        eval_all_clients=args.eval_all_clients
    )
    print()
    
    # 3) DataLoader erstellen
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    
    # 4) Modell laden
    print("🤖 Loading model...")
    model = MLP(in_dim=args.in_dim)
    model.to(DEVICE)
    
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print(f"   ✅ Model loaded from {args.model_path}")
    print()
    
    # 5) Loss-Funktion erstellen (mit Class Weights)
    # ✅ WICHTIG: Muss mit Client/Server übereinstimmen!
    # Nutze CrossEntropyLoss (Standard), nicht FocalLoss
    crit = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    
    # 6) Evaluieren
    print("🔍 Evaluating...")
    avg_loss, n_samples, metrics = evaluate_multi_threshold(
        model, val_loader, crit
    )
    print(f"   ✅ Evaluation complete!")
    print(f"   Avg Loss: {avg_loss:.6f}")
    print(f"   Total Samples: {n_samples}")
    print()
    
    # 7) Ergebnisse speichern
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_dict = {
        "round": args.round,
        "metrics": metrics,
        "model_checkpoint": str(args.model_path),
    }
    
    output_file = output_dir / f"round_{args.round}_eval.json"
    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    print()
    
    # 8) Summary
    print("=" * 70)
    print("📈 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Round:          {args.round}")
    print(f"Val Samples:    {n_samples}")
    print(f"Avg Loss:       {avg_loss:.6f}")
    print(f"Best Threshold: {metrics['all_thresholds'][-3]['threshold']:.2f}")
    print(f"  -> Recall:    {metrics['all_thresholds'][-3]['recall']:.4f}")
    print(f"  -> Spec:      {metrics['all_thresholds'][-3]['spec']:.4f}")
    print(f"  -> F1:        {metrics['all_thresholds'][-3]['f1']:.4f}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
