#!/usr/bin/env python3
"""
Simple Batch Eval - Nutzt die GLEICHE evaluate_multi_threshold Funktion aus client_app.py!

Kein neuer Code, kein FocalLoss Gebastele - einfach das, was während Training lief
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# IMPORTS von client_app (deine bestehenden Funktionen!)
# ============================================================
from federated_learning.client_app import MLP, evaluate_multi_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")


def load_validation_data(split_path: str, parquet_path: str, stats_path: str):
    """Lade ALLE Val-Clients (wie dein Training)"""
    
    with open(split_path) as f:
        split_data = json.load(f)
    
    with open(stats_path) as f:
        stats = json.load(f)
    
    class_weights = torch.tensor(
        [stats["neg_weight"], stats["pos_weight"]], 
        dtype=torch.float32
    )
    
    # Sammle ALLE Val-Samples
    all_val_row_ids = []
    
    if "train" in split_data and "val" in split_data:
        val_data_dict = split_data["val"]
        
        # Nutze val_client_range aus Meta
        if "meta" in split_data and "val_client_range" in split_data["meta"]:
            val_range = split_data["meta"]["val_client_range"]
            min_cid = val_range.get("min", 0)
            max_cid = val_range.get("max", len(val_data_dict) - 1)
            
            for cid in range(min_cid, max_cid + 1):
                cid_str = str(cid)
                if cid_str in val_data_dict and len(val_data_dict[cid_str]) > 0:
                    all_val_row_ids.extend(val_data_dict[cid_str])
    
    print(f"📊 Loading {len(all_val_row_ids)} val samples from {max_cid - min_cid + 1} clients...")
    
    # Lade Parquet
    df = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", all_val_row_ids)]
    )
    
    target_col = stats["target"]
    y_val = df[target_col].astype(int).values
    X_val = df.drop(columns=[target_col, "__row_id__"]).values.astype("float32")
    
    y_val = (y_val >= 1).astype("int64")
    
    print(f"   ✅ {X_val.shape[0]} samples, {X_val.shape[1]} features\n")
    
    return X_val, y_val, class_weights


def main():
    # ============================================================
    # CONFIG
    # ============================================================
    SPLIT = "splits_iid_scaling/splits_iid_16384_clients.json"
    DATA = "data/diabetes_normalized.parquet"
    STATS = "data/norm_stats.json"
    MODEL_DIR = "result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/"
    OUTPUT_DIR = "result/splits_iid_scaling/splits_iid_16384_clients.json/eval_all_rounds/"
    
    # Runden zu evaluieren
    rounds_1_70 = [1, 10, 20, 30, 40, 50, 60, 70]
    rounds_71_80 = list(range(71, 81))
    all_rounds = rounds_1_70 + rounds_71_80
    
    # ============================================================
    # SETUP
    # ============================================================
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🔬 BATCH EVALUATION")
    print("=" * 70)
    print(f"Rounds:  {all_rounds}")
    print(f"Output:  {OUTPUT_DIR}\n")
    
    # Lade Val-Daten NUR EINMAL
    X_val, y_val, class_weights = load_validation_data(SPLIT, DATA, STATS)
    
    # Erstelle DataLoader
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    # ============================================================
    # EVALUATE ALLE RUNDEN
    # ============================================================
    results_summary = []
    
    for i, round_num in enumerate(all_rounds, 1):
        print(f"[{i}/{len(all_rounds)}] Evaluating Round {round_num}...")
        model_path = Path(MODEL_DIR) / f"model_round_{round_num}.pt"
        
        if not model_path.exists():
            print(f"⚠️  Skipping round {round_num} (model not found)")
            continue
        
        # Lade Modell
        model = MLP(in_dim=21)
        model.to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        
        # ✅ Nutze DEINE Eval-Funktion aus client_app!
        # CrossEntropyLoss mit Class-Weights (wie beim Training)
        crit = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
        
        # ✅ GENAU die gleiche Funktion wie während Training!
        avg_loss, n_samples, metrics = evaluate_multi_threshold(
            model, val_loader, crit, 
            threshold_grid=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        )
        
        # Rekonstruiere all_thresholds aus JSON-Strings (GLEICHES FORMAT wie round_80_eval.json!!)
        thresholds = json.loads(metrics["thresholds_json"])
        tp_list = json.loads(metrics["tp_json"])
        fp_list = json.loads(metrics["fp_json"])
        tn_list = json.loads(metrics["tn_json"])
        fn_list = json.loads(metrics["fn_json"])
        
        # Berechne komplette Metriken für jeden Threshold
        all_thresholds_list = []
        for idx, thr in enumerate(thresholds):
            tp = tp_list[idx]
            fp = fp_list[idx]
            tn = tn_list[idx]
            fn = fn_list[idx]
            
            n_pos = tp + fn  # Total positives
            n_neg = tn + fp  # Total negatives
            
            tpr = tp / n_pos if n_pos > 0 else 0.0
            fpr = fp / n_neg if n_neg > 0 else 0.0
            spec = 1.0 - fpr
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            precision = ppv
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            balanced_accuracy = (recall + spec) / 2.0
            youden = recall + spec - 1.0
            
            # Prevalence und alerts
            prevalence = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            alerts_per_1000 = (tp + fp) * 1000 / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0.0
            
            all_thresholds_list.append({
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "tpr": float(tpr),
                "recall": float(recall),
                "fpr": float(fpr),
                "spec": float(spec),
                "ppv": float(ppv),
                "precision": float(precision),
                "npv": float(npv),
                "f1": float(f1),
                "balanced_accuracy": float(balanced_accuracy),
                "youden": float(youden),
                "prevalence": float(prevalence),
                "alerts_per_1000": float(alerts_per_1000),
            })
        
        # Speichere JSON mit alle Infos (GENAU wie round_80_eval.json)
        result_dict = {
            "round": round_num,
            "metrics": {
                "auc": metrics.get("auc", 0.0),
                "n_samples": metrics.get("n_samples", 0),
                "all_thresholds": all_thresholds_list,
                "calib_edges_json": metrics.get("calib_edges_json", "[]"),
                "calib_bin_n_json": metrics.get("calib_bin_n_json", "[]"),
                "calib_bin_sum_pred_json": metrics.get("calib_bin_sum_pred_json", "[]"),
                "calib_bin_sum_true_json": metrics.get("calib_bin_sum_true_json", "[]"),
                "risk_edges_json": metrics.get("risk_edges_json", "[]"),
                "hist_pred_y0_json": metrics.get("hist_pred_y0_json", "[]"),
                "hist_pred_y1_json": metrics.get("hist_pred_y1_json", "[]"),
            },
            "model_checkpoint": str(model_path),
        }
        
        output_file = output_dir / f"round_{round_num}_eval.json"
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        # Summary - Nutze already computed best threshold (Index -3 = 0.55)
        best_threshold_data = all_thresholds_list[-3]
        
        results_summary.append({
            "round": round_num,
            "loss": avg_loss,
            "recall": best_threshold_data["recall"],
            "precision": best_threshold_data["precision"],
            "f1": best_threshold_data["f1"],
            "balanced_acc": best_threshold_data["balanced_accuracy"],
        })
    
    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 90)
    print("📊 SUMMARY")
    print("=" * 90)
    print(f"{'Round':<8} {'Loss':<12} {'Recall':<12} {'Precision':<12} {'F1':<12} {'Bal.Acc':<12}")
    print("-" * 90)
    for s in results_summary:
        print(f"{s['round']:<8.0f} {s['loss']:<12.6f} {s['recall']:<12.4f} {s['precision']:<12.4f} {s['f1']:<12.4f} {s['balanced_acc']:<12.4f}")
    
    # Speichere Summary
    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n✅ Results saved to:", output_dir)


if __name__ == "__main__":
    main()
