# tools/prepare_data.py
import json
import os
from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare(csv_path, out_parquet, out_stats, test_size=0.2, val_size=0.1, seed=123):
    """
    Bereitet Daten vor mit Train/Val/Test Split:
    - Train: 70% (für Client-Training)
    - Validation: 10% (für Threshold-Tuning, client-lokal)
    - Test: 20% (für finale Evaluation, global)
    """
    # Load Dataset from Kaggle with  253,680 survey responses, its binary so 1 is prediabetic/diabetic and 0 is not
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(csv_path)
    
    # Divide feature set and target
    target_col = "Diabetes_binary"
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col]).astype(float)

    # ✅ FIX: Split in 3 Sets (Train, Val, Test)
    idx = np.arange(len(df))
    
    # 1) Train vs. (Val+Test)
    tr_idx, temp_idx = train_test_split(
        idx, 
        test_size=(test_size + val_size),  # 30% = 20% Test + 10% Val
        random_state=seed, 
        stratify=y
    )

    # 2) Val vs. Test
    y_temp = y[temp_idx]
    val_idx, te_idx = train_test_split(
        temp_idx,
        test_size=test_size / (test_size + val_size),  # 20/(20+10) = 2/3
        random_state=seed + 1,
        stratify=y_temp
    )

    # Normalisierungs-Statistiken (nur auf Train!)
    mean = X.iloc[tr_idx].mean().to_dict()
    std  = (X.iloc[tr_idx].std().replace(0, 1)).to_dict()

    # Speichere Parquet mit Row-IDs
    X[target_col] = y
    X["__row_id__"] = idx
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True) 
    X.to_parquet(out_parquet, index=False)

    # Speichere Stats
    stats = {
        "mean": mean, 
        "std": std, 
        "train_idx": tr_idx.tolist(), 
        "val_idx": val_idx.tolist(), 
        "test_idx": te_idx.tolist(), 
        "target": target_col
    }
    Path(out_stats).write_text(json.dumps(stats))
    
    print(" Daten vorbereitet.")
    print(f"   Train: {len(tr_idx)} samples ({len(tr_idx)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_idx)} samples ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(te_idx)} samples ({len(te_idx)/len(df)*100:.1f}%)")
    print(f"   Saved: {out_parquet}, {out_stats}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)   
    a = p.parse_args()
    prepare(a.csv, a.parquet, a.stats, a.test_size, a.val_size)
