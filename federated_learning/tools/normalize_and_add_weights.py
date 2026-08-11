"""
Normalisiert ein bestehendes Parquet und ergänzt Class-Weights in norm_stats.json.

  Nutzt VORHANDENE Splits aus norm_stats.json
  Normalisiert nur Train/Val/Test (keine neuen Splits!)
  Berechnet Class-Weights auf Train
  Deine bestehenden Client-Splits bleiben unverändert!

Einmalig ausführen:
    python3 federated_learning/tools/normalize_and_add_weights.py --parquet data/diabetes.parquet --stats data/norm_stats.json --output data/diabetes_normalized.parquet --pos-weight-boost 1.5
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def normalize_parquet_and_add_weights(
    parquet_path: str,
    stats_path: str,
    output_parquet: str,
    pos_weight_boost: float = 2.0
):
    """
    Normalisiert ein Parquet und ergänzt Class-Weights in norm_stats.json.
    """
    print("=" * 60)
    print("🔧 PARQUET NORMALISIERUNG + CLASS-WEIGHTS")
    print("=" * 60)
    
    # 1. Lade Daten
    print(f"\n📂 Lade Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   Shape: {df.shape}")
    
    # 2. Lade Stats
    print(f"\n📋 Lade Stats: {stats_path}")
    meta = json.loads(Path(stats_path).read_text())
    
    target_col = meta["target"]
    mean = pd.Series(meta["mean"])
    std = pd.Series(meta["std"])
    train_idx = np.array(meta["train_idx"])
    
    # 3. Normalisierung
    print(f"\n🔄 Normalisiere Features...")
    y = df[target_col]
    row_ids = df["__row_id__"]
    
    X = df.drop(columns=[target_col, "__row_id__"])
    
    #   Normalisiere (mit Stats aus Training!)
    std_safe = std.replace(0, 1)  # guard: prevent division-by-zero for zero-variance features
    X_normalized = (X - mean) / std_safe
    
    # 4. Baue normalisiertes Parquet
    df_normalized = X_normalized.copy()
    df_normalized[target_col] = y
    df_normalized["__row_id__"] = row_ids
    
    # 5. Speichere normalisiertes Parquet
    print(f"\n💾 Speichere normalisiertes Parquet: {output_parquet}")
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_parquet(output_parquet, index=False)
    
    # 6. Berechne Class-Weights (auf Train-Set)
    print(f"\n⚖️  Berechne Class-Weights (auf Train-Set)...")
    
    # Filtere Train-Rows
    train_row_ids = set(train_idx)
    df_train = df_normalized[df_normalized["__row_id__"].isin(train_row_ids)]
    
    y_train = df_train[target_col].astype(int).values
    y_train_binary = (y_train >= 1).astype(int)  # 0=healthy, 1=prediabetic+diabetic
    
    pos = int(y_train_binary.sum())
    neg = int((1 - y_train_binary).sum())
    tot = max(1, pos + neg)
    
    # Class-Weights
    pos_weight = (neg / tot) * pos_weight_boost
    neg_weight = pos / tot
    
    # 7. Ergänze Stats
    meta["pos_weight"] = float(pos_weight)
    meta["neg_weight"] = float(neg_weight)
    meta["pos_weight_boost"] = float(pos_weight_boost)
    meta["train_pos_count"] = int(pos)
    meta["train_neg_count"] = int(neg)
    meta["train_pos_ratio"] = float(pos / tot)
    meta["train_neg_ratio"] = float(neg / tot)
    
    # 8. Speichere aktualisierte Stats
    print(f"\n💾 Aktualisiere Stats: {stats_path}")
    Path(stats_path).write_text(json.dumps(meta, indent=2))
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("  ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 60)
    print(f"\n📦 Normalisiertes Parquet:")
    print(f"   {output_parquet}")
    print(f"   Shape: {df_normalized.shape}")
    print(f"   Columns: {list(df_normalized.columns[:5])}... + {len(df_normalized.columns)-5} more")
    
    print(f"\n⚖️  Class-Weights (binary: healthy vs. at-risk):")
    print(f"   Positive (diabetic+prediabetic): {pos_weight:.4f} (count={pos:,}, ratio={pos/tot:.1%})")
    print(f"   Negative (healthy):              {neg_weight:.4f} (count={neg:,}, ratio={neg/tot:.1%})")
    print(f"   Boost Factor:                    {pos_weight_boost}")
    
    print(f"\n📋 Aktualisierte Stats:")
    print(f"   {stats_path}")
    print(f"   Neue Felder: pos_weight, neg_weight, train_pos_count, train_neg_count")
    
    print("\n💡 Nächste Schritte:")
    print(f"   1. Aktualisiere pyproject.toml:")
    print(f"      prepared-parquet = \"{output_parquet}\"")
    print(f"   2. Deine Splits ({meta.get('num_clients', '?')} Clients) bleiben unverändert!")
    print(f"   3. task.py lädt jetzt normalisierte Daten + vorberechnete Weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalisiert Parquet und ergänzt Class-Weights in norm_stats.json"
    )
    parser.add_argument("--parquet", required=True, help="Path to input parquet")
    parser.add_argument("--stats", required=True, help="Path to norm_stats.json")
    parser.add_argument("--output", required=True, help="Path to output normalized parquet")
    parser.add_argument("--pos-weight-boost", type=float, default=2.0,
                        help="Boost factor for positive class (default: 2.0)")
    
    args = parser.parse_args()
    
    normalize_parquet_and_add_weights(
        parquet_path=args.parquet,
        stats_path=args.stats,
        output_parquet=args.output,
        pos_weight_boost=args.pos_weight_boost
    )