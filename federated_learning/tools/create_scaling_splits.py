# federated_learning/tools/create_scaling_splits.py
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from make_splits import dirichlet_partitions, proportional_val_split

"""
    Run Script with:

    python3 federated_learning/tools/create_scaling_splits.py \
        --parquet data/diabetes.parquet \
        --stats data/norm_stats.json \
        --output-dir splits_scaling \
        --alpha 0.3
"""


def create_scaling_partitions(parquet_path: str, stats_path: str, output_dir: str, alpha: float = 0.3, seed: int = 123):
    """
    Erstellt Partitionierungen für Skalierungsanalyse:
    2, 4, 8, 16, 32, 64, 128, 256, ... Clients bis max. Anzahl Samples
    """
    # Lade Daten
    df = pd.read_parquet(parquet_path)
    meta = json.loads(Path(stats_path).read_text())
    
    train_idx = np.array(meta["train_idx"])
    val_idx = np.array(meta["val_idx"])
    
    # Berechne maximale Client-Anzahl (1 Client pro Sample)
    max_clients = len(train_idx)
    
    # Client-Anzahlen: Potenzen von 2, dann finale Anzahl
    client_counts = []
    n = 2
    while n < max_clients:
        client_counts.append(n)
        n *= 2
    
    # Finale Anzahl hinzufügen (falls nicht schon drin)
    if max_clients not in client_counts:
        client_counts.append(max_clients)
    
    print(f"📊 Erstelle Splits für {len(client_counts)} verschiedene Client-Anzahlen:")
    print(f"   Range: {min(client_counts)} → {max(client_counts)} Clients")
    print(f"   Total Samples: {len(train_idx):,} (Train) + {len(val_idx):,} (Val)")
    
    # Output-Verzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Mapping für DataFrame-Positionen
    row_id_to_pos = {int(row_id): pos for pos, row_id in enumerate(df["__row_id__"])}
    train_pos = np.array([row_id_to_pos[int(rid)] for rid in train_idx])
    val_pos = np.array([row_id_to_pos[int(rid)] for rid in val_idx])
    
    # Labels für Dirichlet-Partitionierung
    y_train = df.iloc[train_pos][meta["target"]].astype(int).values
    
    results = []
    
    for num_clients in client_counts:
        print(f"\n Erstelle Split für {num_clients} Clients...")
        
        # Dirichlet-Partitionierung auf Train-Daten
        local_train = dirichlet_partitions(y_train, num_clients, alpha=alpha, seed=seed)
        
        # Mappe zu globalen Row-IDs
        global_train_map = {
            cid: [int(train_idx[i]) for i in idxs]
            for cid, idxs in local_train.items()
        }
        
        # Proportionale Val-Verteilung, seed=seed+1, weil anders wie Train -> verhindert systematische Muster (zb verstecke Sortierung im Datensatz zb Alter)
        global_val_map = proportional_val_split(global_train_map, val_idx, seed=seed + 1)
        
        # Output-Datei
        output_file = output_path / f"splits_dirichlet_{num_clients}_a{str(alpha).replace('.', '')}.json"
        
        output = {
            "train": global_train_map,
            "val": global_val_map,
            "meta": {
                "num_clients": num_clients,
                "alpha": alpha,
                "total_train_samples": len(train_idx),
                "total_val_samples": len(val_idx),
                "seed": seed
            }
        }
        
        # Speichere Split
        output_file.write_text(json.dumps(output, indent=2))
        
        # Statistiken
        train_sizes = [len(idxs) for idxs in global_train_map.values()]
        val_sizes = [len(idxs) for idxs in global_val_map.values()]
        
        results.append({
            "num_clients": int(num_clients),  # Explizit zu Python int
            "file": str(output_file),
            "avg_train_per_client": float(np.mean(train_sizes)),  # Explizit zu Python float
            "std_train_per_client": float(np.std(train_sizes)),
            "min_train_per_client": int(np.min(train_sizes)),
            "max_train_per_client": int(np.max(train_sizes)),
            "avg_val_per_client": float(np.mean(val_sizes)),
        })
        
        print(f"   {output_file.name}")
        print(f"      Train/Client: {np.mean(train_sizes):.1f} ± {np.std(train_sizes):.1f} (range: {np.min(train_sizes)}-{np.max(train_sizes)})")
        print(f"      Val/Client:   {np.mean(val_sizes):.1f} ± {np.std(val_sizes):.1f}")
    
    # Summary-Report erstellen
    summary_file = output_path / "scaling_splits_summary.json"
    summary = {
        "created_splits": results,
        "parameters": {
            "alpha": float(alpha),  # Auch hier konvertieren
            "seed": int(seed),
            "total_train_samples": int(len(train_idx)),
            "total_val_samples": int(len(val_idx)),
            "parquet_path": parquet_path,
            "stats_path": stats_path
        }
    }
    
    summary_file.write_text(json.dumps(summary, indent=2))
    
    print(f"\n Summary Report: {summary_file}")
    print(f" Created {len(client_counts)} split files in {output_dir}/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Create scaling splits for federated learning analysis")
    parser.add_argument("--parquet", required=True, help="Path to prepared parquet file")
    parser.add_argument("--stats", required=True, help="Path to normalization stats JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for split files")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha parameter")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    
    args = parser.parse_args()
    
    create_scaling_partitions(
        parquet_path=args.parquet,
        stats_path=args.stats,
        output_dir=args.output_dir,
        alpha=args.alpha,
        seed=args.seed
    )


if __name__ == "__main__":
    main()