# federated_learning/tools/create_iid_scaling_splits.py
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from make_splits import iid_partitions, proportional_val_split

"""
    IID Scaling Analysis - Perfekt vergleichbare Skalierung
    
    Run Script with:
    python3 federated_learning/tools/create_iid_scaling_splits.py \
        --parquet data/diabetes.parquet \
        --stats data/norm_stats.json \
        --output-dir splits_iid_scaling \
        --seed 123
"""

def create_iid_scaling_partitions(
    parquet_path: str, 
    stats_path: str, 
    output_dir: str, 
    seed: int = 123,
    min_samples_per_client: int = 1
):
    """
    Erstellt IID-Skalierungsanalyse für vergleichbare FL-Performance.
    
    - IID = zufällige, gleichmäßige Verteilung (OHNE Berücksichtigung der Labels)
    - Client-Anzahlen: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    - Mindestens min_samples_per_client pro Client für sinnvolle Ergebnisse
    """
    # Lade Daten
    df = pd.read_parquet(parquet_path)
    meta = json.loads(Path(stats_path).read_text())
    
    train_idx = np.array(meta["train_idx"])
    val_idx = np.array(meta["val_idx"])
    
    print(f"📊 IID SCALING ANALYSIS")
    print(f"   ════════════════════════════════════════")
    print(f"   Total Train Samples: {len(train_idx):,}")
    print(f"   Total Val Samples:   {len(val_idx):,}")
    print(f"   Min Samples/Client:  {min_samples_per_client}")
    print(f"   Distribution:        Random uniform (IID)")
    
    # Berechne sinnvolle Client-Anzahlen (Potenzen von 2)
    max_clients = len(train_idx)
    
    client_counts = []
    n = 2
    while n < max_clients:
        client_counts.append(n)
        n *= 2
    
    # Finale Anzahl hinzufügen (falls nicht schon drin) 1 Client pro Sample
    if max_clients not in client_counts:
        client_counts.append(max_clients)
    
    print(f"   Max feasible clients: {max_clients:,}")
    print(f"   Testing: {client_counts}")
    
    # Output-Verzeichnis erstellen
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Mapping für DataFrame-Positionen (für Label-Statistiken)
    row_id_to_pos = {int(row_id): pos for pos, row_id in enumerate(df["__row_id__"])}
    train_pos = np.array([row_id_to_pos[int(rid)] for rid in train_idx])
    
    # Labels für Statistiken (nur zur Analyse, NICHT für Verteilung verwendet!)
    y_train = df.iloc[train_pos][meta["target"]].astype(int).values
    
    results = []
    
    for num_clients in client_counts:
        print(f"\n🔄 Creating IID split for {num_clients} clients...")
        
        avg_samples = len(train_idx) / num_clients
        
        # IID-Partitionierung: REIN zufällige Verteilung, Labels werden IGNORIERT
        local_train = iid_partitions(len(train_idx), num_clients, seed=seed)
        
        # Mappe lokale Indices zu globalen Row-IDs
        global_train_map = {
            cid: [int(train_idx[i]) for i in idxs]
            for cid, idxs in local_train.items()
        }
        
        # Proportionale Val-Verteilung
        global_val_map = proportional_val_split(global_train_map, val_idx, seed=seed + 1)
        
        # Output-Datei
        output_file = output_path / f"splits_iid_{num_clients}_clients.json"
        
        output = {
            "train": global_train_map,
            "val": global_val_map,
            "meta": {
                "paradigm": "iid_scaling",
                "num_clients": num_clients,
                "split_type": "iid",
                "total_train_samples": len(train_idx),
                "total_val_samples": len(val_idx),
                "seed": seed,
                "avg_samples_per_client": avg_samples,
                "distribution": "random_uniform_iid"
            }
        }
        
        # Speichere Split
        output_file.write_text(json.dumps(output, indent=2))
        
        # Statistiken berechnen
        train_sizes = [len(idxs) for idxs in global_train_map.values()]
        val_sizes = [len(idxs) for idxs in global_val_map.values()]
        
        # Label-Verteilung pro Client analysieren (nur zur Kontrolle!)
        client_label_stats = []
        for cid, train_samples in global_train_map.items():
            # Hole Labels für diesen Client
            client_pos = [row_id_to_pos[int(rid)] for rid in train_samples]
            client_labels = df.iloc[client_pos][meta["target"]].astype(int).values
            
            pos_ratio = (client_labels == 1).mean() if len(client_labels) > 0 else 0.0
            client_label_stats.append(pos_ratio)
        
        # Wie stark weicht die Label-Verteilung zwischen Clients ab?
        label_std = np.std(client_label_stats) if client_label_stats else 0.0
        global_pos_ratio = (y_train == 1).mean()
        
        results.append({
            "num_clients": int(num_clients),
            "file": str(output_file),
            "avg_train_per_client": float(np.mean(train_sizes)),
            "std_train_per_client": float(np.std(train_sizes)),
            "min_train_per_client": int(np.min(train_sizes)),
            "max_train_per_client": int(np.max(train_sizes)),
            "avg_val_per_client": float(np.mean(val_sizes)),
            "label_distribution_std": float(label_std),  # Natürliche Variation bei IID
            "global_positive_ratio": float(global_pos_ratio),
            "paradigm": "iid_scaling"
        })
        
        print(f"   ✅ {output_file.name}")
        print(f"      Samples/Client: {avg_samples:.1f} (range: {np.min(train_sizes)}-{np.max(train_sizes)})")
        print(f"      Val/Client:     {np.mean(val_sizes):.1f}")
        print(f"      Label Std:      {label_std:.4f} (natural IID variation)")
        print(f"      Global Pos%:    {global_pos_ratio:.1%}")
        
        # Warnung bei zu wenigen Samples
        if avg_samples < min_samples_per_client:
            print(f"   ⚠️  Warning: Avg samples ({avg_samples:.1f}) below minimum ({min_samples_per_client})")
    
    # Summary-Report erstellen
    summary_file = output_path / "iid_scaling_summary.json"
    summary = {
        "analysis_type": "iid_scaling", 
        "description": "IID scaling analysis - random uniform data distribution across clients",
        "advantages": [
            "Perfect comparability across client counts",
            "Pure scaling effects (no confounding from data heterogeneity)",
            "Baseline for comparing with non-IID Dirichlet results",
            "Natural label variation due to random sampling"
        ],
        "created_splits": results,
        "parameters": {
            "seed": int(seed),
            "min_samples_per_client": int(min_samples_per_client),
            "total_train_samples": int(len(train_idx)),
            "total_val_samples": int(len(val_idx)),
            "parquet_path": parquet_path,
            "stats_path": stats_path
        },
        "global_statistics": {
            "positive_ratio": float((y_train == 1).mean()),
            "negative_ratio": float((y_train == 0).mean()),
            "total_samples": int(len(y_train))
        }
    }
    
    summary_file.write_text(json.dumps(summary, indent=2))
    
    print(f"\n📋 IID Scaling Summary:")
    print(f"   Created Splits:     {len(client_counts)}")
    print(f"   Client Range:       {min(client_counts)} → {max(client_counts)}")
    print(f"   Data Distribution:  Random uniform (IID)")
    print(f"   Summary Report:     {summary_file}")
    print(f"   Output Directory:   {output_dir}/")
    
    # Vergleichstabelle
    print(f"\n📊 Quick Overview:")
    print(f"   {'Clients':>8} {'Avg/Client':>12} {'Min':>6} {'Max':>6} {'Label Std':>10}")
    print(f"   {'-'*8} {'-'*12} {'-'*6} {'-'*6} {'-'*10}")
    for result in results:
        n = result['num_clients']
        avg = result['avg_train_per_client']
        min_s = result['min_train_per_client'] 
        max_s = result['max_train_per_client']
        std = result['label_distribution_std'] 
        print(f"   {n:8d} {avg:12.1f} {min_s:6d} {max_s:6d} {std:10.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Create IID scaling splits for federated learning")
    parser.add_argument("--parquet", required=True, help="Path to prepared parquet file")
    parser.add_argument("--stats", required=True, help="Path to normalization stats JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for split files")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum samples per client")
    
    args = parser.parse_args()
    
    create_iid_scaling_partitions(
        parquet_path=args.parquet,
        stats_path=args.stats,
        output_dir=args.output_dir,
        seed=args.seed,
        min_samples_per_client=args.min_samples
    )


if __name__ == "__main__":
    main()