# tools/make_splits.py
# Jeder Client liest später dieses Mapping ein und lädt nur die Zeilen, die ihm zugewiesen wurden, als Trainingsdaten.
# entweder IID (gleichverteilte, zufällige Aufteilung) oder Dirichlet (nicht-iid, realistischere Verteilung)
import json, numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd

# IID Partitionen: Daten werden zufällig und gleichmäßig auf die Clients verteilt
def iid_partitions(n, k, seed=123):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, k)
    return {str(i): parts[i].tolist() for i in range(k)}

# Dirichlet Partitionen: Daten werden ungleichmäßig verteilt, basierend auf einer Dirichlet-Verteilung
# Verteilung der Klassen pro Client variiert, was realistischere Szenarien simuliert
def dirichlet_partitions(labels, k, alpha=0.3, seed=123):
    rng = np.random.default_rng(seed)
    label2idx = defaultdict(list)
    for i, y in enumerate(labels):
        label2idx[int(y)].append(i)
    mapping = {str(i): [] for i in range(k)}
    for _, cls_idx in label2idx.items():
        cls_idx = np.array(cls_idx)
        rng.shuffle(cls_idx)
        p = rng.dirichlet([alpha]*k)
        splits = (np.cumsum(p)*len(cls_idx)).astype(int)
        parts = np.split(cls_idx, splits[:-1])
        for i in range(k):
            mapping[str(i)].extend(parts[i].tolist())
    for i in range(k): rng.shuffle(mapping[str(i)])
    return mapping

def proportional_val_split(train_mapping, val_idx, seed=123):
    """
    Verteilt Val-Daten PROPORTIONAL zu Train-Größen.
    
    Beispiel:
        Client 0: Train=20000 (10% von total) → Val=2500 (10% von total)
        Client 1: Train=50000 (25% von total) → Val=6250 (25% von total)
        Client 2: Train=5000  (2.5% von total) → Val=625  (2.5% von total)
    
    → Jeder Client hat ca. 10-15% Val relativ zu seinem Train!
    """
    rng = np.random.default_rng(seed)
    
    # Berechne Train-Größen pro Client
    train_sizes = {cid: len(idxs) for cid, idxs in train_mapping.items()}
    total_train = sum(train_sizes.values())
    
    # Shuffle Val-Indices (für Fairness)
    val_idx_shuffled = val_idx.copy()
    rng.shuffle(val_idx_shuffled)
    
    val_mapping = {}
    start = 0
    
    for cid in sorted(train_mapping.keys(), key=int):
        # Proportionaler Anteil: Client mit 10% Train → 10% Val
        proportion = train_sizes[cid] / total_train
        n_val = int(len(val_idx) * proportion)
        
        # Mindestens 1 Sample (für sehr kleine Clients)
        n_val = max(1, n_val)
        
        # Extrahiere Val-Samples
        end = min(start + n_val, len(val_idx_shuffled))
        val_mapping[cid] = val_idx_shuffled[start:end].tolist()
        start = end
    
    # Rest-Samples an letzten Client (falls Rundungsfehler)
    if start < len(val_idx_shuffled):
        last_cid = sorted(train_mapping.keys(), key=int)[-1]
        val_mapping[last_cid].extend(val_idx_shuffled[start:].tolist())
    
    return val_mapping

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-partitions", type=int, required=True)
    p.add_argument("--mode", choices=["iid","dirichlet"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()

    # Lade Daten und Metadaten
    df = pd.read_parquet(a.parquet)
    meta = json.loads(Path(a.stats).read_text())

    # indizes der Trainings- und Validierungsdaten im Parquet
    train_idx = np.array(meta["train_idx"])
    val_idx = np.array(meta["val_idx"])
    
    # Erstelle Mapping: row_id -> DataFrame-Position
    row_id_to_pos = {int(row_id): pos for pos, row_id in enumerate(df["__row_id__"])}
    
    # Konvertiere train_idx (globale Row-IDs) zu DataFrame-Positionen
    train_pos = np.array([row_id_to_pos[int(rid)] for rid in train_idx])
    val_pos = np.array([row_id_to_pos[int(rid)] for rid in val_idx])
    
    # Nutze train_pos/val_pos statt meta["train_idx"]!
    y_train = df.iloc[train_pos][meta["target"]].astype(int).values
    y_val = df.iloc[val_pos][meta["target"]].astype(int).values

    # Erstelle Partitionen basierend auf dem gewählten Modus
    if a.mode == "iid":
        local_train = iid_partitions(len(train_idx), a.num_partitions, seed=a.seed)
    else:
        # Dirichlet gibt lokale Indices zurück (0, 1, 2, ..., len(y_train)-1)
        local_train = dirichlet_partitions(y_train, a.num_partitions, alpha=a.alpha, seed=a.seed)

    # Mappe lokale Indices → train_idx Positionen → globale Row-IDs
    global_train_map = {
        cid: [int(train_idx[i]) for i in idxs]
        for cid, idxs in local_train.items()
    }

    # Val PROPORTIONAL zu Train aufteilen
    global_val_map = proportional_val_split(global_train_map, val_idx, seed=a.seed + 1)

    output = {
        "train": global_train_map,
        "val": global_val_map
    }

    Path(a.out).write_text(json.dumps(output))

    print(f"   Created {a.num_partitions} partitions ({a.mode}, alpha={a.alpha if a.mode=='dirichlet' else 'N/A'})")
    print(f"   Train partitions: {len(global_train_map)} clients")
    print(f"   Val partitions:   {len(global_val_map)} clients")
    
    # Zeige Val/Train Ratio pro Client
    for cid in sorted(global_train_map.keys(), key=int):
        train_n = len(global_train_map[cid])
        val_n = len(global_val_map[cid])
        ratio = val_n / train_n * 100 if train_n else 0
        print(f"   Client {cid}: Train={train_n:5d}, Val={val_n:4d} ({ratio:5.1f}%)")
    
    print(f"   Output: {a.out}")

            # --> 
            # global_map = {
            #   "0": [10, 30, 50, ...], // ein Krankenhaus mit den Partient:innen
            #   "1": [20, 40, 60, ...],
            #   ...
            # }
