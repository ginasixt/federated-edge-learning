"""
Passt Val-Daten in splits_iid_8192_clients.json an:
- Konzentriert Val-Samples auf weniger Clients
- Ziel: avg_val_per_client ≈ avg_train_per_client (10-11 statt 1.5)
"""
import json
import numpy as np
from pathlib import Path

# Lade aktuellen Split
split_path = Path("splits_iid_scaling/splits_iid_8192_clients.json")
with open(split_path) as f:
    data = json.load(f)

train_map = data["train"]
val_indices_old = []

# Flache alle alten Val-Indices
for entries in data["val"].values():
    val_indices_old.extend(entries)

val_indices_old = np.array(val_indices_old)
total_val_samples = len(val_indices_old)

print(f"📊 Aktuella statisitk:")
print(f"   Train Clients: {len(train_map)}")
print(f"   Total Val-Samples: {total_val_samples}")
print(f"   Avg Val/Client (alt): {total_val_samples / len(train_map):.2f}")
print()

# --- Strategie: Val auf weniger Clients verteilen ---
# Ziel: ~10 Val-Samples pro Client
target_val_per_client = total_val_samples
num_val_clients = int(np.ceil(total_val_samples / target_val_per_client))

print(f"🎯 Neue Verteilung:")
print(f"   Target Val-Samples/Client: {target_val_per_client}")
print(f"   Benötigte Clients: {num_val_clients}")
print(f"   Avg Val/Client (neu): {total_val_samples / num_val_clients:.2f}")
print()

# Shuffle Val-Indices
rng = np.random.default_rng(seed=123)
val_indices_shuffled = val_indices_old.copy()
rng.shuffle(val_indices_shuffled)

# Verteile Val-Samples auf die ersten num_val_clients
val_map_new = {}
samples_per_client = np.full(num_val_clients, target_val_per_client, dtype=int)

# Verteile restliche Samples (Rundungsfehler)
remainder = total_val_samples - samples_per_client.sum()
for i in range(remainder):
    samples_per_client[i] += 1

start = 0
client_ids = sorted(train_map.keys(), key=int)

for i in range(num_val_clients):
    cid = client_ids[i]
    n = samples_per_client[i]
    end = start + n
    val_map_new[cid] = val_indices_shuffled[start:end].tolist()
    start = end
    
    if i < 10 or i >= num_val_clients - 5:  # Zeige erste 10 und letzte 5
        print(f"   Client {cid}: {n} Val-Samples")
    elif i == 10:
        print(f"   ...")

print()

# ✅ WICHTIG: NUR Clients mit echten Val-Daten speichern (keine leeren Listen!)
# Clients ohne Val-Daten existieren einfach nicht im val_map

print(f"✅ Zusammenfassung:")
print(f"   Clients MIT Val-Daten: {len(val_map_new)}")
print(f"   Clients OHNE Val-Daten: {len(client_ids) - len(val_map_new)}")
print(f"   Total Val-Samples verteilt: {sum(len(v) for v in val_map_new.values())}")
print()

# Speichere Val-Client-Range im Meta (für effiziente Evaluation!)
val_client_ids_sorted = sorted([int(cid) for cid in val_map_new.keys()])
val_client_range = {
    "min": val_client_ids_sorted[0] if val_client_ids_sorted else 0,
    "max": val_client_ids_sorted[-1] if val_client_ids_sorted else 0
}

print(f"📊 Val-Client-Range für effiziente Evaluation:")
print(f"   Min: {val_client_range['min']}, Max: {val_client_range['max']}")
print()

# Neuen Split speichern
data["val"] = val_map_new
if "meta" not in data:
    data["meta"] = {}
data["meta"]["val_client_range"] = val_client_range

output_path = Path("splits_iid_scaling/splits_iid_8192_clients_adjusted.json")
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"💾 Gespeichert: {output_path}")
print()
print(f"📌 Verwendung in deinem Code:")
print(f"   Ändere Zeile in task.py/server_app.py:")
print(f"   FROM: 'splits_iid_scaling/splits_iid_8192_clients.json'")
print(f"   TO:   'splits_iid_scaling/splits_iid_8192_clients_adjusted.json'")
