"""Create a centralized validation split from an existing split JSON.

Default behavior:
- keep the train mapping unchanged
- move all validation samples to a separate key (not a client)
- store centralized validation metadata for server-side evaluation
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="splits_iid_scaling/splits_iid_16_clients.json",
        help="Input split JSON path",
    )
    parser.add_argument(
        "--output",
        default="splits_iid_scaling/splits_iid_16_clients_centralized_val.json",
        help="Output split JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r") as f:
        data = json.load(f)

    train_map = data.get("train", {})
    val_map = data.get("val", {})

    # Sammle alle Val-Row-IDs
    all_val_row_ids = []
    for entries in val_map.values():
        all_val_row_ids.extend(entries)

    # Deduplicate
    seen = set()
    all_val_row_ids = [rid for rid in all_val_row_ids if not (rid in seen or seen.add(rid))]

    # ✅ NEU: Val als separate Liste, nicht als Client
    data["val"] = {}  # Leer – kein Client hat Val-Daten
    data["centralized_val_row_ids"] = all_val_row_ids  # Server lädt diese direkt

    # Meta aktualisieren
    if "meta" not in data:
        data["meta"] = {}

    data["meta"]["centralized_val"] = True
    data["meta"]["centralized_val_num_samples"] = len(all_val_row_ids)
    
    # Diese Felder nicht mehr nötig, aber für Kompatibilität entfernen
    data["meta"].pop("centralized_val_client_id", None)
    data["meta"].pop("val_client_range", None)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Centralized validation split created")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Train clients: {len(train_map)}")
    print(f"Validation samples: {len(all_val_row_ids)} (server-side)")


if __name__ == "__main__":
    main()
