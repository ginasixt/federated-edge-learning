#!/usr/bin/env python3
"""Analyse der Klassenverteilung über Clients hinweg.

Das Skript liest ein Split-JSON mit `train`/`val`-Clients und lädt die
benötigten Zeilen über `__row_id__` aus dem Parquet. Danach werden pro Client
und global Kennzahlen zur Label-Verteilung berechnet.

Beispiel:
    python3 federated_learning/tools/analyze_client_label_distribution.py \
        --split splits_iid_scaling/splits_iid_16384_clients.json \
        --parquet data/diabetes_normalized.parquet \
        --stats data/norm_stats.json \
        --output-dir result/label_analysis/16384_clients
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_split(split_path: str) -> Dict[str, Any]:
    return json.loads(Path(split_path).read_text())


def _load_stats(stats_path: str) -> Dict[str, Any]:
    return json.loads(Path(stats_path).read_text())


def _collect_row_ids(split_data: Dict[str, Any], include_val: bool = False) -> List[int]:
    row_ids: List[int] = []

    for cid, samples in split_data.get("train", {}).items():
        row_ids.extend(int(row_id) for row_id in samples)

    if include_val:
        for cid, samples in split_data.get("val", {}).items():
            row_ids.extend(int(row_id) for row_id in samples)

    return row_ids


def _load_rows_from_parquet(parquet_path: str, row_ids: Sequence[int]) -> pd.DataFrame:
    if not row_ids:
        return pd.DataFrame()

    try:
        return pd.read_parquet(parquet_path, filters=[("__row_id__", "in", list(row_ids))])
    except Exception:
        df = pd.read_parquet(parquet_path)
        return df[df["__row_id__"].isin(row_ids)].copy()


def _gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def _entropy_from_ratio(ratio: float) -> float:
    if ratio <= 0.0 or ratio >= 1.0:
        return 0.0
    return float(-(ratio * np.log2(ratio) + (1.0 - ratio) * np.log2(1.0 - ratio)))


def _safe_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _analyze_split_part(
    df: pd.DataFrame,
    split_dict: Dict[str, List[int]],
    target_col: str,
    split_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if df.empty or not split_dict:
        return [], {
            "split_name": split_name,
            "clients": 0,
            "total_samples": 0,
            "total_pos": 0,
            "total_neg": 0,
            "global_positive_ratio": 0.0,
            "global_negative_ratio": 0.0,
        }

    row_id_to_label = {
        int(row_id): int(label)
        for row_id, label in zip(df["__row_id__"].values, df[target_col].astype(int).values)
    }

    client_rows: List[Dict[str, Any]] = []

    for client_id_str, row_ids in split_dict.items():
        client_id = int(client_id_str)
        labels = np.array([row_id_to_label[int(row_id)] for row_id in row_ids], dtype=int)
        total = int(labels.size)
        pos = int(np.sum(labels == 1))
        neg = int(np.sum(labels == 0))
        pos_ratio = float(pos / total) if total > 0 else 0.0
        neg_ratio = float(neg / total) if total > 0 else 0.0

        client_rows.append(
            {
                "client_id": client_id,
                "samples": total,
                "positives": pos,
                "negatives": neg,
                "positive_ratio": pos_ratio,
                "negative_ratio": neg_ratio,
                "label_entropy": _entropy_from_ratio(pos_ratio),
            }
        )

    samples = [row["samples"] for row in client_rows]
    positives = [row["positives"] for row in client_rows]
    pos_ratios = [row["positive_ratio"] for row in client_rows]
    entropies = [row["label_entropy"] for row in client_rows]

    total_samples = int(np.sum(samples))
    total_pos = int(np.sum(positives))
    total_neg = int(total_samples - total_pos)
    global_positive_ratio = float(total_pos / total_samples) if total_samples > 0 else 0.0
    global_negative_ratio = float(total_neg / total_samples) if total_samples > 0 else 0.0

    summary = {
        "split_name": split_name,
        "clients": int(len(client_rows)),
        "total_samples": total_samples,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "global_positive_ratio": global_positive_ratio,
        "global_negative_ratio": global_negative_ratio,
        "samples_per_client": {
            "mean": _safe_float(np.mean(samples)),
            "std": _safe_float(np.std(samples)),
            "min": int(np.min(samples)),
            "max": int(np.max(samples)),
            "median": _safe_float(np.median(samples)),
        },
        "positives_per_client": {
            "mean": _safe_float(np.mean(positives)),
            "std": _safe_float(np.std(positives)),
            "min": int(np.min(positives)),
            "max": int(np.max(positives)),
            "median": _safe_float(np.median(positives)),
        },
        "positive_ratio_per_client": {
            "mean": _safe_float(np.mean(pos_ratios)),
            "std": _safe_float(np.std(pos_ratios)),
            "min": _safe_float(np.min(pos_ratios)),
            "max": _safe_float(np.max(pos_ratios)),
            "median": _safe_float(np.median(pos_ratios)),
            "q10": _safe_float(np.quantile(pos_ratios, 0.10)),
            "q90": _safe_float(np.quantile(pos_ratios, 0.90)),
        },
        "label_entropy_per_client": {
            "mean": _safe_float(np.mean(entropies)),
            "std": _safe_float(np.std(entropies)),
            "min": _safe_float(np.min(entropies)),
            "max": _safe_float(np.max(entropies)),
        },
        "clients_with_zero_positive": int(sum(1 for value in positives if value == 0)),
        "clients_with_only_positive": int(sum(1 for row in client_rows if row["samples"] > 0 and row["negatives"] == 0)),
        "positive_ratio_gini": _gini(pos_ratios),
        "samples_gini": _gini(samples),
    }

    return client_rows, summary


def _print_summary(summary: Dict[str, Any], top_clients: List[Dict[str, Any]], bottom_clients: List[Dict[str, Any]]) -> None:
    print("=" * 78)
    print(f"{summary['split_name'].upper()} LABEL ANALYSIS")
    print("=" * 78)
    print(f"Clients:                 {summary['clients']}")
    print(f"Total samples:           {summary['total_samples']}")
    print(f"Positives / Negatives:   {summary['total_pos']} / {summary['total_neg']}")
    print(f"Global positive ratio:    {summary['global_positive_ratio']:.4f}")
    print()
    print("Per-client sample counts")
    print(
        f"  mean={summary['samples_per_client']['mean']:.2f}, std={summary['samples_per_client']['std']:.2f}, "
        f"min={summary['samples_per_client']['min']}, max={summary['samples_per_client']['max']}, "
        f"median={summary['samples_per_client']['median']:.2f}"
    )
    print("Per-client positive counts")
    print(
        f"  mean={summary['positives_per_client']['mean']:.2f}, std={summary['positives_per_client']['std']:.2f}, "
        f"min={summary['positives_per_client']['min']}, max={summary['positives_per_client']['max']}, "
        f"median={summary['positives_per_client']['median']:.2f}"
    )
    print("Per-client positive ratios")
    print(
        f"  mean={summary['positive_ratio_per_client']['mean']:.4f}, std={summary['positive_ratio_per_client']['std']:.4f}, "
        f"min={summary['positive_ratio_per_client']['min']:.4f}, max={summary['positive_ratio_per_client']['max']:.4f}, "
        f"q10={summary['positive_ratio_per_client']['q10']:.4f}, q90={summary['positive_ratio_per_client']['q90']:.4f}"
    )
    print(f"Clients with zero positives:    {summary['clients_with_zero_positive']}")
    print(f"Clients with only positives:    {summary['clients_with_only_positive']}")
    print(f"Gini of positive ratios:        {summary['positive_ratio_gini']:.4f}")
    print(f"Entropy mean / std:             {summary['label_entropy_per_client']['mean']:.4f} / {summary['label_entropy_per_client']['std']:.4f}")
    print()

    if top_clients:
        print("Highest positive ratios")
        for row in top_clients:
            print(
                f"  client {row['client_id']:>6}: ratio={row['positive_ratio']:.4f}, "
                f"pos={row['positives']}, neg={row['negatives']}, n={row['samples']}"
            )
        print()

    if bottom_clients:
        print("Lowest positive ratios")
        for row in bottom_clients:
            print(
                f"  client {row['client_id']:>6}: ratio={row['positive_ratio']:.4f}, "
                f"pos={row['positives']}, neg={row['negatives']}, n={row['samples']}"
            )
        print()


def analyze_client_label_distribution(
    split_path: str,
    parquet_path: str,
    stats_path: str,
    output_dir: str,
    include_val: bool = False,
    top_k: int = 10,
) -> Dict[str, Any]:
    split_data = _load_split(split_path)
    stats = _load_stats(stats_path)
    target_col = stats.get("target", "target")

    row_ids = _collect_row_ids(split_data, include_val=include_val)
    df = _load_rows_from_parquet(parquet_path, row_ids)

    if df.empty:
        raise ValueError("No rows could be loaded from the parquet file for the provided split.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows, train_summary = _analyze_split_part(df, split_data.get("train", {}), target_col, "train")
    val_rows: List[Dict[str, Any]] = []
    val_summary: Optional[Dict[str, Any]] = None

    if include_val and split_data.get("val"):
        val_rows, val_summary = _analyze_split_part(df, split_data.get("val", {}), target_col, "val")

    train_rows_sorted = sorted(train_rows, key=lambda row: row["positive_ratio"], reverse=True)
    top_clients = train_rows_sorted[:top_k]
    bottom_clients = list(reversed(train_rows_sorted[-top_k:])) if train_rows_sorted else []

    train_rows_file = output_path / "client_label_distribution_train.csv"
    pd.DataFrame(train_rows).sort_values("client_id").to_csv(train_rows_file, index=False)

    if val_rows:
        val_rows_file = output_path / "client_label_distribution_val.csv"
        pd.DataFrame(val_rows).sort_values("client_id").to_csv(val_rows_file, index=False)
    else:
        val_rows_file = None

    summary: Dict[str, Any] = {
        "split_path": split_path,
        "parquet_path": parquet_path,
        "stats_path": stats_path,
        "target_column": target_col,
        "include_val": include_val,
        "train": train_summary,
        "train_rows_file": str(train_rows_file),
    }

    if val_summary is not None:
        summary["val"] = val_summary
        summary["val_rows_file"] = str(val_rows_file) if val_rows_file else None

    summary_file = output_path / "client_label_distribution_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    _print_summary(train_summary, top_clients, bottom_clients)

    print(f"Saved summary: {summary_file}")
    print(f"Saved train rows: {train_rows_file}")
    if val_rows_file is not None:
        print(f"Saved val rows:   {val_rows_file}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze label distribution across clients")
    parser.add_argument("--split", required=True, help="Path to split JSON file")
    parser.add_argument("--parquet", required=True, help="Path to prepared parquet file")
    parser.add_argument("--stats", default="data/norm_stats.json", help="Path to normalization stats JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    parser.add_argument("--include-val", action="store_true", help="Also analyze validation clients")
    parser.add_argument("--top-k", type=int, default=10, help="How many clients to show in top/bottom lists")

    args = parser.parse_args()

    analyze_client_label_distribution(
        split_path=args.split,
        parquet_path=args.parquet,
        stats_path=args.stats,
        output_dir=args.output_dir,
        include_val=args.include_val,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
