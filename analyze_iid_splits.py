"""
Analyze IID FL split JSON files for a scalability study.

For each split file, the script computes client-level training statistics such as:
- number of clients
- total train samples
- mean/std/min/max train samples per client
- mean/std/min/max positive samples per client
- clients without positive samples
- class ratio / positive rate

Expected split format:
{
  "train": {"0": [idx1, idx2, ...], "1": [...]},
  "val":   {"0": [idx1, idx2, ...], ...},
  "test":  [...]   # optional
}

Example usage:
python analyze_iid_splits.py \
  --splits-dir splits_iid_scaling \
  --data-path data/diabetes_normalized.parquet \
  --label-col Diabetes_binary

Output:
- split_scaling_summary.csv
- split_scaling_summary.xlsx (if openpyxl is installed)
- split_client_details/<split_name>_client_stats.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_LABEL_CANDIDATES = [
    "Diabetes_binary",
    "diabetes_binary",
    "label",
    "target",
    "y",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze client-level statistics of FL split JSON files."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("splits_iid_scaling"),
        help="Directory containing split JSON files.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("diabetes_normalized.parquet"),
        help="Parquet file containing the dataset and label column.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Diabetes_binary",
        help="Name of the binary label column in the parquet file.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="splits_iid_*_clients*.json",
        help="Glob pattern for split files inside --splits-dir.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("split_scaling_summary.csv"),
        help="Path of the summary CSV output.",
    )
    parser.add_argument(
        "--output-xlsx",
        type=Path,
        default=Path("split_scaling_summary.xlsx"),
        help="Path of the summary Excel output.",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("split_client_details"),
        help="Directory for per-client detail CSV files.",
    )
    parser.add_argument(
        "--include-adjusted",
        action="store_true",
        help="Also include files containing 'adjusted' in the filename.",
    )
    return parser.parse_args()


def extract_num_clients_from_filename(path: Path) -> Optional[int]:
    """Extract client count from names like splits_iid_16384_clients.json."""
    match = re.search(r"(\d+)_clients", path.name)
    return int(match.group(1)) if match else None


def load_labels(data_path: Path, label_col: str) -> np.ndarray:
    """Load the binary label column from parquet."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {data_path}")

    try:
        df_label = pd.read_parquet(data_path, columns=[label_col])
    except Exception as first_error:
        # Fallback: read full dataframe to show available columns in the error message.
        try:
            df_full = pd.read_parquet(data_path)
        except Exception as second_error:
            raise RuntimeError(
                "Could not read parquet file. Install a parquet engine first, e.g.:\n"
                "  pip install pyarrow\n\n"
                f"Original error: {second_error}"
            ) from second_error

        if label_col not in df_full.columns:
            for candidate in DEFAULT_LABEL_CANDIDATES:
                if candidate in df_full.columns:
                    print(
                        f"⚠️ Label column '{label_col}' not found. "
                        f"Using detected column '{candidate}' instead."
                    )
                    label_col = candidate
                    break
            else:
                raise ValueError(
                    f"Label column '{label_col}' not found. Available columns are:\n"
                    f"{list(df_full.columns)}"
                ) from first_error

        df_label = df_full[[label_col]]

    labels = df_label[label_col].to_numpy()

    unique_values = set(pd.Series(labels).dropna().unique().tolist())
    if not unique_values.issubset({0, 1, 0.0, 1.0, False, True}):
        print(
            f"⚠️ Warning: Label column '{label_col}' is not strictly binary. "
            f"Observed values: {sorted(unique_values)[:10]}"
        )

    return labels.astype(int)


def load_split(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_sorted_client_items(client_map: Dict[str, List[int]]) -> List[tuple[str, List[int]]]:
    return sorted(client_map.items(), key=lambda item: int(item[0]))


def safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float | int]:
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0,
            f"{prefix}_max": 0,
            f"{prefix}_median": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_min": int(np.min(values)),
        f"{prefix}_max": int(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def analyze_client_map(
    client_map: Dict[str, List[int]], labels: np.ndarray
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Compute aggregate stats and per-client stats for one split's train map."""
    rows = []

    for cid, indices in as_sorted_client_items(client_map):
        idx = np.asarray(indices, dtype=int)
        n_samples = int(len(idx))

        if n_samples > 0:
            if idx.min() < 0 or idx.max() >= len(labels):
                raise IndexError(
                    f"Client {cid} contains index outside dataset range: "
                    f"min={idx.min()}, max={idx.max()}, dataset_size={len(labels)}"
                )
            n_pos = int(labels[idx].sum())
        else:
            n_pos = 0

        n_neg = n_samples - n_pos
        pos_rate = n_pos / n_samples if n_samples > 0 else 0.0

        rows.append(
            {
                "client_id": int(cid),
                "num_samples": n_samples,
                "num_positive": n_pos,
                "num_negative": n_neg,
                "positive_rate": pos_rate,
                "has_positive": n_pos > 0,
            }
        )

    detail_df = pd.DataFrame(rows)

    if detail_df.empty:
        aggregate = {
            "num_clients": 0,
            "total_samples": 0,
            "total_positive_samples": 0,
            "total_negative_samples": 0,
            "global_positive_rate": 0.0,
            "clients_without_positive": 0,
            "clients_without_positive_pct": 0.0,
            "clients_with_positive": 0,
            "clients_with_positive_pct": 0.0,
        }
        aggregate.update(safe_stats(np.array([]), "samples_per_client"))
        aggregate.update(safe_stats(np.array([]), "positive_samples_per_client"))
        return aggregate, detail_df

    sample_counts = detail_df["num_samples"].to_numpy()
    positive_counts = detail_df["num_positive"].to_numpy()

    total_samples = int(sample_counts.sum())
    total_positive = int(positive_counts.sum())
    total_negative = total_samples - total_positive
    num_clients = int(len(detail_df))
    clients_without_positive = int((positive_counts == 0).sum())
    clients_with_positive = num_clients - clients_without_positive

    aggregate = {
        "num_clients": num_clients,
        "total_samples": total_samples,
        "total_positive_samples": total_positive,
        "total_negative_samples": total_negative,
        "global_positive_rate": total_positive / total_samples if total_samples else 0.0,
        "clients_without_positive": clients_without_positive,
        "clients_without_positive_pct": clients_without_positive / num_clients if num_clients else 0.0,
        "clients_with_positive": clients_with_positive,
        "clients_with_positive_pct": clients_with_positive / num_clients if num_clients else 0.0,
    }
    aggregate.update(safe_stats(sample_counts, "samples_per_client"))
    aggregate.update(safe_stats(positive_counts, "positive_samples_per_client"))

    return aggregate, detail_df


def format_for_ba(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table with the most relevant BA columns."""
    cols = [
        "split_file",
        "number_of_clients",
        "total_train_samples",
        "mean_local_sample_size_per_client",
        "min_local_sample_size_per_client",
        "max_local_sample_size_per_client",
        "mean_positive_samples_per_client",
        "clients_without_positive_samples",
        "clients_without_positive_samples_pct",
        "global_positive_rate",
    ]
    return summary_df[cols]


def main() -> None:
    args = parse_args()

    labels = load_labels(args.data_path, args.label_col)

    split_files = sorted(
        args.splits_dir.glob(args.pattern),
        key=lambda p: (extract_num_clients_from_filename(p) or 10**12, p.name),
    )

    if not args.include_adjusted:
        split_files = [p for p in split_files if "adjusted" not in p.name]

    if not split_files:
        raise FileNotFoundError(
            f"No split files found in {args.splits_dir} with pattern '{args.pattern}'."
        )

    args.details_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("📊 Analyzing split files:")
    for split_path in split_files:
        data = load_split(split_path)
        if "train" not in data:
            print(f"⚠️ Skipping {split_path.name}: no 'train' key found.")
            continue

        train_map = data["train"]
        stats, detail_df = analyze_client_map(train_map, labels)

        number_from_filename = extract_num_clients_from_filename(split_path)
        number_of_clients = stats["num_clients"]

        detail_out = args.details_dir / f"{split_path.stem}_client_stats.csv"
        detail_df.to_csv(detail_out, index=False)

        row = {
            "split_file": split_path.name,
            "number_of_clients": number_of_clients,
            "number_of_clients_from_filename": number_from_filename,
            "total_train_samples": stats["total_samples"],
            "total_positive_train_samples": stats["total_positive_samples"],
            "total_negative_train_samples": stats["total_negative_samples"],
            "global_positive_rate": stats["global_positive_rate"],
            "mean_local_sample_size_per_client": stats["samples_per_client_mean"],
            "std_local_sample_size_per_client": stats["samples_per_client_std"],
            "min_local_sample_size_per_client": stats["samples_per_client_min"],
            "max_local_sample_size_per_client": stats["samples_per_client_max"],
            "median_local_sample_size_per_client": stats["samples_per_client_median"],
            "mean_positive_samples_per_client": stats[
                "positive_samples_per_client_mean"
            ],
            "std_positive_samples_per_client": stats[
                "positive_samples_per_client_std"
            ],
            "min_positive_samples_per_client": stats[
                "positive_samples_per_client_min"
            ],
            "max_positive_samples_per_client": stats[
                "positive_samples_per_client_max"
            ],
            "median_positive_samples_per_client": stats[
                "positive_samples_per_client_median"
            ],
            "clients_without_positive_samples": stats["clients_without_positive"],
            "clients_without_positive_samples_pct": stats[
                "clients_without_positive_pct"
            ],
            "clients_with_positive_samples": stats["clients_with_positive"],
            "clients_with_positive_samples_pct": stats["clients_with_positive_pct"],
            "client_detail_csv": str(detail_out),
        }
        summary_rows.append(row)

        print(
            f"   {split_path.name}: "
            f"clients={number_of_clients}, "
            f"mean n={row['mean_local_sample_size_per_client']:.2f}, "
            f"min-max n={row['min_local_sample_size_per_client']}-"
            f"{row['max_local_sample_size_per_client']}, "
            f"mean positives={row['mean_positive_samples_per_client']:.2f}, "
            f"zero-positive clients={row['clients_without_positive_samples']} "
            f"({row['clients_without_positive_samples_pct']:.2%})"
        )

    summary_df = pd.DataFrame(summary_rows)

    # Sort by number of clients for a clean scaling table.
    if not summary_df.empty:
        summary_df = summary_df.sort_values("number_of_clients").reset_index(drop=True)

    summary_df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Saved full summary CSV: {args.output_csv}")

    ba_df = format_for_ba(summary_df)
    ba_csv = args.output_csv.with_name(args.output_csv.stem + "_ba_table.csv")
    ba_df.to_csv(ba_csv, index=False)
    print(f"💾 Saved compact BA table CSV: {ba_csv}")

    try:
        with pd.ExcelWriter(args.output_xlsx) as writer:
            summary_df.to_excel(writer, sheet_name="full_summary", index=False)
            ba_df.to_excel(writer, sheet_name="ba_table", index=False)
        print(f"💾 Saved Excel file: {args.output_xlsx}")
    except Exception as exc:
        print(
            "⚠️ Could not write Excel file. CSV outputs were created. "
            "Install openpyxl if you want Excel output:\n"
            "   pip install openpyxl\n"
            f"Reason: {exc}"
        )

    print("\n📌 Most important BA columns:")
    if not ba_df.empty:
        print(ba_df.to_string(index=False))


if __name__ == "__main__":
    main()
