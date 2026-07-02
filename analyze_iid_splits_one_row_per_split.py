#!/usr/bin/env python3
"""
Analyze IID scaling split files for a Federated Learning scalability study.

Output:
- One summary row per split file
- Optional client-level details for debugging

For each split, the script computes:
- Number of clients
- Total train samples
- Mean local sample size per client
- Min-max local sample size per client
- Mean positive samples per client
- Number and percentage of clients without positive samples
- Global positive rate in the training split

Expected split JSON structure:
{
  "train": {
    "0": [idx1, idx2, ...],
    "1": [idx3, idx4, ...]
  },
  "val": {...}
}

The indices are used to look up labels in the parquet file.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def infer_num_clients_from_filename(path: Path) -> int | None:
    """Try to infer client count from filenames like splits_iid_16384_clients.json."""
    match = re.search(r"(\d+)_clients", path.name)
    if match:
        return int(match.group(1))
    return None


def load_labels(data_path: Path, label_col: str) -> np.ndarray:
    """Load label column from parquet file."""
    df = pd.read_parquet(data_path)

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' was not found in {data_path}.\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    labels = df[label_col].to_numpy()
    return labels


def analyze_split(split_path: Path, labels: np.ndarray, split_name: str = "train") -> dict:
    """Analyze one split file and return one aggregated summary row."""
    with open(split_path, "r") as f:
        data = json.load(f)

    if split_name not in data:
        raise KeyError(
            f"Split file {split_path} does not contain key '{split_name}'. "
            f"Available keys: {list(data.keys())}"
        )

    client_map = data[split_name]

    if not isinstance(client_map, dict):
        raise TypeError(
            f"Expected data['{split_name}'] to be a dictionary mapping client IDs to indices."
        )

    client_ids = sorted(client_map.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    local_sample_sizes = []
    positive_samples_per_client = []
    negative_samples_per_client = []

    for cid in client_ids:
        indices = np.asarray(client_map[cid], dtype=int)

        # Some clients could theoretically be empty, so handle that safely
        local_n = len(indices)

        if local_n == 0:
            positives = 0
        else:
            if indices.max() >= len(labels):
                raise IndexError(
                    f"Split {split_path} contains index {indices.max()}, "
                    f"but labels only have length {len(labels)}."
                )
            positives = int(labels[indices].sum())

        negatives = int(local_n - positives)

        local_sample_sizes.append(local_n)
        positive_samples_per_client.append(positives)
        negative_samples_per_client.append(negatives)

    local_sample_sizes = np.asarray(local_sample_sizes, dtype=float)
    positive_samples_per_client = np.asarray(positive_samples_per_client, dtype=float)
    negative_samples_per_client = np.asarray(negative_samples_per_client, dtype=float)

    number_of_clients = len(client_ids)
    total_samples = int(local_sample_sizes.sum())
    total_positive_samples = int(positive_samples_per_client.sum())
    total_negative_samples = int(negative_samples_per_client.sum())

    clients_without_positive = int((positive_samples_per_client == 0).sum())

    if total_samples > 0:
        global_positive_rate = total_positive_samples / total_samples
    else:
        global_positive_rate = np.nan

    filename_num_clients = infer_num_clients_from_filename(split_path)

    row = {
        # Identification
        "split_file": split_path.name,
        "split": split_name,
        "number_of_clients": number_of_clients,
        "number_of_clients_from_filename": filename_num_clients,

        # Local sample size distribution
        "total_samples": total_samples,
        "mean_local_sample_size_per_client": local_sample_sizes.mean() if number_of_clients else np.nan,
        "std_local_sample_size_per_client": local_sample_sizes.std(ddof=0) if number_of_clients else np.nan,
        "min_local_sample_size_per_client": int(local_sample_sizes.min()) if number_of_clients else np.nan,
        "max_local_sample_size_per_client": int(local_sample_sizes.max()) if number_of_clients else np.nan,
        "min_max_local_sample_size_per_client": (
            f"{int(local_sample_sizes.min())}-{int(local_sample_sizes.max())}"
            if number_of_clients else ""
        ),

        # Positive-class distribution
        "total_positive_samples": total_positive_samples,
        "total_negative_samples": total_negative_samples,
        "global_positive_rate": global_positive_rate,
        "mean_positive_samples_per_client": positive_samples_per_client.mean() if number_of_clients else np.nan,
        "std_positive_samples_per_client": positive_samples_per_client.std(ddof=0) if number_of_clients else np.nan,
        "min_positive_samples_per_client": int(positive_samples_per_client.min()) if number_of_clients else np.nan,
        "max_positive_samples_per_client": int(positive_samples_per_client.max()) if number_of_clients else np.nan,

        # Clients without positive samples
        "clients_without_positive_samples": clients_without_positive,
        "clients_without_positive_samples_pct": (
            clients_without_positive / number_of_clients * 100 if number_of_clients else np.nan
        ),
    }

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze IID split JSON files and create one summary row per split."
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
        help="Path to parquet file containing the label column.",
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
        default="splits_iid_*_clients.json",
        help="Glob pattern for split files.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Which split inside the JSON to analyze. Usually 'train'.",
    )
    parser.add_argument(
        "--include-adjusted",
        action="store_true",
        help="Include files with 'adjusted' in the filename.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("split_scaling_summary_one_row_per_split.csv"),
        help="Output CSV path for the summary table.",
    )
    parser.add_argument(
        "--excel-output",
        type=Path,
        default=Path("split_scaling_summary_one_row_per_split.xlsx"),
        help="Output Excel path for the summary table.",
    )

    args = parser.parse_args()

    if not args.splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {args.splits_dir}")

    if not args.data_path.exists():
        raise FileNotFoundError(f"Parquet data file not found: {args.data_path}")

    print(f"Loading labels from: {args.data_path}")
    labels = load_labels(args.data_path, args.label_col)

    split_files = sorted(
        args.splits_dir.glob(args.pattern),
        key=lambda p: infer_num_clients_from_filename(p) or 10**18,
    )

    if not args.include_adjusted:
        split_files = [p for p in split_files if "adjusted" not in p.name]

    if not split_files:
        raise FileNotFoundError(
            f"No split files found in {args.splits_dir} with pattern '{args.pattern}'."
        )

    print(f"Found {len(split_files)} split files.")

    rows = []
    for split_path in split_files:
        print(f"Analyzing: {split_path.name}")
        row = analyze_split(split_path, labels, split_name=args.split_name)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sort by actual number of clients
    summary_df = summary_df.sort_values("number_of_clients").reset_index(drop=True)

    # Round columns for a cleaner BA table
    rounded_df = summary_df.copy()
    for col in [
        "mean_local_sample_size_per_client",
        "std_local_sample_size_per_client",
        "global_positive_rate",
        "mean_positive_samples_per_client",
        "std_positive_samples_per_client",
        "clients_without_positive_samples_pct",
    ]:
        if col in rounded_df.columns:
            rounded_df[col] = rounded_df[col].round(4)

    # More readable percentage column
    rounded_df["global_positive_rate_pct"] = (summary_df["global_positive_rate"] * 100).round(2)

    # BA-focused table: exactly the kind of table you described
    ba_columns = [
        "number_of_clients",
        "total_samples",
        "mean_local_sample_size_per_client",
        "min_max_local_sample_size_per_client",
        "mean_positive_samples_per_client",
        "clients_without_positive_samples",
        "clients_without_positive_samples_pct",
        "global_positive_rate_pct",
        "split_file",
    ]

    ba_table = rounded_df[ba_columns].rename(
        columns={
            "number_of_clients": "Number of Clients",
            "total_samples": "Total Training Samples",
            "mean_local_sample_size_per_client": "Mean Local Sample Size per Client",
            "min_max_local_sample_size_per_client": "Min-Max Local Sample Size per Client",
            "mean_positive_samples_per_client": "Mean Positive Samples per Client",
            "clients_without_positive_samples": "Clients without Positive Samples",
            "clients_without_positive_samples_pct": "Clients without Positive Samples (%)",
            "global_positive_rate_pct": "Global Positive Rate (%)",
            "split_file": "Split File",
        }
    )

    # Save full and BA-focused versions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rounded_df.to_csv(args.output, index=False)

    ba_output = args.output.with_name(args.output.stem + "_ba_table.csv")
    ba_table.to_csv(ba_output, index=False)

    try:
        with pd.ExcelWriter(args.excel_output) as writer:
            ba_table.to_excel(writer, sheet_name="BA_Table", index=False)
            rounded_df.to_excel(writer, sheet_name="Full_Summary", index=False)
    except ImportError:
        print("Excel output skipped. Install openpyxl if you want .xlsx output: pip install openpyxl")

    print()
    print("Done.")
    print(f"Full summary saved to: {args.output}")
    print(f"BA table saved to:    {ba_output}")
    print(f"Excel saved to:       {args.excel_output}")
    print()
    print("Preview:")
    print(ba_table.to_string(index=False))


if __name__ == "__main__":
    main()
