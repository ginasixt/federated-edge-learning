#!/usr/bin/env python3
"""
Final test-set evaluation for saved Flower/PyTorch checkpoints.

Important:
    Use this only after the model checkpoint and decision threshold have been
    selected on the validation set. The test set should not be used for
    checkpoint or threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Same architecture as in client_app.py."""

    def __init__(self, in_dim: int, hidden_dims: list[int] | None = None, out_dim: int = 2):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one selected checkpoint once on the reserved test set."
    )
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data", required=True, help="Path to normalized parquet file")
    parser.add_argument("--stats", required=True, help="Path to norm_stats.json")
    parser.add_argument(
        "--test-rows",
        required=True,
        help=(
            "Path to test row IDs. Supported formats: split JSON with a test key, "
            "JSON list/dict, or CSV with a __row_id__/row_id column."
        ),
    )
    parser.add_argument(
        "--test-key",
        default="test",
        help=(
            "JSON key containing test row IDs. If omitted, the script also tries "
            "common keys such as test_idx."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold selected on validation data",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="test_eval.json", help="Output JSON path")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device",
    )
    return parser.parse_args()


def flatten_row_ids(value) -> list[int]:
    """Flatten row IDs from a list or a client->list mapping."""
    if isinstance(value, list):
        return [int(v) for v in value]

    if isinstance(value, dict):
        row_ids: list[int] = []
        for client_rows in value.values():
            if isinstance(client_rows, list):
                row_ids.extend(int(v) for v in client_rows)
        return row_ids

    raise ValueError(f"Unsupported row-id structure: {type(value)}")


def load_test_row_ids(path: str, test_key: str) -> list[int]:
    row_path = Path(path)
    suffix = row_path.suffix.lower()

    if suffix == ".json":
        with row_path.open("r") as f:
            data = json.load(f)

        if isinstance(data, dict) and test_key in data:
            return flatten_row_ids(data[test_key])

        if isinstance(data, dict):
            for fallback_key in ("test_idx", "test_indices", "test_rows", "test"):
                if fallback_key in data:
                    return flatten_row_ids(data[fallback_key])

        return flatten_row_ids(data)

    if suffix == ".csv":
        df = pd.read_csv(row_path)
        for col in ("__row_id__", "row_id", "rowid", "id"):
            if col in df.columns:
                return [int(v) for v in df[col].dropna().tolist()]
        raise ValueError(
            f"CSV {path} must contain one of these columns: __row_id__, row_id, rowid, id"
        )

    raise ValueError("Unsupported --test-rows format. Use JSON or CSV.")


def load_test_data(
    parquet_path: str,
    stats_path: str,
    test_row_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list[str]]:
    with open(stats_path, "r") as f:
        stats = json.load(f)

    row_ids = list(dict.fromkeys(int(rid) for rid in test_row_ids))
    if not row_ids:
        raise ValueError("No test row IDs found.")

    df = pd.read_parquet(parquet_path, filters=[("__row_id__", "in", row_ids)])
    if df.empty:
        raise ValueError("No rows loaded from parquet. Check --data and --test-rows.")

    target_col = stats["target"]
    feature_cols = [c for c in df.columns if c not in {target_col, "__row_id__"}]

    row_id_to_idx = {int(row_id): idx for idx, row_id in enumerate(df["__row_id__"])}
    missing = [rid for rid in row_ids if rid not in row_id_to_idx]
    if missing:
        raise ValueError(f"{len(missing)} test row IDs were not found in the parquet file.")

    ordered_idx = [row_id_to_idx[rid] for rid in row_ids]
    df = df.iloc[ordered_idx]

    y = df[target_col].astype(int).values
    y = (y >= 1).astype("int64")
    X = df[feature_cols].values.astype("float32")

    class_weights = torch.tensor(
        [stats["neg_weight"], stats["pos_weight"]],
        dtype=torch.float32,
    )
    return X, y, class_weights, feature_cols


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, probs))
    except Exception:
        return None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    probs_all: list[torch.Tensor] = []
    y_all: list[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            probs_all.append(probs.cpu())
            y_all.append(yb.cpu())

    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())

    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    youden = recall + specificity - 1.0
    prevalence = safe_div(tp + fn, tp + fp + tn + fn)
    alerts_per_1000 = safe_div(tp + fp, tp + fp + tn + fn) * 1000.0

    return {
        "n_samples": int(n_samples),
        "threshold": float(threshold),
        "loss": float(total_loss / max(1, n_samples)),
        "auc": compute_auc(y, probs),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "youden": youden,
        "prevalence": prevalence,
        "alerts_per_1000": alerts_per_1000,
    }


def load_state_dict(path: str, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state_dict or contain 'model_state_dict'.")

    return checkpoint


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    test_row_ids = load_test_row_ids(args.test_rows, args.test_key)
    X_test, y_test, class_weights, feature_cols = load_test_data(
        args.data,
        args.stats,
        test_row_ids,
    )

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = MLP(in_dim=X_test.shape[1]).to(device)
    state_dict = load_state_dict(args.model, device)
    model.load_state_dict(state_dict, strict=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    metrics = evaluate(model, test_loader, criterion, args.threshold, device)

    output = {
        "model_checkpoint": str(args.model),
        "data": str(args.data),
        "stats": str(args.stats),
        "test_rows": str(args.test_rows),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("\nFinal test evaluation")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Test rows:   {len(test_row_ids)}")
    print(f"Threshold:   {args.threshold:.4f}")
    print(f"Loss:        {metrics['loss']:.6f}")
    print(f"AUC:         {metrics['auc'] if metrics['auc'] is not None else 'n/a'}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print(f"Bal. Acc.:   {metrics['balanced_accuracy']:.4f}")
    print(f"Counts:      TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"\nSaved:       {output_path}")


if __name__ == "__main__":
    main()
