#!/usr/bin/env python3
"""
Threshold-dependent SCAFFOLD evaluation for validation-AP-selected checkpoints.

For every scaling point and run, this script:

1. Loads the checkpoint already selected by highest validation Average Precision
   from:
       result/splits_iid_scaling/
         splits_iid_<N>_clients.json/SCAFFOLD/bestPRROC/run_<r>/

2. Uses the complete centralized validation set to select:
   a) the threshold maximizing validation MCC;
   b) the threshold maximizing validation specificity subject to a prespecified
      minimum validation recall.

3. Applies both thresholds unchanged to the centralized test set.

4. Writes one compact file required by plot-thr-dependent-scaffold.py:
       result/splits_iid_scaling/final_threshold_analysis/SCAFFOLD/
         all_threshold_results.csv

No checkpoint or threshold is selected on the test set.

Important SCAFFOLD split handling
---------------------------------
Validation IDs are collected from ALL lists under split_data["val"]. This works
both when the complete validation set is stored under val["0"] and when an older
split distributes validation IDs across several client keys. For every scaling
point, the union is required to match norm_stats.json["val_idx"] exactly.

Example
-------
python3 evaluate-thr-dependent-scaffold.py --min-recall 0.80

The minimum recall must be chosen before inspecting the threshold-dependent test
results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from federated_learning.client_app import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_RESULT_ROOT = Path("result/splits_iid_scaling")
DEFAULT_SPLIT_ROOT = Path("splits_iid_scaling")
DEFAULT_DATA_PARQUET = Path("data/diabetes_normalized.parquet")
DEFAULT_NORM_STATS = Path("data/norm_stats.json")
DEFAULT_STRATEGY = "SCAFFOLD"
DEFAULT_OUTPUT = (
    DEFAULT_RESULT_ROOT
    / "final_threshold_analysis"
    / DEFAULT_STRATEGY
    / "all_threshold_results.csv"
)

DEFAULT_SCALING_POINTS: Tuple[int, ...] = (
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)

CHECKPOINT_PATTERN = re.compile(
    r"model_round_(\d+).*\.pt\s*$",
    re.IGNORECASE,
)

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "strategy",
    "scaling_point",
    "run",
    "checkpoint_selection",
    "selected_round",
    "selected_checkpoint",
    "threshold_regime",
    "selected_threshold",
    "minimum_validation_recall_requirement",
    "validation_candidate_threshold_count",
    "validation_eligible_threshold_count",
    "validation_roc_auc",
    "validation_average_precision",
    "test_roc_auc",
    "test_average_precision",
    "validation_tp",
    "validation_fp",
    "validation_tn",
    "validation_fn",
    "validation_mcc",
    "validation_recall",
    "validation_specificity",
    "validation_precision",
    "validation_f1",
    "validation_balanced_accuracy",
    "test_tp",
    "test_fp",
    "test_tn",
    "test_fn",
    "test_mcc",
    "test_recall",
    "test_specificity",
    "test_precision",
    "test_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_false_positive_rate",
    "test_false_negative_rate",
    "test_predicted_positive_rate",
    "test_false_positives_per_1000_negatives",
    "validation_n_samples",
    "test_n_samples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select exact validation thresholds for SCAFFOLD bestPRROC "
            "checkpoints and evaluate them unchanged on the test set."
        )
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        required=True,
        help=(
            "Prespecified minimum validation recall for the constrained "
            "operating point, for example 0.80."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
    )
    parser.add_argument(
        "--data-parquet",
        type=Path,
        default=DEFAULT_DATA_PARQUET,
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=DEFAULT_NORM_STATS,
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Single CSV written for the plotting script.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--scaling-points",
        type=int,
        nargs="+",
        default=list(DEFAULT_SCALING_POINTS),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--in-dim",
        type=int,
        default=21,
    )
    return parser.parse_args()


def finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_global_indices(metadata: Mapping[str, Any]) -> Dict[str, List[int]]:
    names = ("train_idx", "val_idx", "test_idx")
    missing = [name for name in names if name not in metadata]
    if missing:
        raise KeyError(
            "norm_stats.json is missing: " + ", ".join(missing)
        )

    splits: Dict[str, List[int]] = {}
    for name in names:
        values = [int(value) for value in metadata[name]]
        if not values:
            raise ValueError(f"norm_stats.json contains an empty {name}.")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs found in norm_stats.json:{name}.")
        splits[name] = values

    train = set(splits["train_idx"])
    val = set(splits["val_idx"])
    test = set(splits["test_idx"])
    overlaps = {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }
    if any(overlaps.values()):
        raise ValueError(f"Global train/validation/test overlap: {overlaps}")

    return splits


def client_key_sort(key: Any) -> Tuple[int, Any]:
    text = str(key)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def validation_ids_from_scaffold_split(
    split_root: Path,
    n_clients: int,
    expected_global_ids: Sequence[int],
) -> List[int]:
    """
    Flatten every list under split_data["val"] and verify exact global membership.

    This deliberately does not assume that only val["0"] is populated.
    """
    split_path = split_root / f"splits_iid_{n_clients}_clients.json"
    payload = load_json(split_path)
    val_mapping = payload.get("val")
    if not isinstance(val_mapping, dict) or not val_mapping:
        raise KeyError(f"No non-empty 'val' mapping in {split_path}")

    flattened: List[int] = []
    non_empty_clients = 0

    for client_id in sorted(val_mapping, key=client_key_sort):
        values = val_mapping[client_id]
        if not isinstance(values, list):
            raise TypeError(
                f"split_data['val'][{client_id!r}] is not a list in {split_path}"
            )
        if values:
            non_empty_clients += 1
            flattened.extend(int(value) for value in values)

    if not flattened:
        raise ValueError(f"No validation row IDs found in {split_path}")
    if len(flattened) != len(set(flattened)):
        raise ValueError(
            f"Validation IDs are duplicated across client entries in {split_path}"
        )

    observed = set(flattened)
    expected = set(int(value) for value in expected_global_ids)
    missing = expected - observed
    unexpected = observed - expected

    if missing or unexpected:
        raise ValueError(
            f"Validation membership mismatch for {n_clients} clients: "
            f"missing={len(missing)}, unexpected={len(unexpected)}. "
            "The threshold must be selected on the complete global validation set."
        )

    print(
        f"   ✓ Validation split verified: {len(flattened)} rows "
        f"across {non_empty_clients} non-empty val client entr"
        f"{'y' if non_empty_clients == 1 else 'ies'}"
    )
    return flattened


def load_normalized_split(
    parquet_path: Path,
    metadata: Mapping[str, Any],
    row_ids: Sequence[int],
    split_name: str,
    batch_size: int,
    expected_features: int,
) -> Tuple[DataLoader, Dict[str, Any]]:
    requested = [int(value) for value in row_ids]
    if not requested:
        raise ValueError(f"No row IDs supplied for {split_name}.")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate row IDs supplied for {split_name}.")

    dataframe = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", requested)],
    )

    target = str(metadata["target"])
    required_columns = {"__row_id__", target}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(
            f"Missing columns for {split_name}: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe["__row_id__"].duplicated().any():
        raise ValueError(
            f"Duplicate __row_id__ values loaded for {split_name}."
        )

    loaded_ids = set(dataframe["__row_id__"].astype(int))
    requested_set = set(requested)
    missing = requested_set - loaded_ids
    unexpected = loaded_ids - requested_set
    if missing or unexpected:
        raise ValueError(
            f"Loaded {split_name} rows differ from requested membership: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )

    order = {row_id: position for position, row_id in enumerate(requested)}
    dataframe = dataframe.assign(
        __requested_order__=dataframe["__row_id__"].astype(int).map(order)
    ).sort_values("__requested_order__", kind="stable")

    labels = dataframe[target].astype(int).to_numpy()
    labels = (labels >= 1).astype(np.int64, copy=False)

    features = dataframe.drop(
        columns=[target, "__row_id__", "__requested_order__"]
    ).to_numpy(dtype=np.float32)

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {split_name}, "
            f"loaded {features.shape[1]}."
        )
    if np.unique(labels).size != 2:
        raise ValueError(f"{split_name} does not contain both classes.")

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    info = {
        "n_samples": int(len(labels)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive / len(labels)),
    }
    return loader, info


def scaling_strategy_dir(
    result_root: Path,
    strategy: str,
    n_clients: int,
) -> Path:
    return (
        result_root
        / f"splits_iid_{n_clients}_clients.json"
        / strategy
    )


def parse_round_from_checkpoint(path: Path) -> Optional[int]:
    match = CHECKPOINT_PATTERN.search(path.name.strip())
    return int(match.group(1)) if match else None


def discover_best_pr_checkpoint(
    result_root: Path,
    strategy: str,
    n_clients: int,
    run_tag: int,
    expected_validation_samples: int,
) -> Dict[str, Any]:
    selection_dir = (
        scaling_strategy_dir(result_root, strategy, n_clients)
        / "bestPRROC"
        / f"run_{run_tag}"
    )
    if not selection_dir.is_dir():
        raise FileNotFoundError(
            f"Missing AP-selected checkpoint directory: {selection_dir}"
        )

    best_info = load_optional_json(selection_dir / "best_info.json")

    expected_round_raw = best_info.get("best_round")
    try:
        expected_round = (
            int(expected_round_raw)
            if expected_round_raw is not None
            else None
        )
    except (TypeError, ValueError):
        expected_round = None

    candidates = sorted(
        path
        for path in selection_dir.iterdir()
        if path.is_file() and CHECKPOINT_PATTERN.search(path.name.strip())
    )

    if expected_round is not None:
        matching = [
            path
            for path in candidates
            if parse_round_from_checkpoint(path) == expected_round
        ]
        if matching:
            candidates = matching

    if not candidates:
        for key in ("copied_to", "source_checkpoint"):
            raw_path = best_info.get(key)
            if raw_path:
                candidate = Path(str(raw_path))
                if candidate.is_file():
                    candidates = [candidate]
                    break

    if not candidates:
        raise FileNotFoundError(
            f"No selected checkpoint found in {selection_dir}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple bestPRROC checkpoints in {selection_dir}: "
            + ", ".join(path.name for path in candidates)
        )

    checkpoint = candidates[0]
    selected_round = expected_round or parse_round_from_checkpoint(checkpoint)
    if selected_round is None:
        raise ValueError(
            f"Cannot determine selected round from {checkpoint}"
        )

    # Safety check for the previous SCAFFOLD validation issue.
    recorded_ap = best_info.get("pr_auc")
    if recorded_ap is not None:
        try:
            recorded_ap = float(recorded_ap)
        except (TypeError, ValueError):
            recorded_ap = float("nan")
        if not np.isfinite(recorded_ap):
            raise ValueError(
                f"{selection_dir}/best_info.json contains no finite validation "
                "AP. The AP checkpoint must first be reselected on the complete "
                "validation set."
            )

    recorded_n = best_info.get("n_samples")
    if recorded_n is not None:
        try:
            recorded_n_int = int(recorded_n)
        except (TypeError, ValueError):
            recorded_n_int = -1
        if recorded_n_int != expected_validation_samples:
            raise ValueError(
                f"{selection_dir}/best_info.json reports {recorded_n_int} "
                f"validation samples, expected {expected_validation_samples}. "
                "The checkpoint appears not to have been selected on the "
                "complete validation set."
            )

    return {
        "checkpoint": checkpoint,
        "selected_round": int(selected_round),
        "best_info": best_info,
    }


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    if state and all(isinstance(key, str) for key in state):
        if all(key.startswith("module.") for key in state):
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
    return state


def load_model(checkpoint_path: Path, in_dim: int) -> torch.nn.Module:
    model = MLP(in_dim=in_dim).to(DEVICE)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for features, targets in loader:
        features = features.to(DEVICE)
        logits = model(features)
        positive_probabilities = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(
            positive_probabilities.detach().cpu().numpy()
        )
        labels.append(targets.numpy())

    if not probabilities:
        raise ValueError("DataLoader contains no samples.")

    return (
        np.concatenate(probabilities).astype(np.float64, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full_like(
        numerator,
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )
    return result


def exact_threshold_sweep(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Evaluate every distinct operating point induced by validation probabilities.

    Predictions use probability >= threshold. Samples with equal scores are
    added together, so no arbitrary numerical threshold grid is used.
    """
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)

    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be aligned 1D arrays.")
    if len(y) == 0:
        raise ValueError("Cannot optimize a threshold on an empty set.")
    if not np.all(np.isfinite(p)):
        raise ValueError("Probabilities contain NaN or infinity.")
    if np.unique(y).size != 2:
        raise ValueError("Threshold optimization requires both classes.")

    order = np.argsort(-p, kind="stable")
    sorted_probabilities = p[order]
    sorted_labels = y[order]

    group_ends = np.flatnonzero(
        np.r_[
            sorted_probabilities[1:] != sorted_probabilities[:-1],
            True,
        ]
    )
    thresholds = sorted_probabilities[group_ends]

    cumulative_tp = np.cumsum(
        sorted_labels == 1,
        dtype=np.int64,
    )
    cumulative_fp = np.cumsum(
        sorted_labels == 0,
        dtype=np.int64,
    )

    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    fn = positives - tp
    tn = negatives - fp

    tp_f = tp.astype(np.float64)
    fp_f = fp.astype(np.float64)
    tn_f = tn.astype(np.float64)
    fn_f = fn.astype(np.float64)

    recall = safe_divide(tp_f, tp_f + fn_f)
    specificity = safe_divide(tn_f, tn_f + fp_f)
    precision = safe_divide(tp_f, tp_f + fp_f)
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = safe_divide(
        2.0 * tp_f,
        2.0 * tp_f + fp_f + fn_f,
    )

    denominator = np.sqrt(
        (tp_f + fp_f)
        * (tp_f + fn_f)
        * (tn_f + fp_f)
        * (tn_f + fn_f)
    )
    mcc = np.zeros_like(tp_f, dtype=np.float64)
    np.divide(
        tp_f * tn_f - fp_f * fn_f,
        denominator,
        out=mcc,
        where=denominator != 0,
    )

    return {
        "threshold": thresholds,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
    }


def sweep_row(
    sweep: Mapping[str, np.ndarray],
    index: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, values in sweep.items():
        value = values[index]
        if key in {"tp", "fp", "tn", "fn"}:
            result[key] = int(value)
        else:
            number = float(value)
            result[key] = number if np.isfinite(number) else None
    return result


def select_mcc_threshold(
    sweep: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    mcc = np.asarray(sweep["mcc"], dtype=np.float64)
    thresholds = np.asarray(
        sweep["threshold"],
        dtype=np.float64,
    )

    maximum = float(np.nanmax(mcc))
    tied = np.flatnonzero(
        np.isclose(
            mcc,
            maximum,
            rtol=1e-10,
            atol=1e-12,
        )
    )
    selected_index = int(
        tied[np.argmax(thresholds[tied])]
    )

    result = sweep_row(sweep, selected_index)
    result["candidate_threshold_count"] = int(len(thresholds))
    return result


def select_recall_constrained_threshold(
    sweep: Mapping[str, np.ndarray],
    minimum_recall: float,
) -> Dict[str, Any]:
    recall = np.asarray(
        sweep["recall"],
        dtype=np.float64,
    )
    specificity = np.asarray(
        sweep["specificity"],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        sweep["threshold"],
        dtype=np.float64,
    )

    eligible = np.flatnonzero(
        recall >= minimum_recall - 1e-12
    )
    if eligible.size == 0:
        raise RuntimeError(
            f"No validation threshold satisfies recall >= {minimum_recall:.6f}."
        )

    maximum_specificity = float(
        np.nanmax(specificity[eligible])
    )
    tied = eligible[
        np.isclose(
            specificity[eligible],
            maximum_specificity,
            rtol=1e-10,
            atol=1e-12,
        )
    ]
    selected_index = int(
        tied[np.argmax(thresholds[tied])]
    )

    result = sweep_row(sweep, selected_index)
    result["minimum_recall"] = float(minimum_recall)
    result["candidate_threshold_count"] = int(len(thresholds))
    result["eligible_threshold_count"] = int(len(eligible))
    return result


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    predicted = (
        np.asarray(probabilities, dtype=np.float64)
        >= float(threshold)
    )

    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    tn = int(np.sum((~predicted) & (y == 0)))
    fn = int(np.sum((~predicted) & (y == 1)))

    def divide(numerator: float, denominator: float) -> Optional[float]:
        return (
            float(numerator / denominator)
            if denominator
            else None
        )

    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    accuracy = divide(
        tp + tn,
        tp + tn + fp + fn,
    )
    balanced_accuracy = (
        None
        if recall is None or specificity is None
        else float((recall + specificity) / 2.0)
    )
    f1 = divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    denominator = math.sqrt(
        (tp + fp)
        * (tp + fn)
        * (tn + fp)
        * (tn + fn)
    )
    mcc = (
        float((tp * tn - fp * fn) / denominator)
        if denominator
        else 0.0
    )

    false_positive_rate = (
        None
        if specificity is None
        else float(1.0 - specificity)
    )
    false_negative_rate = (
        None
        if recall is None
        else float(1.0 - recall)
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "predicted_positive_rate": divide(tp + fp, len(y)),
        "false_positives_per_1000_negatives": (
            None
            if false_positive_rate is None
            else 1000.0 * false_positive_rate
        ),
        "n_samples": int(len(y)),
    }


def flatten_result(
    *,
    strategy: str,
    n_clients: int,
    run_tag: int,
    selected_round: int,
    checkpoint: Path,
    regime: str,
    validation_selection: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    validation_roc_auc: float,
    validation_average_precision: float,
    test_roc_auc: float,
    test_average_precision: float,
    validation_n_samples: int,
    test_n_samples: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "strategy": strategy,
        "scaling_point": int(n_clients),
        "run": int(run_tag),
        "checkpoint_selection": "highest_validation_average_precision",
        "selected_round": int(selected_round),
        "selected_checkpoint": str(checkpoint),
        "threshold_regime": regime,
        "selected_threshold": validation_selection["threshold"],
        "minimum_validation_recall_requirement": (
            validation_selection.get("minimum_recall")
        ),
        "validation_candidate_threshold_count": (
            validation_selection.get("candidate_threshold_count")
        ),
        "validation_eligible_threshold_count": (
            validation_selection.get("eligible_threshold_count")
        ),
        "validation_roc_auc": validation_roc_auc,
        "validation_average_precision": validation_average_precision,
        "test_roc_auc": test_roc_auc,
        "test_average_precision": test_average_precision,
        "validation_n_samples": validation_n_samples,
        "test_n_samples": test_n_samples,
    }

    for key in (
        "tp",
        "fp",
        "tn",
        "fn",
        "mcc",
        "recall",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
    ):
        row[f"validation_{key}"] = validation_selection.get(key)

    for key in (
        "tp",
        "fp",
        "tn",
        "fn",
        "mcc",
        "recall",
        "specificity",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "predicted_positive_rate",
        "false_positives_per_1000_negatives",
    ):
        row[f"test_{key}"] = test_metrics.get(key)

    return row


def evaluate_run(
    *,
    args: argparse.Namespace,
    n_clients: int,
    run_tag: int,
    validation_loader: DataLoader,
    validation_info: Mapping[str, Any],
    test_loader: DataLoader,
    test_info: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    selected = discover_best_pr_checkpoint(
        result_root=args.result_root,
        strategy=args.strategy,
        n_clients=n_clients,
        run_tag=run_tag,
        expected_validation_samples=int(validation_info["n_samples"]),
    )

    checkpoint: Path = selected["checkpoint"]
    model = load_model(
        checkpoint,
        in_dim=args.in_dim,
    )

    validation_probabilities, validation_labels = predict_probabilities(
        model,
        validation_loader,
    )
    test_probabilities, test_labels = predict_probabilities(
        model,
        test_loader,
    )

    validation_roc_auc = float(
        roc_auc_score(
            validation_labels,
            validation_probabilities,
        )
    )
    validation_average_precision = float(
        average_precision_score(
            validation_labels,
            validation_probabilities,
        )
    )
    test_roc_auc = float(
        roc_auc_score(
            test_labels,
            test_probabilities,
        )
    )
    test_average_precision = float(
        average_precision_score(
            test_labels,
            test_probabilities,
        )
    )

    sweep = exact_threshold_sweep(
        validation_labels,
        validation_probabilities,
    )
    mcc_selection = select_mcc_threshold(sweep)
    recall_selection = select_recall_constrained_threshold(
        sweep,
        minimum_recall=args.min_recall,
    )

    mcc_test = metrics_at_threshold(
        test_labels,
        test_probabilities,
        float(mcc_selection["threshold"]),
    )
    recall_test = metrics_at_threshold(
        test_labels,
        test_probabilities,
        float(recall_selection["threshold"]),
    )

    print(
        f"      ✓ Run {run_tag}, round {selected['selected_round']}: "
        f"MCC threshold={mcc_selection['threshold']:.6f}, "
        f"test MCC={mcc_test['mcc']:.4f}; "
        f"recall threshold={recall_selection['threshold']:.6f}, "
        f"test recall={recall_test['recall']:.4f}, "
        f"test specificity={recall_test['specificity']:.4f}"
    )

    common = {
        "strategy": args.strategy,
        "n_clients": n_clients,
        "run_tag": run_tag,
        "selected_round": selected["selected_round"],
        "checkpoint": checkpoint,
        "validation_roc_auc": validation_roc_auc,
        "validation_average_precision": validation_average_precision,
        "test_roc_auc": test_roc_auc,
        "test_average_precision": test_average_precision,
        "validation_n_samples": int(validation_info["n_samples"]),
        "test_n_samples": int(test_info["n_samples"]),
    }

    return [
        flatten_result(
            **common,
            regime="mcc_optimal",
            validation_selection=mcc_selection,
            test_metrics=mcc_test,
        ),
        flatten_result(
            **common,
            regime="minimum_recall",
            validation_selection=recall_selection,
            test_metrics=recall_test,
        ),
    ]


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_COLUMNS),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row.get(column)
                    for column in OUTPUT_COLUMNS
                }
            )

    temporary.replace(path)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.min_recall <= 1.0:
        raise ValueError("--min-recall must be in the interval (0, 1].")
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if not args.scaling_points:
        raise ValueError("At least one scaling point is required.")

    print("=" * 84)
    print("THRESHOLD-DEPENDENT SCAFFOLD EVALUATION")
    print(f"Strategy       : {args.strategy}")
    print(f"Device         : {DEVICE}")
    print(f"Minimum recall : {args.min_recall:.4f} (selected on validation)")
    print(f"Scaling points : {len(args.scaling_points)}")
    print(f"Runs per point : {args.n_runs}")
    print(f"Output         : {args.output}")
    print("=" * 84)

    metadata = load_json(args.norm_stats)
    global_indices = validate_global_indices(metadata)

    # Verify every SCAFFOLD split before evaluating any model. This catches the
    # old 16k layout while still reconstructing the complete validation set.
    for n_clients in args.scaling_points:
        print(f"\n🔎 Verifying {n_clients} clients")
        validation_ids_from_scaffold_split(
            split_root=args.split_root,
            n_clients=n_clients,
            expected_global_ids=global_indices["val_idx"],
        )

    # Membership is identical across all scaling points, so load both
    # centralized datasets only once.
    validation_loader, validation_info = load_normalized_split(
        parquet_path=args.data_parquet,
        metadata=metadata,
        row_ids=global_indices["val_idx"],
        split_name="centralized validation",
        batch_size=args.batch_size,
        expected_features=args.in_dim,
    )
    test_loader, test_info = load_normalized_split(
        parquet_path=args.data_parquet,
        metadata=metadata,
        row_ids=global_indices["test_idx"],
        split_name="centralized test",
        batch_size=args.batch_size,
        expected_features=args.in_dim,
    )

    print(
        f"\n✓ Validation loaded: {validation_info['n_samples']} samples "
        f"({validation_info['n_positive']} positive, "
        f"{validation_info['n_negative']} negative)"
    )
    print(
        f"✓ Test loaded      : {test_info['n_samples']} samples "
        f"({test_info['n_positive']} positive, "
        f"{test_info['n_negative']} negative)"
    )

    rows: List[Dict[str, Any]] = []

    for n_clients in args.scaling_points:
        print("\n" + "-" * 84)
        print(f"{n_clients} clients")
        print("-" * 84)

        for run_tag in range(1, args.n_runs + 1):
            rows.extend(
                evaluate_run(
                    args=args,
                    n_clients=n_clients,
                    run_tag=run_tag,
                    validation_loader=validation_loader,
                    validation_info=validation_info,
                    test_loader=test_loader,
                    test_info=test_info,
                )
            )

    expected_rows = (
        len(args.scaling_points)
        * args.n_runs
        * 2
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} output rows, produced {len(rows)}."
        )

    rows.sort(
        key=lambda row: (
            int(row["scaling_point"]),
            int(row["run"]),
            str(row["threshold_regime"]),
        )
    )
    write_csv_atomic(args.output, rows)

    print("\n" + "=" * 84)
    print(f"✓ Finished: {len(rows)} rows written")
    print(f"✓ CSV: {args.output}")
    print()
    print("Next step:")
    print(
        "python3 plot-thr-dependent-scaffold.py "
        f"--input {args.output}"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
