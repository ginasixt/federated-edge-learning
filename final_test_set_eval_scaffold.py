#!/usr/bin/env python3
"""
Second Script!!!

Final Test-Set Evaluation for the FL Scaling Study
===================================================

This script evaluates ONLY the checkpoints that were previously selected on
THE VALIDATION SET STORED UNDER split_data["val"]["0"]:

    bestROC/run_<r>/    -> highest validation ROC-AUC
    bestPRROC/run_<r>/  -> highest validation Average Precision (PR-AUC)
    bestLoss/run_<r>/   -> lowest validation loss

No model, round, threshold, or hyperparameter is selected on the test set.
For SCAFFOLD, validation row IDs were read from the split file at
``split_data["val"]["0"]``; client ID ``"0"`` is a string key and is not
interpreted as an empty or missing client.
The test set is used exactly once for final reporting. Its row IDs are read
from ``norm_stats.json["test_idx"]`` and are therefore identical for every
scaling point.

Expected selected-checkpoint layout
-----------------------------------
result/splits_iid_scaling/
  splits_iid_<N>_clients.json/
    SCAFFOLD/
      bestROC/run_<r>/
      bestPRROC/run_<r>/
      bestLoss/run_<r>/

Output layout
-------------
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
  test_set_info.json
  all_test_results.csv
  all_test_aggregate.csv
  final_test_summary.json
  splits_iid_<N>_clients/
    test_results.csv
    test_aggregate.csv
    test_summary.json
    bestROC/run_<r>/
      test_metrics.json
      test_curves.json
    bestPRROC/run_<r>/
      test_metrics.json
      test_curves.json
    bestLoss/run_<r>/
      test_metrics.json
      test_curves.json

For every selected model, the script stores:
  * weighted Cross-Entropy test loss
  * ROC-AUC
  * Average Precision (the PR metric used during validation selection)
  * trapezoidal area under the stored PR curve
  * complete ROC and precision-recall curve points
  * test-set class counts and prevalence
  * source checkpoint and validation-selection metadata

The CSV files contain the scalar metrics needed for comparisons and plots.
The JSON files retain the complete curve data and provenance information.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from federated_learning.client_app import MLP


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# CONFIG
# =============================================================================
RESULT_ROOT = Path("result/splits_iid_scaling")
DATA_PARQUET = Path("data/diabetes_normalized.parquet")
NORM_STATS = Path("data/norm_stats.json")

STRATEGY = "SCAFFOLD"
VALIDATION_SELECTION_SET = "validation_client_0"
VALIDATION_INDEX_SOURCE = "split_file:val['0']"
N_RUNS = 5
IN_DIM = 21
BATCH_SIZE = 256

# These are the three validation-based checkpoint selection criteria produced
# by scaling_eval_fixed.py.
SELECTION_METRICS: Tuple[str, ...] = ("ROC", "PRROC", "Loss")

SCALING_POINTS: List[int] = [
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
    32768,
]

FINAL_OUTPUT_ROOT = RESULT_ROOT / "final_test_set_eval" / STRATEGY
# =============================================================================


CHECKPOINT_PATTERN = re.compile(
    r"model_round_(\d+).*\.pt\s*$",
    re.IGNORECASE,
)

RAW_RESULT_FIELDS: Tuple[str, ...] = (
    "strategy",
    "scaling_point",
    "selection_metric",
    "run",
    "selected_round",
    "test_roc_auc",
    "test_pr_auc",
    "test_pr_average_precision",
    "test_pr_auc_trapezoidal",
    "test_loss",
    "n_samples",
    "n_positive",
    "n_negative",
    "positive_prevalence",
    "validation_roc_auc",
    "validation_pr_auc",
    "validation_loss",
    "selected_checkpoint",
    "original_source_checkpoint",
    "test_metrics_file",
    "test_curves_file",
)

AGGREGATE_FIELDS: Tuple[str, ...] = (
    "strategy",
    "scaling_point",
    "selection_metric",
    "n_runs",
    "test_roc_auc_mean",
    "test_roc_auc_std",
    "test_pr_auc_mean",
    "test_pr_auc_std",
    "test_pr_auc_trapezoidal_mean",
    "test_pr_auc_trapezoidal_std",
    "test_loss_mean",
    "test_loss_std",
)


def scaling_strategy_dir(n_clients: int) -> Path:
    """Directory containing bestROC/bestPRROC/bestLoss for one scale."""
    return (
        RESULT_ROOT
        / f"splits_iid_{n_clients}_clients.json"
        / STRATEGY
    )


def scaling_output_dir(n_clients: int) -> Path:
    """Final test-result directory for one scale."""
    return FINAL_OUTPUT_ROOT / f"splits_iid_{n_clients}_clients"


def finite_or_none(value: Any) -> Optional[float]:
    """Return a finite Python float, otherwise JSON null."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def write_json(path: Path, payload: Any) -> None:
    """Write strict JSON; NaN and Infinity are never emitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    """Write a deterministic UTF-8 CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _require_unique_indices(name: str, values: Sequence[int]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"norm_stats.json enthält doppelte IDs in '{name}'.")


def validate_split_indices(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Check that train/validation/test membership is a clean partition."""
    split_names = ("train_idx", "val_idx", "test_idx")
    missing_keys = [name for name in split_names if name not in meta]
    if missing_keys:
        raise KeyError(
            "In norm_stats.json fehlen Split-Indizes: " + ", ".join(missing_keys)
        )

    splits: Dict[str, List[int]] = {
        name: [int(value) for value in meta[name]]
        for name in split_names
    }
    for name, values in splits.items():
        _require_unique_indices(name, values)

    train = set(splits["train_idx"])
    val = set(splits["val_idx"])
    test = set(splits["test_idx"])

    overlaps = {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }
    if any(overlaps.values()):
        raise ValueError(f"Train/Val/Test-Indizes überlappen: {overlaps}")

    return {
        "train_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
        "total_unique_rows": len(train | val | test),
        "overlaps": overlaps,
    }


def test_id_hash(test_row_ids: Sequence[int]) -> str:
    """Stable membership hash for documenting the exact test split."""
    canonical = ",".join(str(value) for value in sorted(test_row_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_centralized_test(
    parquet_path: Path,
    stats_path: Path,
) -> Tuple[DataLoader, torch.Tensor, Dict[str, Any]]:
    """
    Load the final centralized test set.

    The procedure mirrors task.load_centralized_val, but obtains the row IDs
    directly from norm_stats.json["test_idx"], verifies exact membership, and
    restores the requested row order before constructing tensors.

    The parquet data are already normalized; no second normalization is applied.
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Norm-Statistiken fehlen: {stats_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Normalisierte Parquet-Datei fehlt: {parquet_path}")

    meta = json.loads(stats_path.read_text(encoding="utf-8"))
    split_validation = validate_split_indices(meta)

    test_row_ids = [int(value) for value in meta["test_idx"]]
    requested_set = set(test_row_ids)

    dataframe = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", test_row_ids)],
    )

    required_columns = {"__row_id__", meta["target"]}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(
            "Parquet-Datei enthält erforderliche Spalten nicht: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe["__row_id__"].duplicated().any():
        duplicated = dataframe.loc[
            dataframe["__row_id__"].duplicated(), "__row_id__"
        ].astype(int).tolist()
        raise ValueError(
            "Parquet-Datei enthält doppelte __row_id__-Werte im Testset: "
            f"{duplicated[:10]}"
        )

    loaded_set = set(dataframe["__row_id__"].astype(int).tolist())
    missing_ids = requested_set - loaded_set
    unexpected_ids = loaded_set - requested_set
    if missing_ids or unexpected_ids:
        raise ValueError(
            "Geladene Testzeilen stimmen nicht mit norm_stats['test_idx'] überein. "
            f"Fehlend={len(missing_ids)}, unerwartet={len(unexpected_ids)}"
        )

    # The order has no effect on ROC/PR/loss, but restoring test_idx order makes
    # the stored predictions and curves exactly reproducible.
    order_by_id = {row_id: position for position, row_id in enumerate(test_row_ids)}
    dataframe = dataframe.assign(
        __test_order__=dataframe["__row_id__"].astype(int).map(order_by_id)
    ).sort_values("__test_order__", kind="stable")

    target_column = str(meta["target"])
    y_test = dataframe[target_column].astype(int).to_numpy()
    y_test = (y_test >= 1).astype("int64", copy=False)

    X_test = dataframe.drop(
        columns=[target_column, "__row_id__", "__test_order__"]
    ).to_numpy(dtype="float32")

    if X_test.shape[1] != IN_DIM:
        raise ValueError(
            f"Erwartet wurden {IN_DIM} Features, geladen wurden {X_test.shape[1]}."
        )
    if len(y_test) != len(test_row_ids):
        raise ValueError(
            f"Erwartet wurden {len(test_row_ids)} Test-Samples, "
            f"geladen wurden {len(y_test)}."
        )

    class_weights = torch.tensor(
        [float(meta["neg_weight"]), float(meta["pos_weight"])],
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_positive = int(np.sum(y_test == 1))
    n_negative = int(np.sum(y_test == 0))
    test_info = {
        "evaluation_set": "centralized_test",
        "selection_set": VALIDATION_SELECTION_SET,
        "validation_index_source": VALIDATION_INDEX_SOURCE,
        "selection_policy": (
            "Checkpoints are fixed before test evaluation; no model, round, "
            "threshold, or hyperparameter is selected on the test set."
        ),
        "parquet_path": str(parquet_path),
        "norm_stats_path": str(stats_path),
        "test_index_source": "norm_stats.json:test_idx",
        "test_row_ids_sha256_sorted": test_id_hash(test_row_ids),
        "n_samples": int(len(y_test)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive / len(y_test)),
        "n_features": int(X_test.shape[1]),
        "target": target_column,
        "binary_label_rule": "y = 1 if original target >= 1 else 0",
        "data_already_normalized": True,
        "class_weights": {
            "negative": float(class_weights[0]),
            "positive": float(class_weights[1]),
        },
        "split_integrity": split_validation,
    }
    return loader, class_weights, test_info


def make_criterion(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    """
    Weighted Cross-Entropy identical to the preceding SCAFFOLD validation evaluation.

    reduction='sum' plus global division by the sum of target-class weights
    reproduces one weighted mean over the complete test set and is independent
    of batch composition.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        reduction="sum",
    )


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Support direct state_dicts and common checkpoint wrapper formats."""
    state = checkpoint
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError("Checkpoint enthält kein gültiges PyTorch state_dict.")

    # Also support checkpoints produced through DataParallel/DDP.
    if state and all(isinstance(key, str) for key in state):
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}

    return state


def load_model(checkpoint_path: Path) -> nn.Module:
    """Load one selected checkpoint into the project MLP architecture."""
    model = MLP(in_dim=IN_DIM).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(extract_state_dict(checkpoint))
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate one fixed checkpoint once on the complete test set."""
    model.eval()

    total_weighted_loss = 0.0
    loss_normalizer = 0.0
    n_samples = 0
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for features, targets in loader:
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(features)
        loss_sum = criterion(logits, targets)

        total_weighted_loss += float(loss_sum.item())
        if criterion.weight is None:
            loss_normalizer += float(targets.numel())
        else:
            loss_normalizer += float(criterion.weight[targets].sum().item())
        n_samples += int(targets.numel())

        positive_probability = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(positive_probability.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())

    if not probabilities:
        raise ValueError("Der Test-DataLoader enthält keine Samples.")

    probs = np.concatenate(probabilities).astype(np.float64, copy=False)
    y_true = np.concatenate(labels).astype(np.int64, copy=False)

    average_loss = total_weighted_loss / max(
        loss_normalizer,
        np.finfo(float).eps,
    )

    if np.unique(y_true).size < 2:
        roc_auc_value = None
        average_precision_value = None
    else:
        roc_auc_value = float(roc_auc_score(y_true, probs))
        average_precision_value = float(average_precision_score(y_true, probs))

    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))

    metrics = {
        "loss": float(average_loss),
        "roc_auc": roc_auc_value,
        # Compatibility with the validation evaluator: pr_auc means AP.
        "pr_auc": average_precision_value,
        "pr_average_precision": average_precision_value,
        "pr_auc_definition": "average_precision_score",
        "n_samples": int(n_samples),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive / n_samples),
    }
    return metrics, probs, y_true


def compute_curves(probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Create complete ROC and PR curves from the same test predictions."""
    n_samples = int(len(labels))
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))

    if np.unique(labels).size < 2:
        return {
            "roc_auc": None,
            "roc_auc_from_curve": None,
            "pr_auc": None,
            "pr_average_precision": None,
            "pr_auc_trapezoidal": None,
            "positive_prevalence": (
                float(n_positive / n_samples) if n_samples else None
            ),
            "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities, pos_label=1)
    roc_auc_value = float(roc_auc_score(labels, probabilities))
    roc_auc_from_curve = float(auc(fpr, tpr))

    precision, recall, pr_thresholds = precision_recall_curve(
        labels,
        probabilities,
        pos_label=1,
    )
    average_precision_value = float(
        average_precision_score(labels, probabilities)
    )
    # sklearn returns recall in descending order.
    trapezoidal_pr_auc = float(auc(recall[::-1], precision[::-1]))

    if not np.isclose(
        roc_auc_value,
        roc_auc_from_curve,
        rtol=1e-10,
        atol=1e-12,
    ):
        print(
            "      ⚠️ ROC-AUC aus roc_auc_score und gespeicherter ROC-Kurve "
            "stimmen nicht überein."
        )

    return {
        "roc_auc": roc_auc_value,
        "roc_auc_from_curve": roc_auc_from_curve,
        "pr_auc": average_precision_value,
        "pr_average_precision": average_precision_value,
        "pr_auc_trapezoidal": trapezoidal_pr_auc,
        "pr_auc_definition": "average_precision_score",
        "positive_prevalence": float(n_positive / n_samples),
        "roc_curve": {
            "fpr": [float(value) for value in fpr],
            "tpr": [float(value) for value in tpr],
            # sklearn may use +inf for the first ROC threshold.
            "thresholds": [finite_or_none(value) for value in roc_thresholds],
            "length_relation": "len(fpr) = len(tpr) = len(thresholds)",
        },
        "pr_curve": {
            "precision": [float(value) for value in precision],
            "recall": [float(value) for value in recall],
            "thresholds": [float(value) for value in pr_thresholds],
            "length_relation": (
                "len(precision) = len(recall) = len(thresholds) + 1; "
                "the final precision/recall point has no threshold"
            ),
        },
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"      ⚠️ Konnte {path} nicht lesen: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_round_from_checkpoint(path: Path) -> Optional[int]:
    match = CHECKPOINT_PATTERN.search(path.name.strip())
    return int(match.group(1)) if match else None


def discover_selected_checkpoint(
    n_clients: int,
    selection_metric: str,
    run_tag: int,
) -> Optional[Dict[str, Any]]:
    """
    Locate the checkpoint already selected on validation data.

    Local files inside best<Metric>/run_<r>/ are authoritative. best_info.json
    is used for provenance and for disambiguating multiple checkpoint files.
    """
    selection_dir = (
        scaling_strategy_dir(n_clients)
        / f"best{selection_metric}"
        / f"run_{run_tag}"
    )
    if not selection_dir.is_dir():
        print(f"      ⚠️ Auswahlordner fehlt: {selection_dir}")
        return None

    best_info = load_optional_json(selection_dir / "best_info.json")
    expected_round = best_info.get("best_round")
    try:
        expected_round = int(expected_round) if expected_round is not None else None
    except (TypeError, ValueError):
        expected_round = None

    candidates = sorted(
        path
        for path in selection_dir.iterdir()
        if path.is_file() and CHECKPOINT_PATTERN.search(path.name.strip())
    )

    if expected_round is not None:
        matching_round = [
            path
            for path in candidates
            if parse_round_from_checkpoint(path) == expected_round
        ]
        if len(matching_round) == 1:
            candidates = matching_round
        elif len(matching_round) > 1:
            candidates = matching_round

    # Fallback to paths recorded in best_info.json if the local copy is absent.
    if not candidates:
        for key in ("copied_to", "source_checkpoint"):
            raw_path = best_info.get(key)
            if not raw_path:
                continue
            candidate = Path(str(raw_path))
            if candidate.is_file():
                candidates.append(candidate)
                break

    if not candidates:
        print(f"      ⚠️ Kein ausgewählter Checkpoint in {selection_dir}")
        return None

    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Mehrere ausgewählte Checkpoints in {selection_dir}: {names}. "
            "best_info.json muss best_round eindeutig angeben."
        )

    checkpoint = candidates[0]
    selected_round = expected_round
    if selected_round is None:
        selected_round = parse_round_from_checkpoint(checkpoint)
    if selected_round is None:
        raise ValueError(
            f"Rundennummer konnte aus Checkpoint nicht ermittelt werden: {checkpoint}"
        )

    return {
        "selection_dir": selection_dir,
        "checkpoint": checkpoint,
        "selected_round": int(selected_round),
        "best_info": best_info,
    }


def validation_metadata(best_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validation metrics without treating them as test results."""
    return {
        "roc_auc": finite_or_none(best_info.get("roc_auc")),
        "pr_auc": finite_or_none(best_info.get("pr_auc")),
        "loss": finite_or_none(best_info.get("loss")),
        "n_samples": best_info.get("n_samples"),
        "source_checkpoint": best_info.get("source_checkpoint"),
        "curves_file": best_info.get("curves_file"),
    }


def result_cache_key(
    n_clients: int,
    run_tag: int,
    selected_round: int,
) -> Tuple[int, int, int]:
    """Same run/round may have been selected by multiple validation criteria."""
    return n_clients, run_tag, selected_round


def evaluate_selected_model(
    n_clients: int,
    selection_metric: str,
    run_tag: int,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    cache: Dict[Tuple[int, int, int], Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    selected = discover_selected_checkpoint(
        n_clients=n_clients,
        selection_metric=selection_metric,
        run_tag=run_tag,
    )
    if selected is None:
        return None

    checkpoint: Path = selected["checkpoint"]
    selected_round = int(selected["selected_round"])
    best_info: Dict[str, Any] = selected["best_info"]
    key = result_cache_key(n_clients, run_tag, selected_round)

    if key in cache:
        test_metrics, curves = cache[key]
        print(
            f"      ↪ best{selection_metric} Run {run_tag}: Runde "
            f"{selected_round} bereits evaluiert; Ergebnis wiederverwendet."
        )
    else:
        model = load_model(checkpoint)
        test_metrics, probabilities, labels = evaluate_model(
            model,
            test_loader,
            criterion,
        )
        curves = compute_curves(probabilities, labels)
        cache[key] = (test_metrics, curves)
        print(
            f"      ✅ best{selection_metric} Run {run_tag}: Runde "
            f"{selected_round} | ROC={test_metrics['roc_auc']:.4f} "
            f"PR={test_metrics['pr_auc']:.4f} "
            f"Loss={test_metrics['loss']:.4f}"
        )

    output_dir = (
        scaling_output_dir(n_clients)
        / f"best{selection_metric}"
        / f"run_{run_tag}"
    )
    metrics_path = output_dir / "test_metrics.json"
    curves_path = output_dir / "test_curves.json"

    validation = validation_metadata(best_info)
    common_metadata = {
        "strategy": STRATEGY,
        "scaling_point": n_clients,
        "selection_metric": selection_metric,
        "selection_set": VALIDATION_SELECTION_SET,
        "validation_index_source": VALIDATION_INDEX_SOURCE,
        "evaluation_set": "centralized_test",
        "run": run_tag,
        "selected_round": selected_round,
        "selected_checkpoint": str(checkpoint),
        "original_source_checkpoint": best_info.get("source_checkpoint"),
        "validation_selection_metrics": validation,
    }

    metrics_payload = {
        **common_metadata,
        "test_metrics": {
            **test_metrics,
            "pr_auc_trapezoidal": curves.get("pr_auc_trapezoidal"),
            "roc_auc_from_curve": curves.get("roc_auc_from_curve"),
        },
        "test_curves_file": str(curves_path),
    }
    curves_payload = {
        **common_metadata,
        "test_loss": test_metrics["loss"],
        **curves,
    }

    write_json(metrics_path, metrics_payload)
    write_json(curves_path, curves_payload)

    return {
        "strategy": STRATEGY,
        "scaling_point": n_clients,
        "selection_metric": selection_metric,
        "run": run_tag,
        "selected_round": selected_round,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_pr_average_precision": test_metrics["pr_average_precision"],
        "test_pr_auc_trapezoidal": curves.get("pr_auc_trapezoidal"),
        "test_loss": test_metrics["loss"],
        "n_samples": test_metrics["n_samples"],
        "n_positive": test_metrics["n_positive"],
        "n_negative": test_metrics["n_negative"],
        "positive_prevalence": test_metrics["positive_prevalence"],
        "validation_roc_auc": validation["roc_auc"],
        "validation_pr_auc": validation["pr_auc"],
        "validation_loss": validation["loss"],
        "selected_checkpoint": str(checkpoint),
        "original_source_checkpoint": best_info.get("source_checkpoint"),
        "test_metrics_file": str(metrics_path),
        "test_curves_file": str(curves_path),
    }


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float]]:
    finite_values = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if finite_values.size == 0:
        return None, None
    mean = float(np.mean(finite_values))
    std = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
    return mean, std


def aggregate_results(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate raw test results over runs for each scale and criterion."""
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["scaling_point"]), str(row["selection_metric"]))].append(row)

    aggregate_rows: List[Dict[str, Any]] = []
    for (scaling_point, selection_metric), group in sorted(grouped.items()):
        roc_mean, roc_std = mean_std(row["test_roc_auc"] for row in group)
        pr_mean, pr_std = mean_std(row["test_pr_auc"] for row in group)
        pr_trap_mean, pr_trap_std = mean_std(
            row["test_pr_auc_trapezoidal"] for row in group
        )
        loss_mean, loss_std = mean_std(row["test_loss"] for row in group)

        aggregate_rows.append(
            {
                "strategy": STRATEGY,
                "scaling_point": scaling_point,
                "selection_metric": selection_metric,
                "n_runs": len(group),
                "test_roc_auc_mean": roc_mean,
                "test_roc_auc_std": roc_std,
                "test_pr_auc_mean": pr_mean,
                "test_pr_auc_std": pr_std,
                "test_pr_auc_trapezoidal_mean": pr_trap_mean,
                "test_pr_auc_trapezoidal_std": pr_trap_std,
                "test_loss_mean": loss_mean,
                "test_loss_std": loss_std,
            }
        )
    return aggregate_rows


def process_scaling_point(
    n_clients: int,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> Optional[Dict[str, Any]]:
    source_dir = scaling_strategy_dir(n_clients)
    if not source_dir.is_dir():
        print(f"⏭️  Skip {n_clients} Clients – Strategieordner fehlt: {source_dir}")
        return None

    print("\n" + "=" * 84)
    print(f"🧪 Finales Testset: {n_clients} Clients ({source_dir})")
    print("=" * 84)

    cache: Dict[
        Tuple[int, int, int],
        Tuple[Dict[str, Any], Dict[str, Any]],
    ] = {}
    rows: List[Dict[str, Any]] = []

    for selection_metric in SELECTION_METRICS:
        for run_tag in range(1, N_RUNS + 1):
            row = evaluate_selected_model(
                n_clients=n_clients,
                selection_metric=selection_metric,
                run_tag=run_tag,
                test_loader=test_loader,
                criterion=criterion,
                cache=cache,
            )
            if row is not None:
                rows.append(row)

    if not rows:
        print(f"⏭️  Keine ausgewählten Modelle für {n_clients} Clients gefunden.")
        return None

    rows.sort(key=lambda row: (row["selection_metric"], row["run"]))
    aggregate_rows = aggregate_results(rows)
    output_dir = scaling_output_dir(n_clients)

    raw_csv = output_dir / "test_results.csv"
    aggregate_csv = output_dir / "test_aggregate.csv"
    summary_json = output_dir / "test_summary.json"

    write_csv(raw_csv, rows, RAW_RESULT_FIELDS)
    write_csv(aggregate_csv, aggregate_rows, AGGREGATE_FIELDS)

    summary = {
        "strategy": STRATEGY,
        "scaling_point": n_clients,
        "selection_metrics": list(SELECTION_METRICS),
        "expected_runs_per_selection_metric": N_RUNS,
        "n_evaluations": len(rows),
        "results": rows,
        "aggregate_over_runs": aggregate_rows,
        "raw_csv": str(raw_csv),
        "aggregate_csv": str(aggregate_csv),
    }
    write_json(summary_json, summary)
    return summary


def main() -> None:
    print("=" * 84)
    print("🚀 FINAL TEST-SET EVALUATION")
    print(f"   Strategy              : {STRATEGY}")
    print(f"   Validation selections : {', '.join('best' + m for m in SELECTION_METRICS)}")
    print(f"   Runs per criterion    : {N_RUNS}")
    print(f"   Output root           : {FINAL_OUTPUT_ROOT}")
    print(f"   Device                : {DEVICE}")
    print("=" * 84)

    FINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    test_loader, class_weights, test_info = load_centralized_test(
        parquet_path=DATA_PARQUET,
        stats_path=NORM_STATS,
    )
    criterion = make_criterion(class_weights)
    write_json(FINAL_OUTPUT_ROOT / "test_set_info.json", test_info)

    print(
        f"✅ Testset geladen: {test_info['n_samples']} Samples | "
        f"positiv={test_info['n_positive']} | negativ={test_info['n_negative']} | "
        f"Prävalenz={test_info['positive_prevalence']:.4f}"
    )

    scaling_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for n_clients in SCALING_POINTS:
        summary = process_scaling_point(
            n_clients=n_clients,
            test_loader=test_loader,
            criterion=criterion,
        )
        if summary is None:
            continue
        scaling_summaries.append(summary)
        all_rows.extend(summary["results"])

    all_rows.sort(
        key=lambda row: (
            int(row["scaling_point"]),
            str(row["selection_metric"]),
            int(row["run"]),
        )
    )
    all_aggregate = aggregate_results(all_rows)

    global_raw_csv = FINAL_OUTPUT_ROOT / "all_test_results.csv"
    global_aggregate_csv = FINAL_OUTPUT_ROOT / "all_test_aggregate.csv"
    global_summary_json = FINAL_OUTPUT_ROOT / "final_test_summary.json"

    write_csv(global_raw_csv, all_rows, RAW_RESULT_FIELDS)
    write_csv(global_aggregate_csv, all_aggregate, AGGREGATE_FIELDS)
    write_json(
        global_summary_json,
        {
            "strategy": STRATEGY,
            "test_set": test_info,
            "selection_metrics": list(SELECTION_METRICS),
            "scaling_points_requested": SCALING_POINTS,
            "scaling_points_evaluated": [
                summary["scaling_point"] for summary in scaling_summaries
            ],
            "n_total_evaluations": len(all_rows),
            "results": all_rows,
            "aggregate_over_runs": all_aggregate,
            "scaling_point_summaries": [
                {
                    "scaling_point": summary["scaling_point"],
                    "n_evaluations": summary["n_evaluations"],
                    "summary_file": str(
                        scaling_output_dir(summary["scaling_point"])
                        / "test_summary.json"
                    ),
                }
                for summary in scaling_summaries
            ],
            "global_raw_csv": str(global_raw_csv),
            "global_aggregate_csv": str(global_aggregate_csv),
        },
    )

    print("\n" + "=" * 84)
    print(
        f"✅ Fertig: {len(all_rows)} ausgewählte Checkpoints aus "
        f"{len(scaling_summaries)} Skalierungspunkten auf dem Testset evaluiert."
    )
    print(f"   Einzelresultate : {global_raw_csv}")
    print(f"   Run-Aggregate   : {global_aggregate_csv}")
    print(f"   Gesamtübersicht : {global_summary_json}")
    print("=" * 84)


if __name__ == "__main__":
    main()
