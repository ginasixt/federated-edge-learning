#!/usr/bin/env python3
"""
Retrospective Test-Set Training-Dynamics Evaluation for FedAdam
===============================================================

For every FedAdam scaling point, run, and saved communication-round checkpoint,
this script evaluates Average Precision (AP) on the centralized test set.

The analysis is retrospective and descriptive. It does not alter training,
hyperparameters, validation-based checkpoint selection, or decision thresholds.

Definitions
-----------
Training speed
    First evaluated communication round at which test AP reaches at least
    AP_FRACTION of the highest test AP observed within the respective run.

Late-training behavior
    Mean test AP and the ordinary least-squares linear slope of test AP over
    the exact communication-round interval LATE_START_ROUND--LATE_END_ROUND.
    The default interval is rounds 35--45, matching the final 10-round interval
    of the 45-round FedAdam runs and containing 11 checkpoint values.

Expected checkpoint layout
--------------------------
result/splits_iid_scaling/
  splits_iid_<N>_clients.json/
    FedAdam/
      all_rounds_run_<r>/
        model_round_<ROUND>*.pt

The older directory scheme all_rounds_<r> is also supported.

Default outputs
---------------
result/splits_iid_scaling/training_dynamics_test/FedAdam/
  test_set_info.json
  all_round_test_ap.csv
  training_dynamics_by_run.csv
  training_dynamics_aggregate.csv
  training_dynamics_summary.json

Example
-------
python3 evaluate-training-dynamics-fedadam-test.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_learning.client_app import MLP


DEFAULT_RESULT_ROOT = Path("result/splits_iid_scaling")
DEFAULT_DATA_PARQUET = Path("data/diabetes_normalized.parquet")
DEFAULT_NORM_STATS = Path("data/norm_stats.json")
DEFAULT_STRATEGY = "FedAdam"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_RESULT_ROOT / "training_dynamics_test" / DEFAULT_STRATEGY
)

DEFAULT_SCALING_POINTS: Tuple[int, ...] = (
    2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
)

CHECKPOINT_PATTERN = re.compile(
    r"model_round_(\d+).*\.pt\s*$",
    re.IGNORECASE,
)

ROUND_RESULT_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "run", "round", "test_ap",
    "n_samples", "n_positive", "n_negative", "checkpoint",
)

RUN_SUMMARY_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "run", "n_evaluated_checkpoints",
    "first_evaluated_round", "final_evaluated_round",
    "best_test_ap", "best_test_ap_round",
    "ap_fraction", "ap_fraction_threshold",
    "first_round_reaching_fraction", "test_ap_at_first_fraction_round",
    "rounds_from_fraction_to_end",
    "late_window_start_round", "late_window_end_round",
    "late_window_expected_rounds", "late_window_interval_length",
    "late_window_observed_rounds", "late_window_complete",
    "late_window_missing_rounds",
    "late_mean_test_ap", "late_test_ap_slope_per_round",
    "late_fitted_test_ap_change", "late_test_ap_std",
    "final_round_test_ap",
)

AGGREGATE_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "n_runs",
    "first_round_reaching_fraction_mean",
    "first_round_reaching_fraction_std",
    "first_round_reaching_fraction_min",
    "first_round_reaching_fraction_max",
    "best_test_ap_mean", "best_test_ap_std",
    "late_mean_test_ap_mean", "late_mean_test_ap_std",
    "late_test_ap_slope_per_round_mean",
    "late_test_ap_slope_per_round_std",
    "late_test_ap_slope_per_round_min",
    "late_test_ap_slope_per_round_max",
    "late_fitted_test_ap_change_mean",
    "late_fitted_test_ap_change_std",
    "complete_late_windows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all saved FedAdam checkpoints on the centralized test set "
            "and summarize AP-based training dynamics."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--data-parquet", type=Path, default=DEFAULT_DATA_PARQUET)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scaling-points",
        nargs="+",
        type=int,
        default=list(DEFAULT_SCALING_POINTS),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--in-dim", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ap-fraction", type=float, default=0.99)
    parser.add_argument("--late-start-round", type=int, default=35)
    parser.add_argument("--late-end-round", type=int, default=45)
    parser.add_argument(
        "--allow-incomplete-late-window",
        action="store_true",
        help=(
            "Use available checkpoints inside the requested late window when "
            "individual round files are missing. The output is marked incomplete."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ap_fraction <= 1.0:
        raise ValueError("--ap-fraction must be in (0, 1].")
    if args.runs < 1 or args.batch_size < 1 or args.in_dim < 1:
        raise ValueError("--runs, --batch-size, and --in-dim must be positive.")
    if args.late_start_round < 1:
        raise ValueError("--late-start-round must be positive.")
    if args.late_end_round <= args.late_start_round:
        raise ValueError(
            "--late-end-round must be greater than --late-start-round."
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def validate_split_indices(meta: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("train_idx", "val_idx", "test_idx")
    missing = [name for name in required if name not in meta]
    if missing:
        raise KeyError(
            "norm_stats.json is missing split indices: " + ", ".join(missing)
        )

    splits = {
        name: [int(value) for value in meta[name]]
        for name in required
    }
    for name, values in splits.items():
        if not values:
            raise ValueError(f"'{name}' is empty.")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate row IDs found in '{name}'.")

    train = set(splits["train_idx"])
    validation = set(splits["val_idx"])
    test = set(splits["test_idx"])
    overlaps = {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }
    if any(overlaps.values()):
        raise ValueError(f"Train/validation/test indices overlap: {overlaps}")

    return {
        "train_size": len(train),
        "validation_size": len(validation),
        "test_size": len(test),
        "total_unique_rows": len(train | validation | test),
        "overlaps": overlaps,
    }


def hash_row_ids(row_ids: Sequence[int]) -> str:
    canonical = ",".join(str(value) for value in sorted(row_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_centralized_test(
    parquet_path: Path,
    stats_path: Path,
    batch_size: int,
    in_dim: int,
) -> Tuple[DataLoader, Dict[str, Any]]:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing normalized parquet: {parquet_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing norm statistics: {stats_path}")

    meta = json.loads(stats_path.read_text(encoding="utf-8"))
    split_integrity = validate_split_indices(meta)
    test_row_ids = [int(value) for value in meta["test_idx"]]

    dataframe = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", test_row_ids)],
    )

    target = str(meta["target"])
    required_columns = {"__row_id__", target}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(
            "Parquet is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if dataframe["__row_id__"].duplicated().any():
        raise ValueError("Duplicate __row_id__ values found in test data.")

    requested = set(test_row_ids)
    loaded = set(dataframe["__row_id__"].astype(int))
    if requested != loaded:
        raise ValueError(
            "Loaded test rows do not match norm_stats.json:test_idx: "
            f"missing={len(requested - loaded)}, "
            f"unexpected={len(loaded - requested)}"
        )

    order = {row_id: position for position, row_id in enumerate(test_row_ids)}
    dataframe = dataframe.assign(
        __test_order__=dataframe["__row_id__"].astype(int).map(order)
    ).sort_values("__test_order__", kind="stable")

    y_test = dataframe[target].astype(int).to_numpy()
    y_test = (y_test >= 1).astype(np.int64, copy=False)
    X_test = dataframe.drop(
        columns=[target, "__row_id__", "__test_order__"]
    ).to_numpy(dtype=np.float32)

    if X_test.ndim != 2 or X_test.shape[1] != in_dim:
        raise ValueError(
            f"Expected test matrix (*, {in_dim}), loaded {X_test.shape}."
        )
    if len(y_test) != len(test_row_ids):
        raise ValueError(
            f"Expected {len(test_row_ids)} test rows, loaded {len(y_test)}."
        )
    if np.unique(y_test).size != 2:
        raise ValueError("Average Precision requires both test classes.")

    dataset = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    n_positive = int(np.sum(y_test == 1))
    n_negative = int(np.sum(y_test == 0))
    info = {
        "evaluation_set": "centralized_test",
        "analysis_type": "retrospective_training_dynamics",
        "selection_context": (
            "Test-set trajectories are used only for retrospective description. "
            "They do not alter training, validation-based checkpoint selection, "
            "threshold selection, or hyperparameters."
        ),
        "parquet_path": str(parquet_path),
        "norm_stats_path": str(stats_path),
        "test_index_source": "norm_stats.json:test_idx",
        "test_row_ids_sha256_sorted": hash_row_ids(test_row_ids),
        "n_samples": int(len(y_test)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_prevalence": float(n_positive / len(y_test)),
        "n_features": int(X_test.shape[1]),
        "target": target,
        "binary_label_rule": "y = 1 if original target >= 1 else 0",
        "data_already_normalized": True,
        "split_integrity": split_integrity,
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


def discover_round_models(run_dir: Path) -> List[Tuple[int, Path]]:
    if not run_dir.is_dir():
        return []

    by_round: Dict[int, Path] = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        match = CHECKPOINT_PATTERN.search(path.name.strip())
        if not match:
            continue

        round_number = int(match.group(1))
        if round_number in by_round:
            raise RuntimeError(
                f"Multiple checkpoints for round {round_number} in {run_dir}: "
                f"{by_round[round_number].name}, {path.name}"
            )
        by_round[round_number] = path

    return sorted(by_round.items())


def discover_run_models(
    strategy_dir: Path,
    run_tag: int,
) -> Tuple[List[Tuple[int, Path]], List[Path]]:
    candidates = (
        strategy_dir / f"all_rounds_run_{run_tag}",
        strategy_dir / f"all_rounds_{run_tag}",
    )
    existing = [path for path in candidates if path.is_dir()]

    by_round: Dict[int, Path] = {}
    for run_dir in existing:
        for round_number, checkpoint in discover_round_models(run_dir):
            if round_number in by_round:
                raise RuntimeError(
                    f"Round {round_number} occurs in multiple directories for "
                    f"run {run_tag}: {by_round[round_number]}, {checkpoint}"
                )
            by_round[round_number] = checkpoint

    return sorted(by_round.items()), existing


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


def load_model(
    checkpoint_path: Path,
    in_dim: int,
    device: torch.device,
) -> nn.Module:
    model = MLP(in_dim=in_dim).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()
    return model


@torch.no_grad()
def evaluate_test_ap(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, int, int, int]:
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for features, targets in loader:
        logits = model(features.to(device))
        positive_probability = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(
            positive_probability.detach().cpu().numpy()
        )
        labels.append(targets.detach().cpu().numpy())

    if not probabilities:
        raise ValueError("The test DataLoader contains no samples.")

    probabilities_array = np.concatenate(probabilities).astype(
        np.float64,
        copy=False,
    )
    labels_array = np.concatenate(labels).astype(
        np.int64,
        copy=False,
    )
    if np.unique(labels_array).size != 2:
        raise ValueError("Average Precision requires both test classes.")

    test_ap = float(
        average_precision_score(
            labels_array,
            probabilities_array,
        )
    )
    n_positive = int(np.sum(labels_array == 1))
    n_negative = int(np.sum(labels_array == 0))
    return test_ap, int(labels_array.size), n_positive, n_negative


def summarize_run(
    round_rows: Sequence[Mapping[str, Any]],
    ap_fraction: float,
    late_start_round: int,
    late_end_round: int,
    allow_incomplete_late_window: bool,
) -> Dict[str, Any]:
    if not round_rows:
        raise ValueError("Cannot summarize an empty run.")

    ordered = sorted(
        round_rows,
        key=lambda row: int(row["round"]),
    )
    rounds = np.asarray(
        [int(row["round"]) for row in ordered],
        dtype=int,
    )
    test_ap = np.asarray(
        [float(row["test_ap"]) for row in ordered],
        dtype=float,
    )

    best_index = int(np.argmax(test_ap))
    best_ap = float(test_ap[best_index])
    best_round = int(rounds[best_index])
    fraction_threshold = float(ap_fraction * best_ap)

    qualifying = np.flatnonzero(test_ap >= fraction_threshold)
    if qualifying.size == 0:
        raise RuntimeError("No checkpoint reached the requested AP fraction.")

    first_index = int(qualifying[0])
    first_fraction_round = int(rounds[first_index])
    first_fraction_ap = float(test_ap[first_index])

    expected_rounds = list(
        range(late_start_round, late_end_round + 1)
    )
    row_by_round = {
        int(row["round"]): row
        for row in ordered
    }
    missing_rounds = [
        round_number
        for round_number in expected_rounds
        if round_number not in row_by_round
    ]

    if missing_rounds and not allow_incomplete_late_window:
        raise RuntimeError(
            f"Late window {late_start_round}-{late_end_round} is incomplete; "
            f"missing rounds: {missing_rounds}. Use "
            "--allow-incomplete-late-window only if intentional."
        )

    late_rows = [
        row_by_round[round_number]
        for round_number in expected_rounds
        if round_number in row_by_round
    ]
    if len(late_rows) < 2:
        raise RuntimeError(
            "At least two checkpoints are required for the late AP trend."
        )

    late_x = np.asarray(
        [int(row["round"]) for row in late_rows],
        dtype=float,
    )
    late_y = np.asarray(
        [float(row["test_ap"]) for row in late_rows],
        dtype=float,
    )
    slope, intercept = np.polyfit(late_x, late_y, deg=1)
    fitted_change = float(
        (slope * late_end_round + intercept)
        - (slope * late_start_round + intercept)
    )

    final_round = int(rounds.max())
    return {
        "strategy": str(ordered[0]["strategy"]),
        "scaling_point": int(ordered[0]["scaling_point"]),
        "run": int(ordered[0]["run"]),
        "n_evaluated_checkpoints": int(len(ordered)),
        "first_evaluated_round": int(rounds.min()),
        "final_evaluated_round": final_round,
        "best_test_ap": best_ap,
        "best_test_ap_round": best_round,
        "ap_fraction": float(ap_fraction),
        "ap_fraction_threshold": fraction_threshold,
        "first_round_reaching_fraction": first_fraction_round,
        "test_ap_at_first_fraction_round": first_fraction_ap,
        "rounds_from_fraction_to_end": int(
            final_round - first_fraction_round
        ),
        "late_window_start_round": int(late_start_round),
        "late_window_end_round": int(late_end_round),
        "late_window_expected_rounds": int(len(expected_rounds)),
        "late_window_interval_length": int(
            late_end_round - late_start_round
        ),
        "late_window_observed_rounds": int(len(late_rows)),
        "late_window_complete": len(missing_rounds) == 0,
        "late_window_missing_rounds": ",".join(
            str(value) for value in missing_rounds
        ),
        "late_mean_test_ap": float(np.mean(late_y)),
        "late_test_ap_slope_per_round": float(slope),
        "late_fitted_test_ap_change": fitted_change,
        "late_test_ap_std": float(np.std(late_y, ddof=1)),
        "final_round_test_ap": float(test_ap[-1]),
    }


def finite_mean_std(
    values: Iterable[Any],
) -> Tuple[Optional[float], Optional[float]]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=float,
    )
    if finite.size == 0:
        return None, None
    mean = float(np.mean(finite))
    std = (
        float(np.std(finite, ddof=1))
        if finite.size > 1
        else 0.0
    )
    return mean, std


def aggregate_run_summaries(
    rows: Sequence[Mapping[str, Any]],
    strategy: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["scaling_point"])].append(row)

    aggregates: List[Dict[str, Any]] = []
    for scaling_point, group in sorted(grouped.items()):
        first_values = np.asarray(
            [
                float(row["first_round_reaching_fraction"])
                for row in group
            ],
            dtype=float,
        )
        slope_values = np.asarray(
            [
                float(row["late_test_ap_slope_per_round"])
                for row in group
            ],
            dtype=float,
        )

        first_mean, first_std = finite_mean_std(first_values)
        best_mean, best_std = finite_mean_std(
            row["best_test_ap"] for row in group
        )
        late_mean_mean, late_mean_std = finite_mean_std(
            row["late_mean_test_ap"] for row in group
        )
        slope_mean, slope_std = finite_mean_std(slope_values)
        fitted_mean, fitted_std = finite_mean_std(
            row["late_fitted_test_ap_change"]
            for row in group
        )

        aggregates.append({
            "strategy": strategy,
            "scaling_point": int(scaling_point),
            "n_runs": int(len(group)),
            "first_round_reaching_fraction_mean": first_mean,
            "first_round_reaching_fraction_std": first_std,
            "first_round_reaching_fraction_min": float(
                np.min(first_values)
            ),
            "first_round_reaching_fraction_max": float(
                np.max(first_values)
            ),
            "best_test_ap_mean": best_mean,
            "best_test_ap_std": best_std,
            "late_mean_test_ap_mean": late_mean_mean,
            "late_mean_test_ap_std": late_mean_std,
            "late_test_ap_slope_per_round_mean": slope_mean,
            "late_test_ap_slope_per_round_std": slope_std,
            "late_test_ap_slope_per_round_min": float(
                np.min(slope_values)
            ),
            "late_test_ap_slope_per_round_max": float(
                np.max(slope_values)
            ),
            "late_fitted_test_ap_change_mean": fitted_mean,
            "late_fitted_test_ap_change_std": fitted_std,
            "complete_late_windows": int(
                sum(bool(row["late_window_complete"]) for row in group)
            ),
        })

    return aggregates


def evaluate_run(
    *,
    strategy: str,
    scaling_point: int,
    run_tag: int,
    checkpoints: Sequence[Tuple[int, Path]],
    test_loader: DataLoader,
    in_dim: int,
    device: torch.device,
    ap_fraction: float,
    late_start_round: int,
    late_end_round: int,
    allow_incomplete_late_window: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    round_rows: List[Dict[str, Any]] = []

    for round_number, checkpoint_path in checkpoints:
        model = load_model(
            checkpoint_path,
            in_dim=in_dim,
            device=device,
        )
        test_ap, n_samples, n_positive, n_negative = evaluate_test_ap(
            model,
            test_loader,
            device,
        )

        round_rows.append({
            "strategy": strategy,
            "scaling_point": int(scaling_point),
            "run": int(run_tag),
            "round": int(round_number),
            "test_ap": test_ap,
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "checkpoint": str(checkpoint_path),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_run(
        round_rows=round_rows,
        ap_fraction=ap_fraction,
        late_start_round=late_start_round,
        late_end_round=late_end_round,
        allow_incomplete_late_window=allow_incomplete_late_window,
    )

    print(
        f"      ✓ first {ap_fraction * 100:.1f}% round="
        f"{summary['first_round_reaching_fraction']} | "
        f"late mean AP={summary['late_mean_test_ap']:.6f} | "
        f"late slope={summary['late_test_ap_slope_per_round']:.8f}/round"
    )
    return round_rows, summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 92)
    print("RETROSPECTIVE FEDADAM TEST-SET TRAINING-DYNAMICS EVALUATION")
    print(f"Strategy       : {args.strategy}")
    print(f"Scaling points : {args.scaling_points}")
    print(f"Runs           : {args.runs}")
    print(f"AP fraction    : {args.ap_fraction:.4f}")
    print(
        f"Late window    : rounds {args.late_start_round}-"
        f"{args.late_end_round} "
        f"({args.late_end_round - args.late_start_round}-round interval)"
    )
    print(f"Device         : {device}")
    print(f"Output         : {args.output_dir}")
    print("=" * 92)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_loader, test_info = load_centralized_test(
        parquet_path=args.data_parquet,
        stats_path=args.norm_stats,
        batch_size=args.batch_size,
        in_dim=args.in_dim,
    )
    print(
        f"\n✓ Centralized test loaded: n={test_info['n_samples']}, "
        f"positive={test_info['n_positive']}, "
        f"negative={test_info['n_negative']}, "
        f"prevalence={test_info['positive_prevalence']:.4f}"
    )

    all_round_rows: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    missing_runs: List[Dict[str, Any]] = []
    skipped_scaling_points: List[Dict[str, Any]] = []

    for scaling_point in args.scaling_points:
        strategy_dir = scaling_strategy_dir(
            args.result_root,
            args.strategy,
            scaling_point,
        )
        if not strategy_dir.is_dir():
            message = f"Missing strategy directory: {strategy_dir}"
            print(f"\n⚠️  Skip {scaling_point} clients: {message}")
            skipped_scaling_points.append({
                "scaling_point": int(scaling_point),
                "reason": message,
            })
            continue

        print("\n" + "-" * 92)
        print(f"{scaling_point} clients")
        print("-" * 92)

        for run_tag in range(1, args.runs + 1):
            checkpoints, run_dirs = discover_run_models(
                strategy_dir,
                run_tag,
            )
            if not checkpoints:
                print(
                    f"   ⚠️  Run {run_tag}: no checkpoints found in "
                    f"all_rounds_run_{run_tag} or all_rounds_{run_tag}"
                )
                missing_runs.append({
                    "scaling_point": int(scaling_point),
                    "run": int(run_tag),
                })
                continue

            used_dirs = ", ".join(path.name for path in run_dirs)
            print(
                f"   Run {run_tag}: {len(checkpoints)} checkpoints "
                f"from {used_dirs}"
            )

            round_rows, run_summary = evaluate_run(
                strategy=args.strategy,
                scaling_point=scaling_point,
                run_tag=run_tag,
                checkpoints=checkpoints,
                test_loader=test_loader,
                in_dim=args.in_dim,
                device=device,
                ap_fraction=args.ap_fraction,
                late_start_round=args.late_start_round,
                late_end_round=args.late_end_round,
                allow_incomplete_late_window=(
                    args.allow_incomplete_late_window
                ),
            )
            all_round_rows.extend(round_rows)
            run_summaries.append(run_summary)

    if not run_summaries:
        raise RuntimeError("No FedAdam runs were evaluated.")

    aggregate_rows = aggregate_run_summaries(
        run_summaries,
        strategy=args.strategy,
    )

    round_path = args.output_dir / "all_round_test_ap.csv"
    run_path = args.output_dir / "training_dynamics_by_run.csv"
    aggregate_path = args.output_dir / "training_dynamics_aggregate.csv"
    info_path = args.output_dir / "test_set_info.json"
    summary_path = args.output_dir / "training_dynamics_summary.json"

    write_csv(round_path, all_round_rows, ROUND_RESULT_FIELDS)
    write_csv(run_path, run_summaries, RUN_SUMMARY_FIELDS)
    write_csv(aggregate_path, aggregate_rows, AGGREGATE_FIELDS)
    write_json(info_path, test_info)

    summary_payload = {
        "strategy": args.strategy,
        "evaluation_set": "centralized_test",
        "analysis_type": "retrospective_training_dynamics",
        "training_speed_definition": (
            f"First evaluated round with test AP >= "
            f"{args.ap_fraction:.6f} * best observed test AP in the run."
        ),
        "late_training_definition": (
            f"Mean test AP and OLS AP slope over rounds "
            f"{args.late_start_round}-{args.late_end_round} "
            f"({args.late_end_round - args.late_start_round}-round interval)."
        ),
        "selection_context": (
            "The test trajectories are descriptive and did not alter training, "
            "validation-based checkpoint selection, thresholds, or hyperparameters."
        ),
        "scaling_points_evaluated": sorted({
            int(row["scaling_point"]) for row in run_summaries
        }),
        "n_run_summaries": int(len(run_summaries)),
        "missing_runs": missing_runs,
        "skipped_scaling_points": skipped_scaling_points,
        "test_set_info_file": str(info_path),
        "round_results_file": str(round_path),
        "run_summary_file": str(run_path),
        "aggregate_file": str(aggregate_path),
        "run_summaries": run_summaries,
    }
    write_json(summary_path, summary_payload)

    print("\n" + "=" * 92)
    print(f"✓ Evaluated run summaries : {len(run_summaries)}")
    print(f"✓ Round-level results     : {round_path}")
    print(f"✓ Run-level results       : {run_path}")
    print(f"✓ Aggregate results       : {aggregate_path}")
    print(f"✓ Summary                 : {summary_path}")
    print()
    print("Next step:")
    print("python3 plot-training-dynamics-fedadam-test.py")
    print("=" * 92)


if __name__ == "__main__":
    main()
