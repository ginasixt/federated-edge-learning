#!/usr/bin/env python3
"""
Retrospective Test-Set Training-Dynamics Evaluation for SCAFFOLD
================================================================

For every SCAFFOLD scaling point, run, and saved communication-round
checkpoint, this script evaluates Average Precision (AP) on the centralized
test set.

The analysis is descriptive and retrospective. It does not change the trained
models, the validation-based checkpoint selection, thresholds, or any
hyperparameters.

Definitions
-----------
Training speed
    First evaluated communication round at which test AP reaches at least
    ``AP_FRACTION`` of the highest test AP observed within the respective run:

        AP_r >= AP_FRACTION * max_r(AP_r)

    With the default AP_FRACTION=0.99, this is the first evaluated round
    reaching 99% of the best observed test AP.

Late-training behavior
    Mean test AP and the ordinary least-squares linear slope of test AP over
    the final ``LATE_ROUNDS`` communication rounds. With 80 total rounds and
    the default LATE_ROUNDS=10, the window is exactly rounds 71-80.

Important
---------
"First evaluated round" is used because only saved checkpoints can be
evaluated. The late-training window, however, is defined by communication-round
numbers, not by the last ten saved checkpoints. By default, all rounds in that
window must be present.

Expected checkpoint layout
--------------------------
result/splits_iid_scaling/
  splits_iid_<N>_clients.json/
    SCAFFOLD/
      all_rounds_run_<r>/
        model_round_<ROUND>*.pt

The older directory scheme ``all_rounds_<r>`` is also supported.

Default outputs
---------------
result/splits_iid_scaling/training_dynamics/SCAFFOLD/
  test_set_info.json
  all_round_test_ap.csv
  training_dynamics_by_run.csv
  training_dynamics_aggregate.csv
  training_dynamics_summary.json

Example
-------
python evaluate-training-dynamics-scaffold.py

To permit an incomplete final-round window while retaining an explicit flag:
python evaluate-training-dynamics-scaffold.py --allow-incomplete-final-window
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_learning.client_app import MLP


DEFAULT_RESULT_ROOT = Path("result/splits_iid_scaling")
DEFAULT_DATA_PARQUET = Path("data/diabetes_normalized.parquet")
DEFAULT_NORM_STATS = Path("data/norm_stats.json")
DEFAULT_STRATEGY = "SCAFFOLD"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "training_dynamics" / DEFAULT_STRATEGY

DEFAULT_SCALING_POINTS: Tuple[int, ...] = (
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
)

CHECKPOINT_PATTERN = re.compile(r"model_round_(\d+).*\.pt\s*$", re.IGNORECASE)

ROUND_RESULT_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "run", "round", "test_ap", "n_samples",
    "n_positive", "n_negative", "checkpoint",
)

RUN_SUMMARY_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "run", "n_evaluated_checkpoints",
    "first_evaluated_round", "final_evaluated_round", "best_test_ap",
    "best_test_ap_round", "ap_fraction", "ap_fraction_threshold",
    "first_round_reaching_fraction", "test_ap_at_first_fraction_round",
    "rounds_from_fraction_to_end", "late_window_start_round",
    "late_window_end_round", "late_window_expected_rounds",
    "late_window_observed_rounds", "late_window_complete",
    "late_window_missing_rounds", "late_mean_test_ap",
    "late_test_ap_slope_per_round", "late_fitted_ap_change",
    "late_test_ap_std", "final_round_test_ap",
)

AGGREGATE_FIELDS: Tuple[str, ...] = (
    "strategy", "scaling_point", "n_runs",
    "first_round_reaching_fraction_mean", "first_round_reaching_fraction_std",
    "first_round_reaching_fraction_min", "first_round_reaching_fraction_max",
    "best_test_ap_mean", "best_test_ap_std", "late_mean_test_ap_mean",
    "late_mean_test_ap_std", "late_test_ap_slope_per_round_mean",
    "late_test_ap_slope_per_round_std", "late_test_ap_slope_per_round_min",
    "late_test_ap_slope_per_round_max", "late_fitted_ap_change_mean",
    "late_fitted_ap_change_std", "complete_late_windows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all saved SCAFFOLD checkpoints on the centralized test "
            "set and summarize AP-based training speed and late-training behavior."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--data-parquet", type=Path, default=DEFAULT_DATA_PARQUET)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scaling-points", nargs="+", type=int, default=list(DEFAULT_SCALING_POINTS)
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--in-dim", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ap-fraction", type=float, default=0.99)
    parser.add_argument("--late-rounds", type=int, default=10)
    parser.add_argument(
        "--allow-incomplete-final-window",
        action="store_true",
        help=(
            "Use available rounds if exact final-window checkpoints are missing; "
            "the output remains explicitly marked as incomplete."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ap_fraction <= 1.0:
        raise ValueError("--ap-fraction must be in the interval (0, 1].")
    if args.late_rounds < 2:
        raise ValueError("--late-rounds must be at least 2 to estimate a slope.")
    if args.runs < 1 or args.batch_size < 1:
        raise ValueError("--runs and --batch-size must be positive.")


def validate_split_indices(meta: Dict[str, Any]) -> Dict[str, Any]:
    split_names = ("train_idx", "val_idx", "test_idx")
    missing = [name for name in split_names if name not in meta]
    if missing:
        raise KeyError("norm_stats.json is missing split indices: " + ", ".join(missing))

    splits = {name: [int(value) for value in meta[name]] for name in split_names}
    for name, values in splits.items():
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


def test_id_hash(test_row_ids: Sequence[int]) -> str:
    canonical = ",".join(str(value) for value in sorted(test_row_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_centralized_test(
    parquet_path: Path,
    stats_path: Path,
    batch_size: int,
    in_dim: int,
) -> Tuple[DataLoader, Dict[str, Any]]:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing norm statistics: {stats_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing normalized parquet file: {parquet_path}")

    meta = json.loads(stats_path.read_text(encoding="utf-8"))
    split_integrity = validate_split_indices(meta)
    test_row_ids = [int(value) for value in meta["test_idx"]]
    requested = set(test_row_ids)

    dataframe = pd.read_parquet(
        parquet_path,
        filters=[("__row_id__", "in", test_row_ids)],
    )

    target = str(meta["target"])
    required_columns = {"__row_id__", target}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(
            "Parquet file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if dataframe["__row_id__"].duplicated().any():
        raise ValueError("Duplicate __row_id__ values found in the test data.")

    loaded = set(dataframe["__row_id__"].astype(int).tolist())
    missing_ids = requested - loaded
    unexpected_ids = loaded - requested
    if missing_ids or unexpected_ids:
        raise ValueError(
            "Loaded test rows do not match norm_stats['test_idx']: "
            f"missing={len(missing_ids)}, unexpected={len(unexpected_ids)}"
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

    if X_test.shape[1] != in_dim:
        raise ValueError(f"Expected {in_dim} features, but loaded {X_test.shape[1]}.")
    if np.unique(y_test).size < 2:
        raise ValueError("Average Precision requires both classes in the test set.")

    dataset = TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    n_positive = int(np.sum(y_test == 1))
    n_negative = int(np.sum(y_test == 0))
    info = {
        "evaluation_set": "centralized_test",
        "analysis_type": "retrospective_checkpoint_trajectory",
        "selection_policy": (
            "Test trajectories are used only for post-hoc description of training "
            "dynamics and do not alter training, validation-based checkpoint "
            "selection, thresholds, or hyperparameters."
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
        "target": target,
        "binary_label_rule": "y = 1 if original target >= 1 else 0",
        "data_already_normalized": True,
        "split_integrity": split_integrity,
    }
    return loader, info


def scaling_strategy_dir(result_root: Path, strategy: str, n_clients: int) -> Path:
    return result_root / f"splits_iid_{n_clients}_clients.json" / strategy


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
    strategy_dir: Path, run_tag: int
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
                    f"Round {round_number} occurs in multiple run directories for "
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
        raise TypeError("Checkpoint does not contain a valid PyTorch state_dict.")
    if state and all(isinstance(key, str) for key in state):
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_model(checkpoint_path: Path, in_dim: int, device: torch.device) -> nn.Module:
    model = MLP(in_dim=in_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    return model


@torch.no_grad()
def evaluate_test_ap(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, int, int, int]:
    model.eval()
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for features, targets in loader:
        logits = model(features.to(device))
        positive_probability = torch.softmax(logits, dim=1)[:, 1]
        probabilities.append(positive_probability.detach().cpu().numpy())
        labels.append(targets.numpy())

    if not probabilities:
        raise ValueError("The test DataLoader contains no samples.")
    probs = np.concatenate(probabilities).astype(np.float64, copy=False)
    y_true = np.concatenate(labels).astype(np.int64, copy=False)
    if np.unique(y_true).size < 2:
        raise ValueError("Average Precision requires both classes.")

    test_ap = float(average_precision_score(y_true, probs))
    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))
    return test_ap, int(len(y_true)), n_positive, n_negative


def summarize_run(
    round_rows: Sequence[Dict[str, Any]],
    ap_fraction: float,
    late_rounds: int,
    allow_incomplete_final_window: bool,
) -> Dict[str, Any]:
    if not round_rows:
        raise ValueError("Cannot summarize an empty run.")

    ordered = sorted(round_rows, key=lambda row: int(row["round"]))
    rounds = np.asarray([int(row["round"]) for row in ordered], dtype=int)
    test_ap = np.asarray([float(row["test_ap"]) for row in ordered], dtype=float)

    best_index = int(np.argmax(test_ap))
    best_ap = float(test_ap[best_index])
    best_round = int(rounds[best_index])
    fraction_threshold = float(ap_fraction * best_ap)
    qualifying = np.flatnonzero(test_ap >= fraction_threshold)
    first_fraction_index = int(qualifying[0])
    first_fraction_round = int(rounds[first_fraction_index])
    first_fraction_ap = float(test_ap[first_fraction_index])

    final_round = int(rounds.max())
    late_start = final_round - late_rounds + 1
    expected_rounds = list(range(late_start, final_round + 1))
    row_by_round = {int(row["round"]): row for row in ordered}
    missing_rounds = [r for r in expected_rounds if r not in row_by_round]

    if missing_rounds and not allow_incomplete_final_window:
        raise RuntimeError(
            "The exact final communication-round window is incomplete. "
            f"Expected rounds {late_start}-{final_round}, missing {missing_rounds}. "
            "Re-run with --allow-incomplete-final-window only if intentional."
        )

    late_rows = [row_by_round[r] for r in expected_rounds if r in row_by_round]
    if len(late_rows) < 2:
        raise RuntimeError("At least two checkpoints are required for the late AP trend.")

    late_x = np.asarray([int(row["round"]) for row in late_rows], dtype=float)
    late_y = np.asarray([float(row["test_ap"]) for row in late_rows], dtype=float)
    slope, intercept = np.polyfit(late_x, late_y, deg=1)
    fitted_change = float(
        (slope * late_x.max() + intercept) - (slope * late_x.min() + intercept)
    )

    return {
        "strategy": ordered[0]["strategy"],
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
        "rounds_from_fraction_to_end": int(final_round - first_fraction_round),
        "late_window_start_round": late_start,
        "late_window_end_round": final_round,
        "late_window_expected_rounds": late_rounds,
        "late_window_observed_rounds": int(len(late_rows)),
        "late_window_complete": len(missing_rounds) == 0,
        "late_window_missing_rounds": ",".join(str(v) for v in missing_rounds),
        "late_mean_test_ap": float(np.mean(late_y)),
        "late_test_ap_slope_per_round": float(slope),
        "late_fitted_ap_change": fitted_change,
        "late_test_ap_std": float(np.std(late_y, ddof=1)),
        "final_round_test_ap": float(test_ap[-1]),
    }


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float]]:
    finite = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(float(v))],
        dtype=float,
    )
    if finite.size == 0:
        return None, None
    return float(np.mean(finite)), (
        float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    )


def aggregate_run_summaries(
    rows: Sequence[Dict[str, Any]], strategy: str
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["scaling_point"])].append(row)

    aggregates: List[Dict[str, Any]] = []
    for scaling_point, group in sorted(grouped.items()):
        first_values = np.asarray(
            [float(row["first_round_reaching_fraction"]) for row in group]
        )
        slope_values = np.asarray(
            [float(row["late_test_ap_slope_per_round"]) for row in group]
        )
        first_mean, first_std = mean_std(first_values)
        best_mean, best_std = mean_std(row["best_test_ap"] for row in group)
        late_mean_mean, late_mean_std = mean_std(
            row["late_mean_test_ap"] for row in group
        )
        slope_mean, slope_std = mean_std(slope_values)
        fitted_mean, fitted_std = mean_std(
            row["late_fitted_ap_change"] for row in group
        )
        aggregates.append({
            "strategy": strategy,
            "scaling_point": scaling_point,
            "n_runs": len(group),
            "first_round_reaching_fraction_mean": first_mean,
            "first_round_reaching_fraction_std": first_std,
            "first_round_reaching_fraction_min": float(np.min(first_values)),
            "first_round_reaching_fraction_max": float(np.max(first_values)),
            "best_test_ap_mean": best_mean,
            "best_test_ap_std": best_std,
            "late_mean_test_ap_mean": late_mean_mean,
            "late_mean_test_ap_std": late_mean_std,
            "late_test_ap_slope_per_round_mean": slope_mean,
            "late_test_ap_slope_per_round_std": slope_std,
            "late_test_ap_slope_per_round_min": float(np.min(slope_values)),
            "late_test_ap_slope_per_round_max": float(np.max(slope_values)),
            "late_fitted_ap_change_mean": fitted_mean,
            "late_fitted_ap_change_std": fitted_std,
            "complete_late_windows": int(
                sum(bool(row["late_window_complete"]) for row in group)
            ),
        })
    return aggregates


def process_run(
    result_root: Path,
    strategy: str,
    n_clients: int,
    run_tag: int,
    test_loader: DataLoader,
    in_dim: int,
    device: torch.device,
    ap_fraction: float,
    late_rounds: int,
    allow_incomplete_final_window: bool,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    strategy_dir = scaling_strategy_dir(result_root, strategy, n_clients)
    models, run_dirs = discover_run_models(strategy_dir, run_tag)
    if not models:
        print(f"   ⚠️  {n_clients} clients, run {run_tag}: no checkpoints found")
        return [], None

    print(
        f"   Run {run_tag}: {len(models)} checkpoints in "
        + ", ".join(path.name for path in run_dirs)
    )
    round_rows: List[Dict[str, Any]] = []
    for round_number, checkpoint in models:
        model = load_model(checkpoint, in_dim, device)
        test_ap, n_samples, n_positive, n_negative = evaluate_test_ap(
            model, test_loader, device
        )
        round_rows.append({
            "strategy": strategy,
            "scaling_point": n_clients,
            "run": run_tag,
            "round": round_number,
            "test_ap": test_ap,
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "checkpoint": str(checkpoint),
        })
        print(f"      round {round_number:>3}: test AP={test_ap:.6f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_run(
        round_rows,
        ap_fraction,
        late_rounds,
        allow_incomplete_final_window,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("RETROSPECTIVE SCAFFOLD TEST-SET TRAINING-DYNAMICS EVALUATION")
    print(f"Strategy       : {args.strategy}")
    print(f"Scaling points : {args.scaling_points}")
    print(f"Runs           : {args.runs}")
    print(f"AP fraction    : {args.ap_fraction:.4f}")
    print(f"Late window    : final {args.late_rounds} communication rounds")
    print(f"Device         : {device}")
    print(f"Output         : {args.output_dir}")
    print("=" * 88)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_loader, test_info = load_centralized_test(
        args.data_parquet, args.norm_stats, args.batch_size, args.in_dim
    )
    test_info.update({
        "strategy": args.strategy,
        "ap_fraction": float(args.ap_fraction),
        "late_rounds": int(args.late_rounds),
        "scaling_points_requested": list(args.scaling_points),
        "runs_requested": int(args.runs),
        "device": str(device),
    })
    write_json(args.output_dir / "test_set_info.json", test_info)

    print(
        f"Test set loaded: n={test_info['n_samples']}, "
        f"positive={test_info['n_positive']}, negative={test_info['n_negative']}, "
        f"prevalence={test_info['positive_prevalence']:.4f}"
    )

    all_round_rows: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []
    missing_runs: List[Dict[str, int]] = []

    for n_clients in args.scaling_points:
        strategy_dir = scaling_strategy_dir(args.result_root, args.strategy, n_clients)
        if not strategy_dir.is_dir():
            print(f"\n⚠️  Skip {n_clients} clients: missing {strategy_dir}")
            continue
        print("\n" + "-" * 88)
        print(f"{n_clients} clients")
        print("-" * 88)
        for run_tag in range(1, args.runs + 1):
            round_rows, summary = process_run(
                args.result_root,
                args.strategy,
                n_clients,
                run_tag,
                test_loader,
                args.in_dim,
                device,
                args.ap_fraction,
                args.late_rounds,
                args.allow_incomplete_final_window,
            )
            all_round_rows.extend(round_rows)
            if summary is None:
                missing_runs.append({"scaling_point": n_clients, "run": run_tag})
            else:
                run_summaries.append(summary)

    all_round_rows.sort(
        key=lambda row: (int(row["scaling_point"]), int(row["run"]), int(row["round"]))
    )
    run_summaries.sort(
        key=lambda row: (int(row["scaling_point"]), int(row["run"]))
    )
    aggregates = aggregate_run_summaries(run_summaries, args.strategy)

    round_path = args.output_dir / "all_round_test_ap.csv"
    run_path = args.output_dir / "training_dynamics_by_run.csv"
    aggregate_path = args.output_dir / "training_dynamics_aggregate.csv"
    summary_path = args.output_dir / "training_dynamics_summary.json"

    write_csv(round_path, all_round_rows, ROUND_RESULT_FIELDS)
    write_csv(run_path, run_summaries, RUN_SUMMARY_FIELDS)
    write_csv(aggregate_path, aggregates, AGGREGATE_FIELDS)
    write_json(summary_path, {
        "strategy": args.strategy,
        "evaluation_set": "centralized_test",
        "analysis_type": "retrospective_training_dynamics",
        "training_speed_definition": (
            "First evaluated round with test AP >= "
            f"{args.ap_fraction:.6f} * best observed test AP in the run."
        ),
        "late_training_definition": (
            f"Mean test AP and linear AP slope over the exact final "
            f"{args.late_rounds} communication rounds."
        ),
        "test_set": test_info,
        "scaling_points_evaluated": sorted(
            {int(row["scaling_point"]) for row in run_summaries}
        ),
        "n_run_summaries": len(run_summaries),
        "missing_runs": missing_runs,
        "round_results_file": str(round_path),
        "run_summary_file": str(run_path),
        "aggregate_file": str(aggregate_path),
        "run_summaries": run_summaries,
        "aggregate_over_runs": aggregates,
    })

    print("\n" + "=" * 88)
    print("Finished.")
    print(f"Round-level AP trajectories : {round_path}")
    print(f"Run-level dynamics          : {run_path}")
    print(f"Scaling-point aggregates    : {aggregate_path}")
    print(f"JSON summary                : {summary_path}")
    if missing_runs:
        print(f"Warning: missing runs: {missing_runs}")
    print("=" * 88)


if __name__ == "__main__":
    main()
