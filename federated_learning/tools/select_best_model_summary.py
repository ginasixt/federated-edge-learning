#!/usr/bin/env python3

"""Select the best rounds per strategy and generate comparison plots.

This script scans a result directory containing folders like
``all_rounds_FedAdam_1/`` and ``all_rounds_FedProx_2/``. For every
``round_*_run_*.json`` file it reconstructs ROC-AUC and AUPRC from the saved
threshold curves, selects the best threshold with a non-medical ranking, then
selects the best round per run and the best run per strategy.

The selection policy intentionally does not use Net Benefit. That keeps the
choice suitable for later comparisons across different client sizes, which is
important for the scalability study.

Outputs:
  - CSV/JSON summaries for all rounds and selected best runs
  - Within-strategy plots for the selected best run
  - Cross-strategy overlays comparing the selected best runs
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "figure.figsize": (12, 6),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)

ROUND_RE = re.compile(r"round_(\d+)_run_(\d+)\.json$")
RUN_RE = re.compile(r"all_rounds_(?P<strategy>.+)_(?P<run>\d+)$")

DEFAULT_STRATEGIES = ["FedAdam", "Scaffold", "FedProx"]
COLORS = {"FedAdam": "#16a34a", "Scaffold": "#2563eb", "FedProx": "#dc2626"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_LABELS = {
    "default": "Default (AUPRC/AUC, thr via MCC)",
    "balanced_mcc": "Balanced Accuracy + MCC (min recall/spec gap)",
    "roc": "Best ROC-AUC round",
    "balanced": "Balanced Accuracy",
    "recspec_mean": "Mean(Recall,Specificity)",
    "recspec_strict": "Recall>=0.7 & Alerts<=500",
    "balanced_strict": "Balanced Accuracy (rec>=0.7 & alerts<=500)",
    "mcc": "MCC (Confusion-Matrix)",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_savefig(path: Path) -> None:
    safe_mkdir(path.parent)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def load_json(path: Path) -> dict:
    with path.open("r") as fh:
        return json.load(fh)


def parse_run_dir(run_dir: Path) -> Optional[Tuple[str, int]]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return None
    return match.group("strategy"), int(match.group("run"))


def compute_stats(raw: dict) -> dict:
    tp = float(raw.get("tp", 0.0))
    fp = float(raw.get("fp", 0.0))
    tn = float(raw.get("tn", 0.0))
    fn = float(raw.get("fn", 0.0))

    recall_denom = tp + fn
    spec_denom = tn + fp
    ppv_denom = tp + fp
    npv_denom = tn + fn
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    recall = tp / recall_denom if recall_denom else 0.0
    spec = tn / spec_denom if spec_denom else 0.0
    fpr = fp / spec_denom if spec_denom else 0.0
    ppv = tp / ppv_denom if ppv_denom else 0.0
    npv = tn / npv_denom if npv_denom else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    balanced_accuracy = 0.5 * (recall + spec)
    youden = recall - fpr
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0
    total = tp + fp + tn + fn
    alerts_per_1000 = ((tp + fp) / total) * 1000.0 if total else 0.0
    prevalence = ((tp + fn) / total) if total else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": recall,
        "recall": recall,
        "fpr": fpr,
        "spec": spec,
        "ppv": ppv,
        "precision": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "youden": youden,
        "mcc": mcc,
        "alerts_per_1000": alerts_per_1000,
        "prevalence": prevalence,
    }


def threshold_frame(round_data: dict) -> pd.DataFrame:
    rows = []
    for raw in round_data.get("metrics", {}).get("all_thresholds", []):
        row = compute_stats(raw)
        row["threshold"] = float(raw.get("threshold", np.nan))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("threshold").reset_index(drop=True)


def roc_auc(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    pts = df[["fpr", "recall"]].dropna().sort_values("fpr")
    if len(pts) < 2:
        return float("nan")
    x = np.concatenate(([0.0], pts["fpr"].to_numpy(dtype=float), [1.0]))
    y = np.concatenate(([0.0], pts["recall"].to_numpy(dtype=float), [1.0]))
    return float(sklearn_auc(x, y))


def pr_auc(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    pts = df[["recall", "precision", "prevalence"]].dropna().sort_values("recall")
    if len(pts) < 2:
        return float("nan")
    prevalence = float(pts["prevalence"].iloc[0])
    x = np.concatenate(([0.0], pts["recall"].to_numpy(dtype=float), [1.0]))
    y = np.concatenate(([1.0], pts["precision"].to_numpy(dtype=float), [prevalence]))
    return float(sklearn_auc(x, y))


def select_threshold(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    work = df.copy()
    work["threshold"] = pd.to_numeric(work["threshold"], errors="coerce")
    work = work.dropna(subset=["threshold"])
    if work.empty:
        return {}
    # General-purpose threshold ranking, no Net Benefit.
    work = work.sort_values(
        by=["mcc", "balanced_accuracy", "youden", "recall", "spec", "alerts_per_1000", "threshold"],
        ascending=[False, False, False, False, False, True, False],
    )
    return work.iloc[0].to_dict()


def select_threshold_by_policy(df: pd.DataFrame, policy: str) -> dict:
    if df.empty:
        return {}

    work = df.copy()
    work["threshold"] = pd.to_numeric(work["threshold"], errors="coerce")
    work = work.dropna(subset=["threshold"])
    if work.empty:
        return {}

    work["rec_spec_gap"] = (work["recall"] - work["spec"]).abs()

    if policy == "default":
        return select_threshold(work)
    if policy == "balanced_mcc":
        ordered = work.sort_values(
            by=["balanced_accuracy", "mcc", "rec_spec_gap", "youden", "alerts_per_1000", "threshold"],
            ascending=[False, False, True, False, True, False],
        )
        return ordered.iloc[0].to_dict()
    if policy == "roc":
        ordered = work.sort_values(
            by=["balanced_accuracy", "youden", "mcc", "recall", "spec", "alerts_per_1000", "threshold"],
            ascending=[False, False, False, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()
    if policy == "balanced":
        ordered = work.sort_values(
            by=["balanced_accuracy", "rec_spec_gap", "mcc", "youden", "alerts_per_1000", "threshold"],
            ascending=[False, True, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()
    if policy == "recspec_mean":
        # Mean(recall, specificity) equals balanced_accuracy; keep as explicit policy for reporting.
        ordered = work.sort_values(
            by=["balanced_accuracy", "rec_spec_gap", "youden", "mcc", "alerts_per_1000", "threshold"],
            ascending=[False, True, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()
    if policy == "mcc":
        ordered = work.sort_values(
            by=["mcc", "balanced_accuracy", "youden", "alerts_per_1000", "threshold"],
            ascending=[False, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()
    if policy == "recspec_strict":
        constrained = work[(work["recall"] >= 0.7) & (work["alerts_per_1000"] <= 500.0)].copy()
        if constrained.empty:
            return {}
        ordered = constrained.sort_values(
            by=["balanced_accuracy", "rec_spec_gap", "mcc", "youden", "alerts_per_1000", "threshold"],
            ascending=[False, True, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()

    # Strict balanced-accuracy policy: apply hard operational filters first,
    # then select by highest balanced_accuracy.
    if policy == "balanced_strict":
        constrained = work[(work["recall"] >= 0.7) & (work["alerts_per_1000"] <= 500.0)].copy()
        if constrained.empty:
            return {}
        ordered = constrained.sort_values(
            by=["balanced_accuracy", "rec_spec_gap", "mcc", "youden", "alerts_per_1000", "threshold"],
            ascending=[False, True, False, False, True, False],
        )
        return ordered.iloc[0].to_dict()

    raise ValueError(f"Unknown policy: {policy}")


def resolve_checkpoint_path(round_data: dict, json_path: Path, round_num: int) -> str:
    candidates: List[Path] = []
    ref = round_data.get("model_checkpoint")
    if ref:
        ref_str = str(ref)
        candidates.append((PROJECT_ROOT / ref_str).resolve())
        candidates.append(Path(ref_str).expanduser().resolve())
        if "/all_rounds/" in ref_str:
            # Some outputs store a generic all_rounds path although checkpoints are in the run-specific folder.
            tail = Path(ref_str).name
            candidates.append((json_path.parent / tail).resolve())
    candidates.append((json_path.parent / f"model_round_{round_num}.pt").resolve())

    # Allow common variant filenames in run folder.
    candidates.extend(sorted(json_path.parent.glob(f"model_round_{round_num}*.pt")))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0]) if candidates else str((json_path.parent / f"model_round_{round_num}.pt").resolve())


def round_sort_key(row: pd.Series) -> Tuple:
    return (
        float(row["selected_balanced_accuracy"]),
        float(row["selected_mcc"]),
        -abs(float(row["selected_recall"]) - float(row["selected_spec"])),
        float(row["selected_youden"]),
        float(row["selected_recall"]),
        float(row["selected_spec"]),
        -float(row["selected_alerts_per_1000"]),
        int(row["round"]),
    )


def run_sort_key(row: pd.Series) -> Tuple:
    return (
        float(row["best_round_auprc"]),
        float(row["best_round_auc"]),
        float(row["best_round_mcc"]),
        -float(row["auprc_std_tail"]),
        -float(row["mcc_std_tail"]),
        int(row["best_round"]),
    )


def summarize_tail(df: pd.DataFrame, window: int) -> dict:
    tail = df.sort_values("round").tail(min(window, len(df)))
    return {
        "dense_rounds": tail["round"].astype(int).tolist(),
        "auprc_std_tail": float(tail["auprc"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "auc_std_tail": float(tail["auc"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "mcc_std_tail": float(tail["selected_mcc"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "recall_std_tail": float(tail["selected_recall"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "spec_std_tail": float(tail["selected_spec"].std(ddof=1)) if len(tail) > 1 else 0.0,
    }


def load_all_runs(base_dir: Path) -> Tuple[Dict[str, Dict[int, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    runs: Dict[str, Dict[int, pd.DataFrame]] = {}
    round_frames: List[pd.DataFrame] = []
    run_rows: List[dict] = []

    for run_dir in sorted(base_dir.glob("all_rounds_*")):
        parsed = parse_run_dir(run_dir)
        if not parsed:
            continue
        strategy, run_num = parsed

        rows: List[dict] = []
        for json_file in sorted(run_dir.glob("round_*_run_*.json")):
            match = ROUND_RE.search(json_file.name)
            if not match:
                continue
            round_num = int(match.group(1))
            data = load_json(json_file)
            df = threshold_frame(data)
            if df.empty:
                continue

            auc_val = roc_auc(df)
            auprc_val = pr_auc(df)
            thr = select_threshold_by_policy(df, "balanced_mcc")

            row = {
                "strategy": strategy,
                "run": run_num,
                "round": round_num,
                "json_path": str(json_file),
                "auc": auc_val,
                "auprc": auprc_val,
                "selected_threshold": float(thr.get("threshold", np.nan)),
                "selected_mcc": float(thr.get("mcc", np.nan)),
                "selected_balanced_accuracy": float(thr.get("balanced_accuracy", np.nan)),
                "selected_youden": float(thr.get("youden", np.nan)),
                "selected_recall": float(thr.get("recall", np.nan)),
                "selected_spec": float(thr.get("spec", np.nan)),
                "selected_ppv": float(thr.get("ppv", np.nan)),
                "selected_npv": float(thr.get("npv", np.nan)),
                "selected_f1": float(thr.get("f1", np.nan)),
                "selected_alerts_per_1000": float(thr.get("alerts_per_1000", np.nan)),
            }
            row["round_key"] = round_sort_key(pd.Series(row))
            rows.append(row)

        if not rows:
            continue

        run_df = pd.DataFrame(rows).sort_values("round").reset_index(drop=True)
        runs.setdefault(strategy, {})[run_num] = run_df
        round_frames.append(run_df)

        best_idx = run_df["round_key"].tolist().index(max(run_df["round_key"].tolist()))
        best_row = run_df.iloc[best_idx].to_dict()
        tail = summarize_tail(run_df, window=10)
        run_rows.append(
            {
                "strategy": strategy,
                "run": run_num,
                "rounds": int(len(run_df)),
                "round_min": int(run_df["round"].min()),
                "round_max": int(run_df["round"].max()),
                "best_round": int(best_row["round"]),
                "best_round_auprc": float(best_row["auprc"]),
                "best_round_auc": float(best_row["auc"]),
                "best_round_mcc": float(best_row["selected_mcc"]),
                "best_round_balanced_accuracy": float(best_row["selected_balanced_accuracy"]),
                "best_round_youden": float(best_row["selected_youden"]),
                "best_round_recall": float(best_row["selected_recall"]),
                "best_round_spec": float(best_row["selected_spec"]),
                "best_round_threshold": float(best_row["selected_threshold"]),
                "best_round_ppv": float(best_row["selected_ppv"]),
                "best_round_npv": float(best_row["selected_npv"]),
                "best_round_f1": float(best_row["selected_f1"]),
                "best_round_alerts_per_1000": float(best_row["selected_alerts_per_1000"]),
                **tail,
                "best_round_key": best_row["round_key"],
            }
        )

    round_df = pd.concat(round_frames, ignore_index=True) if round_frames else pd.DataFrame()
    run_df = pd.DataFrame(run_rows)
    return runs, round_df, run_df


def compute_stability_score(row: pd.Series) -> float:
    """Compute a combined stability score (lower is better).
    
    Combines normalized volatility metrics to assess model consistency.
    A lower score indicates more stable performance across rounds.
    """
    metrics = [
        row.get("mcc_std_tail", 0.0),
        row.get("recall_std_tail", 0.0),
        row.get("spec_std_tail", 0.0),
        row.get("auprc_std_tail", 0.0),
        row.get("auc_std_tail", 0.0),
    ]
    valid_metrics = [m for m in metrics if pd.notna(m) and m > 0.0]
    if not valid_metrics:
        return 0.0
    return float(np.mean(valid_metrics))


def choose_best_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    """Select the best run per strategy, balancing performance and stability.
    
    Prioritizes runs with:
    1. Highest balanced accuracy
    2. Highest MCC
    3. Smallest recall-specificity gap
    4. Lowest combined volatility (most stable)
    5. Best round number (prefer earlier convergence)
    """
    if run_df.empty:
        return run_df
    
    # Add stability score for each run
    run_df = run_df.copy()
    run_df["stability_score"] = run_df.apply(compute_stability_score, axis=1)
    
    chosen = []
    for strategy, group in run_df.groupby("strategy", sort=True):
        ordered = group.sort_values(
            by=[
                "best_round_balanced_accuracy",
                "best_round_mcc",
                "best_round_youden",
                "best_round_recall",
                "best_round_spec",
                "stability_score",  # Lower is better (more stable)
                "mcc_std_tail",      # Explicit MCC stability as tiebreaker
                "best_round",
            ],
            ascending=[False, False, False, False, False, True, True, False],
        ).reset_index(drop=True)
        best_run = ordered.iloc[0].copy()
        best_run["stability_metrics"] = {
            "mcc_volatility": float(best_run.get("mcc_std_tail", 0.0)),
            "recall_volatility": float(best_run.get("recall_std_tail", 0.0)),
            "spec_volatility": float(best_run.get("spec_std_tail", 0.0)),
            "auprc_volatility": float(best_run.get("auprc_std_tail", 0.0)),
            "auc_volatility": float(best_run.get("auc_std_tail", 0.0)),
            "combined_stability_score": float(best_run.get("stability_score", 0.0)),
        }
        chosen.append(best_run)
    return pd.DataFrame(chosen).reset_index(drop=True)


def roc_curve_from_json(json_path: Path) -> pd.DataFrame:
    data = load_json(json_path)
    return threshold_frame(data)


def plot_learning_curves(strategy: str, strategy_runs: Dict[int, pd.DataFrame], selected_run: int, selected_round: int, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
    dense_end = max(df["round"].max() for df in strategy_runs.values())
    dense_start = dense_end - 9 if dense_end >= 10 else dense_end
    metric_specs = [("auprc", "AUPRC"), ("auc", "ROC-AUC"), ("selected_mcc", "MCC")]

    for ax, (metric, label) in zip(axes, metric_specs):
        ax.axvspan(dense_start, dense_end, color="orange", alpha=0.08, label="dense tail")
        for run_num, df in strategy_runs.items():
            color = "#111827" if int(run_num) == int(selected_run) else "#94a3b8"
            alpha = 0.95 if int(run_num) == int(selected_run) else 0.35
            lw = 2.6 if int(run_num) == int(selected_run) else 1.3
            ax.plot(df["round"], df[metric], color=color, alpha=alpha, linewidth=lw)
            if int(run_num) == int(selected_run):
                best = df[df["round"] == int(selected_round)].iloc[0]
                ax.scatter([best["round"]], [best[metric]], color="#111827", s=80, zorder=5)
                ax.axvline(best["round"], color="#111827", linestyle="--", linewidth=1.0)
                ax.annotate(f"Best R{int(best['round'])}", xy=(best["round"], best[metric]), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    axes[0].set_ylabel("Score")
    axes[0].legend(["dense tail", f"selected run R{selected_run}"], fontsize=8, loc="lower right")
    fig.suptitle(f"Within-strategy learning curves – {strategy}", fontweight="bold")
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_learning_curves.png")


def plot_within_strategy_overview(strategy: str, strategy_runs: Dict[int, pd.DataFrame], selected_run: int, selected_round: int, outdir: Path) -> None:
    """Create one compact figure that compares runs within a strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    dense_end = max(df["round"].max() for df in strategy_runs.values())
    dense_start = dense_end - 9 if dense_end >= 10 else dense_end

    metric_axes = [
        (axes[0, 0], "auprc", "AUPRC"),
        (axes[0, 1], "auc", "ROC-AUC"),
        (axes[1, 0], "selected_mcc", "MCC @ selected threshold"),
    ]

    for ax, metric, title in metric_axes:
        ax.axvspan(dense_start, dense_end, color="#f59e0b", alpha=0.08)
        for run_num, df in sorted(strategy_runs.items()):
            is_selected = int(run_num) == int(selected_run)
            color = "#111827" if is_selected else "#94a3b8"
            alpha = 0.95 if is_selected else 0.35
            lw = 2.6 if is_selected else 1.3
            ax.plot(df["round"], df[metric], color=color, alpha=alpha, linewidth=lw)
            if is_selected:
                best = df[df["round"] == int(selected_round)].iloc[0]
                ax.scatter([best["round"]], [best[metric]], color="#111827", s=90, zorder=5)
                ax.axvline(best["round"], color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_xlabel("Round")

    balance_ax = axes[1, 1]
    for run_num, df in sorted(strategy_runs.items()):
        is_selected = int(run_num) == int(selected_run)
        color = "#111827" if is_selected else "#94a3b8"
        alpha = 0.95 if is_selected else 0.3
        lw = 2.4 if is_selected else 1.2
        balance_ax.plot(df["round"], df["selected_recall"], color=color, alpha=alpha, linewidth=lw, linestyle="-", label=f"Run {run_num} recall" if is_selected else None)
        balance_ax.plot(df["round"], df["selected_spec"], color=color, alpha=alpha, linewidth=lw, linestyle=":", label=f"Run {run_num} specificity" if is_selected else None)

        if is_selected:
            best = df[df["round"] == int(selected_round)].iloc[0]
            balance_ax.scatter([best["round"]], [best["selected_recall"]], color="#111827", s=80, zorder=5)
            balance_ax.scatter([best["round"]], [best["selected_spec"]], color="#111827", s=80, marker="s", zorder=5)
            balance_ax.axvline(best["round"], color="#111827", linestyle="--", linewidth=1.0)
            balance_ax.annotate(
                f"Best R{int(best['round'])}\nrec={best['selected_recall']:.3f}, spec={best['selected_spec']:.3f}",
                xy=(best["round"], max(best["selected_recall"], best["selected_spec"])),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=8,
            )

    balance_ax.set_title("Recall vs Specificity over rounds (@ selected threshold)")
    balance_ax.set_xlabel("Round")
    balance_ax.set_ylabel("Score")
    balance_ax.set_ylim(0, 1)
    balance_ax.grid(alpha=0.25)
    balance_ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    balance_ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Within-strategy overview – {strategy}", fontweight="bold")
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_within_strategy_overview.png")


def choose_strategy_candidate(strategy: str, strategy_runs: Dict[int, pd.DataFrame], policy: str) -> dict:
    candidates: List[dict] = []
    for run_num, run_df in strategy_runs.items():
        for _, row in run_df.iterrows():
            round_num = int(row["round"])
            json_path = Path(row["json_path"])
            round_data = load_json(json_path)
            curve = threshold_frame(round_data)
            if curve.empty:
                continue

            thr = select_threshold_by_policy(curve, policy)
            if not thr:
                continue

            rec = float(thr.get("recall", np.nan))
            spec = float(thr.get("spec", np.nan))
            bal = float(thr.get("balanced_accuracy", np.nan))
            mcc = float(thr.get("mcc", np.nan))
            auc_val = float(roc_auc(curve))
            auprc_val = float(pr_auc(curve))
            gap = abs(rec - spec)

            item = {
                "strategy": strategy,
                "policy": policy,
                "run": int(run_num),
                "round": round_num,
                "json_path": str(json_path),
                "checkpoint_path": resolve_checkpoint_path(round_data, json_path, round_num),
                "threshold": float(thr.get("threshold", np.nan)),
                "auc": auc_val,
                "auprc": auprc_val,
                "mcc": mcc,
                "balanced_accuracy": bal,
                "youden": float(thr.get("youden", np.nan)),
                "recall": rec,
                "spec": spec,
                "precision": float(thr.get("precision", np.nan)),
                "alerts_per_1000": float(thr.get("alerts_per_1000", np.nan)),
                "tp": float(thr.get("tp", np.nan)),
                "fp": float(thr.get("fp", np.nan)),
                "tn": float(thr.get("tn", np.nan)),
                "fn": float(thr.get("fn", np.nan)),
                "rec_spec_gap": gap,
                "round_data": round_data,
            }

            if policy == "default":
                key = (
                    item["auprc"],
                    item["auc"],
                    item["mcc"],
                    item["balanced_accuracy"],
                    item["recall"],
                    item["spec"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "balanced_mcc":
                key = (
                    item["balanced_accuracy"],
                    item["mcc"],
                    -item["rec_spec_gap"],
                    item["youden"],
                    item["auprc"],
                    item["auc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "roc":
                key = (
                    item["auc"],
                    item["balanced_accuracy"],
                    item["youden"],
                    item["mcc"],
                    item["auprc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "balanced":
                key = (
                    item["balanced_accuracy"],
                    -item["rec_spec_gap"],
                    item["mcc"],
                    item["auprc"],
                    item["auc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "recspec_mean":
                key = (
                    item["balanced_accuracy"],
                    -item["rec_spec_gap"],
                    item["auc"],
                    item["auprc"],
                    item["mcc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "mcc":
                key = (
                    item["mcc"],
                    item["balanced_accuracy"],
                    item["auc"],
                    item["auprc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "recspec_strict":
                key = (
                    item["balanced_accuracy"],
                    -item["rec_spec_gap"],
                    item["recall"],
                    item["spec"],
                    item["mcc"],
                    item["auc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            elif policy == "balanced_strict":
                key = (
                    item["balanced_accuracy"],
                    -item["rec_spec_gap"],
                    item["mcc"],
                    item["auc"],
                    item["auprc"],
                    -item["alerts_per_1000"],
                    item["round"],
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")

            item["policy_key"] = key
            candidates.append(item)

    if not candidates:
        raise ValueError(f"No candidate rounds for strategy={strategy}, policy={policy}")

    return max(candidates, key=lambda x: x["policy_key"])


def save_best_model_bundle(strategy: str, bundle_name: str, candidate: dict, bundle_dir: Path) -> None:
    safe_mkdir(bundle_dir)

    checkpoint_src = Path(candidate["checkpoint_path"])
    if not checkpoint_src.exists():
        raise FileNotFoundError(f"Checkpoint not found for {strategy} {bundle_name}: {checkpoint_src}")

    checkpoint_dst = bundle_dir / f"model_round_{int(candidate['round'])}.pt"
    shutil.copy2(checkpoint_src, checkpoint_dst)

    payload = {
        "strategy": strategy,
        "bundle_name": bundle_name,
        "policy": candidate.get("policy"),
        "run": int(candidate["run"]),
        "round": int(candidate["round"]),
        "threshold": float(candidate["threshold"]),
        "auc": float(candidate["auc"]),
        "auprc": float(candidate["auprc"]),
        "mcc": float(candidate["mcc"]),
        "balanced_accuracy": float(candidate["balanced_accuracy"]),
        "recall": float(candidate["recall"]),
        "spec": float(candidate["spec"]),
        "precision": float(candidate["precision"]),
        "alerts_per_1000": float(candidate["alerts_per_1000"]),
        "tp": float(candidate["tp"]),
        "fp": float(candidate["fp"]),
        "tn": float(candidate["tn"]),
        "fn": float(candidate["fn"]),
        "rec_spec_gap": float(candidate["rec_spec_gap"]),
        "checkpoint_source": candidate["checkpoint_path"],
        "checkpoint_saved_as": str(checkpoint_dst),
        "source_round_json": candidate["json_path"],
        "selection_note": "ROC curve is reconstructed from a finite threshold grid; points are discrete.",
    }
    with (bundle_dir / "run_1.json").open("w") as fh:
        json.dump(payload, fh, indent=2)

    description = [
        f"# {strategy} / {bundle_name}",
        "",
        f"- Run: {candidate['run']}",
        f"- Round: {candidate['round']}",
        f"- Threshold: {candidate['threshold']:.2f}",
        f"- AUC: {candidate['auc']:.4f}",
        f"- AUPRC: {candidate['auprc']:.4f}",
        f"- Recall: {candidate['recall']:.4f}",
        f"- Specificity: {candidate['spec']:.4f}",
        f"- Balanced Accuracy: {candidate['balanced_accuracy']:.4f}",
        f"- MCC: {candidate['mcc']:.4f}",
        f"- False alarms / 1000: {candidate['alerts_per_1000']:.1f}",
        f"- Checkpoint: {checkpoint_dst}",
        "",
        "ROC curves are reconstructed from a finite threshold grid, so the curve is discrete rather than fully continuous.",
    ]
    (bundle_dir / "selection_description.md").write_text("\n".join(description))


def export_best_model_bundles(strategy: str, strategy_runs: Dict[int, pd.DataFrame], outdir: Path) -> Dict[str, Optional[dict]]:
    bundles: Dict[str, Optional[dict]] = {}
    for bundle_name, policy in [
        ("best_BalancedMCC_model", "balanced_mcc"),
        ("best_MCC_model", "mcc"),
        ("best_ROC_model", "roc"),
        ("best_Balanced", "balanced_strict"),
    ]:
        try:
            candidate = choose_strategy_candidate(strategy, strategy_runs, policy)
        except ValueError:
            candidate = None

        bundles[bundle_name] = candidate
        bundle_dir = outdir / strategy / "best_models" / bundle_name
        safe_mkdir(bundle_dir)
        if candidate is None:
            (bundle_dir / "NO_CANDIDATE_FOUND.txt").write_text(
                "No round/threshold satisfied the requested criteria for this strategy."
            )
            continue

        save_best_model_bundle(strategy, bundle_name, candidate, bundle_dir)

    summary_lines = [f"# {strategy} best_models summary", ""]
    for bundle_name, candidate in bundles.items():
        summary_lines.append(f"## {bundle_name}")
        if candidate is None:
            summary_lines.append("- No candidate found")
        else:
            summary_lines.extend(
                [
                    f"- Run: {candidate['run']}",
                    f"- Round: {candidate['round']}",
                    f"- Threshold: {candidate['threshold']:.2f}",
                    f"- AUC: {candidate['auc']:.4f}",
                    f"- AUPRC: {candidate['auprc']:.4f}",
                    f"- Recall: {candidate['recall']:.4f}",
                    f"- Specificity: {candidate['spec']:.4f}",
                    f"- Balanced Accuracy: {candidate['balanced_accuracy']:.4f}",
                    f"- MCC: {candidate['mcc']:.4f}",
                    "",
                ]
            )
    (outdir / strategy / "best_models" / "summary.md").write_text("\n".join(summary_lines))
    return bundles


def export_best_models_per_run(strategy: str, strategy_runs: Dict[int, pd.DataFrame]) -> None:
    for run_num, run_df in strategy_runs.items():
        run_dir = Path(run_df.iloc[0]["json_path"]).parent
        bundles_dir = run_dir / "best_models"
        safe_mkdir(bundles_dir)

        run_bundles = {}
        for bundle_name, policy in [
            ("best_BalancedMCC_model", "balanced_mcc"),
            ("best_MCC_model", "mcc"),
            ("best_ROC_model", "roc"),
            ("best_Balanced", "balanced_strict"),
        ]:
            try:
                candidate = choose_strategy_candidate(strategy, {run_num: run_df}, policy)
            except ValueError:
                candidate = None

            run_bundles[bundle_name] = candidate
            bundle_dir = bundles_dir / bundle_name
            safe_mkdir(bundle_dir)
            if candidate is None:
                (bundle_dir / "NO_CANDIDATE_FOUND.txt").write_text(
                    "No round/threshold satisfied the requested criteria for this run."
                )
                continue

            save_best_model_bundle(strategy, bundle_name, candidate, bundle_dir)

        summary_lines = [f"# {strategy} run {run_num} best_models summary", ""]
        for bundle_name, candidate in run_bundles.items():
            summary_lines.append(f"## {bundle_name}")
            if candidate is None:
                summary_lines.append("- No candidate found")
            else:
                summary_lines.extend(
                    [
                        f"- Run: {candidate['run']}",
                        f"- Round: {candidate['round']}",
                        f"- Threshold: {candidate['threshold']:.2f}",
                        f"- AUC: {candidate['auc']:.4f}",
                        f"- AUPRC: {candidate['auprc']:.4f}",
                        f"- Recall: {candidate['recall']:.4f}",
                        f"- Specificity: {candidate['spec']:.4f}",
                        f"- Balanced Accuracy: {candidate['balanced_accuracy']:.4f}",
                        f"- MCC: {candidate['mcc']:.4f}",
                        "",
                    ]
                )
        (bundles_dir / "summary.md").write_text("\n".join(summary_lines))


def plot_selected_roc_with_description(strategy: str, candidate: dict, outdir: Path) -> None:
    curve = threshold_frame(load_json(Path(candidate["json_path"]))).sort_values("fpr")
    thr = float(candidate["threshold"])
    idx = (curve["threshold"] - thr).abs().idxmin()
    sel = curve.loc[idx]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.0)
    ax.plot(curve["fpr"], curve["recall"], color=COLORS.get(strategy, "#111827"), linewidth=2.5)
    ax.scatter(curve["fpr"], curve["recall"], color=COLORS.get(strategy, "#111827"), s=24, alpha=0.85)
    ax.scatter([sel["fpr"]], [sel["recall"]], marker="D", s=120, color="#111827", edgecolors="white", linewidths=1.2)
    ax.annotate(
        f"thr={thr:.2f}",
        xy=(sel["fpr"], sel["recall"]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(
        f"{strategy} – best round by {POLICY_LABELS.get(candidate.get('policy', 'default'), candidate.get('policy', 'default'))}\n"
        f"Run {candidate['run']} | Round {candidate['round']} | thr={candidate['threshold']:.2f} | AUC={candidate['auc']:.4f}",
        fontsize=11,
    )
    ax.grid(alpha=0.25)

    desc = (
        "ROC is reconstructed from a finite threshold grid (discrete points), "
        "therefore not a fully continuous ROC curve.\n"
        f"Selected point confusion matrix: TP={int(candidate['tp'])}, FP={int(candidate['fp'])}, "
        f"TN={int(candidate['tn'])}, FN={int(candidate['fn'])}."
    )
    fig.text(0.01, -0.03, desc, ha="left", va="top", fontsize=9)
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_best_selected_roc_with_description.png")


def plot_policy_roc_comparison_within_strategy(strategy: str, candidates: Dict[str, dict], outdir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.0)

    policy_colors = {
        "default": "#111827",
        "balanced": "#2563eb",
        "recspec_mean": "#16a34a",
        "mcc": "#dc2626",
    }

    for policy in ["default", "balanced", "recspec_mean", "mcc"]:
        cand = candidates[policy]
        curve = threshold_frame(load_json(Path(cand["json_path"]))).sort_values("fpr")
        thr = float(cand["threshold"])
        idx = (curve["threshold"] - thr).abs().idxmin()
        sel = curve.loc[idx]
        color = policy_colors[policy]
        label = (
            f"{POLICY_LABELS[policy]} | run={cand['run']} round={cand['round']} thr={cand['threshold']:.2f} "
            f"AUC={cand['auc']:.3f} bal={cand['balanced_accuracy']:.3f}"
        )
        ax.plot(curve["fpr"], curve["recall"], color=color, linewidth=2.2, label=label)
        ax.scatter(curve["fpr"], curve["recall"], color=color, s=22, alpha=0.8)
        ax.scatter([sel["fpr"]], [sel["recall"]], marker="D", s=95, color=color, edgecolors="white", linewidths=1.1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"{strategy} – within-strategy ROC comparison by selection policy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_within_strategy_policy_roc_comparison.png")


def plot_policy_confusion_matrices(strategy: str, candidates: Dict[str, dict], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    policies = ["default", "balanced", "recspec_mean", "mcc"]

    for ax, policy in zip(axes.flatten(), policies):
        cand = candidates[policy]
        cm = np.array([[cand["tn"], cand["fp"]], [cand["fn"], cand["tp"]]], dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title(
            f"{POLICY_LABELS[policy]}\nrun={cand['run']} round={cand['round']} thr={cand['threshold']:.2f}",
            fontsize=10,
        )
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{int(value)}", ha="center", va="center", color="#111827", fontsize=10)
        ax.text(
            0.5,
            -0.18,
            f"rec={cand['recall']:.3f} spec={cand['spec']:.3f} bal={cand['balanced_accuracy']:.3f} mcc={cand['mcc']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle(f"{strategy} – confusion matrix comparison across selection policies", fontweight="bold")
    safe_savefig(outdir / f"{strategy}_within_strategy_policy_confusion_matrices.png")


def export_strategy_policy_description(strategy: str, candidates: Dict[str, dict], outdir: Path, best_run_info: Optional[dict] = None) -> None:
    """Export detailed selection description with performance and stability metrics.\n    \n    Args:\n        strategy: Strategy name\n        candidates: Dict mapping policy names to selected candidate dicts\n        outdir: Output directory for markdown report\n        best_run_info: Optional dict with run-level stability metrics\n    \"\"\"\n    lines = [\n        f\"# {strategy} Selection Summary\",\n        \"\",\n        \"## Selection Methodology\",\n        \"Models are selected using a multi-criteria ranking that prioritizes:\",\n        \"1. **Performance**: Balanced accuracy, MCC, recall-specificity balance\",\n        \"2. **Stability**: Consistency of metrics across the last 10 training rounds\",\n        \"3. **Operational Constraints**: Acceptable alert rates and recall levels\",\n        \"\",\n        \"ROC curves are reconstructed from a finite threshold grid; points are discrete approximations.\",\n        \"\",\n    ]\n    \n    # Add stability metrics summary if available\n    if best_run_info:\n        lines.extend([\n            \"## Run Stability Profile\",\n            f\"- Selected Run: {best_run_info.get('run', 'N/A')}\",\n            f\"- Training Rounds: {best_run_info.get('rounds', 'N/A')}\",\n            \"\",\n            \"**Volatility Metrics (Last 10 Rounds)**\",\n            f\"- MCC Volatility (σ): {best_run_info.get('mcc_std_tail', 0.0):.4f}\",\n            f\"- Recall Volatility (σ): {best_run_info.get('recall_std_tail', 0.0):.4f}\",\n            f\"- Specificity Volatility (σ): {best_run_info.get('spec_std_tail', 0.0):.4f}\",\n            f\"- AUPRC Volatility (σ): {best_run_info.get('auprc_std_tail', 0.0):.4f}\",\n            f\"- AUC Volatility (σ): {best_run_info.get('auc_std_tail', 0.0):.4f}\",\n            f\"- **Stability Score**: {best_run_info.get('stability_score', 0.0):.4f} (lower is more stable)\",\n            \"\",\n        ])\n\n    for policy in [\"balanced_mcc\", \"default\", \"balanced\", \"recspec_mean\", \"mcc\"]:\n        if policy not in candidates:\n            continue\n        c = candidates[policy]\n        lines.extend(\n            [\n                f\"## {POLICY_LABELS.get(policy, policy)}\",\n                f\"- Run: {c['run']}\",\n                f\"- Round: {c['round']}\",\n                f\"- Threshold: {c['threshold']:.2f}\",\n                \"\",\n                \"**Performance Metrics**\",\n                f\"- AUC: {c['auc']:.4f}\",\n                f\"- AUPRC: {c['auprc']:.4f}\",\n                f\"- Recall: {c['recall']:.4f}\",\n                f\"- Specificity: {c['spec']:.4f}\",\n                f\"- Balanced Accuracy: {c['balanced_accuracy']:.4f}\",\n                f\"- MCC: {c['mcc']:.4f}\",\n                f\"- Youden Index: {c.get('youden', 0.0):.4f}\",\n                \"\",\n                \"**Confusion Matrix**\",\n                f\"- TP: {int(c['tp'])}, FP: {int(c['fp'])}, TN: {int(c['tn'])}, FN: {int(c['fn'])}\",\n                f\"- Alerts per 1000: {c.get('alerts_per_1000', 0.0):.2f}\",\n                \"\",\n            ]\n        )\n\n    (outdir / f\"{strategy}_selection_description.md\").write_text(\"\\n\".join(lines))


def plot_within_strategy_thresholds(strategy: str, selected_row: pd.Series, best_round_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    curve = best_round_df.sort_values("threshold").copy()
    thr = float(selected_row["best_round_threshold"])
    sel_idx = (curve["threshold"] - thr).abs().idxmin()
    sel = curve.loc[sel_idx]

    axes[0].plot(curve["spec"], curve["recall"], color=COLORS[strategy], linewidth=2.2)
    axes[0].scatter(curve["spec"], curve["recall"], s=24, color=COLORS[strategy])
    axes[0].scatter([sel["spec"]], [sel["recall"]], s=110, color="#111827", marker="D", edgecolors="white", linewidths=1.2)
    axes[0].annotate(f"thr={thr:.2f}", xy=(sel["spec"], sel["recall"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Specificity")
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall vs specificity")
    axes[0].grid(alpha=0.25)

    axes[1].plot(curve["alerts_per_1000"], curve["recall"], color="#16a34a", linewidth=2.2)
    axes[1].scatter(curve["alerts_per_1000"], curve["recall"], s=24, color="#16a34a")
    axes[1].scatter([sel["alerts_per_1000"]], [sel["recall"]], s=110, color="#111827", marker="D", edgecolors="white", linewidths=1.2)
    axes[1].annotate(f"thr={thr:.2f}", xy=(sel["alerts_per_1000"], sel["recall"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Alerts per 1000")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall vs alerts")
    axes[1].grid(alpha=0.25)

    axes[2].plot(curve["threshold"], curve["mcc"], label="MCC", color="#dc2626", linewidth=2.0)
    axes[2].plot(curve["threshold"], curve["balanced_accuracy"], label="Balanced accuracy", color="#7c3aed", linewidth=2.0)
    axes[2].plot(curve["threshold"], curve["youden"], label="Youden", color="#0f766e", linewidth=2.0)
    axes[2].axvline(thr, color="#111827", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("Threshold")
    axes[2].set_title("Threshold metrics")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"Within-strategy threshold analysis – {strategy} (R{int(selected_row['best_round'])})", fontweight="bold")
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_threshold_analysis.png")


def plot_within_strategy_roc_pr(strategy: str, selected_row: pd.Series, best_round_df: pd.DataFrame, outdir: Path) -> None:
    curve = best_round_df.sort_values("threshold").copy()
    roc_points = curve.sort_values("fpr")
    pr_points = curve.sort_values("recall")
    prevalence = float(curve["prevalence"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.1)
    axes[0].plot(roc_points["fpr"], roc_points["recall"], color=COLORS[strategy], linewidth=2.4)
    axes[0].scatter(roc_points["fpr"], roc_points["recall"], s=24, color=COLORS[strategy])
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUC={selected_row['best_round_auc']:.4f})")
    axes[0].grid(alpha=0.25)

    axes[1].axhline(prevalence, linestyle=":", color="gray", linewidth=1.1, label=f"prevalence={prevalence:.3f}")
    axes[1].plot(pr_points["recall"], pr_points["precision"], color=COLORS[strategy], linewidth=2.4)
    axes[1].scatter(pr_points["recall"], pr_points["precision"], s=24, color=COLORS[strategy])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR curve (AUPRC={selected_row['best_round_auprc']:.4f})")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Best round – {strategy} (R{int(selected_row['best_round'])})", fontweight="bold")
    fig.tight_layout()
    safe_savefig(outdir / f"{strategy}_roc_pr_best_round.png")


def plot_cross_strategy_overlay(best_runs: pd.DataFrame, runs: Dict[str, Dict[int, pd.DataFrame]], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for _, row in best_runs.iterrows():
        strategy = row["strategy"]
        run_num = int(row["run"])
        best_round = int(row["best_round"])
        round_path = Path(
            runs[strategy][run_num].loc[runs[strategy][run_num]["round"] == best_round, "json_path"].iloc[0]
        )
        curve = threshold_frame(load_json(round_path))
        color = COLORS.get(strategy, "#111827")

        roc_points = curve.sort_values("fpr")
        pr_points = curve.sort_values("recall")

        axes[0].plot(roc_points["fpr"], roc_points["recall"], color=color, linewidth=2.4, label=f"{strategy} (R{best_round})")
        axes[0].scatter(roc_points["fpr"], roc_points["recall"], s=24, color=color)
        axes[0].scatter([1 - float(row["best_round_spec"])], [float(row["best_round_recall"])], s=90, color=color, marker="D", edgecolors="white", linewidths=1.0)

        axes[1].plot(pr_points["recall"], pr_points["precision"], color=color, linewidth=2.4, label=f"{strategy} (AUPRC={row['best_round_auprc']:.4f})")
        axes[1].scatter(pr_points["recall"], pr_points["precision"], s=24, color=color)
        axes[1].scatter([float(row["best_round_recall"])], [float(row["best_round_ppv"])], s=90, color=color, marker="D", edgecolors="white", linewidths=1.0)

        axes[2].plot(curve["spec"], curve["recall"], color=color, linewidth=2.4, label=f"{strategy} (thr={row['best_round_threshold']:.2f})")
        axes[2].scatter(curve["spec"], curve["recall"], s=24, color=color)
        axes[2].scatter([float(row["best_round_spec"])], [float(row["best_round_recall"])], s=90, color=color, marker="D", edgecolors="white", linewidths=1.0)

    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC overlay")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR overlay")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].set_xlabel("Specificity")
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Recall vs specificity overlay")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Cross-strategy comparison on selected best runs", fontweight="bold")
    fig.tight_layout()
    safe_savefig(outdir / "cross_strategy_overlay.png")


def export_outputs(round_df: pd.DataFrame, run_df: pd.DataFrame, best_runs: pd.DataFrame, output_dir: Path, base_dir: Path) -> None:\n    safe_mkdir(output_dir)\n    round_df.to_csv(output_dir / \"all_rounds_summary.csv\", index=False)\n    run_df.to_csv(output_dir / \"all_runs_summary.csv\", index=False)\n    best_runs.to_csv(output_dir / \"best_runs_per_strategy.csv\", index=False)\n\n    payload = {\n        \"base_dir\": str(base_dir),\n        \"selection_methodology\": {\n            \"description\": \"Multi-criteria model selection combining performance and stability metrics\",\n            \"stability_metrics\": [\n                \"mcc_std_tail: Standard deviation of MCC across last 10 rounds\",\n                \"recall_std_tail: Standard deviation of recall across last 10 rounds\",\n                \"spec_std_tail: Standard deviation of specificity across last 10 rounds\",\n                \"auprc_std_tail: Standard deviation of AUPRC across last 10 rounds\",\n                \"auc_std_tail: Standard deviation of AUC across last 10 rounds\",\n                \"combined_stability_score: Mean of all volatility metrics (lower is better)\",\n            ],\n        },\n        \"selection_policy\": {\n            \"best_round_ranking\": [\"selected_balanced_accuracy\", \"selected_mcc\", \"-recall_spec_gap\", \"selected_youden\", \"selected_recall\", \"selected_spec\", \"selected_alerts_per_1000\", \"round\"],\n            \"best_threshold_ranking\": [\"balanced_accuracy\", \"mcc\", \"-recall_spec_gap\", \"youden\", \"alerts_per_1000\", \"threshold\"],\n            \"best_run_ranking\": [\"best_round_balanced_accuracy\", \"best_round_mcc\", \"best_round_youden\", \"best_round_recall\", \"best_round_spec\", \"stability_score\", \"mcc_std_tail\", \"best_round\"],\n            \"ranking_explanation\": \"Prioritizes high performance metrics first, then uses stability metrics to select most consistent models. Lower stability scores indicate more reliable performance.\",\n            \"primary_policy\": \"balanced_mcc\",\n            \"alternative_policies\": [\"balanced\", \"recspec_mean\", \"balanced_strict\", \"mcc\", \"roc\"],\n        },\n        \"best_runs\": [\n            {\n                **row.to_dict(),\n                \"stability_metrics\": {\n                    \"combined_score\": float(row.get(\"stability_score\", 0.0)),\n                    \"mcc_volatility\": float(row.get(\"mcc_std_tail\", 0.0)),\n                    \"recall_volatility\": float(row.get(\"recall_std_tail\", 0.0)),\n                    \"spec_volatility\": float(row.get(\"spec_std_tail\", 0.0)),\n                }\n            }\n            for _, row in best_runs.iterrows()\n        ],\n    }\n    with (output_dir / \"best_model_selection_summary.json\").open(\"w\") as fh:\n        json.dump(payload, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best rounds and generate comparison plots.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "result" / "splits_iid_scaling" / "splits_iid_16384_clients.json",
        help="Directory containing all_rounds_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "result" / "plots" / "model_selection_report",
        help="Directory for CSV/JSON outputs and plots.",
    )
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    args = parser.parse_args()

    runs, round_df, run_df = load_all_runs(args.base_dir)
    if round_df.empty or run_df.empty:
        raise SystemExit(f"No usable round files found under {args.base_dir}")

    # Keep only strategies that were found on disk.
    found_strategies = {strategy: strategy_runs for strategy, strategy_runs in runs.items() if strategy_runs}
    if args.strategies:
        found_strategies = {s: found_strategies[s] for s in args.strategies if s in found_strategies}
    if not found_strategies:
        raise SystemExit("No requested strategies found in the result directory.")

    run_df = run_df[run_df["strategy"].isin(found_strategies.keys())].reset_index(drop=True)
    round_df = round_df[round_df["strategy"].isin(found_strategies.keys())].reset_index(drop=True)

    best_runs = choose_best_runs(run_df)
    export_outputs(round_df, run_df, best_runs, args.output_dir, args.base_dir)

    for _, row in best_runs.iterrows():
        strategy = row["strategy"]
        run_num = int(row["run"])
        best_round = int(row["best_round"])
        strat_dir = args.output_dir / strategy
        safe_mkdir(strat_dir)
        strategy_runs = found_strategies[strategy]
        selected_run_df = strategy_runs[run_num]
        selected_round_df = threshold_frame(load_json(Path(selected_run_df.loc[selected_run_df["round"] == best_round, "json_path"].iloc[0])))

        plot_within_strategy_overview(strategy, strategy_runs, run_num, best_round, strat_dir)
        plot_learning_curves(strategy, strategy_runs, run_num, best_round, strat_dir)
        plot_within_strategy_thresholds(strategy, row, selected_round_df, strat_dir)
        plot_within_strategy_roc_pr(strategy, row, selected_round_df, strat_dir)

        candidates = {
            "balanced_mcc": choose_strategy_candidate(strategy, strategy_runs, "balanced_mcc"),
            "default": choose_strategy_candidate(strategy, strategy_runs, "default"),
            "balanced": choose_strategy_candidate(strategy, strategy_runs, "balanced"),
            "recspec_mean": choose_strategy_candidate(strategy, strategy_runs, "recspec_mean"),
            "mcc": choose_strategy_candidate(strategy, strategy_runs, "mcc"),
        }
        
        # Extract stability metrics from best run info
        best_run_stability_info = {
            "run": int(row["run"]),
            "rounds": int(row["rounds"]),
            "mcc_std_tail": float(row.get("mcc_std_tail", 0.0)),
            "recall_std_tail": float(row.get("recall_std_tail", 0.0)),
            "spec_std_tail": float(row.get("spec_std_tail", 0.0)),
            "auprc_std_tail": float(row.get("auprc_std_tail", 0.0)),
            "auc_std_tail": float(row.get("auc_std_tail", 0.0)),
            "stability_score": float(row.get("stability_score", 0.0)),
        }
        
        plot_selected_roc_with_description(strategy, candidates["balanced_mcc"], strat_dir)
        plot_policy_roc_comparison_within_strategy(strategy, candidates, strat_dir)
        plot_policy_confusion_matrices(strategy, candidates, strat_dir)
        export_strategy_policy_description(strategy, candidates, strat_dir, best_run_stability_info)
        export_best_model_bundles(strategy, strategy_runs, args.output_dir)
        export_best_models_per_run(strategy, strategy_runs)

    cross_dir = args.output_dir / "cross_strategy"
    safe_mkdir(cross_dir)
    plot_cross_strategy_overlay(best_runs, found_strategies, cross_dir)

    print("Best runs per strategy:")
    print(
        best_runs[
            [
                "strategy",
                "run",
                "best_round",
                "best_round_auprc",
                "best_round_auc",
                "best_round_mcc",
                "best_round_recall",
                "best_round_spec",
                "best_round_threshold",
                "auprc_std_tail",
                "mcc_std_tail",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()