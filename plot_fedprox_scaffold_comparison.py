#!/usr/bin/env python3
"""
Deep comparison plots for FedProx vs Scaffold - focusing on best runs.
Visualizes stability, convergence, and performance metrics for BA thesis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns


RESULT_DIR = Path("/home/bax9142/federated-edge-learning/result/splits_iid_scaling/splits_iid_16384_clients.json")
PLOT_OUT = Path("/home/bax9142/federated-edge-learning/result/plots/fedprox_scaffold_analysis")

ROUND_FILE_RE = re.compile(r"round_(\d+)_run_1\.json$")
STRATEGY_DIR_RE = re.compile(r"^all_rounds_(?P<family>[A-Za-z]+)_(?P<variant>\d+)$")

# Color scheme
COLORS = {
    "FedProx": "#dc2626",
    "FedProx_4": "#b91c1c",
    "FedProx_5": "#ef4444",
    "Scaffold": "#2563eb",
    "Scaffold_1": "#1d4ed8",
}

BEST_RUNS = {
    "FedProx": [4, 5],
    "Scaffold": [1],
}


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def parse_strategy_dir(path: Path) -> Tuple[str, int]:
    match = STRATEGY_DIR_RE.match(path.name)
    if not match:
        return path.name, 0
    return match.group("family"), int(match.group("variant"))


def get_round_files_for_strategy(family: str, variant: int) -> List[Tuple[int, Path]]:
    """Get all round files for a strategy run, sorted by round number."""
    strategy_dir = RESULT_DIR / f"all_rounds_{family}_{variant}"
    if not strategy_dir.exists():
        return []
    
    files = []
    for json_file in strategy_dir.glob("round_*_run_1.json"):
        match = ROUND_FILE_RE.search(json_file.name)
        if match:
            round_num = int(match.group(1))
            files.append((round_num, json_file))
    
    return sorted(files, key=lambda x: x[0])


def extract_metrics_from_round(round_data: dict) -> dict:
    """Extract key metrics from a round JSON."""
    metrics = round_data.get("metrics", {})
    all_thresholds = metrics.get("all_thresholds", [])
    
    if not all_thresholds:
        return None
    
    # Find best threshold by balanced accuracy
    best_threshold = None
    best_ba = -1
    
    for thr_data in all_thresholds:
        tp = float(thr_data.get("tp", 0))
        fp = float(thr_data.get("fp", 0))
        tn = float(thr_data.get("tn", 0))
        fn = float(thr_data.get("fn", 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ba = (recall + spec) / 2.0
        
        if ba > best_ba:
            best_ba = ba
            best_threshold = thr_data
    
    if not best_threshold:
        return None
    
    # Compute metrics
    tp = float(best_threshold.get("tp", 0))
    fp = float(best_threshold.get("fp", 0))
    tn = float(best_threshold.get("tn", 0))
    fn = float(best_threshold.get("fn", 0))
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    ba = (recall + spec) / 2.0
    
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0
    
    auc_val = compute_auc(all_thresholds)
    auprc_val = compute_auprc(all_thresholds)
    
    return {
        "threshold": float(best_threshold.get("threshold", 0)),
        "recall": recall,
        "spec": spec,
        "ppv": ppv,
        "balanced_accuracy": ba,
        "mcc": mcc,
        "auc": auc_val,
        "auprc": auprc_val,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_auc(thresholds: List[dict]) -> float:
    """Compute AUC from threshold data."""
    fpr_list = []
    recall_list = []
    
    for thr_data in thresholds:
        tp = float(thr_data.get("tp", 0))
        fp = float(thr_data.get("fp", 0))
        tn = float(thr_data.get("tn", 0))
        fn = float(thr_data.get("fn", 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        fpr_list.append(fpr)
        recall_list.append(recall)
    
    if len(fpr_list) < 2:
        return 0.0
    
    fpr_array = np.array(fpr_list)
    recall_array = np.array(recall_list)
    
    sorted_idx = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sorted_idx]
    recall_sorted = recall_array[sorted_idx]
    
    auc = np.trapz(recall_sorted, fpr_sorted)
    return float(np.clip(auc, 0, 1))


def compute_auprc(thresholds: List[dict]) -> float:
    """Compute AUPRC from threshold data."""
    precision_list = []
    recall_list = []
    
    for thr_data in thresholds:
        tp = float(thr_data.get("tp", 0))
        fp = float(thr_data.get("fp", 0))
        fn = float(thr_data.get("fn", 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    if len(recall_list) < 2:
        return 0.0
    
    recall_array = np.array(recall_list)
    precision_array = np.array(precision_list)
    
    sorted_idx = np.argsort(recall_array)
    recall_sorted = recall_array[sorted_idx]
    precision_sorted = precision_array[sorted_idx]
    
    auprc = np.trapz(precision_sorted, recall_sorted)
    return float(np.clip(auprc, 0, 1))


def collect_run_metrics(family: str, variant: int) -> pd.DataFrame:
    """Collect all metrics for a run across all rounds."""
    files = get_round_files_for_strategy(family, variant)
    rows = []
    
    for round_num, json_path in files:
        data = load_json(json_path)
        if not data:
            continue
        
        metrics = extract_metrics_from_round(data)
        if not metrics:
            continue
        
        row = {"round": round_num, **metrics}
        rows.append(row)
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_tail_volatility(df: pd.DataFrame, window: int = 10) -> dict:
    """Compute volatility metrics for the tail of the run."""
    tail = df.sort_values("round").tail(min(window, len(df)))
    
    return {
        "recall_std": float(tail["recall"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "spec_std": float(tail["spec"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "ba_std": float(tail["balanced_accuracy"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "mcc_std": float(tail["mcc"].std(ddof=1)) if len(tail) > 1 else 0.0,
        "threshold_std": float(tail["threshold"].std(ddof=1)) if len(tail) > 1 else 0.0,
    }


def plot_convergence_curves(outdir: Path) -> None:
    """Plot convergence curves for FedProx and Scaffold best runs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    metrics_to_plot = [
        ("balanced_accuracy", "Balanced Accuracy", axes[0, 0]),
        ("mcc", "MCC", axes[0, 1]),
        ("auc", "AUC", axes[1, 0]),
        ("auprc", "AUPRC", axes[1, 1]),
    ]
    
    for metric_key, metric_label, ax in metrics_to_plot:
        # FedProx runs
        for variant in BEST_RUNS["FedProx"]:
            df = collect_run_metrics("FedProx", variant)
            if df.empty:
                continue
            
            color = COLORS.get(f"FedProx_{variant}", "#dc2626")
            ax.plot(df["round"], df[metric_key], marker="o", linewidth=2.2, 
                   label=f"FedProx_{variant}", color=color, markersize=4)
        
        # Scaffold runs
        for variant in BEST_RUNS["Scaffold"]:
            df = collect_run_metrics("Scaffold", variant)
            if df.empty:
                continue
            
            color = COLORS.get(f"Scaffold_{variant}", "#2563eb")
            ax.plot(df["round"], df[metric_key], marker="s", linewidth=2.2, 
                   label=f"Scaffold_{variant}", color=color, markersize=4)
        
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} Convergence", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    
    fig.suptitle("FedProx vs Scaffold: Convergence Curves (Best Runs)", 
                fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(outdir / "convergence_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved convergence curves: {outdir / 'convergence_curves.png'}")


def plot_stability_analysis(outdir: Path) -> None:
    """Plot stability and volatility analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    volatility_metrics = [
        ("recall_std", "Recall Volatility (σ)", axes[0, 0]),
        ("spec_std", "Specificity Volatility (σ)", axes[0, 1]),
        ("mcc_std", "MCC Volatility (σ)", axes[1, 0]),
        ("threshold_std", "Threshold Volatility (σ)", axes[1, 1]),
    ]
    
    for metric_key, metric_label, ax in volatility_metrics:
        # Collect tail volatility for all runs
        volatility_data = []
        
        for variant in BEST_RUNS["FedProx"]:
            df = collect_run_metrics("FedProx", variant)
            if not df.empty:
                tail_vol = compute_tail_volatility(df)
                volatility_data.append({
                    "strategy": f"FedProx_{variant}",
                    "volatility": tail_vol.get(metric_key, 0),
                    "color": COLORS.get(f"FedProx_{variant}", "#dc2626"),
                })
        
        for variant in BEST_RUNS["Scaffold"]:
            df = collect_run_metrics("Scaffold", variant)
            if not df.empty:
                tail_vol = compute_tail_volatility(df)
                volatility_data.append({
                    "strategy": f"Scaffold_{variant}",
                    "volatility": tail_vol.get(metric_key, 0),
                    "color": COLORS.get(f"Scaffold_{variant}", "#2563eb"),
                })
        
        if not volatility_data:
            continue
        
        strategies = [v["strategy"] for v in volatility_data]
        values = [v["volatility"] for v in volatility_data]
        colors_list = [v["color"] for v in volatility_data]
        
        bars = ax.bar(range(len(strategies)), values, color=colors_list, alpha=0.8, edgecolor="black", linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} - Last 10 Rounds", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    
    fig.suptitle("Stability Analysis: Tail Volatility (Lower is Better)", 
                fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(outdir / "stability_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved stability analysis: {outdir / 'stability_analysis.png'}")


def plot_threshold_stability(outdir: Path) -> None:
    """Plot decision threshold stability across rounds."""
    fig, axes = plt.subplots(1, len(BEST_RUNS["FedProx"]) + len(BEST_RUNS["Scaffold"]), 
                            figsize=(5 * (len(BEST_RUNS["FedProx"]) + len(BEST_RUNS["Scaffold"])), 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    plot_idx = 0
    
    for variant in BEST_RUNS["FedProx"]:
        ax = axes[plot_idx]
        df = collect_run_metrics("FedProx", variant)
        if not df.empty:
            color = COLORS.get(f"FedProx_{variant}", "#dc2626")
            ax.plot(df["round"], df["threshold"], marker="o", linewidth=2.2, 
                   color=color, markersize=6, alpha=0.8)
            ax.fill_between(df["round"], 
                           df["threshold"] - df["threshold"].std(), 
                           df["threshold"] + df["threshold"].std(), 
                           alpha=0.2, color=color)
            
            ax.set_title(f"FedProx_{variant}\n(σ={df['threshold'].std(ddof=0):.4f})", 
                        fontsize=11, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Decision Threshold")
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 0.6)
        
        plot_idx += 1
    
    for variant in BEST_RUNS["Scaffold"]:
        ax = axes[plot_idx]
        df = collect_run_metrics("Scaffold", variant)
        if not df.empty:
            color = COLORS.get(f"Scaffold_{variant}", "#2563eb")
            ax.plot(df["round"], df["threshold"], marker="s", linewidth=2.2, 
                   color=color, markersize=6, alpha=0.8)
            ax.fill_between(df["round"], 
                           df["threshold"] - df["threshold"].std(), 
                           df["threshold"] + df["threshold"].std(), 
                           alpha=0.2, color=color)
            
            ax.set_title(f"Scaffold_{variant}\n(σ={df['threshold'].std(ddof=0):.4f})", 
                        fontsize=11, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Decision Threshold")
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 0.6)
        
        plot_idx += 1
    
    fig.suptitle("Decision Threshold Stability Across Rounds", 
                fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "threshold_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved threshold stability: {outdir / 'threshold_stability.png'}")


def create_metrics_table(outdir: Path) -> None:
    """Create a comprehensive metrics table for all best runs."""
    rows = []
    
    for variant in BEST_RUNS["FedProx"]:
        df = collect_run_metrics("FedProx", variant)
        if df.empty:
            continue
        
        best_idx = df["balanced_accuracy"].idxmax()
        best_row = df.loc[best_idx]
        
        tail_vol = compute_tail_volatility(df)
        
        rows.append({
            "Strategy": f"FedProx_{variant}",
            "Best Round": int(best_row["round"]),
            "Best BA": f"{best_row['balanced_accuracy']:.4f}",
            "Best MCC": f"{best_row['mcc']:.4f}",
            "Best Threshold": f"{best_row['threshold']:.2f}",
            "Recall": f"{best_row['recall']:.4f}",
            "Specificity": f"{best_row['spec']:.4f}",
            "AUC": f"{best_row['auc']:.4f}",
            "AUPRC": f"{best_row['auprc']:.4f}",
            "Threshold σ (tail)": f"{tail_vol['threshold_std']:.4f}",
            "Recall σ (tail)": f"{tail_vol['recall_std']:.4f}",
            "MCC σ (tail)": f"{tail_vol['mcc_std']:.4f}",
        })
    
    for variant in BEST_RUNS["Scaffold"]:
        df = collect_run_metrics("Scaffold", variant)
        if df.empty:
            continue
        
        best_idx = df["balanced_accuracy"].idxmax()
        best_row = df.loc[best_idx]
        
        tail_vol = compute_tail_volatility(df)
        
        rows.append({
            "Strategy": f"Scaffold_{variant}",
            "Best Round": int(best_row["round"]),
            "Best BA": f"{best_row['balanced_accuracy']:.4f}",
            "Best MCC": f"{best_row['mcc']:.4f}",
            "Best Threshold": f"{best_row['threshold']:.2f}",
            "Recall": f"{best_row['recall']:.4f}",
            "Specificity": f"{best_row['spec']:.4f}",
            "AUC": f"{best_row['auc']:.4f}",
            "AUPRC": f"{best_row['auprc']:.4f}",
            "Threshold σ (tail)": f"{tail_vol['threshold_std']:.4f}",
            "Recall σ (tail)": f"{tail_vol['recall_std']:.4f}",
            "MCC σ (tail)": f"{tail_vol['mcc_std']:.4f}",
        })
    
    df_table = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = outdir / "metrics_comparison_table.csv"
    df_table.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics table CSV: {csv_path}")
    
    # Create nice markdown table
    md_path = outdir / "metrics_comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Performance & Stability Metrics Comparison\n\n")
        f.write("## Best Runs: FedProx (4, 5) vs Scaffold (1)\n\n")
        f.write(df_table.to_markdown(index=False))
        f.write("\n\n## Interpretation Guide\n\n")
        f.write("- **Best BA**: Balanced Accuracy at peak\n")
        f.write("- **Best Threshold**: Decision threshold at peak BA\n")
        f.write("- **Threshold σ (tail)**: Standard deviation last 10 rounds (lower = more stable)\n")
        f.write("- **Recall σ (tail)**: Recall volatility last 10 rounds\n")
        f.write("- **MCC σ (tail)**: MCC volatility last 10 rounds\n\n")
        f.write("## Key Insights\n\n")
        f.write("- **FedProx**: Good performance with moderate stability\n")
        f.write("- **Scaffold**: Slightly lower peak, but exceptional stability\n")
        f.write("- **Threshold stability**: Critical for production deployment\n")
    
    print(f"[INFO] Saved metrics table markdown: {md_path}")


def plot_performance_vs_stability(outdir: Path) -> None:
    """Create scatter plot showing performance vs stability trade-off."""
    data = []
    
    for variant in BEST_RUNS["FedProx"]:
        df = collect_run_metrics("FedProx", variant)
        if not df.empty:
            best_idx = df["balanced_accuracy"].idxmax()
            best_ba = df.loc[best_idx, "balanced_accuracy"]
            tail_vol = compute_tail_volatility(df)
            combined_vol = np.mean([tail_vol["recall_std"], tail_vol["spec_std"], tail_vol["mcc_std"]])
            
            data.append({
                "strategy": f"FedProx_{variant}",
                "performance": best_ba,
                "stability": combined_vol,
                "color": COLORS.get(f"FedProx_{variant}", "#dc2626"),
                "marker": "o",
                "size": 200,
            })
    
    for variant in BEST_RUNS["Scaffold"]:
        df = collect_run_metrics("Scaffold", variant)
        if not df.empty:
            best_idx = df["balanced_accuracy"].idxmax()
            best_ba = df.loc[best_idx, "balanced_accuracy"]
            tail_vol = compute_tail_volatility(df)
            combined_vol = np.mean([tail_vol["recall_std"], tail_vol["spec_std"], tail_vol["mcc_std"]])
            
            data.append({
                "strategy": f"Scaffold_{variant}",
                "performance": best_ba,
                "stability": combined_vol,
                "color": COLORS.get(f"Scaffold_{variant}", "#2563eb"),
                "marker": "s",
                "size": 200,
            })
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for item in data:
        ax.scatter(item["stability"], item["performance"], 
                  color=item["color"], marker=item["marker"], s=item["size"],
                  alpha=0.7, edgecolors="black", linewidth=2, label=item["strategy"])
        
        # Add labels
        ax.annotate(item["strategy"], 
                   (item["stability"], item["performance"]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Volatility (Lower = More Stable)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Best Balanced Accuracy (Higher = Better Performance)", fontsize=12, fontweight="bold")
    ax.set_title("Performance vs Stability Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    
    # Add quadrant lines
    ax.axhline(y=0.74, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=0.01, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    fig.tight_layout()
    fig.savefig(outdir / "performance_vs_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved performance vs stability plot: {outdir / 'performance_vs_stability.png'}")


def plot_threshold_adaptation_analysis(outdir: Path) -> None:
    """Analyze if threshold jumping correlates with model improvement (good) or is just noise (bad)."""
    fig, axes = plt.subplots(2, len(BEST_RUNS["FedProx"]) + len(BEST_RUNS["Scaffold"]), 
                            figsize=(6 * (len(BEST_RUNS["FedProx"]) + len(BEST_RUNS["Scaffold"])), 10))
    
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    plot_idx = 0
    correlations = []
    
    for variant in BEST_RUNS["FedProx"]:
        df = collect_run_metrics("FedProx", variant)
        if not df.empty:
            # Top panel: BA and Threshold on same plot
            ax_top = axes[0, plot_idx]
            ax_top_twin = ax_top.twinx()
            
            color_ba = COLORS.get(f"FedProx_{variant}", "#dc2626")
            color_thr = "#fca5a5"
            
            line1 = ax_top.plot(df["round"], df["balanced_accuracy"], marker="o", 
                              linewidth=2.2, color=color_ba, label="BA", markersize=5)
            line2 = ax_top_twin.plot(df["round"], df["threshold"], marker="s", 
                                    linewidth=2.2, color=color_thr, label="Threshold", 
                                    linestyle="--", markersize=5)
            
            ax_top.set_ylabel("Balanced Accuracy", fontsize=10, color=color_ba, fontweight="bold")
            ax_top_twin.set_ylabel("Decision Threshold", fontsize=10, color=color_thr, fontweight="bold")
            ax_top.set_title(f"FedProx_{variant}: BA & Threshold Co-movement", fontsize=11, fontweight="bold")
            ax_top.grid(alpha=0.2)
            ax_top.tick_params(axis="y", labelcolor=color_ba)
            ax_top_twin.tick_params(axis="y", labelcolor=color_thr)
            
            # Bottom panel: Threshold change vs BA change
            ax_bot = axes[1, plot_idx]
            
            ba_change = df["balanced_accuracy"].diff()
            thr_change = df["threshold"].diff()
            
            correlation = ba_change.corr(thr_change)
            correlations.append({
                "strategy": f"FedProx_{variant}",
                "correlation": correlation,
                "interpretation": "GOOD (adaptive)" if correlation > 0.3 else "UNCLEAR" if correlation > -0.2 else "NOISE"
            })
            
            ax_bot.scatter(thr_change, ba_change, color=color_ba, alpha=0.6, s=60, edgecolors="black", linewidth=1)
            ax_bot.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
            ax_bot.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
            
            ax_bot.set_xlabel("Threshold Change (Δ)", fontsize=10)
            ax_bot.set_ylabel("BA Change (Δ)", fontsize=10)
            ax_bot.set_title(f"Correlation: {correlation:.3f}\n({correlations[-1]['interpretation']})", 
                           fontsize=10, fontweight="bold")
            ax_bot.grid(alpha=0.2)
        
        plot_idx += 1
    
    for variant in BEST_RUNS["Scaffold"]:
        df = collect_run_metrics("Scaffold", variant)
        if not df.empty:
            # Top panel: BA and Threshold on same plot
            ax_top = axes[0, plot_idx]
            ax_top_twin = ax_top.twinx()
            
            color_ba = COLORS.get(f"Scaffold_{variant}", "#2563eb")
            color_thr = "#93c5fd"
            
            line1 = ax_top.plot(df["round"], df["balanced_accuracy"], marker="o", 
                              linewidth=2.2, color=color_ba, label="BA", markersize=5)
            line2 = ax_top_twin.plot(df["round"], df["threshold"], marker="s", 
                                    linewidth=2.2, color=color_thr, label="Threshold", 
                                    linestyle="--", markersize=5)
            
            ax_top.set_ylabel("Balanced Accuracy", fontsize=10, color=color_ba, fontweight="bold")
            ax_top_twin.set_ylabel("Decision Threshold", fontsize=10, color=color_thr, fontweight="bold")
            ax_top.set_title(f"Scaffold_{variant}: BA & Threshold Co-movement", fontsize=11, fontweight="bold")
            ax_top.grid(alpha=0.2)
            ax_top.tick_params(axis="y", labelcolor=color_ba)
            ax_top_twin.tick_params(axis="y", labelcolor=color_thr)
            
            # Bottom panel: Threshold change vs BA change
            ax_bot = axes[1, plot_idx]
            
            ba_change = df["balanced_accuracy"].diff()
            thr_change = df["threshold"].diff()
            
            correlation = ba_change.corr(thr_change)
            correlations.append({
                "strategy": f"Scaffold_{variant}",
                "correlation": correlation,
                "interpretation": "GOOD (adaptive)" if correlation > 0.3 else "UNCLEAR" if correlation > -0.2 else "NOISE"
            })
            
            ax_bot.scatter(thr_change, ba_change, color=color_ba, alpha=0.6, s=60, edgecolors="black", linewidth=1)
            ax_bot.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
            ax_bot.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
            
            ax_bot.set_xlabel("Threshold Change (Δ)", fontsize=10)
            ax_bot.set_ylabel("BA Change (Δ)", fontsize=10)
            ax_bot.set_title(f"Correlation: {correlation:.3f}\n({correlations[-1]['interpretation']})", 
                           fontsize=10, fontweight="bold")
            ax_bot.grid(alpha=0.2)
        
        plot_idx += 1
    
    fig.suptitle("Is Threshold Jumping Good or Bad? Analysis of Threshold-Performance Correlation\n" + 
                "Top: BA & Threshold Co-movement | Bottom: Correlation of changes (>0.3 = adaptive/good, <-0.2 = noise/bad)", 
                fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "threshold_adaptation_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Print interpretation
    print("\n" + "="*60)
    print("THRESHOLD ADAPTATION ANALYSIS")
    print("="*60)
    for item in correlations:
        print(f"{item['strategy']:20} | Corr: {item['correlation']:+.3f} | {item['interpretation']}")
    print("="*60)
    print("INTERPRETATION:")
    print("  • GOOD (corr > 0.3): Threshold adapts WITH model improvement (positive learning)")
    print("  • UNCLEAR (-0.2 to 0.3): No clear relationship")
    print("  • NOISE (corr < -0.2): Threshold jumps negatively with BA (likely oscillation)")
    print("="*60 + "\n")
    
    print(f"[INFO] Saved threshold adaptation analysis: {outdir / 'threshold_adaptation_analysis.png'}")


def main() -> None:
    ensure_outdir(PLOT_OUT)
    
    print("[INFO] Creating FedProx vs Scaffold comparison plots...")
    print(f"[INFO] Output directory: {PLOT_OUT}")
    
    plot_convergence_curves(PLOT_OUT)
    plot_stability_analysis(PLOT_OUT)
    plot_threshold_stability(PLOT_OUT)
    plot_threshold_adaptation_analysis(PLOT_OUT)
    plot_performance_vs_stability(PLOT_OUT)
    create_metrics_table(PLOT_OUT)
    
    print("\n[INFO] All plots and tables created successfully!")
    print(f"[INFO] Check {PLOT_OUT} for all outputs")


if __name__ == "__main__":
    main()
