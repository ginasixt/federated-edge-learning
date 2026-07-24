#!/usr/bin/env python3
"""Plot per-strategy metric panels for the IID scaling experiment.

For each strategy family (FedAdam, FedProx, Scaffold), this script creates one
figure with three panels:

* Panel A: best MCC per round
* Panel B: best balanced accuracy per round
* Panel C: G-Mean per round

run with python3 plot_strategy_metric_panels.py

Each run gets a fixed color within its strategy figure. The panels show the
actual metric values across runs, and the peak point of each run is annotated
with the threshold of the best value for that metric when space allows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "result/splits_iid_scaling/splits_iid_16384_clients.json"
)
DEFAULT_OUT = Path(
    "result/plots/splits_iid_scaling_metric_panels"
)

ROUND_FILE_RE = re.compile(r"round_(\d+)_run_1\.json$")
STRATEGY_DIR_RE = re.compile(r"^all_rounds_(?P<family>[A-Za-z]+)_(?P<variant>\d+)$")

VARIANT_PALETTE = [
    "#1D4ED8",
    "#DC2626",
    "#16A34A",
    "#7C3AED",
    "#F59E0B",
    "#0F766E",
    "#DB2777",
    "#4B5563",
    "#65A30D",
    "#EA580C",
]


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def parse_strategy_dir(path: Path) -> Tuple[str, int]:
    match = STRATEGY_DIR_RE.match(path.name)
    if not match:
        return path.name, 0
    return match.group("family"), int(match.group("variant"))


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def safe_idxmax(series: pd.Series) -> Optional[int]:
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return int(cleaned.idxmax())


def variant_color(variant: int) -> str:
    if variant <= 0:
        return "#6B7280"
    return VARIANT_PALETTE[(variant - 1) % len(VARIANT_PALETTE)]


def darken_color(hex_color: str, factor: float) -> str:
    """Blend a color toward black.

    factor=0 keeps the original color, factor=1 makes it black.
    """
    factor = float(max(0.0, min(1.0, factor)))
    rgb = np.array(mcolors.to_rgb(hex_color), dtype=float)
    shaded = rgb * (1.0 - factor)
    return mcolors.to_hex(shaded)


def summarize_threshold_rows(threshold_rows: Iterable[dict]) -> Tuple[Dict[str, float], pd.DataFrame]:
    df = pd.DataFrame(list(threshold_rows))
    if df.empty:
        return {}, df

    for col in ["tp", "fp", "tn", "fn", "recall", "spec", "threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "balanced_accuracy" not in df.columns and {"recall", "spec"}.issubset(df.columns):
        df["balanced_accuracy"] = (df["recall"] + df["spec"]) / 2.0

    if "mcc" not in df.columns and {"tp", "fp", "tn", "fn"}.issubset(df.columns):
        tp = df["tp"].astype(float)
        fp = df["fp"].astype(float)
        tn = df["tn"].astype(float)
        fn = df["fn"].astype(float)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        numerator = (tp * tn) - (fp * fn)
        df["mcc"] = np.where(denom > 0, numerator / denom, np.nan)

    if "gmean" not in df.columns and {"recall", "spec"}.issubset(df.columns):
        df["gmean"] = np.sqrt(np.clip(df["recall"].astype(float), 0.0, None) * np.clip(df["spec"].astype(float), 0.0, None))

    summary: Dict[str, float] = {}
    metric_map = {
        "mcc": "best_mcc",
        "balanced_accuracy": "best_balanced_accuracy",
        "gmean": "best_gmean",
    }

    for metric, prefix in metric_map.items():
        if metric not in df.columns:
            continue
        idx = safe_idxmax(df[metric])
        if idx is None:
            continue
        row = df.loc[idx]
        summary[f"{prefix}_value"] = float(row.get(metric, np.nan))
        summary[f"{prefix}_threshold"] = float(row.get("threshold", np.nan))
        summary[f"{prefix}_recall"] = float(row.get("recall", np.nan))
        summary[f"{prefix}_spec"] = float(row.get("spec", np.nan))
        summary[f"{prefix}_mcc"] = float(row.get("mcc", np.nan))
        summary[f"{prefix}_balanced_accuracy"] = float(row.get("balanced_accuracy", np.nan))
        summary[f"{prefix}_gmean"] = float(row.get("gmean", np.nan))

    return summary, df


def collect_family_summaries(root: Path) -> Dict[str, pd.DataFrame]:
    rows: List[dict] = []

    for strategy_dir in sorted([d for d in root.glob("all_rounds_*") if d.is_dir()]):
        family, variant = parse_strategy_dir(strategy_dir)
        strategy_id = f"{family}_{variant}" if variant else family

        for round_file in sorted(strategy_dir.glob("round_*_run_1.json")):
            match = ROUND_FILE_RE.search(round_file.name)
            if not match:
                continue

            data = load_json(round_file)
            if not data:
                continue

            metrics = data.get("metrics", {})
            threshold_summary, _ = summarize_threshold_rows(metrics.get("all_thresholds", []))
            if not threshold_summary:
                continue

            row = {
                "family": family,
                "variant": variant,
                "strategy_id": strategy_id,
                "round": int(match.group(1)),
                "source_file": str(round_file),
                "model_checkpoint": data.get("model_checkpoint", ""),
            }
            row.update(threshold_summary)
            rows.append(row)

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    numeric_cols = [col for col in df.columns if col not in {"family", "strategy_id", "source_file", "model_checkpoint"}]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    family_frames: Dict[str, pd.DataFrame] = {}
    for family, family_df in df.groupby("family"):
        family_frames[family] = family_df.sort_values(["variant", "round"]).reset_index(drop=True)
    return family_frames


def lookup_threshold_row(threshold_df: pd.DataFrame, threshold_value: float) -> Optional[pd.Series]:
    if threshold_df.empty or "threshold" not in threshold_df.columns:
        return None
    matches = threshold_df[np.isclose(threshold_df["threshold"].astype(float), float(threshold_value), atol=1e-6)]
    if not matches.empty:
        return matches.iloc[0]
    closest_idx = (threshold_df["threshold"].astype(float) - float(threshold_value)).abs().idxmin()
    return threshold_df.loc[closest_idx]


def get_round_files_for_strategy(root: Path, family: str, variant: int) -> List[Path]:
    strategy_dir = root / f"all_rounds_{family}_{variant}"
    if not strategy_dir.exists():
        return []
    files = [p for p in strategy_dir.glob("round_*_run_1.json") if ROUND_FILE_RE.search(p.name)]
    # sort by numeric round index to avoid lexicographic ordering issues (round_1, round_10, ...)
    files_sorted = sorted(files, key=lambda p: int(ROUND_FILE_RE.search(p.name).group(1)))
    return files_sorted


def collect_threshold_series_for_run(root: Path, family: str, variant: int, thresholds: Sequence[float]) -> Tuple[List[int], Dict[float, List[Optional[pd.Series]]]]:
    round_files = get_round_files_for_strategy(root, family, variant)
    rounds: List[int] = []
    thr_rows_by_thr: Dict[float, List[Optional[pd.Series]]] = {float(t): [] for t in thresholds}

    for rf in round_files:
        match = ROUND_FILE_RE.search(rf.name)
        if not match:
            continue
        rounds.append(int(match.group(1)))
        data = load_json(rf)
        if not data:
            for t in list(thr_rows_by_thr.keys()):
                thr_rows_by_thr[t].append(None)
            continue

        thr_df = pd.DataFrame(data.get("metrics", {}).get("all_thresholds", []))
        for t in list(thr_rows_by_thr.keys()):
            if thr_df.empty:
                thr_rows_by_thr[t].append(None)
            else:
                thr_rows_by_thr[t].append(lookup_threshold_row(thr_df, t))

    return rounds, thr_rows_by_thr


def draw_run_threshold_comparison(ax_a: plt.Axes, ax_b: plt.Axes, root: Path, family: str, variant: int, thresholds: Sequence[float]) -> bool:
    """Draw one run-comparison row into two axes.

    Returns True if any data was drawn.
    """
    rounds, thr_rows_by_thr = collect_threshold_series_for_run(root, family, variant, thresholds)
    if not rounds:
        return False

    palette = VARIANT_PALETTE
    for i, (t, rows) in enumerate(thr_rows_by_thr.items()):
        rec = [float(r.get("recall", np.nan)) if r is not None else np.nan for r in rows]
        spec = [float(r.get("spec", np.nan)) if r is not None else np.nan for r in rows]
        base_color = palette[i % len(palette)]
        ax_a.scatter(spec, rec, label=f"thr={t:.2f}", c=base_color, s=32, alpha=0.92, edgecolors="none")
        ax_b.plot(
            rounds,
            rec,
            color=base_color,
            linewidth=1.0,
            marker="o",
            markersize=3.0,
            markerfacecolor=base_color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=f"rec thr={t:.2f}",
        )
        ax_b.plot(
            rounds,
            spec,
            color=base_color,
            linewidth=1.0,
            linestyle="--",
            marker="o",
            markersize=2.6,
            markerfacecolor="white",
            markeredgecolor=base_color,
            markeredgewidth=0.7,
            label=f"spec thr={t:.2f}",
        )

    ax_a.set_xlabel("Specificity")
    ax_a.set_ylabel("Recall")
    ax_a.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_a.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.grid(alpha=0.2)

    ax_b.set_xlabel("Round")
    ax_b.set_ylabel("Metric")
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if rounds:
        ax_b.set_xlim(-1, max(rounds) + 1)
    ax_b.grid(alpha=0.2)
    return True


def plot_run_threshold_comparison(root: Path, outdir: Path, family: str, variant: int, thresholds: Sequence[float]) -> Optional[Path]:
    """Create a two-panel figure for a single run (family+variant).

    Panel A: spec vs recall scatter for the specified thresholds (points per round).
    Panel B: recall and spec vs round for each specified threshold (lines).
    """
    ensure_outdir(outdir)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))
    has_data = draw_run_threshold_comparison(ax_a, ax_b, root, family, variant, thresholds)
    if not has_data:
        print(f"[WARN] No round files for {family}_{variant}")
        plt.close(fig)
        return None

    ax_a.set_title(f"{family}_{variant}: Spec vs Recall for thresholds")
    ax_a.legend()
    ax_b.set_title(f"{family}_{variant}: Recall & Spec vs Round per threshold")
    ax_b.legend(fontsize=8)

    outpath = outdir / f"run_compare_{family}_{variant}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved run compare plot: {outpath}")
    return outpath


def plot_family_threshold_comparison_stack(root: Path, outdir: Path, family: str, run_specs: Sequence[Tuple[int, Sequence[float]]]) -> Optional[Path]:
    """Create one stacked figure per strategy family.

    Each row corresponds to one requested run and contains the same two panels
    as `plot_run_threshold_comparison`.
    """
    if not run_specs:
        return None

    ensure_outdir(outdir)
    fig, axes = plt.subplots(len(run_specs), 2, figsize=(13.5, 5.2 * len(run_specs)), squeeze=False)

    for row_idx, (variant, thresholds) in enumerate(run_specs):
        ax_a, ax_b = axes[row_idx]
        has_data = draw_run_threshold_comparison(ax_a, ax_b, root, family, variant, thresholds)
        if not has_data:
            ax_a.axis("off")
            ax_b.axis("off")
            continue

        ax_a.set_title(f"{family}_{variant}: Spec vs Recall for thresholds")
        ax_b.set_title(f"{family}_{variant}: Recall & Spec vs Round per threshold")
        ax_a.legend(fontsize=8, loc="best")
        ax_b.legend(fontsize=8, loc="best")

    fig.suptitle(f"{family}: stacked run comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    outpath = outdir / f"run_compare_{family}_stack.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved stacked run compare plot: {outpath}")
    return outpath


def format_panel(ax: plt.Axes, family: str = "default") -> None:
    tick_interval = 0.05 if family == "Scaffold" else 0.10
    
    # Collect all y-data from lines to determine actual data range
    all_y = []
    for line in ax.get_lines():
        all_y.extend(line.get_ydata())
    
    if all_y:
        y_min = np.nanmin(all_y)
        y_max = np.nanmax(all_y)
    else:
        y_min, y_max = 0, 1
    
    # Round to nearest tick interval, with small margin
    y_min_rounded = np.floor(y_min / tick_interval) * tick_interval
    y_max_rounded = np.ceil(y_max / tick_interval) * tick_interval
    
    # Ensure some minimum range
    if y_max_rounded - y_min_rounded < tick_interval * 2:
        y_min_rounded = max(0, y_min_rounded - tick_interval)
        y_max_rounded = min(1, y_max_rounded + tick_interval)
    
    ax.set_ylim(y_min_rounded, y_max_rounded)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.25)


def annotate_peak_threshold(ax: plt.Axes, x: float, y: float, threshold: float, color: str) -> None:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(threshold):
        return
    ax.scatter([x], [y], color=color, s=60, marker="*", edgecolors="black", linewidths=0.6, zorder=5)
    ax.annotate(
        f"thr={threshold:.2f}",
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7,
        color=color,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.85},
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.7},
        zorder=6,
    )


def plot_family_panels(family: str, family_df: pd.DataFrame, outdir: Path) -> Path:
    ensure_outdir(outdir)
    if family_df.empty:
        raise ValueError(f"No data available for family {family}")

    metrics = [
        ("best_mcc_value", "best_mcc_threshold", "Best MCC per Round"),
        ("best_balanced_accuracy_value", "best_balanced_accuracy_threshold", "Best Balanced Accuracy per Round"),
        ("best_gmean_value", "best_gmean_threshold", "Best G-Mean per Round"),
    ]

    # For each metric we create a main panel + a small threshold panel beneath it
    pairs = len(metrics)
    fig, axes = plt.subplots(pairs * 2, 1, figsize=(15.5, 4.5 * pairs), sharex=True)
    axes = np.atleast_1d(axes)

    variants = sorted(int(v) for v in family_df["variant"].dropna().unique().tolist())
    if not variants:
        raise ValueError(f"No variants found for family {family}")

    legend_handles = []
    legend_labels = []
    best_legend_handles = []
    best_legend_labels = []
    # track best overall run per metric: map metric index -> (best_value, variant)
    best_overall: Dict[int, Tuple[float, Optional[int]]] = {i: (float("nan"), None) for i in range(len(metrics))}
    metric_variant_best_values: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(metrics))}

    for variant in variants:
        group = family_df[family_df["variant"] == variant].sort_values("round")
        if group.empty:
            continue

        color = variant_color(variant)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.0))
        legend_labels.append(f"Run {variant}")
        # plot each metric into its main axis and the selected threshold into the small axis below
        for i, (value_col, threshold_col, _) in enumerate(metrics):
            main_ax = axes[i * 2]
            thr_ax = axes[i * 2 + 1]

            series = group[value_col].astype(float)
            # update best overall per metric
            try:
                cur_max = float(np.nanmax(series.values))
            except Exception:
                cur_max = float("nan")
            prev_best, prev_variant = best_overall[i]
            if np.isfinite(cur_max):
                metric_variant_best_values[i].append((variant, cur_max))
            if not np.isfinite(prev_best) or (np.isfinite(cur_max) and cur_max > prev_best):
                best_overall[i] = (cur_max, variant)
            main_ax.plot(
                group["round"].astype(int),
                series.values,
                color=color,
                linewidth=1.0,
                alpha=0.95,
                marker="o",
                markersize=3.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

            # threshold series for the chosen best metric in this round
            thr_series = group[threshold_col].astype(float)
            thr_ax.plot(
                group["round"].astype(int),
                thr_series.values,
                color=color,
                linewidth=1.0,
                alpha=0.9,
                marker="s",
                markersize=3,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )

            # annotate volatility measures on threshold subplot (std and mean abs diff)
            try:
                thr_std = float(thr_series.std(ddof=0))
            except Exception:
                thr_std = float("nan")
            try:
                mean_abs_diff = float(thr_series.diff().abs().mean())
            except Exception:
                mean_abs_diff = float("nan")

            # place small volatility label at right side
            thr_ax.text(
                0.98,
                0.85 - 0.08 * (variant % 4),
                f"Run {variant}: σ={thr_std:.2f}, Δ̄={mean_abs_diff:.2f}",
                transform=thr_ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
                color=color,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.9},
            )

            best_idx = safe_idxmax(series)
            if best_idx is not None:
                best_row = group.loc[best_idx]
                annotate_peak_threshold(
                    main_ax,
                    float(best_row["round"]),
                    float(best_row[value_col]),
                    float(best_row.get(threshold_col, np.nan)),
                    color,
                )

    # After plotting all variants, capture the best overall run for each metric
    for i, (value_col, threshold_col, _) in enumerate(metrics):
        best_val, best_variant = best_overall.get(i, (float("nan"), None))
        if best_variant is None:
            continue

        # find the group for the best variant and re-plot it with stronger styling
        best_group = family_df[family_df["variant"] == best_variant].sort_values("round")
        if best_group.empty:
            continue

        highlight_color = variant_color(int(best_variant))
        main_ax = axes[i * 2]
        # keep the original line visible; only mark the best run with a small star
        for line in main_ax.get_lines():
            if len(line.get_xdata()) == len(best_group) and np.allclose(line.get_xdata(), best_group["round"].astype(int).values):
                line.set_linewidth(max(float(line.get_linewidth()), 1.45))
                line.set_alpha(0.95)
                line.set_zorder(3)
                break

        best_idx = safe_idxmax(best_group[value_col].astype(float))
        if best_idx is not None:
            best_row = best_group.loc[best_idx]
            main_ax.scatter(
                [float(best_row["round"])],
                [float(best_row[value_col])],
                s=88,
                marker="*",
                color=highlight_color,
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )

        ranked = sorted(metric_variant_best_values.get(i, []), key=lambda item: item[1], reverse=True)
        second_variant: Optional[int] = None
        second_val = float("nan")
        if len(ranked) > 1:
            second_variant, second_val = ranked[1]

        gap_text = "n/a"
        if np.isfinite(best_val) and np.isfinite(second_val):
            gap_text = f"{best_val - second_val:.3f}"

        metric_label = metrics[i][2]
        best_legend_handles.append(
            Line2D([0], [0], marker="*", linestyle="None", color=highlight_color, markeredgecolor="black", markersize=10)
        )
        if second_variant is None:
            best_legend_labels.append(f"{metric_label}: Run {best_variant} = {best_val:.3f}")
        else:
            best_legend_labels.append(
                f"{metric_label}: Run {best_variant} = {best_val:.3f} | 2nd Run {second_variant} = {second_val:.3f} | Δ={gap_text}"
            )

    # set titles and formatting: main and threshold axes
    for i, (value_col, threshold_col, title) in enumerate(metrics):
        main_ax = axes[i * 2]
        thr_ax = axes[i * 2 + 1]

        main_ax.set_title(f"{title}", fontsize=12, fontweight="bold")
        main_ax.set_ylabel("Metric Value")
        format_panel(main_ax, family)

        thr_ax.set_ylabel("Decision Threshold")
        # Use finer grid for Scaffold, standard for others
        tick_interval = 0.05 if family == "Scaffold" else 0.10
        thr_ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_interval))
        thr_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        thr_ax.tick_params(axis="y", which="major", length=4)
        thr_ax.grid(alpha=0.18)

    axes[-1].set_xlabel("Round")
    # Set X-axis limits to start at 0 for consistency
    max_round = family_df["round"].max() if not family_df.empty else 1
    axes[-1].set_xlim(-1, max_round + 1)
    
    run_legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=min(5, len(legend_labels)),
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.add_artist(run_legend)
    if best_legend_handles:
        fig.legend(
            best_legend_handles,
            best_legend_labels,
            loc="lower right",
            frameon=True,
            framealpha=0.95,
            fontsize=8,
            bbox_to_anchor=(0.995, 0.01),
            title="Best overall",
            title_fontsize=8,
        )
    fig.suptitle(
        f"{family}: Metric Panels with Annotated Best Decision Thresholds",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))

    outpath = outdir / f"metric_panels_{family}.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_all_plots(root: Path, outdir: Path) -> List[Path]:
    family_frames = collect_family_summaries(root)
    if not family_frames:
        raise RuntimeError(f"No round files with threshold metrics found under {root}")

    ensure_outdir(outdir)
    outputs: List[Path] = []
    for family in ["FedAdam", "FedProx", "Scaffold"]:
        family_df = family_frames.get(family)
        if family_df is None or family_df.empty:
            continue
        outputs.append(plot_family_panels(family, family_df, outdir))

    if not outputs:
        raise RuntimeError(f"No strategy plots could be created under {root}")

    print("[INFO] Saved plots:")
    for output in outputs:
        print(f"  - {output}")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Creates per-strategy metric panels for the IID scaling experiment.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root directory containing all_rounds_* folders")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for the generated plots")
    parser.add_argument(
        "--run-compare",
        type=str,
        nargs="*",
        help=(
            "Per-run compare specifier(s) of the form Family_variant:thr1,thr2. "
            "Example: FedAdam_1:0.5,0.65 FedProx_4:0.65,0.45"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    make_all_plots(args.root, args.out)

    run_compare_entries = getattr(args, "run_compare", None) or []
    if not run_compare_entries:
        return

    grouped_specs: Dict[str, List[Tuple[int, Sequence[float]]]] = {}

    # parse entries like Family_variant:thr1,thr2 and group them by family
    for entry in run_compare_entries:
        try:
            left, right = entry.split(":")
            fam_part = left.split("_")
            family = "_".join(fam_part[:-1]) if len(fam_part) > 1 else fam_part[0]
            variant = int(fam_part[-1])
            thresholds = [float(x) for x in right.split(",") if x.strip()]
        except Exception as exc:
            print(f"[WARN] Failed to parse run-compare entry '{entry}': {exc}")
            continue

        grouped_specs.setdefault(family, []).append((variant, thresholds))

    for family, run_specs in grouped_specs.items():
        plot_family_threshold_comparison_stack(args.root, args.out, family, run_specs)


if __name__ == "__main__":
    main()