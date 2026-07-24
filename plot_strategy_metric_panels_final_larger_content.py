#!/usr/bin/env python3
"""Plot and analyse per-strategy MCC curves for the IID scaling experiment.

For each strategy family (FedAdam, FedProx, Scaffold), the script creates one
clean two-panel figure:

* Panel A: best MCC per communication round
* Panel B: decision threshold that produced the best MCC in each round

For every run, it additionally reports:

* mean MCC over the final k rounds (plateau MCC),
* standard deviation over the same plateau window,
* the first round from which MCC reaches and remains above a fixed fraction of
  its plateau MCC (default: 90%).

The figures are saved as both high-resolution PNG and vector PDF files. The
numerical results are saved as CSV files and printed to the terminal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "result/splits_iid_scaling/splits_iid_16384_clients.json"
)
DEFAULT_OUT = Path(
    "result/plots/splits_iid_scaling_metric_panels"
)

DEFAULT_PLATEAU_ROUNDS = 10
DEFAULT_TARGET_FRACTION = 0.90
DEFAULT_X_TICK_INTERVAL = 5
DEFAULT_FIGURE_WIDTH = 12.0
DEFAULT_FIGURE_HEIGHT = 7.4

ROUND_FILE_RE = re.compile(r"round_(\d+)_run_1\.json$")
STRATEGY_DIR_RE = re.compile(r"^all_rounds_(?P<family>[A-Za-z]+)_(?P<variant>\d+)$")

# UHH-based palette. The first three colors are the specified UHH colors;
# violet and ochre extend the palette while remaining clearly distinguishable
# on a white background. Further entries are fallbacks for more than five runs.
VARIANT_PALETTE = [
    "#E2001A",  # UHH red
    "#0271BB",  # UHH blue
    "#3B515B",  # UHH slate
    "#7A3E9D",  # complementary violet
    "#F28E2B",  # complementary orange
    "#00867A",  # teal fallback
    "#A64B73",  # muted magenta fallback
    "#5B6F2A",  # olive fallback
    "#6B7280",  # neutral gray fallback
    "#C45A00",  # burnt orange fallback
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


def summarize_threshold_rows(threshold_rows: Iterable[dict]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return the threshold row with the highest MCC.

    MCC is calculated from the confusion-matrix counts when it is not already
    present in the input JSON. Balanced accuracy and G-Mean are intentionally
    no longer calculated or exported.
    """
    df = pd.DataFrame(list(threshold_rows))
    if df.empty:
        return {}, df

    for col in ["tp", "fp", "tn", "fn", "recall", "spec", "threshold", "mcc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mcc" not in df.columns and {"tp", "fp", "tn", "fn"}.issubset(df.columns):
        tp = df["tp"].astype(float)
        fp = df["fp"].astype(float)
        tn = df["tn"].astype(float)
        fn = df["fn"].astype(float)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        numerator = (tp * tn) - (fp * fn)
        df["mcc"] = np.where(denominator > 0, numerator / denominator, np.nan)

    if "mcc" not in df.columns:
        return {}, df

    best_idx = safe_idxmax(df["mcc"])
    if best_idx is None:
        return {}, df

    best_row = df.loc[best_idx]
    summary = {
        "best_mcc_value": float(best_row.get("mcc", np.nan)),
        "best_mcc_threshold": float(best_row.get("threshold", np.nan)),
        "best_mcc_recall": float(best_row.get("recall", np.nan)),
        "best_mcc_spec": float(best_row.get("spec", np.nan)),
    }
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



def first_round_reaching_and_holding_target(
    rounds: pd.Series,
    mcc_values: pd.Series,
    target_mcc: float,
) -> Optional[int]:
    """Return the first observed round from which MCC stays at/above target.

    Missing MCC values are removed before the suffix test. Therefore, the result
    refers to all subsequently *observed* rounds in the run.
    """
    valid = pd.DataFrame({"round": rounds, "mcc": mcc_values}).dropna()
    if valid.empty or not np.isfinite(target_mcc):
        return None

    valid = valid.sort_values("round").reset_index(drop=True)
    values = valid["mcc"].to_numpy(dtype=float)
    suffix_minimum = np.minimum.accumulate(values[::-1])[::-1]
    candidate_positions = np.flatnonzero(suffix_minimum >= target_mcc)
    if candidate_positions.size == 0:
        return None
    return int(valid.loc[int(candidate_positions[0]), "round"])


def analyse_family_mcc(
    family: str,
    family_df: pd.DataFrame,
    plateau_rounds: int,
    target_fraction: float,
    max_plateau_std: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate plateau performance, plateau stability and convergence speed."""
    if plateau_rounds <= 0:
        raise ValueError("plateau_rounds must be greater than zero")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in the interval (0, 1]")
    if max_plateau_std is not None and max_plateau_std < 0:
        raise ValueError("max_plateau_std must be non-negative")

    result_rows: List[dict] = []
    variants = sorted(int(v) for v in family_df["variant"].dropna().unique())

    for variant in variants:
        group = family_df[family_df["variant"] == variant].sort_values("round")
        valid = group[["round", "best_mcc_value", "best_mcc_threshold"]].dropna(
            subset=["round", "best_mcc_value"]
        )
        if valid.empty:
            continue

        plateau = valid.tail(plateau_rounds)
        plateau_mean = float(plateau["best_mcc_value"].mean())
        # Population standard deviation: the final k rounds are the complete
        # plateau window being described, not a sample from a larger window.
        plateau_std = float(plateau["best_mcc_value"].std(ddof=0))
        target_mcc = float(target_fraction * plateau_mean)
        rounds_to_target = first_round_reaching_and_holding_target(
            valid["round"], valid["best_mcc_value"], target_mcc
        )

        peak_idx = safe_idxmax(valid["best_mcc_value"])
        peak_round = np.nan
        peak_mcc = np.nan
        peak_threshold = np.nan
        if peak_idx is not None:
            peak_row = valid.loc[peak_idx]
            peak_round = int(peak_row["round"])
            peak_mcc = float(peak_row["best_mcc_value"])
            peak_threshold = float(peak_row.get("best_mcc_threshold", np.nan))

        result_rows.append(
            {
                "family": family,
                "run": variant,
                "rounds_available": int(len(valid)),
                "plateau_rounds_requested": int(plateau_rounds),
                "plateau_rounds_used": int(len(plateau)),
                "plateau_start_round": int(plateau["round"].iloc[0]),
                "plateau_end_round": int(plateau["round"].iloc[-1]),
                "plateau_mcc_mean": plateau_mean,
                "plateau_mcc_std": plateau_std,
                "target_fraction": float(target_fraction),
                "target_mcc": target_mcc,
                "rounds_to_target": rounds_to_target if rounds_to_target is not None else np.nan,
                "reached_and_held_target": rounds_to_target is not None,
                "peak_mcc": peak_mcc,
                "peak_round": peak_round,
                "peak_threshold": peak_threshold,
                "stable_by_cutoff": (
                    plateau_std <= max_plateau_std if max_plateau_std is not None else pd.NA
                ),
            }
        )

    summary = pd.DataFrame(result_rows)
    if summary.empty:
        return summary

    summary["stability_rank"] = summary["plateau_mcc_std"].rank(
        method="min", ascending=True
    ).astype("Int64")
    summary["performance_rank"] = summary["plateau_mcc_mean"].rank(
        method="min", ascending=False
    ).astype("Int64")
    summary["convergence_rank"] = summary["rounds_to_target"].rank(
        method="min", ascending=True, na_option="bottom"
    ).astype("Int64")

    # A recommendation is only produced when the user supplies an explicit,
    # fixed stability cutoff. This avoids data-dependent post-hoc selection.
    summary["practically_tied_with_best"] = False
    summary["recommended"] = False
    if max_plateau_std is not None:
        stable = summary[summary["stable_by_cutoff"] == True].copy()  # noqa: E712
        if not stable.empty:
            best_performance_index = stable["plateau_mcc_mean"].idxmax()
            best_performance_row = stable.loc[best_performance_index]
            best_mean = float(best_performance_row["plateau_mcc_mean"])
            best_std = float(best_performance_row["plateau_mcc_std"])

            # Treat plateau means as practically equivalent when their difference
            # is no larger than the larger of the two plateau fluctuations.
            stable["practically_tied_with_best"] = stable.apply(
                lambda row: (best_mean - float(row["plateau_mcc_mean"]))
                <= max(best_std, float(row["plateau_mcc_std"])),
                axis=1,
            )
            summary.loc[
                stable.index,
                "practically_tied_with_best",
            ] = stable["practically_tied_with_best"]

            tied = stable[stable["practically_tied_with_best"]].copy()
            tied["rounds_to_target_sort"] = tied["rounds_to_target"].fillna(np.inf)
            winner_index = tied.sort_values(
                ["rounds_to_target_sort", "plateau_mcc_mean"],
                ascending=[True, False],
            ).index[0]
            summary.loc[winner_index, "recommended"] = True

    return summary.sort_values("run").reset_index(drop=True)


def print_family_mcc_summary(summary: pd.DataFrame, max_plateau_std: Optional[float]) -> None:
    if summary.empty:
        return

    family = str(summary["family"].iloc[0])
    target_pct = 100.0 * float(summary["target_fraction"].iloc[0])
    print(f"\n[RESULT] {family}: plateau MCC and rounds-to-{target_pct:.0f}%")

    display = summary[
        [
            "run",
            "plateau_start_round",
            "plateau_end_round",
            "plateau_mcc_mean",
            "plateau_mcc_std",
            "target_mcc",
            "rounds_to_target",
            "peak_mcc",
            "peak_round",
        ]
    ].copy()
    for col in ["plateau_mcc_mean", "plateau_mcc_std", "target_mcc", "peak_mcc"]:
        display[col] = display[col].map(lambda value: f"{value:.4f}" if pd.notna(value) else "n/a")
    display["rounds_to_target"] = display["rounds_to_target"].map(
        lambda value: str(int(value)) if pd.notna(value) else "not reached/held"
    )
    print(display.to_string(index=False))

    if max_plateau_std is None:
        print(
            "[INFO] No automatic winner selected. Set --max-plateau-std to apply "
            "the stability-first selection rule with a pre-defined cutoff."
        )
    else:
        winner = summary[summary["recommended"]]
        if winner.empty:
            print(
                f"[INFO] No run satisfies the plateau SD cutoff "
                f"({max_plateau_std:.4f})."
            )
        else:
            row = winner.iloc[0]
            rounds_text = (
                str(int(row["rounds_to_target"]))
                if pd.notna(row["rounds_to_target"])
                else "not reached/held"
            )
            print(
                f"[SELECTED] Run {int(row['run'])}: plateau MCC "
                f"{row['plateau_mcc_mean']:.4f} ± {row['plateau_mcc_std']:.4f}; "
                f"rounds-to-target = {rounds_text}."
            )

def format_panel(ax: plt.Axes, family: str = "default") -> None:
    # Use the same 0.05 MCC spacing for every strategy so that differences
    # between runs remain directly comparable across figures.
    tick_interval = 0.05
    
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
    ax.grid(alpha=0.25, linewidth=0.75)


def plot_family_panels(
    family: str,
    family_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outdir: Path,
    x_tick_interval: int,
    figure_width: float,
    figure_height: float,
) -> List[Path]:
    """Create an MCC panel and a compact, horizontally dodged threshold panel.

    Runs that select the same decision threshold in the same communication
    round are shifted slightly along the x-axis. Their true threshold values
    therefore remain unchanged while all points stay visible.

    FedAdam receives a larger threshold panel because its broader threshold
    range produces more y-axis levels and denser trajectories.
    """
    ensure_outdir(outdir)
    if family_df.empty:
        raise ValueError(f"No data available for family {family}")
    if x_tick_interval <= 0:
        raise ValueError("x_tick_interval must be greater than zero")
    if figure_width <= 0 or figure_height <= 0:
        raise ValueError("figure_width and figure_height must be greater than zero")

    variants = sorted(int(v) for v in family_df["variant"].dropna().unique())
    if not variants:
        raise ValueError(f"No variants found for family {family}")

    is_fedadam = family.lower() == "fedadam"

    # FedAdam spans more threshold levels and therefore receives additional
    # vertical space in the lower panel. The larger hspace is used for every
    # strategy to separate the two panels more clearly.
    if is_fedadam:
        effective_height = max(figure_height, 8.6)
        height_ratios = [3.65, 1.75]
    else:
        effective_height = max(figure_height, 7.6)
        height_ratios = [4.15, 1.20]

    panel_gap = 0.20
    fig, (main_ax, threshold_ax) = plt.subplots(
        2,
        1,
        figsize=(figure_width, effective_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": panel_gap,
        },
    )

    # A small horizontal displacement prevents identical threshold points from
    # covering each other. No vertical displacement is applied, so the y-value
    # always remains the actual selected threshold.
    if len(variants) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.34, 0.34, len(variants))

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for offset, variant in zip(offsets, variants):
        group = family_df[family_df["variant"] == variant].sort_values("round")
        if group.empty:
            continue

        color = variant_color(variant)
        rounds = group["round"].to_numpy(dtype=float)
        mcc_values = group["best_mcc_value"].to_numpy(dtype=float)
        thresholds = group["best_mcc_threshold"].to_numpy(dtype=float)

        # Main MCC curves retain the original line and marker styling.
        main_ax.plot(
            rounds,
            mcc_values,
            color=color,
            linewidth=1.45,
            alpha=0.97,
            marker="o",
            markersize=4.3,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

        # A very light line preserves the temporal order without dominating the
        # panel. The visible points are horizontally dodged when runs coincide.
        shifted_rounds = rounds + float(offset)
        threshold_ax.plot(
            shifted_rounds,
            thresholds,
            color=color,
            linewidth=0.85,
            alpha=0.28,
            zorder=1,
        )
        threshold_ax.scatter(
            shifted_rounds,
            thresholds,
            color=color,
            s=34,
            marker="o",
            edgecolors="white",
            linewidths=0.55,
            alpha=0.98,
            zorder=3,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.45,
                marker="o",
                markersize=5.4,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        )

        # Restore the plateau and convergence statistics for every run.
        run_summary = summary_df[summary_df["run"] == variant]
        if run_summary.empty:
            legend_labels.append(f"Run {variant}")
        else:
            row = run_summary.iloc[0]
            target_pct = 100.0 * float(row["target_fraction"])
            rounds_to_target = (
                str(int(row["rounds_to_target"]))
                if pd.notna(row["rounds_to_target"])
                else "n/a"
            )
            legend_labels.append(
                f"Run {variant}: plateau "
                f"{float(row['plateau_mcc_mean']):.3f} ± "
                f"{float(row['plateau_mcc_std']):.3f}, "
                f"R{target_pct:.0f}={rounds_to_target}"
            )

    max_round = int(family_df["round"].max())

    # Titles are intentionally omitted because the APA-style figure number and
    # title are placed in the thesis document rather than inside the plot.
    main_ax.set_ylabel("MCC", fontsize=12)
    format_panel(main_ax, family)

    threshold_ax.set_ylabel("Decision Threshold", fontsize=12, labelpad=10)
    threshold_ax.set_xlabel("Communication Round", fontsize=12)

    # Decision thresholds were evaluated on a fixed 0.05 grid.
    threshold_tick_interval = 0.05
    threshold_values = pd.to_numeric(
        family_df["best_mcc_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_thresholds = threshold_values[np.isfinite(threshold_values)]

    if finite_thresholds.size:
        threshold_min = (
            np.floor(finite_thresholds.min() / threshold_tick_interval)
            * threshold_tick_interval
        )
        threshold_max = (
            np.ceil(finite_thresholds.max() / threshold_tick_interval)
            * threshold_tick_interval
        )

        if np.isclose(threshold_min, threshold_max):
            threshold_min -= threshold_tick_interval
            threshold_max += threshold_tick_interval

        # Half a grid step of padding keeps outer points away from the axes.
        threshold_ax.set_ylim(
            max(0.0, threshold_min - threshold_tick_interval / 2.0),
            min(1.0, threshold_max + threshold_tick_interval / 2.0),
        )

    threshold_ax.yaxis.set_major_locator(
        mticker.MultipleLocator(threshold_tick_interval)
    )
    threshold_ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2f")
    )
    threshold_ax.grid(axis="y", which="major", alpha=0.24, linewidth=0.75)
    threshold_ax.grid(axis="x", which="major", alpha=0.10, linewidth=0.70)

    for axis in (main_ax, threshold_ax):
        axis.set_xlim(-0.5, max_round + 0.5)
        axis.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_interval))
        axis.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        axis.tick_params(axis="x", which="major", labelsize=10.5, length=4.5, width=1.0)
        axis.tick_params(axis="y", which="major", labelsize=10.5, length=4.5, width=1.0)
        axis.tick_params(axis="x", which="minor", length=2.8, width=0.8, color="0.50")

        # Open-axis layout.
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.05)
        axis.spines["bottom"].set_linewidth(1.05)

    # FedAdam has more threshold labels; the larger panel above provides the
    # main remedy, while a marginally smaller tick font avoids crowding.
    if is_fedadam:
        threshold_ax.tick_params(axis="y", which="major", labelsize=9.5)

    main_ax.tick_params(labelbottom=False)

    # Centre the legend relative to the actual plotting axis rather than the
    # complete figure canvas. Three columns keep the detailed labels readable.
    threshold_ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38 if is_fedadam else -0.44),
        ncol=min(3, len(legend_labels)),
        frameon=True,
        framealpha=0.95,
        fontsize=9.2,
        borderaxespad=0.0,
        columnspacing=1.5,
        handlelength=2.2,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.985,
        bottom=0.235 if is_fedadam else 0.225,
        hspace=panel_gap,
    )

    png_path = outdir / f"mcc_panels_{family}.png"
    pdf_path = outdir / f"mcc_panels_{family}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]

def make_all_plots(
    root: Path,
    outdir: Path,
    plateau_rounds: int,
    target_fraction: float,
    x_tick_interval: int,
    max_plateau_std: Optional[float],
    figure_width: float,
    figure_height: float,
) -> List[Path]:
    family_frames = collect_family_summaries(root)
    if not family_frames:
        raise RuntimeError(f"No round files with threshold metrics found under {root}")

    ensure_outdir(outdir)
    outputs: List[Path] = []
    all_summaries: List[pd.DataFrame] = []

    for family in ["FedAdam", "FedProx", "Scaffold"]:
        family_df = family_frames.get(family)
        if family_df is None or family_df.empty:
            continue

        summary = analyse_family_mcc(
            family,
            family_df,
            plateau_rounds=plateau_rounds,
            target_fraction=target_fraction,
            max_plateau_std=max_plateau_std,
        )
        if summary.empty:
            continue

        all_summaries.append(summary)
        print_family_mcc_summary(summary, max_plateau_std)

        family_csv = outdir / f"mcc_plateau_summary_{family}.csv"
        summary.to_csv(family_csv, index=False)
        outputs.append(family_csv)
        outputs.extend(
            plot_family_panels(
                family,
                family_df,
                summary,
                outdir,
                x_tick_interval=x_tick_interval,
                figure_width=figure_width,
                figure_height=figure_height,
            )
        )

    if not all_summaries:
        raise RuntimeError(f"No strategy plots could be created under {root}")

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_csv = outdir / "mcc_plateau_summary_all_strategies.csv"
    combined_summary.to_csv(combined_csv, index=False)
    outputs.append(combined_csv)

    print("\n[INFO] Saved outputs:")
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
        "--plateau-rounds",
        type=int,
        default=DEFAULT_PLATEAU_ROUNDS,
        help="Number of final rounds used for plateau mean and standard deviation (default: 10)",
    )
    parser.add_argument(
        "--target-fraction",
        type=float,
        default=DEFAULT_TARGET_FRACTION,
        help="Fraction of plateau MCC used for rounds-to-target (default: 0.90)",
    )
    parser.add_argument(
        "--x-tick-interval",
        type=int,
        default=DEFAULT_X_TICK_INTERVAL,
        help="Distance between labelled x-axis ticks in rounds (default: 5)",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=DEFAULT_FIGURE_WIDTH,
        help=(
            "Figure width in inches. The compact default keeps communication "
            "rounds closer together and increases readability when fitted to page width "
            "(default: 12.0)."
        ),
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT,
        help="Figure height in inches (default: 7.4)",
    )
    parser.add_argument(
        "--max-plateau-std",
        type=float,
        default=None,
        help=(
            "Optional fixed maximum plateau MCC standard deviation. When set, only runs "
            "at or below this cutoff are considered stable; among them, the highest "
            "plateau MCC wins, with rounds-to-target as tie-breaker."
        ),
    )
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
    make_all_plots(
        args.root,
        args.out,
        plateau_rounds=args.plateau_rounds,
        target_fraction=args.target_fraction,
        x_tick_interval=args.x_tick_interval,
        max_plateau_std=args.max_plateau_std,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )

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