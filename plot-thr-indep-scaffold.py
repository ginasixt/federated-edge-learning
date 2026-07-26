#!/usr/bin/env python3
"""
Third Script!!

APA-style figures for the threshold-independent SCAFFOLD scalability study.

The preferred input is the run-level CSV created by final_test_set_eval.py:

    result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/all_test_results.csv

Required run-level columns:
    strategy, scaling_point, selection_metric, run,
    test_roc_auc, test_pr_auc, test_loss

The script also accepts the aggregated ``scaling_plot_statistics.csv`` created
by the previous plotting script. In that fallback mode, individual run points
cannot be reconstructed; the absolute figure therefore shows the recorded
minimum-to-maximum range around each mean.

Generated figures
-----------------
Figure 1: Absolute test performance
    A. ROC-AUC for models selected by validation ROC-AUC
    B. Average precision for models selected by validation AP
    C. Weighted loss for models selected by validation loss

Figure 2: Relative change from the two-client baseline
    A. Relative ROC-AUC change
    B. Relative average-precision change
    C. Relative increase in weighted loss

Figure 3: Run-to-run stability heatmap
    Maximum relative deviation of an individual run from the five-run mean
    (or coefficient of variation when --stability cv is selected).

Design principles
-----------------
* UHH-based colors, line widths, markers, axes, fonts, and legend styling
  matching the supplied configuration-selection figures
* no decorative figure titles inside the plots
* bold panel labels A-C without internal panel titles
* sans-serif font, restrained gridlines, no top/right borders
* vector PDF/SVG and high-resolution PNG output
* one neutral shared legend centered below all three panels
* deterministic horizontal jitter for the five individual runs
* no test-set model selection or threshold optimization

APA figure numbers, italicized titles, notes, and captions should normally be
added in the thesis document rather than embedded inside the graphic. Suggested
captions are written to ``suggested_figure_captions.txt``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULTS
# =============================================================================
DEFAULT_INPUT = Path(
    "result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/"
    "all_test_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "apa_scalability_figures"
DEFAULT_STRATEGY = "SCAFFOLD"
EXPECTED_RUNS = 5
BASELINE_CLIENTS = 2
DEFAULT_FORMATS: Tuple[str, ...] = ("pdf", "png")
PNG_DPI = 300
FIGURE_WIDTH = 12.0  # inches; matches the configuration-selection figures

# Match the visual language of the configuration-selection plots.
LINE_WIDTH = 1.80
MARKER_SIZE = 3.00
MARKER_EDGE_WIDTH = 0.50
INDIVIDUAL_POINT_SIZE = 12
ERRORBAR_WIDTH = 1.00
GRID_ALPHA = 0.22
GRID_LINEWIDTH = 0.55

# Each threshold-independent metric is paired only with the validation
# criterion used to choose its final checkpoint. Only the APA panel labels
# A-C are drawn; descriptive panel titles belong in the figure caption.
PANEL_SPECS: Tuple[Mapping[str, object], ...] = (
    {
        "panel": "A",
        "selection": "ROC",
        "metric": "test_roc_auc",
        "label": "ROC-AUC",
        "relative_label": "ROC-AUC change (%)",
        "heatmap_label": "ROC-AUC",
        "decimals": 4,
        "minimum_span": 0.004,
    },
    {
        "panel": "B",
        "selection": "PRROC",
        "metric": "test_pr_auc",
        "label": "AP",
        "relative_label": "AP change (%)",
        "heatmap_label": "AP",
        "decimals": 3,
        "minimum_span": 0.015,
    },
    {
        "panel": "C",
        "selection": "Loss",
        "metric": "test_loss",
        "label": "Weighted loss",
        "relative_label": "Weighted loss increase (%)",
        "heatmap_label": "Weighted loss",
        "decimals": 3,
        "minimum_span": 0.040,
    },
)

SELECTIONS = {str(spec["selection"]) for spec in PANEL_SPECS}
METRICS = {str(spec["metric"]) for spec in PANEL_SPECS}

# UHH-based palette used by the other thesis figures in this project.
# Each panel has one stable color, while individual runs use a transparent
# version of the corresponding panel color.
UHH_RED = "#E2001A"
UHH_BLUE = "#0271BB"
UHH_SLATE = "#3B515B"
UHH_VIOLET = "#7A3E9D"
UHH_ORANGE = "#F28E2B"
INK = "#202124"
REFERENCE_GRAY = "#6B7280"
INDIVIDUAL_RUN_GRAY = "#9AA0A6"
SHARED_LEGEND_GRAY = UHH_SLATE

PANEL_COLORS = {
    "ROC": UHH_RED,
    "PRROC": UHH_BLUE,
    "Loss": UHH_SLATE,
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "uhh_slate",
    ["#FFFFFF", "#DCE5E8", "#91A2A9", UHH_SLATE],
)


def panel_color(spec: Mapping[str, object]) -> str:
    """Return the fixed color assigned to one metric panel."""
    return PANEL_COLORS.get(str(spec["selection"]), UHH_SLATE)


# =============================================================================
# ARGUMENTS AND STYLE
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create APA-style threshold-independent SCAFFOLD scalability "
            "figures from final test-set results."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Run-level all_test_results.csv (preferred) or aggregated "
            "scaling_plot_statistics.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Strategy to plot (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--baseline-clients",
        type=int,
        default=BASELINE_CLIENTS,
        help="Scaling point used as the relative-change baseline (default: 2).",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
        help="Expected runs per scaling point and selection criterion (default: 5).",
    )
    parser.add_argument(
        "--stability",
        choices=("maxdev", "cv"),
        default="maxdev",
        help=(
            "Stability measure for Figure 3: maximum absolute relative "
            "deviation from the mean or coefficient of variation."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg", "tiff"),
        default=list(DEFAULT_FORMATS),
        help="Output formats (default: pdf png).",
    )
    parser.add_argument(
        "--font",
        default="Arial",
        help=(
            "Preferred sans-serif font. Arial is attempted first and falls "
            "back to Liberation Sans or DejaVu Sans when unavailable."
        ),
    )
    parser.add_argument(
        "--width",
        type=float,
        default=FIGURE_WIDTH,
        help="Figure width in inches (default: 12.0).",
    )
    return parser.parse_args()


def available_font(preferred: str) -> str:
    """Return the first available APA-compatible sans-serif font."""
    for candidate in (preferred, "Liberation Sans", "DejaVu Sans"):
        try:
            findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            continue
    return "DejaVu Sans"


def configure_matplotlib(font_name: str) -> None:
    """Apply the same publication styling as the other thesis plots."""
    mpl.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.90,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "savefig.dpi": PNG_DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# =============================================================================
# DATA HANDLING
# =============================================================================
def read_and_prepare(
    path: Path,
    strategy: str,
    expected_runs: int,
) -> Tuple[str, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Read either run-level or aggregated input.

    Returns
    -------
    mode:
        ``"raw"`` or ``"summary"``.
    raw:
        Filtered run-level rows, or None for aggregated input.
    summary:
        One row per strategy/scaling point/selection criterion.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run final_test_set_eval.py first or provide --input."
        )

    df = pd.read_csv(path)
    raw_required = {
        "strategy",
        "scaling_point",
        "selection_metric",
        "run",
        "test_roc_auc",
        "test_pr_auc",
        "test_loss",
    }
    summary_required = {
        "strategy",
        "scaling_point",
        "selection_metric",
        "n_runs",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_roc_auc_min",
        "test_roc_auc_max",
        "test_pr_auc_mean",
        "test_pr_auc_std",
        "test_pr_auc_min",
        "test_pr_auc_max",
        "test_loss_mean",
        "test_loss_std",
        "test_loss_min",
        "test_loss_max",
    }

    if raw_required.issubset(df.columns):
        mode = "raw"
        raw = validate_raw(df, strategy)
        summary = aggregate_raw(raw)
    elif summary_required.issubset(df.columns):
        mode = "summary"
        raw = None
        summary = validate_summary(df, strategy)
        ensure_summary_variability_columns(summary)
        print(
            "⚠️ Aggregated input detected. Individual run points cannot be "
            "reconstructed; Figure 1 will show min-to-max ranges instead."
        )
    else:
        missing_raw = sorted(raw_required - set(df.columns))
        missing_summary = sorted(summary_required - set(df.columns))
        raise ValueError(
            "Input is neither a supported run-level nor aggregated CSV.\n"
            f"Missing for run-level mode: {', '.join(missing_raw)}\n"
            f"Missing for summary mode: {', '.join(missing_summary)}"
        )

    validate_run_counts(summary, expected_runs)
    return mode, raw, summary.sort_values(
        ["scaling_point", "selection_metric"]
    ).reset_index(drop=True)


def validate_raw(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    raw = df.copy()
    for column in ("scaling_point", "run", *sorted(METRICS)):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    invalid = raw[["scaling_point", "run", *sorted(METRICS)]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Run-level CSV contains {int(invalid.sum())} invalid rows.")

    raw["scaling_point"] = raw["scaling_point"].astype(int)
    raw["run"] = raw["run"].astype(int)
    raw["selection_metric"] = raw["selection_metric"].astype(str)
    raw["strategy"] = raw["strategy"].astype(str)

    raw = raw[
        (raw["strategy"] == strategy)
        & (raw["selection_metric"].isin(SELECTIONS))
    ].copy()
    if raw.empty:
        raise ValueError(f"No supported rows found for strategy '{strategy}'.")

    key = ["strategy", "scaling_point", "selection_metric", "run"]
    duplicated = raw.duplicated(key, keep=False)
    if duplicated.any():
        rows = raw.loc[duplicated, key].sort_values(key)
        raise ValueError("Duplicate run rows found:\n" + rows.to_string(index=False))

    return raw.sort_values(["scaling_point", "selection_metric", "run"])


def validate_summary(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    summary = df.copy()
    numeric = [
        "scaling_point",
        "n_runs",
        *[
            f"{metric}_{suffix}"
            for metric in sorted(METRICS)
            for suffix in ("mean", "std", "min", "max")
        ],
    ]
    for column in numeric:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    invalid = summary[numeric].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Aggregated CSV contains {int(invalid.sum())} invalid rows.")

    summary["scaling_point"] = summary["scaling_point"].astype(int)
    summary["n_runs"] = summary["n_runs"].astype(int)
    summary["selection_metric"] = summary["selection_metric"].astype(str)
    summary["strategy"] = summary["strategy"].astype(str)
    summary = summary[
        (summary["strategy"] == strategy)
        & (summary["selection_metric"].isin(SELECTIONS))
    ].copy()
    if summary.empty:
        raise ValueError(f"No supported rows found for strategy '{strategy}'.")
    return summary


def aggregate_raw(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(
        ["strategy", "scaling_point", "selection_metric"], sort=True
    )
    records: List[Dict[str, object]] = []

    for (strategy, clients, selection), group in grouped:
        record: Dict[str, object] = {
            "strategy": strategy,
            "scaling_point": int(clients),
            "selection_metric": selection,
            "n_runs": int(group["run"].nunique()),
        }
        for metric in sorted(METRICS):
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            max_abs_deviation = float(np.max(np.abs(values - mean)))
            denominator = abs(mean)
            relative_max = (
                100.0 * max_abs_deviation / denominator
                if denominator > np.finfo(float).eps
                else float("nan")
            )
            cv = (
                100.0 * std / denominator
                if denominator > np.finfo(float).eps
                else float("nan")
            )
            record.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_min": minimum,
                    f"{metric}_max": maximum,
                    f"{metric}_max_abs_deviation": max_abs_deviation,
                    f"{metric}_relative_max_deviation_percent": relative_max,
                    f"{metric}_coefficient_of_variation_percent": cv,
                }
            )
        records.append(record)

    return pd.DataFrame.from_records(records)


def ensure_summary_variability_columns(summary: pd.DataFrame) -> None:
    """Create variability columns when an older aggregate file lacks them."""
    for metric in sorted(METRICS):
        mean = summary[f"{metric}_mean"].astype(float)
        std = summary[f"{metric}_std"].astype(float)
        minimum = summary[f"{metric}_min"].astype(float)
        maximum = summary[f"{metric}_max"].astype(float)
        denominator = mean.abs().replace(0.0, np.nan)

        max_dev_col = f"{metric}_max_abs_deviation"
        rel_col = f"{metric}_relative_max_deviation_percent"
        cv_col = f"{metric}_coefficient_of_variation_percent"

        if max_dev_col not in summary.columns:
            summary[max_dev_col] = np.maximum(
                (mean - minimum).abs(), (maximum - mean).abs()
            )
        if rel_col not in summary.columns:
            summary[rel_col] = 100.0 * summary[max_dev_col] / denominator
        if cv_col not in summary.columns:
            summary[cv_col] = 100.0 * std / denominator


def validate_run_counts(summary: pd.DataFrame, expected_runs: int) -> None:
    mismatches = summary[summary["n_runs"] != expected_runs]
    if not mismatches.empty:
        print(
            f"⚠️ {len(mismatches)} groups do not contain exactly "
            f"{expected_runs} runs:"
        )
        print(
            mismatches[
                ["scaling_point", "selection_metric", "n_runs"]
            ].to_string(index=False)
        )


def relevant_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep only the metric-selection pair used in the three main panels."""
    pieces = []
    for spec in PANEL_SPECS:
        selection = str(spec["selection"])
        metric = str(spec["metric"])
        subset = summary[summary["selection_metric"] == selection].copy()
        subset["display_metric"] = str(spec["label"])
        subset["metric_key"] = metric
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True)


# =============================================================================
# PLOT HELPERS
# =============================================================================
def client_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value:,}"


def client_positions(clients: Sequence[int]) -> np.ndarray:
    return np.log2(np.asarray(clients, dtype=float))


def set_client_axis(ax: plt.Axes, clients: Sequence[int], show_label: bool) -> None:
    positions = client_positions(clients)
    ax.set_xticks(positions)
    ax.set_xticklabels([client_label(int(v)) for v in clients])
    ax.set_xlim(float(positions.min() - 0.35), float(positions.max() + 0.35))
    if show_label:
        ax.set_xlabel("Number of clients (log2 scale)")
    else:
        ax.set_xlabel("")


def style_axis(ax: plt.Axes) -> None:
    """Use the same open-axis layout and restrained grid as the MCC plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.90)
    ax.spines["bottom"].set_linewidth(0.90)
    ax.tick_params(
        colors=INK,
        direction="out",
        labelsize=10.5,
        length=4.0,
        width=0.85,
    )
    ax.grid(True, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, linestyle="-")
    ax.set_axisbelow(True)


def set_panel_label(ax: plt.Axes, panel: str) -> None:
    """Place only the bold APA panel letter above the upper-left corner."""
    ax.text(
        -0.075,
        1.020,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def trimmed_decimal_formatter(max_decimals: int) -> FuncFormatter:
    """Format ticks without unnecessary trailing zeros.

    For example, 0.8240 is displayed as 0.824 and 0.8000 as 0.8.
    The maximum precision remains metric-specific so small differences are not
    hidden by forcing every panel to only two decimal places.
    """

    def _format(value: float, _position: object) -> str:
        if not np.isfinite(value):
            return ""
        rounded = f"{value:.{max_decimals}f}"
        trimmed = rounded.rstrip("0").rstrip(".")
        return "0" if trimmed in {"-0", ""} else trimmed

    return FuncFormatter(_format)


def padded_limits(
    values: Iterable[float],
    minimum_span: float,
    include_zero: bool = False,
    padding_fraction: float = 0.12,
) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-1.0, 1.0)

    low = float(np.min(arr))
    high = float(np.max(arr))
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)

    center = (low + high) / 2.0
    span = max(high - low, minimum_span)
    low = center - span / 2.0
    high = center + span / 2.0
    pad = span * padding_fraction
    return low - pad, high + pad


def deterministic_jitter(n: int, width: float = 0.18) -> np.ndarray:
    """Symmetric deterministic offsets in log2 client-axis units."""
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.linspace(-width / 2.0, width / 2.0, n)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, object] = {}
        if fmt == "png":
            kwargs["dpi"] = PNG_DPI
        elif fmt == "tiff":
            kwargs["dpi"] = PNG_DPI
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=fmt, **kwargs)
        print(f"   ✓ {path}")


# =============================================================================
# FIGURE 1: ABSOLUTE PERFORMANCE
# =============================================================================
def plot_absolute_performance(
    mode: str,
    raw: Optional[pd.DataFrame],
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> pd.DataFrame:
    clients = sorted(summary["scaling_point"].unique().tolist())
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width, width * 1.30),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.12, hspace=0.25)

    export_rows: List[Dict[str, object]] = []

    for index, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        selection = str(spec["selection"])
        metric = str(spec["metric"])
        color = panel_color(spec)
        subset = summary[summary["selection_metric"] == selection].sort_values(
            "scaling_point"
        )
        x = client_positions(subset["scaling_point"].to_numpy(dtype=int))
        means = subset[f"{metric}_mean"].to_numpy(dtype=float)

        if mode == "raw" and raw is not None:
            raw_subset = raw[raw["selection_metric"] == selection]
            all_values: List[float] = []
            for clients_value, x_center in zip(
                subset["scaling_point"].astype(int), x
            ):
                values = (
                    raw_subset[raw_subset["scaling_point"] == clients_value]
                    .sort_values("run")[metric]
                    .to_numpy(dtype=float)
                )
                jitter = deterministic_jitter(len(values))
                ax.scatter(
                    x_center + jitter,
                    values,
                    s=INDIVIDUAL_POINT_SIZE,
                    marker="o",
                    facecolor=REFERENCE_GRAY,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.65,
                    zorder=2,
                )
                all_values.extend(values.tolist())
        else:
            minimum = subset[f"{metric}_min"].to_numpy(dtype=float)
            maximum = subset[f"{metric}_max"].to_numpy(dtype=float)
            lower = means - minimum
            upper = maximum - means
            ax.errorbar(
                x,
                means,
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor=INDIVIDUAL_RUN_GRAY,
                elinewidth=ERRORBAR_WIDTH,
                capsize=2.5,
                capthick=0.8,
                alpha=0.80,
                zorder=2,
            )
            all_values = np.concatenate([minimum, maximum]).tolist()

        ax.plot(
            x,
            means,
            color=color,
            linewidth=LINE_WIDTH,
            alpha=0.97,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )

        style_axis(ax)
        set_panel_label(ax, str(spec["panel"]))
        ax.set_ylabel(str(spec["label"]))
        ax.yaxis.set_major_formatter(
            trimmed_decimal_formatter(int(spec["decimals"]))
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylim(
            *padded_limits(
                all_values,
                minimum_span=float(spec["minimum_span"]),
            )
        )
        set_client_axis(ax, clients, show_label=index == len(PANEL_SPECS) - 1)

        for _, row in subset.iterrows():
            export_rows.append(
                {
                    "panel": spec["panel"],
                    "selection_metric": selection,
                    "metric": metric,
                    "scaling_point": int(row["scaling_point"]),
                    "n_runs": int(row["n_runs"]),
                    "mean": float(row[f"{metric}_mean"]),
                    "std": float(row[f"{metric}_std"]),
                    "min": float(row[f"{metric}_min"]),
                    "max": float(row[f"{metric}_max"]),
                }
            )

    # One neutral, figure-level legend makes clear that the point/line
    # encoding applies identically to all three metric panels. Panel colors
    # identify the metric, while the legend explains the statistical summary.
    if mode == "raw":
        shared_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=INDIVIDUAL_RUN_GRAY,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=5.0,
                label="Individual run",
            ),
            Line2D(
                [0],
                [0],
                color=SHARED_LEGEND_GRAY,
                marker="o",
                markerfacecolor=SHARED_LEGEND_GRAY,
                markeredgecolor="white",
                markeredgewidth=MARKER_EDGE_WIDTH,
                linewidth=LINE_WIDTH,
                markersize=5.0,
                label="Mean across runs",
            ),
        ]
    else:
        shared_handles = [
            Line2D(
                [0],
                [0],
                color=INDIVIDUAL_RUN_GRAY,
                marker="|",
                linewidth=ERRORBAR_WIDTH,
                markersize=8,
                label="Observed min–max range",
            ),
            Line2D(
                [0],
                [0],
                color=SHARED_LEGEND_GRAY,
                marker="o",
                markerfacecolor=SHARED_LEGEND_GRAY,
                markeredgecolor="white",
                markeredgewidth=MARKER_EDGE_WIDTH,
                linewidth=LINE_WIDTH,
                markersize=5.0,
                label="Mean across runs",
            ),
        ]

    bottom_ax = axes[-1]
    ax_position = bottom_ax.get_position()
    legend_x = ax_position.x0 + ax_position.width / 2

    fig.legend(
        handles=shared_handles,
        loc="lower center",
        bbox_to_anchor=(legend_x, 0.04),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=12,
        borderaxespad=0.0,
        columnspacing=1.5,
        handlelength=2.2,
    )

    save_figure(
        fig,
        output_dir,
        "figure_1_absolute_threshold_independent_performance",
        formats,
    )
    plt.close(fig)
    return pd.DataFrame(export_rows)


# =============================================================================
# FIGURE 2: RELATIVE CHANGE
# =============================================================================
def compute_relative_changes(
    summary: pd.DataFrame,
    baseline_clients: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in PANEL_SPECS:
        selection = str(spec["selection"])
        metric = str(spec["metric"])
        subset = summary[summary["selection_metric"] == selection].sort_values(
            "scaling_point"
        )
        baseline_rows = subset[subset["scaling_point"] == baseline_clients]
        if baseline_rows.empty:
            raise ValueError(
                f"Baseline scaling point {baseline_clients} is missing for {selection}."
            )
        baseline = float(baseline_rows.iloc[0][f"{metric}_mean"])
        if abs(baseline) <= np.finfo(float).eps:
            raise ValueError(f"Baseline mean is zero for metric {metric}.")

        for _, row in subset.iterrows():
            mean = float(row[f"{metric}_mean"])
            relative = 100.0 * (mean - baseline) / abs(baseline)
            rows.append(
                {
                    "panel": spec["panel"],
                    "selection_metric": selection,
                    "metric": metric,
                    "scaling_point": int(row["scaling_point"]),
                    "baseline_clients": int(baseline_clients),
                    "baseline_mean": baseline,
                    "mean": mean,
                    "relative_change_percent": relative,
                }
            )
    return pd.DataFrame(rows)


def plot_relative_changes(
    relative: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> None:
    clients = sorted(relative["scaling_point"].unique().tolist())
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width, width * 1.30),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.075, hspace=0.25)

    for index, (ax, spec) in enumerate(zip(axes, PANEL_SPECS)):
        selection = str(spec["selection"])
        color = panel_color(spec)
        subset = relative[relative["selection_metric"] == selection].sort_values(
            "scaling_point"
        )
        x = client_positions(subset["scaling_point"].to_numpy(dtype=int))
        values = subset["relative_change_percent"].to_numpy(dtype=float)

        ax.axhline(0.0, color=REFERENCE_GRAY, linewidth=0.90, linestyle="--", zorder=1)
        ax.plot(
            x,
            values,
            color=color,
            linewidth=LINE_WIDTH,
            alpha=0.97,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )

        style_axis(ax)
        set_panel_label(ax, str(spec["panel"]))
        ax.set_ylabel(str(spec["relative_label"]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(trimmed_decimal_formatter(1))

        observed_span = float(np.ptp(values)) if len(values) else 0.0
        minimum_span = max(0.6, observed_span * 0.25)
        ax.set_ylim(
            *padded_limits(
                values,
                minimum_span=minimum_span,
                include_zero=True,
                padding_fraction=0.15,
            )
        )
        set_client_axis(ax, clients, show_label=index == len(PANEL_SPECS) - 1)

        # Label the final scaling point only; this emphasizes the endpoint
        # without turning the figure into a data table.
        if len(values):
            final_value = values[-1]
            display_final = 0.0 if abs(final_value) < 0.05 else final_value
            offset = 6
            ax.annotate(
                f"{display_final:+.1f}%",
                xy=(x[-1], final_value),
                xytext=(-4, offset),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=9.0,
                color=color,
            )

    save_figure(
        fig,
        output_dir,
        "figure_2_relative_change_from_baseline",
        formats,
    )
    plt.close(fig)


# =============================================================================
# FIGURE 3: STABILITY HEATMAP
# =============================================================================
def compute_stability(
    summary: pd.DataFrame,
    measure: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    suffix = (
        "relative_max_deviation_percent"
        if measure == "maxdev"
        else "coefficient_of_variation_percent"
    )
    for spec in PANEL_SPECS:
        selection = str(spec["selection"])
        metric = str(spec["metric"])
        subset = summary[summary["selection_metric"] == selection].sort_values(
            "scaling_point"
        )
        column = f"{metric}_{suffix}"
        for _, row in subset.iterrows():
            rows.append(
                {
                    "panel": spec["panel"],
                    "display_metric": spec["heatmap_label"],
                    "selection_metric": selection,
                    "metric": metric,
                    "scaling_point": int(row["scaling_point"]),
                    "stability_measure": measure,
                    "stability_percent": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def heatmap_text(value: float) -> str:
    if value < 0.10:
        return f"{value:.2f}"
    if value < 10.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def plot_stability_heatmap(
    stability: pd.DataFrame,
    measure: str,
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> None:
    clients = sorted(stability["scaling_point"].unique().tolist())
    row_order = [str(spec["heatmap_label"]) for spec in PANEL_SPECS]
    pivot = stability.pivot(
        index="display_metric",
        columns="scaling_point",
        values="stability_percent",
    ).reindex(index=row_order, columns=clients)

    matrix = pivot.to_numpy(dtype=float)
    finite = matrix[np.isfinite(matrix)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    vmax = max(maximum * 1.05, 0.5)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(width, max(3.0, width * 0.48)))
    fig.subplots_adjust(left=0.20, right=0.88, top=0.92, bottom=0.25)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=HEATMAP_CMAP,
        norm=norm,
    )

    ax.set_xticks(np.arange(len(clients)))
    ax.set_xticklabels([client_label(value) for value in clients], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Number of clients", fontsize=12.0)
    ax.set_ylabel("")
    ax.tick_params(length=0, colors=INK, labelsize=10.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                label = "–"
                text_color = INK
            else:
                label = heatmap_text(float(value))
                text_color = "white" if norm(value) > 0.58 else INK
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9.2,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.034, pad=0.025)
    if measure == "maxdev":
        cbar.set_label("Max. deviation from mean (%)", rotation=90, fontsize=12.0)
    else:
        cbar.set_label("Coefficient of variation (%)", rotation=90, fontsize=12.0)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=10.5, width=0.7, length=3.0)

    save_figure(
        fig,
        output_dir,
        "figure_3_run_stability_heatmap",
        formats,
    )
    plt.close(fig)


# =============================================================================
# CAPTIONS AND MAIN
# =============================================================================
def write_caption_suggestions(
    output_dir: Path,
    mode: str,
    expected_runs: int,
    stability_measure: str,
    baseline_clients: int,
    strategy: str,
) -> None:
    if mode == "raw":
        spread_sentence = (
            "Small gray points represent individual runs; filled colored circles "
            "and connecting lines represent means across runs."
        )
    else:
        spread_sentence = (
            "Gray ranges represent the observed minimum and maximum; filled colored "
            "circles and connecting lines represent means across runs."
        )

    stability_sentence = (
        "Each cell reports the maximum absolute deviation of an individual "
        "run from the mean, expressed as a percentage of the mean."
        if stability_measure == "maxdev"
        else "Each cell reports the sample standard deviation as a percentage "
        "of the mean (coefficient of variation)."
    )

    text = f"""Suggested APA-style figure captions
===================================

Figure 1
Threshold-independent test performance of {strategy} across client scaling points.
Note. Panel A shows ROC-AUC for checkpoints selected by validation ROC-AUC,
Panel B shows average precision for checkpoints selected by validation average
precision, and Panel C shows weighted cross-entropy loss for checkpoints
selected by validation loss. {spread_sentence} Each scaling point contains up
to n = {expected_runs} independent runs. The client axis is logarithmic to base 2.
Restricted y-axis ranges are used to make small differences visible.

Figure 2
Relative change in threshold-independent {strategy} test performance from the
{baseline_clients}-client baseline.
Note. Values are calculated from the mean across runs at each scaling point.
Positive values in Panels A and B indicate higher ROC-AUC or average precision
than at {baseline_clients} clients. Positive values in Panel C indicate an increase in weighted
loss and therefore worse performance. The dashed horizontal line denotes no
change from the baseline.

Figure 3
Run-to-run variability of threshold-independent {strategy} test performance.
Note. {stability_sentence} ROC-AUC, average precision, and weighted loss are
reported for checkpoints selected by their corresponding validation metric.
Values are percentages.
"""
    (output_dir / "suggested_figure_captions.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    font_name = available_font(args.font)
    configure_matplotlib(font_name)

    print("=" * 78)
    print(f"APA-STYLE {args.strategy.upper()} SCALABILITY FIGURES")
    print(f"Input     : {args.input}")
    print(f"Output    : {args.output_dir}")
    print(f"Strategy  : {args.strategy}")
    print(f"Font      : {font_name}")
    print(f"Formats   : {', '.join(args.formats)}")
    print("=" * 78)

    mode, raw, summary = read_and_prepare(
        args.input,
        strategy=args.strategy,
        expected_runs=args.expected_runs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the complete aggregate table and export focused plotting tables.
    summary.to_csv(args.output_dir / "all_aggregated_test_statistics.csv", index=False)

    absolute_table = plot_absolute_performance(
        mode=mode,
        raw=raw,
        summary=summary,
        output_dir=args.output_dir,
        formats=args.formats,
        width=args.width,
    )
    absolute_table.to_csv(
        args.output_dir / "figure_1_absolute_values.csv", index=False
    )

    relative = compute_relative_changes(summary, args.baseline_clients)
    relative.to_csv(args.output_dir / "figure_2_relative_changes.csv", index=False)
    plot_relative_changes(
        relative=relative,
        output_dir=args.output_dir,
        formats=args.formats,
        width=args.width,
    )

    stability = compute_stability(summary, args.stability)
    stability.to_csv(args.output_dir / "figure_3_run_stability.csv", index=False)
    plot_stability_heatmap(
        stability=stability,
        measure=args.stability,
        output_dir=args.output_dir,
        formats=args.formats,
        width=args.width,
    )

    write_caption_suggestions(
        output_dir=args.output_dir,
        mode=mode,
        expected_runs=args.expected_runs,
        stability_measure=args.stability,
        baseline_clients=args.baseline_clients,
        strategy=args.strategy,
    )

    print("=" * 78)
    print("Finished. Main outputs:")
    print("  Figure 1: absolute threshold-independent performance")
    print("  Figure 2: relative change from the two-client baseline")
    print("  Figure 3: run-to-run stability heatmap")
    print("  Captions: suggested_figure_captions.txt")
    print("=" * 78)


if __name__ == "__main__":
    main()
