#!/usr/bin/env python3
"""
APA-style comparison figures for FedProx, SCAFFOLD, and FedAdam.

The script combines six already aggregated CSV files:

Threshold-dependent inputs
--------------------------
    FedProx_threshold_dependent_aggregate.csv
    SCAFFOLD_threshold_dependent_aggregate.csv
    FedAdam_threshold_dependent_aggregate.csv

Threshold-independent inputs
----------------------------
    FedProx_all_test_aggregate.csv
    SCAFFOLD_all_test_aggregate.csv
    FedAdam_all_test_aggregate.csv

Only mean lines are plotted; individual-run points are intentionally omitted.
By default, the script uses only scaling points present in all six input files.
This normally yields the common comparison range from 2 to 16,384 clients.

Generated figures
-----------------
Figure 1: MCC-optimal operating point
    A. Mean validation-selected decision threshold
    B. Mean test MCC
    C. Mean test recall and specificity

Figure 2: Fixed validation recall requirement
    A. Mean validation-selected decision threshold
    B. Mean test recall and specificity, with the validation-recall reference

Figure 3: Threshold-independent test performance
    A. Mean test ROC-AUC for validation-ROC-selected checkpoints
    B. Mean test AP for validation-AP-selected checkpoints
    C. Mean weighted test loss for validation-loss-selected checkpoints

The visual design follows the existing strategy-specific thesis figures:
Arial-compatible typography, UHH-based restrained colors, bold panel labels,
base-2 client axis, restricted panel-specific y ranges, no internal titles,
and PDF plus high-resolution PNG output.

Default directory layout
------------------------
result/all_strategy_comparison_output/
    FedProx_threshold_dependent_aggregate.csv
    SCAFFOLD_threshold_dependent_aggregate.csv
    FedAdam_threshold_dependent_aggregate.csv
    FedProx_all_test_aggregate.csv
    SCAFFOLD_all_test_aggregate.csv
    FedAdam_all_test_aggregate.csv

Example
-------
python3 plot-combined-strategy-comparison.py

Or provide all six paths explicitly:

python3 plot-combined-strategy-comparison.py \
  --fedprox-threshold-dependent path/to/FedProx_threshold_dependent_aggregate.csv \
  --scaffold-threshold-dependent path/to/SCAFFOLD_threshold_dependent_aggregate.csv \
  --fedadam-threshold-dependent path/to/FedAdam_threshold_dependent_aggregate.csv \
  --fedprox-threshold-independent path/to/FedProx_all_test_aggregate.csv \
  --scaffold-threshold-independent path/to/SCAFFOLD_all_test_aggregate.csv \
  --fedadam-threshold-independent path/to/FedAdam_all_test_aggregate.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULTS AND VISUAL CONSTANTS
# =============================================================================
DEFAULT_INPUT_DIR = Path("result/all_strategy_comparison")
DEFAULT_OUTPUT_DIR = Path("result/all_strategy_comparison_output")
DEFAULT_FORMATS: Tuple[str, ...] = ("pdf", "png")
PNG_DPI = 300
FIGURE_WIDTH = 12.0
MINIMUM_VALIDATION_RECALL = 0.80

LINE_WIDTH = 1.80
MARKER_SIZE = 4.20
MARKER_EDGE_WIDTH = 0.60
GRID_ALPHA = 0.22
GRID_LINEWIDTH = 0.55

UHH_RED = "#E2001A"
UHH_BLUE = "#0271BB"
UHH_SLATE = "#3B515B"
INK = "#202124"
REFERENCE_GRAY = "#6B7280"

STRATEGY_ORDER: Tuple[str, ...] = ("FedProx", "SCAFFOLD", "FedAdam")
STRATEGY_STYLE: Mapping[str, Mapping[str, str]] = {
    "FedProx": {"color": UHH_BLUE, "marker": "o"},
    "SCAFFOLD": {"color": UHH_RED, "marker": "s"},
    "FedAdam": {"color": UHH_SLATE, "marker": "^"},
}

THRESHOLD_REQUIRED_COLUMNS = {
    "strategy",
    "threshold_regime",
    "scaling_point",
    "n_runs",
    "selected_threshold_mean",
    "test_mcc_mean",
    "test_recall_mean",
    "test_specificity_mean",
}

THRESHOLD_INDEPENDENT_REQUIRED_COLUMNS = {
    "strategy",
    "scaling_point",
    "selection_metric",
    "n_runs",
    "test_roc_auc_mean",
    "test_pr_auc_mean",
    "test_loss_mean",
}

THRESHOLD_REGIME_ALIASES = {
    "maximum_validation_mcc": "mcc_optimal",
    "maximum_mcc": "mcc_optimal",
    "mcc-optimal": "mcc_optimal",
    "recall_constrained": "minimum_recall",
    "fixed_recall": "minimum_recall",
    "minimum-recall": "minimum_recall",
}

SELECTION_ALIASES = {
    "roc": "ROC",
    "roc_auc": "ROC",
    "prroc": "PRROC",
    "ap": "PRROC",
    "average_precision": "PRROC",
    "loss": "Loss",
    "weighted_loss": "Loss",
}


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create three APA-style mean-line comparison figures for FedProx, "
            "SCAFFOLD, and FedAdam."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--fedprox-threshold-dependent", type=Path)
    parser.add_argument("--scaffold-threshold-dependent", type=Path)
    parser.add_argument("--fedadam-threshold-dependent", type=Path)
    parser.add_argument("--fedprox-threshold-independent", type=Path)
    parser.add_argument("--scaffold-threshold-independent", type=Path)
    parser.add_argument("--fedadam-threshold-independent", type=Path)

    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=MINIMUM_VALIDATION_RECALL,
        help="Prespecified validation recall requirement (default: 0.80).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg", "tiff"),
        default=list(DEFAULT_FORMATS),
    )
    parser.add_argument("--font", default="Arial")
    parser.add_argument("--width", type=float, default=FIGURE_WIDTH)
    return parser.parse_args()


# =============================================================================
# INPUT DISCOVERY
# =============================================================================
def expected_filename(strategy: str, kind: str) -> str:
    if kind == "threshold_dependent":
        return f"{strategy}_threshold_dependent_aggregate.csv"
    if kind == "threshold_independent":
        return f"{strategy}_all_test_aggregate.csv"
    raise ValueError(f"Unknown input kind: {kind}")


def strategy_tokens(strategy: str) -> Tuple[str, ...]:
    if strategy == "SCAFFOLD":
        return ("scaffold",)
    return (strategy.lower(),)


def resolve_input_path(
    explicit_path: Optional[Path],
    input_dir: Path,
    strategy: str,
    kind: str,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    exact = input_dir / expected_filename(strategy, kind)
    if exact.is_file():
        return exact

    base_name = (
        "threshold_dependent_aggregate"
        if kind == "threshold_dependent"
        else "all_test_aggregate"
    )

    # Also support one subdirectory per strategy with the original filename.
    subdir_candidates = [
        input_dir / strategy / f"{base_name}.csv",
        input_dir / strategy.lower() / f"{base_name}.csv",
    ]
    for candidate in subdir_candidates:
        if candidate.is_file():
            return candidate

    matches: List[Path] = []
    for candidate in input_dir.rglob("*.csv"):
        lowered = candidate.name.lower()
        if base_name not in lowered:
            continue
        if any(token in lowered or token in str(candidate.parent).lower()
               for token in strategy_tokens(strategy)):
            matches.append(candidate)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        formatted = "\n  ".join(str(path) for path in unique_matches)
        raise RuntimeError(
            f"Multiple possible {kind} inputs found for {strategy}:\n  {formatted}\n"
            "Pass the intended file explicitly on the command line."
        )

    raise FileNotFoundError(
        f"Could not find the {kind} aggregate for {strategy}.\n"
        f"Expected, for example:\n  {exact}\n"
        "Alternatively pass the file explicitly."
    )


def resolve_all_inputs(args: argparse.Namespace) -> Dict[str, Dict[str, Path]]:
    explicit = {
        "FedProx": {
            "threshold_dependent": args.fedprox_threshold_dependent,
            "threshold_independent": args.fedprox_threshold_independent,
        },
        "SCAFFOLD": {
            "threshold_dependent": args.scaffold_threshold_dependent,
            "threshold_independent": args.scaffold_threshold_independent,
        },
        "FedAdam": {
            "threshold_dependent": args.fedadam_threshold_dependent,
            "threshold_independent": args.fedadam_threshold_independent,
        },
    }

    resolved: Dict[str, Dict[str, Path]] = {}
    for strategy in STRATEGY_ORDER:
        resolved[strategy] = {}
        for kind in ("threshold_dependent", "threshold_independent"):
            resolved[strategy][kind] = resolve_input_path(
                explicit[strategy][kind], args.input_dir, strategy, kind
            )
    return resolved


# =============================================================================
# DATA READING AND VALIDATION
# =============================================================================
def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    disposable = [
        column
        for column in cleaned.columns
        if column.lower().startswith("unnamed") or column.lower() == "index"
    ]
    return cleaned.drop(columns=disposable, errors="ignore")


def canonical_strategy(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
        "fedadam": "FedAdam",
    }
    if text not in mapping:
        raise ValueError(f"Unknown strategy value: {value!r}")
    return mapping[text]


def read_threshold_aggregate(path: Path, expected_strategy: str) -> pd.DataFrame:
    frame = clean_columns(pd.read_csv(path))
    missing = THRESHOLD_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Threshold-dependent file is missing columns ({path}):\n  "
            + "\n  ".join(sorted(missing))
        )

    frame = frame.copy()
    frame["strategy"] = frame["strategy"].map(canonical_strategy)
    observed = set(frame["strategy"].dropna().unique())
    if observed != {expected_strategy}:
        raise ValueError(
            f"Expected strategy {expected_strategy} in {path}, found {sorted(observed)}."
        )

    frame["threshold_regime"] = (
        frame["threshold_regime"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(THRESHOLD_REGIME_ALIASES)
    )
    frame = frame[
        frame["threshold_regime"].isin({"mcc_optimal", "minimum_recall"})
    ].copy()

    numeric_columns = [
        "scaling_point",
        "n_runs",
        "selected_threshold_mean",
        "test_mcc_mean",
        "test_recall_mean",
        "test_specificity_mean",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any():
        invalid = frame[frame[numeric_columns].isna().any(axis=1)]
        raise ValueError(
            f"Invalid threshold-dependent numeric values in {path}:\n"
            + invalid.head(10).to_string(index=False)
        )

    frame["scaling_point"] = frame["scaling_point"].astype(int)
    frame["n_runs"] = frame["n_runs"].astype(int)

    duplicate_key = ["strategy", "threshold_regime", "scaling_point"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate threshold aggregate rows in {path}:\n"
            + frame.loc[duplicates, duplicate_key].to_string(index=False)
        )

    regimes = set(frame["threshold_regime"].unique())
    if regimes != {"mcc_optimal", "minimum_recall"}:
        raise ValueError(
            f"Expected both threshold regimes in {path}, found {sorted(regimes)}."
        )
    return frame.sort_values(["threshold_regime", "scaling_point"]).reset_index(drop=True)


def canonical_selection(value: Any) -> str:
    text = str(value).strip()
    key = text.lower().replace("-", "_").replace(" ", "_")
    return SELECTION_ALIASES.get(key, text)


def read_threshold_independent_aggregate(
    path: Path, expected_strategy: str
) -> pd.DataFrame:
    frame = clean_columns(pd.read_csv(path))
    missing = THRESHOLD_INDEPENDENT_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Threshold-independent file is missing columns ({path}):\n  "
            + "\n  ".join(sorted(missing))
        )

    frame = frame.copy()
    frame["strategy"] = frame["strategy"].map(canonical_strategy)
    observed = set(frame["strategy"].dropna().unique())
    if observed != {expected_strategy}:
        raise ValueError(
            f"Expected strategy {expected_strategy} in {path}, found {sorted(observed)}."
        )

    frame["selection_metric"] = frame["selection_metric"].map(canonical_selection)
    frame = frame[frame["selection_metric"].isin({"ROC", "PRROC", "Loss"})].copy()

    numeric_columns = [
        "scaling_point",
        "n_runs",
        "test_roc_auc_mean",
        "test_pr_auc_mean",
        "test_loss_mean",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric_columns].isna().any().any():
        invalid = frame[frame[numeric_columns].isna().any(axis=1)]
        raise ValueError(
            f"Invalid threshold-independent numeric values in {path}:\n"
            + invalid.head(10).to_string(index=False)
        )

    frame["scaling_point"] = frame["scaling_point"].astype(int)
    frame["n_runs"] = frame["n_runs"].astype(int)

    duplicate_key = ["strategy", "selection_metric", "scaling_point"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Duplicate threshold-independent aggregate rows in {path}:\n"
            + frame.loc[duplicates, duplicate_key].to_string(index=False)
        )

    selections = set(frame["selection_metric"].unique())
    if selections != {"ROC", "PRROC", "Loss"}:
        raise ValueError(
            f"Expected ROC, PRROC, and Loss rows in {path}, found {sorted(selections)}."
        )
    return frame.sort_values(["selection_metric", "scaling_point"]).reset_index(drop=True)


def read_all_data(
    paths: Mapping[str, Mapping[str, Path]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    threshold_frames: List[pd.DataFrame] = []
    independent_frames: List[pd.DataFrame] = []

    for strategy in STRATEGY_ORDER:
        threshold_frames.append(
            read_threshold_aggregate(
                paths[strategy]["threshold_dependent"], strategy
            )
        )
        independent_frames.append(
            read_threshold_independent_aggregate(
                paths[strategy]["threshold_independent"], strategy
            )
        )

    threshold = pd.concat(threshold_frames, ignore_index=True)
    independent = pd.concat(independent_frames, ignore_index=True)
    return threshold, independent


def common_scaling_points(
    threshold: pd.DataFrame, independent: pd.DataFrame
) -> List[int]:
    point_sets: List[set[int]] = []

    for strategy in STRATEGY_ORDER:
        for regime in ("mcc_optimal", "minimum_recall"):
            subset = threshold[
                (threshold["strategy"] == strategy)
                & (threshold["threshold_regime"] == regime)
            ]
            point_sets.append(set(subset["scaling_point"].astype(int)))

        for selection in ("ROC", "PRROC", "Loss"):
            subset = independent[
                (independent["strategy"] == strategy)
                & (independent["selection_metric"] == selection)
            ]
            point_sets.append(set(subset["scaling_point"].astype(int)))

    common = sorted(set.intersection(*point_sets)) if point_sets else []
    if not common:
        raise ValueError("The six inputs have no common scaling points.")

    for value in common:
        if value <= 0 or value & (value - 1) != 0:
            raise ValueError(
                f"Scaling point {value} is not a positive power of two."
            )
    return common


def filter_common_points(
    threshold: pd.DataFrame,
    independent: pd.DataFrame,
    clients: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    allowed = set(int(value) for value in clients)
    return (
        threshold[threshold["scaling_point"].isin(allowed)].copy(),
        independent[independent["scaling_point"].isin(allowed)].copy(),
    )


# =============================================================================
# PLOT STYLE HELPERS
# =============================================================================
def available_font(preferred: str) -> str:
    for candidate in (preferred, "Liberation Sans", "DejaVu Sans"):
        try:
            findfont(candidate, fallback_to_default=False)
            return candidate
        except ValueError:
            continue
    return "DejaVu Sans"


def configure_matplotlib(font_name: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "axes.linewidth": 0.90,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def client_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value:,}"


def client_positions(clients: Sequence[int]) -> np.ndarray:
    return np.log2(np.asarray(clients, dtype=float))


def set_client_axis(ax: plt.Axes, clients: Sequence[int], show_label: bool) -> None:
    positions = client_positions(clients)
    ax.set_xticks(positions)
    ax.set_xticklabels([client_label(int(value)) for value in clients])
    ax.set_xlim(float(positions.min() - 0.35), float(positions.max() + 0.35))
    ax.set_xlabel("Number of clients (log2 scale)" if show_label else "")


def style_axis(ax: plt.Axes) -> None:
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
    def _format(value: float, _position: object) -> str:
        if not np.isfinite(value):
            return ""
        text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text

    return FuncFormatter(_format)


def padded_limits(
    values: Iterable[float],
    minimum_span: float,
    include_value: Optional[float] = None,
    padding_fraction: float = 0.12,
    hard_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (0.0, 1.0)

    low = float(np.min(array))
    high = float(np.max(array))
    if include_value is not None:
        low = min(low, include_value)
        high = max(high, include_value)

    center = (low + high) / 2.0
    span = max(high - low, minimum_span)
    low = center - span / 2.0
    high = center + span / 2.0
    padding = span * padding_fraction
    low -= padding
    high += padding

    if hard_bounds is not None:
        low = max(hard_bounds[0], low)
        high = min(hard_bounds[1], high)
    if np.isclose(low, high):
        high = low + minimum_span
    return low, high


def strategy_series(
    frame: pd.DataFrame,
    strategy: str,
    clients: Sequence[int],
    metric: str,
) -> np.ndarray:
    subset = (
        frame[frame["strategy"] == strategy]
        .set_index("scaling_point")
        .reindex(clients)
    )
    if subset[metric].isna().any():
        missing = [
            int(client)
            for client, value in zip(clients, subset[metric].tolist())
            if pd.isna(value)
        ]
        raise ValueError(
            f"Missing {metric} values for {strategy} at scaling points {missing}."
        )
    return subset[metric].to_numpy(dtype=float)


def plot_strategy_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    clients: Sequence[int],
    metric: str,
) -> List[float]:
    x = client_positions(clients)
    all_values: List[float] = []
    for strategy in STRATEGY_ORDER:
        style = STRATEGY_STYLE[strategy]
        y = strategy_series(frame, strategy, clients, metric)
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=LINE_WIDTH,
            linestyle="-",
            alpha=0.97,
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=3,
        )
        all_values.extend(y.tolist())
    return all_values


def plot_recall_specificity(
    ax: plt.Axes,
    frame: pd.DataFrame,
    clients: Sequence[int],
) -> List[float]:
    x = client_positions(clients)
    all_values: List[float] = []
    for strategy in STRATEGY_ORDER:
        color = STRATEGY_STYLE[strategy]["color"]
        recall = strategy_series(frame, strategy, clients, "test_recall_mean")
        specificity = strategy_series(
            frame, strategy, clients, "test_specificity_mean"
        )
        ax.plot(
            x,
            recall,
            color=color,
            linewidth=LINE_WIDTH,
            linestyle="-",
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            alpha=0.97,
            zorder=3,
        )
        ax.plot(
            x,
            specificity,
            color=color,
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker="s",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGE_WIDTH,
            alpha=0.97,
            zorder=3,
        )
        all_values.extend(recall.tolist())
        all_values.extend(specificity.tolist())
    return all_values


def strategy_legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for strategy in STRATEGY_ORDER:
        style = STRATEGY_STYLE[strategy]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=LINE_WIDTH,
                marker=style["marker"],
                markersize=5.0,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=MARKER_EDGE_WIDTH,
                label="FedAvg" if strategy == "FedProx" else strategy,
            )
        )
    return handles


def metric_legend_handles(include_reference: Optional[float] = None) -> List[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=REFERENCE_GRAY,
            linewidth=LINE_WIDTH,
            linestyle="-",
            marker="o",
            markersize=4.5,
            markerfacecolor=REFERENCE_GRAY,
            markeredgecolor="white",
            label="Recall",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_GRAY,
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker="s",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=REFERENCE_GRAY,
            label="Specificity",
        ),
    ]
    if include_reference is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REFERENCE_GRAY,
                linewidth=1.0,
                linestyle=":",
                label=f"Validation recall requirement ({include_reference:.2f})",
            )
        )
    return handles


def add_strategy_legend(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    bottom_axis = axes[-1]
    position = bottom_axis.get_position()
    legend_x = position.x0 + position.width / 2.0
    fig.legend(
        handles=strategy_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(legend_x, 0.018),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=9.2,
        borderaxespad=0.0,
        columnspacing=1.7,
        handlelength=2.4,
    )


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        kwargs: Dict[str, Any] = {}
        if fmt == "png":
            kwargs["dpi"] = PNG_DPI
        elif fmt == "tiff":
            kwargs["dpi"] = PNG_DPI
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(output_path, format=fmt, **kwargs)
        print(f"  ✓ {output_path}")


# =============================================================================
# FIGURE 1: MCC-OPTIMAL OPERATING POINT
# =============================================================================
def plot_mcc_optimal(
    threshold: pd.DataFrame,
    clients: Sequence[int],
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> pd.DataFrame:
    regime = threshold[threshold["threshold_regime"] == "mcc_optimal"].copy()

    fig, axes = plt.subplots(
        3, 1, figsize=(width, 8.4), sharex=True, constrained_layout=False
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.975, bottom=0.135, hspace=0.28
    )

    values_a = plot_strategy_metric(
        axes[0], regime, clients, "selected_threshold_mean"
    )
    style_axis(axes[0])
    set_panel_label(axes[0], "A")
    axes[0].set_ylabel("Decision threshold")
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_formatter(trimmed_decimal_formatter(2))
    axes[0].set_ylim(
        *padded_limits(values_a, minimum_span=0.08, hard_bounds=(0.0, 1.0))
    )
    set_client_axis(axes[0], clients, show_label=False)

    values_b = plot_strategy_metric(axes[1], regime, clients, "test_mcc_mean")
    style_axis(axes[1])
    set_panel_label(axes[1], "B")
    axes[1].set_ylabel("MCC")
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].yaxis.set_major_formatter(trimmed_decimal_formatter(3))
    axes[1].set_ylim(
        *padded_limits(values_b, minimum_span=0.025, hard_bounds=(-1.0, 1.0))
    )
    set_client_axis(axes[1], clients, show_label=False)

    values_c = plot_recall_specificity(axes[2], regime, clients)
    style_axis(axes[2])
    set_panel_label(axes[2], "C")
    axes[2].set_ylabel("Metric value")
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[2].yaxis.set_major_formatter(trimmed_decimal_formatter(2))
    axes[2].set_ylim(
        *padded_limits(values_c, minimum_span=0.12, hard_bounds=(0.0, 1.0))
    )
    set_client_axis(axes[2], clients, show_label=True)
    axes[2].legend(
        handles=metric_legend_handles(),
        loc="best",
        frameon=False,
        fontsize=9.2,
    )

    add_strategy_legend(fig, axes)
    save_figure(
        fig,
        output_dir,
        "figure_1_strategy_comparison_mcc_optimal_operating_point",
        formats,
    )
    plt.close(fig)

    columns = [
        "strategy",
        "threshold_regime",
        "scaling_point",
        "n_runs",
        "selected_threshold_mean",
        "test_mcc_mean",
        "test_recall_mean",
        "test_specificity_mean",
    ]
    return regime[columns].sort_values(["strategy", "scaling_point"])


# =============================================================================
# FIGURE 2: FIXED VALIDATION RECALL REQUIREMENT
# =============================================================================
def plot_fixed_recall(
    threshold: pd.DataFrame,
    clients: Sequence[int],
    minimum_recall: float,
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> pd.DataFrame:
    regime = threshold[threshold["threshold_regime"] == "minimum_recall"].copy()

    fig, axes = plt.subplots(
        2, 1, figsize=(width, 6.2), sharex=True, constrained_layout=False
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.970, bottom=0.175, hspace=0.30
    )

    values_a = plot_strategy_metric(
        axes[0], regime, clients, "selected_threshold_mean"
    )
    style_axis(axes[0])
    set_panel_label(axes[0], "A")
    axes[0].set_ylabel("Decision threshold")
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].yaxis.set_major_formatter(trimmed_decimal_formatter(2))
    axes[0].set_ylim(
        *padded_limits(values_a, minimum_span=0.08, hard_bounds=(0.0, 1.0))
    )
    set_client_axis(axes[0], clients, show_label=False)

    values_b = plot_recall_specificity(axes[1], regime, clients)
    axes[1].axhline(
        minimum_recall,
        color=REFERENCE_GRAY,
        linewidth=1.0,
        linestyle=":",
        zorder=1,
    )
    style_axis(axes[1])
    set_panel_label(axes[1], "B")
    axes[1].set_ylabel("Test metric")
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[1].yaxis.set_major_formatter(trimmed_decimal_formatter(2))
    axes[1].set_ylim(
        *padded_limits(
            values_b,
            minimum_span=0.12,
            include_value=minimum_recall,
            hard_bounds=(0.0, 1.0),
        )
    )
    set_client_axis(axes[1], clients, show_label=True)
    axes[1].legend(
        handles=metric_legend_handles(include_reference=minimum_recall),
        loc="best",
        frameon=False,
        fontsize=9.2,
    )

    add_strategy_legend(fig, axes)
    save_figure(
        fig,
        output_dir,
        "figure_2_strategy_comparison_fixed_validation_recall",
        formats,
    )
    plt.close(fig)

    columns = [
        "strategy",
        "threshold_regime",
        "scaling_point",
        "n_runs",
        "selected_threshold_mean",
        "test_recall_mean",
        "test_specificity_mean",
    ]
    return regime[columns].sort_values(["strategy", "scaling_point"])


# =============================================================================
# FIGURE 3: THRESHOLD-INDEPENDENT TEST PERFORMANCE
# =============================================================================
def selected_metric_frame(
    independent: pd.DataFrame, selection_metric: str
) -> pd.DataFrame:
    return independent[
        independent["selection_metric"] == selection_metric
    ].copy()


def plot_threshold_independent(
    independent: pd.DataFrame,
    clients: Sequence[int],
    output_dir: Path,
    formats: Sequence[str],
    width: float,
) -> pd.DataFrame:
    panel_specs: Tuple[Mapping[str, Any], ...] = (
        {
            "panel": "A",
            "selection": "ROC",
            "metric": "test_roc_auc_mean",
            "ylabel": "ROC-AUC",
            "decimals": 3,
            "minimum_span": 0.015,
            "bounds": (0.0, 1.0),
        },
        {
            "panel": "B",
            "selection": "PRROC",
            "metric": "test_pr_auc_mean",
            "ylabel": "Average precision",
            "decimals": 3,
            "minimum_span": 0.030,
            "bounds": (0.0, 1.0),
        },
        {
            "panel": "C",
            "selection": "Loss",
            "metric": "test_loss_mean",
            "ylabel": "Weighted loss",
            "decimals": 3,
            "minimum_span": 0.035,
            "bounds": (0.0, np.inf),
        },
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(width, 8.4), sharex=True, constrained_layout=False
    )
    fig.subplots_adjust(
        left=0.105, right=0.985, top=0.975, bottom=0.135, hspace=0.28
    )

    figure_rows: List[pd.DataFrame] = []
    for index, spec in enumerate(panel_specs):
        panel_frame = selected_metric_frame(independent, str(spec["selection"]))
        values = plot_strategy_metric(
            axes[index], panel_frame, clients, str(spec["metric"])
        )
        style_axis(axes[index])
        set_panel_label(axes[index], str(spec["panel"]))
        axes[index].set_ylabel(str(spec["ylabel"]))
        axes[index].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axes[index].yaxis.set_major_formatter(
            trimmed_decimal_formatter(int(spec["decimals"]))
        )
        bounds = spec["bounds"]
        hard_bounds = None
        if isinstance(bounds, tuple) and np.isfinite(bounds[1]):
            hard_bounds = bounds
        elif isinstance(bounds, tuple):
            hard_bounds = (float(bounds[0]), float(max(values) * 2.0 + 1.0))
        axes[index].set_ylim(
            *padded_limits(
                values,
                minimum_span=float(spec["minimum_span"]),
                hard_bounds=hard_bounds,
            )
        )
        set_client_axis(
            axes[index], clients, show_label=(index == len(panel_specs) - 1)
        )

        selected_columns = [
            "strategy",
            "scaling_point",
            "selection_metric",
            "n_runs",
            str(spec["metric"]),
        ]
        figure_rows.append(panel_frame[selected_columns].copy())

    add_strategy_legend(fig, axes)
    save_figure(
        fig,
        output_dir,
        "figure_3_strategy_comparison_threshold_independent_test_performance",
        formats,
    )
    plt.close(fig)

    return pd.concat(figure_rows, ignore_index=True).sort_values(
        ["selection_metric", "strategy", "scaling_point"]
    )


# =============================================================================
# CAPTIONS AND MAIN
# =============================================================================
def write_captions(
    output_dir: Path,
    clients: Sequence[int],
    minimum_recall: float,
) -> Path:
    path = output_dir / "suggested_strategy_comparison_captions.txt"
    first_client = int(min(clients))
    last_client = int(max(clients))
    n_points = len(clients)

    text = f"""Suggested APA-style figure captions
===================================

Figure X
Threshold-Dependent Test Performance of FedProx, SCAFFOLD, and FedAdam at the MCC-Optimal Operating Point

Note. For each strategy, run, and scaling point, the checkpoint with the highest validation average precision was selected before threshold optimization. The decision threshold was then selected by maximizing MCC on the validation set and transferred unchanged to the centralized test set. Panel A shows the mean validation-selected threshold, Panel B shows mean test MCC, and Panel C shows mean test recall and specificity at the corresponding operating points. Lines represent means across five repeated runs. Colors distinguish the optimization strategies; solid circles and dashed open squares distinguish recall and specificity, respectively. The comparison includes the {n_points} scaling points shared by all strategies ({first_client:,}-{last_client:,} clients). The client axis is logarithmic to base 2.

Figure X
Threshold-Dependent Test Performance of FedProx, SCAFFOLD, and FedAdam Under a Fixed Validation Recall Requirement

Note. For each strategy, run, and scaling point, the checkpoint with the highest validation average precision was selected before threshold optimization. Among the validation operating points satisfying recall >= {minimum_recall:.2f}, the threshold yielding the highest validation specificity was selected and transferred unchanged to the centralized test set. Panel A shows the mean validation-selected threshold, and Panel B shows mean test recall and specificity. The dotted horizontal line denotes the prespecified validation recall requirement. Lines represent means across five repeated runs. Colors distinguish the optimization strategies; solid circles and dashed open squares distinguish recall and specificity, respectively. The comparison includes the {n_points} scaling points shared by all strategies ({first_client:,}-{last_client:,} clients). The client axis is logarithmic to base 2.

Figure X
Threshold-Independent Test-Set Performance of FedProx, SCAFFOLD, and FedAdam Across Client Scaling Points

Note. Panel A shows mean test ROC-AUC for checkpoints selected by validation ROC-AUC, Panel B shows mean test average precision for checkpoints selected by validation average precision, and Panel C shows mean weighted test loss for checkpoints selected by validation loss. Lines represent means across five repeated runs; colors and markers distinguish the optimization strategies. The comparison includes the {n_points} scaling points shared by all strategies ({first_client:,}-{last_client:,} clients). The client axis is logarithmic to base 2. No model, round, threshold, or hyperparameter was selected on the test set.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum_recall <= 1.0:
        raise ValueError("--minimum-recall must be in the interval (0, 1].")

    font_name = available_font(args.font)
    configure_matplotlib(font_name)

    paths = resolve_all_inputs(args)

    print("=" * 94)
    print("APA-STYLE STRATEGY COMPARISON FIGURES")
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Font            : {font_name}")
    print(f"Formats         : {', '.join(args.formats)}")
    print(f"Minimum recall  : {args.minimum_recall:.2f}")
    print("Resolved inputs:")
    for strategy in STRATEGY_ORDER:
        print(f"  {strategy}")
        print(f"    threshold-dependent  : {paths[strategy]['threshold_dependent']}")
        print(f"    threshold-independent: {paths[strategy]['threshold_independent']}")
    print("=" * 94)

    threshold, independent = read_all_data(paths)
    clients = common_scaling_points(threshold, independent)
    threshold, independent = filter_common_points(
        threshold, independent, clients
    )

    print(f"Common scaling points ({len(clients)}): {clients}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_threshold_path = (
        args.output_dir / "combined_threshold_dependent_common_points.csv"
    )
    combined_independent_path = (
        args.output_dir / "combined_threshold_independent_common_points.csv"
    )
    threshold.to_csv(combined_threshold_path, index=False)
    independent.to_csv(combined_independent_path, index=False)

    figure_1_data = plot_mcc_optimal(
        threshold,
        clients,
        args.output_dir,
        args.formats,
        args.width,
    )
    figure_1_path = args.output_dir / "figure_1_mcc_optimal_values.csv"
    figure_1_data.to_csv(figure_1_path, index=False)

    figure_2_data = plot_fixed_recall(
        threshold,
        clients,
        args.minimum_recall,
        args.output_dir,
        args.formats,
        args.width,
    )
    figure_2_path = args.output_dir / "figure_2_fixed_recall_values.csv"
    figure_2_data.to_csv(figure_2_path, index=False)

    figure_3_data = plot_threshold_independent(
        independent,
        clients,
        args.output_dir,
        args.formats,
        args.width,
    )
    figure_3_path = args.output_dir / "figure_3_threshold_independent_values.csv"
    figure_3_data.to_csv(figure_3_path, index=False)

    caption_path = write_captions(
        args.output_dir, clients, args.minimum_recall
    )

    print()
    print("Finished. Main outputs:")
    print("  Figure 1: strategy comparison at MCC-optimal operating points")
    print("  Figure 2: strategy comparison under fixed validation recall")
    print("  Figure 3: threshold-independent strategy comparison")
    print(f"  Combined threshold-dependent data  : {combined_threshold_path}")
    print(f"  Combined threshold-independent data: {combined_independent_path}")
    print(f"  Figure 1 values: {figure_1_path}")
    print(f"  Figure 2 values: {figure_2_path}")
    print(f"  Figure 3 values: {figure_3_path}")
    print(f"  Captions       : {caption_path}")
    print("=" * 94)


if __name__ == "__main__":
    main()
