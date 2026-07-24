#!/usr/bin/env python3
"""
Create the complete run-to-run dispersion table for the threshold-independent
FedAdam scaling study.

Default usage
-------------
Run from the project root:

    python table_plot.py

Default input:
    result/splits_iid_scaling/final_test_set_eval/FedAdam/all_test_results.csv

Default output directory:
    result/splits_iid_scaling/final_test_set_eval/FedAdam/
    apa_scalability_figures/

The script reads the run-level results directly, calculates point-specific
statistics across the five runs at every scaling point, summarizes them across
the 15 scaling points, and creates:

    run_dispersion_table.csv
    run_dispersion_summary_full.csv
    run_dispersion_table.md
    run_dispersion_text_summary.txt
    table_run_to_run_dispersion.pdf
    table_run_to_run_dispersion.png
    figure_run_to_run_dispersion_by_scaling.pdf
    figure_run_to_run_dispersion_by_scaling.png
    run_dispersion_by_scaling.csv
    suggested_dispersion_figure_caption.txt

It also accepts an already aggregated statistics CSV as input.

Metric-specific checkpoint selection
------------------------------------
ROC-AUC       -> selection_metric == "ROC"
AP            -> selection_metric == "PRROC"
Weighted loss -> selection_metric == "Loss"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULTS
# =============================================================================
DEFAULT_INPUT = Path(
    "result/splits_iid_scaling/final_test_set_eval/FedAdam/"
    "all_test_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "apa_scalability_figures"
DEFAULT_STRATEGY = "FedAdam"
EXPECTED_SCALING_POINTS = 15
EXPECTED_RUNS = 5
DEFAULT_FORMATS: Tuple[str, ...] = ("pdf", "png")
PNG_DPI = 300

# Visual language shared with the other threshold-independent thesis figures.
FIGURE_WIDTH = 12.0
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
SHARED_LEGEND_GRAY = UHH_SLATE

PANEL_COLORS = {
    "ROC": UHH_RED,
    "PRROC": UHH_BLUE,
    "Loss": UHH_SLATE,
}

METRIC_SPECS: Tuple[Mapping[str, str], ...] = (
    {
        "label": "ROC-AUC",
        "selection_metric": "ROC",
        "metric": "test_roc_auc",
    },
    {
        "label": "AP",
        "selection_metric": "PRROC",
        "metric": "test_pr_auc",
    },
    {
        "label": "Weighted loss",
        "selection_metric": "Loss",
        "metric": "test_loss",
    },
)


# =============================================================================
# ARGUMENTS
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot the run-to-run dispersion table for "
            "threshold-independent FedAdam test performance."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Run-level or aggregated input CSV (default: {DEFAULT_INPUT})",
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
        help=f"Strategy to analyze (default: {DEFAULT_STRATEGY})",
    )
    parser.add_argument(
        "--expected-scaling-points",
        type=int,
        default=EXPECTED_SCALING_POINTS,
        help=(
            "Expected number of scaling points per metric "
            f"(default: {EXPECTED_SCALING_POINTS})"
        ),
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
        help=f"Expected runs per scaling point (default: {EXPECTED_RUNS})",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg", "tiff"),
        default=list(DEFAULT_FORMATS),
        help="Rendered table formats (default: pdf png)",
    )
    parser.add_argument(
        "--font",
        default="Arial",
        help="Preferred font (default: Arial)",
    )
    return parser.parse_args()


# =============================================================================
# INPUT AND AGGREGATION
# =============================================================================
def read_and_prepare(path: Path, strategy: str) -> Tuple[str, pd.DataFrame]:
    """Read run-level or aggregated input and return a common summary table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found:\n  {path}\n\n"
            "Run the script from the project root or provide another input:\n"
            "  python table_plot.py --input PATH/TO/FILE.csv"
        )

    data = pd.read_csv(path)

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
        *{
            f"{spec['metric']}_{suffix}"
            for spec in METRIC_SPECS
            for suffix in ("mean", "std", "min", "max")
        },
    }

    if raw_required.issubset(data.columns):
        return "run-level", aggregate_run_level(data, strategy)

    if summary_required.issubset(data.columns):
        return "aggregated", prepare_aggregated(data, strategy)

    missing_raw = sorted(raw_required - set(data.columns))
    missing_summary = sorted(summary_required - set(data.columns))
    raise ValueError(
        "Unsupported input format.\n\n"
        "Missing columns for run-level mode:\n  "
        + "\n  ".join(missing_raw)
        + "\n\nMissing columns for aggregated mode:\n  "
        + "\n  ".join(missing_summary)
    )


def aggregate_run_level(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Calculate point-specific statistics across the repeated runs."""
    frame = data.copy()

    numeric_columns = [
        "scaling_point",
        "run",
        "test_roc_auc",
        "test_pr_auc",
        "test_loss",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[numeric_columns].isna().any().any():
        raise ValueError("Run-level input contains invalid numeric values.")

    frame["strategy"] = frame["strategy"].astype(str)
    frame["selection_metric"] = frame["selection_metric"].astype(str)
    frame["scaling_point"] = frame["scaling_point"].astype(int)
    frame["run"] = frame["run"].astype(int)

    selections = {spec["selection_metric"] for spec in METRIC_SPECS}
    frame = frame[
        (frame["strategy"] == strategy)
        & frame["selection_metric"].isin(selections)
    ].copy()

    if frame.empty:
        raise ValueError(f"No supported rows found for strategy '{strategy}'.")

    key = ["strategy", "scaling_point", "selection_metric", "run"]
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate run rows found:\n"
            + frame.loc[duplicated, key].sort_values(key).to_string(index=False)
        )

    records: List[Dict[str, object]] = []
    grouped = frame.groupby(
        ["strategy", "scaling_point", "selection_metric"],
        sort=True,
    )

    for (group_strategy, clients, selection), group in grouped:
        record: Dict[str, object] = {
            "strategy": group_strategy,
            "scaling_point": int(clients),
            "selection_metric": selection,
            "n_runs": int(group["run"].nunique()),
        }

        for spec in METRIC_SPECS:
            metric = spec["metric"]
            values = group[metric].to_numpy(dtype=float)

            mean = float(np.mean(values))
            sample_sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            max_abs_deviation = float(np.max(np.abs(values - mean)))

            denominator = abs(mean)
            if denominator <= np.finfo(float).eps:
                cv_percent = float("nan")
                relative_max_deviation_percent = float("nan")
            else:
                cv_percent = 100.0 * sample_sd / denominator
                relative_max_deviation_percent = (
                    100.0 * max_abs_deviation / denominator
                )

            record.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": sample_sd,
                    f"{metric}_min": minimum,
                    f"{metric}_max": maximum,
                    f"{metric}_max_abs_deviation": max_abs_deviation,
                    f"{metric}_relative_max_deviation_percent": (
                        relative_max_deviation_percent
                    ),
                    f"{metric}_coefficient_of_variation_percent": cv_percent,
                }
            )

        records.append(record)

    return pd.DataFrame.from_records(records)


def prepare_aggregated(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Validate aggregated input and derive missing dispersion columns."""
    frame = data.copy()

    frame["strategy"] = frame["strategy"].astype(str)
    frame["selection_metric"] = frame["selection_metric"].astype(str)
    frame["scaling_point"] = pd.to_numeric(
        frame["scaling_point"], errors="coerce"
    )
    frame["n_runs"] = pd.to_numeric(frame["n_runs"], errors="coerce")

    if frame[["scaling_point", "n_runs"]].isna().any().any():
        raise ValueError("Aggregated input contains invalid scaling points or run counts.")

    frame["scaling_point"] = frame["scaling_point"].astype(int)
    frame["n_runs"] = frame["n_runs"].astype(int)

    selections = {spec["selection_metric"] for spec in METRIC_SPECS}
    frame = frame[
        (frame["strategy"] == strategy)
        & frame["selection_metric"].isin(selections)
    ].copy()

    if frame.empty:
        raise ValueError(f"No supported rows found for strategy '{strategy}'.")

    for spec in METRIC_SPECS:
        metric = spec["metric"]
        required = [
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_min",
            f"{metric}_max",
        ]

        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if frame[required].isna().any().any():
            raise ValueError(f"Invalid aggregated values for metric '{metric}'.")

        mean = frame[f"{metric}_mean"].astype(float)
        sample_sd = frame[f"{metric}_std"].astype(float)
        minimum = frame[f"{metric}_min"].astype(float)
        maximum = frame[f"{metric}_max"].astype(float)
        denominator = mean.abs().replace(0.0, np.nan)

        max_abs_column = f"{metric}_max_abs_deviation"
        relative_column = f"{metric}_relative_max_deviation_percent"
        cv_column = f"{metric}_coefficient_of_variation_percent"

        if max_abs_column not in frame.columns:
            frame[max_abs_column] = np.maximum(
                (mean - minimum).abs(),
                (maximum - mean).abs(),
            )
        else:
            frame[max_abs_column] = pd.to_numeric(
                frame[max_abs_column], errors="coerce"
            )

        if relative_column not in frame.columns:
            frame[relative_column] = (
                100.0 * frame[max_abs_column] / denominator
            )
        else:
            frame[relative_column] = pd.to_numeric(
                frame[relative_column], errors="coerce"
            )

        if cv_column not in frame.columns:
            frame[cv_column] = 100.0 * sample_sd / denominator
        else:
            frame[cv_column] = pd.to_numeric(
                frame[cv_column], errors="coerce"
            )

    return frame


# =============================================================================
# SUMMARY CALCULATION
# =============================================================================
def value_and_client(
    values: pd.Series,
    clients: pd.Series,
    mode: str,
) -> Tuple[float, int]:
    if mode == "min":
        index = values.idxmin()
    elif mode == "max":
        index = values.idxmax()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return float(values.loc[index]), int(clients.loc[index])


def calculate_metric_summary(
    summary: pd.DataFrame,
    spec: Mapping[str, str],
    expected_scaling_points: int,
    expected_runs: int,
) -> Dict[str, object]:
    """Summarize point-specific dispersion across all scaling points."""
    selection = spec["selection_metric"]
    metric = spec["metric"]

    subset = summary[
        summary["selection_metric"] == selection
    ].sort_values("scaling_point").reset_index(drop=True)

    if len(subset) != expected_scaling_points:
        raise ValueError(
            f"{spec['label']}: expected {expected_scaling_points} scaling "
            f"points, found {len(subset)}."
        )

    if subset["scaling_point"].duplicated().any():
        raise ValueError(f"{spec['label']}: duplicate scaling points found.")

    incorrect = subset[subset["n_runs"] != expected_runs]
    if not incorrect.empty:
        details = ", ".join(
            f"{int(row.scaling_point)} clients: n={int(row.n_runs)}"
            for row in incorrect.itertuples()
        )
        raise ValueError(
            f"{spec['label']}: expected {expected_runs} runs at every point; "
            f"found {details}."
        )

    clients = subset["scaling_point"].astype(int)
    metric_means = subset[f"{metric}_mean"].astype(float)
    standard_deviations = subset[f"{metric}_std"].astype(float)
    cvs = subset[f"{metric}_coefficient_of_variation_percent"].astype(float)
    relative_deviations = subset[
        f"{metric}_relative_max_deviation_percent"
    ].astype(float)

    minimum_sd, client_at_minimum_sd = value_and_client(
        standard_deviations, clients, "min"
    )
    maximum_sd, client_at_maximum_sd = value_and_client(
        standard_deviations, clients, "max"
    )
    minimum_cv, client_at_minimum_cv = value_and_client(cvs, clients, "min")
    maximum_cv, client_at_maximum_cv = value_and_client(cvs, clients, "max")
    minimum_relative, client_at_minimum_relative = value_and_client(
        relative_deviations, clients, "min"
    )
    maximum_relative, client_at_maximum_relative = value_and_client(
        relative_deviations, clients, "max"
    )

    return {
        "metric": spec["label"],
        "selection_metric": selection,
        "n_scaling_points": int(len(subset)),
        "runs_per_scaling_point": int(expected_runs),

        # Performance context for the detailed verification output.
        "mean_metric_value_across_scaling_points": float(metric_means.mean()),
        "minimum_point_specific_mean": float(metric_means.min()),
        "maximum_point_specific_mean": float(metric_means.max()),

        # Absolute run-to-run dispersion.
        "mean_sd_across_scaling_points": float(standard_deviations.mean()),
        "minimum_sd": minimum_sd,
        "client_at_minimum_sd": client_at_minimum_sd,
        "maximum_sd": maximum_sd,
        "client_at_maximum_sd": client_at_maximum_sd,

        # Relative run-to-run dispersion.
        "mean_cv_percent": float(cvs.mean()),
        "minimum_cv_percent": minimum_cv,
        "client_at_minimum_cv": client_at_minimum_cv,
        "maximum_cv_percent": maximum_cv,
        "client_at_maximum_cv": client_at_maximum_cv,

        # Complementary worst-case single-run measure.
        "mean_pointwise_max_relative_deviation_percent": float(
            relative_deviations.mean()
        ),
        "minimum_relative_single_run_deviation_percent": minimum_relative,
        "client_at_minimum_relative_single_run_deviation": (
            client_at_minimum_relative
        ),
        "maximum_relative_single_run_deviation_percent": maximum_relative,
        "client_at_maximum_relative_single_run_deviation": (
            client_at_maximum_relative
        ),
    }


# =============================================================================
# OUTPUT TABLES
# =============================================================================
def format_client(value: int) -> str:
    return f"{value:,}"


def format_sd(value: float) -> str:
    return f"{value:.4f}"


def format_percent(value: float) -> str:
    return f"{value:.2f}"


def compact_row(summary: Mapping[str, object]) -> Dict[str, object]:
    """Main-text table: absolute and relative dispersion, min/max CV locations."""
    return {
        "Metric": summary["metric"],
        "Mean SD across scaling points": format_sd(
            float(summary["mean_sd_across_scaling_points"])
        ),
        "SD range": (
            f"{format_sd(float(summary['minimum_sd']))}"
            f"–{format_sd(float(summary['maximum_sd']))}"
        ),
        "Mean CV (%)": format_percent(float(summary["mean_cv_percent"])),
        "Min. CV (%)": format_percent(float(summary["minimum_cv_percent"])),
        "Client at min. CV": format_client(
            int(summary["client_at_minimum_cv"])
        ),
        "Max. CV (%)": format_percent(float(summary["maximum_cv_percent"])),
        "Client at max. CV": format_client(
            int(summary["client_at_maximum_cv"])
        ),
        "Max. relative single-run deviation (%)": format_percent(
            float(summary["maximum_relative_single_run_deviation_percent"])
        ),
        "Client at max. relative deviation": format_client(
            int(summary["client_at_maximum_relative_single_run_deviation"])
        ),
    }


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            "---" if index == 0 else "---:"
            for index in range(len(headers))
        )
        + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return "\n".join(lines)


def make_text_summary(summaries: Sequence[Mapping[str, object]]) -> str:
    by_metric = {str(item["metric"]): item for item in summaries}

    lines = [
        "Run-to-run dispersion summary",
        "==============================",
        "",
        "Maximum relative single-run deviations:",
    ]

    for metric in ("ROC-AUC", "AP", "Weighted loss"):
        item = by_metric[metric]
        lines.append(
            f"- {metric}: "
            f"{float(item['maximum_relative_single_run_deviation_percent']):.2f}% "
            f"at {int(item['client_at_maximum_relative_single_run_deviation']):,} "
            "clients"
        )

    lines.extend(
        [
            "",
            "CV minima and maxima:",
        ]
    )

    for metric in ("ROC-AUC", "AP", "Weighted loss"):
        item = by_metric[metric]
        lines.append(
            f"- {metric}: minimum CV "
            f"{float(item['minimum_cv_percent']):.2f}% at "
            f"{int(item['client_at_minimum_cv']):,} clients; maximum CV "
            f"{float(item['maximum_cv_percent']):.2f}% at "
            f"{int(item['client_at_maximum_cv']):,} clients"
        )

    return "\n".join(lines) + "\n"


# =============================================================================
# APA-STYLE RENDERED TABLE
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
    """Apply the same publication styling as the other thesis figures."""
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
        }
    )


def plot_table(
    compact_frame: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """Render a compact APA-like table with horizontal rules only."""
    display_columns = [
        "Metric",
        "Mean SD\nacross points",
        "SD range",
        "Mean CV\n(%)",
        "Min. CV\n(%)",
        "Client at\nmin. CV",
        "Max. CV\n(%)",
        "Client at\nmax. CV",
        "Max. relative\nsingle-run dev. (%)",
        "Client at max.\nrelative deviation",
    ]

    table_values = compact_frame.copy()
    table_values.columns = display_columns

    fig, ax = plt.subplots(figsize=(15.0, 2.85))
    ax.axis("off")

    table = ax.table(
        cellText=table_values.values,
        colLabels=table_values.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.12, 1.0, 0.78],
        colWidths=[
            0.115,  # Metric
            0.120,  # Mean SD
            0.105,  # SD range
            0.075,  # Mean CV
            0.075,  # Min CV
            0.095,  # Client min CV
            0.075,  # Max CV
            0.095,  # Client max CV
            0.135,  # Max relative deviation
            0.110,  # Client max relative deviation
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.45)

    n_rows = len(table_values)
    n_cols = len(table_values.columns)

    # Remove all default borders.
    for cell in table.get_celld().values():
        cell.set_linewidth(0.0)
        cell.set_edgecolor("white")
        cell.set_facecolor("white")
        cell.PAD = 0.03

    # Header formatting.
    for column in range(n_cols):
        header = table[(0, column)]
        header.set_text_props(weight="bold")
        header.visible_edges = "TB"
        header.set_edgecolor("#202124")
        header.set_linewidth(0.8)

    # Bottom rule under final row.
    for column in range(n_cols):
        cell = table[(n_rows, column)]
        cell.visible_edges = "B"
        cell.set_edgecolor("#202124")
        cell.set_linewidth(0.8)

    # Left-align metric labels.
    for row in range(1, n_rows + 1):
        table[(row, 0)].set_text_props(ha="left")

    ax.text(
        0.0,
        1.03,
        "Run-to-Run Dispersion in Threshold-Independent FedAdam Test Performance",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontstyle="italic",
    )

    note = (
        "Note. Statistics are based on five repeated runs at each of 15 scaling "
        "points. Mean SD is the arithmetic mean of the point-specific sample "
        "standard deviations. CV denotes the point-specific coefficient of "
        "variation."
    )
    ax.text(
        0.0,
        0.01,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        wrap=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"table_run_to_run_dispersion.{fmt}"
        kwargs: Dict[str, object] = {}
        if fmt == "png":
            kwargs["dpi"] = PNG_DPI
        elif fmt == "tiff":
            kwargs["dpi"] = PNG_DPI
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

        fig.savefig(path, format=fmt, **kwargs)
        print(f"  ✓ {path}")

    plt.close(fig)



def make_dispersion_plot_data(point_summary: pd.DataFrame) -> pd.DataFrame:
    """Create one tidy row per metric and scaling point for the appendix plot."""
    records: List[Dict[str, object]] = []

    for spec in METRIC_SPECS:
        metric = spec["metric"]
        selection = spec["selection_metric"]

        subset = point_summary[
            point_summary["selection_metric"] == selection
        ].sort_values("scaling_point")

        for row in subset.itertuples(index=False):
            records.append(
                {
                    "metric": spec["label"],
                    "selection_metric": selection,
                    "scaling_point": int(row.scaling_point),
                    "cv_percent": float(
                        getattr(
                            row,
                            f"{metric}_coefficient_of_variation_percent",
                        )
                    ),
                    "maximum_relative_single_run_deviation_percent": float(
                        getattr(
                            row,
                            f"{metric}_relative_max_deviation_percent",
                        )
                    ),
                }
            )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No point-specific dispersion data available for plotting.")

    return frame



def client_label(value: int) -> str:
    """Use the same compact client labels as the other scalability figures."""
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return f"{value:,}"


def client_positions(clients: Sequence[int]) -> np.ndarray:
    return np.log2(np.asarray(clients, dtype=float))


def style_axis(ax: plt.Axes) -> None:
    """Open APA-style axes with restrained gridlines."""
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


def percent_formatter(value: float, _position: object) -> str:
    """Show up to two decimals without unnecessary trailing zeros."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def plot_dispersion_by_scaling(
    plot_data: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    """
    Plot relative run-to-run dispersion across all client scaling points.

    The visual style matches the threshold-independent performance plots:
    metric-specific UHH colors, panel letters only, open axes, restrained
    gridlines, compact log2 client labels, and one shared neutral legend.
    """
    panel_specs = (
        {
            "metric": "ROC-AUC",
            "selection": "ROC",
            "panel": "A",
        },
        {
            "metric": "AP",
            "selection": "PRROC",
            "panel": "B",
        },
        {
            "metric": "Weighted loss",
            "selection": "Loss",
            "panel": "C",
        },
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(FIGURE_WIDTH, 8.4),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.975,
        bottom=0.135,
        hspace=0.28,
    )

    all_clients = sorted(
        int(value) for value in plot_data["scaling_point"].unique()
    )
    x_ticks = client_positions(all_clients)

    for index, (ax, spec) in enumerate(zip(axes, panel_specs)):
        metric_label = str(spec["metric"])
        selection = str(spec["selection"])
        panel = str(spec["panel"])
        color = PANEL_COLORS[selection]

        subset = plot_data[
            plot_data["metric"] == metric_label
        ].sort_values("scaling_point")

        clients = subset["scaling_point"].to_numpy(dtype=int)
        x = client_positions(clients)
        cv = subset["cv_percent"].to_numpy(dtype=float)
        maximum_deviation = subset[
            "maximum_relative_single_run_deviation_percent"
        ].to_numpy(dtype=float)

        # Same panel color for both measures; marker and line style encode
        # the statistical quantity, while panel color identifies the metric.
        ax.plot(
            x,
            cv,
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
            maximum_deviation,
            color=color,
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker="s",
            markersize=MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.90,
            alpha=0.97,
            zorder=3,
        )

        style_axis(ax)
        set_panel_label(ax, panel)
        ax.set_ylabel("Relative dispersion (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        upper = max(
            float(np.nanmax(cv)),
            float(np.nanmax(maximum_deviation)),
        )
        padding = max(0.04, upper * 0.12)
        ax.set_ylim(0.0, upper + padding)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([client_label(value) for value in all_clients])
        ax.set_xlim(float(x_ticks.min() - 0.35), float(x_ticks.max() + 0.35))
        ax.set_xlabel(
            "Number of clients (log2 scale)"
            if index == len(panel_specs) - 1
            else ""
        )

    shared_handles = [
        Line2D(
            [0],
            [0],
            color=SHARED_LEGEND_GRAY,
            linewidth=LINE_WIDTH,
            linestyle="-",
            marker="o",
            markersize=5.0,
            markerfacecolor=SHARED_LEGEND_GRAY,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            label="Coefficient of variation",
        ),
        Line2D(
            [0],
            [0],
            color=SHARED_LEGEND_GRAY,
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker="s",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=SHARED_LEGEND_GRAY,
            markeredgewidth=0.90,
            label="Maximum relative single-run deviation",
        ),
    ]

    bottom_position = axes[-1].get_position()
    legend_x = bottom_position.x0 + bottom_position.width / 2
    fig.legend(
        handles=shared_handles,
        loc="lower center",
        bbox_to_anchor=(legend_x, 0.018),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=9.2,
        borderaxespad=0.0,
        columnspacing=1.5,
        handlelength=2.4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_dir / (
            f"figure_run_to_run_dispersion_by_scaling.{fmt}"
        )
        save_kwargs: Dict[str, object] = {}
        if fmt == "png":
            save_kwargs["dpi"] = PNG_DPI
        elif fmt == "tiff":
            save_kwargs["dpi"] = PNG_DPI
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}

        fig.savefig(output_path, format=fmt, **save_kwargs)
        print(f"  ✓ {output_path}")

    plt.close(fig)




def write_dispersion_figure_caption(output_dir: Path) -> Path:
    """Write the title and note outside the plot, as required for APA figures."""
    caption_path = output_dir / "suggested_dispersion_figure_caption.txt"
    caption = """Figure A.X
Run-to-Run Dispersion in Threshold-Independent FedAdam Test Performance Across Client Scaling Points

Note. Panel A shows ROC-AUC, Panel B shows average precision (AP), and Panel C shows weighted binary cross-entropy loss calculated with a positive-class weight of 1.5. Solid lines with filled circles represent the point-specific coefficient of variation across five repeated training runs. Dashed lines with open squares represent the maximum absolute deviation of an individual run from the corresponding point-specific mean, expressed as a percentage of that mean. Checkpoints were selected on the validation set by maximizing ROC-AUC, maximizing AP, and minimizing weighted validation loss, respectively. The client axis is logarithmic to base 2.
"""
    caption_path.write_text(caption, encoding="utf-8")
    return caption_path


def write_outputs(
    summaries: Sequence[Mapping[str, object]],
    point_summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    font_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_frame = pd.DataFrame(summaries)
    compact_frame = pd.DataFrame(
        [compact_row(summary) for summary in summaries]
    )

    full_path = output_dir / "run_dispersion_summary_full.csv"
    compact_path = output_dir / "run_dispersion_table.csv"
    markdown_path = output_dir / "run_dispersion_table.md"
    text_path = output_dir / "run_dispersion_text_summary.txt"
    plot_data_path = output_dir / "run_dispersion_by_scaling.csv"
    caption_path = write_dispersion_figure_caption(output_dir)

    plot_data = make_dispersion_plot_data(point_summary)

    full_frame.to_csv(full_path, index=False)
    compact_frame.to_csv(compact_path, index=False)
    plot_data.to_csv(plot_data_path, index=False)
    text_path.write_text(make_text_summary(summaries), encoding="utf-8")

    note = (
        "Note. Statistics are based on five repeated runs at each of 15 scaling "
        "points. Mean SD is the arithmetic mean of the 15 sample standard "
        "deviations calculated separately across the five runs at each scaling "
        "point; it is not a pooled standard deviation. The coefficient of "
        "variation (CV) is the point-specific sample standard deviation divided "
        "by the corresponding point-specific mean and expressed as a percentage. "
        "ROC-AUC, AP, and weighted loss are based on checkpoints selected using "
        "their corresponding validation metrics."
    )

    markdown = (
        "**Table X**\n\n"
        "*Run-to-Run Dispersion in Threshold-Independent FedAdam "
        "Test Performance*\n\n"
        f"{dataframe_to_markdown(compact_frame)}\n\n"
        f"*{note}*\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    configure_matplotlib(font_name)
    plot_table(compact_frame, output_dir, formats)
    plot_dispersion_by_scaling(plot_data, output_dir, formats)

    print()
    print(markdown)
    print("Generated data files:")
    print(f"  ✓ {compact_path}")
    print(f"  ✓ {full_path}")
    print(f"  ✓ {markdown_path}")
    print(f"  ✓ {text_path}")
    print(f"  ✓ {plot_data_path}")
    print(f"  ✓ {caption_path}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    font_name = available_font(args.font)

    print("=" * 78)
    print("FEDADAM RUN-TO-RUN DISPERSION TABLE")
    print(f"Input      : {args.input}")
    print(f"Output     : {args.output_dir}")
    print(f"Strategy   : {args.strategy}")
    print(f"Font       : {font_name}")
    print(
        f"Expected   : {args.expected_scaling_points} scaling points, "
        f"{args.expected_runs} runs per point"
    )
    print("=" * 78)

    mode, point_summary = read_and_prepare(args.input, args.strategy)
    print(f"Input mode : {mode}")

    summaries = [
        calculate_metric_summary(
            point_summary,
            spec,
            expected_scaling_points=args.expected_scaling_points,
            expected_runs=args.expected_runs,
        )
        for spec in METRIC_SPECS
    ]

    write_outputs(
        summaries=summaries,
        point_summary=point_summary,
        output_dir=args.output_dir,
        formats=args.formats,
        font_name=font_name,
    )


if __name__ == "__main__":
    main()
