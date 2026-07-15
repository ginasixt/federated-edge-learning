#!/usr/bin/env python3
"""
Publication-ready plots for the final FL scaling-study test results.

The script reads the run-level output produced by ``final_test_set_eval.py``:

    result/splits_iid_scaling/final_test_set_eval/FedAdam/
        all_test_results.csv

For every scaling point and validation-selection criterion, the five test-set
runs are aggregated to exactly one plotted point per metric:

    * mean across runs -> marker position
    * min/max across runs -> asymmetric whiskers (default)
    * sample standard deviation -> available as an alternative

Generated figures
-----------------
1. test_roc_auc_by_scaling.pdf/png
2. test_average_precision_by_scaling.pdf/png
3. test_loss_by_scaling.pdf/png
4. test_metrics_overview.pdf/png
5. relative_run_variability.pdf/png

Generated tables
----------------
1. scaling_plot_statistics.csv
2. relative_run_variability.csv

The figures use a base-2 logarithmic client axis, vector PDF output, embedded
TrueType fonts, distinct markers and line styles, and dimensions suitable for
an A4 bachelor thesis. No test-set model selection is performed here; this
script only visualizes the already finalized test evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_INPUT = Path(
    "result/splits_iid_scaling/final_test_set_eval/FedAdam/"
    "all_test_results.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "figures"
EXPECTED_RUNS = 5

SELECTION_ORDER: Tuple[str, ...] = ("ROC", "PRROC", "Loss")
SELECTION_LABELS: Mapping[str, str] = {
    "ROC": "Selected by validation ROC-AUC",
    "PRROC": "Selected by validation AP",
    "Loss": "Selected by validation loss",
}

# Marker and line-style redundancy keeps the plots distinguishable in
# grayscale printouts without relying on color alone.
MARKERS: Mapping[str, str] = {
    "ROC": "o",
    "PRROC": "s",
    "Loss": "^",
}
LINESTYLES: Mapping[str, object] = {
    "ROC": "-",
    "PRROC": "--",
    "Loss": ":",
}

METRIC_SPECS: Mapping[str, Dict[str, str]] = {
    "test_roc_auc": {
        "label": "Test ROC-AUC",
        "short_name": "ROC-AUC",
        "filename": "test_roc_auc_by_scaling",
    },
    "test_pr_auc": {
        "label": "Test average precision",
        "short_name": "Average precision",
        "filename": "test_average_precision_by_scaling",
    },
    "test_loss": {
        "label": "Weighted test loss",
        "short_name": "Loss",
        "filename": "test_loss_by_scaling",
    },
}

OUTPUT_FORMATS: Tuple[str, ...] = ("pdf", "png")
PNG_DPI = 400

# Figure dimensions in inches. 6.8 in fits comfortably within an A4 text
# block; individual figures are compact enough for thesis placement.
INDIVIDUAL_FIGSIZE = (6.8, 3.7)
OVERVIEW_FIGSIZE = (6.8, 8.5)
VARIABILITY_FIGSIZE = (6.8, 7.5)
# =============================================================================


def configure_matplotlib() -> None:
    """Set a clean, paper-oriented Matplotlib configuration."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
            "figure.dpi": 120,
            "savefig.dpi": PNG_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready scaling-study plots from the final "
            "run-level test-set results."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Run-level CSV created by final_test_set_eval.py (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figures and plot tables (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--spread",
        choices=("minmax", "std", "none"),
        default="minmax",
        help=(
            "Uncertainty shown in the main plots: min/max whiskers, one sample "
            "standard deviation, or none (default: minmax)."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=list(OUTPUT_FORMATS),
        help="Output formats (default: pdf png).",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=EXPECTED_RUNS,
        help="Expected number of runs per scaling point and selection criterion.",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Do not generate the three individual metric figures.",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Do not generate the combined three-panel overview.",
    )
    parser.add_argument(
        "--no-variability",
        action="store_true",
        help="Do not generate the relative run-variability figure.",
    )
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Run final_test_set_eval.py first or provide --input."
        )

    df = pd.read_csv(path)
    required = {
        "strategy",
        "scaling_point",
        "selection_metric",
        "run",
        *METRIC_SPECS.keys(),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    numeric_columns = [
        "scaling_point",
        "run",
        *METRIC_SPECS.keys(),
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_rows = df[
        df[["scaling_point", "run", *METRIC_SPECS.keys()]].isna().any(axis=1)
    ]
    if not invalid_rows.empty:
        raise ValueError(
            f"Input CSV contains {len(invalid_rows)} rows with invalid numeric values."
        )

    df["scaling_point"] = df["scaling_point"].astype(int)
    df["run"] = df["run"].astype(int)
    df["selection_metric"] = df["selection_metric"].astype(str)

    unknown = sorted(set(df["selection_metric"]) - set(SELECTION_ORDER))
    if unknown:
        print(
            "⚠️ Ignoring unknown selection criteria: " + ", ".join(unknown)
        )
        df = df[df["selection_metric"].isin(SELECTION_ORDER)].copy()

    if df.empty:
        raise ValueError("No supported result rows remain after validation.")

    duplicate_keys = ["strategy", "scaling_point", "selection_metric", "run"]
    duplicates = df.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        duplicated = df.loc[duplicates, duplicate_keys].sort_values(duplicate_keys)
        raise ValueError(
            "Duplicate run-level result rows found:\n" + duplicated.to_string(index=False)
        )

    return df.sort_values(
        ["strategy", "scaling_point", "selection_metric", "run"]
    ).reset_index(drop=True)


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all statistics needed for plots and thesis reporting."""
    group_columns = ["strategy", "scaling_point", "selection_metric"]
    rows: List[Dict[str, float | int | str]] = []

    for keys, group in df.groupby(group_columns, sort=True):
        strategy, scaling_point, selection_metric = keys
        row: Dict[str, float | int | str] = {
            "strategy": str(strategy),
            "scaling_point": int(scaling_point),
            "selection_metric": str(selection_metric),
            "n_runs": int(group["run"].nunique()),
        }

        for metric in METRIC_SPECS:
            values = finite_array(group[metric])
            if values.size == 0:
                for suffix in (
                    "mean",
                    "std",
                    "min",
                    "max",
                    "mean_minus_min",
                    "max_minus_mean",
                    "max_abs_deviation",
                    "relative_max_deviation_percent",
                    "coefficient_of_variation_percent",
                ):
                    row[f"{metric}_{suffix}"] = np.nan
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            lower_distance = mean - minimum
            upper_distance = maximum - mean
            max_abs_deviation = max(lower_distance, upper_distance)
            denominator = abs(mean)
            relative_max_deviation = (
                100.0 * max_abs_deviation / denominator
                if denominator > np.finfo(float).eps
                else np.nan
            )
            coefficient_of_variation = (
                100.0 * std / denominator
                if denominator > np.finfo(float).eps
                else np.nan
            )

            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_min": minimum,
                    f"{metric}_max": maximum,
                    f"{metric}_mean_minus_min": lower_distance,
                    f"{metric}_max_minus_mean": upper_distance,
                    f"{metric}_max_abs_deviation": max_abs_deviation,
                    f"{metric}_relative_max_deviation_percent": relative_max_deviation,
                    f"{metric}_coefficient_of_variation_percent": coefficient_of_variation,
                }
            )

        rows.append(row)

    result = pd.DataFrame(rows)
    selection_rank = {name: index for index, name in enumerate(SELECTION_ORDER)}
    result["_selection_rank"] = result["selection_metric"].map(selection_rank)
    result = result.sort_values(
        ["strategy", "scaling_point", "_selection_rank"]
    ).drop(columns="_selection_rank")
    return result.reset_index(drop=True)


def validate_run_counts(stats: pd.DataFrame, expected_runs: int) -> None:
    unexpected = stats[stats["n_runs"] != expected_runs]
    if unexpected.empty:
        print(f"✅ Every plotted group contains {expected_runs} runs.")
        return

    print(
        f"⚠️ {len(unexpected)} group(s) do not contain the expected "
        f"{expected_runs} runs:"
    )
    print(
        unexpected[
            ["strategy", "scaling_point", "selection_metric", "n_runs"]
        ].to_string(index=False)
    )


def short_client_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def selection_present(stats: pd.DataFrame) -> List[str]:
    present = set(stats["selection_metric"].astype(str))
    return [metric for metric in SELECTION_ORDER if metric in present]


def metric_bounds(stats: pd.DataFrame, metric: str, spread: str) -> Tuple[float, float]:
    if spread == "minmax":
        lower = finite_array(stats[f"{metric}_min"])
        upper = finite_array(stats[f"{metric}_max"])
    elif spread == "std":
        means = stats[f"{metric}_mean"].to_numpy(dtype=float)
        stds = stats[f"{metric}_std"].to_numpy(dtype=float)
        lower = finite_array(means - stds)
        upper = finite_array(means + stds)
    else:
        means = finite_array(stats[f"{metric}_mean"])
        lower = means
        upper = means

    if lower.size == 0 or upper.size == 0:
        return 0.0, 1.0
    return float(np.min(lower)), float(np.max(upper))


def set_y_limits(ax: plt.Axes, stats: pd.DataFrame, metric: str, spread: str) -> None:
    minimum, maximum = metric_bounds(stats, metric, spread)
    data_range = maximum - minimum

    if metric in ("test_roc_auc", "test_pr_auc"):
        padding = max(0.004, data_range * 0.14)
        lower = max(0.0, minimum - padding)
        upper = min(1.0, maximum + padding)
        if math.isclose(lower, upper):
            lower = max(0.0, lower - 0.01)
            upper = min(1.0, upper + 0.01)
    else:
        padding = max(0.002, data_range * 0.14)
        lower = max(0.0, minimum - padding)
        upper = maximum + padding
        if math.isclose(lower, upper):
            upper = upper + max(0.01, abs(upper) * 0.05)

    ax.set_ylim(lower, upper)


def configure_client_axis(ax: plt.Axes, scaling_points: Sequence[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(scaling_points)
    ax.set_xticklabels(
        [short_client_label(value) for value in scaling_points],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(min(scaling_points) / 1.18, max(scaling_points) * 1.18)
    ax.set_xlabel("Number of clients")
    ax.grid(True, axis="both", which="major")
    ax.set_axisbelow(True)


def spread_yerr(group: pd.DataFrame, metric: str, spread: str) -> np.ndarray | None:
    if spread == "none":
        return None
    if spread == "std":
        std = group[f"{metric}_std"].to_numpy(dtype=float)
        return np.vstack([std, std])
    lower = group[f"{metric}_mean_minus_min"].to_numpy(dtype=float)
    upper = group[f"{metric}_max_minus_mean"].to_numpy(dtype=float)
    return np.vstack([lower, upper])


def plot_metric_on_axis(
    ax: plt.Axes,
    stats: pd.DataFrame,
    metric: str,
    spread: str,
    *,
    show_xlabel: bool = True,
    show_legend: bool = True,
) -> None:
    scaling_points = sorted(stats["scaling_point"].unique().astype(int))

    for selection_metric in selection_present(stats):
        group = stats[stats["selection_metric"] == selection_metric].sort_values(
            "scaling_point"
        )
        x = group["scaling_point"].to_numpy(dtype=int)
        y = group[f"{metric}_mean"].to_numpy(dtype=float)
        yerr = spread_yerr(group, metric, spread)

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            label=SELECTION_LABELS[selection_metric],
            marker=MARKERS[selection_metric],
            linestyle=LINESTYLES[selection_metric],
            capsize=2.2 if yerr is not None else 0.0,
            capthick=0.8,
            elinewidth=0.8,
            markeredgewidth=0.8,
            zorder=3,
        )

    configure_client_axis(ax, scaling_points)
    if not show_xlabel:
        ax.set_xlabel("")
    ax.set_ylabel(METRIC_SPECS[metric]["label"])
    set_y_limits(ax, stats, metric, spread)

    if show_legend:
        ax.legend(frameon=False, ncol=1, loc="best")


def spread_note(spread: str, expected_runs: int) -> str:
    if spread == "minmax":
        return (
            f"Markers show the mean across {expected_runs} runs; "
            "whiskers span the observed minimum to maximum."
        )
    if spread == "std":
        return (
            f"Markers show the mean across {expected_runs} runs; "
            "whiskers show ±1 sample standard deviation."
        )
    return f"Markers show the mean across {expected_runs} runs."


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"format": fmt}
        if fmt == "png":
            save_kwargs["dpi"] = PNG_DPI
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def create_individual_metric_figures(
    stats: pd.DataFrame,
    output_dir: Path,
    spread: str,
    formats: Sequence[str],
    expected_runs: int,
) -> List[Path]:
    outputs: List[Path] = []
    for metric, spec in METRIC_SPECS.items():
        fig, ax = plt.subplots(figsize=INDIVIDUAL_FIGSIZE, constrained_layout=True)
        plot_metric_on_axis(ax, stats, metric, spread)
        fig.text(
            0.01,
            -0.01,
            spread_note(spread, expected_runs),
            ha="left",
            va="top",
            fontsize=7.2,
        )
        outputs.extend(
            save_figure(fig, output_dir, spec["filename"], formats)
        )
    return outputs


def create_overview_figure(
    stats: pd.DataFrame,
    output_dir: Path,
    spread: str,
    formats: Sequence[str],
    expected_runs: int,
) -> List[Path]:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=OVERVIEW_FIGSIZE,
        sharex=True,
        constrained_layout=True,
    )

    handles = labels = None
    for index, (ax, metric) in enumerate(zip(axes, METRIC_SPECS.keys())):
        plot_metric_on_axis(
            ax,
            stats,
            metric,
            spread,
            show_xlabel=index == len(axes) - 1,
            show_legend=False,
        )
        ax.text(
            0.01,
            0.96,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(3, len(labels)),
            frameon=False,
        )

    fig.text(
        0.01,
        -0.005,
        spread_note(spread, expected_runs),
        ha="left",
        va="top",
        fontsize=7.2,
    )
    return save_figure(fig, output_dir, "test_metrics_overview", formats)


def create_variability_table(stats: pd.DataFrame) -> pd.DataFrame:
    columns = ["strategy", "scaling_point", "selection_metric", "n_runs"]
    output = stats[columns].copy()
    for metric in METRIC_SPECS:
        output[f"{metric}_relative_max_deviation_percent"] = stats[
            f"{metric}_relative_max_deviation_percent"
        ]
        output[f"{metric}_coefficient_of_variation_percent"] = stats[
            f"{metric}_coefficient_of_variation_percent"
        ]
    return output


def create_variability_figure(
    stats: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> List[Path]:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=VARIABILITY_FIGSIZE,
        sharex=True,
        constrained_layout=True,
    )
    scaling_points = sorted(stats["scaling_point"].unique().astype(int))
    handles = labels = None

    for index, (ax, metric) in enumerate(zip(axes, METRIC_SPECS.keys())):
        column = f"{metric}_relative_max_deviation_percent"
        for selection_metric in selection_present(stats):
            group = stats[
                stats["selection_metric"] == selection_metric
            ].sort_values("scaling_point")
            ax.plot(
                group["scaling_point"],
                group[column],
                marker=MARKERS[selection_metric],
                linestyle=LINESTYLES[selection_metric],
                label=SELECTION_LABELS[selection_metric],
            )

        configure_client_axis(ax, scaling_points)
        if index < len(axes) - 1:
            ax.set_xlabel("")
        ax.set_ylabel("Maximum deviation\nfrom mean (%)")
        ax.text(
            0.01,
            0.96,
            f"({chr(ord('a') + index)}) {METRIC_SPECS[metric]['short_name']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        lower, upper = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=max(upper, 0.1))
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(3, len(labels)),
            frameon=False,
        )

    fig.text(
        0.01,
        -0.005,
        "For each point: max(|minimum − mean|, |maximum − mean|) / |mean| × 100.",
        ha="left",
        va="top",
        fontsize=7.2,
    )
    return save_figure(fig, output_dir, "relative_run_variability", formats)


def write_manifest(
    path: Path,
    input_path: Path,
    stats_path: Path,
    variability_path: Path,
    figure_paths: Sequence[Path],
    spread: str,
    expected_runs: int,
) -> None:
    payload = {
        "input_csv": str(input_path),
        "aggregation": "arithmetic mean across run-level test results",
        "expected_runs_per_point": expected_runs,
        "main_plot_spread": spread,
        "statistics_csv": str(stats_path),
        "variability_csv": str(variability_path),
        "figures": [str(path) for path in figure_paths],
        "metric_definitions": {
            "test_roc_auc": "ROC area under the curve on the held-out test set",
            "test_pr_auc": (
                "Average precision on the held-out test set; this is the PR-AUC "
                "selection metric used by the evaluation pipeline"
            ),
            "test_loss": "Weighted cross-entropy loss on the held-out test set",
        },
        "relative_variability_definition": (
            "100 * max(abs(minimum - mean), abs(maximum - mean)) / abs(mean)"
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    results = read_results(args.input)
    stats = aggregate_results(results)
    validate_run_counts(stats, args.expected_runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "scaling_plot_statistics.csv"
    variability_path = args.output_dir / "relative_run_variability.csv"
    stats.to_csv(stats_path, index=False, float_format="%.10g")
    create_variability_table(stats).to_csv(
        variability_path,
        index=False,
        float_format="%.10g",
    )

    figure_paths: List[Path] = []
    if not args.no_individual:
        figure_paths.extend(
            create_individual_metric_figures(
                stats,
                args.output_dir,
                args.spread,
                args.formats,
                args.expected_runs,
            )
        )
    if not args.no_overview:
        figure_paths.extend(
            create_overview_figure(
                stats,
                args.output_dir,
                args.spread,
                args.formats,
                args.expected_runs,
            )
        )
    if not args.no_variability:
        figure_paths.extend(
            create_variability_figure(
                stats,
                args.output_dir,
                args.formats,
            )
        )

    manifest_path = args.output_dir / "figure_manifest.json"
    write_manifest(
        manifest_path,
        args.input,
        stats_path,
        variability_path,
        figure_paths,
        args.spread,
        args.expected_runs,
    )

    print("\n" + "=" * 78)
    print("✅ Scaling-study figures created")
    print(f"   Input             : {args.input}")
    print(f"   Output directory  : {args.output_dir}")
    print(f"   Plot statistics   : {stats_path}")
    print(f"   Variability table : {variability_path}")
    for path in figure_paths:
        print(f"   Figure            : {path}")
    print(f"   Manifest          : {manifest_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
