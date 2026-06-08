import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/home/bax9142/federated-edge-learning/result/splits_iid_scaling/splits_iid_16384_clients.json"
)
DEFAULT_OUT = Path(
    "/home/bax9142/federated-edge-learning/result/plots/splits_iid_scaling_models"
)

ROUND_FILE_RE = re.compile(r"round_(\d+)_run_1\.json$")
STRATEGY_DIR_RE = re.compile(r"^all_rounds_(?P<family>[A-Za-z]+)_(?P<variant>\d+)$")

FAMILY_COLORS = {
    "FedAdam": "#1D4ED8",
    "FedProx": "#DC2626",
    "Scaffold": "#16A34A",
}

VARIANT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


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


def summarize_threshold_rows(threshold_rows: Iterable[dict]) -> Dict[str, float]:
    df = pd.DataFrame(list(threshold_rows))
    if df.empty:
        return {}

    summary: Dict[str, float] = {}
    metrics_to_maximize = {
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "youden": "youden",
        "recall": "recall",
        "precision": "precision",
        "ppv": "ppv",
    }

    for prefix, metric in metrics_to_maximize.items():
        idx = safe_idxmax(df[metric])
        if idx is None:
            continue
        row = df.loc[idx]
        summary[f"best_{prefix}_value"] = float(row.get(metric, np.nan))
        summary[f"best_{prefix}_threshold"] = float(row.get("threshold", np.nan))
        summary[f"best_{prefix}_recall"] = float(row.get("recall", np.nan))
        summary[f"best_{prefix}_spec"] = float(row.get("spec", np.nan))
        summary[f"best_{prefix}_precision"] = float(row.get("precision", np.nan))
        summary[f"best_{prefix}_fpr"] = float(row.get("fpr", np.nan))
        summary[f"best_{prefix}_alerts_per_1000"] = float(row.get("alerts_per_1000", np.nan))

    return summary


def summarize_round_thresholds(threshold_rows: Iterable[dict]) -> Tuple[Dict[str, float], pd.DataFrame]:
    df = pd.DataFrame(list(threshold_rows))
    if df.empty:
        return {}, df

    summary: Dict[str, float] = {}
    ba_idx = safe_idxmax(df["balanced_accuracy"])
    if ba_idx is not None:
        row = df.loc[ba_idx]
        summary["best_balanced_accuracy_value"] = float(row.get("balanced_accuracy", np.nan))
        summary["best_balanced_accuracy_threshold"] = float(row.get("threshold", np.nan))
        summary["best_balanced_accuracy_recall"] = float(row.get("recall", np.nan))
        summary["best_balanced_accuracy_spec"] = float(row.get("spec", np.nan))
        summary["best_balanced_accuracy_precision"] = float(row.get("precision", np.nan))
        summary["best_balanced_accuracy_fpr"] = float(row.get("fpr", np.nan))
        summary["best_balanced_accuracy_alerts_per_1000"] = float(row.get("alerts_per_1000", np.nan))

    return summary, df


def collect_round_summary(root: Path) -> pd.DataFrame:
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
            threshold_rows = metrics.get("all_thresholds", [])
            threshold_summary = summarize_threshold_rows(threshold_rows)

            row = {
                "family": family,
                "variant": variant,
                "strategy_id": strategy_id,
                "round": int(match.group(1)),
                "auc": metrics.get("auc", np.nan),
                "model_checkpoint": data.get("model_checkpoint", ""),
                "source_file": str(round_file),
            }
            row.update(threshold_summary)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [c for c in df.columns if c not in {"family", "strategy_id", "model_checkpoint", "source_file"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df.sort_values(["family", "variant", "round"]).reset_index(drop=True)


def collect_strategy_bundles(root: Path) -> Dict[str, dict]:
    bundles: Dict[str, dict] = {}

    for strategy_dir in sorted([d for d in root.glob("all_rounds_*") if d.is_dir()]):
        family, variant = parse_strategy_dir(strategy_dir)
        strategy_id = f"{family}_{variant}" if variant else family
        rows: List[dict] = []
        threshold_by_round: Dict[int, pd.DataFrame] = {}

        for round_file in sorted(strategy_dir.glob("round_*_run_1.json")):
            match = ROUND_FILE_RE.search(round_file.name)
            if not match:
                continue

            data = load_json(round_file)
            if not data:
                continue

            round_num = int(match.group(1))
            metrics = data.get("metrics", {})
            round_summary, threshold_df = summarize_round_thresholds(metrics.get("all_thresholds", []))
            threshold_by_round[round_num] = threshold_df

            row = {
                "family": family,
                "variant": variant,
                "strategy_id": strategy_id,
                "round": round_num,
                "auc": metrics.get("auc", np.nan),
                "model_checkpoint": data.get("model_checkpoint", ""),
                "source_file": str(round_file),
            }
            row.update(summarize_threshold_rows(metrics.get("all_thresholds", [])))
            row.update(round_summary)
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(["family", "variant", "round"]).reset_index(drop=True)
        bundles[strategy_id] = {
            "family": family,
            "variant": variant,
            "strategy_dir": strategy_dir,
            "summary": summary_df,
            "threshold_by_round": threshold_by_round,
        }

    return bundles


def style_for_strategy(strategy_id: str) -> Tuple[str, str]:
    family = strategy_id.split("_")[0]
    variant_part = strategy_id.split("_")[-1]
    marker_index = 0
    try:
        marker_index = max(0, int(variant_part) - 1)
    except ValueError:
        marker_index = 0
    color = FAMILY_COLORS.get(family, "#6B7280")
    marker = VARIANT_MARKERS[marker_index % len(VARIANT_MARKERS)]
    return color, marker


def hex_to_rgb(hexcol: str) -> Tuple[float, float, float]:
    hexcol = hexcol.lstrip("#")
    lv = len(hexcol)
    return tuple(int(hexcol[i : i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(clip01(c) * 255) for c in rgb)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def variant_shaded_color(base_hex: str, idx: int, total: int) -> str:
    base = hex_to_rgb(base_hex)
    if total <= 1:
        return base_hex
    # factor in [0.25, 0.75]
    factor = 0.25 + 0.5 * (idx / max(1, total - 1))
    # interpolate towards white to get lighter shades for larger idx
    shaded = tuple(clip01((1 - factor) * c + factor * 1.0) for c in base)
    return rgb_to_hex(shaded)


def plot_learning_curves(df: pd.DataFrame, outdir: Path) -> Path:
    ensure_outdir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    metrics = [("auc", "AUROC"), ("best_f1_value", "Bestes F1 pro Runde")]

    for ax, (metric, title) in zip(axes, metrics):
        for strategy_id, group in df.groupby("strategy_id"):
            color, marker = style_for_strategy(strategy_id)
            ax.plot(
                group["round"],
                group[metric],
                label=strategy_id,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=2,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Lernkurven für die Modelle im IID-Scaling-Experiment", fontsize=13, fontweight="bold")
    fig.tight_layout()
    outpath = outdir / "01_learning_curves.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_threshold_quality_curves(df: pd.DataFrame, outdir: Path) -> Path:
    ensure_outdir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    metrics = [("best_balanced_accuracy_value", "Best Balanced Accuracy"), ("best_youden_value", "Best Youden-Index")]

    for ax, (metric, title) in zip(axes, metrics):
        for strategy_id, group in df.groupby("strategy_id"):
            color, marker = style_for_strategy(strategy_id)
            ax.plot(
                group["round"],
                group[metric],
                label=strategy_id,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=2,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Threshold-basierte Qualitätsmetriken über die Runden", fontsize=13, fontweight="bold")
    fig.tight_layout()
    outpath = outdir / "02_threshold_quality_curves.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_best_round_bar_chart(df: pd.DataFrame, outdir: Path) -> Path:
    ensure_outdir(outdir)
    summary_rows = []
    for strategy_id, group in df.groupby("strategy_id"):
        best_idx = safe_idxmax(group["auc"])
        if best_idx is None:
            continue
        best_row = group.loc[best_idx]
        summary_rows.append(
            {
                "strategy_id": strategy_id,
                "round": int(best_row["round"]),
                "auc": float(best_row.get("auc", np.nan)),
                "best_f1_value": float(best_row.get("best_f1_value", np.nan)),
                "best_balanced_accuracy_value": float(best_row.get("best_balanced_accuracy_value", np.nan)),
                "best_youden_value": float(best_row.get("best_youden_value", np.nan)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return outdir / "03_best_round_bar_chart.png"

    metrics = ["auc", "best_f1_value", "best_balanced_accuracy_value", "best_youden_value"]
    labels = ["AUROC", "F1", "Balanced Acc.", "Youden"]
    x = np.arange(len(summary))
    width = 0.18

    fig, ax = plt.subplots(figsize=(16, 6))
    for offset, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(
            x + (offset - 1.5) * width,
            summary[metric],
            width=width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{sid}\nR{int(r)}" for sid, r in zip(summary["strategy_id"], summary["round"])], rotation=0)
    ax.set_ylabel("Wert")
    ax.set_title("Bestes Round pro Modell, ausgewertet mit mehreren Metriken")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    outpath = outdir / "03_best_round_bar_chart.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_tradeoff_best_round(df: pd.DataFrame, outdir: Path) -> Path:
    ensure_outdir(outdir)
    strategies = list(df["strategy_id"].dropna().unique())
    if not strategies:
        return outdir / "04_tradeoff_best_round.png"

    cols = min(3, len(strategies))
    rows = math.ceil(len(strategies) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[len(strategies):]:
        ax.axis("off")

    for ax, strategy_id in zip(axes_flat, strategies):
        group = df[df["strategy_id"] == strategy_id]
        best_idx = safe_idxmax(group["auc"])
        if best_idx is None:
            ax.axis("off")
            continue

        best_row = group.loc[best_idx]
        round_file = Path(best_row["source_file"])
        data = load_json(round_file)
        if not data:
            ax.axis("off")
            continue

        threshold_rows = data.get("metrics", {}).get("all_thresholds", [])
        threshold_df = pd.DataFrame(threshold_rows)
        if threshold_df.empty:
            ax.axis("off")
            continue

        color, _ = style_for_strategy(strategy_id)
        ax.scatter(
            threshold_df["spec"],
            threshold_df["recall"],
            c=threshold_df["threshold"],
            cmap="viridis",
            s=45,
            edgecolors="none",
        )
        ax.plot(threshold_df["spec"], threshold_df["recall"], color=color, linewidth=1.5, alpha=0.65)
        ax.scatter([best_row.get("best_f1_spec", np.nan)], [best_row.get("best_f1_recall", np.nan)], color=color, s=90, marker="*", edgecolors="black", linewidths=0.8)
        ax.set_title(f"{strategy_id}  |  beste Round {int(best_row['round'])}")
        ax.set_xlabel("Specificity")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)

    fig.suptitle("Recall-Specificity-Trade-off der besten Runden", fontsize=13, fontweight="bold")
    fig.tight_layout()
    outpath = outdir / "04_tradeoff_best_round.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def lookup_threshold_row(threshold_df: pd.DataFrame, threshold_value: float) -> Optional[pd.Series]:
    if threshold_df.empty or "threshold" not in threshold_df.columns:
        return None
    matches = threshold_df[np.isclose(threshold_df["threshold"].astype(float), float(threshold_value), atol=1e-6)]
    if not matches.empty:
        return matches.iloc[0]
    closest_idx = (threshold_df["threshold"].astype(float) - float(threshold_value)).abs().idxmin()
    return threshold_df.loc[closest_idx]


def compute_near_optimal_interval(threshold_df: pd.DataFrame, delta: float = 0.01) -> Tuple[float, float, float, float]:
    if threshold_df.empty or "balanced_accuracy" not in threshold_df.columns:
        return np.nan, np.nan, np.nan, np.nan

    ba_max = float(threshold_df["balanced_accuracy"].max())
    near_opt = threshold_df[threshold_df["balanced_accuracy"] >= ba_max - delta]
    if near_opt.empty:
        return ba_max, np.nan, np.nan, np.nan

    best_idx = safe_idxmax(threshold_df["balanced_accuracy"])
    best_thr = float(threshold_df.loc[best_idx, "threshold"]) if best_idx is not None else np.nan
    return ba_max, best_thr, float(near_opt["threshold"].min()), float(near_opt["threshold"].max())


def format_threshold_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_threshold_stability_per_strategy(root: Path, outdir: Path, include_panel_c: bool = True) -> List[Path]:
    bundles = collect_strategy_bundles(root)
    outputs: List[Path] = []

    for strategy_id, bundle in bundles.items():
        summary_df = bundle["summary"]
        threshold_by_round = bundle["threshold_by_round"]
        if summary_df.empty:
            continue

        rounds = sorted(summary_df["round"].dropna().astype(int).unique().tolist())
        if not rounds:
            continue

        final_round = max(rounds)
        final_row = summary_df.loc[summary_df["round"] == final_round].sort_values("variant").iloc[-1]
        final_threshold = float(final_row.get("best_balanced_accuracy_threshold", np.nan))

        panel_count = 3 if include_panel_c else 2
        fig, axes = plt.subplots(panel_count, 1, figsize=(15, 4.2 * panel_count), sharex=True)
        if panel_count == 1:
            axes = [axes]

        ax_a = axes[0]
        ax_b = axes[1]
        ax_c = axes[2] if include_panel_c else None

        variants_sorted = sorted(summary_df["variant"].dropna().unique().tolist())
        n_variants = len(variants_sorted)
        for variant, group in summary_df.groupby("variant"):
            group = group.sort_values("round")
            try:
                variant_index = variants_sorted.index(variant)
            except ValueError:
                variant_index = 0
            family = strategy_id.split("_")[0]
            base_hex = FAMILY_COLORS.get(family, "#6B7280")
            color = variant_shaded_color(base_hex, variant_index, n_variants)
            marker = VARIANT_MARKERS[variant_index % len(VARIANT_MARKERS)]

            best_thresholds = []
            lower_bounds = []
            upper_bounds = []
            ba_values = []
            ba_fixed_threshold_values = []
            recall_values = []
            spec_values = []
            selected_rounds = []

            for _, row in group.iterrows():
                round_num = int(row["round"])
                threshold_df = threshold_by_round.get(round_num, pd.DataFrame())
                if threshold_df.empty:
                    continue

                ba_max, best_thr, lower_thr, upper_thr = compute_near_optimal_interval(threshold_df)
                best_row = threshold_df.loc[safe_idxmax(threshold_df["balanced_accuracy"])]
                fixed_row = lookup_threshold_row(threshold_df, final_threshold)

                selected_rounds.append(round_num)
                best_thresholds.append(best_thr)
                lower_bounds.append(lower_thr)
                upper_bounds.append(upper_thr)
                ba_values.append(ba_max)
                ba_fixed_threshold_values.append(float(fixed_row.get("balanced_accuracy", np.nan)) if fixed_row is not None else np.nan)
                recall_values.append(float(best_row.get("recall", np.nan)))
                spec_values.append(float(best_row.get("spec", np.nan)))

            if not selected_rounds:
                continue

            selected_rounds_arr = np.asarray(selected_rounds)
            best_thresholds_arr = np.asarray(best_thresholds, dtype=float)
            lower_bounds_arr = np.asarray(lower_bounds, dtype=float)
            upper_bounds_arr = np.asarray(upper_bounds, dtype=float)
            ba_values_arr = np.asarray(ba_values, dtype=float)
            ba_fixed_threshold_arr = np.asarray(ba_fixed_threshold_values, dtype=float)
            recall_values_arr = np.asarray(recall_values, dtype=float)
            spec_values_arr = np.asarray(spec_values, dtype=float)

            ax_a.step(selected_rounds_arr, best_thresholds_arr, where="post", color=color, linewidth=2.0, label=f"Run {variant}")
            ax_a.scatter(
                selected_rounds_arr[-1],
                best_thresholds_arr[-1],
                color=color,
                s=110,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )

            ax_b.plot(
                selected_rounds_arr,
                ba_values_arr,
                color=color,
                linewidth=1.0,
                alpha=0.25,
                marker=marker,
                markersize=3,
                label=f"Run {variant} best BA per round",
            )
            ax_b.plot(
                selected_rounds_arr,
                ba_fixed_threshold_arr,
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.7,
                marker=marker,
                markersize=3,
                label=f"Run {variant} BA at final threshold",
            )

            max_idx = int(np.nanargmax(ba_values_arr))
            ax_b.scatter(selected_rounds_arr[max_idx], ba_values_arr[max_idx], color=color, s=85, marker="*", edgecolors="black", linewidths=0.6, zorder=5)

            if ax_c is not None:
                ax_c.plot(selected_rounds_arr, recall_values_arr, color=color, linestyle="-", linewidth=1.8, alpha=0.9, marker=marker, markersize=3, label=f"Recall run {variant}")
                ax_c.plot(selected_rounds_arr, spec_values_arr, color=color, linestyle="--", linewidth=1.8, alpha=0.9, marker=marker, markersize=3, label=f"Specificity run {variant}")

        ax_a.set_title(f"{strategy_id}: Best threshold trajectory and near-optimal interval")
        ax_a.set_ylabel("Selected threshold")
        format_threshold_axis(ax_a)
        ax_a.grid(alpha=0.25)
        ax_a.legend(fontsize=8, ncol=2)

        ax_b.set_title("Best balanced accuracy per round and BA at final selected threshold")
        ax_b.set_ylabel("Balanced accuracy")
        ax_b.grid(alpha=0.25)
        ax_b.legend(fontsize=8, ncol=2)

        if ax_c is not None:
            ax_c.set_title("Recall and specificity at the selected threshold")
            ax_c.set_xlabel("Round")
            ax_c.set_ylabel("Metric value")
            ax_c.set_ylim(0.0, 1.02)
            ax_c.grid(alpha=0.25)
            ax_c.legend(fontsize=7, ncol=2)

        ax_b.set_xlabel("Round")
        fig.suptitle(f"Threshold stability and performance across rounds - {strategy_id}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        strategy_outdir = outdir / strategy_id
        ensure_outdir(strategy_outdir)
        outpath = strategy_outdir / f"05_threshold_stability_{strategy_id}.png"
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(outpath)

    return outputs


def plot_threshold_stability_by_family(root: Path, outdir: Path, include_panel_c: bool = True) -> List[Path]:
    bundles = collect_strategy_bundles(root)
    # Group bundles by family
    families: Dict[str, List[str]] = {}
    for sid, b in bundles.items():
        fam = b.get("family", sid.split("_")[0])
        families.setdefault(fam, []).append(sid)

    outputs: List[Path] = []
    for fam, sids in families.items():
        # consolidate summary and threshold_by_round per variant
        strategy_outdir = outdir / fam
        ensure_outdir(strategy_outdir)

        # find union of rounds
        all_rounds = set()
        for sid in sids:
            all_rounds.update(bundles[sid]["summary"]["round"].dropna().astype(int).unique().tolist())
        if not all_rounds:
            continue
        rounds = sorted(all_rounds)
        final_round = max(rounds)

        fig, axes = plt.subplots(3 if include_panel_c else 2, 1, figsize=(15, 4.2 * (3 if include_panel_c else 2)), sharex=True)
        if (3 if include_panel_c else 2) == 1:
            axes = [axes]
        ax_a = axes[0]
        ax_b = axes[1]
        ax_c = axes[2] if include_panel_c else None

        for sid in sids:
            b = bundles[sid]
            summary_df = b["summary"]
            threshold_by_round = b["threshold_by_round"]
            if summary_df.empty:
                continue

            # pick final threshold from the last round of this variant if available
            if final_round in threshold_by_round:
                final_thr_row = summary_df[summary_df["round"] == final_round]
                if not final_thr_row.empty:
                    final_threshold = float(final_thr_row.iloc[-1].get("best_balanced_accuracy_threshold", np.nan))
                else:
                    final_threshold = np.nan
            else:
                # fallback: take the last available round for this variant
                if not summary_df.empty:
                    final_threshold = float(summary_df.iloc[-1].get("best_balanced_accuracy_threshold", np.nan))
                else:
                    final_threshold = np.nan

            # collect arrays
            selected_rounds = []
            best_thresholds = []
            lower_bounds = []
            upper_bounds = []
            ba_values = []
            ba_fixed_threshold_values = []
            recall_values = []
            spec_values = []

            for round_num in rounds:
                threshold_df = threshold_by_round.get(round_num, pd.DataFrame())
                if threshold_df.empty:
                    continue
                ba_max, best_thr, lower_thr, upper_thr = compute_near_optimal_interval(threshold_df)
                best_row = threshold_df.loc[safe_idxmax(threshold_df["balanced_accuracy"])]
                fixed_row = lookup_threshold_row(threshold_df, final_threshold) if not np.isnan(final_threshold) else None

                selected_rounds.append(round_num)
                best_thresholds.append(best_thr)
                lower_bounds.append(lower_thr)
                upper_bounds.append(upper_thr)
                ba_values.append(ba_max)
                ba_fixed_threshold_values.append(float(fixed_row.get("balanced_accuracy", np.nan)) if fixed_row is not None else np.nan)
                recall_values.append(float(best_row.get("recall", np.nan)))
                spec_values.append(float(best_row.get("spec", np.nan)))

            if not selected_rounds:
                continue

            selected_rounds_arr = np.asarray(selected_rounds)
            best_thresholds_arr = np.asarray(best_thresholds, dtype=float)
            lower_bounds_arr = np.asarray(lower_bounds, dtype=float)
            upper_bounds_arr = np.asarray(upper_bounds, dtype=float)
            ba_values_arr = np.asarray(ba_values, dtype=float)
            ba_fixed_threshold_arr = np.asarray(ba_fixed_threshold_values, dtype=float)
            recall_values_arr = np.asarray(recall_values, dtype=float)
            spec_values_arr = np.asarray(spec_values, dtype=float)

            # choose shaded color per variant within family
            try:
                variant_index = sids.index(sid)
            except ValueError:
                variant_index = 0
            n_variants = len(sids)
            family_name = b.get("family", sid.split("_")[0])
            base_hex = FAMILY_COLORS.get(family_name, "#6B7280")
            color = variant_shaded_color(base_hex, variant_index, n_variants)
            marker = VARIANT_MARKERS[variant_index % len(VARIANT_MARKERS)]
            label = sid

            ax_a.step(selected_rounds_arr, best_thresholds_arr, where="post", color=color, linewidth=2.0, label=label)
            ax_a.scatter(
                selected_rounds_arr[-1],
                best_thresholds_arr[-1],
                color=color,
                s=110,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )

            ax_b.plot(selected_rounds_arr, ba_values_arr, color=color, linewidth=1.0, alpha=0.25, marker=marker, markersize=3, label=f"{label} best BA per round")
            ax_b.plot(selected_rounds_arr, ba_fixed_threshold_arr, color=color, linewidth=1.6, linestyle="--", alpha=0.7, marker=marker, markersize=3, label=f"{label} BA at final threshold")

            if ax_c is not None:
                ax_c.plot(selected_rounds_arr, recall_values_arr, color=color, linestyle="-", linewidth=1.6, alpha=0.9, marker=marker, markersize=3, label=f"Recall {label}")
                ax_c.plot(selected_rounds_arr, spec_values_arr, color=color, linestyle="--", linewidth=1.6, alpha=0.9, marker=marker, markersize=3, label=f"Spec {label}")

        ax_a.set_title(f"{fam}: Best threshold trajectory (all runs)")
        ax_a.set_ylabel("Selected threshold")
        format_threshold_axis(ax_a)
        ax_a.grid(alpha=0.25)
        ax_a.legend(fontsize=7, ncol=2)

        ax_b.set_title(f"{fam}: Best balanced accuracy per round and BA at final selected threshold (all runs)")
        ax_b.set_ylabel("Balanced accuracy")
        ax_b.grid(alpha=0.25)
        ax_b.legend(fontsize=7, ncol=2)

        if ax_c is not None:
            ax_c.set_title(f"{fam}: Recall & Specificity at selected threshold (all runs)")
            ax_c.set_ylabel("Metric value")
            ax_c.set_ylim(0.0, 1.02)
            ax_c.grid(alpha=0.25)
            ax_c.legend(fontsize=7, ncol=2)

        ax_b.set_xlabel("Round")
        fig.suptitle(f"Threshold stability and performance across rounds - {fam} (all runs)", fontsize=14, fontweight="bold")
        fig.tight_layout()

        outpath = strategy_outdir / f"05_threshold_stability_{fam}_all_runs.png"
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(outpath)

    return outputs


def plot_combined_strategy_view(root: Path, outdir: Path, include_panel_c: bool = True) -> List[Path]:
    bundles = collect_strategy_bundles(root)
    outputs: List[Path] = []

    for strategy_id, bundle in bundles.items():
        summary_df = bundle["summary"]
        threshold_by_round = bundle["threshold_by_round"]
        if summary_df.empty:
            continue

        rounds = sorted(summary_df["round"].dropna().astype(int).unique().tolist())
        if not rounds:
            continue

        final_round = max(rounds)
        final_row = summary_df.loc[summary_df["round"] == final_round].sort_values("variant").iloc[-1]
        final_threshold = float(final_row.get("best_balanced_accuracy_threshold", np.nan))

        cols = 3 if include_panel_c else 2
        fig, axes = plt.subplots(1, cols, figsize=(5.6 * cols, 5.2), sharex=False)
        if cols == 1:
            axes = [axes]

        ax_a = axes[0]
        ax_b = axes[1]
        ax_c = axes[2] if include_panel_c else None

        variants_sorted = sorted(summary_df["variant"].dropna().unique().tolist())
        n_variants = len(variants_sorted)
        for variant, group in summary_df.groupby("variant"):
            group = group.sort_values("round")
            try:
                variant_index = variants_sorted.index(variant)
            except ValueError:
                variant_index = 0
            family = strategy_id.split("_")[0]
            base_hex = FAMILY_COLORS.get(family, "#6B7280")
            color = variant_shaded_color(base_hex, variant_index, n_variants)
            marker = VARIANT_MARKERS[variant_index % len(VARIANT_MARKERS)]

            best_thresholds = []
            lower_bounds = []
            upper_bounds = []
            ba_values = []
            ba_fixed_threshold_values = []
            recall_values = []
            spec_values = []
            selected_rounds = []

            for _, row in group.iterrows():
                round_num = int(row["round"])
                threshold_df = threshold_by_round.get(round_num, pd.DataFrame())
                if threshold_df.empty:
                    continue

                ba_max, best_thr, lower_thr, upper_thr = compute_near_optimal_interval(threshold_df)
                best_row = threshold_df.loc[safe_idxmax(threshold_df["balanced_accuracy"])]
                fixed_row = lookup_threshold_row(threshold_df, final_threshold)

                selected_rounds.append(round_num)
                best_thresholds.append(best_thr)
                lower_bounds.append(lower_thr)
                upper_bounds.append(upper_thr)
                ba_values.append(ba_max)
                ba_fixed_threshold_values.append(float(fixed_row.get("balanced_accuracy", np.nan)) if fixed_row is not None else np.nan)
                recall_values.append(float(best_row.get("recall", np.nan)))
                spec_values.append(float(best_row.get("spec", np.nan)))

            if not selected_rounds:
                continue

            selected_rounds_arr = np.asarray(selected_rounds)
            best_thresholds_arr = np.asarray(best_thresholds, dtype=float)
            lower_bounds_arr = np.asarray(lower_bounds, dtype=float)
            upper_bounds_arr = np.asarray(upper_bounds, dtype=float)
            ba_values_arr = np.asarray(ba_values, dtype=float)
            ba_fixed_threshold_arr = np.asarray(ba_fixed_threshold_values, dtype=float)
            recall_values_arr = np.asarray(recall_values, dtype=float)
            spec_values_arr = np.asarray(spec_values, dtype=float)

            # Panel A - threshold trajectory
            ax_a.plot(selected_rounds_arr, best_thresholds_arr, color=color, linewidth=2.0, marker=marker, markersize=4, label=f"Run {variant}")
            ax_a.fill_between(selected_rounds_arr, lower_bounds_arr, upper_bounds_arr, color=color, alpha=0.15)
            ax_a.set_xlabel("Round")
            ax_a.set_ylabel("Selected threshold")
            ax_a.grid(alpha=0.2)

            # Panel B - best BA
            ax_b.plot(selected_rounds_arr, ba_values_arr, color=color, linewidth=1.0, alpha=0.25, marker=marker, markersize=3, label=f"Run {variant} best BA per round")
            ax_b.plot(selected_rounds_arr, ba_fixed_threshold_arr, color=color, linewidth=1.6, linestyle="--", alpha=0.7, marker=marker, markersize=3, label=f"Run {variant} BA at final threshold")
            max_idx = int(np.nanargmax(ba_values_arr))
            ax_b.scatter(selected_rounds_arr[max_idx], ba_values_arr[max_idx], color=color, s=85, marker="*", edgecolors="black", linewidths=0.6, zorder=5)
            ax_b.set_xlabel("Round")
            ax_b.set_ylabel("Balanced accuracy")
            ax_b.grid(alpha=0.2)

            # Panel C - recall & specificity
            if ax_c is not None:
                ax_c.plot(selected_rounds_arr, recall_values_arr, color=color, linestyle="-", linewidth=1.8, alpha=0.9, marker=marker, markersize=3, label=f"Recall run {variant}")
                ax_c.plot(selected_rounds_arr, spec_values_arr, color=color, linestyle="--", linewidth=1.8, alpha=0.9, marker=marker, markersize=3, label=f"Spec run {variant}")
                ax_c.set_xlabel("Round")
                ax_c.set_ylabel("Metric value")
                ax_c.set_ylim(0.0, 1.02)
                ax_c.grid(alpha=0.2)

        ax_a.set_title("Best threshold trajectory and near-optimal interval")
        ax_b.set_title("Best balanced accuracy per round and BA at final selected threshold")
        if ax_c is not None:
            ax_c.set_title("Recall and specificity at the selected threshold")

        # Layout and save
        strategy_outdir = outdir / strategy_id
        ensure_outdir(strategy_outdir)
        outpath = strategy_outdir / f"06_combined_view_{strategy_id}.png"
        fig.suptitle(f"Combined view - {strategy_id}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(outpath)

    return outputs


def make_all_plots(root: Path, outdir: Path) -> pd.DataFrame:
    df = collect_round_summary(root)
    if df.empty:
        print(f"[WARN] Keine Round-Dateien unter {root} gefunden.")
        return df

    ensure_outdir(outdir)
    outputs = [
        plot_learning_curves(df, outdir),
        plot_threshold_quality_curves(df, outdir),
        plot_best_round_bar_chart(df, outdir),
        plot_tradeoff_best_round(df, outdir),
    ]
    outputs.extend(plot_threshold_stability_per_strategy(root, outdir, include_panel_c=True))
    outputs.extend(plot_threshold_stability_by_family(root, outdir, include_panel_c=True))
    print("[INFO] Saved plots:")
    for output in outputs:
        print(f"  - {output}")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Erstellt Vergleichsplots für die Modelle im IID-Scaling-Experiment.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root-Verzeichnis mit all_rounds_* Unterordnern")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ausgabeordner für die erzeugten Plots")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    make_all_plots(args.root, args.out)


if __name__ == "__main__":
    main()