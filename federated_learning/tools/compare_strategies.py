import json
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import auc as sklearn_auc

plt.rcParams.update({
    "figure.dpi": 130,
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLORS = {"Scaffold": "#2563EB", "FedProx": "#DC2626"}
BASE = "/home/bax9142/federated-edge-learning/result/splits_iid_scaling"
OUTPUT_DIR = "/home/bax9142/federated-edge-learning/result/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATHS = {
    "Scaffold": {
        "rounds_dir": f"{BASE}/splits_iid_16384_clients.json/all_rounds_boost2",
        "best_model": f"{BASE}/splits_iid_16384_clients.json/best_model/run_1.json",
    },
    "FedProx": {
        "rounds_dir": f"{BASE}/splits_iid_16384_clients_FedProx.json/all_rounds",
        "best_model": f"{BASE}/splits_iid_16384_clients_FedProx.json/best_model/run_1.json",
    },
}

prevalence = 0.13934878587196467


# ── Data loading ──────────────────────────────────────────────────────────────

def load_best_model(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_round_files(rounds_dir: str) -> dict:
    result = {}
    for fp in glob.glob(os.path.join(rounds_dir, "round_*_run_1.json")):
        m = re.search(r"round_(\d+)_run_1\.json", fp)
        if m:
            r = int(m.group(1))
            with open(fp) as fh:
                result[r] = json.load(fh)
    return dict(sorted(result.items()))


def rounds_table_df(strategy: str, best_models: dict) -> pd.DataFrame:
    rows = best_models[strategy]["screening_summary"]["rounds_table"]
    df = pd.DataFrame(rows)
    df["strategy"] = strategy
    return df


def get_pr_points(strategy: str, round_num: int, round_files: dict, best_models: dict):
    thresholds = round_files[strategy][round_num]["metrics"]["all_thresholds"]
    pts = sorted(
        [(t["recall"], t["precision"], t["threshold"]) for t in thresholds if "recall" in t],
        key=lambda x: x[0],
    )
    recalls    = [0.0] + [p[0] for p in pts] + [1.0]
    precisions = [1.0] + [p[1] for p in pts] + [prevalence]
    thrs       = [None] + [p[2] for p in pts] + [None]
    return np.array(recalls), np.array(precisions), thrs


def get_selected_threshold(strategy: str, round_num: int, best_models: dict):
    bm = best_models[strategy]
    for row in bm["screening_summary"]["rounds_table"]:
        if row["round"] == round_num:
            return row.get("threshold")
    return bm.get("selected_threshold")


def threshold_df(strategy: str, round_num: int, round_files: dict) -> pd.DataFrame:
    thresholds = round_files[strategy][round_num]["metrics"]["all_thresholds"]
    df = pd.DataFrame([t for t in thresholds if "tpr" in t])
    df["strategy"] = strategy
    df["round"] = round_num
    return df


# ── Plot 1: Learning curves AUROC & AUPRC ────────────────────────────────────

def plot_auc_auprc(dfs: dict, best_models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, metric, title in zip(axes, ["auc", "auprc"], ["AUROC", "AUPRC"]):
        ax.axvspan(71, 80, alpha=0.08, color="orange", label="Dense window (71–80)")
        for s, df in dfs.items():
            best_r = best_models[s]["round"]
            ax.plot(df["round"], df[metric], marker="o", markersize=4,
                    color=COLORS[s], label=s, linewidth=2)
            ax.axvline(best_r, color=COLORS[s], linestyle="--", linewidth=1.2, alpha=0.7)
            val = df.loc[df["round"] == best_r, metric].values
            if len(val):
                ax.annotate(f"Best R{best_r}", xy=(best_r, val[0]),
                            xytext=(3, 6), textcoords="offset points",
                            color=COLORS[s], fontsize=8)
        ax.set_xlabel("Training Round")
        ax.set_ylabel(title)
        ax.set_title(f"{title} über Runden")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Lernkurven: AUROC und AUPRC (FedProx vs. Scaffold)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_auroc_auprc.png"), bbox_inches="tight")
    plt.close()


# ── Plot 2: MCC & F1 learning curves ─────────────────────────────────────────

def plot_mcc_f1(dfs: dict, best_models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_mcc_info = {}
    for s, df in dfs.items():
        peak_row = df.loc[df["mcc"].idxmax()]
        max_mcc_info[s] = (int(peak_row["round"]), float(peak_row["mcc"]))

    for ax, metric, title in zip(axes, ["mcc", "f1"], ["MCC", "F1-Score"]):
        ax.axvspan(71, 80, alpha=0.08, color="orange", label="Dense window (71–80)")
        for s, df in dfs.items():
            ax.plot(df["round"], df[metric], marker="o", markersize=4,
                    color=COLORS[s], label=s, linewidth=2)
            if metric == "mcc":
                pr, pm = max_mcc_info[s]
                ax.scatter(pr, pm, s=80, zorder=5, color=COLORS[s], marker="*")
                ax.annotate(f"Peak {pm:.4f}\n(R{pr})", xy=(pr, pm),
                            xytext=(5, -12), textcoords="offset points",
                            color=COLORS[s], fontsize=8)
        ax.set_xlabel("Training Round")
        ax.set_ylabel(title)
        ax.set_title(f"{title} über Runden")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(axis="y", alpha=0.3)

    ax = axes[0]
    for s, df in dfs.items():
        dense = df[df["round"] >= 71].sort_values("round")
        if len(dense) >= 2:
            slope = np.polyfit(dense["round"], dense["mcc"], 1)[0]
            sign = "↑" if slope > 0 else "↓"
            ax.annotate(f"{s} slope: {slope:+.5f}/R {sign}",
                        xy=(0.02, 0.05 if s == "FedProx" else 0.12),
                        xycoords="axes fraction", color=COLORS[s], fontsize=8.5)

    plt.suptitle("Lernkurven: MCC und F1-Score (FedProx vs. Scaffold)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_mcc_f1.png"), bbox_inches="tight")
    plt.close()


# ── Plot 3: PR curve best model ───────────────────────────────────────────────

def plot_pr_best(best_models: dict, round_files: dict):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1.5,
               label=f"Baseline (Prävalenz ≈ {prevalence:.1%})")
    for s in ["Scaffold", "FedProx"]:
        best_r = best_models[s]["round"]
        recalls, precisions, thrs = get_pr_points(s, best_r, round_files, best_models)
        auprc_val = sklearn_auc(recalls, precisions)
        ax.plot(recalls, precisions, color=COLORS[s], linewidth=2.5,
                label=f"{s} (Runde {best_r}, AUPRC={auprc_val:.4f})")
        sel_thr = get_selected_threshold(s, best_r, best_models)
        if sel_thr:
            for r, p, t in zip(recalls, precisions, thrs):
                if t is not None and abs(t - sel_thr) < 0.001:
                    ax.scatter(r, p, s=130, zorder=5, color=COLORS[s], marker="D",
                               edgecolors="white", linewidths=1)
                    ax.annotate(f"  thr={t:.2f}\n  R={r:.3f}\n  P={p:.3f}",
                                xy=(r, p), xytext=(8, 0), textcoords="offset points",
                                color=COLORS[s], fontsize=8.5)
                    break
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"Precision-Recall-Kurve – bestes Modell je Strategie\n"
                 f"(16 384 Clients, IID, Prävalenz ≈ {prevalence:.1%})", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_pr_best_model.png"), bbox_inches="tight")
    plt.close()


# ── Plot 4: PR curve evolution over rounds ────────────────────────────────────

def plot_pr_evolution(best_models: dict, round_files: dict):
    highlight_rounds = [1, 10, 40, 70, 80]
    available_rounds = {s: sorted(round_files[s].keys()) for s in PATHS}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, s in zip(axes, ["Scaffold", "FedProx"]):
        ax.axhline(prevalence, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Baseline (Prävalenz ≈ {prevalence:.1%})")
        rounds_to_plot = [r for r in highlight_rounds if r in available_rounds[s]]
        cmap = plt.cm.get_cmap("Blues" if s == "Scaffold" else "Reds", len(rounds_to_plot) + 2)
        for i, r in enumerate(rounds_to_plot):
            recalls, precisions, _ = get_pr_points(s, r, round_files, best_models)
            auprc_val = sklearn_auc(recalls, precisions)
            lw = 2.5 if r == rounds_to_plot[-1] else 1.4
            ax.plot(recalls, precisions, color=cmap(i + 2), linewidth=lw,
                    label=f"R{r} ({auprc_val:.3f})")
        best_r = best_models[s]["round"]
        if best_r not in rounds_to_plot and best_r in available_rounds[s]:
            recalls, precisions, _ = get_pr_points(s, best_r, round_files, best_models)
            auprc_val = sklearn_auc(recalls, precisions)
            ax.plot(recalls, precisions, color=COLORS[s], linewidth=2.5,
                    label=f"R{best_r} BEST ({auprc_val:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{s} – PR-Kurven Entwicklung")
        ax.legend(title="Runde (AUPRC)", fontsize=9)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    plt.suptitle("PR-Kurven Entwicklung über Runden", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_pr_evolution.png"), bbox_inches="tight")
    plt.close()


# ── Plot 5: Threshold analysis ────────────────────────────────────────────────

def plot_threshold_analysis(best_models: dict, round_files: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = [
        ("tpr",    "TPR (Recall / Sensitivity)"),
        ("fpr",    "FPR (1 – Specificity)"),
        ("f1",     "F1-Score"),
        ("youden", "Youden-Index J"),
    ]
    for ax, (met, label) in zip(axes.flatten(), metrics_to_plot):
        for s in ["Scaffold", "FedProx"]:
            best_r = best_models[s]["round"]
            tdf = threshold_df(s, best_r, round_files)
            ax.plot(tdf["threshold"], tdf[met], color=COLORS[s], linewidth=2,
                    marker="o", markersize=5, label=f"{s} (R{best_r})")
            sel_thr = get_selected_threshold(s, best_r, best_models)
            if sel_thr is not None:
                sel_row = tdf[tdf["threshold"] == sel_thr]
                if not sel_row.empty:
                    ax.scatter(sel_row["threshold"], sel_row[met], s=100,
                               marker="D", zorder=5, color=COLORS[s],
                               edgecolors="white", linewidths=1.2)
        ax.set_xlabel("Schwellenwert (threshold)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(0.1, 0.75)
    plt.suptitle("Threshold-Analyse des besten Modells – FedProx vs. Scaffold\n"
                 "(◆ = gewählter Betriebspunkt)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_threshold_analysis.png"), bbox_inches="tight")
    plt.close()


# ── Plot 6: Screening efficiency ──────────────────────────────────────────────

def plot_screening_efficiency(best_models: dict, round_files: dict):
    fig, ax = plt.subplots(figsize=(10, 7))
    recall_line = np.linspace(0.01, 1.0, 200)
    ideal_alerts = recall_line / prevalence * 10
    ax.plot(recall_line, ideal_alerts, color="gray", linestyle=":", linewidth=1.5,
            label="Ideal (perfekte Precision)")
    for s in ["Scaffold", "FedProx"]:
        best_r = best_models[s]["round"]
        tdf = threshold_df(s, best_r, round_files).sort_values("recall")
        ax.plot(tdf["recall"], tdf["alerts_per_1000"], color=COLORS[s],
                linewidth=2.5, label=f"{s} (R{best_r})")
        for _, row in tdf.iterrows():
            ax.annotate(f"{row['threshold']:.2f}",
                        xy=(row["recall"], row["alerts_per_1000"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7.5, color=COLORS[s], alpha=0.85)
        sel_thr = get_selected_threshold(s, best_r, best_models)
        if sel_thr is not None:
            sel_row = tdf[tdf["threshold"] == sel_thr]
            if not sel_row.empty:
                ax.scatter(sel_row["recall"], sel_row["alerts_per_1000"], s=130,
                           zorder=6, color=COLORS[s], marker="D", edgecolors="white",
                           linewidths=1.5, label=f"{s} Betriebspunkt (thr={sel_thr})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Alerts pro 1 000 Patienten")
    ax.set_title("Screening-Effizienz: Alerts/1000 vs. Recall\n"
                 "Je links-unten, desto effizienter  ◆ = gewählter Betriebspunkt", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0, 1050)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_screening_efficiency.png"), bbox_inches="tight")
    plt.close()


# ── Plot 7: Convergence analysis ──────────────────────────────────────────────

def plot_convergence(dfs: dict, best_models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, s in zip(axes, ["Scaffold", "FedProx"]):
        df = dfs[s]
        dense = df[df["round"] >= 71].sort_values("round").copy()
        ax.plot(df.sort_values("round")["round"], df.sort_values("round")["mcc"],
                color=COLORS[s], alpha=0.3, linewidth=1.5, linestyle="--")
        ax.plot(dense["round"], dense["mcc"], color=COLORS[s], linewidth=2.5,
                marker="o", markersize=6)
        ax.fill_between(dense["round"],
                        dense["mcc"] - dense["mcc"].std(),
                        dense["mcc"] + dense["mcc"].std(),
                        alpha=0.15, color=COLORS[s])
        if len(dense) >= 3:
            rolling = dense["mcc"].rolling(3, center=True).mean()
            ax.plot(dense["round"], rolling, color=COLORS[s], linewidth=2,
                    linestyle="-.", alpha=0.7, label="Rolling mean (w=3)")
        rounds_arr = dense["round"].values
        mcc_arr    = dense["mcc"].values
        slope, intercept = np.polyfit(rounds_arr, mcc_arr, 1)
        ax.plot(rounds_arr, slope * rounds_arr + intercept, color="black",
                linestyle=":", linewidth=1.5, label=f"Trend: {slope:+.5f}/Runde")
        oi = best_models[s]["overtraining_info"]
        ax.axvline(oi["peak_round"], color=COLORS[s], linewidth=1, linestyle="--", alpha=0.6)
        ax.annotate(f"Peak R{oi['peak_round']}\nMCC={oi['peak_mcc']:.4f}",
                    xy=(oi["peak_round"], oi["peak_mcc"]),
                    xytext=(6, -18), textcoords="offset points",
                    fontsize=8.5, color=COLORS[s])
        conv = best_models[s]["convergence_info"]
        info = (f"Konvergenz: {'✓ Plateau' if conv['converged'] else '✗ noch verbessernd'}\n"
                f"Grund: {conv['reason']}\n"
                f"MCC-Std: {conv['mcc_std']:.5f}\n"
                f"AUPRC-Std: {conv['auprc_std']:.5f}\n"
                f"Übertraining: {'✓' if oi['overtraining'] else '✗'}\n"
                f"Drop vom Peak: {oi['relative_drop_pct']:.1f}%\n"
                f"Slope: {oi['trend_slope_per_round']:+.6f}/R")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=8.5,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor=COLORS[s], alpha=0.9))
        ax.set_xlabel("Runde")
        ax.set_ylabel("MCC")
        ax.set_title(f"{s} – MCC im dichten Fenster (R71–80)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
    plt.suptitle("Konvergenz-Analyse: MCC-Entwicklung im dichten Evaluierungs-Fenster",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_convergence.png"), bbox_inches="tight")
    plt.close()


# ── Plot 8: Threshold selection over rounds ───────────────────────────────────

def plot_threshold_selection(dfs: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, s in zip(axes, ["Scaffold", "FedProx"]):
        df = dfs[s].sort_values("round")
        ax2 = ax.twinx()
        ax.fill_between(df["round"], df["threshold"] - 0.02, df["threshold"] + 0.02,
                        alpha=0.2, color=COLORS[s])
        ax.plot(df["round"], df["threshold"], color=COLORS[s], linewidth=2.5,
                marker="o", markersize=6, label="Threshold")
        ax2.plot(df["round"], df["recall"], color="green", linewidth=1.8,
                 linestyle="--", marker="s", markersize=4, alpha=0.8, label="Recall")
        ax.axvspan(71, 80, alpha=0.08, color="orange")
        ax.set_xlabel("Runde")
        ax.set_ylabel("Gewählter Schwellenwert", color=COLORS[s])
        ax2.set_ylabel("Recall", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax.set_title(f"{s} – Threshold & Recall über Runden")
        ax.set_ylim(0.1, 0.85)
        ax2.set_ylim(0.3, 0.95)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.grid(alpha=0.2)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    plt.suptitle("Threshold-Selektion über Runden (recall_constrained)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_threshold_selection.png"), bbox_inches="tight")
    plt.close()


# ── Summary table (printed to stdout) ────────────────────────────────────────

def print_summary(best_models: dict):
    rows = []
    for s in ["Scaffold", "FedProx"]:
        bm = best_models[s]
        ss = bm["screening_summary"]
        oi = bm["overtraining_info"]
        ci = bm["convergence_info"]
        rows.append({
            "Strategie":               s,
            "Bestes Modell (Runde)":   bm["round"],
            "AUC":                     f"{bm['auc']:.4f}",
            "AUPRC":                   f"{bm['auprc']:.4f}",
            "MCC":                     f"{bm['mcc']:.4f}",
            "Recall":                  f"{bm['metrics']['recall']:.4f}",
            "Specificity":             f"{bm['metrics']['spec']:.4f}",
            "F1":                      f"{bm['metrics']['f1']:.4f}",
            "NPV":                     f"{bm['metrics']['npv']:.4f}",
            "Alerts/1000":             f"{bm['metrics']['alerts_per_1000']:.1f}",
            "Threshold":               bm.get("selected_threshold"),
            "AUC mean±std":            f"{ss['auc']['mean']:.4f}±{ss['auc']['std']:.4f}",
            "AUPRC mean±std":          f"{ss['auprc']['mean']:.4f}±{ss['auprc']['std']:.4f}",
            "MCC mean±std":            f"{ss['mcc']['mean']:.4f}±{ss['mcc']['std']:.4f}",
            "MCC max":                 f"{ss['mcc']['max']:.4f}",
            "Alerts/1000 mean":        f"{ss['alerts_per_1000']['mean']:.1f}",
            "Alerts/1000 min":         f"{ss['alerts_per_1000']['min']:.1f}",
            "Konvergiert":             "✓" if ci["converged"] else "✗",
            "Konvergenz-Grund":        ci["reason"],
            "Übertraining":            "✓" if oi["overtraining"] else "✗",
            "Peak MCC (Runde)":        f"{oi['peak_mcc']} (R{oi['peak_round']})",
            "Drop vom Peak (%)":       f"{oi['relative_drop_pct']:.1f}%",
            "Trend-Slope/Runde":       f"{oi['trend_slope_per_round']:+.6f}",
        })
    df = pd.DataFrame(rows).set_index("Strategie").T
    pd.set_option("display.max_colwidth", 30)
    print("\n" + "=" * 60)
    print("Kennzahlen-Vergleich: FedProx vs. Scaffold (16 384 Clients, IID)")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    best_models = {s: load_best_model(PATHS[s]["best_model"]) for s in PATHS}
    round_files  = {s: load_round_files(PATHS[s]["rounds_dir"]) for s in PATHS}

    for s, rf in round_files.items():
        print(f"{s}: {len(rf)} round files – rounds {sorted(rf.keys())}")

    dfs = {s: rounds_table_df(s, best_models) for s in PATHS}

    print("\n[1/8] AUROC & AUPRC learning curves")
    plot_auc_auprc(dfs, best_models)

    print("[2/8] MCC & F1 learning curves")
    plot_mcc_f1(dfs, best_models)

    print("[3/8] PR curve – best model")
    plot_pr_best(best_models, round_files)

    print("[4/8] PR curve evolution")
    plot_pr_evolution(best_models, round_files)

    print("[5/8] Threshold analysis")
    plot_threshold_analysis(best_models, round_files)

    print("[6/8] Screening efficiency")
    plot_screening_efficiency(best_models, round_files)

    print("[7/8] Convergence analysis")
    plot_convergence(dfs, best_models)

    print("[8/8] Threshold selection over rounds")
    plot_threshold_selection(dfs)

    print_summary(best_models)


if __name__ == "__main__":
    main()