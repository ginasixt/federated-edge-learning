#!/usr/bin/env python3
"""
Calibration & Risk Distribution Plots über alle evaluierten Runden
- Reliability Diagram (Calibration Curve)
- Risk Distribution Histograms (y0 vs y1)
- Performance Metrics über Runden
- Best Threshold Identification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

# ============================================================
# 1. LOAD ALL EVAL DATA
# ============================================================
def load_eval_rounds(eval_dir: Path) -> Dict[int, dict]:
    """Lade alle round_*_eval.json Dateien und returne {round_num: data}"""
    eval_data = {}
    eval_path = Path(eval_dir)
    
    # Finde alle round_*_eval.json Dateien
    json_files = sorted(eval_path.glob("round_*_eval.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                round_num = data['round']
                eval_data[round_num] = data
                print(f"✓ Loaded round {round_num}")
        except Exception as e:
            print(f"✗ Error loading {json_file}: {e}")
    
    return eval_data


# ============================================================
# 2. CALIBRATION METRICS
# ============================================================
def extract_calibration_data(metrics: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrahiere Calibration Daten aus metrics
    Returns: (edges, avg_predicted, empirical_rate, bin_counts)
    """
    import json
    
    calib_edges = np.array(json.loads(metrics['calib_edges_json']), dtype=float)
    calib_bin_n = np.array(json.loads(metrics['calib_bin_n_json']), dtype=float)
    calib_bin_sum_pred = np.array(json.loads(metrics['calib_bin_sum_pred_json']), dtype=float)
    calib_bin_sum_true = np.array(json.loads(metrics['calib_bin_sum_true_json']), dtype=float)
    
    # Berechne Durchschnitte pro Bin
    avg_predicted = np.divide(
        calib_bin_sum_pred, 
        calib_bin_n, 
        where=calib_bin_n > 0,
        out=np.zeros_like(calib_bin_sum_pred)
    )
    
    empirical_rate = np.divide(
        calib_bin_sum_true,
        calib_bin_n,
        where=calib_bin_n > 0,
        out=np.zeros_like(calib_bin_sum_true, dtype=float)
    )
    
    return calib_edges, avg_predicted, empirical_rate, calib_bin_n


def extract_risk_data(metrics: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrahiere Risk Distribution Daten
    Returns: (edges, hist_y0, hist_y1)
    """
    import json
    
    risk_edges = np.array(json.loads(metrics['risk_edges_json']))
    hist_y0 = np.array(json.loads(metrics['hist_pred_y0_json']))
    hist_y1 = np.array(json.loads(metrics['hist_pred_y1_json']))
    
    return risk_edges, hist_y0, hist_y1


# ============================================================
# 3. PLOTTING FUNCTIONS
# ============================================================
def plot_calibration_curves(eval_data: Dict[int, dict], output_dir: Path):
    """
    Erstelle Reliability Diagram für alle Runden
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Calibration Curves (Reliability Diagrams) Over Rounds', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Wähle interessante Runden zum Plotten
    rounds_to_plot = sorted(eval_data.keys())
    if len(rounds_to_plot) > 6:
        # Nimm First, Last und 4 gleichmäßig verteilte dazwischen
        step = len(rounds_to_plot) // 5
        rounds_to_plot = [
            rounds_to_plot[0],
            rounds_to_plot[step],
            rounds_to_plot[2*step],
            rounds_to_plot[3*step],
            rounds_to_plot[4*step],
            rounds_to_plot[-1]
        ]
    
    for idx, round_num in enumerate(rounds_to_plot):
        ax = axes[idx]
        metrics = eval_data[round_num]['metrics']
        
        calib_edges, avg_pred, empirical, bin_counts = extract_calibration_data(metrics)
        
        # Filter empty bins
        mask = bin_counts > 0
        avg_pred_filtered = avg_pred[mask]
        empirical_filtered = empirical[mask]
        bin_counts_filtered = bin_counts[mask]
        
        # Plot: Perfect Calibration (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.5)
        
        # Plot: Actual Calibration (bubble size = sample count)
        scatter = ax.scatter(
            avg_pred_filtered, 
            empirical_filtered, 
            s=bin_counts_filtered * 0.5,  # Size proportional zu Anzahl
            alpha=0.6,
            c=avg_pred_filtered,
            cmap='viridis',
            edgecolors='black',
            linewidth=1
        )
        
        # Verbinde die Punkte
        ax.plot(avg_pred_filtered, empirical_filtered, 'o-', alpha=0.3, linewidth=1)
        
        # Berechne Calibration Error (Expected Calibration Error)
        calibration_error = np.mean(np.abs(avg_pred_filtered - empirical_filtered))
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=10)
        ax.set_ylabel('Empirical Positive Rate', fontsize=10)
        ax.set_title(f'Round {round_num}\nECE={calibration_error:.4f}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'calibration_curves.png'}")
    plt.close()


def plot_risk_distributions(eval_data: Dict[int, dict], output_dir: Path):
    """
    Erstelle Risk Distribution mit VIOLIN PLOTS (oben) + HISTOGRAMS (unten)
    Violin Plot zeigt Dichteverteilung, Histogramm zeigt absolute Häufigkeiten
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wähle interessante Runden
    rounds_to_plot = sorted(eval_data.keys())
    if len(rounds_to_plot) > 6:
        step = len(rounds_to_plot) // 5
        rounds_to_plot = [
            rounds_to_plot[0],
            rounds_to_plot[step],
            rounds_to_plot[2*step],
            rounds_to_plot[3*step],
            rounds_to_plot[4*step],
            rounds_to_plot[-1]
        ]
    
    # Erstelle 2 Reihen (Violins oben, Histogramme unten)
    fig, axes = plt.subplots(2, len(rounds_to_plot), figsize=(5*len(rounds_to_plot), 12))
    if len(rounds_to_plot) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle('Risk Distribution: Violin Plots (Probability Density) + Histograms', 
                 fontsize=16, fontweight='bold')
    
    for col_idx, round_num in enumerate(rounds_to_plot):
        metrics = eval_data[round_num]['metrics']
        risk_edges, hist_y0, hist_y1 = extract_risk_data(metrics)
        
        # Rekonstruiere die ursprünglichen Wahrscheinlichkeiten aus den Bins
        # (Damit können wir Violin Plots machen)
        prob_y0 = []
        prob_y1 = []
        
        for bin_idx in range(len(hist_y0)):
            # Bin center als representative Wahrscheinlichkeit
            bin_center = (risk_edges[bin_idx] + risk_edges[bin_idx+1]) / 2
            prob_y0.extend([bin_center] * hist_y0[bin_idx])
            prob_y1.extend([bin_center] * hist_y1[bin_idx])
        
        # ==================== VIOLIN PLOT (oben) ====================
        ax_violin = axes[0, col_idx]
        
        # Prepare data für Violin Plot
        data_for_violin = {
            'Negative (y=0)': prob_y0,
            'Positive (y=1)': prob_y1
        }
        
        positions = [1, 2]
        parts = ax_violin.violinplot(
            [prob_y0, prob_y1],
            positions=positions,
            widths=0.7,
            showmeans=True,
            showmedians=True
        )
        
        # Färbe die Violin Plots
        colors = ['blue', 'red']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
        
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(['Negative (y=0)', 'Positive (y=1)'])
        ax_violin.set_ylabel('Predicted Probability', fontsize=10)
        ax_violin.set_title(f'Round {round_num} - Probability Distribution', fontweight='bold')
        ax_violin.set_ylim(-0.05, 1.05)
        ax_violin.grid(True, alpha=0.3, axis='y')
        
        # ==================== HISTOGRAM (unten) ====================
        ax_hist = axes[1, col_idx]
        
        bin_centers = (risk_edges[:-1] + risk_edges[1:]) / 2
        bin_width = risk_edges[1] - risk_edges[0]
        
        # Normalisiere zu Proportionen
        hist_y0_norm = hist_y0 / hist_y0.sum() if hist_y0.sum() > 0 else hist_y0
        hist_y1_norm = hist_y1 / hist_y1.sum() if hist_y1.sum() > 0 else hist_y1
        
        # Plot Histogramme nebeneinander
        ax_hist.bar(bin_centers - bin_width/4, hist_y0_norm, width=bin_width/2, 
                   label='Negative (y=0)', alpha=0.7, color='blue')
        ax_hist.bar(bin_centers + bin_width/4, hist_y1_norm, width=bin_width/2, 
                   label='Positive (y=1)', alpha=0.7, color='red')
        
        ax_hist.set_xlabel('Predicted Probability', fontsize=10)
        ax_hist.set_ylabel('Normalized Frequency', fontsize=10)
        ax_hist.set_title(f'Histogram (Normalized)', fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3, axis='y')
        ax_hist.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distributions_with_violins.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'risk_distributions_with_violins.png'}")
    plt.close()


def plot_performance_over_rounds(eval_data: Dict[int, dict], output_dir: Path, threshold: float = 0.45):
    """
    Plotte Key Metrics über alle Runden für einen bestimmten Threshold
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rounds = sorted(eval_data.keys())
    
    # Extrahiere Metriken für jeden Threshold
    metrics_by_round = {}
    for round_num in rounds:
        threshold_data = eval_data[round_num]['metrics']['all_thresholds']
        
        # Finde den nächsten Threshold
        closest_thr = min(threshold_data, key=lambda x: abs(x['threshold'] - threshold))
        metrics_by_round[round_num] = closest_thr
    
    # Extrahiere einzelne Metriken
    f1_scores = [metrics_by_round[r]['f1'] for r in rounds]
    recall = [metrics_by_round[r]['recall'] for r in rounds]
    precision = [metrics_by_round[r]['ppv'] for r in rounds]  # PPV = precision
    balanced_acc = [metrics_by_round[r]['balanced_accuracy'] for r in rounds]
    youden = [metrics_by_round[r]['youden'] for r in rounds]
    
    # Calibration Error über Runden
    calib_errors = []
    for round_num in rounds:
        metrics = eval_data[round_num]['metrics']
        calib_edges, avg_pred, empirical, bin_counts = extract_calibration_data(metrics)
        mask = bin_counts > 0
        calib_error = np.mean(np.abs(avg_pred[mask] - empirical[mask]))
        calib_errors.append(calib_error)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Performance Metrics Over Rounds (Threshold={threshold:.2f})', 
                 fontsize=16, fontweight='bold')
    
    # F1 Score
    axes[0, 0].plot(rounds, f1_scores, 'o-', linewidth=2, markersize=6, color='green')
    axes[0, 0].set_title('F1 Score', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=rounds[np.argmax(f1_scores)], color='red', linestyle='--', alpha=0.5)
    
    # Recall
    axes[0, 1].plot(rounds, recall, 'o-', linewidth=2, markersize=6, color='blue')
    axes[0, 1].set_title('Recall (Sensitivity)', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.75, color='orange', linestyle='--', alpha=0.5, label='Target (0.75)')
    axes[0, 1].legend()
    
    # Precision
    axes[0, 2].plot(rounds, precision, 'o-', linewidth=2, markersize=6, color='purple')
    axes[0, 2].set_title('Precision (PPV)', fontweight='bold')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Balanced Accuracy
    axes[1, 0].plot(rounds, balanced_acc, 'o-', linewidth=2, markersize=6, color='brown')
    axes[1, 0].set_title('Balanced Accuracy', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Youden's Index (best balance)
    axes[1, 1].plot(rounds, youden, 'o-', linewidth=2, markersize=6, color='red')
    axes[1, 1].set_title("Youden's Index (Recall + Spec - 1)", fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=rounds[np.argmax(youden)], color='darkred', linestyle='--', alpha=0.5)
    
    # Calibration Error
    axes[1, 2].plot(rounds, calib_errors, 'o-', linewidth=2, markersize=6, color='orange')
    axes[1, 2].set_title('Expected Calibration Error (ECE)', fontweight='bold')
    axes[1, 2].set_ylabel('ECE (lower = better)')
    axes[1, 2].set_xlabel('Round')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(x=rounds[np.argmin(calib_errors)], color='darkgreen', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'performance_over_rounds_thr{threshold:.2f}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'performance_over_rounds_thr{threshold:.2f}.png'}")
    plt.close()


def plot_threshold_analysis(eval_data: Dict[int, dict], output_dir: Path):
    """
    Wie verändern sich Metriken über verschiedene Thresholds?
    (Für die letzte evaluierte Runde)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nutze die neueste Runde
    latest_round = max(eval_data.keys())
    threshold_data = eval_data[latest_round]['metrics']['all_thresholds']
    
    thresholds = [t['threshold'] for t in threshold_data]
    recall_list = [t['recall'] for t in threshold_data]
    precision_list = [t['ppv'] for t in threshold_data]
    f1_list = [t['f1'] for t in threshold_data]
    spec_list = [t['spec'] for t in threshold_data]
    youden_list = [t['youden'] for t in threshold_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Threshold Analysis - Round {latest_round}', fontsize=14, fontweight='bold')
    
    # Plot 1: Recall, Precision, Specificity
    axes[0].plot(thresholds, recall_list, 'o-', label='Recall', linewidth=2)
    axes[0].plot(thresholds, precision_list, 's-', label='Precision', linewidth=2)
    axes[0].plot(thresholds, spec_list, '^-', label='Specificity', linewidth=2)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Recall, Precision, Specificity vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Threshold
    axes[1].plot(thresholds, f1_list, 'o-', linewidth=2, color='green')
    best_f1_idx = np.argmax(f1_list)
    axes[1].scatter(thresholds[best_f1_idx], f1_list[best_f1_idx], 
                   s=200, color='red', marker='*', zorder=5, label=f'Best F1={f1_list[best_f1_idx]:.3f}')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Youden's Index
    axes[2].plot(thresholds, youden_list, 'o-', linewidth=2, color='red')
    best_youden_idx = np.argmax(youden_list)
    axes[2].scatter(thresholds[best_youden_idx], youden_list[best_youden_idx],
                   s=200, color='green', marker='*', zorder=5, 
                   label=f"Best Youden={youden_list[best_youden_idx]:.3f}")
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel("Youden's Index")
    axes[2].set_title("Youden's Index vs Threshold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis_latest_round.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'threshold_analysis_latest_round.png'}")
    plt.close()


def print_best_round_summary(eval_data: Dict[int, dict]):
    """
    Drucke eine Zusammenfassung der besten Runden nach verschiedenen Metriken
    """
    print("\n" + "="*80)
    print("BEST ROUND SUMMARY")
    print("="*80)
    
    rounds = sorted(eval_data.keys())
    
    # Für jeden Threshold berechne beste Runde
    threshold_to_check = 0.45
    
    best_f1_round = None
    best_f1_score = -1
    best_youden_round = None
    best_youden_score = -1
    best_calib_round = None
    best_calib_error = float('inf')
    best_recall_round = None
    best_recall = -1
    
    for round_num in rounds:
        metrics = eval_data[round_num]['metrics']
        
        # Finde Threshold
        threshold_data = metrics['all_thresholds']
        closest_thr = min(threshold_data, key=lambda x: abs(x['threshold'] - threshold_to_check))
        
        # F1
        if closest_thr['f1'] > best_f1_score:
            best_f1_score = closest_thr['f1']
            best_f1_round = round_num
        
        # Youden
        if closest_thr['youden'] > best_youden_score:
            best_youden_score = closest_thr['youden']
            best_youden_round = round_num
        
        # Recall (für Safety)
        if closest_thr['recall'] > best_recall:
            best_recall = closest_thr['recall']
            best_recall_round = round_num
        
        # Calibration
        calib_edges, avg_pred, empirical, bin_counts = extract_calibration_data(metrics)
        mask = bin_counts > 0
        calib_error = np.mean(np.abs(avg_pred[mask] - empirical[mask]))
        if calib_error < best_calib_error:
            best_calib_error = calib_error
            best_calib_round = round_num
    
    print(f"\nBest F1 Score:       Round {best_f1_round:3d}  |  F1={best_f1_score:.4f}")
    print(f"Best Youden Index:   Round {best_youden_round:3d}  |  Youden={best_youden_score:.4f}")
    print(f"Best Recall:         Round {best_recall_round:3d}  |  Recall={best_recall:.4f}")
    print(f"Best Calibration:    Round {best_calib_round:3d}  |  ECE={best_calib_error:.4f}")
    print("\n" + "="*80 + "\n")


# ============================================================
# 4. MAIN
# ============================================================
if __name__ == "__main__":
    # Pfade
    eval_dir = Path("/home/bax9142/federated-edge-learning/result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/eval_all_rounds/")
    output_dir = Path("/home/bax9142/federated-edge-learning/plots_out_calibration_risk/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Loading evaluation data from: {eval_dir}")
    eval_data = load_eval_rounds(eval_dir)
    
    if not eval_data:
        print("❌ No evaluation data found!")
        exit(1)
    
    print(f"\n✓ Loaded {len(eval_data)} rounds")
    
    # Generate Plots
    print("\n📈 Generating plots...")
    plot_calibration_curves(eval_data, output_dir)
    plot_risk_distributions(eval_data, output_dir)
    plot_performance_over_rounds(eval_data, output_dir, threshold=0.45)
    plot_threshold_analysis(eval_data, output_dir)
    
    # Summary
    print_best_round_summary(eval_data)
    
    print(f"\n✓ All plots saved to: {output_dir}")
