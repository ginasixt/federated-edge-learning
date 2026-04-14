import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

def plot_roc_curve(json_file, output_file=None):
    """
    Plot ROC curve from metrics JSON file.
    
    Args:
        json_file: Path to JSON file with metrics
        output_file: Optional path to save the plot (default: next to json file)
    """
    # Load metrics
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract TPR and FPR from all_thresholds
    thresholds = []
    tprs = []
    fprs = []
    
    for threshold_data in data['metrics']['all_thresholds']:
        thresholds.append(threshold_data['threshold'])
        tprs.append(threshold_data['tpr'])
        fprs.append(threshold_data['fpr'])
    
    # Sort by FPR for proper curve plotting
    sorted_pairs = sorted(zip(fprs, tprs))
    fprs_sorted = [x[0] for x in sorted_pairs]
    tprs_sorted = [x[1] for x in sorted_pairs]
    
    # Add (0,0) and (1,1) points for complete ROC curve
    fprs_plot = [0] + fprs_sorted + [1]
    tprs_plot = [0] + tprs_sorted + [1]
    
    # Calculate AUC (trapezoidal rule)
    auc = np.trapz(tprs_plot, fprs_plot)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fprs_plot, tprs_plot, 'b-o', linewidth=2, markersize=6, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Diagonal')
    
    # Add annotations for selected thresholds
    for i, thr in enumerate(thresholds[::2]):  # Annotate every other threshold to avoid clutter
        idx = i * 2
        plt.annotate(f'{thresholds[idx]:.2f}', 
                    xy=(fprs[idx], tprs[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve - Round {data.get("round", "?")}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # Save plot
    if output_file is None:
        json_path = Path(json_file)
        output_file = json_path.parent / f"roc_curve_round_{data.get('round', 'unknown')}.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    print(f"AUC: {auc:.4f}")
    print(f"Number of thresholds: {len(thresholds)}")
    
    plt.close()


def plot_calibration_curve(json_file, output_file=None):
    """
    Plot calibration curve (reliability diagram) from metrics JSON file.
    Shows predicted probability vs. actual positive fraction.
    
    Args:
        json_file: Path to JSON file with metrics
        output_file: Optional path to save the plot
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    round_num = data['round']
    
    # Extract calibration data
    calib_edges = json.loads(metrics['calib_edges_json'])
    calib_bin_n = json.loads(metrics['calib_bin_n_json'])
    calib_bin_sum_pred = json.loads(metrics['calib_bin_sum_pred_json'])
    calib_bin_sum_true = json.loads(metrics['calib_bin_sum_true_json'])
    
    # Compute calibration metrics per bin
    bin_centers = []
    actual_fractions = []
    
    for i in range(len(calib_bin_n)):
        bin_center = (calib_edges[i] + calib_edges[i+1]) / 2
        bin_centers.append(bin_center)
        
        # Average predicted probability in this bin
        avg_pred = calib_bin_sum_pred[i] / calib_bin_n[i] if calib_bin_n[i] > 0 else 0
        
        # Actual positive fraction in this bin
        actual_frac = calib_bin_sum_true[i] / calib_bin_n[i] if calib_bin_n[i] > 0 else 0
        actual_fractions.append(actual_frac)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Plot calibration curve
    plt.plot(bin_centers, actual_fractions, 'b-o', linewidth=2, markersize=8, label='Model Calibration')
    
    # Calculate Expected Calibration Error (ECE)
    ece = sum([calib_bin_n[i] / sum(calib_bin_n) * abs(bin_centers[i] - actual_fractions[i]) 
               for i in range(len(calib_bin_n)) if calib_bin_n[i] > 0])
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve - Round {round_num} (ECE = {ece:.4f})', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # Save plot
    if output_file is None:
        json_path = Path(json_file)
        output_file = json_path.parent / f"calibration_round_{round_num}.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Calibration plot saved to: {output_file}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    plt.close()


def plot_risk_distribution(json_file, output_file=None):
    """
    Plot risk/probability distribution (histogram) showing predicted probability distribution
    for positive and negative samples.
    
    Visualizes how well the model separates the two classes based on predicted probabilities.
    
    Args:
        json_file: Path to JSON file with metrics
        output_file: Optional path to save the plot
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    round_num = data['round']
    
    # Extract risk distribution data
    risk_edges = json.loads(metrics['risk_edges_json'])
    hist_pred_y0 = json.loads(metrics['hist_pred_y0_json'])  # Negatives (y=0)
    hist_pred_y1 = json.loads(metrics['hist_pred_y1_json'])  # Positives (y=1)
    
    # Compute bin centers
    bin_centers = [(risk_edges[i] + risk_edges[i+1]) / 2 for i in range(len(risk_edges) - 1)]
    bin_width = risk_edges[1] - risk_edges[0]
    
    # Create figure with stacked histogram
    fig, ax = plt.subplots(figsize=(13, 6))
    
    # Normalize counts to percentages for better comparison
    total_samples = sum(hist_pred_y0) + sum(hist_pred_y1)
    hist_y0_norm = [100 * x / total_samples for x in hist_pred_y0]
    hist_y1_norm = [100 * x / total_samples for x in hist_pred_y1]
    
    # Stacked bar plot
    ax.bar(bin_centers, hist_y0_norm, width=bin_width*0.9, label='Negatives (y=0)', 
           alpha=0.8, color='#1f77b4')
    ax.bar(bin_centers, hist_y1_norm, width=bin_width*0.9, bottom=hist_y0_norm,
           label='Positives (y=1)', alpha=0.8, color='#ff7f0e')
    
    # Add a reference line showing perfect separation (0 at 0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Decision Boundary (0.5)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Samples (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk Distribution (Probability Distribution) - Round {round_num}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper center')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([-0.02, 1.02])
    
    # Save plot
    if output_file is None:
        json_path = Path(json_file)
        output_file = json_path.parent / f"risk_distribution_round_{round_num}.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Risk distribution plot saved to: {output_file}")
    
    # Print summary statistics
    neg_at_low = sum(hist_pred_y0[:10])   # Samples with pred prob < 0.5
    pos_at_high = sum(hist_pred_y1[10:])  # Samples with pred prob >= 0.5
    
    print(f"  Negatives at low prob (<0.5): {neg_at_low} ({100*neg_at_low/sum(hist_pred_y0):.1f}%)")
    print(f"  Positives at high prob (>=0.5): {pos_at_high} ({100*pos_at_high/sum(hist_pred_y1):.1f}%)")
    print(f"  Total samples: {total_samples}")
    
    plt.close()


def plot_multi_round_risk_comparison(json_files: List[str], output_file=None):
    """
    Plot risk distributions for multiple rounds for comparison.
    Shows how model's confidence/calibration improves over training.
    
    Args:
        json_files: List of paths to JSON files with metrics
        output_file: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    round_data = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metrics = data['metrics']
        round_num = data['round']
        
        # Extract risk distribution data
        risk_edges = json.loads(metrics['risk_edges_json'])
        hist_pred_y0 = json.loads(metrics['hist_pred_y0_json'])
        hist_pred_y1 = json.loads(metrics['hist_pred_y1_json'])
        
        # Compute bin centers and widths
        bin_centers = [(risk_edges[i] + risk_edges[i+1]) / 2 for i in range(len(risk_edges) - 1)]
        bin_width = risk_edges[1] - risk_edges[0]
        
        # Normalize
        total_samples = sum(hist_pred_y0) + sum(hist_pred_y1)
        hist_y0_norm = [100 * x / total_samples for x in hist_pred_y0]
        hist_y1_norm = [100 * x / total_samples for x in hist_pred_y1]
        
        # Calculate separation quality metric
        neg_at_low = sum(hist_pred_y0[:10])  # How many negatives in <0.5 range
        pos_at_high = sum(hist_pred_y1[10:])  # How many positives in >=0.5 range
        separation_score = (neg_at_low + pos_at_high) / sum(hist_pred_y0 + hist_pred_y1)
        
        round_data.append({
            'round': round_num,
            'bin_centers': bin_centers,
            'bin_width': bin_width,
            'hist_y0_norm': hist_y0_norm,
            'hist_y1_norm': hist_y1_norm,
            'separation_score': separation_score,
            'total_samples': total_samples,
        })
    
    # Sort by round number
    round_data.sort(key=lambda x: x['round'])
    
    # Plot each round
    for idx, data in enumerate(round_data):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Stacked bar plot
        ax.bar(data['bin_centers'], data['hist_y0_norm'], width=data['bin_width']*0.9,
               label='Negatives (y=0)', alpha=0.8, color='#1f77b4')
        ax.bar(data['bin_centers'], data['hist_y1_norm'], width=data['bin_width']*0.9,
               bottom=data['hist_y0_norm'], label='Positives (y=1)', alpha=0.8, color='#ff7f0e')
        
        # Reference line
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title(f'Round {data["round"]} (Separation: {data["separation_score"]:.3f})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim([-0.02, 1.02])
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper center')
    
    # Hide unused subplots
    for idx in range(len(round_data), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Risk Distribution Comparison - Multiple Rounds', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        base_dir = Path(json_files[0]).parent
        output_file = base_dir / "risk_distribution_comparison_multi_rounds.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nMulti-round risk distribution plot saved to: {output_file}")
    print(f"Compared {len(round_data)} rounds:")
    for data in round_data:
        print(f"  Round {data['round']}: Separation Score = {data['separation_score']:.4f}")
    
    plt.close()


def plot_combined_metrics_comparison(json_files: List[str], output_file=None):
    """
    Create a comprehensive comparison dashboard with multiple metrics across rounds:
    - Calibration (ECE)
    - Risk separation score
    - Best threshold performance (F1, Recall, Precision)
    
    Perfect for evaluating model progression!
    
    Args:
        json_files: List of paths to JSON files with metrics
        output_file: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    round_nums = []
    ece_scores = []
    separation_scores = []
    f1_scores = []
    recall_scores = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metrics = data['metrics']
        round_num = data['round']
        round_nums.append(round_num)
        
        # === ECE (Calibration) ===
        calib_edges = json.loads(metrics['calib_edges_json'])
        calib_bin_n = json.loads(metrics['calib_bin_n_json'])
        calib_bin_sum_pred = json.loads(metrics['calib_bin_sum_pred_json'])
        calib_bin_sum_true = json.loads(metrics['calib_bin_sum_true_json'])
        
        bin_centers_calib = [(calib_edges[i] + calib_edges[i+1]) / 2 for i in range(len(calib_edges) - 1)]
        ece = sum([calib_bin_n[i] / sum(calib_bin_n) * abs(bin_centers_calib[i] - 
                  (calib_bin_sum_true[i] / calib_bin_n[i] if calib_bin_n[i] > 0 else 0))
                  for i in range(len(calib_bin_n)) if calib_bin_n[i] > 0])
        ece_scores.append(ece)
        
        # === Risk Separation ===
        risk_edges = json.loads(metrics['risk_edges_json'])
        hist_pred_y0 = json.loads(metrics['hist_pred_y0_json'])
        hist_pred_y1 = json.loads(metrics['hist_pred_y1_json'])
        
        neg_at_low = sum(hist_pred_y0[:10])
        pos_at_high = sum(hist_pred_y1[10:])
        separation_score = (neg_at_low + pos_at_high) / (sum(hist_pred_y0) + sum(hist_pred_y1))
        separation_scores.append(separation_score)
        
        # === Best Threshold Metrics (index -3 = 0.55) ===
        best_threshold_idx = -3
        best_threshold = data['metrics']['all_thresholds'][best_threshold_idx]
        f1_scores.append(best_threshold['f1'])
        recall_scores.append(best_threshold['recall'])
    
    # === Plot 1: ECE (Calibration) - Lower is Better ===
    axes[0, 0].plot(round_nums, ece_scores, 'o-', linewidth=2.5, markersize=8, color='#d62728')
    axes[0, 0].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Expected Calibration Error (ECE)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Model Calibration Over Rounds\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, max(ece_scores) * 1.1])
    
    # === Plot 2: Separation Score (Risk Distribution) - Higher is Better ===
    axes[0, 1].plot(round_nums, separation_scores, 'o-', linewidth=2.5, markersize=8, color='#2ca02c')
    axes[0, 1].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Separation Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Class Separation Quality Over Rounds\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # === Plot 3: F1 Score ===
    axes[1, 0].plot(round_nums, f1_scores, 'o-', linewidth=2.5, markersize=8, color='#1f77b4')
    axes[1, 0].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('F1 Score at Threshold 0.55\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # === Plot 4: Recall Score ===
    axes[1, 1].plot(round_nums, recall_scores, 'o-', linewidth=2.5, markersize=8, color='#ff7f0e')
    axes[1, 1].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Recall at Threshold 0.55\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    fig.suptitle('Comprehensive Model Evaluation - Multiple Rounds', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        base_dir = Path(json_files[0]).parent
        output_file = base_dir / "combined_metrics_comparison.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nCombined metrics comparison plot saved to: {output_file}")
    
    plt.close()
    """
    Plot calibration curves for multiple rounds in one figure for comparison.
    
    Args:
        json_files: List of paths to JSON files with metrics
        output_file: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    round_data = []
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metrics = data['metrics']
        round_num = data['round']
        
        # Extract calibration data
        calib_edges = json.loads(metrics['calib_edges_json'])
        calib_bin_n = json.loads(metrics['calib_bin_n_json'])
        calib_bin_sum_pred = json.loads(metrics['calib_bin_sum_pred_json'])
        calib_bin_sum_true = json.loads(metrics['calib_bin_sum_true_json'])
        
        # Compute calibration metrics per bin
        bin_centers = []
        actual_fractions = []
        
        for i in range(len(calib_bin_n)):
            bin_center = (calib_edges[i] + calib_edges[i+1]) / 2
            bin_centers.append(bin_center)
            
            actual_frac = calib_bin_sum_true[i] / calib_bin_n[i] if calib_bin_n[i] > 0 else 0
            actual_fractions.append(actual_frac)
        
        # Calculate ECE
        ece = sum([calib_bin_n[i] / sum(calib_bin_n) * abs(bin_centers[i] - actual_fractions[i]) 
                   for i in range(len(calib_bin_n)) if calib_bin_n[i] > 0])
        
        round_data.append({
            'round': round_num,
            'bin_centers': bin_centers,
            'actual_fractions': actual_fractions,
            'ece': ece
        })
    
    # Sort by round number for better visualization
    round_data.sort(key=lambda x: x['round'])
    
    # Plot perfectly calibrated line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2.5, label='Perfect Calibration', alpha=0.8)
    
    # Color palette for multiple rounds
    colors = plt.cm.viridis(np.linspace(0, 1, len(round_data)))
    
    # Plot calibration curve for each round
    for idx, data in enumerate(round_data):
        plt.plot(data['bin_centers'], data['actual_fractions'], 
                marker='o', linewidth=2, markersize=6,
                label=f"Round {data['round']} (ECE={data['ece']:.4f})",
                color=colors[idx], alpha=0.8)
    
    plt.xlabel('Mean Predicted Probability', fontsize=13)
    plt.ylabel('Fraction of Positives', fontsize=13)
    plt.title('Calibration Comparison - Multiple Rounds', fontsize=15, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    # Save plot
    if output_file is None:
        base_dir = Path(json_files[0]).parent
        output_file = base_dir / "calibration_comparison_multi_rounds.png"
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nMulti-round calibration plot saved to: {output_file}")
    print(f"Compared {len(round_data)} rounds:")
    for data in round_data:
        print(f"  Round {data['round']}: ECE = {data['ece']:.4f}")
    
    plt.close()

if __name__ == "__main__":
    # ============================================================
    # EXAMPLE 1: Single ROC Curve
    # ============================================================
    print("=" * 70)
    print("EXAMPLE 1: Plot single ROC curve")
    print("=" * 70)
    
    json_file = "result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/eval_all_rounds/round_80_eval.json"
    
    if Path(json_file).exists():
        plot_roc_curve(json_file)
    else:
        print(f"File not found: {json_file}")
    
    
    # ============================================================
    # EXAMPLE 2: Single Calibration Curve
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Plot single calibration curve")
    print("=" * 70)
    
    if Path(json_file).exists():
        plot_calibration_curve(json_file)
    else:
        print(f"File not found: {json_file}")
    
    
    # ============================================================
    # EXAMPLE 3: Single Risk Distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Plot single risk distribution")
    print("=" * 70)
    
    if Path(json_file).exists():
        plot_risk_distribution(json_file)
    else:
        print(f"File not found: {json_file}")
    
    
    # ============================================================
    # EXAMPLE 4: Multi-Round Calibration Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Compare calibration across multiple rounds")
    print("=" * 70)
    
    # Find all eval JSON files in the directory
    eval_dir = Path("result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/eval_all_rounds/")
    
    if eval_dir.exists():
        json_files = sorted(eval_dir.glob("round_*_eval.json"))
        
        if json_files:
            # Select specific rounds for comparison
            selected_rounds = [1, 10, 20, 30, 40, 50, 60, 70, 80]
            selected_files = [f for f in json_files if int(f.stem.split('_')[1]) in selected_rounds]
            
            if selected_files:
                print(f"Plotting calibration for rounds: {[int(f.stem.split('_')[1]) for f in selected_files]}")
                plot_multi_round_calibration(selected_files)
            else:
                print(f"No files found for selected rounds. Available files: {[f.stem for f in json_files]}")
        else:
            print(f"No eval JSON files found in {eval_dir}")
    else:
        print(f"Directory not found: {eval_dir}")
    
    
    # ============================================================
    # EXAMPLE 5: Multi-Round Risk Distribution Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Compare risk distribution across multiple rounds")
    print("=" * 70)
    
    if eval_dir.exists():
        json_files = sorted(eval_dir.glob("round_*_eval.json"))
        
        if json_files:
            selected_rounds = [1, 10, 20, 30, 40, 50, 60, 70, 80]
            selected_files = [f for f in json_files if int(f.stem.split('_')[1]) in selected_rounds]
            
            if selected_files:
                print(f"Plotting risk distribution for rounds: {[int(f.stem.split('_')[1]) for f in selected_files]}")
                plot_multi_round_risk_comparison(selected_files)
            else:
                print(f"No files found for selected rounds")
        else:
            print(f"No eval JSON files found in {eval_dir}")
    else:
        print(f"Directory not found: {eval_dir}")
    
    
    # ============================================================
    # EXAMPLE 6: Combined Comprehensive Comparison (BEST!)
    # ============================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 6: COMPREHENSIVE comparison - All metrics in one dashboard")
    print("=" * 70)
    
    if eval_dir.exists():
        json_files = sorted(eval_dir.glob("round_*_eval.json"))
        
        if json_files:
            selected_rounds = [1, 10, 20, 30, 40, 50, 60, 70, 80]
            selected_files = [f for f in json_files if int(f.stem.split('_')[1]) in selected_rounds]
            
            if selected_files:
                print(f"Creating comprehensive dashboard for rounds: {[int(f.stem.split('_')[1]) for f in selected_files]}")
                plot_combined_metrics_comparison(selected_files)
            else:
                print(f"No files found for selected rounds")
        else:
            print(f"No eval JSON files found in {eval_dir}")
    else:
        print(f"Directory not found: {eval_dir}")
    
    print("\n" + "=" * 70)
    print("✅ All plots generated!")
    print("=" * 70)
