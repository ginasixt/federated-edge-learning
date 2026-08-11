"""
Finale zentrale Test-Evaluation nach FL-Training.
Verwendet FESTEN Threshold aus Validation (run_X.json).

Usage:
    python federated_learning/tools/final_test_evaluation_with_val_threshold.py \
        --result-json result/alpha03/multi_thr/run_1.json \
        --parquet data/diabetes.parquet \
        --stats data/norm_stats.json \
        --output final_evaluation
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict

from federated_learning.client_app import MLP
from federated_learning.task import load_prepared


def load_training_result(result_json: Path) -> Dict:
    """Lädt Trainings-Ergebnis mit bestem Threshold aus Validation."""
    if not result_json.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_json}")
    
    result = json.loads(result_json.read_text())
    
    # Extrahiere Informationen
    checkpoint_path = Path(result["model_checkpoint"])
    best_threshold = result["metrics"]["best_threshold"]
    round_num = result["round"]
    
    print(f" Loaded training result:")
    print(f"   Best Round:      {round_num}")
    print(f"   Val Threshold:   {best_threshold:.4f}")
    print(f"   Val AUC:         {result['metrics']['auc']:.4f}")
    print(f"   Val Recall:      {result['metrics']['recall']:.4f}")
    print(f"   Val Specificity: {result['metrics']['spec']:.4f}")
    print(f"   Checkpoint:      {checkpoint_path}")
    
    return {
        "checkpoint": checkpoint_path,
        "threshold": best_threshold,
        "round": round_num,
        "val_metrics": result["metrics"]
    }


def evaluate_on_test(
    model: MLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fixed_threshold: float,
    output_dir: Path
) -> Dict:
    """
    Evaluiert Modell auf Test-Set mit FESTEM Threshold aus Validation.
    
    Args:
        model: Trainiertes MLP
        X_test: Test-Features
        y_test: Test-Labels
        fixed_threshold: Threshold aus Validation (NICHT aus Test!)
        output_dir: Ausgabe-Verzeichnis
        
    Returns:
        Dict mit allen Metriken (ohne roc_data)
    """
    model.eval()
    
    # 1) Predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    
    # 2) AUC (threshold-unabhängig)
    auc = roc_auc_score(y_test, probs)
    
    # 3) ROC-Kurve für Visualisierung
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    
    # 4)  Verwende FESTEN Threshold aus Training
    preds = (probs >= fixed_threshold).astype(int)
    
    # 5) Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 6) Berechne alle Metriken
    def safe_div(a, b):
        return float(a) / float(b) if b > 0 else 0.0
    
    recall = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    bal_acc = (recall + spec) / 2
    youden = recall + spec - 1.0
    prevalence = safe_div(tp + fn, len(y_test))
    alerts_per_1000 = safe_div(tp + fp, len(y_test)) * 1000
    
    # 7)  Separiere ROC-Daten (für Plot) und Report (für JSON)
    roc_data = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }
    
    # 8) Report erstellen (OHNE roc_data für kompaktes JSON)
    report = {
        "test_samples": int(len(y_test)),
        "test_prevalence": float(prevalence),
        "auc": float(auc),
        "threshold_used": float(fixed_threshold),
        "threshold_source": "validation",
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), 
            "fn": int(fn), "tp": int(tp)
        },
        "metrics": {
            "recall": float(recall),
            "tpr": float(recall),
            "specificity": float(spec),
            "precision": float(precision),
            "ppv": float(precision),
            "npv": float(npv),
            "f1": float(f1),
            "balanced_accuracy": float(bal_acc),
            "youden": float(youden),
            "alerts_per_1000": float(alerts_per_1000),
            "fpr": float(1 - spec)
        }
    }
    
    # 9) Speichere Report (kompakt)
    report_path = output_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    # 10)  Speichere ROC-Daten separat (nur für internen Gebrauch)
    roc_path = output_dir / "test_roc_data.npz"
    np.savez(roc_path, fpr=fpr, tpr=tpr, thresholds=thresholds)
    print(f"  ROC data saved separately: {roc_path}")
    
    return report, roc_data  #  Gebe beides zurück (für Plot)


def plot_scientific_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    auc: float,
    fixed_threshold: float,
    metrics: Dict,
    output_path: Path
):
    """
    Erstellt wissenschaftliche ROC-Kurve mit optimalem Threshold.
    """
    # Finde Punkt für festen Threshold
    idx = np.argmin(np.abs(thresholds - fixed_threshold))
    
    # Erstelle Figure mit professionellem Layout
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ROC-Kurve (dickere Linie für bessere Sichtbarkeit)
    ax.plot(fpr, tpr, linewidth=2.5, color='#1f77b4', 
           label=f'ROC Curve (AUC = {auc:.3f})', zorder=2)
    
    # Random Classifier (Diagonale)
    ax.plot([0, 1], [0, 1], '--', linewidth=1.5, color='#7f7f7f', 
           alpha=0.6, label='Random Classifier', zorder=1)
    
    #  Operating Point: Dezenter Marker statt Stern
    ax.plot(fpr[idx], tpr[idx], 'o', markersize=10, 
            color='#d62728', markeredgecolor='black', 
            markeredgewidth=1.5, zorder=3,
            label=f'Operating Point (t = {fixed_threshold:.3f})')
    
    # Annotation mit wichtigsten Metriken
    annotation_text = (
        f'Sensitivity: {tpr[idx]:.3f}\n'
        f'Specificity: {1-fpr[idx]:.3f}\n'
        f'Precision: {metrics["precision"]:.3f}'
    )
    
    # Positioniere Annotation intelligent (oben rechts wenn Punkt unten links)
    if fpr[idx] < 0.5 and tpr[idx] < 0.5:
        xytext = (fpr[idx] + 0.15, tpr[idx] + 0.15)
        ha = 'left'
    else:
        xytext = (fpr[idx] - 0.15, tpr[idx] - 0.15)
        ha = 'right'
    
    ax.annotate(
        annotation_text,
        xy=(fpr[idx], tpr[idx]),
        xytext=xytext,
        fontsize=10,
        ha=ha,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='#d62728', alpha=0.9, linewidth=1.5),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='#d62728')
    )
    
    # Achsenbeschriftungen (wissenschaftlicher Standard)
    ax.set_xlabel('False Positive Rate (1 − Specificity)', 
                 fontsize=13, fontweight='normal')
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                 fontsize=13, fontweight='normal')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Legende (kompakter)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
             edgecolor='black', frameon=True)
    
    # Grid (dezent)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
    # Achsengrenzen
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    # Aspect Ratio 1:1
    ax.set_aspect('equal', adjustable='box')
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Tight Layout
    plt.tight_layout()
    
    # Speichern mit hoher Auflösung (für Publikationen)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ROC curve saved: {output_path}")


def plot_confusion_matrix(
    cm_dict: Dict,
    metrics: Dict,
    threshold: float,
    output_path: Path
):
    """Plot Confusion Matrix mit Metriken."""
    tn, fp = cm_dict["tn"], cm_dict["fp"]
    fn, tp = cm_dict["fn"], cm_dict["tp"]
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Heatmap
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    
    # Colorbar
    #cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12)
    
    # Annotationen
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n\n{cm[i, j]:,}",
                   ha="center", va="center",
                   fontsize=18, fontweight='bold',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # Achsen
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'], 
                       fontsize=12, fontweight='bold')
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'], 
                       fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Test Set Confusion Matrix\n(Validation-Optimized Threshold)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Metriken-Footer
    footer = (
        f"Threshold: {threshold:.3f} (from validation) | "
        f"Sensitivity: {metrics['recall']:.3f} | "
        f"Specificity: {metrics['specificity']:.3f} | "
        f"Precision: {metrics['precision']:.3f} | "
        f"F1-Score: {metrics['f1']:.3f}"
    )
    fig.text(0.5, 0.02, footer, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Confusion matrix saved: {output_path}")


def print_report(
    auc: float,
    report: Dict,
    threshold: float,
    y_test: np.ndarray,
    val_metrics: Dict
):
    """Druckt ausführlichen Test-Report."""
    
    #  Extrahiere Metriken aus report
    metrics = report["metrics"]
    cm = report["confusion_matrix"]
    
    print("\n" + "="*80)
    print("🎯 FINAL TEST EVALUATION REPORT")
    print("="*80)
    
    print(f"\n  Test Dataset Statistics:")
    print(f"   Total samples:      {len(y_test):,}")
    print(f"   Positive cases:     {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
    print(f"   Negative cases:     {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
    
    print(f"\n🎯 Threshold Configuration:")
    print(f"   Source:             Validation Set")
    print(f"   Value:              {threshold:.4f}")
    print(f"   ⚠️  NOT optimized on test data (prevents data leakage)")
    
    print(f"\n📈 Test Set Performance:")
    print(f"   ┌─────────────────────────────────┬──────────┐")
    print(f"   │ Metric                          │  Value   │")
    print(f"   ├─────────────────────────────────┼──────────┤")
    print(f"   │ AUC                             │  {auc:.4f}  │")
    print(f"   │ Sensitivity (Recall/TPR)        │  {metrics['recall']:.4f}  │")
    print(f"   │ Specificity (TNR)               │  {metrics['specificity']:.4f}  │")
    print(f"   │ Precision (PPV)                 │  {metrics['precision']:.4f}  │")
    print(f"   │ Negative Predictive Value (NPV) │  {metrics['npv']:.4f}  │")
    print(f"   │ F1-Score                        │  {metrics['f1']:.4f}  │")
    print(f"   │ Balanced Accuracy               │  {metrics['balanced_accuracy']:.4f}  │")
    print(f"   │ Youden's Index                  │  {metrics['youden']:.4f}  │")
    print(f"   └─────────────────────────────────┴──────────┘")
    
    print(f"\n⚠️  Clinical Resource Impact:")
    print(f"   Alerts per 1000 patients: {metrics['alerts_per_1000']:.1f}")
    print(f"   (Patients requiring follow-up: TP + FP)")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   ┌──────────────┬──────────────┐")
    print(f"   │ TN = {cm['tn']:6,} │ FP = {cm['fp']:6,} │  Predicted Negative / Positive")
    print(f"   ├──────────────┼──────────────┤")
    print(f"   │ FN = {cm['fn']:6,} │ TP = {cm['tp']:6,} │")
    print(f"   └──────────────┴──────────────┘")
    print(f"     Actual Neg      Actual Pos")
    
    print(f"\n  Validation vs. Test Performance:")
    print(f"   ┌──────────────────┬────────────┬────────────┬────────────┐")
    print(f"   │ Metric           │ Validation │    Test    │    Δ       │")
    print(f"   ├──────────────────┼────────────┼────────────┼────────────┤")
    print(f"   │ AUC              │   {val_metrics['auc']:6.4f}   │  {auc:6.4f}  │  {auc - val_metrics['auc']:+6.4f}  │")
    print(f"   │ Recall           │   {val_metrics['recall']:6.4f}   │  {metrics['recall']:6.4f}  │  {metrics['recall'] - val_metrics['recall']:+6.4f}  │")
    print(f"   │ Specificity      │   {val_metrics['spec']:6.4f}   │  {metrics['specificity']:6.4f}  │  {metrics['specificity'] - val_metrics['spec']:+6.4f}  │")
    print(f"   │ F1-Score         │   {val_metrics['f1']:6.4f}   │  {metrics['f1']:6.4f}  │  {metrics['f1'] - val_metrics['f1']:+6.4f}  │")
    print(f"   └──────────────────┴────────────┴────────────┴────────────┘")
    
    # Generalization Assessment
    auc_drop = val_metrics['auc'] - auc
    recall_drop = val_metrics['recall'] - metrics['recall']
    spec_drop = val_metrics['spec'] - metrics['specificity']
    
    print(f"\n🔍 Generalization Assessment:")
    if abs(auc_drop) < 0.02 and abs(recall_drop) < 0.05 and abs(spec_drop) < 0.05:
        print(f"    EXCELLENT: Model generalizes well to unseen data")
    elif abs(auc_drop) < 0.05 and abs(recall_drop) < 0.10 and abs(spec_drop) < 0.10:
        print(f"   ✓  GOOD: Acceptable generalization performance")
    else:
        print(f"   ⚠️  CAUTION: Significant performance drop on test set")
        print(f"      Consider: More training data, regularization, or different architecture")
    
    print("\n" + "="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Final test evaluation with FIXED threshold from validation"
    )
    parser.add_argument(
        "--result-json",
        required=True,
        type=Path,
        help="Path to run_X.json (contains best threshold from validation)"
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to prepared.parquet"
    )
    parser.add_argument(
        "--stats",
        required=True,
        help="Path to norm_stats.json"
    )
    parser.add_argument(
        "--output",
        default="final_evaluation",
        type=Path,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # 1) Lade Training-Ergebnis (mit bestem Threshold aus Val)
    print("\n" + "="*80)
    print("Loading Training Results...")
    print("="*80)
    training_result = load_training_result(args.result_json)
    
    # 2) Lade Test-Daten
    print("\nLoading Test Data...")
    X, y, train_idx, val_idx, test_idx = load_prepared(args.parquet, args.stats)
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f" Test set loaded: {len(test_idx):,} samples")
    
    # 3) Lade Modell
    print("\nLoading Model...")
    model = MLP(in_dim=X_test.shape[1])
    checkpoint = torch.load(training_result["checkpoint"], map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    print(f" Model loaded from checkpoint")
    
    # 4)  Evaluiere mit FESTEM Threshold aus Training
    print("\nEvaluating on Test Set...")
    args.output.mkdir(parents=True, exist_ok=True)
    report, roc_data = evaluate_on_test(  #  Beides empfangen
        model,
        X_test,
        y_test,
        training_result["threshold"],
        args.output
    )
    
    # 5) Plot ROC (mit separaten roc_data)
    print("\nGenerating Plots...")
    plot_scientific_roc(
        roc_data["fpr"],
        roc_data["tpr"],
        roc_data["thresholds"],
        report["auc"],
        training_result["threshold"],
        report["metrics"],
        args.output / "test_roc_curve.png"
    )
    
    # 6) Plot Confusion Matrix
    plot_confusion_matrix(
        report["confusion_matrix"],
        report["metrics"],
        training_result["threshold"],
        args.output / "test_confusion_matrix.png"
    )
    
    # 7) Print Report
    print_report(
        report["auc"],
        report,
        training_result["threshold"],
        y_test,
        training_result["val_metrics"]
    )
    
    # 8) Final Summary
    print(f" Evaluation Complete!")
    print(f"\n📁 Results saved to: {args.output}/")
    print(f"   • test_report.json           (all metrics as JSON)")
    print(f"   • test_roc_curve.png         (publication-ready ROC)")
    print(f"   • test_confusion_matrix.png  (confusion matrix)")
    print(f"\n⚠️  Threshold used: {training_result['threshold']:.4f} (from validation set)")
    print(f"   This prevents data leakage and reflects real-world deployment!\n")


if __name__ == "__main__":
    main()
