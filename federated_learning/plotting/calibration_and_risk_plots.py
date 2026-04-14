"""
Calibration Plot & Risk Distribution Plot Generator
Für Federated Learning Model Evaluation

Diese Utility generiert Plots basierend auf aggregierten Metriken vom Server.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_calibration_curve(
    calibration_results: List[Dict],
    title: str = "Calibration Plot (Aggregated)",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[Path]:
    """
    Generiert Calibration Plot.
    
    Perfekte Kalibrierung: Punkte liegen auf der Diagonale y=x
    Über-kalibriert: Punkte unter der Diagonale (Modell zu zuversichtlich)
    Unter-kalibriert: Punkte über der Diagonale
    
    Args:
        calibration_results: Liste von Bin-Dicts mit:
            - mean_predicted_prob: Durchschnittliche vorhergesagte Wahrscheinlichkeit
            - mean_observed_freq: Beobachtete Häufigkeit (Anteil positiver Samples)
            - n_samples: Anzahl Samples in diesem Bin
        title: Plot-Titel
        save_path: Pfad zum Speichern (optional)
        show: Ob Plot angezeigt werden soll
    
    Returns:
        Pfad zur gespeicherten Datei (falls save_path gegeben)
    """
    
    if not calibration_results:
        print("⚠️  Keine Calibration-Daten!")
        return None
    
    # Extrahiere Daten
    mean_preds = [r["mean_predicted_prob"] for r in calibration_results]
    mean_obs = [r["mean_observed_freq"] for r in calibration_results]
    n_bins = [r["n_samples"] for r in calibration_results]
    
    # Filtere leere Bins (n=0)
    valid_indices = [i for i, n in enumerate(n_bins) if n > 0]
    mean_preds = [mean_preds[i] for i in valid_indices]
    mean_obs = [mean_obs[i] for i in valid_indices]
    n_bins = [n_bins[i] for i in valid_indices]
    
    # Erstelle Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot: Perfekte Kalibrierung (Diagonale)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration", linewidth=2)
    
    # Plot: Tatsächliche Kalibrierung (mit Bubble-Größe = Anzahl Samples)
    scatter = ax.scatter(mean_preds, mean_obs, s=[max(n/10, 50) for n in n_bins], 
                        alpha=0.6, edgecolors='black', linewidth=1.5, c='steelblue')
    
    # Verbinde Punkte mit einer Kurve
    sorted_indices = np.argsort(mean_preds)
    sorted_preds = [mean_preds[i] for i in sorted_indices]
    sorted_obs = [mean_obs[i] for i in sorted_indices]
    ax.plot(sorted_preds, sorted_obs, color='steelblue', alpha=0.4, linewidth=1)
    
    # Labels
    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Frequency (Empirical)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Farbe für Bubbles
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("# Samples", fontsize=11)
    
    fig.tight_layout()
    
    # Speichere
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Calibration plot saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return save_path


def plot_risk_distribution(
    risk_distribution: List[Dict],
    title: str = "Risk Distribution (Aggregated)",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[Path]:
    """
    Generiert Risk Distribution Plot (Histogramme für y=0 und y=1).
    
    Gute Trennung: Histogramme überlappen kaum
    Schlechte Trennung: Histogramme überlappen stark
    
    Args:
        risk_distribution: Liste von Bin-Dicts mit:
            - count_y0: Anzahl Samples (negative class) in diesem Bin
            - count_y1: Anzahl Samples (positive class) in diesem Bin
            - bin_edge_lower, bin_edge_upper: Bin-Grenzen
        title: Plot-Titel
        save_path: Pfad zum Speichern (optional)
        show: Ob Plot angezeigt werden soll
    
    Returns:
        Pfad zur gespeicherten Datei (falls save_path gegeben)
    """
    
    if not risk_distribution:
        print("⚠️  Keine Risk Distribution-Daten!")
        return None
    
    # Extrahiere Daten
    counts_y0 = np.array([r["count_y0"] for r in risk_distribution])
    counts_y1 = np.array([r["count_y1"] for r in risk_distribution])
    bin_edges = [r["bin_edge_lower"] for r in risk_distribution] + \
                [risk_distribution[-1]["bin_edge_upper"]]
    bin_centers = np.array([risk_distribution[i]["bin_edge_lower"] + 
                           (risk_distribution[i]["bin_edge_upper"] - risk_distribution[i]["bin_edge_lower"]) / 2 
                           for i in range(len(risk_distribution))])
    bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 0.05
    
    # Erstelle Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogramme
    ax.bar(bin_centers - bin_width/2.2, counts_y0, width=bin_width*0.9, 
           label="y=0 (Healthy)", alpha=0.7, color='steelblue', edgecolor='black')
    ax.bar(bin_centers + bin_width/2.2, counts_y1, width=bin_width*0.9, 
           label="y=1 (Prediabetic/Diabetic)", alpha=0.7, color='coral', edgecolor='black')
    
    # Labels
    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])
    
    fig.tight_layout()
    
    # Speichere
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Risk distribution plot saved: {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)
    return save_path


def generate_both_plots(
    metrics_dict: Dict,
    output_dir: Optional[Path] = None,
    round_num: Optional[int] = None,
    show: bool = False
) -> Dict[str, Optional[Path]]:
    """
    Generiert beide Plots aus den aggregierten Server-Metriken.
    
    Args:
        metrics_dict: Dictionary mit Schlüsseln "calibration" und "risk_distribution"
        output_dir: Verzeichnis zum Speichern (optional)
        round_num: Runde (wird in Dateinamen verwendet)
        show: Ob Plots angezeigt werden sollen
    
    Returns:
        Dict mit beiden Plot-Pfaden
    """
    
    results = {}
    
    # Calibration Plot
    if "calibration" in metrics_dict and metrics_dict["calibration"]:
        calib_path = None
        if output_dir:
            round_str = f"_round_{round_num}" if round_num is not None else ""
            calib_path = Path(output_dir) / f"calibration{round_str}.png"
        
        results["calibration"] = plot_calibration_curve(
            metrics_dict["calibration"],
            title=f"Calibration Plot (Round {round_num})" if round_num else "Calibration Plot",
            save_path=calib_path,
            show=show
        )
    
    # Risk Distribution Plot
    if "risk_distribution" in metrics_dict and metrics_dict["risk_distribution"]:
        risk_path = None
        if output_dir:
            round_str = f"_round_{round_num}" if round_num is not None else ""
            risk_path = Path(output_dir) / f"risk_distribution{round_str}.png"
        
        results["risk_distribution"] = plot_risk_distribution(
            metrics_dict["risk_distribution"],
            title=f"Risk Distribution (Round {round_num})" if round_num else "Risk Distribution",
            save_path=risk_path,
            show=show
        )
    
    return results


def load_and_plot_from_json(
    json_path: str,
    output_dir: Optional[Path] = None,
    show: bool = False
):
    """
    Lädt aggregierte Metriken aus JSON und generiert Plots.
    
    Nützlich für Post-Training Visualisierung.
    
    Args:
        json_path: Pfad zur JSON-Datei mit aggregierten Metriken
        output_dir: Verzeichnis zum Speichern
        show: Ob Plots angezeigt werden sollen
    """
    
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    return generate_both_plots(metrics, output_dir, show=show)


if __name__ == "__main__":
    # Beispiel: Generiere Plots aus gespeicherten Metriken
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        load_and_plot_from_json(json_file, output_dir, show=True)
    else:
        print("Usage: python calibration_and_risk_plots.py <metrics.json> [output_dir]")
