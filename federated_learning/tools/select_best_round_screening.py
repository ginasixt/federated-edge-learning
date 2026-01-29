"""
Post-Training Screening Policy: Wähle beste Runde aus allen gespeicherten Checkpoints.

Usage:
    python federated_learning/tools/select_best_round_screening.py \
        --rounds-dir result/splits_iid_scaling/splits_iid_65536_clients.json/all_rounds/ \
        --output-dir result/splits_iid_scaling/splits_iid_65536_clients.json/best_model/
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from federated_learning.screening_policy import ScreeningPolicy


def load_all_rounds(rounds_dir: Path) -> List[Dict]:
    """Lade alle gespeicherten Runden-Metriken."""
    rounds = []
    
    for json_file in sorted(rounds_dir.glob("round_*_run_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            rounds.append({
                "round": data["round"],
                "metrics": data["metrics"],
                "checkpoint": Path(data["model_checkpoint"]),
                "json_path": json_file
            })
        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")
    
    return rounds


def apply_screening(
    rounds: List[Dict],
    min_recall: float = 0.66,
    min_spec: float = 0.60
) -> Optional[Dict]:
    """
    Wende Screening Policy auf alle Runden an.
    
    Returns:
        Dict mit bester Runde oder None
    """
    if not rounds:
        return None
    
    # ✅ Initialisiere Screening Policy
    screening = ScreeningPolicy(min_recall=min_recall, min_spec=min_spec)
    
    # ✅ Füge alle Runden hinzu
    for r in rounds:
        screening.add_round(r["round"], r["metrics"])
    
    # ✅ Wähle beste Runde
    best = screening.best()
    
    if not best:
        print("⚠️  No round passed screening criteria!")
        return None
    
    # ✅ Finde zugehörigen Checkpoint
    best_round = best["round"]
    best_data = next(r for r in rounds if r["round"] == best_round)
    
    return {
        "round": best_round,
        "metrics": best["metrics"],
        "checkpoint": best_data["checkpoint"],
        "summary": screening.get_summary()
    }


def save_best_model(best: Dict, output_dir: Path, run_tag: str = "1"):
    """Kopiere bestes Modell + Metrics in Output-Verzeichnis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Kopiere Checkpoint
    src_checkpoint = best["checkpoint"]
    dst_checkpoint = output_dir / f"model_round_{best['round']}.pt"
    shutil.copy2(src_checkpoint, dst_checkpoint)
    
    # 2) Speichere Metrics
    result = {
        "round": best["round"],
        "metrics": best["metrics"],
        "model_checkpoint": str(dst_checkpoint),
        "screening_summary": best["summary"]
    }
    
    json_path = output_dir / f"run_{run_tag}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Best model saved:")
    print(f"   Round:       {best['round']}")
    print(f"   Checkpoint:  {dst_checkpoint}")
    print(f"   Metrics:     {json_path}")
    print(f"\n📊 Performance:")
    print(f"   Recall:      {best['metrics']['recall']:.3f}")
    print(f"   Specificity: {best['metrics']['spec']:.3f}")
    print(f"   F1-Score:    {best['metrics']['f1']:.3f}")
    print(f"   AUC:         {best['metrics']['auc']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply Screening Policy to select best training round"
    )
    parser.add_argument(
        "--rounds-dir",
        type=Path,
        required=True,
        help="Directory with all_rounds/*.json files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for best model"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.66,
        help="Minimum recall constraint (default: 0.66)"
    )
    parser.add_argument(
        "--min-spec",
        type=float,
        default=0.60,
        help="Minimum specificity preference (default: 0.60)"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="1",
        help="Run tag for output filename (default: 1)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("POST-TRAINING SCREENING POLICY")
    print(f"{'='*80}\n")
    
    # 1) Lade alle Runden
    print(f"Loading rounds from: {args.rounds_dir}")
    rounds = load_all_rounds(args.rounds_dir)
    print(f"✅ Loaded {len(rounds)} evaluated rounds\n")
    
    if not rounds:
        print("❌ No rounds found!")
        return
    
    # 2) Screening
    print(f"Applying Screening Policy:")
    print(f"  - Min Recall:      {args.min_recall:.2f}")
    print(f"  - Min Specificity: {args.min_spec:.2f}\n")
    
    best = apply_screening(
        rounds,
        min_recall=args.min_recall,
        min_spec=args.min_spec
    )
    
    if not best:
        print("❌ No round passed screening!")
        return
    
    # 3) Speichere bestes Modell
    save_best_model(best, args.output_dir, args.run_tag)
    
    print(f"\n{'='*80}")
    print("SCREENING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()