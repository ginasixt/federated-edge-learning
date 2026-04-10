"""
Post-Training Screening Policy: select the best round and threshold from all saved checkpoints.

Handles the mixed evaluation schedule of this FL setup:
  - Rounds 1-70: evaluated every 10 rounds (sparse)
  - Rounds 71-80: evaluated every round (dense)

Uses Net Benefit strategy (recommended for medical screening):
  NB = recall - cost_ratio × (1 - specificity)
  
  For diabetes screening, cost_ratio ≈ 0.3-0.4 (FN is 2.5-3x more costly than FP).

The policy uses the full multi-threshold curve from each round to:
  1. Select the optimal threshold for each round (max NB + feasibility constraints)
  2. Recompute AUC from the ROC curve (server field is unreliable)
  3. Compare rounds on their best achievable Net Benefit
  4. Detect convergence and overtraining across rounds

Usage:
    python3 -m federated_learning.tools.select_best_round_screening \
        --rounds-dir result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds_FedProx_5/ \
        --output-dir result/splits_iid_scaling/splits_iid_16384_clients.json/best_model/ \
        --min-recall 0.70 \
        --max-alerts 500 \
        --strategy net_benefit \
        --cost-ratio 0.5 \
        --dense-window 10

"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from federated_learning.screening_policy import ScreeningPolicy


def resolve_checkpoint_path(
    checkpoint_str: str,
    rounds_dir: Path,
    json_path: Path,
    round_num: int,
) -> Path:
    """Resolve checkpoint path robustly when JSON contains stale paths."""
    raw = Path(checkpoint_str)

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(json_path.parent / raw)

    candidates.append(rounds_dir / raw.name)
    candidates.append(rounds_dir / f"model_round_{round_num}.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return raw


def load_all_rounds(rounds_dir: Path) -> List[Dict]:
    """Load all saved round metric files from an all_rounds/ directory."""
    rounds = []
    for json_file in sorted(rounds_dir.glob("round_*_run_*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            checkpoint = resolve_checkpoint_path(
                checkpoint_str=data["model_checkpoint"],
                rounds_dir=rounds_dir,
                json_path=json_file,
                round_num=data["round"],
            )
            rounds.append({
                "round":      data["round"],
                "metrics":    data["metrics"],
                "checkpoint": checkpoint,
                "json_path":  json_file,
            })
        except Exception as e:
            print(f"  WARNING: Failed to load {json_file.name}: {e}")
    return rounds


def apply_screening(
    rounds: List[Dict],
    min_recall: float = 0.70,
    max_alerts_per_1000: float = 500.0,
    threshold_strategy: str = "net_benefit",
    cost_ratio: float = 0.3,
    dense_window: int = 10,
) -> Optional[Dict]:
    """
    Apply the ScreeningPolicy to all loaded rounds.

    Parameters
    ----------
    rounds : list of round dicts from load_all_rounds()
    min_recall : hard recall constraint (patient safety)
    max_alerts_per_1000 : operational capacity constraint
    threshold_strategy : how to pick the best threshold per round
    cost_ratio : cost ratio for net_benefit strategy (FP_cost / FN_cost)
    dense_window : number of rounds with consecutive evaluations at the end
                   (used for convergence/overtraining detection)

    Returns a dict with: round, auc, net_benefit, selected_threshold, metrics,
    all_threshold_entries, convergence_info, overtraining_info, checkpoint, 
    summary, best_thresholds_per_round
    """
    if not rounds:
        return None

    policy = ScreeningPolicy(
        min_recall=min_recall,
        max_alerts_per_1000=max_alerts_per_1000,
        threshold_strategy=threshold_strategy,
        cost_ratio=cost_ratio,
        convergence_window=dense_window,
    )

    for r in sorted(rounds, key=lambda x: x["round"]):
        policy.add_round(r["round"], r["metrics"])

    best = policy.best()
    if not best:
        print("  WARNING: No round passed screening criteria!")
        return None

    best_data = next(r for r in rounds if r["round"] == best["round"])
    return {**best, "checkpoint": best_data["checkpoint"], "summary": policy.get_summary()}


def save_best_model(best: Dict, output_dir: Path, run_tag: str = "1") -> None:
    """Copy the best checkpoint and write the full result JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    src  = best["checkpoint"]
    dst  = output_dir / f"model_round_{best['round']}.pt"

    if not src.exists():
        raise FileNotFoundError(
            "Checkpoint file not found for selected round. "
            f"Resolved path: {src}. "
            "Please verify round JSON files and checkpoint locations."
        )

    shutil.copy2(src, dst)

    m = best.get("metrics") or {}
    result = {
        "round":              best["round"],
        "auc":                best["auc"],
        "auprc":              best.get("auprc"),
        "mcc":                best.get("mcc"),
        "net_benefit":        best.get("net_benefit"),
        "selected_threshold": best["selected_threshold"],
        "metrics":            m,
        "model_checkpoint":   str(dst),
        "composite_score":    best.get("composite_score"),
        "convergence_info":   best.get("convergence_info"),
        "overtraining_info":  best.get("overtraining_info"),
        "best_thresholds_per_round": best.get("best_thresholds_per_round"),
        "screening_summary":  best.get("summary"),
    }

    json_path = output_dir / f"run_{run_tag}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    ot = best.get("overtraining_info", {})
    ot_flag = "YES - see overtraining_info" if ot.get("overtraining") else "No"

    auprc = best.get("auprc", 0.0) or 0.0
    mcc   = best.get("mcc",   0.0) or 0.0
    nb    = best.get("net_benefit", 0.0) or 0.0

    print(f"\n  Best model saved:")
    print(f"    Round:               {best['round']}")
    print(f"    Threshold:           {best['selected_threshold']}")
    print(f"    ROC-AUC (recomp):    {best['auc']:.4f}  (reference — TN-inflated)")
    print(f"    AUPRC:               {auprc:.4f}  (primary)")
    print(f"    MCC:                 {mcc:.4f}  (primary)")
    print(f"    Net Benefit:         {nb:.4f}")
    print(f"    Recall:              {m.get('recall', 0):.4f}")
    print(f"    Specificity:         {m.get('spec', 0):.4f}")
    print(f"    PPV:                 {m.get('ppv', 0):.4f}")
    print(f"    NPV:                 {m.get('npv', 0):.4f}")
    print(f"    Alerts/1000:         {m.get('alerts_per_1000', 0):.1f}")
    print(f"    Overtraining?        {ot_flag}")
    if ot.get("overtraining"):
        print(f"    >> {ot.get('reason', '')}")
    print(f"    Checkpoint:          {dst}")
    print(f"    Metrics JSON:        {json_path}")
    
    # Print summary of best thresholds per round
    best_thresholds = best.get("best_thresholds_per_round", [])
    if best_thresholds:
        print(f"\n  Best Thresholds per Round ({len(best_thresholds)} rounds):")
        print(f"  {'Round':<8} {'Threshold':<12} {'Recall':<10} {'Spec':<10} {'PPV':<10} {'NPV':<10} {'Alerts/1k':<12} {'NB':<10}")
        print(f"  {'-'*82}")
        for entry in best_thresholds:
            print(
                f"  {entry['round']:<8} {entry['threshold']:<12.2f} "
                f"{entry['recall']:<10.4f} {entry['spec']:<10.4f} {entry['ppv']:<10.4f} "
                f"{entry['npv']:<10.4f} {entry['alerts_per_1000']:<12.1f} {entry['net_benefit']:<10.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Select best FL training round using ScreeningPolicy with Net Benefit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rounds-dir",  type=Path, required=True,
                        help="Directory containing round_*_run_*.json files")
    parser.add_argument("--output-dir",  type=Path, required=True,
                        help="Output directory for best model and result JSON")
    parser.add_argument("--min-recall",  type=float, default=0.70,
                        help="Hard recall constraint (patient safety)")
    parser.add_argument("--max-alerts",  type=float, default=500.0,
                        help="Soft max alerts per 1000 patients (operational capacity)")
    parser.add_argument("--strategy",    type=str,   default="net_benefit",
                        choices=["youden", "recall_constrained",
                                 "balanced_accuracy", "f1", "npv_spec", "net_benefit"],
                        help="Threshold selection strategy per round")
    parser.add_argument("--cost-ratio",  type=float, default=0.3,
                        help="Cost ratio for net_benefit strategy (FP_cost / FN_cost). "
                             "For diabetes screening: 0.3-0.4")
    parser.add_argument("--dense-window", type=int,  default=10,
                        help="Number of consecutively evaluated rounds at the end "
                             "(used for convergence and overtraining detection)")
    parser.add_argument("--run-tag",     type=str,   default="1",
                        help="Tag appended to the output JSON filename")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("  POST-TRAINING SCREENING POLICY (NET BENEFIT)")
    print(f"{'='*70}")
    print(f"  Rounds dir:      {args.rounds_dir}")
    print(f"  Min recall:      {args.min_recall:.2f}")
    print(f"  Max alerts/1000: {args.max_alerts:.0f}")
    print(f"  Strategy:        {args.strategy}")
    if args.strategy == "net_benefit":
        print(f"  Cost ratio:      {args.cost_ratio:.2f}  (FP cost / FN cost)")
    print(f"  Dense window:    {args.dense_window} rounds")
    print(f"{'='*70}\n")

    rounds = load_all_rounds(args.rounds_dir)
    if not rounds:
        print("  ERROR: No round files found in", args.rounds_dir)
        return
    print(f"  Loaded {len(rounds)} round files "
          f"(rounds {min(r['round'] for r in rounds)}-{max(r['round'] for r in rounds)})\n")

    best = apply_screening(
        rounds,
        min_recall=args.min_recall,
        max_alerts_per_1000=args.max_alerts,
        threshold_strategy=args.strategy,
        cost_ratio=args.cost_ratio,
        dense_window=args.dense_window,
    )

    if not best:
        print("  ERROR: No round passed screening criteria.")
        return

    save_best_model(best, args.output_dir, args.run_tag)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()