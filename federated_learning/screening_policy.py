import json
from typing import List, Dict, Any, Optional, Union
import numpy as np

class ScreeningPolicy:
    """
    After every round, the screening policy balances:
    1) High recall (don't miss diabetes cases)
    2) Acceptable specificity (avoid overwhelming follow-up capacity)
    3) Stability across rounds (robust model)
    4) Composite score for final selection

    The policy tracks round metrics and selects the best round
    """

    def __init__(self, min_recall: float = 0.70, min_spec: float = 0.70):
        self.history: List[Dict[str, Any]] = [] # List of the Metrics of all rounds {"round": int, "metrics": dict}
        self.min_recall = min_recall  # Hard constraint: must catch 70%+ of cases
        self.min_spec = min_spec      # Soft constraint: prefer 70%+ specificity

    def add_round(self, rnd: int, metrics: Dict[str, Any]) -> None:
        """
        Add round metrics to history.
        """
        # Filter: Konvertiere nur numeric values zu float, behalte strings
        processed_metrics = {}
        for k, v in metrics.items():
            if k == "all_thresholds":
                # Behalte Liste von Threshold-Results
                processed_metrics[k] = v
            elif isinstance(v, (int, float)):
                processed_metrics[k] = float(v)
            elif isinstance(v, str):
                processed_metrics[k] = v  # Strings beibehalten!
            else:
                # Fallback: Versuche float-Konvertierung, sonst string
                try:
                    processed_metrics[k] = float(v)
                except (ValueError, TypeError):
                    processed_metrics[k] = str(v)
        
        entry = {"round": int(rnd), "metrics": processed_metrics}
        self.history.append(entry)

    def _spec(self, m): 
        return float(m.get("spec", 0.0))
    
    def _recall(self, m): 
        return float(m.get("recall", m.get("tpr", 0.0)))
    
    def _f1(self, m):
        return float(m.get("f1", 0.0))

    def _alerts_per_1000(self, m):
        """Lower is better (less resource strain)"""
        return float(m.get("alerts_per_1000", 1000.0))

    def _stability_score(self, rnd: int, window: int = 3) -> float:
        """
        Check if metrics are stable in recent rounds.
        Higher score = more stable (less variance).
        Its calculated by looking at the variance of recall and specificity over the last `window` rounds.
        """
        recent = [h for h in self.history 
                  if h["round"] >= rnd - window and h["round"] <= rnd]
        if len(recent) < 2:
            return 1.0  # Not enough data = assume stable
        
        recalls = [self._recall(h["metrics"]) for h in recent]
        specs = [self._spec(h["metrics"]) for h in recent]
        
        # Lower variance = higher stability
        recall_var = np.var(recalls) if len(recalls) > 1 else 0.0
        spec_var = np.var(specs) if len(specs) > 1 else 0.0
        
        # Normalize: low variance = high score (max ~1.0)
        stability = 1.0 / (1.0 + 10 * (recall_var + spec_var))
        return stability

    def _screening_score(self, h: Dict[str, Any]) -> float:
        """
        Composite score for medical screening:
        - Recall: 40% weight (most important: catch positives)
        - Specificity: 30% weight (control false positives)
        - F1: 20% weight (overall balance)
        - Stability: 10% weight (robustness)
        """
        m = h["metrics"]
        # 
        stability = self._stability_score(h["round"])
        
        score = (
            0.40 * self._recall(m) +      # High recall is critical
            0.30 * self._spec(m) +        # But we need acceptable spec too
            0.20 * self._f1(m) +          # Balance measure
            0.10 * stability              # Prefer stable models
        )
        return score

    def _is_pareto_optimal(self, h: Dict[str, Any], candidates: List[Dict[str, Any]]) -> bool:
        """
        Check if h is Pareto-optimal:
        No other candidate is better in BOTH recall AND spec.
        """
        for other in candidates:
            if (self._recall(other["metrics"]) >= self._recall(h["metrics"]) and 
                self._spec(other["metrics"]) >= self._spec(h["metrics"]) and
                (self._recall(other["metrics"]) > self._recall(h["metrics"]) or 
                 self._spec(other["metrics"]) > self._spec(h["metrics"]))):
                return False  # h is dominated by other
        return True

    def _choose_best(self) -> Optional[Dict[str, Any]]:

        if not self.history:
            return None

        # 1) Hard filter by minimum recall (safety requirement)
        candidates = [h for h in self.history 
                     if self._recall(h["metrics"]) >= self.min_recall]
        
        if not candidates:
            print(f"WARNING: No rounds meet min recall {self.min_recall}")
            print(f"           Using best available recall instead")
            candidates = self.history

        # 2) Prefer candidates with good specificity
        good_spec = [h for h in candidates 
                    if self._spec(h["metrics"]) >= self.min_spec]
        
        # TODO(ginasixt): Finde else hier noch keimn gutes Fallbackverhalten
        if good_spec:
            candidates = good_spec
            print(f"Found {len(candidates)} rounds with recall≥{self.min_recall} AND spec≥{self.min_spec}")
        else:
            print(f"⚠️  No rounds meet spec≥{self.min_spec}, using recall filter only")

        # 3) Filter to Pareto-optimal solutions
        # It checks if any candidate is dominated in both recall & spec, if none dominates the other both are Pareto-optimal and are being kept
        # Round 1:  Recall=0.757, Spec=0.725
        # Round 10: Recall=0.790, Spec=0.706 -> Round 10 hat höheren Recall, aber niedrigeren Spec
        # --> beide sind Pareto-optimal! (keiner dominiert den anderen)
        pareto = [h for h in candidates if self._is_pareto_optimal(h, candidates)]
        print(f" {len(pareto)} Pareto-optimal rounds (not dominated in recall & spec)")

        # 4) Among Pareto-optimal, choose highest composite score
        best = max(pareto, key=lambda h: self._screening_score(h))
        
        # Log the decision
        m = best["metrics"]
        print(f"\n🎯 SELECTED ROUND {best['round']}:")
        print(f"   Recall:     {self._recall(m):.3f}")
        print(f"   Specificity: {self._spec(m):.3f}")
        print(f"   F1-Score:    {self._f1(m):.3f}")
        print(f"   Stability:   {self._stability_score(best['round']):.3f}")
        print(f"   Alerts/1000: {self._alerts_per_1000(m):.1f}")
        print(f"   Composite:   {self._screening_score(best):.3f}\n")
        
        return best

    def best(self) -> Optional[Dict[str, Any]]:
        return self._choose_best()

    def save_best(self, out_path: str) -> None:
        best = self.best()
        if best is None:
            return
        with open(out_path, "w") as f:
            json.dump(best, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all rounds"""
        if not self.history:
            return {}
        
        recalls = [self._recall(h["metrics"]) for h in self.history]
        specs = [self._spec(h["metrics"]) for h in self.history]
        f1s = [self._f1(h["metrics"]) for h in self.history]
        
        return {
            "total_rounds": len(self.history),
            "recall": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls)), "max": float(max(recalls))},
            "spec": {"mean": float(np.mean(specs)), "std": float(np.std(specs)), "max": float(max(specs))},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "max": float(max(f1s))},
        }