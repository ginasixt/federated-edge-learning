"""
ScreeningPolicy for imbalanced Federated Learning.

Design principles:
- Works with the actual data format: metrics = {auc: ..., all_thresholds: [...]}
- Selects an optimal threshold per round (configurable strategy)
- Computes AUC from the threshold curve (trapezoidal rule) since aggregated AUC is unreliable
- Compares rounds on their best achievable operating point (not an arbitrary single threshold)
- Hard constraints: min_recall (patient safety), max_alerts_per_1000 (operational capacity)
- Convergence detection: signals when FL rounds have plateaued
- Pareto-filtering on (AUC, Youden's J) — threshold-independent quality signal
- No silent state mutation: best() is pure / read-only
"""

import json
from typing import List, Dict, Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b else default


def _compute_auc_from_curve(threshold_entries: List[Dict]) -> float:
    """
    Compute ROC-AUC using the trapezoidal rule from aggregated threshold results.
    Entries must contain 'fpr' and 'tpr' (= recall).

    The server's aggregated AUC field is unreliable (often 0.0 due to per-client
    single-class predictions), so we recompute it from the global counts.
    """
    if not threshold_entries:
        return 0.0

    # Sort by FPR ascending (lower threshold → more positives → higher FPR)
    pts = sorted(threshold_entries, key=lambda e: float(e.get("fpr", 0.0)))

    fprs = [float(e.get("fpr", 0.0)) for e in pts]
    tprs = [float(e.get("tpr", e.get("recall", 0.0))) for e in pts]

    # Add anchor points (0,0) and (1,1)
    fprs = [0.0] + fprs + [1.0]
    tprs = [0.0] + tprs + [1.0]

    auc = float(np.trapz(tprs, fprs))
    return max(0.0, min(1.0, auc))


def _compute_auprc_from_curve(threshold_entries: List[Dict]) -> float:
    """
    Compute Precision-Recall AUC (AUPRC) using the trapezoidal rule.

    AUPRC is the correct curve metric for imbalanced data (~14% prevalence here):
    - ROC-AUC is dominated by the large number of true negatives and stays
      misleadingly high even for poor models on imbalanced sets.
    - AUPRC only measures precision vs recall — both focused on the positive class —
      so it reflects actual screening quality without TN inflation.

    Entries are sorted by recall ascending (higher threshold → fewer predicted positives
    → lower recall). Anchored at (recall=0, prec=1.0). 
    """
    if not threshold_entries:
        return 0.0

    pts = sorted(threshold_entries,
                 key=lambda e: float(e.get("recall", e.get("tpr", 0.0))))
    recalls    = [float(e.get("recall", e.get("tpr", 0.0)))       for e in pts]
    precisions = [float(e.get("ppv",    e.get("precision", 0.0))) for e in pts]

    # Anchor: at infinite threshold recall=0, precision=1
    recalls    = [0.0] + recalls
    precisions = [1.0] + precisions

    auprc = float(np.trapz(precisions, recalls))
    return max(0.0, min(1.0, auprc))


def _compute_mcc_from_entry(e: Dict) -> float:
    """
    Matthews Correlation Coefficient for a single threshold entry.

    MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    MCC is the preferred single-number metric for imbalanced classification:
    - Ranges from −1 (perfect inverse) through 0 (random) to +1 (perfect).
    - Unlike F1, Youden's J, or accuracy, MCC uses all four confusion-matrix
      quadrants symmetrically and is not inflated by class imbalance.
    - For 14% prevalence (6:1 ratio) it gives much more informative signal
      than Youden's J (= TPR + TNR − 1) which can look good even when the
      model produces many false positives on the majority class.
    """
    tp = float(e.get("tp", 0)); fp = float(e.get("fp", 0))
    tn = float(e.get("tn", 0)); fn = float(e.get("fn", 0))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def _compute_net_benefit(e: Dict, cost_ratio: float = 0.3) -> float:
    """
    Compute Net Benefit for medical decision-making.

    NB = recall - cost_ratio * (1 - specificity)

    This is used in Decision Curve Analysis (DCA) for medical screening.
    
    Parameters
    ----------
    e : dict
        Threshold entry with 'recall' and 'spec' fields
    cost_ratio : float
        Cost ratio: cost_of_false_positive / cost_of_false_negative
        For diabetes screening: ~0.3 (FN is ~3x more costly than FP)
    
    Returns
    -------
    float
        Net Benefit score (higher is better)
    """
    tp = float(e.get("tp", 0.0))
    fp = float(e.get("fp", 0.0))
    fn = float(e.get("fn", 0.0))
    tn = float(e.get("tn", 0.0))
    t = cost_ratio

    N = tp + fp + tn + fn

    return (tp / N) - (fp / N) * (t / (1 - t))


def _select_threshold(
    threshold_entries: List[Dict],
    strategy: str,
    min_recall: float,
    max_alerts_per_1000: float,
    cost_ratio: float = 0.3,
) -> Optional[Dict]:
    """
    Pick the best threshold entry from a round's all_thresholds list.

    Strategies
    ----------
    youden
        Maximise Youden's J = recall + specificity - 1.
        Best single-number summary for imbalanced data (prevalence ~14% here).

    recall_constrained
        Among entries with recall >= min_recall, maximise specificity.
        Enforces patient-safety first, then minimises false-positive burden.
        Falls back to highest Youden if no entry satisfies the recall constraint.

    balanced_accuracy
        Maximise balanced accuracy = 0.5 * (recall + specificity).

    f1
        Maximise F1-score.

    npv_spec
        Among entries with recall >= min_recall, maximise NPV * specificity.
        Use when missing a positive is catastrophic but over-alerting is very costly.

    net_benefit
        Among entries with recall >= min_recall, maximise Net Benefit.
        NB = recall - cost_ratio * (1 - specificity)
        cost_ratio is the ratio of FP cost to FN cost.
        For medical screening, this enforces patient safety while minimising
        operational burden (alerts_per_1000).
    """
    if not threshold_entries:
        return None

    # Operational capacity filter (soft — relax if nothing fits)
    feasible = [e for e in threshold_entries
                if float(e.get("alerts_per_1000", 1000.0)) <= max_alerts_per_1000]
    pool = feasible if feasible else threshold_entries

    if strategy == "youden":
        return max(pool, key=lambda e: float(e.get("youden", 0.0)))

    elif strategy == "recall_constrained":
        high_recall = [e for e in pool
                       if float(e.get("recall", e.get("tpr", 0.0))) >= min_recall]
        if high_recall:
            return max(high_recall, key=lambda e: float(e.get("spec", 0.0)))
        # Fallback: highest Youden available
        return max(pool, key=lambda e: float(e.get("f1", 0.0)))

    elif strategy == "balanced_accuracy":
        return max(pool, key=lambda e: float(e.get("balanced_accuracy", 0.0)))

    elif strategy == "f1":
        return max(pool, key=lambda e: float(e.get("f1", 0.0)))

    elif strategy == "npv_spec":
        high_recall = [e for e in pool
                       if float(e.get("recall", e.get("tpr", 0.0))) >= min_recall]
        if high_recall:
            return max(high_recall,
                       key=lambda e: float(e.get("npv", 0.0)) * float(e.get("spec", 0.0)))
        return max(pool, key=lambda e: float(e.get("youden", 0.0)))

    elif strategy == "net_benefit":
        high_recall = [e for e in pool
                       if float(e.get("recall", e.get("tpr", 0.0))) >= min_recall]
        if high_recall:
            return max(high_recall, 
                       key=lambda e: _compute_net_benefit(e, cost_ratio))
        # Fallback: highest net benefit available
        return max(pool, key=lambda e: _compute_net_benefit(e, cost_ratio))

    else:
        raise ValueError(
            f"Unknown threshold strategy: {strategy!r}. "
            f"Choose from: youden, recall_constrained, balanced_accuracy, f1, npv_spec, net_benefit"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScreeningPolicy:
    """
    Post-training round selection for imbalanced FL medical screening.

    After training, call add_round() for every evaluated round (in any order),
    then call best() to get the recommended round + threshold.

    Parameters
    ----------
    min_recall : float
        Hard constraint: the selected threshold must achieve at least this recall.
        Medical screening default = 0.70 (catch >=70% of positive cases).
    max_alerts_per_1000 : float
        Operational constraint: the selected threshold must not exceed this number
        of alerts per 1000 patients. Default = 500.
    threshold_strategy : str
        How to pick the best threshold within each round.
        One of: "youden", "recall_constrained", "balanced_accuracy", "f1", "npv_spec", "net_benefit".
        Default = "recall_constrained" (safety-first).
    cost_ratio : float
        Cost ratio for net_benefit strategy: cost_of_FP / cost_of_FN.
        For diabetes screening: ~0.3-0.4 (FN is 2.5-3x more costly than FP).
        Only used if threshold_strategy == "net_benefit".
        Default = 0.3.
    convergence_window : int
        Number of recent rounds used for convergence/stability detection.
        Set this to the number of rounds you evaluated densely at the end
        (e.g. 10 if you ran rounds 71-80 every round). Default = 10.
    convergence_delta : float
        Maximum std of MCC over the window to consider training converged
        (plateau). Below this → converged. Default = 0.005.
    overtraining_drop : float
        Relative drop in MCC from peak to recent mean that triggers an
        overtraining warning. Default = 0.03 (3% relative drop).
    """

    def __init__(
        self,
        min_recall: float = 0.70,
        max_alerts_per_1000: float = 500.0,
        threshold_strategy: str = "recall_constrained",
        cost_ratio: float = 0.5,
        convergence_window: int = 10,
        convergence_delta: float = 0.005,
        overtraining_drop: float = 0.03,
    ):
        self.min_recall = min_recall
        self.max_alerts_per_1000 = max_alerts_per_1000
        self.threshold_strategy = threshold_strategy
        self.cost_ratio = cost_ratio
        self.convergence_window = convergence_window
        self.convergence_delta = convergence_delta
        self.overtraining_drop = overtraining_drop

        # Internal history list — each entry is an enriched round dict:
        #   {round, raw_metrics, all_threshold_entries, selected, auc, net_benefit}
        self._history: List[Dict[str, Any]] = []
        
        # Track best thresholds per round for summary
        self._best_thresholds_per_round: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_round(self, rnd: int, metrics: Dict[str, Any]) -> None:
        """
        Register one round's aggregated metrics from the server.

        `metrics` is expected to contain:
            "auc"            : float  (often 0.0 / unreliable — will be recomputed)
            "all_thresholds" : list   (one dict per threshold operating point)

        Each threshold dict must have at least:
            threshold, tp, fp, tn, fn, tpr/recall, fpr, spec, f1,
            youden, balanced_accuracy, npv, ppv, alerts_per_1000
        """
        entries = metrics.get("all_thresholds", [])

        # Recompute AUC from the global aggregated curve (server field unreliable)
        auc = _compute_auc_from_curve(entries)

        # Select the optimal threshold for this round
        selected = _select_threshold(
            entries,
            strategy=self.threshold_strategy,
            min_recall=self.min_recall,
            max_alerts_per_1000=self.max_alerts_per_1000,
            cost_ratio=self.cost_ratio,
        )

        # AUPRC — PR-curve AUC, correct metric for imbalanced data
        auprc = _compute_auprc_from_curve(entries)

        # MCC at the selected operating point
        mcc = _compute_mcc_from_entry(selected) if selected else 0.0
        
        # Net Benefit at the selected operating point
        net_benefit = _compute_net_benefit(selected, self.cost_ratio) if selected else 0.0

        self._history.append({
            "round": int(rnd),
            "raw_metrics": metrics,
            "all_threshold_entries": entries,
            "selected": selected,    # best threshold entry for this round
            "auc":   auc,
            "auprc": auprc,
            "mcc":   mcc,
            "net_benefit": net_benefit,
        })
        
        # Track best threshold per round for summary
        if selected:
            self._best_thresholds_per_round.append({
                "round": int(rnd),
                "threshold": selected.get("threshold"),
                "recall": float(selected.get("recall", selected.get("tpr", 0.0))),
                "spec": float(selected.get("spec", 0.0)),
                "ppv": float(selected.get("ppv", selected.get("precision", 0.0))),
                "npv": float(selected.get("npv", 0.0)),
                "alerts_per_1000": float(selected.get("alerts_per_1000", 0.0)),
                "net_benefit": net_benefit,
            })

    def best(self) -> Optional[Dict[str, Any]]:
        """
        Pure read-only selection: returns the best round + threshold.

        Returns a dict with keys:
            round, auc, selected_threshold, metrics (the threshold entry),
            all_threshold_entries, composite_score, convergence_info
        """
        return self._choose_best()

    def save_best(self, out_path: str) -> None:
        """Serialise the best-round result to JSON."""
        result = self.best()
        if result is None:
            return
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    def is_converged(self) -> bool:
        """Return True if the last convergence_window rounds show no meaningful improvement."""
        return self._convergence_info()["converged"]

    def get_summary(self) -> Dict[str, Any]:
        """Summary statistics across all rounds (uses recomputed AUC + selected thresholds)."""
        if not self._history:
            return {}

        aucs    = [h["auc"]               for h in self._history]
        auprcs  = [h["auprc"]             for h in self._history]
        mccs    = [h["mcc"]               for h in self._history]
        youdens = [self._youden(h)         for h in self._history]
        recalls = [self._recall(h)         for h in self._history]
        specs   = [self._spec(h)           for h in self._history]
        f1s     = [self._f1(h)             for h in self._history]
        npvs    = [self._npv(h)            for h in self._history]
        alerts  = [self._alerts(h)         for h in self._history]

        # Per-round table for inspection
        rounds_table = [
            {
                "round":       h["round"],
                "auc":         round(h["auc"],   4),
                "auprc":       round(h["auprc"], 4),
                "mcc":         round(h["mcc"],   4),
                "youden":      round(self._youden(h), 4),
                "recall":      round(self._recall(h), 4),
                "spec":        round(self._spec(h), 4),
                "f1":          round(self._f1(h), 4),
                "npv":         round(self._npv(h), 4),
                "alerts_1000": round(self._alerts(h), 1),
                "threshold":   (h.get("selected") or {}).get("threshold"),
                "composite":   round(self._composite_score(h), 4),
            }
            for h in sorted(self._history, key=lambda x: x["round"])
        ]

        return {
            "total_rounds":       len(self._history),
            "threshold_strategy": self.threshold_strategy,
            "auc":   {"mean": float(np.mean(aucs)),   "std": float(np.std(aucs)),   "max": float(max(aucs))},
            "auprc": {"mean": float(np.mean(auprcs)), "std": float(np.std(auprcs)), "max": float(max(auprcs))},
            "mcc":   {"mean": float(np.mean(mccs)),   "std": float(np.std(mccs)),   "max": float(max(mccs))},
            "youden": {"mean": float(np.mean(youdens)), "std": float(np.std(youdens)), "max": float(max(youdens))},
            "recall": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls)), "max": float(max(recalls))},
            "spec":   {"mean": float(np.mean(specs)),   "std": float(np.std(specs)),   "max": float(max(specs))},
            "f1":     {"mean": float(np.mean(f1s)),     "std": float(np.std(f1s)),     "max": float(max(f1s))},
            "npv":    {"mean": float(np.mean(npvs)),    "std": float(np.std(npvs)),    "max": float(max(npvs))},
            "alerts_per_1000": {"mean": float(np.mean(alerts)), "min": float(min(alerts))},
            "convergence":   self._convergence_info(),
            "overtraining":  self._overtraining_info(),
            "rounds_table":  rounds_table,
        }

    # ------------------------------------------------------------------
    # Accessors on enriched history entries
    # ------------------------------------------------------------------

    def _recall(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("recall", s.get("tpr", 0.0)))

    def _spec(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("spec", 0.0))

    def _f1(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("f1", 0.0))

    def _youden(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("youden", 0.0))

    def _mcc(self, h: Dict) -> float:
        """MCC at the selected operating point — cached in h['mcc']."""
        return float(h.get("mcc", _compute_mcc_from_entry(h.get("selected") or {})))

    def _auprc(self, h: Dict) -> float:
        """Precision-Recall AUC — cached in h['auprc']."""
        return float(h.get("auprc", 0.0))

    def _npv(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("npv", 0.0))

    def _alerts(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("alerts_per_1000", 1000.0))

    def _balanced_acc(self, h: Dict) -> float:
        s = h.get("selected") or {}
        return float(s.get("balanced_accuracy", 0.0))
    
    def _net_benefit(self, h: Dict) -> float:
        """Net Benefit at the selected operating point — cached in h['net_benefit']."""
        return float(h.get("net_benefit", 0.0))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _stability_score(self, rnd: int) -> float:
        """
        Stability of MCC over recent rounds ending at `rnd`.
        Uses MCC because it is the most informative single metric for
        imbalanced data (~14% prevalence here) — symmetric over all four
        confusion-matrix quadrants and not inflated by true negatives.
        Lower variance → higher stability score (max 1.0).

        Dense window note: the last convergence_window rounds are typically
        evaluated every round, while earlier rounds are sparse (every 10).
        We use round-number proximity, not count-based indexing, so that
        a round at position 75 considers rounds 65-75 as its window.
        """
        window = self.convergence_window
        recent = [h for h in self._history
                  if h["round"] >= rnd - window and h["round"] <= rnd]
        if len(recent) < 2:
            return 1.0
        mccs = [self._mcc(h) for h in recent]
        var = float(np.var(mccs))
        return 1.0 / (1.0 + 20.0 * var)

    def _composite_score(self, h: Dict) -> float:
        """
        Round quality score for imbalanced medical screening.

        Weights (sum = 1.0) — chosen for ~14% prevalence (6:1 imbalance):
            AUPRC             0.35  — PR-curve AUC; not inflated by TN; best
                                      threshold-independent metric for imbalanced data
            MCC               0.30  — uses all 4 confusion-matrix quadrants;
                                      imbalance-robust operating-point quality
            Recall            0.20  — hard safety driver (catch positives)
            NPV               0.10  — safety of negative prediction
            Stability         0.05  — MCC variance over recent rounds

        Youden's J and ROC-AUC are kept in get_summary() for reference but are
        NOT used in scoring — both are inflated by the large TN count.
        alerts_per_1000 is a hard constraint in _choose_best(), not a soft weight.
        """
        stability = self._stability_score(h["round"])
        return (
            0.35 * self._auprc(h) +
            0.30 * self._mcc(h) +
            0.20 * self._recall(h) +
            0.10 * self._npv(h) +
            0.05 * stability
        )

    def _is_pareto_optimal(self, h: Dict, candidates: List[Dict]) -> bool:
        """
        Pareto-optimality on (AUPRC, MCC).

        AUPRC — threshold-independent PR quality; not inflated by TN.
        MCC   — operating-point quality using all four quadrants.

        Both metrics are correct for imbalanced data (~14% prevalence).
        ROC-AUC and Youden's J are NOT used here because they are heavily
        inflated by the large majority-class (TN) count.

        A round h is dominated if another candidate is strictly better
        on at least one dimension and at least as good on the other.
        """
        for other in candidates:
            if other is h:
                continue
            if (self._auprc(other) >= self._auprc(h) and
                self._mcc(other)   >= self._mcc(h) and
                (self._auprc(other) > self._auprc(h) or
                 self._mcc(other)   > self._mcc(h))):
                return False
        return True

    # ------------------------------------------------------------------
    # Convergence & overtraining
    # ------------------------------------------------------------------

    def _convergence_info(self) -> Dict[str, Any]:
        """
        Detect whether FL training has plateaued.

        Uses the last convergence_window rounds (by count) — these correspond
        to the dense evaluation phase (e.g. rounds 71-80 if window=10).
        Criterion: std of MCC is below convergence_delta.

        MCC is used instead of Youden's J because it reflects all four
        confusion-matrix cells and is not inflated by the majority class.
        """
        if len(self._history) < self.convergence_window:
            return {
                "converged":   False,
                "reason":      "not enough rounds",
                "mcc_std":     None,
                "auprc_std":   None,
                "dense_rounds": [],
            }

        # Sort by round number to make sure we take the last N chronologically
        sorted_hist = sorted(self._history, key=lambda h: h["round"])
        recent = sorted_hist[-self.convergence_window:]
        mccs   = [self._mcc(h)   for h in recent]
        auprcs = [self._auprc(h) for h in recent]

        mcc_std   = float(np.std(mccs))
        auprc_std = float(np.std(auprcs))
        converged = mcc_std < self.convergence_delta

        return {
            "converged":    converged,
            "window":       self.convergence_window,
            "dense_rounds": [h["round"] for h in recent],
            "mcc_std":      round(mcc_std, 5),
            "auprc_std":    round(auprc_std, 5),
            "delta":        self.convergence_delta,
            "reason":       "plateau detected" if converged else "still improving",
        }

    def _overtraining_info(self) -> Dict[str, Any]:
        """
        Detect if performance peaked earlier and has since declined.

        The key insight for this FL setup:
        - Rounds 1-70 are sparse (every 10 rounds) — an early trend
        - Rounds 71-80 are dense (every round) — the fine-grained signal

        Overtraining is flagged when:
          mean(MCC in recent dense window) < peak_mcc * (1 - overtraining_drop)
          AND the peak round is not among the recent dense rounds.

        MCC is used instead of Youden's J — it is imbalance-robust and
        reflects improvement / decline more accurately at ~14% prevalence.

        Also detects a clear monotone decline trend in the dense window
        using a simple linear regression slope.
        """
        if len(self._history) < 2:
            return {"overtraining": False, "reason": "not enough rounds"}

        sorted_hist = sorted(self._history, key=lambda h: h["round"])
        mccs   = [self._mcc(h) for h in sorted_hist]
        rounds = [h["round"]   for h in sorted_hist]

        # Global peak (by MCC)
        peak_idx  = int(np.argmax(mccs))
        peak_mcc  = mccs[peak_idx]
        peak_round = rounds[peak_idx]

        # Recent dense window (last convergence_window by count)
        recent      = sorted_hist[-self.convergence_window:]
        recent_rnds = [h["round"]  for h in recent]
        recent_mccs = [self._mcc(h) for h in recent]
        recent_mean = float(np.mean(recent_mccs))

        # 1) Value-based drop: peak is before dense window AND MCC dropped
        peak_in_dense = peak_round in recent_rnds
        relative_drop = (peak_mcc - recent_mean) / peak_mcc if peak_mcc > 0 else 0.0
        value_drop    = (not peak_in_dense) and (relative_drop >= self.overtraining_drop)

        # 2) Trend-based: linear slope over dense window (negative = declining)
        trend_slope = 0.0
        if len(recent) >= 3:
            x   = np.array(recent_rnds, dtype=float)
            y   = np.array(recent_mccs, dtype=float)
            x_c = x - x.mean()
            trend_slope = float(np.dot(x_c, y) / np.dot(x_c, x_c)) if np.dot(x_c, x_c) > 0 else 0.0

        # Slope threshold: MCC decline of >0.001/round is meaningful
        trend_declining = len(recent) >= 3 and trend_slope < -0.001

        overtraining = value_drop or (trend_declining and not peak_in_dense)

        if overtraining:
            reason = (
                f"Peak MCC={peak_mcc:.4f} at round {peak_round}, "
                f"recent mean MCC={recent_mean:.4f} "
                f"(drop={relative_drop*100:.1f}%, slope={trend_slope:+.5f}/round)"
            )
        elif trend_declining and peak_in_dense:
            reason = (
                f"Declining MCC trend in dense window (slope={trend_slope:+.5f}/round) "
                f"but peak is still within window — may still converge"
            )
        else:
            reason = "No overtraining detected"

        return {
            "overtraining":          overtraining,
            "peak_round":            peak_round,
            "peak_mcc":              round(peak_mcc, 4),
            "recent_mean_mcc":       round(recent_mean, 4),
            "relative_drop_pct":     round(relative_drop * 100, 2),
            "trend_slope_per_round": round(trend_slope, 6),
            "trend_declining":       trend_declining,
            "dense_window_rounds":   recent_rnds,
            "reason":                reason,
        }

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def _choose_best(self) -> Optional[Dict[str, Any]]:
        """
        Select the best round using a layered funnel:

        For net_benefit strategy:
        1. Hard recall filter  — patient safety (non-negotiable)
        2. Soft alerts filter  — operational capacity
        3. Select round with max net_benefit
        4. Tie-breaking: if NB diff < 0.01, prefer higher PPV or lower alerts

        For other strategies:
        1. Hard recall filter  — patient safety (non-negotiable)
        2. Soft alerts filter  — operational capacity
        3. Pareto filter       — remove rounds dominated in (AUPRC, MCC)
        4. Composite score     — weighted multi-metric score
        """
        if not self._history:
            return None

        # ── 1. Hard recall constraint ──────────────────────────────────
        passing = [h for h in self._history if self._recall(h) >= self.min_recall]

        if not passing:
            print(f"  WARNING: No rounds meet min_recall={self.min_recall:.2f}.")
            print(f"           Using best available recall (clinical review required).")
            sorted_by_recall = sorted(self._history, key=self._recall, reverse=True)
            passing = sorted_by_recall[: max(3, len(sorted_by_recall) // 2)]

        # ── 2. Soft alerts constraint ──────────────────────────────────
        feasible = [h for h in passing
                    if self._alerts(h) <= self.max_alerts_per_1000]

        if feasible:
            candidates = feasible
            print(f"  {len(candidates)} rounds pass recall >= {self.min_recall:.2f} "
                  f"and alerts <= {self.max_alerts_per_1000:.0f}/1000")
        else:
            candidates = passing
            best_alerts = min(self._alerts(h) for h in candidates)
            print(f"  WARNING: No round meets alerts <= {self.max_alerts_per_1000:.0f}/1000.")
            print(f"           Best available: {best_alerts:.0f}/1000 (capacity review needed).")

        # ── 3. Strategy-specific selection ──────────────────────────────
        if self.threshold_strategy == "net_benefit":
            # Find max net benefit
            best = max(candidates, key=self._net_benefit)
            best_nb = self._net_benefit(best)
            
            # Tie-breaking: if other rounds within 0.01 of best NB
            #   prefer higher PPV or fewer alerts
            tied_tolerance = 0.01
            tied_candidates = [
                h for h in candidates 
                if abs(self._net_benefit(h) - best_nb) < tied_tolerance
            ]
            
            if len(tied_candidates) > 1:
                print(f"  Tie-breaking: {len(tied_candidates)} rounds within NB tolerance {tied_tolerance}")
                # Primary: max PPV; Secondary: min alerts
                best = max(tied_candidates, 
                          key=lambda h: (
                              float((h.get("selected") or {}).get("ppv", 0.0)),
                              -float((h.get("selected") or {}).get("alerts_per_1000", 1000.0))
                          ))
        else:
            # ── 3. Pareto filter on (AUPRC, MCC) ─────────────────────────
            pareto = [h for h in candidates if self._is_pareto_optimal(h, candidates)]
            if pareto:
                candidates = pareto
                print(f"  {len(pareto)} Pareto-optimal rounds (AUPRC vs MCC)")

            # ── 4. Composite score ─────────────────────────────────────────
            best = max(candidates, key=self._composite_score)

        # ── Logging ───────────────────────────────────────────────────
        s    = best.get("selected") or {}
        conv = self._convergence_info()
        ot   = self._overtraining_info()

        print(f"\n  SELECTED ROUND {best['round']}:")
        print(f"    ROC-AUC (recomp):   {best['auc']:.4f}  (reference only — TN-inflated)")
        print(f"    AUPRC:              {self._auprc(best):.4f}  (primary metric)")
        print(f"    MCC:                {self._mcc(best):.4f}  (primary metric)")
        print(f"    Youden's J:         {self._youden(best):.4f}  (reference)")
        print(f"    Recall:             {self._recall(best):.4f}")
        print(f"    Specificity:        {self._spec(best):.4f}")
        print(f"    NPV:                {self._npv(best):.4f}")
        print(f"    F1:                 {self._f1(best):.4f}")
        print(f"    Balanced accuracy:  {self._balanced_acc(best):.4f}")
        print(f"    Alerts/1000:        {self._alerts(best):.1f}")
        if self.threshold_strategy == "net_benefit":
            print(f"    Net Benefit:        {self._net_benefit(best):.4f}  (cost_ratio={self.cost_ratio})")
        print(f"    Threshold used:     {s.get('threshold', '?')}")
        if self.threshold_strategy != "net_benefit":
            print(f"    Composite score:    {self._composite_score(best):.4f}")
        print(f"    Stability score:    {self._stability_score(best['round']):.4f}")
        print(f"    Convergence:        {conv['reason']} "
              f"(MCC std={conv.get('mcc_std', '?')} "
              f"over rounds {conv.get('dense_rounds', [])})")

        if ot["overtraining"]:
            print(f"\n  *** OVERTRAINING WARNING ***")
            print(f"    {ot['reason']}")
            print(f"    Peak round {ot['peak_round']} (MCC={ot['peak_mcc']:.4f}) "
                  f"is not the selected round — consider using round {ot['peak_round']} instead.")
        elif ot.get("trend_declining"):
            print(f"\n  NOTE: Declining MCC trend in recent rounds "
                  f"(slope={ot['trend_slope_per_round']:+.5f}/round). "
                  f"Monitor for overtraining.")
        print()

        return {
            "round":                 best["round"],
            "auc":                   best["auc"],
            "auprc":                 self._auprc(best),
            "mcc":                   self._mcc(best),
            "net_benefit":           self._net_benefit(best),
            "selected_threshold":    s.get("threshold"),
            "metrics":               s,
            "all_threshold_entries": best["all_threshold_entries"],
            "composite_score":       self._composite_score(best) if self.threshold_strategy != "net_benefit" else None,
            "convergence_info":      conv,
            "overtraining_info":     ot,
            "best_thresholds_per_round": self._best_thresholds_per_round,
        }