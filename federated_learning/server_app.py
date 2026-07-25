# federated_learning/server_app.py
from __future__ import annotations

import torch
import json
import numpy as np
from pathlib import Path

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import (
    Context, Parameters, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.common.record import ConfigRecord

try:
    from flwr.server.app import ServerAppComponents
except ImportError:
    from flwr.server import ServerAppComponents

from federated_learning.client_app import MLP, _serialize_ndarrays, _deserialize_ndarrays


class ScaffoldFedAvg(FedAvg):
    """
    FedAvg extended with SCAFFOLD control-variate aggregation.

    Extra responsibilities vs. plain FedAvg:
      - Maintains a server-side global control variate  c  (one tensor per parameter).
      - Injects  c  into every fit-config so clients can correct their gradients.
      - After each fit round aggregates the  Δc_i  reports from clients and
        updates  c  ←  c + (1/N) * Σ Δc_i   (N = total number of supernodes).
      - Caches aggregated parameters in RAM and writes checkpoints + JSON metrics
        after each evaluation round (unchanged from the original FedAvgWithScreening).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model_dim: int,
        run_config: dict,
        num_total_clients: int,       # N in the SCAFFOLD paper
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.model_dim      = model_dim
        self.run_config     = run_config
        self.num_total_clients = max(1, num_total_clients)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Template model – used for state-dict key mapping & parameter shapes
        self.template_model = MLP(in_dim=model_dim)

        # Global control variate c – one zero-vector per parameter tensor.
        # Initialised lazily on the first aggregate_fit call so we know shapes.
        self.c_global: list[np.ndarray] | None = None

        # RAM cache: {server_round: Parameters}
        self.parameters_cache: dict[int, Parameters] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_c_global(self) -> list[np.ndarray]:
        """Return a list of zero arrays matching the model's parameter shapes."""
        return [
            np.zeros_like(v.detach().cpu().numpy())
            for v in self.template_model.parameters()
        ]

    # ------------------------------------------------------------------
    # aggregate_fit  – FedAvg aggregation  +  SCAFFOLD c update
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[any, FitRes]],
        failures: list[tuple[any, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, any]]:
        """Aggregiere Gewichte und speichere im RAM."""

        # 1) Standard FedAvg weight aggregation 
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is None:
            return None, {}

        # ── 1b) Apply global step size η_g = 1 (SCAFFOLD paper default) ───
        # FedAvg already computes weighted average of client updates.
        # η_g = 1 means we use the aggregated parameters as-is (no server damping).
        # If you ever want η_g < 1, scale the delta from the previous round here.
        # (Currently a no-op since η_g = 1, but explicit for clarity)
        # global_lr = 1.0  → no scaling needed

        # # ── 2) Cache aggregated parameters for checkpoint saving ───────────
        # self.parameters_cache[server_round] = aggregated_parameters
        # if len(self.parameters_cache) > 3:
        #     del self.parameters_cache[min(self.parameters_cache)]

        run_tag = str(self.run_config.get("run-tag", "1"))
        checkpoint_path = self.checkpoint_dir / f"all_rounds_run_{run_tag}" / f"model_round_{server_round}_run_{run_tag}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(server_round, aggregated_parameters, checkpoint_path)
        self._save_checkpoint(server_round, aggregated_parameters, checkpoint_path)

        # ── 3) Lazy-init global control variate ───────────────────────────
        if self.c_global is None:
            self.c_global = self._init_c_global()

        # ── 4) Aggregate Δc_i from clients and update c ───────────────────
        #   c ← c + (1/N) * Σ Δc_i
        #   N = total number of supernodes (not just sampled ones!)
        delta_sum: list[np.ndarray] | None = None
        num_contributors = 0

        for _client_proxy, fit_res in results:
            raw = fit_res.metrics.get("scaffold_delta_ci", None)
            if raw is None:
                continue
            try:
                delta_ci = _deserialize_ndarrays(raw)
            except Exception as e:
                print(f"⚠️  Could not deserialize scaffold_delta_ci: {e}")
                continue

            if delta_sum is None:
                delta_sum = [np.zeros_like(d) for d in delta_ci]

            for idx, d in enumerate(delta_ci):
                delta_sum[idx] = delta_sum[idx] + d

            num_contributors += 1

        if delta_sum is not None and num_contributors > 0:
            scale = 1.0 / self.num_total_clients   # divide by N, not by sampled count
            for idx in range(len(self.c_global)):
                self.c_global[idx] = self.c_global[idx] + scale * delta_sum[idx]

            print(
                f"🔧 Round {server_round}: Updated c_global from "
                f"{num_contributors} client Δc_i reports (N={self.num_total_clients})"
            )
        else:
            print(f"⚠️  Round {server_round}: No scaffold_delta_ci received – c_global unchanged")

        return aggregated_parameters, aggregated_metrics

    # ------------------------------------------------------------------
    # aggregate_evaluate  – unchanged checkpoint / JSON logic
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[any, any]],
        failures: list[tuple[any, any] | BaseException],
    ) -> tuple[float | None, dict[str, any]]:
        """Save checkpoint + JSON metrics for every evaluated round."""

        if not results:
            return None, {}

        valid_results = [
            (cp, res) for cp, res in results if res.num_examples > 0
        ]
        if not valid_results:
            print(f"⏭️  Round {server_round}: Evaluation skipped (all n_val=0)")
            return None, {}

        loss, metrics = super().aggregate_evaluate(server_round, valid_results, failures)
        if not metrics:
            return loss, {}

        run_tag   = str(self.run_config.get("run-tag", "1"))
        json_path = self.checkpoint_dir / f"all_rounds_run_{run_tag}" / f"round_{server_round}_run_{run_tag}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(
                {"round": server_round, "metrics": metrics, "loss": loss},
                f, indent=2,
            )
        print(f"💾 Round {server_round}: Saved {json_path.name}")

        return loss, metrics

    # ------------------------------------------------------------------
    # _save_checkpoint  – unchanged
    # ------------------------------------------------------------------

    def _save_checkpoint(self, server_round: int, parameters: Parameters, checkpoint_path: Path):
        ndarrays        = parameters_to_ndarrays(parameters)
        state_dict_keys = list(self.template_model.state_dict().keys())

        if len(ndarrays) != len(state_dict_keys):
            print(f"⚠️  Parameter count mismatch! "
                  f"Expected {len(state_dict_keys)}, got {len(ndarrays)}")
            return

        state_dict = {
            k: torch.tensor(arr, dtype=torch.float32)
            for k, arr in zip(state_dict_keys, ndarrays)
        }
        torch.save(state_dict, checkpoint_path)
        print(f"   💾 Checkpoint saved: {checkpoint_path}")


# -----------------------------------------------------------------------
# server_fn
# -----------------------------------------------------------------------

def server_fn(context: Context) -> ServerAppComponents:
    """Build ScaffoldFedAvg strategy + server config."""
    rc = dict(context.run_config)

    checkpoint_dir     = Path(f"result/{rc.get('split-path', 'default')}/all_rounds/")
    model_dim          = 21
    total_rounds       = int(rc.get("num-server-rounds", 80))
    num_total_clients  = int(rc.get("min-available-clients", 32768))

    # Holds a reference to the strategy so on_fit_config_fn can read c_global.
    # We use a one-element list as a mutable closure cell.
    strategy_ref: list[ScaffoldFedAvg] = []

    def on_fit_config_fn(rnd: int) -> dict:
        """Send training hyper-params + current global control variate c to clients."""
        lr = float(rc.get("lr", 1e-2))

        cfg: dict = {
            "epochs":         int(rc.get("local-epochs", 2)),
            "lr":             lr,
            "weight-decay":   float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
        }

        # Inject current c_global so clients can apply gradient correction
        if strategy_ref and strategy_ref[0].c_global is not None:
            cfg["scaffold_c"] = _serialize_ndarrays(strategy_ref[0].c_global)
        # Round 1: c_global is still None → clients fall back to zeros automatically

        return cfg

    def on_evaluate_config_fn(rnd: int) -> dict:
        # First and last round always evaluate
        if rnd == 1 or rnd == total_rounds:
            pass
        elif rnd <= 70:
            if rnd % 10 != 0:
                return {}  # Skip evaluation this round

        threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        return {"threshold_grid": json.dumps(threshold_grid)}

    # ── metric helpers ─────────────────────────────────────────────────

    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
        tpr     = _safe_div(tp, tp + fn)
        fpr     = _safe_div(fp, fp + tn)
        spec    = 1.0 - fpr
        ppv     = _safe_div(tp, tp + fp)
        npv     = _safe_div(tn, tn + fn)
        f1      = _safe_div(2 * ppv * tpr, ppv + tpr)
        bal_acc = 0.5 * (tpr + spec)
        youden  = tpr + spec - 1.0
        prev    = _safe_div(tp + fn, tp + fp + tn + fn)
        alerts_per_1000 = _safe_div(tp + fp, tp + fp + tn + fn) * 1000.0
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "recall": tpr,
            "fpr": fpr, "spec": spec,
            "ppv": ppv, "precision": ppv, "npv": npv,
            "f1": f1, "balanced_accuracy": bal_acc, "youden": youden,
            "prevalence": prev, "alerts_per_1000": alerts_per_1000,
        }

    def evaluate_metrics_aggregation_fn(eval_metrics: list[tuple[int, dict]]) -> dict:
        """
        Aggregate evaluation metrics from clients using multi-threshold optimization.
        
        """
        # Konvertiere ConfigRecord
        processed_metrics = []
        for n, md in eval_metrics:  # n = number of samples (val) and md = metrics dict
            if isinstance(md, ConfigRecord):
                md_dict = dict(md.items())
            elif isinstance(md, dict):
                md_dict = md
            else:
                print(f"Unknown metrics type: {type(md)}")
                continue
            
            # FILTER: Ignoriere Clients mit 0 Validation Samples
            if int(n) == 0:
                print(f"⚠️  Skipping client with n_val=0")
                continue
            
            # FILTER: Ignoriere Clients mit leeren Metriken (aus deinem Client-Fallback)
            n_samples = md_dict.get("n_samples", 0)
            if int(n_samples) == 0:
                print(f"⚠️  Skipping client with n_samples=0 in metrics")
                continue
            
            processed_metrics.append((n, md_dict))
        
        # 1) AUC aggregieren
        auc_weighted_sum = 0.0
        total_weight_for_auc = 0
        
        for n, md in processed_metrics:
            auc = md.get("auc", None)
            if auc is not None and auc > 0:  # ✅ Ignoriere auc=0.0
                w = int(n) if n else 1
                auc_weighted_sum += float(auc) * w
                total_weight_for_auc += w
        
        aggregated_auc = auc_weighted_sum / total_weight_for_auc if total_weight_for_auc else 0.0
        
        # 2) Deserialize und Aggregiere Threshold-Counts
        threshold_aggregated = {}
        
        for n, md in processed_metrics:
            #  Parse JSON-Strings zu Listen
            try:
                thresholds = json.loads(md.get("thresholds_json", "[]"))
                tp_list = json.loads(md.get("tp_json", "[]"))
                fp_list = json.loads(md.get("fp_json", "[]"))
                tn_list = json.loads(md.get("tn_json", "[]"))
                fn_list = json.loads(md.get("fn_json", "[]"))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error: {e}")
                continue
            
            # Aggregiere Counts
            for i, thr in enumerate(thresholds):
                if thr not in threshold_aggregated:
                    threshold_aggregated[thr] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
                
                threshold_aggregated[thr]["tp"] += int(tp_list[i]) if i < len(tp_list) else 0
                threshold_aggregated[thr]["fp"] += int(fp_list[i]) if i < len(fp_list) else 0
                threshold_aggregated[thr]["tn"] += int(tn_list[i]) if i < len(tn_list) else 0
                threshold_aggregated[thr]["fn"] += int(fn_list[i]) if i < len(fn_list) else 0
        
        # NAch Aggregation:
        # threshold_aggregated = {
        #   0.30: {tp: 3192, fp: 9554, tn: 12279, fn: 343},  # SUMME aller Clients!
        #   0.35: {tp: 3083, fp: 8489, tn: 13344, fn: 452},
        #   ...
        # }
        
        # 3) Berechne Metriken 
        threshold_results = []
        
        # Berechne Metriken für jeden Threshold
        for thr in sorted(threshold_aggregated.keys()):
            counts = threshold_aggregated[thr]
            metrics = _metrics_from_counts(
                counts["tp"], counts["fp"], counts["tn"], counts["fn"]
            )
            metrics["threshold"] = thr
            threshold_results.append(metrics)
        
        # 4) Minimales Logging (für Live-Monitoring während Training)
        print(f"\n📊 Multi-Threshold Aggregation:")
        print(f"   Valid clients: {len(processed_metrics)}")
        print(f"   Evaluated {len(threshold_results)} thresholds:")
        
        for result in threshold_results[:3]:
            print(f"     Thr={result['threshold']:.2f}: "
                  f"Rec={result['recall']:.3f}, Spec={result['spec']:.3f}, F1={result['f1']:.3f}")
        
        if len(threshold_results) > 3:
            print(f"     ... (+ {len(threshold_results)-3} more)\n")
        
        # 5) Gib ALLE Threshold-Ergebnisse zurück (Post-Training Screening wählt beste)
        return {
            "auc": aggregated_auc,
            "all_thresholds": threshold_results  # Alle Thresholds für Screening-Tool
        }
    
    def fit_metrics_aggregation_fn(metrics):
        """Weighted average of scalar fit metrics (excludes scaffold_delta_ci)."""
        n_sum = sum(n for n, _ in metrics) or 1
        # Only aggregate numeric scalars; skip the serialized control-variate string
        scalar_keys = set()
        for _, m in metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)) and k != "scaffold_delta_ci":
                    scalar_keys.add(k)
        return {
            k: sum(n * float(m.get(k, 0.0)) for n, m in metrics) / n_sum
            for k in scalar_keys
        }

    # ── build strategy ─────────────────────────────────────────────────

    strategy = ScaffoldFedAvg(
        checkpoint_dir=checkpoint_dir,
        model_dim=model_dim,
        run_config=rc,
        num_total_clients=num_total_clients,
        # FedAvg parameters
        fraction_fit=float(rc.get("fraction-fit", 0.75)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 6)),
        min_available_clients=int(rc.get("min-available-clients", 8)),
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 8)),
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    # Wire strategy into the closure so on_fit_config_fn can read c_global
    strategy_ref.append(strategy)

    cfg = ServerConfig(num_rounds=total_rounds)
    return ServerAppComponents(config=cfg, strategy=strategy)


app = ServerApp(server_fn=server_fn)


