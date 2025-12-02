# federated_learning/server_app.py
from __future__ import annotations

import torch
from pathlib import Path
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.record import ConfigRecord
from federated_learning.screening_policy import ScreeningPolicy
from federated_learning.client_app import MLP
import json
import numpy as np

try:
    from flwr.server.app import ServerAppComponents
except ImportError:
    from flwr.server import ServerAppComponents


class FedAvgWithScreening(FedAvg):
    """
    Custom FedAvg Strategy that:
    1. Caches parameters in RAM after aggregation
    2. Tracks metrics with ScreeningPolicy
    3. Saves checkpoints for best rounds
    """
    
    def __init__(
        self,
        screening_policy: ScreeningPolicy,
        checkpoint_dir: Path,
        model_dim: int,  # Feature-Dimension für MLP
        run_config: dict,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.screening = screening_policy
        self.checkpoint_dir = checkpoint_dir
        self.model_dim = model_dim
        self.run_config = run_config
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Template für State Dict Keys
        self.template_model = MLP(in_dim=model_dim)
        
        # RAM-Cache: {server_round: Parameters}
        self.parameters_cache: dict[int, Parameters] = {}
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[any, FitRes]],
        failures: list[tuple[any, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, any]]:
        """Aggregiere Gewichte und speichere im RAM."""
        
        # 1) Standard FedAvg Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is None:
            return None, {}
        
        # 2) Speichere im RAM
        self.parameters_cache[server_round] = aggregated_parameters
        
        # 3) Cleanup: Behalte nur die letzten 3 Runden im RAM
        if len(self.parameters_cache) > 3:
            oldest = min(self.parameters_cache.keys())
            del self.parameters_cache[oldest]
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[any, any]],
        failures: list[tuple[any, any] | BaseException],
    ) -> tuple[float | None, dict[str, any]]:
        """
        Nach Evaluation: Prüfe ob beste Runde und speichere nach Eval beste Runde Checkpoint.
        """
        
        # 1) Standard Aggregation (ruft evaluate_metrics_aggregation_fn auf)
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if metrics is None:
            return loss, {}
        
        # 2) Füge Metriken zu Screening hinzu
        self.screening.add_round(server_round, metrics)
        
        # 3) Prüfe ob diese Runde die beste ist
        best = self.screening.best()
        
        if best and best["round"] == server_round:
            print(f"\n✅ Round {server_round} is NEW BEST - Saving checkpoint...")
            
            # Hole Parameters aus RAM-Cache
            if server_round not in self.parameters_cache:
                print(f"⚠️  Warning: Parameters for round {server_round} not in cache!")
                return loss, metrics
            
            parameters = self.parameters_cache[server_round]
            
            # Speichere Checkpoint
            checkpoint_path = self.checkpoint_dir / f"model_round_{server_round}.pt"
            self._save_checkpoint(server_round, parameters, checkpoint_path)
            
            # Update JSON mit Checkpoint-Path
            best["model_checkpoint"] = str(checkpoint_path)
            
            run_tag = str(self.run_config.get("run-tag", "none"))
            json_path = self.checkpoint_dir / f"run_{run_tag}.json"
            self.screening.save_best(str(json_path))
            
            print(f"   💾 Best model saved: {checkpoint_path}")
            print(f"   📊 Metrics saved: {json_path}")
        
        return loss, metrics
    
    def _save_checkpoint(
        self, 
        server_round: int, 
        parameters: Parameters,
        checkpoint_path: Path
    ):
        """Speichert Flower Parameters als PyTorch State Dict.
        
        Args:
            server_round: Aktuelle Runde
            parameters: Flower Parameters-Objekt
            checkpoint_path: Wo speichern
        """
        # 1) Konvertiere Flower Parameters zu NumPy Arrays
        ndarrays = parameters_to_ndarrays(parameters)
        
        # 2) Mappe zu PyTorch State Dict
        state_dict_keys = list(self.template_model.state_dict().keys())
        
        if len(ndarrays) != len(state_dict_keys):
            print(f"⚠️  Parameter count mismatch! "
                  f"Expected {len(state_dict_keys)}, got {len(ndarrays)}")
            return
        
        state_dict = {
            key: torch.tensor(arr, dtype=torch.float32)
            for key, arr in zip(state_dict_keys, ndarrays)
        }
        
        # Speichere auf Disk
        torch.save(state_dict, checkpoint_path)
        print(f"   💾 Checkpoint saved: {checkpoint_path}")


# flwr run lädt über pyproject.toml serverapp = "…server_app:app".
# Flower ruft server_fn(context) auf.
#  context.run_config (die TOML-Werte) werden gelesen
# bauen FedAvg(..., on_fit_config_fn=..., evaluate_metrics_aggregation_fn=...).
# udn geben ServerAppComponents(config, strategy) zurück.
# Ab dann orchestriert Flower die Runden (Sampling, Fit, Evaluate).
def server_fn(context: Context) -> ServerAppComponents:
    """Build strategy + config. FedAvg + multi-threshold optimization + che"""
    rc = dict(context.run_config)
    
    # Screening-Policy (OHNE round_counter!)
    screening = ScreeningPolicy()
    
    # Checkpoint-Verzeichnis
    checkpoint_dir = Path(f"result/{rc.get('split-path','default')}/multi_thr/")
    
    # Model-Dimension (aus prepared data)
    model_dim = 21  # Deine Feature-Anzahl (kannst du auch dynamisch laden)
    
    # Threshold-Grid
    threshold_grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    
    def on_fit_config_fn(rnd: int) -> dict:
        """Config für Training"""
        lr = float(rc.get("lr", 1e-2)) if rnd < 3 else float(rc.get("lr-after", 5e-3))
        return {
            "epochs": int(rc.get("local-epochs", 1)),
            "lr": lr,
            "mu": float(rc.get("mu", 1e-3)),
            "weight-decay": float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
        }
    
    def on_evaluate_config_fn(rnd: int) -> dict:
        """Config für Evaluation"""
        return {
            "threshold_grid": json.dumps(threshold_grid),
        }
    
    def _safe_div(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0
    
    def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
        """Berechne Metriken aus TP/FP/TN/FN"""
        tpr = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        spec = 1.0 - fpr
        ppv = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        f1 = _safe_div(2*ppv*tpr, ppv + tpr) if (ppv + tpr) else 0.0
        bal_acc = 0.5 * (tpr + spec)
        youden = tpr + spec - 1.0
        prev = _safe_div(tp + fn, tp + fp + tn + fn)
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
        for n, md in eval_metrics:
            if isinstance(md, ConfigRecord):
                md_dict = dict(md.items())
            elif isinstance(md, dict):
                md_dict = md
            else:
                print(f"Unknown metrics type: {type(md)}")
                continue
            processed_metrics.append((n, md_dict))
        
        # 1) AUC aggregieren
        auc_weighted_sum = 0.0
        total_weight_for_auc = 0
        
        for n, md in processed_metrics:
            if not isinstance(md, dict):
                continue
            auc = md.get("auc", None)
            if auc is not None:
                w = int(n) if n else 1
                auc_weighted_sum += float(auc) * w
                total_weight_for_auc += w
        
        aggregated_auc = auc_weighted_sum / total_weight_for_auc if total_weight_for_auc else 0.0
        
        # 2) Deserialize und Aggregiere Threshold-Counts
        threshold_aggregated = {}
        
        for n, md in processed_metrics:
            if not isinstance(md, dict):
                continue
            
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
        
        # 4) Wähle besten Threshold
        best_thr_result = None
        best_score = -1.0
        
        MIN_RECALL = 0.75
        MIN_SPEC = 0.70
        
        # Durchlaufe alle Threshold-Ergebnisse um den besten zu finden
        for result in threshold_results:
            recall = result["recall"]
            spec = result["spec"]
            f1 = result["f1"]
            
            # 4.1) Filtere nach MIN_RECALL
            if recall < MIN_RECALL:
                continue
            
            # 4.2) Spec Penalty is calculated if spec is below MIN_SPEC
            spec_penalty = 0.0 if spec >= MIN_SPEC else (MIN_SPEC - spec) * 0.5
            
            # 4.3) Score Berechnung
            score = (
                0.40 * recall +     # Recall wichtigster Faktor
                0.30 * spec +       # Spec auch 
                0.20 * f1 +
                0.10 * (1.0 - spec_penalty) 
            )
            
            # 4.4) Wähle basierend auf bestem Score den Threshold
            if score > best_score:
                best_score = score
                best_thr_result = result
        
        # Falls kein Threshold MIN_RECALL erfüllt, wähle besten Youden-Index
        if best_thr_result is None:
            print(f"⚠️  No threshold meets min_recall={MIN_RECALL}, using best youden index.")
            best_thr_result = max(threshold_results, key=lambda x: x["youden"])
        
        # 5) Logging
        print(f"\n🎯 Multi-Threshold Aggregation:")
        print(f"   ═══════════════════════════════════════════════")
        print(f"   Evaluated {len(threshold_results)} thresholds:")
        
        for result in threshold_results[:3]:
            print(f"     • Thr={result['threshold']:.2f}: "
                  f"Recall={result['recall']:.3f}, Spec={result['spec']:.3f}, F1={result['f1']:.3f}")
        
        print(f"   ...")
        print(f"\n   BEST Threshold: {best_thr_result['threshold']:.2f}")
        print(f"     • Recall:      {best_thr_result['recall']:.3f}")
        print(f"     • Specificity: {best_thr_result['spec']:.3f}")
        print(f"     • F1-Score:    {best_thr_result['f1']:.3f}")
        print(f"     • Score:       {best_score:.3f}")
        print(f"   ═══════════════════════════════════════════════\n")
        
        # Verwenden metrics_aggregated als Träger
        agg = dict(best_thr_result)
        agg["auc"] = aggregated_auc
        agg["best_threshold"] = best_thr_result["threshold"]
        agg["best_score"] = best_score
        
        return agg
    
    def fit_metrics_aggregation_fn(metrics):
        """Aggregiere Fit-Metriken."""
        n_sum = sum(n for n, _ in metrics) or 1
        keys = set().union(*(m.keys() for _, m in metrics))
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / n_sum for k in keys}
    
    # ✅ Eine saubere Strategy-Klasse
    strategy = FedAvgWithScreening(
        screening_policy=screening,
        checkpoint_dir=checkpoint_dir,
        model_dim=model_dim,
        run_config=rc,
        # FedAvg-Parameter
        fraction_fit=float(rc.get("fraction-fit", 0.8)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 6)), # is the number of clients that are sampled to be trained in each round
        min_available_clients=int(rc.get("min-available-clients", 8)), 
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 8)),
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    
    cfg = ServerConfig(num_rounds=int(rc.get("num-server-rounds", 20)))
    return ServerAppComponents(config=cfg, strategy=strategy)

# Create ServerApp
# ServerApp is the main entry point for the Flower server
# It orchestrates the federated learning process, manages clients, and controls training and evaluation.
# When we flwr run, flower reads from the pyproject.toml file and loads the configurations. 
# 
# server_fn prepares everything we need to run the server
# creates the model, defines the strategy, and sets the server config (numberof rounds)

app = ServerApp(server_fn=server_fn)


