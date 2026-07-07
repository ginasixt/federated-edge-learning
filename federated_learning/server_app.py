# federated_learning/server_app.py
from __future__ import annotations

import torch
import json
import numpy as np
import random
from pathlib import Path
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAdam
from flwr.common import Context, Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.record import ConfigRecord
from federated_learning.client_app import MLP

try:
    from flwr.server.app import ServerAppComponents
except ImportError:
    from flwr.server import ServerAppComponents


class FedAdamWithScreening(FedAdam):
    """
    Custom FedAdam Strategy that:
    1. Caches parameters in RAM after aggregation
    2. Tracks metrics with ScreeningPolicy
    3. Saves checkpoints for best rounds
    4. ✅ Only samples clients with actual validation data for evaluation (efficiency!)
    
    FedAdam = Federated Adaptive Moment Estimation (optimizer wie Adam, aber für FL)
    Vorteile gegenüber FedAvg:
    - Adaptive learning rates pro Parameter
    - Robuster gegen heterogene Datenverteilungen
    - Schnellere Konvergenz bei komplexen Problemen
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        model_dim: int,  # Feature-Dimension für MLP
        run_config: dict,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.model_dim = model_dim
        self.run_config = run_config
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Template für State Dict Keys
        self.template_model = MLP(in_dim=model_dim)
        
        # ✅ Lade Val-Client-Range aus Split Meta für effiziente Evaluation
        split_path = run_config.get("split-path")
        self.val_client_range = None
        if split_path:
            try:
                with open(split_path, "r") as f:
                    split_data = json.load(f)
                    meta = split_data.get("meta", {})
                    self.val_client_range = meta.get("val_client_range")
                    if self.val_client_range:
                        print(f"✅ Val-Client-Range geladen: {self.val_client_range['min']}-{self.val_client_range['max']}")
            except Exception as e:
                print(f"⚠️  Konnte Val-Client-Range nicht laden: {e}")
                self.val_client_range = None
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[any, FitRes]],
        failures: list[tuple[any, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, any]]:
        """Aggregiere Gewichte und speichere den Checkpoint direkt auf Disk."""
        
        # 1) Standard FedAdam Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is None:
            return None, {}
        
        # 2) Speichere das aggregierte Modell direkt als Checkpoint auf Disk
        run_tag = str(self.run_config.get("run-tag", "1"))
        checkpoint_path = self.checkpoint_dir / f"model_round_{server_round}_run_{run_tag}.pt"
        self._save_checkpoint(server_round, aggregated_parameters, checkpoint_path)
        
        return aggregated_parameters, aggregated_metrics
    
    def select_for_evaluate(self, server_round: int, available_clients, num_to_select: int):
        """
        ✅ OPTIMIZATION: Nur Val-Clients evaluieren!
        
        Nutzt die Val-Client-Range aus dem Split Meta (viel effizienter als Liste laden).
        Standard Flower sampelt zufällig alle Clients.
        Aber Clients ohne Val-Daten→ n_val=0 (Rechenaufwand verschwendet!)
        
        Diese Methode sampelt nur aus Clients mit echten Val-Daten.
        → Spart Rechenaufwand + schneller Evaluation
        """
        # Speichere alle verfügbaren Client-IDs (als Ints)
        all_cids = [int(client.node_id) for client in available_clients]
        
        # Filtere nur Val-Clients (aus Range)
        if self.val_client_range:
            min_cid = self.val_client_range.get("min", 0)
            max_cid = self.val_client_range.get("max", 0)
            val_clients = [c for c in all_cids if min_cid <= c <= max_cid]
        else:
            # Fallback: Wenn Range nicht geladen, nehme alle
            val_clients = all_cids
        
        # Sample aus Val-Clients
        if val_clients:
            num_to_select = min(num_to_select, len(val_clients))
            selected_ids = set(random.sample(val_clients, num_to_select))
        else:
            selected_ids = set()
        
        # Gebe nur die selected Clients zurück
        selected_clients = [c for c in available_clients if int(c.node_id) in selected_ids]
        
        return selected_clients
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[any, any]],
        failures: list[tuple[any, any] | BaseException],
    ) -> tuple[float | None, dict[str, any]]:
        """Nach Evaluation: Prüfe ob beste Runde und speichere Checkpoint."""
        
        # ✅ 1) Early Exit: Keine Results 
        if not results:
            return None, {}
        
        # ✅ 2) Filtering: Nur Clients mit n_val > 0 
        valid_results = []
        for client_proxy, evaluate_res in results:
            if evaluate_res.num_examples > 0:
                valid_results.append((client_proxy, evaluate_res))
        
        # ✅ 3) Check: Alle n_val=0?
        if not valid_results:
            print(f"⏭Round {server_round}: Evaluation skipped")
            return None, {}
        
        # ✅ 4) Standard Aggregation 
        loss, metrics = super().aggregate_evaluate(server_round, valid_results, failures)
        
        if not metrics:
            return loss, {}
        
        # 5) Speichere Metriken als JSON und verknüpfe den bereits gespeicherten Checkpoint
        run_tag = str(self.run_config.get("run-tag", "1"))
        checkpoint_path = self.checkpoint_dir / f"model_round_{server_round}_run_{run_tag}.pt"
        json_path = self.checkpoint_dir / f"round_{server_round}_run_{run_tag}.json"
        
        if checkpoint_path.exists():
            metrics_with_meta = {
                "round": server_round,
                "metrics": metrics,
                "model_checkpoint": str(checkpoint_path)
            }

            with open(json_path, "w") as f:
                json.dump(metrics_with_meta, f, indent=2)

            print(f"💾 Round {server_round}: Saved {checkpoint_path.name} + {json_path.name}")
        else:
            print(f"⚠️  Round {server_round}: Checkpoint missing at evaluation time: {checkpoint_path}")

        # 6) OPTIONAL: Generiere Calibration & Risk Distribution Plots
        # try:
        #     from federated_learning.plotting.calibration_and_risk_plots import generate_both_plots
        #     
        #     plots_dir = self.checkpoint_dir / "plots"
        #     generate_both_plots(
        #         metrics,
        #         output_dir=plots_dir,
        #         round_num=server_round,
        #         show=False  # Nicht im Background anzeigen
        #     )
        # except ImportError:
        #     pass  # Plots sind optional
        # except Exception as e:
        #     print(f"⚠️  Plot generation failed: {e}")
        
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
            print(f"Parameter count mismatch! "
                  f"Expected {len(state_dict_keys)}, got {len(ndarrays)}")
            return
        
        state_dict = {
            key: torch.tensor(arr, dtype=torch.float32)
            for key, arr in zip(state_dict_keys, ndarrays)
        }
        
        # Speichere auf Disk
        torch.save(state_dict, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


# flwr run lädt über pyproject.toml serverapp = "…server_app:app".
# Flower ruft server_fn(context) auf.
#  context.run_config (die TOML-Werte) werden gelesen
# bauen FedAvg(..., on_fit_config_fn=..., evaluate_metrics_aggregation_fn=...).
# udn geben ServerAppComponents(config, strategy) zurück.
# Ab dann orchestriert Flower die Runden (Sampling, Fit, Evaluate).
def server_fn(context: Context) -> ServerAppComponents:
    """Build strategy + config. FedAvg + multi-threshold optimization + checkpointing"""
    rc = dict(context.run_config)
    
    # Checkpoint-Verzeichnis (alle Runden für Post-Training Screening)
    checkpoint_dir = Path(f"result/{rc.get('split-path','default')}/all_rounds/")
    
    # Model-Dimension (aus prepared data)
    model_dim = 21  # Deine Feature-Anzahl (kannst du auch dynamisch laden)

    total_rounds = int(rc.get("num-server-rounds", 80))

    # --- Robust Schedule für viele Clients mit kleinen, unbalancierten Local-Splits ---
    warmup_rounds = int(rc.get("warmup-rounds", 8))
    warmup_lr_start = float(rc.get("warmup-lr-start", 1e-3))
    warmup_lr_end = float(rc.get("warmup-lr-end", 3e-3))
    warmup_mu_start = float(rc.get("warmup-mu-start", 0.0))
    warmup_mu_end = float(rc.get("warmup-mu-end", 1e-5))

    lr_main = float(rc.get("lr", 1e-2))
    lr_after = float(rc.get("lr-after", 5e-3))
    lr_after_round = int(rc.get("lr-after-round", 60))
    mu_main = float(rc.get("mu", 1e-4))

    def _linear_round_schedule(rnd: int, total: int, start: float, end: float) -> float:
        """Linear schedule: mostly for warmup phase."""
        if total <= 1:
            return float(end)
        progress = float(rnd - 1) / float(total - 1)
        progress = max(0.0, min(1.0, progress))
        return float(start + (end - start) * progress)
    
    def _cosine_annealing_schedule(rnd: int, total_rounds: int, lr_max: float, lr_min: float) -> float:
        """
        Cosine Annealing Schedule (smooth, mathematically elegant).
        
        Uses: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
        
        This creates a smooth decay from lr_max to lr_min over total_rounds.
        - Start: lr_max (rnd=1)
        - Middle: smooth decline
        - End (rnd=total_rounds): approaches lr_min
        """
        if total_rounds <= 1:
            return float(lr_min)
        
        t = float(rnd - 1)  # Progress from 0 to total_rounds-1
        T = float(total_rounds - 1)
        
        # Cosine annealing formula
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * t / T))
        return float(lr)
    
    
    def on_fit_config_fn(rnd: int) -> dict:
        """
        ✅ 3-PHASE TRAINING SCHEDULE (Optimal für Federated Learning):
        
        PHASE 1: Warmup (Runden 1-10)
          - LR linear: 1e-3 → 5e-2 (sanfte Stabilisierung)
          - Mu linear: 0.0 → 5e-4 (FedProx hochfahren)
        
        PHASE 2: Stable Training (Runden 11-60)
          - LR konstant: 5e-2 (aggressives Lernen)
          - Mu konstant: 5e-4 (starke Regularisierung)
          ⭐ Das Herzstück des Trainings!
        
        PHASE 3: Cosine Annealing (Runden 61-75)
          - LR cosine decay: 5e-2 → 2e-2 (sanfte Konvergenz)
          - Mu leicht reduzieren: 5e-4 → 1e-4 (weniger Ridge am Ende)
        
        Vorteile:
        - Stabil: Warmup → Plateau → Decay
        - LR: Aggressive dann graduell weniger
        - Mu: KONSTANT während Haupttraining (Ridge-Stärke!)
        """
        
        # Phase 1: Warmup (Runden 1-10)
        if rnd <= warmup_rounds:
            lr = _linear_round_schedule(rnd, warmup_rounds, warmup_lr_start, warmup_lr_end)
            mu = _linear_round_schedule(rnd, warmup_rounds, warmup_mu_start, warmup_mu_end)
        
        # Phase 2: Stable Training (Runden 11-60) - KONSTANTE LR & MU!
        elif rnd <= 60:
            lr = lr_main              # Konstant: 5e-2
            mu = mu_main              # Konstant: 5e-4 (volle FedProx Stärke!)
        
        # Phase 3: Cosine Annealing (Runden 61-80, dynamisch berechnet)
        else:
            remaining_rounds = total_rounds - 60  # Z.B. 80 - 60 = 20 Runden für Annealing
            round_in_cosine = rnd - 60  # 1-indexed in cosine phase
            
            # LR smooth decay: 5e-2 → 2e-2
            lr = _cosine_annealing_schedule(
                rnd=round_in_cosine,
                total_rounds=remaining_rounds,
                lr_max=lr_main,      # 5e-2
                lr_min=lr_after       # 2e-2
            )
            
            # Mu: Nur leicht reduzieren (Ridge bleibt wichtig!)
            # Decay von 5e-4 → 1e-4 (aber nicht so aggressiv wie LR)
            mu = _cosine_annealing_schedule(
                rnd=round_in_cosine,
                total_rounds=remaining_rounds,
                lr_max=mu_main,       # 5e-4
                lr_min=float(1e-4)    # 1e-4 (weniger aggressiv als wegfallen)
            )
        
        return {
            "epochs": int(rc.get("local-epochs", 1)),
            "lr": lr,
            "mu": mu,
            "weight-decay": float(rc.get("weight-decay", 1e-4)),
            "clip-grad-norm": float(rc.get("clip-grad-norm", 5.0)),
        }
    
    def on_evaluate_config_fn(rnd: int) -> dict:
        """
        ADAPTIVE Evaluation Schedule (Optimiert für schnelle Konvergenz):
        - Runden 1-34: Alle 5 Runden (Warmup/Early Training, sparsamer evaluieren)
        - Runden 35+: JEDE Runde (Dense Monitoring um Plateau zu erkennen)
        - Runde 1 & letzte Runde: IMMER evaluieren
        """
        # Erste und letzte Runde IMMER evaluieren
        if rnd == 1 or rnd == total_rounds:
            pass  # Evaluation läuft
        # Runden 2-20: Alle 5 Runden evaluieren
        elif rnd < 15:
            if rnd % 5 != 0:
                return {}  # Skip Evaluation, senden leere Config
        # Runden 35+: JEDE Runde evaluieren (dense monitoring)
        
        # ✅ Threshold Grid (nur wenn Eval läuft)
        threshold_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        
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
        
        # ============================================================
        # 4) CALIBRATION PLOT AGGREGATION (10 Bins)
        # ============================================================
        calib_aggregated = {}
        
        for n, md in processed_metrics:
            try:
                calib_edges = json.loads(md.get("calib_edges_json", "[]"))
                calib_bin_n = json.loads(md.get("calib_bin_n_json", "[]"))
                calib_bin_sum_pred = json.loads(md.get("calib_bin_sum_pred_json", "[]"))
                calib_bin_sum_true = json.loads(md.get("calib_bin_sum_true_json", "[]"))
            except json.JSONDecodeError:
                continue
            
            if not calib_bin_n:  # Skip wenn leer
                continue
            
            # Initialize aggregate bins if not done yet
            if not calib_aggregated:
                num_bins = len(calib_bin_n)
                calib_aggregated = {
                    "edges": calib_edges,
                    "bin_n": [0] * num_bins,
                    "bin_sum_pred": [0.0] * num_bins,
                    "bin_sum_true": [0] * num_bins,
                }
            
            # Aggregiere Bin-Counts
            for i in range(len(calib_bin_n)):
                calib_aggregated["bin_n"][i] += int(calib_bin_n[i])
                calib_aggregated["bin_sum_pred"][i] += float(calib_bin_sum_pred[i])
                calib_aggregated["bin_sum_true"][i] += int(calib_bin_sum_true[i])
        
        # Berechne Calibration Metriken
        calibration_results = []
        if calib_aggregated:
            for i in range(len(calib_aggregated["bin_n"])):
                n_bin = calib_aggregated["bin_n"][i]
                if n_bin > 0:
                    mean_pred = calib_aggregated["bin_sum_pred"][i] / n_bin
                    mean_obs = calib_aggregated["bin_sum_true"][i] / n_bin
                else:
                    mean_pred = 0.0
                    mean_obs = 0.0
                
                calibration_results.append({
                    "bin_index": i,
                    "bin_edge_lower": float(calib_aggregated["edges"][i]),
                    "bin_edge_upper": float(calib_aggregated["edges"][i + 1]) if i + 1 < len(calib_aggregated["edges"]) else 1.0,
                    "n_samples": n_bin,
                    "mean_predicted_prob": mean_pred,
                    "mean_observed_freq": mean_obs,
                })
        
        # ============================================================
        # 5) RISK DISTRIBUTION AGGREGATION (20 Bins)
        # ============================================================
        risk_aggregated = {}
        
        for n, md in processed_metrics:
            try:
                risk_edges = json.loads(md.get("risk_edges_json", "[]"))
                hist_pred_y0 = json.loads(md.get("hist_pred_y0_json", "[]"))
                hist_pred_y1 = json.loads(md.get("hist_pred_y1_json", "[]"))
            except json.JSONDecodeError:
                continue
            
            if not hist_pred_y0:  # Skip wenn leer
                continue
            
            # Initialize aggregate histograms if not done yet
            if not risk_aggregated:
                num_bins = len(hist_pred_y0)
                risk_aggregated = {
                    "edges": risk_edges,
                    "hist_y0": [0] * num_bins,
                    "hist_y1": [0] * num_bins,
                }
            
            # Aggregiere Histogramm-Counts
            for i in range(len(hist_pred_y0)):
                risk_aggregated["hist_y0"][i] += int(hist_pred_y0[i])
                risk_aggregated["hist_y1"][i] += int(hist_pred_y1[i])
        
        # Berechne Risk Distribution Metriken
        risk_distribution = []
        if risk_aggregated:
            for i in range(len(risk_aggregated["hist_y0"])):
                risk_distribution.append({
                    "bin_index": i,
                    "bin_edge_lower": float(risk_aggregated["edges"][i]),
                    "bin_edge_upper": float(risk_aggregated["edges"][i + 1]) if i + 1 < len(risk_aggregated["edges"]) else 1.0,
                    "count_y0": risk_aggregated["hist_y0"][i],
                    "count_y1": risk_aggregated["hist_y1"][i],
                })
        
        # ============================================================
        # 6) Minimales Logging (für Live-Monitoring während Training)
        # ============================================================
        print(f"\n📊 Multi-Threshold Aggregation:")
        print(f"   Valid clients: {len(processed_metrics)}")
        print(f"   Evaluated {len(threshold_results)} thresholds:")
        
        for result in threshold_results[:3]:
            print(f"     Thr={result['threshold']:.2f}: "
                  f"Rec={result['recall']:.3f}, Spec={result['spec']:.3f}, F1={result['f1']:.3f}")
        
        if len(threshold_results) > 3:
            print(f"     ... (+ {len(threshold_results)-3} more)\n")
        
        print(f"📈 Calibration Plot: {len(calibration_results)} bins aggregated")
        print(f"📊 Risk Distribution: {len(risk_distribution)} bins aggregated\n")
        
        # ============================================================
        # 7) Gib ALLE Ergebnisse zurück (Plots & Post-Training Screening)
        # ============================================================
        return {
            "auc": aggregated_auc,
            "all_thresholds": threshold_results,  # Alle Thresholds für Screening-Tool
            "calibration": calibration_results,   # Für Calibration Plot
            "risk_distribution": risk_distribution,  # Für Risk Distribution Plot
        }
    
    def fit_metrics_aggregation_fn(metrics):
        """Aggregiere Fit-Metriken."""
        n_sum = sum(n for n, _ in metrics) or 1
        keys = set().union(*(m.keys() for _, m in metrics))
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / n_sum for k in keys}
    
    # ✅ Erzeuge initiale Parameter aus dem Modell
    initial_model = MLP(in_dim=model_dim)
    initial_weights = [
        v.detach().clone().numpy() 
        for v in initial_model.state_dict().values()
    ]
    initial_parameters = ndarrays_to_parameters(initial_weights)
    
    # ✅ FedAdam Strategy mit adaptive Server-Parametern
    strategy = FedAdamWithScreening(
        checkpoint_dir=checkpoint_dir,
        model_dim=model_dim,
        run_config=rc,
        initial_parameters=initial_parameters,
        # FedAdam-Parameter (Aggregation & Sampling)
        fraction_fit=float(rc.get("fraction-fit", 0.8)),
        fraction_evaluate=float(rc.get("fraction-evaluate", 1.0)),
        min_fit_clients=int(rc.get("min-fit-clients", 6)), # is the number of clients that are sampled to be trained in each round
        min_available_clients=int(rc.get("min-available-clients", 8)), 
        min_evaluate_clients=int(rc.get("min-evaluate-clients", 8)),
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        # ✅ FedAdam-spezifische Parameter (korrekte Flower-Namen!)
        eta=float(rc.get("eta", 0.1)),              # Server Learning Rate
        eta_l=float(rc.get("eta-l", 0.1)),          # Client Learning Rate (optional über Config)
        beta_1=float(rc.get("beta-1", 0.9)),        # Adam momentum für Gradienten
        beta_2=float(rc.get("beta-2", 0.99)),       # Adam momentum für quadr. Gradienten
        tau=float(rc.get("tau", 1e-9)),             # Epsilon für numerische Stabilität
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


