"""federated-learning: A Flower / PyTorch app."""
# federated_learning/client_app.py
# Hält MLP und Training-/Eval-Funktionen.
from __future__ import annotations

import json
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from federated_learning.task import load_client_data, make_loaders_from_arrays

DEVICE = torch.device("cpu")


# define a simple MLP model for Classification
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int] = [256, 128], out_dim: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

        # Kaiming-Init für Hidden Layers
        for m in self.net:
           if isinstance(m, nn.Linear):
               nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
        
        # Letzte Schicht: noch kleinere Gewichte (für FEL dann)
        # final_layer = list(self.net.children())[-1]
        # if isinstance(final_layer, nn.Linear):
        #     nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)  # sehr kleine Gewichte
        #     if final_layer.bias is not None:
        #         nn.init.constant_(final_layer.bias, 0)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification (focuses on hard examples)."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor [w_neg, w_pos]
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        p = torch.exp(-ce)
        focal = (1 - p) ** self.gamma * ce
        return focal.mean()




# --- Training/Eval-Utilities ---
def evaluate_multi_threshold(
    model: nn.Module, 
    loader, 
    crit,
    threshold_grid: List[float]
) -> Tuple[float, int, dict]:
    """
    Evaluates model on validation data across multiple thresholds.
    Returns: 
            avg_loss (float): average loss,
            n_samples (int): number of samples,
            metrics (json-serializable dict): includes
            TN, FN, TP and FP for each threshold, number of samples and AUC.
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    probs_all = []
    y_all = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs.cpu())
            y_all.append(yb.cpu())
    
    probs = torch.cat(probs_all).numpy()
    y = torch.cat(y_all).numpy()
    
    auc = 0.0
    
    # ============================================================
    # 1) MULTI-THRESHOLD METRIKEN (wie vorher)
    # ============================================================
    thresholds = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    
    for thr in threshold_grid:
        preds = (probs >= thr).astype(int)
        
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        
        thresholds.append(float(thr))
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
    
    # ============================================================
    # 2) CALIBRATION PLOT METRIKEN (10 Bins) - Privacy-Safe
    # ============================================================
    num_calib_bins = 10
    calib_edges = np.linspace(0.0, 1.0, num_calib_bins + 1)
    calib_bin_n = []
    calib_bin_sum_pred = []
    calib_bin_sum_true = []
    
    for i in range(num_calib_bins):
        lower = calib_edges[i]
        upper = calib_edges[i + 1]
        
        # Maske für Samples in diesem Bin
        if i == num_calib_bins - 1:
            mask = (probs >= lower) & (probs <= upper)  # Letztes Bin inclusive
        else:
            mask = (probs >= lower) & (probs < upper)
        
        n_in_bin = int(mask.sum())
        sum_pred = float(probs[mask].sum()) if n_in_bin > 0 else 0.0
        sum_true = int(y[mask].sum()) if n_in_bin > 0 else 0
        
        calib_bin_n.append(n_in_bin)
        calib_bin_sum_pred.append(sum_pred)
        calib_bin_sum_true.append(sum_true)
    
    # ============================================================
    # 3) RISK DISTRIBUTION PLOT METRIKEN (20 Bins) - Privacy-Safe
    # ============================================================
    num_risk_bins = 20
    risk_edges = np.linspace(0.0, 1.0, num_risk_bins + 1)
    
    # Histogram für y=0 (negative class)
    hist_pred_y0 = []
    for i in range(num_risk_bins):
        lower = risk_edges[i]
        upper = risk_edges[i + 1]
        
        if i == num_risk_bins - 1:
            mask = (probs >= lower) & (probs <= upper) & (y == 0)
        else:
            mask = (probs >= lower) & (probs < upper) & (y == 0)
        
        count = int(mask.sum())
        hist_pred_y0.append(count)
    
    # Histogram für y=1 (positive class)
    hist_pred_y1 = []
    for i in range(num_risk_bins):
        lower = risk_edges[i]
        upper = risk_edges[i + 1]
        
        if i == num_risk_bins - 1:
            mask = (probs >= lower) & (probs <= upper) & (y == 1)
        else:
            mask = (probs >= lower) & (probs < upper) & (y == 1)
        
        count = int(mask.sum())
        hist_pred_y1.append(count)
    
    # JSON-Serialisierung
    metrics = {
        "auc": auc,
        "n_samples": n_samples,
        # Multi-Threshold
        "thresholds_json": json.dumps(thresholds), 
        "tp_json": json.dumps(tp_list),             
        "fp_json": json.dumps(fp_list),
        "tn_json": json.dumps(tn_list),
        "fn_json": json.dumps(fn_list),
        # Calibration Plot (10 Bins)
        "calib_edges_json": json.dumps(calib_edges.tolist()),
        "calib_bin_n_json": json.dumps(calib_bin_n),
        "calib_bin_sum_pred_json": json.dumps(calib_bin_sum_pred),
        "calib_bin_sum_true_json": json.dumps(calib_bin_sum_true),
        # Risk Distribution (20 Bins)
        "risk_edges_json": json.dumps(risk_edges.tolist()),
        "hist_pred_y0_json": json.dumps(hist_pred_y0),
        "hist_pred_y1_json": json.dumps(hist_pred_y1),
    }
    
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, n_samples, metrics


    


def evaluate(model: nn.Module, loader, crit, threshold: float) -> Tuple[float, int, dict]:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    # Zähler für globale Metriken
    tp = fp = tn = fn = 0
    probs_all = []
    y_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

            # Wahrscheinlichkeiten für Klasse 1
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()

            tp += int(((preds == 1) & (yb == 1)).sum())
            fp += int(((preds == 1) & (yb == 0)).sum())
            tn += int(((preds == 0) & (yb == 0)).sum())
            fn += int(((preds == 0) & (yb == 1)).sum())

            probs_all.append(probs.detach().cpu())
            y_all.append(yb.detach().cpu())

    # AUC ist threshold-unabhängig
    # try:
    #     if probs_all:
    #         p = torch.cat(probs_all).numpy()
    #         y = torch.cat(y_all).numpy()
    #         auc = float(roc_auc_score(y, p))
    #     else:
    #         auc = 0.0
    # except Exception:
    auc = 0.0

    metrics = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "auc": auc}
    avg_loss = total_loss / max(1, n_samples)
    return avg_loss, n_samples, metrics


# --- Flower-Client ---
class FlowerClient(NumPyClient):
    def __init__(self, cid: str, rc: dict):
        super().__init__()
        # Client-ID, Run-Config von client_fn
        self.cid = str(cid)
        self.rc = rc
        
        # 1. Split-Mapping laden
        split_path = rc.get("split-path")
        if not split_path:
            raise RuntimeError("run_config['split-path'] fehlt.")
        
        with open(split_path, "r") as f:
            split_data = json.load(f)
        
        # 2. Client-spezifische Indices/ Row IDs extrahieren
        train_mapping = split_data.get("train", {})
        val_mapping = split_data.get("val", {})
        
        if self.cid not in train_mapping:
            raise KeyError(f"cid {self.cid} fehlt in train mapping")
        
        # Client-spezifische Indices
        client_train_idx = train_mapping[self.cid]
        # ✅ Val ist optional: Manche Clients haben keine Val-Daten
        client_val_idx = val_mapping.get(self.cid, [])

        print(f"[Client {self.cid}] Data split: Train:{len(client_train_idx)}, Validation: {len(client_val_idx)} ")
        
        # 3. Lade Daten + Class-Weights (global berechnet, nicht für echtes FEL scenario :( ), hab ich jetzt schon vorher in normalice and add weights angewand, könnte man nochmal umschrieben.
        boost_factor = float(rc.get("pos-weight-boost", 2.0))
        
        X_train, y_train, X_val, y_val, class_weights = load_client_data(
            parquet_path=rc["prepared-parquet"],
            stats_path=rc["norm-stats-json"],
            train_row_ids=client_train_idx,
            val_row_ids=client_val_idx,
        )
        
        # 4. DataLoader erstellen
        bs = int(rc.get("batch-size", 128))
        self.train_loader, self.val_loader = make_loaders_from_arrays(
            X_train, y_train, X_val, y_val, batch_size=bs
        )
        
        # 5. Modell
        self.model = MLP(in_dim=X_train.shape[1]).to(DEVICE)
        
        # 6. Loss mit Class-Weights (bereits auf CPU, muss nur zu GPU)
        class_weights = class_weights.to(DEVICE)
        
        use_focal = rc.get("use-focal-loss", False)
        if use_focal:
            gamma = float(rc.get("focal-gamma", 2.0))
            self.crit = FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            self.crit = nn.CrossEntropyLoss(weight=class_weights)

        # 7. Defaults
        self.default_lr = float(rc.get("lr", 1e-2))
        
        # Anzahl der lokalen Epochen aus config, (default 1)
        # the config arument is send from the server (run-config) to the client
        # the server can set the number of epochs, learning rate, etc. (look at server_app.py)
        # in the next step we then train our local model.
        self.local_epochs = int(rc.get("local-epochs", 1))
        self.eval_threshold = float(rc.get("eval-threshold", 0.35))


     # FOR EVERY ROUND THE FOLLOWING METHODS ARE CALLED:
     # 1) fit is called to train the model locally for each client
         # 1) the current weights are set
         # 2) the model is trained on the local data for a number of epochs
         # 3) the new weights are returned to the server
    def fit(self, parameters, config):
        # Sets the model weights and trains the model on the local data
        self.set_parameters(parameters)

        lr = float(config.get("lr", self.default_lr))
        epochs = int(config.get("epochs", self.local_epochs))
        wd = float(config.get("weight-decay", 1e-4))
        clip = float(config.get("clip-grad-norm", 5.0))

        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # Training
        fit_total = 0
        fit_correct = 0
        fit_ce_sum = 0.0

        self.model.train()
        for _ in range(epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = self.model(xb)
                ce = self.crit(logits, yb)

                # ✅ FedAdam: Nur Cross-Entropy Loss (keine FedProx-Regularisierung!)
                loss = ce
                loss.backward()
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                opt.step()

                bs = xb.size(0)
                fit_total += bs
                fit_correct += (logits.argmax(1) == yb).sum().item()
                fit_ce_sum += ce.item() * bs  # nur CE loggen

        fit_metrics = {
            "fit_loss": fit_ce_sum / max(1, fit_total),
            "fit_accuracy": fit_correct / max(1, fit_total),
        }
        return self.get_parameters({}), len(self.train_loader.dataset), fit_metrics

  
    # 2) this method is called to evaluate the local clients model
        # 1) the new global weights are set and and 
        #    the optimal threshold is found on the local validation data
        # 2) the model is evaluated on the local test data
        # 3) Loss and accurany are returned
    def evaluate(self, parameters, config):
        """Evaluate nur wenn config nicht leer"""
    
        # SKIP wenn Server keine Evaluation will
        if not config or len(config) == 0:
            return 0.0, 0, {}  # Dummy-Return
    
        # 1) Set parameters
        self.set_parameters(parameters)
        
        # 2) Get threshold grid
        threshold_grid_str = config.get(
            "threshold_grid", 
            json.dumps([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        )
        threshold_grid = json.loads(threshold_grid_str)
        
        # 3) Evaluate (nur wenn Daten vorhanden)
        if len(self.val_loader.dataset) >= 1:
            loss, n_val, metrics = evaluate_multi_threshold(
                self.model, 
                self.val_loader,
                self.crit,
                threshold_grid
            )
        else:
            loss = 0.0
            n_val = 0
            metrics = {
                "auc": 0.0,
                "n_samples": 0,
                "thresholds_json": json.dumps([]),
                "tp_json": json.dumps([]),
                "fp_json": json.dumps([]),
                "tn_json": json.dumps([]),
                "fn_json": json.dumps([]),
            }
    
        return float(loss), int(n_val), metrics
    
    def set_parameters(self, parameters):
        keys = list(self.model.state_dict().keys())
        state = {k: torch.from_numpy(v).float() for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)

    # 1.2) After the training is complete, we return the current model weights to the server
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]


def client_fn(context: Context):
    # run_config als dict holen
    rc = dict(context.run_config)
    cid = None
    if hasattr(context, "node_config"):
        # cid(partition-id) wird aus der run_config gholt
        cid = context.node_config.get("partition-id")
    if cid is None:
        raise RuntimeError("No client id (cid) found in context.node_config or run_config.")
    # erzeugen von FlowerClient mit Client-ID und Run-Config
    return FlowerClient(str(cid), rc).to_client()

app = ClientApp(client_fn=client_fn)
















# UNUSED METHODs FOR THE CASE OF THE CASE:

# def find_optimal_threshold(model: nn.Module, loader, min_recall=0.75, min_spec=0.70):
#     """
#     Client findet optimalen Threshold lokal basierend auf:
#     - Min. Recall (Safety: muss mind. 75% der Fälle finden)
#     - Beste Specificity (bei gegebenem Recall)
#     """
#     model.eval()
#     probs_all = []
#     y_all = []
    
#     # Sammle alle Wahrscheinlichkeiten (lokal, werden NICHT gesendet!)
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             logits = model(xb)
#             probs = torch.softmax(logits, dim=1)[:, 1]
#             probs_all.append(probs.cpu())
#             y_all.append(yb.cpu())
    
#     probs = torch.cat(probs_all).numpy()
#     y = torch.cat(y_all).numpy()
    
#     # Berechne ROC-Kurve (lokal!)
#     from sklearn.metrics import roc_curve
#     fpr, tpr, thresholds = roc_curve(y, probs)
    
#     # Finde Thresholds, die min_recall erfüllen
#     valid_indices = np.where(tpr >= min_recall)[0]
    
#     if len(valid_indices) == 0:
#         # Fallback: Bester verfügbarer Recall
#         optimal_idx = np.argmax(tpr)
#     else:
#         # Unter gültigen: maximiere Specificity (= minimiere FPR)
#         valid_fprs = fpr[valid_indices]
#         best_among_valid = valid_indices[np.argmin(valid_fprs)]
#         optimal_idx = best_among_valid
    
#     optimal_threshold = float(thresholds[optimal_idx])
#     optimal_recall = float(tpr[optimal_idx])
#     optimal_spec = float(1.0 - fpr[optimal_idx])
    
#     # Berechne auch Youden's Index zum Vergleich
#     youden_index = tpr - fpr
#     youden_optimal_idx = np.argmax(youden_index)
#     youden_threshold = float(thresholds[youden_optimal_idx])
    
#     return {
#         "optimal_threshold": optimal_threshold,
#         "optimal_recall": optimal_recall,
#         "optimal_spec": optimal_spec,
#         "youden_threshold": youden_threshold,  # Alternative Metrik
#         "num_samples": len(y),
#     }


# def find_optimal_threshold_adaptive(
#     model: nn.Module, 
#     loader, 
#     min_recall=0.72, 
#     min_spec=0.70,
#     fallback_min_recall=0.65,  #Relaxed fallback
#     fallback_min_spec=0.60):
#     """
#     Adaptive threshold search with graceful degradation:
#     1) Try to find threshold with (min_recall, min_spec)
#     2) If fails, try relaxed constraints (fallback_min_recall, fallback_min_spec)
#     3) If still fails, use Youden's Index (best balance)
#     """
#     model.eval()
#     probs_all = []
#     y_all = []
    
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             logits = model(xb)
#             probs = torch.softmax(logits, dim=1)[:, 1]
#             probs_all.append(probs.cpu())
#             y_all.append(yb.cpu())
    
#     probs = torch.cat(probs_all).numpy()
#     y = torch.cat(y_all).numpy()
    
#     from sklearn.metrics import roc_curve
#     fpr, tpr, thresholds = roc_curve(y, probs)
    
#     # Schritt 1: Versuche primäre Constraints (strikt)
#     valid_indices = np.where((tpr >= min_recall) & ((1 - fpr) >= min_spec))[0]
    
#     if len(valid_indices) > 0:
#         # Erfolg: Wähle besten unter gültigen (minimiere FPR)
#         best_idx = valid_indices[np.argmin(fpr[valid_indices])]
#         strategy = "primary"
#     else:
#         # Schritt 2: Fallback zu relaxed Constraints
#         print(f"⚠️  Client: No threshold meets primary constraints (recall≥{min_recall}, spec≥{min_spec})")
#         print(f"   Trying relaxed constraints (recall≥{fallback_min_recall}, spec≥{fallback_min_spec})")
        
#         valid_indices = np.where((tpr >= fallback_min_recall) & ((1 - fpr) >= fallback_min_spec))[0]
        
#         if len(valid_indices) > 0:
#             best_idx = valid_indices[np.argmin(fpr[valid_indices])]
#             strategy = "fallback"
#         else:
#             # Schritt 3: Ultimate Fallback - Youden's Index
#             print(f"⚠️  Client: Relaxed constraints also failed")
#             print(f"   Using Youden's Index (best balance between recall & spec)")
            
#             youden_index = tpr - fpr
#             best_idx = np.argmax(youden_index)
#             strategy = "youden"
    
#     optimal_threshold = float(thresholds[best_idx])
#     optimal_recall = float(tpr[best_idx])
#     optimal_spec = float(1.0 - fpr[best_idx])
    
#     # ✅ Zusätzlich: Youden's Threshold als Referenz
#     youden_index = tpr - fpr
#     youden_idx = np.argmax(youden_index)
#     youden_threshold = float(thresholds[youden_idx])
    
#     # ✅ Quality Score: Wie gut ist dieser Threshold?
#     quality_score = 0.6 * optimal_recall + 0.4 * optimal_spec  # Gewichtet für Screening
    
#     return {
#         "optimal_threshold": optimal_threshold,
#         "optimal_recall": optimal_recall,
#         "optimal_spec": optimal_spec,
#         "youden_threshold": youden_threshold,
#         "num_samples": len(y),
#         "strategy": strategy,  # "primary", "fallback", or "youden"
#         "quality_score": quality_score,  # 0-1, higher = better
#     }
