"""federated-learning: A Flower / PyTorch app."""
# federated_learning/client_app.py
from __future__ import annotations

import json
import pickle
import base64
from typing import List, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.record import ConfigRecord

from federated_learning.task import load_client_data, make_loaders_from_arrays



DEVICE = torch.device("cpu")


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
        final_layer = list(self.net.children())[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)  # sehr kleine Gewichte
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0)



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


# --- Serialization helpers for control variates ---
def _serialize_ndarrays(arrays: list[np.ndarray]) -> str:
    """Serialize list of numpy arrays to base64-encoded pickle string."""
    return base64.b64encode(pickle.dumps(arrays, protocol=4)).decode("ascii")


def _deserialize_ndarrays(s: str) -> list[np.ndarray]:
    """Deserialize base64-encoded pickle string back to list of numpy arrays."""
    return pickle.loads(base64.b64decode(s.encode("ascii")))


# --- Training/Eval-Utilities ---
def train_one_epoch(model: nn.Module, loader, opt, crit, clip_norm: float):
    """Train one epoch - FedProx removed, SCAFFOLD correction handled in FlowerClient.fit"""
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        opt.step()


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
    
    # Berechne Metriken
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
    
    # JSON-Serialisierung
    metrics = {
        "auc": auc,
        "n_samples": n_samples,
        "thresholds_json": json.dumps(thresholds), 
        "tp_json": json.dumps(tp_list),             
        "fp_json": json.dumps(fp_list),
        "tn_json": json.dumps(tn_list),
        "fn_json": json.dumps(fn_list),
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


class FlowerClient(NumPyClient):
    def __init__(self, cid: str, rc: dict, context: Context):  # accept context
        self.cid = cid
        self.context = context  # store for state access

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
        if self.cid not in val_mapping:
            raise KeyError(f"cid {self.cid} fehlt in val mapping")
        
        # Client-spezifische Indices
        client_train_idx = train_mapping[self.cid]
        client_val_idx = val_mapping[self.cid]

        print(f"[Client {self.cid}] Data split:")
        print(f"   Train:      {len(client_train_idx)} samples (client-local)")
        print(f"   Validation: {len(client_val_idx)} samples (client-local)")
        
        # 3. Lade Daten + Class-Weights (global berechnet, nicht für echtes FEL scenario :( ), hab ich jetzt schon vorher in normalice and add weights angewand, könnte man nochmal umschrieben.
        # boost_factor = float(1.3)
        
        X_train, y_train, X_val, y_val, class_weights = load_client_data(
            parquet_path=rc["prepared-parquet"],
            stats_path=rc["norm-stats-json"],
            train_row_ids=client_train_idx,
            val_row_ids=client_val_idx,
        )
        
        # 4. DataLoader erstellen
        bs = int(rc.get("batch-size", 128))

        # # 4. DataLoader erstellen – batch_size ≈ 20% of local training data
        # bs_cfg = rc.get("batch-size", None)
        # if bs_cfg is not None:
        #     bs = int(bs_cfg)
        # else:
        #     bs = max(1, int(len(X_train) * 0.2))  # ~20% of local data per batch
        #     print(f"[Client {self.cid}] Dynamic batch size: {bs} (~20% of {len(X_train)} train samples)")

        self.train_loader, self.val_loader = make_loaders_from_arrays(
            X_train, y_train, X_val, y_val, batch_size=bs
        )
        
        # 5. Modell
        self.model = MLP(in_dim=X_train.shape[1]).to(DEVICE)
        
        # 6. Loss mit Class-Weights (bereits auf CPU, muss nur zu GPU)
        # norm_stats wurde mit boost=2.0 vorberechnet → zur Laufzeit rescalen
        # PRECOMPUTED_BOOST = 2.0
        # class_weights[1] = class_weights[1] * (boost_factor / PRECOMPUTED_BOOST)
        class_weights = class_weights.to(DEVICE)
        
        self.crit = nn.CrossEntropyLoss(weight=class_weights)

        # 7. Defaults
        self.default_lr = float(rc.get("lr", 1e-2))
        self.local_epochs = int(rc.get("local-epochs", 2))
        self.eval_threshold = float(rc.get("eval-threshold", 0.35))

        # ── SCAFFOLD state: c_i initialised to zeros ──────────────────────
        # Will be lazily initialized to correct shape on first fit() call
        self.ci: list[torch.Tensor] | None = None

    # FOR EVERY ROUND THE FOLLOWING METHODS ARE CALLED:
    def fit(self, parameters, config):
        """Train locally with SCAFFOLD gradient correction."""
        # 1) Set current global weights & snapshot x (global model before update)
        self.set_parameters(parameters)
        x_global = [p.detach().clone().cpu() for p in self.model.parameters()]

        lr    = float(config.get("lr", self.default_lr))
        epochs = int(config.get("epochs", self.local_epochs))
        wd    = float(config.get("weight-decay", 0))
        clip  = float(config.get("clip-grad-norm", 5.0))

        # 2) Receive server control variate c from config (JSON-serialized)
        if "scaffold_c" in config:
            c_arrays = _deserialize_ndarrays(config["scaffold_c"])
            c_list = [
                torch.tensor(a, device=DEVICE, dtype=torch.float32)
                for a in c_arrays
            ]
        else:
            # Round 1 fallback: server hasn't sent c yet → treat as zeros
            c_list = [
                torch.zeros_like(p, device=DEVICE)
                for p in self.model.parameters()
            ]

        # 3) Load ci from persistent state (replaces self.ci check)
        self.ci = self._load_ci()
        if self.ci is None:
            print(f"[Client {self.cid}] ⚠️  ci initialized to zeros (first round)")
            self.ci = [torch.zeros_like(p, device=DEVICE) for p in self.model.parameters()]
        else:
            print(f"[Client {self.cid}] ✅ ci loaded from persistent state")

        # SCAFFOLD uses vanilla SGD (no momentum) so the math stays exact
        # (momentum would shift the effective gradient and break the variance-reduction proof)
        opt = optim.SGD(self.model.parameters(), lr=lr, weight_decay=0)

        # Count total local steps K = epochs × batches (used for c_i update)
        num_batches = len(self.train_loader)
        K = max(1, epochs * num_batches)

        # ── SCAFFOLD Option II c_i update ──────────────────────────────────
        # c_i_new = c_i - c + (x - y) / (K * lr)
        # With full-batch (K=1) and small lr, the correction explodes.
        # Clamp the correction term to avoid divergence.
        MAX_CI_NORM = 1.0  # tune if needed

        fit_total   = 0
        fit_correct = 0
        fit_ce_sum  = 0.0

        self.model.train()
        for _ in range(epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                opt.zero_grad()
                logits = self.model(xb)
                ce = self.crit(logits, yb)
                ce.backward()

                # Clip BEFORE SCAFFOLD correction
                if clip and clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)

                # SCAFFOLD correction applied after clipping
                with torch.no_grad():
                    for p, ci_t, c_t in zip(self.model.parameters(), self.ci, c_list):
                        if p.grad is not None:
                            p.grad.add_(c_t - ci_t)

                opt.step()

                bs = xb.size(0)
                fit_total   += bs
                fit_correct += (logits.argmax(1) == yb).sum().item()
                fit_ce_sum  += ce.item() * bs

        # 4) Control variate update (SCAFFOLD Option II):
        #    c_i_new = c_i - c + (1 / K*lr) * (x - y_i)
        #    delta_ci = c_i_new - c_i  →  sent to server for aggregation
        y_local = [p.detach().clone().cpu() for p in self.model.parameters()]

        delta_ci: list[np.ndarray] = []
        new_ci:   list[torch.Tensor] = []

        with torch.no_grad():
            for ci_t, c_t, x_t, y_t in zip(self.ci, c_list, x_global, y_local):
                # c_i_new = c_i - c + (x - y) / (K * lr)
                correction = (x_t - y_t) / (K * lr)
                # Clamp correction norm per tensor
                norm = correction.norm()
                if norm > MAX_CI_NORM:
                    correction = correction * (MAX_CI_NORM / norm)
                ci_new = ci_t.cpu() - c_t.cpu() + correction
                new_ci.append(ci_new.to(DEVICE))
                delta_ci.append((ci_new - ci_t.cpu()).numpy())

        # Persist updated c_i for next round
        self.ci = new_ci
        self._save_ci(new_ci)  # ← ADD THIS LINE

        fit_metrics = {
            "fit_loss":     fit_ce_sum / max(1, fit_total),
            "fit_accuracy": fit_correct / max(1, fit_total),
            # Send Δc_i to server so it can update the global control variate c
            "scaffold_delta_ci": _serialize_ndarrays(delta_ci),
        }

        return self.get_parameters({}), len(self.train_loader.dataset), fit_metrics

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
            json.dumps([0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])  # default grid if not provided
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
    
    # ── SCAFFOLD persistence helpers (uses Flower context.state) ────────────
    _CI_STATE_KEY = "scaffold_ci"

    def _save_ci(self, ci: list[torch.Tensor]) -> None:
        """Persist c_i into Flower's built-in per-client state (no disk I/O)."""
        arrays = [t.detach().cpu().numpy().astype(np.float32) for t in ci]
        serialized = _serialize_ndarrays(arrays)  # pickle+base64, shape-safe
        record = ConfigRecord({self._CI_STATE_KEY: serialized})
        self.context.state.config_records[self._CI_STATE_KEY] = record

    def _load_ci(self) -> list[torch.Tensor] | None:
        """Load c_i from Flower's per-client state. Returns None on first round."""
        if self._CI_STATE_KEY not in self.context.state.config_records:
            return None
        record = self.context.state.config_records[self._CI_STATE_KEY]
        serialized = record[self._CI_STATE_KEY]
        arrays = _deserialize_ndarrays(serialized)  # restores exact shapes
        return [
            torch.tensor(a, device=DEVICE, dtype=torch.float32)
            for a in arrays
        ]

    def set_parameters(self, parameters):
        keys  = list(self.model.state_dict().keys())
        state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state, strict=True)

    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]


def client_fn(context: Context):
    cid = context.node_config.get("partition-id", context.node_id)
    rc  = dict(context.run_config)
    return FlowerClient(str(cid), rc, context).to_client()  # pass context


app = ClientApp(client_fn=client_fn)














