#!/usr/bin/env python3
"""
Scaling-Study Evaluation
========================

Für jeden Skalierungspunkt (Anzahl Clients) und jeden der 5 Runs wird jedes
gespeicherte Runden-Modell auf dem ZENTRALISIERTEN Validierungsset ausgewertet.

Pro Run wird dann das jeweils beste Modell nach jeder Metrik ermittelt:
  - höchste ROC-AUC
  - höchste PR-AUC (Average Precision)
  - niedrigster Loss   (CrossEntropy mit Class-Weights, exakt wie im Training)

Die Gewinner-Checkpoints werden nach
  <FedAdam>/best<Metric>/run_<r>/  kopiert
und eine Zusammenfassung als JSON + CSV geschrieben.

WICHTIG - unterscheidet sich bewusst vom alten eval_batch.py:
  * Val-Set ist jetzt CENTRALIZED  -> split_data["centralized_val_row_ids"]
    (server_app.py: _load_centralized_val + task.load_centralized_val).
    Das alte eval_batch.py sammelte noch pro-Client Val-Rows -> veraltet.
  * evaluate_multi_threshold() liefert auc=0.0 (sklearn auskommentiert),
    daher werden ROC-AUC und PR-AUC hier direkt aus probs/y berechnet.
  * Loss = CrossEntropyLoss(weight=class_weights) - identisch zum Training.

Nutzt die ECHTEN Bausteine deines Projekts:
  from federated_learning.client_app import MLP
  from federated_learning.task       import load_centralized_val
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# --- deine echten Projekt-Bausteine -----------------------------------------
from federated_learning.client_app import MLP
from federated_learning.task import load_centralized_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CONFIG
# ============================================================================
# Wurzel, unter der die Skalierungs-Splits liegen.
RESULT_ROOT = Path("result/splits_iid_scaling")

# Wo liegen die Split-JSONs (für centralized_val_row_ids + Parquet-Filter)?
SPLIT_ROOT = Path("splits_iid_scaling")

DATA_PARQUET = "data/diabetes_normalized.parquet"
NORM_STATS = "data/norm_stats.json"

STRATEGY = "FedAdam"          # Unterordner mit den Runs
N_RUNS = 5                    # all_rounds_run_1 .. all_rounds_run_5
IN_DIM = 21                   # Feature-Dimension (server_app: model_dim = 21)
BATCH_SIZE = 256             # wie server-seitige Val (server_app: batch_size=256)
USE_FOCAL = False            # Training nutzte CrossEntropy (FedAdam-Pfad)
FOCAL_GAMMA = 2.0

# Alle Skalierungspunkte (Anzahl Clients)
SCALING_POINTS: List[int] = [
    2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
]

# Metriken: name -> (Richtung, wie aus einem Eval-Result der Vergleichswert kommt)
#   "max" = größer ist besser, "min" = kleiner ist besser
METRICS: Dict[str, Tuple[str, str]] = {
    "ROC":  ("max", "roc_auc"),   # -> Ordner best<...> = bestROC
    "PRROC": ("max", "pr_auc"),   # PR-AUC / Average Precision -> bestPRROC
    "Loss": ("min", "loss"),      # -> bestLoss
}
# ============================================================================


def split_json_path(n_clients: int) -> Path:
    """Pfad zur Split-Datei eines Skalierungspunkts."""
    return SPLIT_ROOT / f"splits_iid_{n_clients}_clients.json"


def scaling_dir(n_clients: int) -> Path:
    """result/.../splits_iid_<n>_clients.json/<STRATEGY>"""
    return RESULT_ROOT / f"splits_iid_{n_clients}_clients.json" / STRATEGY


def build_val_loader(n_clients: int) -> Tuple[Optional[DataLoader], Optional[torch.Tensor]]:
    """
    Baut den zentralisierten Val-Loader für einen Skalierungspunkt.
    Nutzt centralized_val_row_ids aus der Split-Datei (wie der Server).
    """
    sp = split_json_path(n_clients)
    if not sp.exists():
        print(f"   ⚠️  Split-Datei fehlt: {sp}")
        return None, None

    with open(sp) as f:
        split_data = json.load(f)

    val_row_ids = split_data.get("centralized_val_row_ids", [])
    if not val_row_ids:
        print(f"   ⚠️  Keine centralized_val_row_ids in {sp.name}")
        return None, None

    # Exakt die Server-Ladefunktion -> gleiche Distribution, gleiche Labels
    X_val, y_val, class_weights = load_centralized_val(
        parquet_path=DATA_PARQUET,
        stats_path=NORM_STATS,
        val_row_ids=val_row_ids,
    )

    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   ✅ Val geladen: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    return val_loader, class_weights


def make_criterion(class_weights: torch.Tensor) -> nn.Module:
    """Loss identisch zum Training (FedAdam-Pfad -> CrossEntropy mit Weights)."""
    if USE_FOCAL:
        from federated_learning.client_app import FocalLoss
        return FocalLoss(alpha=class_weights.to(DEVICE), gamma=FOCAL_GAMMA)
    return nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, crit: nn.Module) -> Dict[str, float]:
    """
    Ein Durchlauf über das Val-Set. Sammelt probs/y und berechnet:
      loss    (gewichtete CE, wie Training),
      roc_auc (sklearn),
      pr_auc  (sklearn average_precision_score).
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        probs_all.append(probs.cpu().numpy())
        y_all.append(yb.cpu().numpy())

    probs = np.concatenate(probs_all)
    y = np.concatenate(y_all)
    avg_loss = total_loss / max(1, n_samples)

    # AUC-Metriken brauchen beide Klassen
    if len(np.unique(y)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y, probs))
        pr_auc = float(average_precision_score(y, probs))

    return {
        "loss": float(avg_loss),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_samples": int(n_samples),
    }


def discover_round_models(run_dir: Path, run_tag: int) -> List[Tuple[int, Path]]:
    """
    Findet alle model_round_<r>_run_<run>.pt in einem Run-Ordner.
    Rückgabe sortiert nach Rundennummer.
    """
    if not run_dir.exists():
        return []

    found: List[Tuple[int, Path]] = []
    for p in run_dir.glob(f"model_round_*_run_{run_tag}.pt"):
        stem = p.stem  # model_round_<r>_run_<run>
        try:
            # ..._round_<r>_run_<run>
            round_num = int(stem.split("_round_")[1].split("_run_")[0])
        except (IndexError, ValueError):
            print(f"      ⚠️  Kann Rundennummer nicht parsen: {p.name}")
            continue
        found.append((round_num, p))

    found.sort(key=lambda t: t[0])
    return found


def load_model(checkpoint_path: Path) -> nn.Module:
    model = MLP(in_dim=IN_DIM).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model


def pick_best(per_round: List[Dict], metric_key: str, direction: str) -> Optional[Dict]:
    """Wählt aus allen Runden-Results das beste nach einer Metrik (NaN ignoriert)."""
    valid = [r for r in per_round if not np.isnan(r[metric_key])]
    if not valid:
        return None
    if direction == "max":
        return max(valid, key=lambda r: r[metric_key])
    return min(valid, key=lambda r: r[metric_key])


def process_run(
    n_clients: int,
    run_tag: int,
    val_loader: DataLoader,
    crit: nn.Module,
) -> Dict:
    """
    Evaluiert alle Runden eines Runs und ermittelt die besten Modelle je Metrik.
    Kopiert die Gewinner in best<Metric>/run_<run>/ und gibt eine Zusammenfassung
    zurück.
    """
    base = scaling_dir(n_clients)
    run_dir = base / f"all_rounds_run_{run_tag}"
    models = discover_round_models(run_dir, run_tag)

    if not models:
        print(f"   [Run {run_tag}] keine Modelle in {run_dir}")
        return {"run": run_tag, "n_models": 0, "best": {}, "per_round": []}

    print(f"   [Run {run_tag}] {len(models)} Runden gefunden -> evaluiere ...")

    per_round: List[Dict] = []
    for round_num, ckpt in models:
        model = load_model(ckpt)
        res = evaluate_model(model, val_loader, crit)
        res["round"] = round_num
        res["checkpoint"] = str(ckpt)
        per_round.append(res)

    # Beste Modelle je Metrik ermitteln + kopieren
    best_summary: Dict[str, Dict] = {}
    for mname, (direction, key) in METRICS.items():
        best = pick_best(per_round, key, direction)
        if best is None:
            print(f"      ⚠️  Keine gültige Metrik '{mname}' (alle NaN?)")
            continue

        dest_dir = base / f"best{mname}" / f"run_{run_tag}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        src = Path(best["checkpoint"])
        dst_model = dest_dir / src.name
        shutil.copy2(src, dst_model)

        # Zusätzlich: kleine Metrik-Datei neben dem kopierten Modell
        info = {
            "scaling_point": n_clients,
            "run": run_tag,
            "metric": mname,
            "direction": direction,
            "best_round": best["round"],
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
            "loss": best["loss"],
            "n_samples": best["n_samples"],
            "source_checkpoint": best["checkpoint"],
            "copied_to": str(dst_model),
        }
        with open(dest_dir / "best_info.json", "w") as f:
            json.dump(info, f, indent=2)

        best_summary[mname] = info
        print(
            f"      ✅ best{mname}: Round {best['round']:>3} "
            f"(ROC={best['roc_auc']:.4f} PR={best['pr_auc']:.4f} Loss={best['loss']:.4f})"
        )

    return {
        "run": run_tag,
        "n_models": len(models),
        "best": best_summary,
        "per_round": per_round,
    }


def process_scaling_point(n_clients: int) -> Optional[Dict]:
    base = scaling_dir(n_clients)
    if not base.exists():
        print(f"⏭️  Skip {n_clients} clients - Ordner fehlt: {base}")
        return None

    print("\n" + "=" * 78)
    print(f"🔬 Skalierungspunkt: {n_clients} Clients   ({base})")
    print("=" * 78)

    val_loader, class_weights = build_val_loader(n_clients)
    if val_loader is None:
        print(f"⏭️  Skip {n_clients} clients - kein Val-Set")
        return None

    crit = make_criterion(class_weights)

    runs_summary = []
    for run_tag in range(1, N_RUNS + 1):
        runs_summary.append(process_run(n_clients, run_tag, val_loader, crit))

    # Alle Runden-Metriken dieses Skalierungspunkts als CSV ablegen
    csv_rows = ["scaling_point,run,round,roc_auc,pr_auc,loss,n_samples"]
    for rs in runs_summary:
        for r in rs["per_round"]:
            csv_rows.append(
                f"{n_clients},{rs['run']},{r['round']},"
                f"{r['roc_auc']:.6f},{r['pr_auc']:.6f},{r['loss']:.6f},{r['n_samples']}"
            )
    csv_path = base / "all_rounds_metrics.csv"
    csv_path.write_text("\n".join(csv_rows))

    summary = {
        "scaling_point": n_clients,
        "n_runs": N_RUNS,
        "runs": [
            {"run": rs["run"], "n_models": rs["n_models"], "best": rs["best"]}
            for rs in runs_summary
        ],
    }
    with open(base / "scaling_point_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    print("=" * 78)
    print("🚀 SCALING-STUDY EVALUATION")
    print(f"   Strategy : {STRATEGY}")
    print(f"   Runs     : {N_RUNS}")
    print(f"   Metriken : {', '.join(METRICS.keys())} (best<Metric>/run_<r>/)")
    print(f"   Device   : {DEVICE}")
    print("=" * 78)

    global_summary = []
    for n_clients in SCALING_POINTS:
        s = process_scaling_point(n_clients)
        if s is not None:
            global_summary.append(s)

    # Globale Zusammenfassung über alle Skalierungspunkte
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULT_ROOT / f"{STRATEGY}_scaling_summary.json"
    with open(out, "w") as f:
        json.dump(global_summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"✅ Fertig. {len(global_summary)} Skalierungspunkte ausgewertet.")
    print(f"   Globale Zusammenfassung: {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
