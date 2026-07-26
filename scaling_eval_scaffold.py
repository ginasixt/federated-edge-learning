#!/usr/bin/env python3
"""
First Script!!!
Als erstes, muss aber davor noch distrubution file ausüben

SCAFFOLD Scaling-Study Evaluation
=================================

Für jeden Skalierungspunkt (Anzahl Clients) und jeden der 5 Runs wird jedes
gespeicherte Runden-Modell auf dem Validierungsset ausgewertet.

Bei den SCAFFOLD-Split-Dateien liegt das vollständige Validierungsset unter:
    split_data["val"]["0"]

Der JSON-Schlüssel ist ausdrücklich der String "0". Client 0 wird daher nicht
über einen Wahrheitswert geprüft und auch nicht versehentlich als fehlend oder
leer interpretiert.

Pro Run wird das jeweils beste Modell nach jeder Metrik ermittelt:
  - höchste ROC-AUC
  - höchste PR-AUC (Average Precision)
  - niedrigster Loss (CrossEntropy mit Class-Weights, wie im Training)

Die Gewinner-Checkpoints werden nach
  <SCAFFOLD>/best<Metric>/run_<r>/
kopiert und eine Zusammenfassung als JSON + CSV geschrieben.

ROC-AUC und Average Precision werden direkt aus den Modellwahrscheinlichkeiten
und Labels berechnet. Für die Gewinner-Checkpoints werden außerdem die
vollständigen ROC- und Precision-Recall-Kurven gespeichert.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# --- deine echten Projekt-Bausteine -----------------------------------------
from federated_learning.client_app import MLP
from federated_learning.task import load_client_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CONFIG
# ============================================================================
# Wurzel, unter der die Skalierungs-Splits liegen.
RESULT_ROOT = Path("result/splits_iid_scaling")

# Wo liegen die Split-JSONs mit val["0"] und den übrigen Client-Splits?
SPLIT_ROOT = Path("splits_iid_scaling")

DATA_PARQUET = "data/diabetes_normalized.parquet"
NORM_STATS = "data/norm_stats.json"

STRATEGY = "SCAFFOLD"          # Unterordner mit den Runs
N_RUNS = 5                    # all_rounds_run_1 .. all_rounds_run_5
IN_DIM = 21                   # Feature-Dimension (server_app: model_dim = 21)
BATCH_SIZE = 256             # Batchgröße der nachträglichen Val-Evaluation
USE_FOCAL = False            # SCAFFOLD-Training nutzte CrossEntropy
FOCAL_GAMMA = 2.0
VAL_CLIENT_ID = "0"          # JSON-Schlüssel: vollständiges Val-Set liegt bei Client 0

# Alle Skalierungspunkte (Anzahl Clients)
SCALING_POINTS: List[int] = [
    2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384,
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
    return SPLIT_ROOT / f"splits_iid_{n_clients}_clients_adjusted.json"


def scaling_dir(n_clients: int) -> Path:
    """result/.../splits_iid_<n>_clients.json/<STRATEGY>"""
    return RESULT_ROOT / f"splits_iid_{n_clients}_clients.json" / STRATEGY


def build_val_loader(n_clients: int) -> Tuple[Optional[DataLoader], Optional[torch.Tensor]]:
    """
    Baut den Validation-Loader für einen Skalierungspunkt aus val["0"].

    In den SCAFFOLD-Split-Dateien liegt das gesamte Validation Set im Eintrag
    split_data["val"]["0"]. JSON-Objektschlüssel werden als Strings geladen;
    deshalb wird der Schlüssel "0" ausdrücklich geprüft und anschließend direkt
    indiziert. So kann Client 0 nicht durch einen falschen Truthiness-Check oder
    durch die Verwendung des Integer-Schlüssels 0 als leer behandelt werden.
    """
    sp = split_json_path(n_clients)
    if not sp.exists():
        print(f"   ⚠️  Split-Datei fehlt: {sp}")
        return None, None

    with open(sp, encoding="utf-8") as f:
        split_data = json.load(f)

    val_clients = split_data.get("val")
    if not isinstance(val_clients, dict):
        print(f"   ⚠️  Kein gültiger 'val'-Abschnitt in {sp.name}")
        return None, None

    # Wichtig: Nach json.load() lautet der Client-Schlüssel "0", nicht 0.
    # Wir prüfen die Existenz des Schlüssels und NICHT den Wahrheitswert der ID.
    if VAL_CLIENT_ID not in val_clients:
        available = ", ".join(sorted(str(key) for key in val_clients.keys())) or "keine"
        print(
            f"   ⚠️  Validation-Client '{VAL_CLIENT_ID}' fehlt in {sp.name}; "
            f"vorhandene Val-Clients: {available}"
        )
        return None, None

    val_row_ids = val_clients[VAL_CLIENT_ID]
    if not isinstance(val_row_ids, list):
        print(
            f"   ⚠️  split_data['val']['{VAL_CLIENT_ID}'] ist keine Liste "
            f"in {sp.name}"
        )
        return None, None

    if len(val_row_ids) == 0:
        print(
            f"   ⚠️  Validation-Client '{VAL_CLIENT_ID}' enthält keine Row-IDs "
            f"in {sp.name}"
        )
        return None, None

    # task.py stellt keine Funktion load_centralized_val() bereit.
    # Stattdessen verwenden wir die vorhandene load_client_data()-Funktion.
    # Die Trainingsliste bleibt absichtlich leer; alle IDs aus val["0"] werden
    # dadurch ausschließlich als Validation-Daten zurückgegeben.
    (
        _X_train_unused,
        _y_train_unused,
        X_val,
        y_val,
        class_weights,
    ) = load_client_data(
        parquet_path=DATA_PARQUET,
        stats_path=NORM_STATS,
        train_row_ids=[],
        val_row_ids=val_row_ids,
    )

    # Explizite Dtypes vermeiden stille Abweichungen zwischen Split-Dateien.
    X_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_val, dtype=torch.long)
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

    val_ds = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"   ✅ Val aus Client '{VAL_CLIENT_ID}' geladen: "
        f"{X_val.shape[0]} samples, {X_val.shape[1]} features"
    )
    return val_loader, class_weights

def make_criterion(class_weights: torch.Tensor) -> nn.Module:
    """
    Loss wie im Training.

    Für CrossEntropy wird reduction="sum" verwendet. In evaluate_model() wird
    anschließend durch die Summe der tatsächlich angewandten Klassengewichte
    geteilt. Das entspricht exakt der globalen weighted-mean-Reduktion von
    CrossEntropyLoss und ist unabhängig von der Batch-Aufteilung.
    """
    if USE_FOCAL:
        from federated_learning.client_app import FocalLoss
        return FocalLoss(alpha=class_weights.to(DEVICE), gamma=FOCAL_GAMMA)
    return nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), reduction="sum")


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, crit: nn.Module
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Ein Durchlauf über das Val-Set. Sammelt probs/y und berechnet loss,
    roc_auc, pr_auc. Gibt probs/y mit zurück, damit die vollständigen
    ROC-/PR-Kurven für das Gewinner-Modell aus EXAKT denselben Werten
    berechnet werden können (kein zweiter Val-Durchlauf).
    """
    model.eval()
    total_loss = 0.0
    loss_normalizer = 0.0
    n_samples = 0
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)

        if isinstance(crit, nn.CrossEntropyLoss) and crit.reduction == "sum":
            # Bei gewichteter CE dividiert reduction="mean" nicht durch die
            # Samplezahl, sondern durch die Summe der Zielklassen-Gewichte.
            total_loss += float(loss.item())
            if crit.weight is None:
                loss_normalizer += float(yb.numel())
            else:
                loss_normalizer += float(crit.weight[yb].sum().item())
        else:
            # Fallback für eine ggf. aktivierte FocalLoss mit mean-Reduktion.
            total_loss += float(loss.item()) * xb.size(0)
            loss_normalizer += float(xb.size(0))

        n_samples += xb.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())

    if not probs_all:
        raise ValueError("Der Validierungs-DataLoader enthält keine Samples.")

    probs = np.concatenate(probs_all).astype(np.float64, copy=False)
    y = np.concatenate(y_all).astype(np.int64, copy=False)
    avg_loss = total_loss / max(loss_normalizer, np.finfo(float).eps)

    if len(np.unique(y)) < 2:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y, probs))
        pr_auc = float(average_precision_score(y, probs))

    scalars = {
        "loss": float(avg_loss),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_samples": int(n_samples),
    }
    return scalars, probs, y

def _json_float(value: float) -> Optional[float]:
    """Konvertiert nicht-endliche Werte (z. B. ROC-Threshold inf) zu JSON-null."""
    value = float(value)
    return value if np.isfinite(value) else None


def compute_curves(probs: np.ndarray, y: np.ndarray) -> Dict:
    """
    Berechnet vollständige ROC- und Precision-Recall-Kurven aus denselben
    Wahrscheinlichkeiten, die auch für die Modellselektion verwendet werden.

    PR-AUC wird für die Selektion als Average Precision (AP) definiert. Zusätzlich
    wird die trapezoidale Fläche unter der gespeicherten PR-Kurve ausgegeben,
    weil AP und trapezoidale PR-AUC mathematisch nicht identisch sind.
    """
    n_samples = int(len(y))
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    if len(np.unique(y)) < 2:
        return {
            "roc_auc": None,
            "roc_auc_from_curve": None,
            "pr_auc": None,
            "pr_average_precision": None,
            "pr_auc_trapezoidal": None,
            "positive_prevalence": (
                float(n_positive / n_samples) if n_samples else None
            ),
            "roc_curve": {"fpr": [], "tpr": [], "thresholds": []},
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
            "n_samples": n_samples,
            "n_positive": n_positive,
            "n_negative": n_negative,
        }

    fpr, tpr, roc_thr = roc_curve(y, probs, pos_label=1)
    roc_auc_score_value = float(roc_auc_score(y, probs))
    roc_auc_curve_value = float(auc(fpr, tpr))

    precision, recall, pr_thr = precision_recall_curve(y, probs, pos_label=1)
    pr_average_precision = float(average_precision_score(y, probs))
    # precision_recall_curve liefert Recall absteigend; für eine explizit
    # aufsteigende x-Achse werden beide Arrays vor der Integration umgedreht.
    pr_auc_trapezoidal = float(auc(recall[::-1], precision[::-1]))

    if not np.isclose(roc_auc_score_value, roc_auc_curve_value, rtol=1e-10, atol=1e-12):
        print(
            "      ⚠️  ROC-AUC aus roc_auc_score und aus der gespeicherten "
            "Kurve stimmen nicht überein."
        )

    return {
        # Kompatibler Schlüssel: roc_auc ist die für die Auswahl verwendete AUC.
        "roc_auc": roc_auc_score_value,
        "roc_auc_from_curve": roc_auc_curve_value,
        # Kompatibler Schlüssel: pr_auc ist bewusst Average Precision (AP).
        "pr_auc": pr_average_precision,
        "pr_average_precision": pr_average_precision,
        "pr_auc_trapezoidal": pr_auc_trapezoidal,
        "pr_auc_selection_method": "average_precision_score",
        "positive_prevalence": float(n_positive / n_samples),
        "roc_curve": {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
            # sklearn kann am Anfang +inf liefern. JSON-null bewahrt die
            # Punktausrichtung, ohne ungültiges Infinity in JSON zu schreiben.
            "thresholds": [_json_float(v) for v in roc_thr],
            "length_relation": "len(fpr) = len(tpr) = len(thresholds)",
        },
        "pr_curve": {
            "precision": [float(v) for v in precision],
            "recall": [float(v) for v in recall],
            "thresholds": [float(v) for v in pr_thr],
            "length_relation": (
                "len(precision) = len(recall) = len(thresholds) + 1; "
                "the final precision/recall point has no threshold"
            ),
        },
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def discover_round_models(run_dir: Path) -> List[Tuple[int, Path]]:
    """
    Findet alle Runden-Modelle in einem Run-Ordner.

    Die Run-Zugehörigkeit ergibt sich AUSSCHLIESSLICH aus dem Ordner,
    NICHT aus dem Dateinamen. Unterstützt werden durch process_run() sowohl
    all_rounds_run_<r> als auch das ältere Schema all_rounds_<r>. Der Suffix
    im Dateinamen
    ist bedeutungslos, also liegen z.B. in all_rounds_run_3/ evtl.:
        model_round_12_run_3.pt  oder  model_round_12_run_1.pt  oder  model_round_12.pt
    Alle gehören zu Run 3, weil sie in dessen Ordner liegen.

    Erkannt wird jede Datei, die 'model_round_<N>' enthält und auf '.pt' endet,
    egal was zwischen <N> und .pt steht. <N> ist die Rundennummer.
    """
    if not run_dir.exists():
        return []

    # Tolerantes Matching: Suffix egal, Groß-/Kleinschreibung egal,
    # führende/abschließende Whitespaces (auch NBSP) egal.
    # iterdir() statt glob(): ein unsichtbares Zeichen im Namen kann so
    # nicht heimlich Dateien aussortieren.
    pattern = re.compile(r"model_round_(\d+).*\.pt\s*$", re.IGNORECASE)

    by_round: Dict[int, Path] = {}
    for p in sorted(run_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.strip()
        m = pattern.search(name)
        if not m:
            continue
        round_num = int(m.group(1))

        if round_num in by_round:
            print(
                f"      ⚠️  Mehrere Dateien für Runde {round_num} in "
                f"{run_dir.name}: behalte '{by_round[round_num].name}', "
                f"ignoriere '{p.name}'"
            )
            continue
        by_round[round_num] = p

    found = sorted(by_round.items(), key=lambda t: t[0])
    return found



def discover_run_models(base: Path, run_tag: int) -> Tuple[List[Tuple[int, Path]], List[Path]]:
    """
    Sucht einen Run in beiden verwendeten Ordnerschemata:

      1. all_rounds_run_<r>  (regulär)
      2. all_rounds_<r>      (älteres 16k-Schema)

    Falls beide Ordner existieren, werden eindeutige Runden zusammengeführt.
    Bei doppelten Rundennummern hat das reguläre Schema Priorität.
    """
    candidates = [
        base / f"all_rounds_run_{run_tag}",
        base / f"all_rounds_{run_tag}",
    ]
    existing_dirs = [path for path in candidates if path.is_dir()]

    by_round: Dict[int, Path] = {}
    for run_dir in existing_dirs:
        for round_num, checkpoint in discover_round_models(run_dir):
            if round_num in by_round:
                print(
                    f"      ⚠️  Runde {round_num} existiert in mehreren Run-Ordnern: "
                    f"behalte '{by_round[round_num]}', ignoriere '{checkpoint}'"
                )
                continue
            by_round[round_num] = checkpoint

    return sorted(by_round.items(), key=lambda item: item[0]), existing_dirs


def load_model(checkpoint_path: Path) -> nn.Module:
    model = MLP(in_dim=IN_DIM).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)

    # Unterstützt sowohl direkt gespeicherte state_dicts als auch häufige
    # Wrapper-Formate aus Trainings-Checkpoints.
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

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
    models, run_dirs = discover_run_models(base, run_tag)

    if not models:
        searched = ", ".join(
            str(path) for path in (
                base / f"all_rounds_run_{run_tag}",
                base / f"all_rounds_{run_tag}",
            )
        )
        print(f"   [Run {run_tag}] keine Modelle gefunden; geprüft: {searched}")
        return {
            "run": run_tag,
            "run_directories": [str(path) for path in run_dirs],
            "n_models": 0,
            "best": {},
            "per_round": [],
        }

    used_dirs = ", ".join(path.name for path in run_dirs)
    print(
        f"   [Run {run_tag}] {len(models)} Runden gefunden in "
        f"{used_dirs} -> evaluiere ..."
    )

    per_round: List[Dict] = []
    probs_by_round: Dict[int, np.ndarray] = {}
    y_by_round: Dict[int, np.ndarray] = {}
    for round_num, ckpt in models:
        model = load_model(ckpt)
        res, probs, y = evaluate_model(model, val_loader, crit)   # <-- 3 Rückgabewerte!
        res["round"] = round_num
        res["checkpoint"] = str(ckpt)
        per_round.append(res)
        probs_by_round[round_num] = probs
        y_by_round[round_num] = y

    best_summary: Dict[str, Dict] = {}
    for mname, (direction, key) in METRICS.items():
        best = pick_best(per_round, key, direction)
        if best is None:
            print(f"      ⚠️  Keine gültige Metrik '{mname}' (alle NaN?)")
            continue

        dest_dir = base / f"best{mname}" / f"run_{run_tag}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Bei erneutem Ausführen soll genau ein Gewinner-Checkpoint im Ziel
        # liegen; alte model_round_*.pt-Dateien werden deshalb entfernt.
        for old_checkpoint in dest_dir.iterdir():
            if old_checkpoint.is_file() and re.search(
                r"model_round_(\d+).*\.pt\s*$",
                old_checkpoint.name.strip(),
                re.IGNORECASE,
            ):
                old_checkpoint.unlink()

        src = Path(best["checkpoint"])
        dst_model = dest_dir / src.name
        shutil.copy2(src, dst_model)

        # ✅ Kurven NEU berechnen für das Gewinner-Modell (aus denselben probs/y)
        best_round = best["round"]
        curves = compute_curves(probs_by_round[best_round], y_by_round[best_round])
        curves_payload = {
            "scaling_point": n_clients,
            "run": run_tag,
            "metric": mname,
            "best_round": best_round,
            "loss": best["loss"],
            **curves,
            "source_checkpoint": best["checkpoint"],
        }
        with open(dest_dir / "curves.json", "w") as f:
            json.dump(curves_payload, f, indent=2)

        info = {
            "scaling_point": n_clients,
            "run": run_tag,
            "metric": mname,
            "direction": direction,
            "best_round": best_round,
            "roc_auc": best["roc_auc"],
            "pr_auc": best["pr_auc"],
            "pr_auc_definition": "average_precision_score",
            "loss": best["loss"],
            "n_samples": best["n_samples"],
            "source_checkpoint": best["checkpoint"],
            "copied_to": str(dst_model),
            "curves_file": str(dest_dir / "curves.json"),
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
        "run_directories": [str(path) for path in run_dirs],
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
            {
                "run": rs["run"],
                "run_directories": rs.get("run_directories", []),
                "n_models": rs["n_models"],
                "best": rs["best"],
            }
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
