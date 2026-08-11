# Federated Edge Learning — Projektübersicht

Dieses Repository enthält Experimente und Hilfsprogramme für föderiertes Lernen (Federated Learning) mit Fokus auf medizinische Screening-Aufgaben (z. B. Diabetes-Erkennung). Das Projekt verwendet PyTorch für Modelle und Flower für die föderierte Orchestrierung.

Kurz: Daten bleiben lokal bei Clients, nur Modell-Updates werden zentral aggregiert.

## Inhalte dieser README
- **Kurzbeschreibung & Ziele**
- **Voraussetzungen & Installation**
- **Schnellstart: Daten, Splits, Training, Evaluation, Plots**
- **Projektstruktur**
- **Häufige Befehle**

## Voraussetzungen
- Python 3.10+ (virtuelle Umgebung empfohlen)
- pip
- Optionale: Docker (für Flower-Deployments)

Installation (empfohlen in einem virtualenv):

```bash
# virtuelles Environment erstellen
python -m venv .venv
source .venv/bin/activate
# Abhängigkeiten installieren
pip install -U pip
pip install -r requirements.txt
```

Hinweis: In `pyproject.toml` / `requirements.txt` sind die genutzten Bibliotheken (PyTorch, Flower, numpy, pandas, scikit-learn, matplotlib, etc.) aufgeführt.

## Schnellstart — typische Abläufe

1) Daten vorbereiten

```bash
# Beispiel: CSV → Parquet + Normalisierungsstatistiken
python federated_learning/tools/prepare_data.py \
  --csv <input.csv> \
  --parquet data/diabetes.parquet \
  --stats data/norm_stats.json
```

2) Client-Partitionen erzeugen (z. B. Dirichlet non-IID)

```bash
python federated_learning/tools/make_splits.py \
  --parquet data/diabetes.parquet \
  --stats data/norm_stats.json \
  --out splits_dirichlet_10_a03.json \
  --num-partitions 10 --mode dirichlet --alpha 0.3
```

3) Föderiertes Training starten

```bash
# Standard: Flower starten bzw. eigene Start-Skripte benutzen
flwr run
# oder siehe scripts / server_app.py und client_app.py für lokale Tests
```

4) Beste Runde / Schwellenwert auswerten (Testset)

```bash
python federated_learning/tools/final_test_evaluation_with_val_threshold.py \
  --result-json result/<...>/run_1.json \
  --parquet data/diabetes.parquet \
  --stats data/norm_stats.json \
  --output final_evaluation
```

5) Grafiken erzeugen

```bash
python plot_results.py --root result --out plots_out
```

## Wichtige Konventionen & Ordner
- `data/` — vorbereitete Datendateien (`.parquet`, `norm_stats.json`, Splits)
- `federated_learning/` — zentrale Paketlogik (Server, Client, Daten-Tools, Plotting)
- `result/` — Run-Resultate, Checkpoints, Metriken
- `plots/` — generierte Visualisierungen

## Wichtige Skripte (Kurz)
- `federated_learning/tools/prepare_data.py` — CSV → Parquet + Normalisierung
- `federated_learning/tools/make_splits.py` — Partitionierung (IID/Dirichlet)
- `federated_learning/tools/final_test_evaluation_with_val_threshold.py` — Test-Evaluation mit gewählt. Threshold
- `federated_learning/client_app.py` — Client-Trainingslogik
- `federated_learning/server_app.py` — Server/Aggregationslogik
- `plot_results.py`, `plot_*` — Verschiedene Plot-Skripte zur Auswertung

## Hinweise zur Methodik (kurz)
- Class weighting, FedAvg, FedProx und gradient clipping werden genutzt, um class imbalance und non-IID-Streuung zu adressieren.
- Multi-threshold-Validierung: Clients evaluieren mehrere Schwellen; Server aggregiert Verwirrungsmatrix-Zählungen und wählt Schwellen nach klinisch motivierten Kriterien (Recall-vorrangig).

## Beispielbefehle zum Debugging
- Alle Skripte mit `--help` aufrufen, um Optionen zu sehen, z. B.:

```bash
python federated_learning/tools/make_splits.py --help
python federated_learning/client_app.py --help
```

## Ergebnisformat
- Checkpoints: `result/.../model_round_X.pt`
- Metriken: `result/.../run_1.json` (rundenspezifische Metriken und ausgewählter Threshold)
- Plots: PNG / PDF in `plots/` oder im jeweiligen `result`-Unterordner

## Weiteres / Deployment
- Flower kann lokal, über Docker oder mit dem Deployment Engine betrieben werden. Siehe Flower-Dokumentation für TLS/Authentifizierung.

## Mitwirken
- Issue anlegen oder PR öffnen. Kurze Beschreibung und reproduzierbares Beispiel bitte beilegen.

## Lizenz
- Dieses Repository enthält Forschungs-Code. Lizenz/Disclaimer bitte im Repo-Root ergänzen falls gewünscht.

---

Wenn du möchtest, passe ich die README noch an (z. B. deutsche Übersetzung einzelner Skript-Hilfetexte, zusätzliche Beispiele oder eine kurze Quickstart-Anleitung für Docker/Flower). 
```

