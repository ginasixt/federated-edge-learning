# Calibration & Risk Distribution Feature

## ✅ Was wurde implementiert?

Zwei neue Privacy-Safe Evaluations-Metriken für bessere Model-Beurteilung und Best-Round-Selection.

---

## 📊 1. **Calibration Plot**

**Was ist das?**
- Zeigt, wie gut die Modell-Vorhersagen **kalibriert** sind
- Perfekt kalibriert = Punkte liegen auf Diagonale (y=x)

**Wie funktioniert es?**

**Im Client (`evaluate_multi_threshold`):**
- Segregiere Predictions in **10 Bins** (z.B. [0.0-0.1], [0.1-0.2], ..., [0.9-1.0])
- Für jeden Bin berechne:
  - `n_i` = Anzahl Samples im Bin
  - `sum_pred_i` = Summe der vorhergesagten Wahrscheinlichkeiten
  - `sum_true_i` = Anzahl positiver Labels
- Sende nur diese **Summen** (DSGVO-safe! ✅)

**Im Server (`evaluate_metrics_aggregation_fn`):**
- Aggregiere alle Summen über alle Clients
- Berechne pro Bin:
  - `mean_pred_i = global_sum_pred_i / global_n_i` (was das Modell sagt)
  - `mean_obs_i = global_sum_true_i / global_n_i` (was wirklich passiert)
- Plotte: x-Achse = mean_pred, y-Achse = mean_obs

**Interpretation:**
- **Auf der Diagonale**: Gut kalibriert ✅
- **Unter der Diagonale**: Modell zu optimistisch (overconfident)
- **Über der Diagonale**: Modell zu pessimistisch

---

## 📈 2. **Risk Distribution Plot**

**Was ist das?**
- Zeigt die **Verteilung** der vorhergesagten Wahrscheinlichkeiten für beide Klassen
- Gute Modelle: Histogramme überlappen wenig (gute Trennung)

**Wie funktioniert es?**

**Im Client (`evaluate_multi_threshold`):**
- Segregiere Predictions in **20 Bins**
- Zähle Samples pro Bin, getrennt nach Outcome:
  - `hist_y0[i]` = Counts für y=0 (Healthy) im Bin i
  - `hist_y1[i]` = Counts für y=1 (Prediabetic/Diabetic) im Bin i
- Sende nur diese **Histogramm-Counts** (DSGVO-safe! ✅)

**Im Server (`evaluate_metrics_aggregation_fn`):**
- Aggregiere alle Histogramm-Counts über alle Clients
- Plotte: Nebeneinander-Balkendiagramm
  - Blau = y=0 Histogramm
  - Rot = y=1 Histogramm

**Interpretation:**
- **Geringe Überlappung**: Modell trennt Klassen gut ✅
- **Hohe Überlappung**: Modell hat Schwierigkeiten
- **Links viel y=0, rechts viel y=1**: Ideales Szenario

---

## 🔄 Workflow

### 1. **Während Training (automatisch)**

```bash
flwr run
```

Für jede **evaluierte Runde**:
- Client berechnet Calibration + Risk Distribution Metriken (lokal)
- Server aggregiert diese über alle Clients
- Plots werden **automatisch generiert** in:
  - `result/{split-path}/all_rounds/plots/calibration_round_{N}.png`
  - `result/{split-path}/all_rounds/plots/risk_distribution_round_{N}.png`

### 2. **Post-Training: Plots manuell generieren**

```python
from federated_learning.plotting.calibration_and_risk_plots import load_and_plot_from_json

# Lade aggregierte Metriken einer Runde
load_and_plot_from_json(
    json_path="result/.../round_75_run_1.json",
    output_dir="my_plots/",
    show=True
)
```

Oder **Kommandozeile**:
```bash
python calibration_and_risk_plots.py \
  result/splits_iid_scaling/splits_iid_16384_clients.json/all_rounds/round_75_run_1.json \
  output_plots/
```

---

## 🛡️ Privacy & Safety

✅ **Keine individuellen Patientendaten werden gesendet!**
- Nur aggregierte Summen/Counts pro Bin
- Keine einzelnen Predictions
- DSGVO-konform

✅ **Mathematisch sicher:**
- Binning aggregiert Details weg
- Summen können nicht zurückbrechnet werden zu Individuals

---

## 📋 Metriken in den Results

Nach jeder Evaluation findest du in `round_{N}_run_{tag}.json`:

```json
{
  "round": 75,
  "metrics": {
    "auc": 0.0,
    "all_thresholds": [...],
    
    "calibration": [
      {
        "bin_index": 0,
        "bin_edge_lower": 0.0,
        "bin_edge_upper": 0.1,
        "n_samples": 1250,
        "mean_predicted_prob": 0.05,
        "mean_observed_freq": 0.08
      },
      ...
    ],
    
    "risk_distribution": [
      {
        "bin_index": 0,
        "bin_edge_lower": 0.0,
        "bin_edge_upper": 0.05,
        "count_y0": 850,
        "count_y1": 12
      },
      ...
    ]
  }
}
```

---

## 🎯 Best Round Selection

Mit diesen Metriken kannst du jetzt besser auswählen:

1. **Stabilität**: Beobachte Calibration über Runden hinweg
   - Stabile Calibration = konsistentes Modell
   - Große Schwankungen = Overfitting/Instabilität

2. **Performance**: Schaue auf Risk Distribution
   - Je weniger Überlappung = je besser die Trennung

3. **Kombiniert**: Nutze beide Plots als **holistische Metrik**

---

## 🚀 Next Steps

1. Starte Training: `flwr run`
2. Beobachte Plots in `result/.../plots/`
3. Nutze diese für Round Selection (z.B. `select_best_round_screening.py`)
4. Evaluiere finalist Model auf echtem Test-Set

---

## 📞 Troubleshooting

**Problem**: Plots werden nicht generiert
- ✅ Check: Ist `matplotlib` installiert? (`pip list | grep matplotlib`)
- ✅ Check: Ist Evaluations-Config korrekt?

**Problem**: Leere Bins im Calibration Plot
- ✅ Normal! Bedeutet keine Samples in diesem Probability-Bereich
- ✅ Filter diese weg (done automatically)

**Problem**: Sehr hohe Überlappung in Risk Distribution
- ⚠️ Modell trennt Klassen nicht gut
- 💡 Überprüfe: Label Balance, Feature Quality, Model Architecture

