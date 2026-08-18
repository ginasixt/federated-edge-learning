# Federated Edge Learning — SCAFFOLD Scalability Study

This repository contains the experimental code used for the **SCAFFOLD scalability study** conducted as part of the bachelor thesis on federated edge learning for diabetes screening.

The project uses **PyTorch** for model training and **Flower (FLwr)** for federated orchestration and simulation.

The central objective of the study is to investigate how federated learning behaves when a **fixed global training dataset is distributed across an increasing number of clients**. Increasing the client count therefore does not introduce additional training data, but progressively fragments the same training set into smaller local client datasets.

For SCAFFOLD, the final scalability study evaluates:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512,
1,024, 2,048, 4,096, 8,192, 16,384 clients
```

Each scaling point is repeated five times.

SCAFFOLD extends FedAvg by introducing a **global server control variate** and a **persistent local control variate for each client**. These control variates modify the client gradients to reduce deviations between local and global optimization directions.

---

# Requirements and Installation

A Python virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

The Flower application itself also declares its main runtime dependencies in `pyproject.toml`, including Flower and PyTorch.

---

# Project Workflow

The complete SCAFFOLD workflow is:

```text
Raw BRFSS dataset
        │
        ▼
prepare_data.py
        │
        ▼
diabetes.parquet + norm_stats.json
        │
        ▼
normalize_and_add_weights.py
        │
        ▼
diabetes_normalized.parquet
        │
        ▼
create_iid_scaling_splits.py
        │
        ▼
IID client partitions
        │
        ▼
adjust_val_distribution.py
        │
        ▼
centralized validation client
        │
        ▼
SCAFFOLD training
(server_app.py + client_app.py + task.py)
        │
        ▼
scaling_eval_scaffold.py
        │
        ├── bestROC
        ├── bestPRROC
        └── bestLoss
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
final_test_set_eval_scaffold.py   evaluate-thr-dependent-scaffold.py
        │                               │
        ▼                               ▼
plot-thr-indep-scaffold.py       plot-thr-dep-scaffold.py
        │
        ▼
table_plot_scaffold.py

All saved round checkpoints
        │
        ▼
evaluate-training-dynamics-scaffold.py
        │
        ▼
plot-training-dynamics-scaffold.py
```

The validation set is used for configuration, checkpoint, and threshold selection. The test set is not used for these selections and is evaluated only after the corresponding validation-based decisions have been made.

---

# 1. Core Federated Learning Components

## `server_app.py`

```text
federated_learning/server_app.py
```

Implements the server-side SCAFFOLD strategy.

The custom `ScaffoldFedAvg` strategy extends Flower's standard FedAvg implementation. Model parameters are still aggregated through the usual sample-size-weighted FedAvg procedure, but the strategy additionally maintains the global SCAFFOLD control variate `c`.

During every training round, the server:

1. selects the participating clients,
2. sends the current global model,
3. sends the current global control variate `c`,
4. receives locally updated models,
5. aggregates the client models using FedAvg,
6. receives the client control-variate updates `Δc_i`,
7. updates the global control variate,
8. stores model checkpoints and evaluation results.

The global control variate is initialized with zeros. After a round, the server updates it using the sum of the participating clients' `Δc_i` values divided by the total client population.

The strategy also stores checkpoints under:

```text
result/<split-path>/SCAFFOLD/all_rounds_run_<run>/
```

and writes round-specific evaluation information as JSON files.

---

## `client_app.py`

```text
federated_learning/client_app.py
```

Implements the Flower client, the neural-network model, and the local SCAFFOLD optimization step.

The model is a multilayer perceptron with:

```text
21 input features
256-unit hidden layer + ReLU
128-unit hidden layer + ReLU
2 output logits
```

The hidden layers use Kaiming initialization.

For every selected client, `fit()`:

1. loads the current global model,
2. receives the server control variate `c`,
3. restores the client's persistent local control variate `c_i`,
4. trains locally using SGD,
5. applies gradient clipping,
6. applies the SCAFFOLD correction `c - c_i`,
7. updates the local model,
8. computes the updated local control variate,
9. stores `c_i` in Flower's persistent client state,
10. returns the updated model and `Δc_i` to the server.

Gradient clipping is applied **before** adding the SCAFFOLD correction term.

The client control variate follows the computationally cheaper SCAFFOLD Option II update.

---

## `task.py`

```text
federated_learning/task.py
```

Contains the data-loading utilities used by each Flower client.

For a given client, the script:

1. receives the client's training and validation row IDs,
2. loads only these rows from the normalized Parquet dataset,
3. separates features and labels,
4. retrieves the precomputed class weights from `norm_stats.json`,
5. creates PyTorch `DataLoader` objects.

The Parquet data are already standardized before training, so no normalization is performed inside `task.py`.

---

# 2. Data Preparation

## `prepare_data.py`

```text
federated_learning/tools/prepare_data.py
```

Downloads the binary BRFSS 2015 diabetes dataset and creates a reproducible global train/validation/test split.

The default split is:

```text
Training:    70%
Validation:  10%
Test:        20%
```

The split is stratified by the target class.

The script stores:

```text
data/diabetes.parquet
data/norm_stats.json
```

The Parquet file contains the feature values, target label, and stable `__row_id__` identifier.

`norm_stats.json` contains:

```text
train_idx
val_idx
test_idx
feature means
feature standard deviations
target-column name
```

Example:

```bash
python federated_learning/tools/prepare_data.py \
    --csv <input.csv> \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json
```

---

## `normalize_and_add_weights.py`

```text
federated_learning/tools/normalize_and_add_weights.py
```

Standardizes the dataset and calculates the class weights used during training.

It uses the previously created global train/validation/test split and does **not** create new splits.

Feature standardization is based exclusively on the mean and standard deviation calculated from the training subset.

The script also calculates global class weights from the training labels and applies the configurable positive-class boost.

Example:

```bash
python federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5
```

The output is:

```text
data/diabetes_normalized.parquet
```

and `norm_stats.json` is extended with fields such as:

```text
pos_weight
neg_weight
pos_weight_boost
train_pos_count
train_neg_count
```

---

# 3. Creating the IID Client Partitions

## `make_splits.py`

```text
federated_learning/tools/make_splits.py
```

Contains general partitioning helper functions.

The relevant function for the final scalability study is:

```python
iid_partitions(...)
```

which randomly shuffles the training observations and distributes them approximately evenly across the requested number of clients.

The file also contains Dirichlet partitioning functionality. This was used in earlier experiments but is **not part of the final IID scalability study**.

---

## `create_iid_scaling_splits.py`

```text
federated_learning/tools/create_iid_scaling_splits.py
```

Generates IID client partitions for increasing client counts.

Example:

```bash
python federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123
```

Training observations are randomly and approximately evenly distributed across clients.

Labels are **not used for the assignment**. Label statistics are calculated only after partitioning to characterize the resulting natural client-level variation.

The generated files follow the naming scheme:

```text
splits_iid_scaling/
├── splits_iid_2_clients.json
├── splits_iid_4_clients.json
├── splits_iid_8_clients.json
├── ...
└── splits_iid_16384_clients.json
```

Each file contains:

```json
{
  "train": {...},
  "val": {...},
  "meta": {...}
}
```

The total training dataset remains fixed across all scaling points.

---

## `create_scaling_splits.py`

```text
federated_learning/tools/create_scaling_splits.py
```

Creates Dirichlet-based non-IID scaling partitions.

This file belongs to earlier non-IID experiments and is **not required for reproducing the final IID SCAFFOLD scalability study**.

---

# 4. Centralizing the Validation Set

## `adjust_val_distribution.py`

```text
adjust_val_distribution.py
```

Modifies the generated scaling split files so that the complete validation dataset is assigned to a single validation client.

This reduces evaluation overhead during the large-client simulations and allows later validation-based checkpoint selection to use one complete centralized validation set.

The adjusted split files are subsequently used by `scaling_eval_scaffold.py`.

---

# 5. Configuring SCAFFOLD

The main Flower configuration is stored in:

```text
pyproject.toml
```

The Flower application entry points are:

```toml
[tool.flwr.app.components]
serverapp = "federated_learning.server_app:app"
clientapp = "federated_learning.client_app:app"
```

This means that:

```bash
flwr run .
```

starts the SCAFFOLD server and client implementations contained in these two files.

---

## Main SCAFFOLD Configuration

The relevant configuration is located under:

```toml
[tool.flwr.app.config]
```

The supplied configuration contains:

```toml
num-server-rounds = 81
fraction-fit = 0.5
fraction-evaluate = 1.0

local-epochs = 2

batch-size = 9000
lr = 1e-2
weight-decay = 1e-5
pos-weight-boost = 1.5
clip-grad-norm = 5.0
```

For the actual local SCAFFOLD update, the client uses vanilla SGD and explicitly sets optimizer weight decay to zero inside the client implementation.

The important SCAFFOLD-specific settings are therefore primarily:

```text
Learning rate:          1e-2
Local epochs:           2
Gradient clipping:      5.0
Positive-class boost:   1.5
```

The large configured batch size ensures that small local datasets are processed in a single batch at the highly fragmented scaling points.

---

## Ray Simulation Configuration

Ray resources are configured in `pyproject.toml`.

The supplied configuration uses:

```toml
num_cpus = 32
options.backend.client-resources.num-cpus = 1.0
options.backend.client-resources.num-gpus = 0.0
options.backend.max-workers = 32
```

These parameters control how many simulated clients can run concurrently.

They do **not** change the conceptual number of federated clients participating in a communication round.

The total simulated client population is controlled through:

```toml
options.num-supernodes = ...
```

---

# 6. Running SCAFFOLD

There are two main ways to start training.

## Single Flower Run

A single experiment can be started directly with:

```bash
flwr run .
```

This uses the current values in:

```text
pyproject.toml
```

including:

```text
split-path
number of rounds
fraction-fit
local epochs
learning rate
gradient clipping
Ray resources
```

For a different experiment, edit the corresponding values in `pyproject.toml` before starting the run.

---

## Scaling Experiments

The automated launcher is:

```text
federated_learning/tools/run_iid_scaling.sh
```

Run:

```bash
./federated_learning/tools/run_iid_scaling.sh
```

If necessary:

```bash
chmod +x federated_learning/tools/run_iid_scaling.sh
```

The script:

1. iterates over the configured client counts,
2. selects the corresponding IID split file,
3. updates `options.num-supernodes`,
4. calculates the required number of participating clients,
5. starts the Flower simulation,
6. repeats each scaling point multiple times,
7. stores run-specific logs,
8. stops and cleans Ray between experiments,
9. restores the original `pyproject.toml` afterward.

---

## Selecting Client Counts

At the beginning of the launcher:

```bash
CLIENT_COUNTS=(...)
```

determines which scaling points are executed.

For the complete SCAFFOLD thesis range, use:

```bash
CLIENT_COUNTS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
```

For a test run:

```bash
CLIENT_COUNTS=(4096)
```

---

## Selecting the Number of Runs

The final study uses:

```bash
RUNS_PER_SPLIT=5
```

For debugging:

```bash
RUNS_PER_SPLIT=1
```

---

## Run-Specific Flower Overrides

For every simulation, the launcher calls Flower using:

```bash
flwr run . --run-config "..."
```

and overrides parameters including:

```text
split-path
min-fit-clients
min-available-clients
min-evaluate-clients
num-server-rounds
run-tag
```

The corresponding split file is automatically selected as:

```text
splits_iid_scaling/splits_iid_<N>_clients.json
```

---

## Important Launcher Note

The currently supplied `run_iid_scaling.sh` contains several historical range-dependent settings.

For reproducing the final SCAFFOLD study, the launcher should be configured consistently for:

```text
2–16,384 clients
5 repeated runs
80 communication rounds
selected SCAFFOLD participation setting
```

before launching the complete reproduction.

---

# 7. Validation-Based Checkpoint Selection

## `scaling_eval_scaffold.py`

```text
scaling_eval_scaffold.py
```

This is the **first evaluation step after training**.

For every SCAFFOLD scaling point and each of the five runs, the script evaluates every saved communication-round checkpoint on the centralized validation set.

It independently selects:

```text
bestROC
    highest validation ROC-AUC

bestPRROC
    highest validation Average Precision

bestLoss
    lowest weighted validation loss
```

The selected checkpoints are copied to:

```text
result/splits_iid_scaling/
└── splits_iid_<N>_clients.json/
    └── SCAFFOLD/
        ├── bestROC/
        ├── bestPRROC/
        └── bestLoss/
```

The test set is not used during this step.

Run:

```bash
python scaling_eval_scaffold.py
```

---

# 8. Threshold-Independent Final Test Evaluation

## `final_test_set_eval_scaffold.py`

```text
final_test_set_eval_scaffold.py
```

Evaluates only the checkpoints previously selected on the validation set.

For every scaling point and run, the script evaluates:

```text
bestROC
bestPRROC
bestLoss
```

on the fixed centralized test set.

It calculates:

```text
weighted cross-entropy loss
ROC-AUC
Average Precision
ROC curve
precision-recall curve
test-set prevalence
```

The combined outputs are written to:

```text
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
```

including:

```text
all_test_results.csv
all_test_aggregate.csv
final_test_summary.json
test_set_info.json
```

Run:

```bash
python final_test_set_eval_scaffold.py
```

---

# 9. Threshold-Independent Figures

## `plot-thr-indep-scaffold.py`

```text
plot-thr-indep-scaffold.py
```

Creates the threshold-independent SCAFFOLD scalability figures.

The main absolute-performance figure contains:

```text
Panel A: ROC-AUC
Panel B: Average Precision
Panel C: Weighted loss
```

Each metric uses the checkpoint selected with the corresponding validation criterion.

The script also supports relative-change and run-stability visualizations.

Default input:

```text
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
    all_test_results.csv
```

Run:

```bash
python plot-thr-indep-scaffold.py
```

---

# 10. Run-to-Run Dispersion

## `table_plot_scaffold.py`

```text
table_plot_scaffold.py
```

Analyzes variation across the five repeated SCAFFOLD runs.

For each scaling point and metric, it calculates statistics such as:

```text
mean
standard deviation
coefficient of variation
minimum
maximum
maximum relative deviation from the mean
```

The analysis is performed for:

```text
ROC-AUC
Average Precision
Weighted loss
```

using the corresponding validation-selected checkpoints.

Typical outputs include:

```text
run_dispersion_table.csv
run_dispersion_summary_full.csv
run_dispersion_table.md
run_dispersion_text_summary.txt
run_dispersion_by_scaling.csv

table_run_to_run_dispersion.pdf
table_run_to_run_dispersion.png

figure_run_to_run_dispersion_by_scaling.pdf
figure_run_to_run_dispersion_by_scaling.png
```

Run:

```bash
python table_plot_scaffold.py
```

---

# 11. Threshold-Dependent Evaluation

## `evaluate-thr-dependent-scaffold.py`

```text
evaluate-thr-dependent-scaffold.py
```

Uses the checkpoint previously selected according to highest validation AP (`bestPRROC`).

For every scaling point and run:

1. validation predictions are generated,
2. two operating points are selected on validation,
3. the selected thresholds are transferred unchanged to the test set,
4. final threshold-dependent test metrics are calculated.

Two operating points are used.

### MCC-optimal

The validation threshold that maximizes MCC is selected.

### Minimum validation recall

A minimum validation recall requirement is specified before evaluation.

For the thesis:

```text
Recall >= 0.80
```

Among all eligible thresholds, validation specificity is maximized.

Run:

```bash
python evaluate-thr-dependent-scaffold.py --min-recall 0.80
```

The combined output is:

```text
result/splits_iid_scaling/final_threshold_analysis/SCAFFOLD/
    all_threshold_results.csv
```

---

## `plot-thr-dep-scaffold.py`

```text
plot-thr-dep-scaffold.py
```

Plots the previously generated threshold-dependent results.

It does **not** select a model, communication round, or threshold.

### MCC-optimal figure

```text
Panel A: validation-selected decision threshold
Panel B: test MCC
Panel C: test recall and specificity
```

### Fixed minimum-recall figure

```text
Panel A: validation-selected decision threshold
Panel B: test recall and specificity
```

Run:

```bash
python plot-thr-dep-scaffold.py
```

---

# 12. Training-Dynamics Analysis

## `evaluate-training-dynamics-scaffold.py`

```text
evaluate-training-dynamics-scaffold.py
```

Retrospectively evaluates all saved SCAFFOLD checkpoints on the centralized test set using Average Precision.

The analysis is descriptive only and does not alter training or model selection.

Two quantities are calculated.

### Time to near-best performance

The script determines the first evaluated communication round for which:

```text
test AP >= 0.99 × best observed test AP in the run
```

### Late-training trend

An ordinary least-squares regression is fitted to the AP values over the final configured communication-round window.

The output includes:

```text
all_round_test_ap.csv
training_dynamics_by_run.csv
training_dynamics_aggregate.csv
training_dynamics_summary.json
```

Run:

```bash
python evaluate-training-dynamics-scaffold.py
```

---

## `plot-training-dynamics-scaffold.py`

```text
plot-training-dynamics-scaffold.py
```

Visualizes the training-dynamics results.

The resulting figure contains:

```text
Panel A:
First evaluated round reaching 99% of the best observed test AP

Panel B:
Late-training AP slope per communication round
```

Run:

```bash
python plot-training-dynamics-scaffold.py
```

---

# 13. Complete Execution Order

A complete SCAFFOLD workflow is:

```bash
# 1. Prepare train / validation / test metadata
python federated_learning/tools/prepare_data.py \
    --csv <input.csv> \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json

# 2. Normalize features and calculate class weights
python federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5

# 3. Generate IID scaling splits
python federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123

# 4. Centralize validation data in the split files
python adjust_val_distribution.py

# 5. Run SCAFFOLD scaling experiments
./federated_learning/tools/run_iid_scaling.sh

# 6. Select checkpoints on validation
python scaling_eval_scaffold.py

# 7. Final threshold-independent test evaluation
python final_test_set_eval_scaffold.py

# 8. Generate threshold-independent figures
python plot-thr-indep-scaffold.py

# 9. Calculate run-to-run dispersion
python table_plot_scaffold.py

# 10. Select validation operating points and evaluate on test
python evaluate-thr-dependent-scaffold.py --min-recall 0.80

# 11. Generate threshold-dependent figures
python plot-thr-dep-scaffold.py

# 12. Evaluate training dynamics
python evaluate-training-dynamics-scaffold.py

# 13. Generate training-dynamics figure
python plot-training-dynamics-scaffold.py
```

A single Flower experiment can instead be started using:

```bash
flwr run .
```

---

# 14. Project File Overview

| File                                                    | Purpose                                                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `pyproject.toml`                                        | Flower application, SCAFFOLD hyperparameters, data paths, and Ray simulation resources               |
| `federated_learning/server_app.py`                      | Server-side SCAFFOLD strategy, FedAvg aggregation, global control-variate update, checkpoint storage |
| `federated_learning/client_app.py`                      | MLP, local SCAFFOLD training, client control variate, gradient correction and evaluation             |
| `federated_learning/task.py`                            | Loads client-specific normalized data and class weights                                              |
| `federated_learning/tools/prepare_data.py`              | Creates the fixed global train/validation/test split                                                 |
| `federated_learning/tools/normalize_and_add_weights.py` | Standardizes features and computes class weights                                                     |
| `federated_learning/tools/make_splits.py`               | General IID/Dirichlet partitioning helpers                                                           |
| `federated_learning/tools/create_iid_scaling_splits.py` | Creates the final IID scaling partitions                                                             |
| `federated_learning/tools/create_scaling_splits.py`     | Legacy/non-IID Dirichlet scaling utility                                                             |
| `adjust_val_distribution.py`                            | Moves the full validation set to one validation client                                               |
| `federated_learning/tools/run_iid_scaling.sh`           | Launches repeated Flower simulations across selected client counts                                   |
| `scaling_eval_scaffold.py`                              | Evaluates saved checkpoints on validation and selects best ROC/AP/loss checkpoints                   |
| `final_test_set_eval_scaffold.py`                       | Evaluates validation-selected checkpoints on the final test set                                      |
| `plot-thr-indep-scaffold.py`                            | Creates threshold-independent SCAFFOLD figures                                                       |
| `table_plot_scaffold.py`                                | Calculates run-to-run dispersion statistics and figures                                              |
| `evaluate-thr-dependent-scaffold.py`                    | Selects validation operating points and evaluates them on test                                       |
| `plot-thr-dep-scaffold.py`                              | Creates threshold-dependent figures                                                                  |
| `evaluate-training-dynamics-scaffold.py`                | Retrospective AP-based training-dynamics evaluation                                                  |
| `plot-training-dynamics-scaffold.py`                    | Creates the SCAFFOLD training-dynamics figure                                                        |

---

# 15. Result Directory Structure

A simplified result structure is:

```text
result/
└── splits_iid_scaling/
    │
    ├── splits_iid_2_clients.json/
    │   └── SCAFFOLD/
    │       ├── all_rounds_run_1/
    │       ├── ...
    │       ├── bestROC/
    │       ├── bestPRROC/
    │       └── bestLoss/
    │
    ├── ...
    │
    ├── splits_iid_16384_clients.json/
    │   └── SCAFFOLD/
    │
    ├── final_test_set_eval/
    │   └── SCAFFOLD/
    │       ├── all_test_results.csv
    │       ├── all_test_aggregate.csv
    │       └── ...
    │
    ├── final_threshold_analysis/
    │   └── SCAFFOLD/
    │       └── all_threshold_results.csv
    │
    └── training_dynamics/
        └── SCAFFOLD/
            ├── all_round_test_ap.csv
            ├── training_dynamics_by_run.csv
            └── training_dynamics_aggregate.csv
```

---

# 16. Useful Commands

Start one run:

```bash
flwr run .
```

Start the scaling launcher:

```bash
./federated_learning/tools/run_iid_scaling.sh
```

Start it in the background:

```bash
nohup ./federated_learning/tools/run_iid_scaling.sh &
```

Inspect logs:

```bash
tail -f nohup.out
```

Stop Ray:

```bash
ray stop --force
```

Show script options:

```bash
python <script>.py --help
```

---

# 17. Methodological Notes

### Fixed global dataset

The global training sample size is identical across all client counts. Increasing the number of clients therefore represents increasing **data fragmentation**, not increasing training data.

### IID partitioning

Training samples are randomly and approximately evenly distributed without using the labels during client assignment.

### SCAFFOLD control variates

SCAFFOLD maintains one global control variate and one persistent local control variate per client.

The global model parameters are still aggregated using sample-size-weighted FedAvg.

### Local optimization

Clients use SGD and class-weighted cross-entropy.

Gradient clipping is applied before adding the SCAFFOLD correction.

### Persistent client state

The local control variate `c_i` is stored using Flower's per-client state and restored when the same client participates in later rounds.

### Validation-based selection

Final checkpoints and operating-point thresholds are selected only from validation data.

### Final test evaluation

The final test set does not influence training, hyperparameter selection, checkpoint selection, or threshold selection.

### Retrospective training dynamics

The test trajectories used for training-dynamics analysis are descriptive only and do not alter the training procedure or selected models.

---

# 18. Scope

This README documents the **SCAFFOLD branch of the bachelor-thesis scalability experiments**.

The repository may contain additional scripts from earlier experiments, including Dirichlet partitions and older screening utilities. These are not required for reproducing the final IID SCAFFOLD scalability analysis.
