# Federated Edge Learning — FedAvg Scalability Study

This repository contains the experimental code used for the **FedAvg scalability study** conducted as part of the bachelor thesis on federated edge learning for diabetes screening.

The project uses  **Flower (FLwr)** for federated orchestration and simulation.

The central objective of the scalability study is to investigate how federated learning behaves when a **fixed global training dataset is distributed across an increasing number of clients**. Increasing the client count therefore does not add additional training data, but progressively fragments the same dataset into smaller local subsets.

For FedAvg, the evaluated client configurations are:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512,
1,024, 2,048, 4,096, 8,192, 16,384
```

Each scaling point is repeated **five times**. The selected FedAvg configuration uses **80 communication rounds** and a target client participation fraction of **0.80** with a minimum participation of 2 clients.

> **Legacy naming**
>
> Several evaluation scripts and result directories still use the identifier `FedProx`. These files correspond to the configuration reported as **FedAvg** in the final thesis. The historical names are retained where necessary to preserve compatibility with the existing result directory structure.

---

## Contents

* [Requirements and Installation](#requirements-and-installation)
* [Project Workflow](#project-workflow)
* [Data Preparation](#1-data-preparation)
* [Creating IID Scaling Splits](#2-creating-iid-scaling-splits)
* [Configuring FedAvg](#3-configuring-fedavg)
* [Running the Scaling Study](#4-running-the-fedavg-scaling-study)
* [Evaluation Workflow](#5-evaluation-workflow)
* [Validation-Based Checkpoint Selection](#6-validation-based-checkpoint-selection)
* [Threshold-Independent Test Evaluation](#7-threshold-independent-final-test-evaluation)
* [Run-to-Run Dispersion](#8-run-to-run-dispersion)
* [Threshold-Dependent Evaluation](#9-threshold-dependent-evaluation)
* [Training-Dynamics Analysis](#10-training-dynamics-analysis)
* [Configuration-Selection Plot](#11-configuration-selection-plot)
* [Complete Execution Order](#12-complete-execution-order)
* [Project File Overview](#13-project-file-overview)
* [Result Directory Structure](#14-result-directory-structure)
* [Useful Commands](#15-useful-commands)

---

# Requirements and Installation

A Python virtual environment is recommended.

## Requirements

* Python 3.10+
* `pip`
* Flower
* PyTorch
* NumPy
* pandas
* scikit-learn
* matplotlib
* PyArrow

Additional dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

The Flower application components and the main simulation configuration are defined in:

```text
pyproject.toml
```

---

# Project Workflow

The complete FedAvg workflow is:

```text
Raw dataset
    │
    ▼
Data preparation
    │
    ▼
Feature normalization
and class weights
    │
    ▼
IID scaling splits
    │
    ▼
FedAvg training
    │
    ▼
Validation-based
checkpoint selection
    │
    ├──────────────────────┬─────────────────────────┐
    │                      │                         │
    ▼                      ▼                         ▼
Threshold-independent   Threshold-dependent      Training-dynamics
test evaluation         evaluation               analysis
    │                      │                         │
    ▼                      ▼                         ▼
Performance plots       Operating-point plots    Convergence plots
and dispersion tables
```

The **validation set** is used for:

* configuration selection,
* communication-round model checkpoint selection,
* decision-threshold selection.

The **test set is not used to select models, rounds, thresholds, or hyperparameters**. It is used only for the final performance evaluation and the retrospective descriptive training-dynamics analysis.

---

## Core Federated Learning Components

The federated training application is organized around three core files: `server_app.py`, `client_app.py`, and `task.py`. Together, they define the server-side aggregation, client-side training, and client-specific data loading used during the Flower simulations.

### `server_app.py`

```text
federated_learning/server_app.py
```

Defines the Flower server application and the FedAvg strategy used during training. It coordinates client sampling, distributes the global model, aggregates the returned client updates using sample-size-weighted FedAvg, and handles checkpoint and metric storage during the communication rounds.

### `client_app.py`

```text
federated_learning/client_app.py
```

Defines the Flower client application and the neural network used for local training. Each selected client receives the current global model, trains it on its assigned local dataset using the configured loss, optimizer, learning-rate schedule, and gradient clipping, and returns the updated model parameters to the server.

### `task.py`

```text
federated_learning/task.py
```

Contains the shared data-loading utilities used by the clients. It loads only the rows assigned to the respective client from the normalized Parquet dataset, retrieves the corresponding training and validation samples, loads the predefined class weights, and creates the PyTorch `DataLoader` objects used during local training and evaluation.


---

# 1. Data Preparation

its already done, no need to repeat this steps.

## `prepare_data.py`

```text
federated_learning/tools/prepare_data.py
```

Performs the initial dataset preparation and creates the global train, validation, and test assignment together with the metadata required by later processing steps.

Example:

```bash
python3 federated_learning/tools/prepare_data.py \
    --csv <input.csv> \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json
```

The resulting metadata file contains the stable row indices used to preserve the same global train, validation, and test sets throughout all experiments.

---

## `normalize_and_add_weights.py`

```text
federated_learning/tools/normalize_and_add_weights.py
```

Standardizes the input features using statistics calculated from the training subset and computes the global class weights used during model training.

Example:

```bash
python3 federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5
```

The script:

1. loads the existing train/validation/test assignment,
2. standardizes all features using training-set statistics,
3. writes the normalized dataset to Parquet,
4. calculates the class weights from the training data,
5. applies the configured positive-class weight boost,
6. stores the resulting weights in `norm_stats.json`.

This script does **not** create new train, validation, or test splits.

The resulting files are:

```text
data/
├── diabetes.parquet
├── diabetes_normalized.parquet
└── norm_stats.json
```

---

# 2. Creating IID Scaling Splits

Already created.

## `make_splits.py`

```text
federated_learning/tools/make_splits.py
```

Contains general helper functions for partitioning data across federated clients.

For the final scalability study, its `iid_partitions()` functionality is used to randomly distribute training observations approximately evenly across the clients.

The file also contains functionality for other partitioning schemes, but these are not part of the final FedAvg scalability experiment.

---

## `create_iid_scaling_splits.py`

```text
federated_learning/tools/create_iid_scaling_splits.py
```

Creates the client partitions used for the IID scalability study.

Example:

```bash
python3 federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123
```

The global training dataset remains unchanged across scaling points. Only the number of clients across which the samples are distributed changes.

Labels are **not used for assigning observations to clients**. Differences in local class composition therefore arise naturally from IID random sampling.

The generated files follow the structure:

```text
splits_iid_scaling/
├── splits_iid_2_clients.json
├── splits_iid_4_clients.json
├── splits_iid_8_clients.json
├── splits_iid_16_clients.json
├── ...
├── splits_iid_8192_clients.json
└── splits_iid_16384_clients.json
```

Each split contains a mapping between client IDs and the global row IDs assigned to that client.

---

## `analyze_iid_splits_one_row_per_split.py`

```text
analyze_iid_splits_one_row_per_split.py
```

Analyzes the generated IID scaling splits and creates one summary row for every client configuration.

The script calculates, among other quantities:

* number of clients,
* total number of training samples,
* mean local dataset size,
* minimum and maximum local dataset size,
* mean number of positive samples per client,
* variation in positive samples across clients,
* number of clients without positive training samples,
* percentage of clients without positive training samples,
* global positive-class rate.

This analysis was used to characterize how local data availability and local class composition change under increasing fragmentation.

---

## `adjust_val_distribution.py`

```text
adjust_val_distribution.py
```

Modifies the generated split files so that all validation samples are assigned to a single validation client, reducing the overhead of centralized validation during training and making evaluation faster.


---
# 3. Configuring FedAvg

The main configuration file is:

```text
pyproject.toml
```

It contains the default Flower configuration, model-training parameters, data paths, and Ray simulation settings.

The main experiment settings are located under:

```toml
[tool.flwr.app.config]
```

---

## 3.1 Federation Settings

Important federation parameters include:

```toml
num-server-rounds = 80
fraction-fit = 0.8
local-epochs = 1
```

The final FedAvg scalability experiment uses:

```text
Communication rounds:       80
Target client participation: 80%
Runs per scaling point:      5
Scaling range:                2–16,384 clients
```

At the two-client configuration, **both clients participate**.

For all larger configurations, the target participation fraction is converted to a whole number of clients by rounding down:

```text
2 clients       → 2 participating clients
4 clients       → 3
8 clients       → 6
16 clients      → 12
32 clients      → 25
...
```

---

## 3.2 Data Paths

Relevant data configuration fields include:

```toml
dataset-path = "data/diabetes.csv"
prepared-parquet = "data/diabetes_normalized.parquet"
norm-stats-json = "data/norm_stats.json"
split-path = "splits_iid_scaling/splits_iid_16384_clients.json"
```

The `split-path` entry serves as a default. During the scalability study, `run_iid_scaling.sh` passes the appropriate split path for each client configuration through the Flower run configuration.

---

## 3.3 Selected FedAvg Optimization Configuration

The configuration selected at the 16,384-client reference setting is used unchanged throughout the scalability study.

### Learning-rate schedule

```text
Rounds 1–12:
linear warm-up from 3e-3 to 8e-2

Rounds 13–59:
constant learning rate of 8e-2

Rounds 60–80:
cosine-annealing cool-down from 8e-2 to 3e-2
```

Additional selected settings:

```text
Positive-class weight boost: 1.5
Weight decay:                 5e-4
Gradient clipping norm:       4.0
Local epochs:                 1
Communication rounds:         80
Client participation:         0.80
```

The positive-class weighting addresses the class imbalance of the diabetes dataset, while gradient clipping limits excessively large local gradients.

---

## 3.4 Changing Experiment Parameters

The launcher and `pyproject.toml` serve different purposes.

### `run_iid_scaling.sh`

Use the launcher to change which experiment is executed, including:

* client counts,
* number of repeated runs,
* split file,
* run identifier,
* number of communication rounds,
* number of required participating clients.

### `pyproject.toml`

Use `pyproject.toml` to change the actual model and optimization configuration, including:

* learning rate,
* learning-rate schedule,
* warm-up configuration,
* weight decay,
* gradient clipping,
* positive-class weight boost,
* number of local epochs,
* dataset paths,
* Flower settings,
* Ray CPU and memory resources.

---

# 4. Running the FedAvg Scaling Study

## `run_iid_scaling.sh`

```text
federated_learning/tools/run_iid_scaling.sh
```

This is the main launcher for the IID FedAvg scalability experiments.

Run it from the project root:

```bash
./federated_learning/tools/run_iid_scaling.sh
```

If necessary, first make it executable:

```bash
chmod +x federated_learning/tools/run_iid_scaling.sh
```

---

## 4.1 Selecting the Scaling Points

The client counts are specified near the beginning of the script:

```bash
CLIENT_COUNTS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
```

To run only selected scaling points:

```bash
CLIENT_COUNTS=(1024 2048 4096)
```

To run only one configuration:

```bash
CLIENT_COUNTS=(4096)
```

---

## 4.2 Selecting the Number of Repetitions

The final study uses:

```bash
RUNS_PER_SPLIT=5
```

For a short test run:

```bash
RUNS_PER_SPLIT=1
```

---

## 4.3 Split Selection

For every client count `N`, the launcher uses:

```text
splits_iid_scaling/splits_iid_<N>_clients.json
```

For example:

```text
splits_iid_scaling/splits_iid_4096_clients.json
```

for the 4,096-client configuration.

The number of simulated Flower SuperNodes is adjusted to match the corresponding split.

---

## 4.4 Parameters Passed by the Launcher

The launcher passes run-specific values to Flower using `--run-config`.

These include:

```text
split-path
min-fit-clients
min-available-clients
min-evaluate-clients
num-server-rounds
run-tag
```

The values provided through `--run-config` override the corresponding defaults in `pyproject.toml` for the current simulation.

---

## 4.5 Logs

Run-specific logs are stored under:

```text
logs/iid_scaling/
```

For longer runs on a remote machine, the launcher can be started with:

```bash
nohup ./federated_learning/tools/run_iid_scaling.sh &
```

The launcher cleans up the Ray runtime between experiments and restores temporary changes made to `pyproject.toml`.

---

# 5. Evaluation Workflow

After the training runs are complete, the evaluation follows this order:

```text
Training checkpoints
        │
        ▼
scaling_evaluation_fedavg.py
        │
        │ Validation-based checkpoint selection
        │
        ├── bestROC
        ├── bestPRROC
        └── bestLoss
        │
        ├───────────────────────────────┐
        │                               │
        ▼                               ▼
final_test_set_eval_fedavg.py   evaluate-thr-dependent-fedavg.py
        │                               │
        ▼                               ▼
Threshold-independent            Threshold-dependent
test results                     test results
        │                               │
        ├───────────────┐               ▼
        ▼               ▼        plot-thr-dependent-fedavg.py
plot-thr-indep-    table_plot_
fedavg.py         fedavg.py


All saved round checkpoints
        │
        ▼
evaluate-training-dynamics-fedavg.py
        │
        ▼
plot-training-dynamics-fedavg.py
```

---

# 6. Validation-Based Checkpoint Selection

## `scaling_evaluation_fedavg.py`

```text
scaling_evaluation_fedavg.py
```

This is the **first evaluation step after training**.

For every scaling point and repeated run, the script evaluates the saved communication-round checkpoints on the centralized validation set.

Three checkpoint-selection criteria are considered independently:

```text
bestROC
    highest validation ROC-AUC

bestPRROC
    highest validation Average Precision (AP)

bestLoss
    lowest weighted validation loss
```

The selected checkpoints are stored under directories such as:

```text
result/splits_iid_scaling/
└── splits_iid_4096_clients.json/
    └── FedProx/
        ├── bestROC/
        │   ├── run_1/
        │   ├── ...
        │   └── run_5/
        │
        ├── bestPRROC/
        │   ├── run_1/
        │   ├── ...
        │   └── run_5/
        │
        └── bestLoss/
            ├── run_1/
            ├── ...
            └── run_5/
```

The test set is not used during checkpoint selection.

---

# 7. Threshold-Independent Final Test Evaluation

## `final_test_set_eval_fedavg.py`

```text
final_test_set_eval_fedavg.py
```

Evaluates only the checkpoints that were previously selected using the validation set.

For each client configuration and repeated run, the script evaluates:

```text
bestROC
bestPRROC
bestLoss
```

on the fixed centralized test set.

The evaluation calculates:

* weighted test loss,
* ROC-AUC,
* Average Precision,
* precision-recall curve,
* ROC curve,
* class counts,
* class prevalence,
* checkpoint provenance information.

No model, communication round, threshold, or hyperparameter is selected using the test set.

The main output directory is:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/
```

Important output files include:

```text
all_test_results.csv
all_test_aggregate.csv
final_test_summary.json
test_set_info.json
```

and run-specific result files:

```text
splits_iid_<N>_clients/
├── bestROC/
├── bestPRROC/
└── bestLoss/
```

---

## `plot-thr-indep-fedavg.py`

```text
plot-thr-indep-fedavg.py
```

Creates the threshold-independent FedAvg scalability figures from the run-level final test results.

The final main figure contains:

```text
Panel A: ROC-AUC
Panel B: Average Precision
Panel C: Weighted loss
```

Each metric uses the checkpoint selected with its corresponding validation criterion:

```text
ROC-AUC       → bestROC
AP            → bestPRROC
Weighted loss → bestLoss
```

The plot shows both:

* individual repeated runs,
* mean performance across the five runs.

Default input:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/all_test_results.csv
```

Run:

```bash
python3 plot-thr-indep-fedavg.py
```

The script writes high-resolution PNG and vector PDF output.

---

# 8. Run-to-Run Dispersion

## `table_plot_fedavg.py`

```text
table_plot_fedavg.py
```

Analyzes the variation between the five repeated FedAvg runs at every scaling point.

It reads:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/all_test_results.csv
```

and computes statistics such as:

* mean,
* standard deviation,
* coefficient of variation,
* minimum,
* maximum,
* maximum absolute relative deviation of an individual run from the scaling-point mean.

The analysis is performed separately for:

```text
ROC-AUC
Average Precision
Weighted loss
```

using the corresponding validation-selected checkpoints.

Run:

```bash
python3 table_plot_fedavg.py
```

Important outputs include:

```text
run_dispersion_table.csv
run_dispersion_summary_full.csv
run_dispersion_table.md
run_dispersion_by_scaling.csv

figure_run_to_run_dispersion_by_scaling.png
```

The resulting dispersion figure is used to describe how strongly repeated runs vary at different client counts.

---

# 9. Threshold-Dependent Evaluation

## `evaluate-thr-dependent-fedavg.py`

```text
evaluate-thr-dependent-fedavg.py
```

Evaluates the final models at two validation-selected operating points.

The analysis uses the model checkpoint selected by **validation AP** (`bestPRROC`).

For each scaling point and run:

1. the trained checkpoint is loaded,
2. predictions are generated on the centralized validation set,
3. the decision threshold is selected using validation data,
4. the selected threshold is transferred unchanged to the test set,
5. the final threshold-dependent test metrics are calculated.

Two threshold-selection regimes are used.

---

## 9.1 MCC-Optimal Operating Point

The first regime selects the threshold that maximizes the Matthews correlation coefficient on the validation set:

```text
selected threshold = argmax validation MCC
```

The selected threshold is then applied unchanged to the test predictions.

---

## 9.2 Fixed Minimum-Recall Operating Point

The second regime imposes a validation recall requirement of:

```text
Recall >= 0.80
```

Among the thresholds satisfying this constraint, the threshold with the highest validation specificity is selected.

Run:

```bash
python3 evaluate-thr-dependent-fedavg.py --min-recall 0.80
```

The combined results are stored under:

```text
result/splits_iid_scaling/final_threshold_analysis/FedProx/
```

including:

```text
all_threshold_results.csv
```

and run-specific JSON result files.

---

## `plot-thr-dependent-fedavg.py`

```text
plot-thr-dependent-fedavg.py
```

Creates the threshold-dependent FedAvg figures from the results generated by `evaluate-thr-dependent-fedavg.py`.

The plotting script performs **no additional checkpoint or threshold selection**.

### MCC-optimal figure

```text
Panel A:
Validation-selected decision threshold

Panel B:
Test MCC

Panel C:
Test recall and specificity
```

### Fixed validation-recall figure

```text
Panel A:
Validation-selected decision threshold

Panel B:
Test recall and specificity
```

Run:

```bash
python3 plot-thr-dependent-fedavg.py
```

Default input:

```text
result/splits_iid_scaling/final_threshold_analysis/FedProx/
    all_threshold_results.csv
```

---

# 10. Training-Dynamics Analysis

The training-dynamics analysis examines how many communication rounds are required to approach the best observed predictive performance and whether training is still systematically changing near the end of the 80-round horizon.

Unlike the primary final test evaluation, this is a **retrospective descriptive analysis of the saved training trajectory**.

It does not affect:

* training,
* configuration selection,
* checkpoint selection,
* threshold selection,
* final reported model selection.

---

## `evaluate-training-dynamics-fedavg.py`

```text
evaluate-training-dynamics-fedavg.py
```

Evaluates the saved communication-round checkpoints retrospectively on the test set using Average Precision.

Two quantities are calculated for every run.

### First round reaching 99% of best AP

The script determines:

```text
first evaluated round for which

test AP >= 0.99 × best observed test AP within the same run
```

This provides a descriptive measure of how quickly the run approaches its best observed performance.

---

### Late-training AP trend

An ordinary least-squares linear regression is fitted to test AP over:

```text
rounds 70–80 inclusive
```

The resulting slope describes the late-training trend:

```text
positive slope    → AP still increasing
slope near zero   → little systematic change
negative slope    → AP decreasing
```

Run:

```bash
python3 evaluate-training-dynamics-fedavg.py
```

---

## `plot-training-dynamics-fedavg.py`

```text
plot-training-dynamics-fedavg.py
```

Plots the results from the training-dynamics evaluation.

The figure contains:

```text
Panel A:
First evaluated communication round reaching
99% of the best observed test AP

Panel B:
OLS test-AP slope over communication rounds 70–80
```

Small points represent individual runs and the connected markers represent the mean across the five repeated runs.

Run:

```bash
python3 plot-training-dynamics-fedavg.py
```

---

# 11. Configuration-Selection Plot

## `plot_strategy_mcc.py`

```text
plot_strategy_mcc.py
```

Visualizes the preliminary optimization-configuration experiments conducted at the fixed **16,384-client reference configuration**.

For FedAvg, five candidate configurations were evaluated before the final scalability study.

The plot is based on validation MCC over the communication rounds and summarizes properties including:

* maximum validation MCC,
* late-training plateau performance,
* late-training variation,
* time required to approach the plateau.

The final FedAvg configuration selected from this experiment was subsequently used unchanged across all scaling points.

This script belongs to the **configuration-selection stage** and is not part of the final scaling evaluation pipeline itself.

---

# 12. Complete Execution Order

A full reproduction of the FedAvg study follows the sequence below.

## Step 1 — Prepare the raw dataset

```bash
python3 federated_learning/tools/prepare_data.py \
    --csv <input.csv> \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json
```

## Step 2 — Normalize features and calculate class weights

```bash
python3 federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5
```

## Step 3 — Create the IID client partitions

```bash
python3 federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123
```

## Step 4 — Analyze the generated partitions

```bash
python3 analyze_iid_splits_one_row_per_split.py
```

## Step 5 — Run the FedAvg scaling study

```bash
./federated_learning/tools/run_iid_scaling.sh
```

## Step 6 — Select communication-round checkpoints using validation data

```bash
python3 scaling_evaluation_fedavg.py
```

## Step 7 — Perform the final threshold-independent test evaluation

```bash
python3 final_test_set_eval_fedavg.py
```

## Step 8 — Generate threshold-independent plots

```bash
python3 plot-thr-indep-fedavg.py
```

## Step 9 — Calculate run-to-run dispersion

```bash
python3 table_plot_fedavg.py
```

## Step 10 — Select operating points on validation and evaluate on test

```bash
python3 evaluate-thr-dependent-fedavg.py --min-recall 0.80
```

## Step 11 — Generate threshold-dependent plots

```bash
python3 plot-thr-dependent-fedavg.py
```

## Step 12 — Evaluate training dynamics

```bash
python3 evaluate-training-dynamics-fedavg.py
```

## Step 13 — Generate the training-dynamics figure

```bash
python3 plot-training-dynamics-fedavg.py
```

---

# 13. Project File Overview

## Core Training Files

| File                               | Purpose                                                                           |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| `pyproject.toml`                   | Main Flower, optimization, data-path and Ray simulation configuration             |
| `federated_learning/client_app.py` | Neural-network model and client-side training/evaluation logic                    |
| `federated_learning/server_app.py` | Server-side FedAvg aggregation, client coordination and checkpoint handling       |
| `federated_learning/task.py`       | Loads client-specific normalized observations and constructs PyTorch data loaders |

## Data and Partitioning

| File                                                    | Purpose                                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `federated_learning/tools/prepare_data.py`              | Initial dataset preparation and global train/validation/test metadata            |
| `federated_learning/tools/normalize_and_add_weights.py` | Standardizes features and calculates class weights                               |
| `federated_learning/tools/make_splits.py`               | General client-partitioning helper functions                                     |
| `federated_learning/tools/create_iid_scaling_splits.py` | Generates the IID client partitions used for the scalability study               |
| `analyze_iid_splits_one_row_per_split.py`               | Summarizes local sample availability and class composition across scaling points |

## Training

| File                                          | Purpose                                                            |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `federated_learning/tools/run_iid_scaling.sh` | Executes repeated FedAvg experiments across selected client counts |

## Validation and Test Evaluation

| File                                                 | Purpose                                                                                              |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `scaling_evaluation_fedavg.py`                      | Evaluates saved checkpoints on validation and selects the best ROC-AUC, AP and loss checkpoint       |
| `final_test_set_eval_fedavg.py`                     | Evaluates the validation-selected checkpoints on the fixed final test set                            |
| `evaluate-thr-dependent-fedavg.py`                  | Selects MCC-optimal and recall-constrained operating points on validation and evaluates them on test |
| `evaluate-training-dynamics-fedavg.py` | Retrospectively evaluates saved checkpoints to characterize convergence behavior                     |

## Figures and Tables

| File                                             | Purpose                                                     |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `plot_strategy_mcc.py`               | Configuration-selection visualization at 16,384 clients for all strategies   |
| `plot-thr-indep-fedavg.py`                      | Threshold-independent FedAvg scalability figure             |
| `table_plot_fedavg.py`                          | Run-to-run dispersion statistics, table and appendix figure |
| `plot-thr-dependent-fedavg.py`                  | Threshold-dependent FedAvg figures                          |
| `plot-training-dynamics-fedavg.py` | FedAvg training-dynamics appendix figure                 
| `plot-combined-strategy-comparison.py` | Creates the cross-strategy overview figures comparing FedAvg, SCAFFOLD, and FedAdam for threshold-independent performance and both validation-selected operating points |


---

# 14. Result Directory Structure

Training results are organized by scaling point.

A simplified structure is:

```text
result/
└── splits_iid_scaling/
    │
    ├── splits_iid_2_clients.json/
    │   └── FedProx/
    │       ├── all_rounds/
    │       ├── bestROC/
    │       ├── bestPRROC/
    │       └── bestLoss/
    │
    ├── splits_iid_4_clients.json/
    │   └── FedProx/
    │
    ├── ...
    │
    ├── splits_iid_16384_clients.json/
    │   └── FedProx/
    │
    ├── final_test_set_eval/
    │   └── FedProx/
    │       ├── all_test_results.csv
    │       ├── all_test_aggregate.csv
    │       ├── final_test_summary.json
    │       └── ...
    │
    └── final_threshold_analysis/
        └── FedProx/
            ├── all_threshold_results.csv
            └── ...
```

Again, `FedProx` is a legacy directory identifier for the experiments reported as **FedAvg** in the final thesis.

---

# 15. Output Formats

The main output formats are:

```text
.pt
    PyTorch model checkpoints

.json
    run-specific metrics, selected-checkpoint metadata
    and detailed evaluation results

.csv
    combined run-level and aggregated analysis results

.pdf
    vector figures

.png
    high-resolution raster figures

.log
    Flower simulation logs
```

---

# 16. Useful Commands

## Display command-line options

Most evaluation and utility scripts provide command-line help:

```bash
python3 <script>.py --help
```

For example:

```bash
python3 federated_learning/tools/create_iid_scaling_splits.py --help
python3 evaluate-thr-dependent-fedavg.py --help
python3 plot-thr-dependent-fedavg.py --help
```

---

## Stop Ray

If a previous simulation left Ray processes running:

```bash
ray stop --force
```

---

## Start training in the background

```bash
nohup ./federated_learning/tools/run_iid_scaling.sh &
```

---

## Check background output

```bash
tail -f nohup.out
```

or inspect the run-specific files under:

```text
logs/iid_scaling/
```

---

# 17. Methodological Notes

The final FedAvg scalability study follows several important conventions.

### Fixed global dataset

The total training dataset remains constant across all client configurations. Increasing the client count therefore represents increasing **client-level data fragmentation**, not an increase in training data.

### IID client partitioning

Training observations are randomly and approximately evenly distributed across clients without using the labels during partitioning.

### Fixed configuration across scaling points

The FedAvg optimization configuration is selected at 16,384 clients and subsequently held constant across the complete scaling range.

This allows the experiment to examine how one fixed configuration responds to increasing fragmentation rather than retuning the model independently at every client count.

### Repeated experiments

Each scaling point is repeated five times to characterize run-to-run variation.

### Validation-based model selection

Communication-round checkpoints are selected using the centralized validation set only.

### Validation-based threshold selection

Both the MCC-optimal decision threshold and the fixed minimum-recall operating point are selected using validation predictions.

### Final test evaluation

The test set is only evaluated after the relevant checkpoint and, where applicable, decision threshold have already been selected.

### Training-dynamics analysis

The retrospective evaluation of all saved test-set checkpoints is used only to characterize training behavior and does not alter any training or model-selection decision.

---

# 18. Scope

This README documents the **FedAvg branch of the bachelor-thesis experiments**.

The repository may contain additional scripts from earlier experiments or implementations of other federated optimization strategies. These are not required for reproducing the FedAvg scalability results described above.
