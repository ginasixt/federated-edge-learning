# Federated Edge Learning — SCAFFOLD Scalability Study

This repository contains the experimental code for the **SCAFFOLD scalability study** conducted as part of a bachelor thesis on federated edge learning for diabetes screening.

The implementation uses **PyTorch** for model training and **Flower (FLwr)** for federated orchestration and simulation.

The central experiment investigates how federated learning behaves when a **fixed global training dataset is distributed across an increasing number of clients**. Increasing the number of clients therefore does not add training data, but progressively fragments the same dataset into smaller local client datasets.

The final SCAFFOLD scalability study evaluates:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512,
1,024, 2,048, 4,096, 8,192, 16,384 clients
```

Each scaling point is repeated **five times**.

SCAFFOLD extends FedAvg through control variates. In addition to the global model, the server maintains a global control variate and each client maintains a persistent local control variate. Their difference is used to correct the local optimization step.

---

# Requirements and Installation

A Python virtual environment is recommended.

Create and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -U pip
pip install -r requirements.txt
```

The main project dependencies include Flower, PyTorch, NumPy, pandas, scikit-learn, matplotlib, and PyArrow.

The Flower application and experiment configuration are defined in:

```text
pyproject.toml
```

---

# Overall Workflow

The final SCAFFOLD experiment follows this workflow:

```text
Raw BRFSS dataset
        │
        ▼
prepare_data.py
        │
        ▼
Fixed train / validation / test split
        │
        ▼
normalize_and_add_weights.py
        │
        ▼
Normalized dataset + class weights
        │
        ▼
create_iid_scaling_splits.py
        │
        ▼
IID client partitions
        │
        ▼
SCAFFOLD training
client_app.py + server_app.py + task.py
        │
        ▼
Saved checkpoints from evaluated communication rounds
        │
        ▼
scaling_eval_scaffold.py
        │
        ▼
Validation-based checkpoint selection
   ├── bestROC
   ├── bestPRROC
   └── bestLoss
        │
        ▼
final_test_set_eval_scaffold.py
        │
        ▼
Final threshold-independent test results
        │
        ├── plot-thr-indep-scaffold.py
        └── table_plot_scaffold.py
```

Two additional analyses branch from the saved results.

Threshold-dependent evaluation:

```text
bestPRROC checkpoint
        │
        ▼
evaluate-thr-dependent-scaffold.py
        │
        ▼
Threshold selection on validation
   ├── MCC-optimal threshold
   └── Recall ≥ 0.80, then maximum specificity
        │
        ▼
Apply selected threshold unchanged to test
        │
        ▼
plot-thr-dep-scaffold.py
```

Training-dynamics analysis:

```text
Saved checkpoints from evaluated communication rounds
        │
        ▼
evaluate-training-dynamics-scaffold.py
        │
        ▼
Test AP across communication rounds
   ├── First evaluated round reaching 99% of best test AP
   └── Late-training test-AP slope
        │
        ▼
plot-training-dynamics-scaffold.py
```

The validation set is used for checkpoint and threshold selection. The test set is not used to select final checkpoints or decision thresholds.

The training-dynamics analysis evaluates intermediate checkpoints retrospectively on the test set for descriptive analysis only and does not affect model selection.

---

# Core Federated Learning Components

## `server_app.py`

```text
federated_learning/server_app.py
```

Defines the Flower server application and the server-side SCAFFOLD implementation.

The custom strategy extends Flower's FedAvg aggregation with the SCAFFOLD global control variate.

The server is responsible for:

* coordinating the participating clients,
* maintaining the global model,
* aggregating client model updates,
* maintaining the global SCAFFOLD control variate,
* receiving client control-variate changes,
* saving model checkpoints and evaluation information.

The standard model parameters returned by the clients are aggregated using Flower's FedAvg aggregation.

In addition, participating clients return their local control-variate changes `Δc_i`. These are used to update the global control variate.

The global control variate is initialized to zero and sent to participating clients in subsequent communication rounds.

Saved checkpoints are later used for validation-based checkpoint selection and the retrospective training-dynamics analysis.

---

## `client_app.py`

```text
federated_learning/client_app.py
```

Defines the Flower client application, neural network, local training procedure, and client-side SCAFFOLD control variate.

The model is a multilayer perceptron with:

```text
21 input features
        ↓
256 hidden units + ReLU
        ↓
128 hidden units + ReLU
        ↓
2 output logits
```

For every selected client, the client:

1. receives the current global model,
2. receives the global control variate,
3. restores its persistent local control variate,
4. loads its assigned local training observations,
5. performs local SGD training,
6. applies the SCAFFOLD gradient correction,
7. updates its local control variate,
8. stores the local control variate in Flower's persistent client state,
9. returns the updated model and control-variate change to the server.

The SCAFFOLD correction is applied as:

```text
corrected gradient = local gradient + c - c_i
```

where `c` is the global and `c_i` the client-specific control variate.

Gradient clipping is applied **before** the SCAFFOLD correction.

The implementation uses the computationally cheaper SCAFFOLD Option II update for the client control variate.

Local optimization uses vanilla SGD without momentum or weight decay.

---

## `task.py`

```text
federated_learning/task.py
```

Contains the data-loading utilities used by the Flower clients and the evaluation scripts.

`load_client_data()` loads only the row IDs assigned to the respective client from:

```text
data/diabetes_normalized.parquet
```

The Parquet file is already normalized before training, so no additional feature normalization is performed during client training.

Class weights are loaded from:

```text
data/norm_stats.json
```

using the precomputed:

```text
neg_weight
pos_weight
```

---

# Data Preparation

## `prepare_data.py`

```text
federated_learning/tools/prepare_data.py
```

Prepares the BRFSS 2015 diabetes dataset and creates the fixed global train, validation, and test partition.

The default proportions are:

```text
Training:    70%
Validation:  10%
Test:        20%
```

The split is stratified by the binary target variable.

The script stores stable row IDs so that the same observations remain in the global train, validation, and test sets throughout all scaling experiments.

Run:

```bash
python3 federated_learning/tools/prepare_data.py \
    --csv data/diabetes.csv \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json
```

The current implementation obtains the BRFSS dataset through `kagglehub`; the `--csv` argument is retained by the command-line interface.

Outputs:

```text
data/diabetes.parquet
data/norm_stats.json
```

`norm_stats.json` contains, among other information:

```text
train_idx
val_idx
test_idx
mean
std
target
```

The normalization statistics are calculated using only the training subset.

---

## `normalize_and_add_weights.py`

```text
federated_learning/tools/normalize_and_add_weights.py
```

Normalizes the prepared dataset and calculates the class weights used during training.

Run:

```bash
python3 federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5
```

The script:

1. loads the existing train/validation/test split,
2. normalizes the features using the training-set mean and standard deviation,
3. writes the normalized dataset,
4. calculates class weights from the global training subset,
5. applies the positive-class boost,
6. stores the resulting weights in `norm_stats.json`.

It does **not** create a new train/validation/test split.

Output:

```text
data/diabetes_normalized.parquet
```

and additional entries in `norm_stats.json`, including:

```text
pos_weight
neg_weight
train_pos_count
train_neg_count
pos_weight_boost
```

---

# Creating the IID Scaling Partitions

## `make_splits.py`

```text
federated_learning/tools/make_splits.py
```

Contains general helper functions for client partitioning.

The final scalability experiment uses its IID partitioning functionality.

Other partitioning functionality contained in the file is not required for reproducing the final SCAFFOLD scalability study.

---

## `create_iid_scaling_splits.py`

```text
federated_learning/tools/create_iid_scaling_splits.py
```

Creates the client partitions used in the final scalability study.

Run:

```bash
python3 federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123
```

The fixed global training dataset is randomly and approximately evenly distributed across increasing numbers of clients.

Importantly, **labels are not used when assigning training observations to clients**. Label information is only inspected afterwards to describe the resulting local class distributions.

For the final SCAFFOLD study, the relevant scaling points are:

```text
2
4
8
16
32
64
128
256
512
1,024
2,048
4,096
8,192
16,384
```

The corresponding files follow the naming convention:

```text
splits_iid_scaling/
├── splits_iid_2_clients.json
├── splits_iid_4_clients.json
├── splits_iid_8_clients.json
├── ...
├── splits_iid_8192_clients.json
└── splits_iid_16384_clients.json
```

The global amount of training data remains constant across these files. Only the number and size of the local client datasets changes.

---

# Configuring SCAFFOLD

The main experiment configuration is stored in:

```text
pyproject.toml
```

The Flower application entry points connect the project to:

```text
federated_learning.server_app
federated_learning.client_app
```

The main experiment parameters can be changed under:

```toml
[tool.flwr.app.config]
```

Relevant parameters include:

```text
num-server-rounds
fraction-fit
min-fit-clients
min-available-clients
local-epochs

batch-size
lr
clip-grad-norm
pos-weight-boost

prepared-parquet
norm-stats-json
split-path
run-tag
```

The final SCAFFOLD scalability study uses:

```text
Communication rounds:         80
Client participation:         approximately 0.75
Repeated runs:                5
Scaling range:                2–16,384 clients
Client learning rate:         1e-2
Local epochs:                 2
Gradient clipping norm:       5.0
Positive-class boost:         1.5
```

At the two-client scaling point, both clients participate. For larger client configurations, the target number of participating clients is converted to a whole number of clients.

The SCAFFOLD client uses vanilla SGD:

```text
Optimizer:       SGD
Momentum:        none
Weight decay:    0
```

The configuration used for an actual scaling run is determined by the values in `pyproject.toml` together with values explicitly overridden through Flower's `--run-config`.

In particular, the scaling launcher passes the scaling-point-specific `min-fit-clients` value to Flower and thereby determines the effective number of participating clients.

---

# Starting SCAFFOLD Training

There are two ways to start the training.

## Single Experiment

To run one experiment using the current settings in `pyproject.toml`:

```bash
flwr run .
```

This is useful for individual runs, configuration tests, and debugging.

Before starting, check at least:

```text
split-path
options.num-supernodes
num-server-rounds
fraction-fit
min-fit-clients
local-epochs
lr
clip-grad-norm
run-tag
```

---

## Scaling Study

The automated scaling launcher is:

```text
federated_learning/tools/run_iid_scaling.sh
```

Run:

```bash
./federated_learning/tools/run_iid_scaling.sh
```

If the script is not executable:

```bash
chmod +x federated_learning/tools/run_iid_scaling.sh
```

The launcher is used to run several client configurations and repeated runs automatically.

Its main responsibilities are:

1. iterate over the requested client counts,
2. select the corresponding IID split file,
3. set the simulated number of clients,
4. determine the required number of participating clients,
5. assign a run identifier,
6. start Flower with the corresponding `--run-config`,
7. store logs,
8. stop remaining Ray processes between experiments.

For the complete final SCAFFOLD study, the client-count list should contain:

```bash
CLIENT_COUNTS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
```

and:

```bash
RUNS_PER_SPLIT=5
```

The selected split for `N` clients follows:

```text
splits_iid_scaling/splits_iid_<N>_clients.json
```

For example:

```text
splits_iid_scaling/splits_iid_4096_clients.json
```

The launcher can pass values such as:

```text
split-path
min-fit-clients
min-available-clients
min-evaluate-clients
num-server-rounds
run-tag
```

through Flower's `--run-config`.

These values override the corresponding defaults in `pyproject.toml` for that run.

For exact reproduction of the final scalability study, the launcher should use **80 communication rounds** for all scaling points.

---

## Running in the Background

For long simulations on a remote machine:

```bash
nohup ./federated_learning/tools/run_iid_scaling.sh &
```

Logs can be inspected with:

```bash
tail -f nohup.out
```

or through the run-specific log files produced by the launcher.

---

# Validation-Based Checkpoint Selection

## `scaling_eval_scaffold.py`

```text
scaling_eval_scaffold.py
```

This is the first evaluation step after all required training runs have completed.

The script retrospectively evaluates the saved communication-round checkpoints on the validation set.

For the final SCAFFOLD evaluation, the complete validation set is loaded from:

```text
split_data["val"]["0"]
```

For every scaling point and repeated run, three checkpoints are selected independently:

```text
bestROC
    checkpoint with the highest validation ROC-AUC

bestPRROC
    checkpoint with the highest validation Average Precision

bestLoss
    checkpoint with the lowest weighted validation loss
```

The resulting structure is:

```text
result/splits_iid_scaling/
└── splits_iid_<N>_clients.json/
    └── SCAFFOLD/
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

Checkpoint selection is based exclusively on validation performance.

The test set is not used during this step.

Run:

```bash
python3 scaling_eval_scaffold.py
```

---

# Final Threshold-Independent Test Evaluation

## `final_test_set_eval_scaffold.py`

```text
final_test_set_eval_scaffold.py
```

Evaluates **only the checkpoints that have already been selected on validation data**.

The three selection categories are:

```text
bestROC
bestPRROC
bestLoss
```

For each selected checkpoint, the script evaluates performance on the fixed centralized test set.

It calculates:

* weighted cross-entropy loss,
* ROC-AUC,
* Average Precision,
* ROC curve,
* precision-recall curve,
* test-set class counts,
* test-set prevalence,
* checkpoint and validation-selection provenance.

No model, communication round, threshold, or hyperparameter is selected on the test set.

Run:

```bash
python3 final_test_set_eval_scaffold.py
```

The main output directory is:

```text
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
```

Important combined outputs include:

```text
all_test_results.csv
all_test_aggregate.csv
final_test_summary.json
test_set_info.json
```

Run-specific outputs include:

```text
test_metrics.json
test_curves.json
```

---

# Threshold-Independent Figures

## `plot-thr-indep-scaffold.py`

```text
plot-thr-indep-scaffold.py
```

Creates the final threshold-independent SCAFFOLD scalability figures from:

```text
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
    all_test_results.csv
```

The main absolute-performance figure contains:

```text
Panel A: ROC-AUC
Panel B: Average Precision
Panel C: Weighted loss
```

The checkpoint-selection criterion is matched to the reported metric:

```text
ROC-AUC       → bestROC
AP            → bestPRROC
Weighted loss → bestLoss
```

The plot shows the repeated runs together with the point-specific mean.

Run:

```bash
python3 plot-thr-indep-scaffold.py
```

The script also supports additional visualizations such as relative change from the two-client baseline and run-to-run stability.

---

# Run-to-Run Dispersion

## `table_plot_scaffold.py`

```text
table_plot_scaffold.py
```

Calculates run-to-run dispersion for the threshold-independent SCAFFOLD results.

Input:

```text
result/splits_iid_scaling/final_test_set_eval/SCAFFOLD/
    all_test_results.csv
```

For every scaling point, statistics are calculated across the five repeated runs for:

```text
ROC-AUC
Average Precision
Weighted loss
```

using the corresponding validation-selected checkpoint category.

The outputs include:

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
python3 table_plot_scaffold.py
```

---

# Threshold-Dependent Evaluation

## `evaluate-thr-dependent-scaffold.py`

```text
evaluate-thr-dependent-scaffold.py
```

The threshold-dependent analysis uses the checkpoint selected according to the highest validation Average Precision:

```text
bestPRROC
```

For each scaling point and repeated run, predictions are first generated on the validation set.

Two decision-threshold regimes are then evaluated.

### MCC-Optimal Operating Point

The decision threshold maximizing **validation MCC** is selected.

### Fixed Minimum-Recall Operating Point

A minimum validation recall requirement is specified before the evaluation.

The final study uses:

```text
validation recall ≥ 0.80
```

Among all thresholds satisfying this requirement, the threshold with the highest validation specificity is selected.

Both selected thresholds are subsequently applied **unchanged** to the test set.

No threshold is optimized on test data.

Run:

```bash
python3 evaluate-thr-dependent-scaffold.py --min-recall 0.80
```

The combined output is:

```text
result/splits_iid_scaling/final_threshold_analysis/SCAFFOLD/
└── all_threshold_results.csv
```

---

## `plot-thr-dep-scaffold.py`

```text
plot-thr-dep-scaffold.py
```

Visualizes the results produced by `evaluate-thr-dependent-scaffold.py`.

The script performs **no checkpoint, communication-round, or threshold selection**.

The MCC-optimal figure contains:

```text
Panel A: validation-selected decision threshold
Panel B: test MCC
Panel C: test recall and specificity
```

The fixed-recall figure contains:

```text
Panel A: validation-selected decision threshold
Panel B: test recall
Panel C: test specificity
```

Run:

```bash
python3 plot-thr-dep-scaffold.py
```

---

# Training-Dynamics Analysis

## `evaluate-training-dynamics-scaffold.py`

```text
evaluate-training-dynamics-scaffold.py
```

This is a retrospective **centralized-test-set** analysis of the SCAFFOLD training trajectories.

For every scaling point, run, and saved evaluated communication-round checkpoint, the script calculates test Average Precision.

The analysis is descriptive only and does not modify:

* training,
* hyperparameters,
* checkpoint selection,
* threshold selection,
* final test checkpoint selection.

Two quantities are derived.

### First Round Reaching 99% of Best Test AP

For every run, the script identifies the first evaluated communication round satisfying:

```text
test AP ≥ 0.99 × highest test AP observed in the run
```

This provides a descriptive measure of how quickly the run approached its best observed test performance.

### Late-Training Trend

The script fits an ordinary least-squares linear trend to test AP over the final training interval.

For the thesis analysis, the intended interval is:

```text
Rounds 70–80
```

The outputs are stored under:

```text
result/splits_iid_scaling/training_dynamics/SCAFFOLD/
```

including:

```text
test_set_info.json
all_round_test_ap.csv
training_dynamics_by_run.csv
training_dynamics_aggregate.csv
training_dynamics_summary.json
```

Run:

```bash
python3 evaluate-training-dynamics-scaffold.py
```

---

## `plot-training-dynamics-scaffold.py`

```text
plot-training-dynamics-scaffold.py
```

Visualizes the results produced by the training-dynamics evaluation.

The figure summarizes:

```text
Panel A:
First evaluated communication round reaching
99% of the best observed test AP

Panel B:
Late-training test-AP slope
```

Run:

```bash
python3 plot-training-dynamics-scaffold.py
```

---

# Complete Execution Order

For a complete reproduction of the final SCAFFOLD pipeline:

```bash
# 1. Prepare the fixed train / validation / test split
python3 federated_learning/tools/prepare_data.py \
    --csv data/diabetes.csv \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json

# 2. Normalize the features and calculate class weights
python3 federated_learning/tools/normalize_and_add_weights.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output data/diabetes_normalized.parquet \
    --pos-weight-boost 1.5

# 3. Generate the IID scaling partitions
python3 federated_learning/tools/create_iid_scaling_splits.py \
    --parquet data/diabetes.parquet \
    --stats data/norm_stats.json \
    --output-dir splits_iid_scaling \
    --seed 123

# 4. Run the SCAFFOLD scaling experiments
./federated_learning/tools/run_iid_scaling.sh

# Alternative: run one experiment using pyproject.toml
flwr run .

# 5. Select communication-round checkpoints using validation
python3 scaling_eval_scaffold.py

# 6. Perform final threshold-independent test evaluation
python3 final_test_set_eval_scaffold.py

# 7. Create threshold-independent figures
python3 plot-thr-indep-scaffold.py

# 8. Calculate run-to-run dispersion
python3 table_plot_scaffold.py

# 9. Select validation operating points and evaluate them on test
python3 evaluate-thr-dependent-scaffold.py --min-recall 0.80

# 10. Create threshold-dependent figures
python3 plot-thr-dep-scaffold.py

# 11. Evaluate training dynamics on the centralized test set
python3 evaluate-training-dynamics-scaffold.py

# 12. Create the training-dynamics figure
python3 plot-training-dynamics-scaffold.py
```

---

# Project File Overview

| File                                                    | Purpose                                                                           |
| ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `pyproject.toml`                                        | Main Flower, SCAFFOLD, client-training, data-path, and simulation configuration   |
| `federated_learning/server_app.py`                      | FedAvg model aggregation, global SCAFFOLD control variate, and checkpoint storage |
| `federated_learning/client_app.py`                      | Neural network, local SGD training, and client-specific SCAFFOLD control variate  |
| `federated_learning/task.py`                            | Loads client-specific normalized data and class weights                           |
| `federated_learning/tools/prepare_data.py`              | Creates the fixed global train/validation/test partition                          |
| `federated_learning/tools/normalize_and_add_weights.py` | Normalizes features and computes class weights                                    |
| `federated_learning/tools/make_splits.py`               | Helper functions used for client partitioning                                     |
| `federated_learning/tools/create_iid_scaling_splits.py` | Generates the IID client partitions for the scaling study                         |
| `federated_learning/tools/run_iid_scaling.sh`           | Runs repeated SCAFFOLD experiments across selected client counts                  |
| `scaling_eval_scaffold.py`                              | Selects best ROC-AUC, AP, and loss checkpoints using validation                   |
| `final_test_set_eval_scaffold.py`                       | Evaluates validation-selected checkpoints on the fixed final test set             |
| `plot-thr-indep-scaffold.py`                            | Creates threshold-independent SCAFFOLD scalability figures                        |
| `table_plot_scaffold.py`                                | Calculates and visualizes run-to-run dispersion                                   |
| `evaluate-thr-dependent-scaffold.py`                    | Selects operating-point thresholds on validation and evaluates them on test       |
| `plot-thr-dep-scaffold.py`                              | Creates threshold-dependent SCAFFOLD figures                                      |
| `evaluate-training-dynamics-scaffold.py`                | Retrospective test-AP training-dynamics analysis                                  |
| `plot-training-dynamics-scaffold.py`                    | Creates the training-dynamics figure                                              |

---

# Result Directory Structure

A simplified result structure is:

```text
result/
└── splits_iid_scaling/
    │
    ├── splits_iid_2_clients.json/
    │   └── SCAFFOLD/
    │       ├── all_rounds_run_1/
    │       ├── all_rounds_run_2/
    │       ├── ...
    │       ├── all_rounds_run_5/
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
    │       ├── final_test_summary.json
    │       └── test_set_info.json
    │
    ├── final_threshold_analysis/
    │   └── SCAFFOLD/
    │       └── all_threshold_results.csv
    │
    └── training_dynamics/
        └── SCAFFOLD/
            ├── test_set_info.json
            ├── all_round_test_ap.csv
            ├── training_dynamics_by_run.csv
            ├── training_dynamics_aggregate.csv
            └── training_dynamics_summary.json
```

---

# Useful Commands

Start one Flower run:

```bash
flwr run .
```

Start the scaling launcher:

```bash
./federated_learning/tools/run_iid_scaling.sh
```

Start the scaling launcher in the background:

```bash
nohup ./federated_learning/tools/run_iid_scaling.sh &
```

Inspect background output:

```bash
tail -f nohup.out
```

Stop remaining Ray processes:

```bash
ray stop --force
```

Display the available command-line options of a script:

```bash
python3 <script>.py --help
```

Examples:

```bash
python3 federated_learning/tools/create_iid_scaling_splits.py --help
python3 evaluate-thr-dependent-scaffold.py --help
python3 evaluate-training-dynamics-scaffold.py --help
```

---

# Methodological Notes

## Fixed Global Training Dataset

The amount of global training data remains constant throughout the scaling experiment.

Increasing the number of clients therefore represents increasing **client-level data fragmentation**, not an increase in available training data.

---

## IID Client Partitioning

Training observations are randomly and approximately evenly distributed across clients.

Labels are not used when assigning training observations to clients.

Local class-distribution differences therefore arise naturally from random fragmentation.

---

## Fixed SCAFFOLD Configuration

One SCAFFOLD configuration was selected in the 16,384-client reference setting and subsequently applied across the scalability study.

The final scaling range extends from 2 to 16,384 clients.

---

## Client Participation

SCAFFOLD uses approximately:

```text
0.75
```

client participation per communication round.

At the two-client configuration, both clients participate.

For larger client configurations, the requested participation fraction is converted to a whole number of participating clients.

---

## SCAFFOLD Control Variates

SCAFFOLD maintains a global control variate on the server and a persistent local control variate for each client.

During local training, the difference between these control variates corrects the local gradient before the optimizer step.

After local training, participating clients return their control-variate changes to the server, which uses them to update the global control variate.

---

## Validation-Based Checkpoint Selection

The final reported threshold-independent metrics use different validation-based checkpoint-selection criteria:

```text
ROC-AUC       → checkpoint with highest validation ROC-AUC
AP            → checkpoint with highest validation AP
Weighted loss → checkpoint with lowest validation loss
```

No checkpoint is selected using test performance.

---

## Threshold Selection

Threshold-dependent analyses use the `bestPRROC` checkpoint.

The decision threshold is determined on validation data according to either:

```text
maximum validation MCC
```

or:

```text
validation recall ≥ 0.80
followed by maximum validation specificity
```

The selected threshold is then applied unchanged to the test set.

---

## Final Test Set

The same centralized test set is used for every scaling point and run.

Its membership is defined by:

```text
norm_stats.json["test_idx"]
```

The test set does not influence model training, hyperparameter selection, checkpoint selection, or decision-threshold selection.

---

## Training Dynamics

Training dynamics are evaluated retrospectively on the centralized **test set**.

The analysis determines:

```text
first evaluated round reaching 99% of best test AP
```

and:

```text
late-training test-AP slope
```

The intended late-training interval for the 80-round SCAFFOLD runs is rounds 70–80.

This analysis is descriptive and does not affect model or threshold selection.

---
