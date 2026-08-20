# Federated Edge Learning — FedAvg Scalability Study

This repository contains the experimental code for the **FedAvg scalability study** conducted as part of a bachelor thesis on federated edge learning for diabetes screening.

The implementation uses **PyTorch** for model training and **Flower (FLwr)** for federated orchestration and simulation.

The central experiment investigates how federated learning behaves when a **fixed global training dataset is distributed across an increasing number of clients**. Increasing the number of clients therefore does not add training data, but progressively fragments the same dataset into smaller local client datasets.

The final FedAvg scalability study evaluates:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512,
1,024, 2,048, 4,096, 8,192, 16,384 clients
```

Each scaling point is repeated **five times**.

FedAvg performs local model training on participating clients and aggregates the resulting client models on the server using sample-size-weighted averaging.

> **Legacy naming**
>
> Several internal result directories and legacy identifiers still use the name `FedProx`. These results correspond to the experiments reported as **FedAvg** in the final thesis. The historical identifier is retained where required for compatibility with the existing result structure.

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

The final FedAvg experiment follows this workflow:

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
        ├── analyze_iid_splits_one_row_per_split.py
        │
        ▼
FedAvg training
client_app.py + server_app.py + task.py
        │
        ▼
Saved model checkpoints
        │
        ▼
scaling_evaluation_fedavg.py
        │
        ▼
Validation-based checkpoint selection
   ├── bestROC
   ├── bestPRROC
   └── bestLoss
        │
        ▼
final_test_set_eval_fedavg.py
        │
        ▼
Final threshold-independent test results
        │
        ├── plot-thr-indep-fedavg.py
        └── table_plot_fedavg.py
```

Two additional analyses branch from the saved results.

Threshold-dependent evaluation:

```text
bestPRROC checkpoint
        │
        ▼
evaluate-thr-dependent-fedavg.py
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
plot-thr-dependent-fedavg.py
```

Training-dynamics analysis:

```text
Saved checkpoints from evaluated communication rounds
        │
        ▼
evaluate-training-dynamics-fedavg.py
        │
        ▼
Test AP across communication rounds
   ├── First evaluated round reaching 99% of best test AP
   └── Late-training test-AP slope
        │
        ▼
plot-training-dynamics-fedavg.py
```

Additional analysis and comparison scripts include:

```text
plot_strategy_mcc.py
    Configuration-selection analysis at 16,384 clients

plot-combined-strategy-comparison.py
    Cross-strategy comparison of FedAvg, SCAFFOLD, and FedAdam
```

The validation set is used for checkpoint and threshold selection. The test set is not used to select final checkpoints, decision thresholds, or hyperparameters.

The training-dynamics analysis evaluates intermediate checkpoints retrospectively on the test set for descriptive analysis only and does not affect model or threshold selection.

---

# Core Federated Learning Components

## `server_app.py`

```text
federated_learning/server_app.py
```

Defines the Flower server application and the FedAvg strategy.

The server is responsible for:

* coordinating the participating clients,
* distributing the current global model,
* aggregating the locally updated client models,
* maintaining the global model across communication rounds,
* storing model checkpoints and round information.

FedAvg aggregates the participating client models using the number of local training samples as weights.

The saved communication-round checkpoints are later used for validation-based checkpoint selection and for the retrospective training-dynamics analysis.

---

## `client_app.py`

```text
federated_learning/client_app.py
```

Defines the Flower client application, neural network, and local training procedure.

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
2. loads its assigned local training observations,
3. performs the configured local optimization,
4. applies gradient clipping,
5. returns the updated model parameters to the server.

Local training uses class-weighted cross-entropy.

The selected FedAvg configuration uses SGD with momentum together with weight decay and gradient clipping.

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

The Parquet file is already normalized before training, so no additional feature normalization is performed during local training.

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

Other partitioning functionality contained in the file is not required for reproducing the final FedAvg scalability study.

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

Importantly, **labels are not used when assigning training observations to clients**. Label information is inspected only afterwards to describe the resulting local class distributions.

For the final FedAvg study, the relevant scaling points are:

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

The generated files follow the naming convention:

```text
splits_iid_scaling/
├── splits_iid_2_clients.json
├── splits_iid_4_clients.json
├── splits_iid_8_clients.json
├── ...
├── splits_iid_8192_clients.json
└── splits_iid_16384_clients.json
```

The global amount of training data remains constant across these files. Only the number and size of the local client datasets change.

---

## `analyze_iid_splits_one_row_per_split.py`

```text
analyze_iid_splits_one_row_per_split.py
```

Provides a descriptive analysis of the generated IID scaling partitions.

The script creates one summary row per scaling point and reports quantities including:

* number of clients,
* total number of training samples,
* mean, minimum, and maximum local dataset size,
* mean and variation of positive samples per client,
* minimum and maximum positive samples per client,
* number of clients without positive observations,
* percentage of clients without positive observations,
* global positive-class rate.

This script is used to characterize how local data availability and class composition change as the fixed training dataset becomes increasingly fragmented.

It is an analysis utility and is not required to start federated training.

Run:

```bash
python3 analyze_iid_splits_one_row_per_split.py
```

---

# Configuring FedAvg

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
lr-after
lr-after-round
warmup-rounds
warmup-lr-start
warmup-lr-end
weight-decay
clip-grad-norm
pos-weight-boost

prepared-parquet
norm-stats-json
split-path
run-tag
```

The final FedAvg scalability study uses:

```text
Communication rounds:         80
Target client participation:  0.80
Repeated runs:                5
Scaling range:                2–16,384 clients
Local epochs:                 1
Gradient clipping norm:       4.0
Weight decay:                 5e-4
Positive-class boost:         1.5
```

At the two-client scaling point, both clients participate.

For all larger client configurations, the target participation fraction is converted to a whole number of clients by rounding down.

For example:

```text
2 clients       → 2 participating clients
4 clients       → 3
8 clients       → 6
16 clients      → 12
32 clients      → 25
...
```

---

## Learning-Rate Schedule

The selected FedAvg configuration uses:

```text
Rounds 1–12:
linear warm-up from 3e-3 to 8e-2

Rounds 13–59:
constant learning rate of 8e-2

Rounds 60–80:
cosine-annealing cool-down from 8e-2 to 3e-2
```

The local optimizer configuration is:

```text
Optimizer:          SGD
Momentum:           0.9
Weight decay:       5e-4
Local epochs:       1
Gradient clipping:  4.0
```

The configuration used for an actual scaling run is determined by the values in `pyproject.toml` together with values explicitly overridden through Flower's `--run-config`.

---

# Starting FedAvg Training

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
learning-rate schedule
weight-decay
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
7. store run-specific logs,
8. stop remaining Ray processes between experiments.

For the complete final FedAvg study, the client-count list should contain:

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

or through the run-specific files under:

```text
logs/iid_scaling/
```

---

# Validation-Based Checkpoint Selection

## `scaling_evaluation_fedavg.py`

```text
scaling_evaluation_fedavg.py
```

This is the first evaluation step after all required training runs have completed.

The script retrospectively evaluates the saved communication-round checkpoints on the validation set.

For the final FedAvg evaluation, the complete validation set is loaded from:

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

The selected checkpoints are stored under the legacy strategy directory:

```text
result/splits_iid_scaling/
└── splits_iid_<N>_clients.json/
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

Checkpoint selection is based exclusively on validation performance.

The test set is not used during this step.

Run:

```bash
python3 scaling_evaluation_fedavg.py
```

---

# Final Threshold-Independent Test Evaluation

## `final_test_set_eval_fedavg.py`

```text
final_test_set_eval_fedavg.py
```

Evaluates **only the checkpoints that have already been selected using validation data**.

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
python3 final_test_set_eval_fedavg.py
```

The main output directory retains the legacy strategy identifier:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/
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

## `plot-thr-indep-fedavg.py`

```text
plot-thr-indep-fedavg.py
```

Creates the final threshold-independent FedAvg scalability figures from:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/
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
python3 plot-thr-indep-fedavg.py
```

The script also supports additional visualizations such as relative change from the two-client baseline and run-to-run stability.

---

# Run-to-Run Dispersion

## `table_plot_fedavg.py`

```text
table_plot_fedavg.py
```

Calculates run-to-run dispersion for the threshold-independent FedAvg results.

Input:

```text
result/splits_iid_scaling/final_test_set_eval/FedProx/
    all_test_results.csv
```

For every scaling point, statistics are calculated across the five repeated runs for:

```text
ROC-AUC
Average Precision
Weighted loss
```

using the corresponding validation-selected checkpoint category.

Calculated quantities include:

* mean,
* standard deviation,
* coefficient of variation,
* minimum and maximum,
* maximum relative deviation of an individual run from the scaling-point mean.

Important outputs include:

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
python3 table_plot_fedavg.py
```

---

# Threshold-Dependent Evaluation

## `evaluate-thr-dependent-fedavg.py`

```text
evaluate-thr-dependent-fedavg.py
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
python3 evaluate-thr-dependent-fedavg.py --min-recall 0.80
```

The combined output is stored under:

```text
result/splits_iid_scaling/final_threshold_analysis/FedProx/
└── all_threshold_results.csv
```

---

## `plot-thr-dependent-fedavg.py`

```text
plot-thr-dependent-fedavg.py
```

Visualizes the results produced by `evaluate-thr-dependent-fedavg.py`.

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
Panel B: test recall and specificity
```

The recall requirement is imposed on validation data; the corresponding test recall is reported after transferring the selected threshold unchanged.

Run:

```bash
python3 plot-thr-dependent-fedavg.py
```

---

# Training-Dynamics Analysis

## `evaluate-training-dynamics-fedavg.py`

```text
evaluate-training-dynamics-fedavg.py
```

This is a retrospective **centralized-test-set** analysis of the FedAvg training trajectories.

For every scaling point, run, and saved communication-round checkpoint, the script calculates test Average Precision.

The analysis is descriptive only and does not modify:

* training,
* hyperparameters,
* validation-based checkpoint selection,
* threshold selection,
* final model selection.

Two quantities are derived.

### First Round Reaching 99% of Best Test AP

For every run, the script identifies the first evaluated communication round satisfying:

```text
test AP ≥ 0.99 × highest test AP observed within the run
```

This provides a descriptive measure of how quickly the run approached its best observed test performance.

### Late-Training Trend

The script fits an ordinary least-squares linear trend to test AP over:

```text
Rounds 70–80 inclusive
```

This window contains 11 AP observations spanning ten round-to-round intervals.

The outputs are stored under:

```text
result/splits_iid_scaling/training_dynamics/FedProx/
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
python3 evaluate-training-dynamics-fedavg.py
```

---

## `plot-training-dynamics-fedavg.py`

```text
plot-training-dynamics-fedavg.py
```

Visualizes the results produced by the training-dynamics evaluation.

The figure summarizes:

```text
Panel A:
First evaluated communication round reaching
99% of the best observed test AP

Panel B:
Late-training test-AP slope over rounds 70–80
```

Small points represent individual runs and connected markers represent the mean across repeated runs.

Run:

```bash
python3 plot-training-dynamics-fedavg.py
```

---

# Configuration-Selection Plot

## `plot_strategy_mcc.py`

```text
plot_strategy_mcc.py
```

Visualizes the optimization-configuration experiments conducted at the fixed **16,384-client reference setting** before the final scalability study.

For FedAvg, five candidate configurations were evaluated.

The analysis is based on validation MCC over communication rounds and summarizes properties such as:

* validation MCC,
* late-training plateau performance,
* late-training variation,
* convergence speed.

The selected FedAvg configuration was subsequently held fixed across the scalability study.

This script belongs to the **configuration-selection stage** and is not part of the final scaling evaluation pipeline itself.

Run:

```bash
python3 plot_strategy_mcc.py
```

---

# Cross-Strategy Comparison

## `plot-combined-strategy-comparison.py`

```text
plot-combined-strategy-comparison.py
```

Creates the final overview figures comparing **FedAvg, SCAFFOLD, and FedAdam**.

The script combines the aggregated threshold-independent and threshold-dependent results of all three strategies.

By default, only client scaling points available for all strategies are included. The common comparison range is therefore:

```text
2–16,384 clients
```

The script creates three comparison figures.

### MCC-Optimal Operating Point

```text
Panel A: mean validation-selected decision threshold
Panel B: mean test MCC
Panel C: mean test recall and specificity
```

### Fixed Validation-Recall Operating Point

```text
Panel A: mean validation-selected decision threshold
Panel B: mean test recall and specificity
```

### Threshold-Independent Performance

```text
Panel A: mean test ROC-AUC
Panel B: mean test Average Precision
Panel C: mean weighted test loss
```

Only the strategy-level mean curves are shown; individual repeated-run points are omitted from these overview figures.

The script internally retains the legacy identifier `FedProx` for the FedAvg result files.

Run:

```bash
python3 plot-combined-strategy-comparison.py
```

The default input and output directories are:

```text
comparison_input/
comparison_output/
```

The expected aggregated inputs include:

```text
FedProx_threshold_dependent_aggregate.csv
SCAFFOLD_threshold_dependent_aggregate.csv
FedAdam_threshold_dependent_aggregate.csv

FedProx_all_test_aggregate.csv
SCAFFOLD_all_test_aggregate.csv
FedAdam_all_test_aggregate.csv
```

---

# Complete Execution Order

For a complete reproduction of the final FedAvg pipeline:

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

# 4. Analyze local data availability and class composition
python3 analyze_iid_splits_one_row_per_split.py

# 5. Run the FedAvg scaling experiments
./federated_learning/tools/run_iid_scaling.sh

# Alternative: run one experiment using pyproject.toml
flwr run .

# 6. Select communication-round checkpoints using validation
python3 scaling_evaluation_fedavg.py

# 7. Perform final threshold-independent test evaluation
python3 final_test_set_eval_fedavg.py

# 8. Create threshold-independent figures
python3 plot-thr-indep-fedavg.py

# 9. Calculate run-to-run dispersion
python3 table_plot_fedavg.py

# 10. Select validation operating points and evaluate them on test
python3 evaluate-thr-dependent-fedavg.py --min-recall 0.80

# 11. Create threshold-dependent figures
python3 plot-thr-dependent-fedavg.py

# 12. Evaluate training dynamics on the centralized test set
python3 evaluate-training-dynamics-fedavg.py

# 13. Create the training-dynamics figure
python3 plot-training-dynamics-fedavg.py

# Optional: configuration-selection visualization
python3 plot_strategy_mcc.py

# Optional: cross-strategy overview figures
python3 plot-combined-strategy-comparison.py
```

---

# Project File Overview

| File                                                    | Purpose                                                                          |
| ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `pyproject.toml`                                        | Main Flower, FedAvg, client-training, data-path, and simulation configuration    |
| `federated_learning/server_app.py`                      | FedAvg aggregation, client coordination, and checkpoint storage                  |
| `federated_learning/client_app.py`                      | Neural network and local client training                                         |
| `federated_learning/task.py`                            | Loads client-specific normalized data and class weights                          |
| `federated_learning/tools/prepare_data.py`              | Creates the fixed global train/validation/test partition                         |
| `federated_learning/tools/normalize_and_add_weights.py` | Normalizes features and computes class weights                                   |
| `federated_learning/tools/make_splits.py`               | Helper functions used for client partitioning                                    |
| `federated_learning/tools/create_iid_scaling_splits.py` | Generates the IID client partitions for the scaling study                        |
| `analyze_iid_splits_one_row_per_split.py`               | Summarizes local sample availability and class composition across scaling points |
| `federated_learning/tools/run_iid_scaling.sh`           | Runs repeated FedAvg experiments across selected client counts                   |
| `scaling_evaluation_fedavg.py`                          | Selects best ROC-AUC, AP, and loss checkpoints using validation                  |
| `final_test_set_eval_fedavg.py`                         | Evaluates validation-selected checkpoints on the fixed final test set            |
| `plot-thr-indep-fedavg.py`                              | Creates threshold-independent FedAvg scalability figures                         |
| `table_plot_fedavg.py`                                  | Calculates and visualizes run-to-run dispersion                                  |
| `evaluate-thr-dependent-fedavg.py`                      | Selects operating-point thresholds on validation and evaluates them on test      |
| `plot-thr-dependent-fedavg.py`                          | Creates threshold-dependent FedAvg figures                                       |
| `evaluate-training-dynamics-fedavg.py`                  | Retrospective test-AP training-dynamics analysis                                 |
| `plot-training-dynamics-fedavg.py`                      | Creates the FedAvg training-dynamics figure                                      |
| `plot_strategy_mcc.py`                                  | Visualizes optimization-configuration selection at 16,384 clients                |
| `plot-combined-strategy-comparison.py`                  | Creates the final cross-strategy overview figures                                |

---

# Result Directory Structure

A simplified result structure is:

```text
result/
└── splits_iid_scaling/
    │
    ├── splits_iid_2_clients.json/
    │   └── FedProx/
    │       ├── all_rounds_FedProx_1/
    │       ├── all_rounds_FedProx_2/
    │       ├── ...
    │       ├── all_rounds_FedProx_5/
    │       ├── bestROC/
    │       ├── bestPRROC/
    │       └── bestLoss/
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
    │       └── test_set_info.json
    │
    ├── final_threshold_analysis/
    │   └── FedProx/
    │       └── all_threshold_results.csv
    │
    └── training_dynamics/
        └── FedProx/
            ├── test_set_info.json
            ├── all_round_test_ap.csv
            ├── training_dynamics_by_run.csv
            ├── training_dynamics_aggregate.csv
            └── training_dynamics_summary.json
```

The `FedProx` identifier is retained in these result paths for compatibility with the experiments that are reported as **FedAvg** in the final thesis.

Some older runs may use alternative round-directory names. The final evaluation scripts support the retained legacy layouts where required.

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
python3 evaluate-thr-dependent-fedavg.py --help
python3 evaluate-training-dynamics-fedavg.py --help
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

Local differences in class composition therefore arise naturally from random fragmentation.

---

## Fixed FedAvg Configuration

One FedAvg configuration was selected in the 16,384-client reference setting and subsequently applied unchanged across the scalability study.

The final scaling range extends from 2 to 16,384 clients.

---

## Client Participation

FedAvg uses a target client participation fraction of:

```text
0.80
```

At the two-client configuration, both clients participate.

For larger client configurations, the requested participation fraction is converted to a whole number of participating clients by rounding down.

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

For the 80-round FedAvg runs, the late-training interval is rounds 70–80 inclusive.

This analysis is descriptive and does not affect model or threshold selection.

---
