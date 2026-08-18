# Federated Edge Learning — FedAdam Scalability Study

This repository contains the experimental code for the **FedAdam scalability study** conducted as part of a bachelor thesis on federated edge learning for diabetes screening.

The implementation uses **PyTorch** for model training and **Flower (FLwr)** for federated orchestration and simulation.

The central experiment investigates how federated learning behaves when a **fixed global training dataset is distributed across an increasing number of clients**. Increasing the number of clients therefore does not add training data, but progressively fragments the same dataset into smaller local client datasets.

The final FedAdam scalability study evaluates:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512,
1,024, 2,048, 4,096, 8,192, 16,384, 32,768 clients
```

Each scaling point is repeated **five times**.

FedAdam differs from FedAvg primarily through its adaptive **server-side optimization**. Clients perform local model training, while the server uses first- and second-moment estimates of the aggregated client updates to determine the global update.

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

The final FedAdam experiment follows this workflow:

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
adjust_val_distribution.py
        │
        ▼
Centralized validation mapping
        │
        ▼
FedAdam training
client_app.py + server_app.py + task.py
        │
        ▼
Saved model checkpoint for each communication round
        │
        ▼
scaling_eval.py
        │
        ▼
Validation-based checkpoint selection
   ├── bestROC
   ├── bestPRROC
   └── bestLoss
        │
        ▼
final_test_set_eval.py
        │
        ▼
Final threshold-independent test results
        │
        ├── plot-thr-indep.py
        └── table_plot.py
```

Two additional analyses branch from the saved results.

Threshold-dependent evaluation:

```text
bestPRROC checkpoint
        │
        ▼
evaluate-thr-dependent-fedadam.py
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
plot-thr-dependent-fedadam.py
```

Training-dynamics analysis:

```text
Saved checkpoint from each communication round
        │
        ▼
evaluate-training-dynamics-fedadam.py
        │
        ▼
Validation AP across communication rounds
   ├── First evaluated round reaching 99% of best validation AP
   └── Late-training validation-AP slope
        │
        ▼
plot-training-dynamics-fedadam.py
```

The validation set is used for checkpoint and threshold selection. The test set is not used to select models, communication rounds, thresholds, or hyperparameters.

---

# Core Federated Learning Components

## `server_app.py`

```text
federated_learning/server_app.py
```

Defines the Flower server application and the FedAdam strategy.

The custom strategy extends Flower's `FedAdam` implementation. After receiving the locally trained client models, the server performs the FedAdam server-side optimization step.

The server is also responsible for:

* coordinating the participating clients,
* maintaining the global model,
* applying the FedAdam server optimizer,
* performing centralized validation,
* saving model checkpoints,
* storing validation metrics for the communication rounds.

The FedAdam-specific server parameters are read from the Flower run configuration:

```text
eta
eta-l
beta-1
beta-2
tau
```

where:

* `eta` is the server learning rate,
* `eta-l` is the FedAdam client learning-rate parameter,
* `beta-1` controls the first-moment estimate,
* `beta-2` controls the second-moment estimate,
* `tau` is the FedAdam stability parameter.

The corresponding values are configured in `pyproject.toml`.

The server loads the complete validation set using the `centralized_val_row_ids` stored in the active split file. Validation is therefore performed directly on the server rather than by aggregating client-side validation results.

Each communication-round model is stored as a PyTorch checkpoint, for example:

```text
result/splits_iid_scaling/
└── splits_iid_4096_clients.json/
    └── FedAdam/
        └── all_rounds_run_1/
            ├── model_round_1_run_1.pt
            ├── model_round_2_run_1.pt
            ├── ...
            └── model_round_45_run_1.pt
```

These checkpoints are later used for validation-based checkpoint selection and the retrospective training-dynamics analysis.

---

## `client_app.py`

```text
federated_learning/client_app.py
```

Defines the Flower client application, neural network, and local training procedure.

For every selected client, the client:

1. receives the current global model,
2. loads its assigned local training observations,
3. performs the configured local training,
4. returns the updated model parameters to the server.

The neural network is a multilayer perceptron for binary diabetes classification.

Local optimization parameters such as the learning rate, learning-rate schedule, weight decay, gradient clipping, batch size, and number of local epochs are configured through `pyproject.toml`.

FedAdam itself is applied on the **server side**. The client-side training therefore produces the local model updates on which the server subsequently performs adaptive FedAdam optimization.

---

## `task.py`

```text
federated_learning/task.py
```

Contains the data-loading utilities used by the Flower clients and the centralized validation procedure.

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

The file also provides:

```text
load_centralized_val(...)
```

which is used by the server and later validation-based evaluation scripts to load the complete centralized validation set.

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

Other partitioning functionality contained in the file is not required for reproducing the final FedAdam scalability study.

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

For the final FedAdam study, the relevant scaling points are:

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
32,768
```

The corresponding files follow the naming convention:

```text
splits_iid_scaling/
├── splits_iid_2_clients.json
├── splits_iid_4_clients.json
├── splits_iid_8_clients.json
├── ...
├── splits_iid_16384_clients.json
└── splits_iid_32768_clients.json
```

The global amount of training data remains constant across these files. Only the number and size of the local client datasets changes.

---

## `analyze_client_label_distribution.py`

```text
federated_learning/tools/analyze_client_label_distribution.py
```

Provides a descriptive analysis of the generated client partitions.

It can be used to inspect quantities such as:

* number of samples per client,
* positive samples per client,
* local positive-class proportions,
* variation in local class composition,
* clients containing no positive observations.

This script is an analysis utility and is not required to start the federated training.


---

# Configuring FedAdam

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
weight-decay
clip-grad-norm
pos-weight-boost

eta
eta-l
beta-1
beta-2
tau

prepared-parquet
norm-stats-json
split-path
run-tag
```

The final FedAdam scalability study uses:

```text
Communication rounds:       45
Target client participation: 0.80
Repeated runs:               5
Scaling range:               2–32,768 clients
Positive-class boost:        1.5
```

At the two-client scaling point, both clients participate. For larger client configurations, the target participation fraction is converted to a whole number of clients.

The selected FedAdam configuration uses the following client-side learning-rate schedule:

```text
Rounds 1–8:
linear warm-up from 1e-3 to 5e-2

Rounds 9–39:
constant at 5e-2

Rounds 40–45:
cosine-annealing cool-down from 5e-2 to 2e-2
```

The selected server-side FedAdam settings are:

```text
eta     = 0.1
eta-l   = 0.1
beta-1  = 0.9
beta-2  = 0.99
tau     = 1e-3
```

with:

```text
weight decay          = 5e-4
gradient clipping     = 4.0
positive-class boost  = 1.5
```

The configuration used for an actual run is always determined by the values in `pyproject.toml` together with any values explicitly overridden through Flower's `--run-config`.

---

# Starting FedAdam Training

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
local optimization settings
FedAdam server parameters
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

For the complete final FedAdam study, the client-count list should contain:

```bash
CLIENT_COUNTS=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)
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

or through the run-specific log files produced by the launcher.

---

# Validation-Based Checkpoint Selection

## `scaling_eval.py`

```text
scaling_eval.py
```

This is the first evaluation step after all required training runs have completed.

During training, a model checkpoint is saved for each communication round.

`scaling_eval.py` retrospectively evaluates these saved checkpoints on the complete centralized validation set.

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
    └── FedAdam/
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
python3 scaling_eval.py
```

---

# Final Threshold-Independent Test Evaluation

## `final_test_set_eval.py`

```text
final_test_set_eval.py
```

Evaluates **only the checkpoints that have already been selected on the centralized validation set**.

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
python3 final_test_set_eval.py
```

The main output directory is:

```text
result/splits_iid_scaling/final_test_set_eval/FedAdam/
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

## `plot-thr-indep.py`

```text
plot-thr-indep.py
```

Creates the final threshold-independent FedAdam scalability figures from:

```text
result/splits_iid_scaling/final_test_set_eval/FedAdam/
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
python3 plot-thr-indep.py
```

The script also supports additional visualizations such as relative change from the two-client baseline and run-to-run stability.

---

# Run-to-Run Dispersion

## `table_plot.py`

```text
table_plot.py
```

Calculates run-to-run dispersion for the threshold-independent FedAdam results.

Input:

```text
result/splits_iid_scaling/final_test_set_eval/FedAdam/
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
python3 table_plot.py
```

---

# Threshold-Dependent Evaluation

## `evaluate-thr-dependent-fedadam.py`

```text
evaluate-thr-dependent-fedadam.py
```

The threshold-dependent analysis uses the checkpoint selected according to the highest validation Average Precision:

```text
bestPRROC
```

For each scaling point and repeated run, predictions are first generated on the centralized validation set.

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
python3 evaluate-thr-dependent-fedadam.py --min-recall 0.80
```

The combined output is:

```text
result/splits_iid_scaling/final_threshold_analysis/FedAdam/
└── all_threshold_results.csv
```

---

## `plot-thr-dependent-fedadam.py`

```text
plot-thr-dependent-fedadam.py
```

Visualizes the results produced by `evaluate-thr-dependent-fedadam.py`.

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
python3 plot-thr-dependent-fedadam.py
```

---

# Training-Dynamics Analysis

## `evaluate-training-dynamics-fedadam.py`

```text
evaluate-training-dynamics-fedadam.py
```

This is a retrospective **centralized-validation** analysis of the FedAdam training trajectories.

For every scaling point, run, and saved communication-round checkpoint, the script calculates validation Average Precision.

The analysis is descriptive only and does not modify:

* training,
* hyperparameters,
* checkpoint selection,
* threshold selection,
* final test evaluation.

Two quantities are derived.

### First Round Reaching 99% of Best Validation AP

For every run, the script identifies the first evaluated communication round satisfying:

```text
validation AP ≥ 0.99 × highest validation AP observed in the run
```

This provides a descriptive measure of how quickly the run approached its best observed validation performance.

### Late-Training Trend

The script fits an ordinary least-squares linear trend to validation AP over the final training interval.

For a 45-round FedAdam run, the default interval is:

```text
Rounds 35–45
```

This contains 11 checkpoints and represents a ten-round interval.

The outputs are stored under:

```text
result/splits_iid_scaling/training_dynamics/FedAdam/
```

including:

```text
centralized_validation_set_info.json
all_round_validation_ap.csv
training_dynamics_by_run.csv
training_dynamics_aggregate.csv
training_dynamics_summary.json
```

Run:

```bash
python3 evaluate-training-dynamics-fedadam.py
```

---

## `plot-training-dynamics-fedadam.py`

```text
plot-training-dynamics-fedadam.py
```

Visualizes the results produced by the training-dynamics evaluation.

The figure summarizes:

```text
Panel A:
First evaluated communication round reaching
99% of the best observed validation AP

Panel B:
Late-training validation-AP slope
```

Run:

```bash
python3 plot-training-dynamics-fedadam.py
```

---

# Complete Execution Order

For a complete reproduction of the final FedAdam pipeline:

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

# 4. Prepare centralized validation IDs
python3 adjust_val_distribution.py

# 5. Run the FedAdam scaling experiments
./federated_learning/tools/run_iid_scaling.sh

# Alternative: run one experiment using pyproject.toml
flwr run .

# 6. Select communication-round checkpoints using validation
python3 scaling_eval.py

# 7. Perform final threshold-independent test evaluation
python3 final_test_set_eval.py

# 8. Create threshold-independent figures
python3 plot-thr-indep.py

# 9. Calculate run-to-run dispersion
python3 table_plot.py

# 10. Select validation operating points and evaluate them on test
python3 evaluate-thr-dependent-fedadam.py --min-recall 0.80

# 11. Create threshold-dependent figures
python3 plot-thr-dependent-fedadam.py

# 12. Evaluate training dynamics on centralized validation
python3 evaluate-training-dynamics-fedadam.py

# 13. Create the training-dynamics figure
python3 plot-training-dynamics-fedadam.py
```

---

# Project File Overview

| File                                                            | Purpose                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `pyproject.toml`                                                | Main Flower, FedAdam, client-training, data-path, and Ray simulation configuration               |
| `federated_learning/server_app.py`                              | FedAdam server optimization, centralized validation, client coordination, and checkpoint storage |
| `federated_learning/client_app.py`                              | Neural network and local client training                                                         |
| `federated_learning/task.py`                                    | Loads client-specific normalized data, class weights, and centralized validation data            |
| `federated_learning/tools/prepare_data.py`                      | Creates the fixed global train/validation/test partition                                         |
| `federated_learning/tools/normalize_and_add_weights.py`         | Normalizes features and computes class weights                                                   |
| `federated_learning/tools/make_splits.py`                       | Helper functions used for client partitioning                                                    |
| `federated_learning/tools/create_iid_scaling_splits.py`         | Generates the IID client partitions for the scaling study                                        |
| `federated_learning/tools/analyze_client_label_distribution.py` | Describes local dataset and class-distribution characteristics                                   |
| `adjust_val_distribution.py`                                    | Makes the complete validation set available as centralized validation IDs                        |
| `federated_learning/tools/run_iid_scaling.sh`                   | Runs repeated FedAdam experiments across selected client counts                                  |
| `scaling_eval.py`                                               | Selects best ROC-AUC, AP, and loss checkpoints using centralized validation                      |
| `final_test_set_eval.py`                                        | Evaluates validation-selected checkpoints on the fixed final test set                            |
| `plot-thr-indep.py`                                             | Creates threshold-independent FedAdam scalability figures                                        |
| `table_plot.py`                                                 | Calculates and visualizes run-to-run dispersion                                                  |
| `evaluate-thr-dependent-fedadam.py`                             | Selects operating-point thresholds on validation and evaluates them on test                      |
| `plot-thr-dependent-fedadam.py`                                 | Creates threshold-dependent FedAdam figures                                                      |
| `evaluate-training-dynamics-fedadam.py`                         | Retrospective validation-AP training-dynamics analysis                                           |
| `plot-training-dynamics-fedadam.py`                             | Creates the training-dynamics figure                                                             |

---

# Result Directory Structure

A simplified result structure is:

```text
result/
└── splits_iid_scaling/
    │
    ├── splits_iid_2_clients.json/
    │   └── FedAdam/
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
    ├── splits_iid_32768_clients.json/
    │   └── FedAdam/
    │
    ├── final_test_set_eval/
    │   └── FedAdam/
    │       ├── all_test_results.csv
    │       ├── all_test_aggregate.csv
    │       ├── final_test_summary.json
    │       └── test_set_info.json
    │
    ├── final_threshold_analysis/
    │   └── FedAdam/
    │       └── all_threshold_results.csv
    │
    └── training_dynamics/
        └── FedAdam/
            ├── centralized_validation_set_info.json
            ├── all_round_validation_ap.csv
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
python3 evaluate-thr-dependent-fedadam.py --help
python3 evaluate-training-dynamics-fedadam.py --help
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

## Fixed FedAdam Configuration

One FedAdam configuration was selected in the 16,384-client reference setting and subsequently applied unchanged across the scalability study.

The additional 32,768-client experiment extends the FedAdam scaling range beyond the range evaluated for FedAvg and SCAFFOLD.

---

## Client Participation

FedAdam uses a target client participation fraction of:

```text
0.80
```

At the two-client configuration, both clients participate.

For larger client configurations, the requested participation fraction is converted to a whole number of participating clients.


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

Training dynamics are evaluated retrospectively on the centralized **validation set**.

The analysis determines:

```text
first evaluated round reaching 99% of best validation AP
```

and:

```text
late-training validation-AP slope
```

For the 45-round FedAdam runs, the late-training interval is rounds 35–45.

This analysis is descriptive and does not affect model selection.

---

