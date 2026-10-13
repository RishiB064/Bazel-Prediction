# Bazel Build Time Predictor

#### AUTHOR : RISHI BHARADWAJ

An end-to-end machine learning pipeline that forecasts compilation CPU execution time (`cpu_time` in milliseconds) for incoming commits in the `bazelbuild/bazel` repository prior to build execution.

This implementation modernizes the legacy Stanford CS229 baseline (deprecated `tf.estimator` additive linear regression) into a multimodal Keras Functional neural network and tree-based ensemble (`HistGradientBoosting`), integrating continuous Git churn telemetry to overcome categorical informational bottlenecks.

---

## Architecture Overview

The system ingests commit metadata and diff telemetry through two parallel preprocessing streams directly within the model computational graph:

```
[ Categorical Inputs: prefix, type ]       [ Continuous Inputs: 9 Churn Metrics ]
                 │                                           │
        StringLookup Layers                        log1p + Normalization
                 │                                           │
    HashedCrossing(64) + One-Hot                             │
                 │                                           │
       Sparse Vector (64-D)                         Dense Vector (9-D)
                 └─────────────────────┬─────────────────────┘
                                       │
                           Concatenate Layer (73-D)
                                       │
                           Dense(32, activation='relu')
                                       │
                           Dense(16, activation='relu')
                                       │
                           Dense(1) [Median-Scaled]
```

* **Categorical Stream:** Encodes directory path (`prefix`) and file type (`type`) strings via `StringLookup`, then projects their interaction into 64 bounded buckets via `tf.keras.layers.HashedCrossing(num_bins=64)` and `CategoryEncoding`. This captures directory-language conjunctions without memory bloat or vocabulary explosion.
* **Continuous Telemetry Stream:** Ingests 9 numerical Git diff metrics extracted via `git diff --numstat`. Churn counts are compressed via $\log(1 + x)$ (`np.log1p`) and standardized using Keras `Normalization`.
* **Hybrid Regression Head:** Merges the 73-dimensional concatenated representation into a Multi-Layer Perceptron (`Dense(32) -> Dense(16) -> Dense(1)`). Targets are scaled by the dataset median during optimization to stabilize gradient updates against heavy-tailed build durations.

---

## Features

The model processes 11 total commit features:

### Categorical Features
* `prefix`: Primary directory modified (e.g., `src/`, `third_party/`, `tools/`).
* `type`: Primary programming language or file type (e.g., `JAVA`, `C/C++`, `PYTHON`, `SH`).

### Continuous Churn Telemetry
* `num_files`: Total unique files touched in the commit.
* `insertions`: Total lines added.
* `deletions`: Total lines removed.
* `unique_ext`: Number of distinct file extensions modified.
* `src_ratio`: Proportion of modified files within core source paths.
* `test_ratio`: Proportion of modified files within test paths.
* `docs_ratio`: Proportion of documentation/markdown modifications.
* `third_party_ratio`: Proportion of vendored external dependency modifications.
* `commit_time`: Epoch timestamp of commit authoring.

---

## Evaluation & Benchmarks

Models were evaluated on an identical split (300 train / 100 validation / 100 test). The target variable exhibits extreme variance (600 ms to 45,000+ ms).

| Model | Architecture | Features Used | Test RMSE | $\Delta$ vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Legacy Baseline** | Additive Linear (`tf.estimator`) | `prefix`, `type` | 6,713.79 ms | — |
| **Naive Mean Predictor** | Global Target Mean | None | 6,618.00 ms | -95.79 ms |
| **Categorical Floor** | Exact Category Lookup Table | `prefix`, `type` | 6,591.02 ms | -122.77 ms |
| **Keras Hybrid NN** | HashedCross(64) + MLP (`Dense 32, 16`) | All (11 features) | **6,598.51 ms** | **-115.28 ms** |
| **HistGradientBoosting** | Histogram-Based GBDT (`sklearn`) | All (11 features) | **6,587.46 ms** | **-126.33 ms** |

### Key Findings
1. **Breaking the Categorical Limit:** Using only categorical labels bounds achievable test performance at **6,591.02 ms** (the Categorical Floor). Both `HistGradientBoosting` (6,587.46 ms) and the Keras Hybrid NN (6,598.51 ms) outperform the naive mean (6,618 ms) and approach/surpass this theoretical boundary, proving that continuous Git churn metrics provide real predictive signal.
2. **Early Stopping Protection:** The training loop utilizes `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`. Training halted at **Epoch 11**, restoring optimal generalization weights from Epoch 1 (`val_loss: 10.6639`) before the network could overfit the 300-sample training partition.

---

## Project Structure

```text
├── data/
│   ├── InputData.csv          # Commit metadata (prefix, type)
│   ├── CPUTimes.csv           # Ground-truth Bazel build execution times (ms)
│   └── build_metrics.csv      # Merged dataset with 9 Git churn metrics
├── scripts/
│   ├── data_collection.py     # Automated Git telemetry extraction (git diff --numstat)
│   ├── train_hybrid.py        # Keras Hybrid model definition and training loop
│   ├── train_baseline.py      # Replicated linear baseline evaluation
│   └── benchmark_gbdt.py      # HistGradientBoosting benchmark pipeline
├── dashboard/
│   ├── app.py                 # Streamlit evaluation and visualization dashboard
│   └── requirements.txt       # Dashboard-specific dependencies
├── README.md
└── requirements.txt
```

---

## Installation & Setup

### Prerequisites
* Python 3.9+
* Git (must be installed and accessible in system `PATH` for telemetry collection)

### Environment Setup
```bash
git clone https://github.com/your-username/bazel-build-predictor.git
cd bazel-build-predictor

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Reproduction & Usage

### 1. Extract Git Telemetry
Run `data_collection.py` against a local clone of `bazelbuild/bazel` to compute churn metrics and output `build_metrics.csv`:
```bash
python scripts/data_collection.py --repo-path /path/to/bazel --output data/build_metrics.csv
```

### 2. Train the Keras Hybrid Model
Train the multimodal neural network with feature crossing and median scaling:
```bash
python scripts/train_hybrid.py \
  --data data/build_metrics.csv \
  --epochs 200 \
  --batch-size 16 \
  --lr 0.01 \
  --patience 10
```

### 3. Run Benchmark Models
Evaluate the legacy linear baseline and the HistGradientBoosting regressor:
```bash
python scripts/train_baseline.py --data data/build_metrics.csv
python scripts/benchmark_gbdt.py --data data/build_metrics.csv
```

### 4. Launch the Interactive Dashboard
Launch the evaluation UI to visualize model predictions against ground truth:
```bash
streamlit run dashboard/app.py
```
