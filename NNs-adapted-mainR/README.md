# Python ML Pipelines - Credit Default Prediction

> **Status:** ✅ ACTIVE  
> **Framework:** AutoGluon 1.4+  
> **Python:** 3.11+

---

## 🎯 Overview

This folder contains the main Python/AutoGluon pipelines for credit default prediction. Two main approaches are implemented:

| Script | Split Strategy | Purpose |
|--------|---------------|---------|
| `00_combined_pipeline.py` | Out-of-Time (OOT) | Production simulation - train on past, predict future |
| `00_combined_pipeline_adapted-to-mainR.py` | Out-of-Sample (OOS) | R comparison - matches Main.R methodology exactly |

---

## 📁 Current Structure

```
python_pipelines/
├── 00_combined_pipeline.py                    # OOT temporal pipelines (1a/1b/1c/2)
├── 00_combined_pipeline_adapted-to-mainR.py   # OOS stratified (R-compatible)
├── compare_pipelines.py                       # Auto-finds latest runs, shows metrics table
├── generate_pipeline_summary.py               # Create detailed reports
├── README.md                                  # This file
└── models/
    └── pipeline_runs/                         # Individual run results
        ├── pipeline_1a/                       # All sectors, tree models
        ├── pipeline_1b/                       # With company exclusion
        ├── pipeline_1c/                       # Neural networks
        ├── pipeline_2/                        # Cross-sector
        └── adapted_oos_tree/                  # R-adapted approach
```

**Neural Network Models:**

> ⚠️ **Fixed Hyperparameters:** Both pipelines use **fixed, non-tuned hyperparameter values** (not AutoGluon HPO search spaces). This avoids crashes from invalid `ag_space.Real()` objects and ensures stable, reproducible training.

1. **PyTorch Neural Network (NN_TORCH)**
   ```python
   # FIXED values (both pipelines use these)
   num_epochs: 100              # OOS pipeline
   num_epochs: 50               # OOT pipeline (faster)
   learning_rate: 0.001         # Both pipelines
   dropout_prob: 0.1            # OOS pipeline only
   batch_size: 512              # Tunable via BATCH_SIZE config
   ```

2. **FastAI Tabular (FASTAI)**
   ```python
   # FIXED values (both pipelines use these)
   epochs: 50                   # OOS pipeline
   epochs: 30                   # OOT pipeline (faster)
   learning_rate: 0.001         # OOS pipeline
   learning_rate: 0.01          # OOT pipeline
   batch_size: 512              # Tunable via BATCH_SIZE config
   ```

### Why Fixed Hyperparameters?

When you use `ag_space.Real()` or other search space objects, AutoGluon expects HPO (hyperparameter optimization) to be enabled. Without it, you get:
```
TypeError: unsupported operand type(s) for /: 'Real' and 'float'
```

**Solution:** Define hyperparameters as **fixed scalar values** in the config section:
```python
# NN_TORCH (PyTorch) hyperparameters
NN_TORCH_EPOCHS = 100        # Number of training epochs
NN_TORCH_LR = 0.001          # Learning rate
NN_TORCH_DROPOUT = 0.1       # Dropout probability

# FASTAI hyperparameters
FASTAI_EPOCHS = 50           # Number of training epochs
FASTAI_LR = 0.001            # Learning rate
```

These are then referenced in the hyperparameter dict:
```python
custom_hyperparameters = {
    'NN_TORCH': {
        'num_epochs': NN_TORCH_EPOCHS,    # References config variable
        'learning_rate': NN_TORCH_LR,     # References config variable
        'dropout_prob': NN_TORCH_DROPOUT, # References config variable
        'batch_size': BATCH_SIZE,         # Tunable if GPU OOM occurs
    },
    'FASTAI': {
        'epochs': FASTAI_EPOCHS,          # References config variable
        'lr': FASTAI_LR,                  # References config variable
        'bs': BATCH_SIZE,                 # Tunable if GPU OOM occurs
    },
}
```

**Benefits:**
- ✅ No HPO TypeErrors or crashes
- ✅ Values discoverable in config section (lines ~300-305 in both files)
- ✅ Easy to tune if needed (just edit the XXXX_EPOCHS/LR variables)
- ✅ Reproducible across runs (same seed = same values)


---

## 🚀 Quick Start

### Run Standard OOT Pipeline
```bash
cd /home/martin-mal/Documents/oenb_standalone
source .venv/bin/activate
cd python_pipelines
python 00_combined_pipeline.py
```

### Run R-Adapted OOS Pipeline
```bash
python 00_combined_pipeline_adapted-to-mainR.py
```

---

## 📜 Script Details

### 1️⃣ `00_combined_pipeline.py` — Out-of-Time (OOT) Temporal Split

**Purpose:** Production-realistic evaluation where models are trained on historical data and tested on future data.

**Configuration:**
```python
# Pipeline selection
PIPELINE = # Options: '1a', '1b', '1c', '2'

# Training intensity
TRAIN_MODE_PARAM =  # Options: "short", "medium", "long"

# Threshold tuning
MAX_FNR = # Max acceptable false negative rate, eg. 0.28
```

**Pipeline Variants:**

| Pipeline | Train Years | Test Years | Model Type | Special Feature |
|----------|-------------|------------|------------|-----------------|
| `1a` | 2018-2020 | 2021-2022 | Tree | All sectors pooled |
| `1b` | 2018-2020 | 2021-2022 | Tree | Company exclusion (no data leakage) |
| `1c` | 2018-2021 | 2022 | Neural Networks | All sectors pooled |
| `2` | 2018-2020 | 2021-2022 | Tree | Cross-sector (Industry→Services) |

---

### 2️⃣ `00_combined_pipeline_adapted-to-mainR.py` — Out-of-Sample (OOS) Stratified Split

**Purpose:** Directly comparable to Main.R methodology for benchmarking Python vs R approaches.

**Key Features:**
- ✅ Same random seed: `RANDOM_STATE = 123` (matches `set.seed(123)`)
- ✅ Same train/test ratio: `TRAIN_SIZE = 0.7` (matches `Train_size = 0.7`)
- ✅ Same CV folds: `N_FOLDS = 5` (matches `N_folds <- 5`)
- ✅ Same stratification: by `sector` + `y` (matches `MVstratifiedsampling`)
- ✅ Custom CV groups via AutoGluon's `groups` parameter (matches `MVstratifiedsampling_CV`)
- ✅ Maximum GPU/CPU utilization

**Configuration:**
```python
# Match Main.R settings exactly
RANDOM_STATE = 123      # set.seed(123)
TRAIN_SIZE = 0.7        # Train_size = 0.7
N_FOLDS = 5             # N_folds <- 5

# Model selection
MODEL_TYPE = 'tree'     # Options: 'tree', 'nn', 'all'

# Training intensity
TRAIN_MODE_PARAM = "long"  # Options: "short", "medium", "long"
```

---

### 3️⃣ `compare_pipelines.py` — Side-by-Side Comparison

**Purpose:** Automatically finds the latest OOT and OOS runs and displays a comparison table.

**Usage:**
```bash
python compare_pipelines.py
```

**Sample Output:**
```
📊 PIPELINE COMPARISON: OOT (Temporal) vs OOS (Stratified)
===========================================================================
METRIC                    OOT (Temporal)            OOS (Stratified)
===========================================================================
ROC-AUC                   0.8234                    0.8156
Recall (TPR)              85.00%                    82.50%
Precision                 42.00%                    45.00%
FNR (Miss Rate)           15.00%                    17.50%
...
---------------------------------------------------------------------------
🏆 OOT (Temporal) has higher ROC-AUC by 0.0078
```

---

### 4️⃣ `generate_pipeline_summary.py` — Detailed Report Generator

**Purpose:** Creates comprehensive HTML/Markdown reports with visualizations.

**Usage:**
```bash
python generate_pipeline_summary.py <path_to_pipeline_run>

# Example:
python generate_pipeline_summary.py models/pipeline_runs/pipeline_1c/20260116_122757
```

**Generated Outputs:**
- ROC curve plot
- Precision-Recall curve
- Confusion matrix heatmap
- Threshold analysis chart
- Feature importance (for tree models)
- Full HTML report

---

## 🔄 Custom CV Groups in AutoGluon

The adapted pipeline uses AutoGluon's `groups` parameter to replicate **exactly** the same CV fold assignments as Main.R's `MVstratifiedsampling_CV`.

### How the `groups` Parameter Works

From the [AutoGluon documentation](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html):

> **groups** (str, default = None) – If specified, AutoGluon will use the column named the value of `groups` in `train_data` during `.fit` as the data splitting indices for bagging. This column will not be used as a feature. The data will be split via `sklearn.model_selection.LeaveOneGroupOut`.

### Understanding the Example

The docs give this example:
> *"If you want your data folds to preserve adjacent rows in the table without shuffling, then for 3-fold bagging with 6 rows of data, the groups column values should be `[0, 0, 1, 1, 2, 2]`."*

**What this means:**

The group value tells AutoGluon **which validation fold each row belongs to**. With `LeaveOneGroupOut`, all rows with the same group value will be held out together as validation data.

| Row Index | Group Value | Meaning |
|-----------|-------------|---------|
| 0 | 0 | This row is validation data in Fold 0 |
| 1 | 0 | This row is validation data in Fold 0 |
| 2 | 1 | This row is validation data in Fold 1 |
| 3 | 1 | This row is validation data in Fold 1 |
| 4 | 2 | This row is validation data in Fold 2 |
| 5 | 2 | This row is validation data in Fold 2 |

**The resulting CV splits:**

- **Fold 0:** Train on rows 2,3,4,5 (groups 1,2) → Validate on rows 0,1 (group 0)
- **Fold 1:** Train on rows 0,1,4,5 (groups 0,2) → Validate on rows 2,3 (group 1)
- **Fold 2:** Train on rows 0,1,2,3 (groups 0,1) → Validate on rows 4,5 (group 2)

The example shows **non-shuffled, contiguous** fold assignments (`[0,0,1,1,2,2]`), meaning rows 0-1 stay together, rows 2-3 stay together, etc. This preserves the original row order within folds.

**In our implementation**, we use `StratifiedKFold` with `shuffle=True` and `random_state=123` to create fold assignments that:
1. Are **stratified** by the target variable `y` (ensuring each fold has similar default rates)
2. Match **exactly** the fold assignments from Main.R's `MVstratifiedsampling_CV`

### Our Implementation

We create a `cv_fold_group` column using stratified K-fold (same as Main.R):

```python
# Each row gets assigned to a fold (0, 1, 2, 3, or 4)
train_data['cv_fold_group'] = create_cv_groups_column(
    train_data,
    target_col='y',
    n_folds=5,
    random_state=123  # Same seed as Main.R
)

# Tell AutoGluon to use this column for CV splits
predictor = TabularPredictor(
    label='y',
    groups='cv_fold_group',  # ← This is the key!
    ...
)
```

This ensures the **exact same observations** end up in the same CV folds as in Main.R, making results directly comparable.

---

## 📊 Output Files

### Directory Structure

All Python pipeline results are saved to **`python_pipelines/models/`** (NOT the root `/models/` folder):

```
python_pipelines/
└── models/
    └── pipeline_runs/
        ├── pipeline_1a/                       # OOT: All sectors, tree models
        │   └── 20260116_122757/
        ├── pipeline_1b/                       # OOT: With company exclusion
        ├── pipeline_1c/                       # OOT: Neural networks
        ├── pipeline_2/                        # OOT: Cross-sector
        └── adapted_oos_tree/                  # OOS: R-adapted approach
            └── 20260116_143022/
```

> ⚠️ **Note:** The root-level `/models/` directory is for legacy/notebook results. All pipeline script outputs go to `python_pipelines/models/`.

### Provenance Tracking with `SOURCE_INFO.json`

Each run includes a `SOURCE_INFO.json` file that documents **exactly** where the results came from:

```json
{
  "source_script": "/home/martin-mal/Documents/oenb_standalone/python_pipelines/00_combined_pipeline_adapted-to-mainR.py",
  "source_script_name": "00_combined_pipeline_adapted-to-mainR.py",
  "pipeline_type": "oos_adapted_mainR",
  "methodology": {
    "split_method": "out_of_sample_stratified",
    "train_size": 0.7,
    "cv_folds": 5,
    "random_seed": 123,
    "stratification": ["sector", "y"]
  },
  "execution": {
    "timestamp": "2026-01-16T14:30:22.123456",
    "working_directory": "/home/martin-mal/Documents/oenb_standalone/python_pipelines"
  },
  "hardware": {
    "gpu_count": 1,
    "cpu_count": 16
  }
}
```

This ensures you can always trace back:
- **Which script** produced the results
- **What methodology** was used (OOT vs OOS, split ratios, seeds)
- **When** the run was executed
- **What hardware** was available

### Run Output Contents

Each run creates a timestamped folder with:

```
models/pipeline_runs/pipeline_1c/20260116_122757/
├── models/                    # Trained AutoGluon models
├── SOURCE_INFO.json           # 🆕 Full provenance tracking
├── threshold_tuning.json      # Optimal threshold & metrics
├── run_config.json            # Configuration used
├── leaderboard.csv            # Model comparison
└── predictor.pkl              # Serialized predictor
```

### Key Metrics in `threshold_tuning.json`
```json
{
  "optimal_threshold": 0.1234,
  "metrics_at_optimal": {
    "recall": 0.85,
    "precision": 0.42,
    "fnr": 0.15,
    "fpr": 0.23,
    "mcc": 0.45,
    "tp": 127,
    "fn": 23
  },
  "roc_auc": 0.8234
}
```

---

## 🔄 Comparison: OOT vs OOS

| Aspect | OOT (Temporal) | OOS (Stratified) |
|--------|----------------|------------------|
| **Realism** | ✅ Production-like | ❌ Academic |
| **R Comparison** | ❌ Different split | ✅ Same methodology |
| **Data Leakage Risk** | ✅ None | ⚠️ Possible (same companies) |
| **Default Distribution** | May vary by year | Balanced across splits |
| **Use Case** | Deployment readiness | Algorithm benchmarking |

---

## ⏱️ Progress Tracking

Both main pipeline scripts include **real-time progress tracking**:
- Elapsed time displayed at each stage
- Stage timing breakdown at completion
- Useful for detecting stalled executions

Example output:
```
⏱️  [0.0s elapsed] Starting: PART 1: Data Loading & EDA
✅ [5.2s elapsed] Completed: PART 1: Data Loading & EDA (5.2s)
⏱️  [5.2s elapsed] Starting: PART 2: Data Preparation & Splitting
...
⏱️  TIMING SUMMARY
   PART 1: Data Loading & EDA: 5.2s (0.1%)
   PART 2: Data Preparation & Splitting: 12.3s (0.3%)
   PART 2b: Model Training: 58.2m (97.1%)
   PART 3: Model Evaluation: 45.6s (1.3%)
   TOTAL: 1.0h
```

---

## 💻 Hardware Optimization

The adapted pipeline (`00_combined_pipeline_adapted-to-mainR.py`) automatically maximizes hardware utilization:

```python
# Auto-detected and configured:
GPU_COUNT = torch.cuda.device_count()  # Uses all available GPUs
CPU_COUNT = os.cpu_count()             # Uses all available CPUs

# Model-specific GPU acceleration:
'GBM': {'device': 'gpu'}               # LightGBM on GPU
'CAT': {'task_type': 'GPU'}            # CatBoost on GPU  
'XGB': {'tree_method': 'gpu_hist'}     # XGBoost on GPU
'RF':  {'n_jobs': CPU_COUNT}           # Random Forest parallelized
```

### 🔍 Monitoring Hardware Utilization

Use these commands in a separate terminal to verify GPU/CPU usage during training:

#### GPU Monitoring (NVIDIA)
```bash
# Real-time GPU stats (updates every 1 second)
watch -n 1 nvidia-smi

# Continuous GPU monitoring with timestamps
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1

# Quick one-time check
nvidia-smi
```

**What to look for:**
- `GPU-Util`: Should be **80-100%** during training (if < 50%, GPU may not be used)
- `Memory-Usage`: Should show significant VRAM allocation
- Process list should show `python` using the GPU

#### CPU Monitoring
```bash
# Real-time CPU usage per core (updates every 1 second)
watch -n 1 "mpstat -P ALL 1 1 | head -20"

# Interactive process monitor (press 'q' to quit)
htop

# Simple CPU load average
uptime

# Per-core utilization summary
mpstat -P ALL 1

# Top CPU-consuming processes
top -o %CPU
```

**What to look for:**
- All cores at **80-100%** during parallelized operations (RF, data processing)
- `load average` should be close to your CPU count (e.g., ~16 for 16 cores)

#### Combined Monitoring
```bash
# GPU + CPU in one view (requires gpustat: pip install gpustat)
watch -n 1 "gpustat; echo '---'; uptime"

# Or side-by-side terminals:
# Terminal 1: watch -n 1 nvidia-smi
# Terminal 2: htop
```

#### Memory Monitoring
```bash
# Real-time RAM usage
watch -n 1 free -h

# Detailed memory breakdown
vmstat 1
```

**Signs of good utilization:**
| Resource | Underutilized | Well Utilized |
|----------|---------------|---------------|
| GPU Util | < 50% | 80-100% |
| GPU Memory | < 2GB | > 4GB |
| CPU (all cores) | < 50% avg | 80-100% |
| RAM | < 8GB | 12-32GB |

---

## 📝 Notes

- **GPU Support:** Auto-detected, uses CUDA if available
- **Memory:** Large models may need 16GB+ RAM
- **Time:** "long" mode runs ~60 minutes per pipeline
- **Reproducibility:** Use same seeds for comparable results between R and Python
