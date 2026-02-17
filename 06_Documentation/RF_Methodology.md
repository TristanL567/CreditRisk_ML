# Random Forest Pipeline — Methodology & User Guide

**Project:** Credit Risk ML — OeNB Industry Lab  
**Script:** `01_Code/main_RF.R`  
**Author:** Auto-generated documentation  
**Last updated:** February 2026

---

## Hardware Benchmark (`01_Code/benchmark_hardware.R`)

Run this **once** before `main_RF.R` to find out what your machine can handle. Takes ~30 seconds and outputs a recommended `TRAIN_MODE` setting.

```r
source("CreditRisk_ML/01_Code/benchmark_hardware.R")
```

<details>
<summary>Full benchmark script</summary>

```r
#==============================================================================#
#==== Hardware Benchmark — Recommends TRAIN_MODE for main_RF.R ================#
#==============================================================================#
# Run this ONCE before main_RF.R to find out what your machine can handle.
# Takes ~30 seconds. Outputs a recommended TRAIN_MODE setting.
#==============================================================================#

cat("\n", strrep("=", 60), "\n")
cat("  HARDWARE BENCHMARK\n")
cat(strrep("=", 60), "\n\n")

# ---- 1. System info ----
n_cores <- parallel::detectCores()
ram_gb  <- as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / 1e9
os_name <- Sys.info()["sysname"]

cat(sprintf("  OS:          %s\n", os_name))
cat(sprintf("  CPU cores:   %d\n", n_cores))
cat(sprintf("  RAM:         %.1f GB\n", ram_gb))

# ---- 2. Quick RF benchmark (synthetic data, mimics real workload) ----
cat("\n  Running benchmark (5-fold CV on synthetic data)...\n")

suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3learners)
  library(ranger)
})

set.seed(42)
n_obs  <- 5000L
n_feat <- 13L

bench_data <- as.data.frame(matrix(rnorm(n_obs * n_feat), ncol = n_feat))
colnames(bench_data) <- paste0("x", seq_len(n_feat))
bench_data$y <- factor(sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.97, 0.03)))

task <- TaskClassif$new("bench", backend = bench_data, target = "y", positive = "1")
learner <- lrn("classif.ranger", predict_type = "prob", num.trees = 500L,
               num.threads = n_cores)
resampling <- rsmp("cv", folds = 5L)

t0 <- Sys.time()
rr <- resample(task, learner, resampling, store_models = FALSE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

auc_score <- rr$aggregate(msr("classif.auc"))
time_per_fold <- elapsed / 5

cat(sprintf("  Benchmark AUC:       %.4f\n", auc_score))
cat(sprintf("  Total time:          %.1f s\n", elapsed))
cat(sprintf("  Time per CV fold:    %.2f s\n", time_per_fold))

# ---- 3. Estimate total runtimes ----
# Scale factor: real data is ~3-5x larger, MBO overhead ~1.3x
scale_factor <- 4.0 * 1.3

est_per_eval <- time_per_fold * 5 * scale_factor  # 5 folds per eval

est_fast   <- est_per_eval * 10  / 60   # 10 evals, 3 folds → scale by 3/5
est_medium <- est_per_eval * 30  / 60   # 30 evals, 5 folds
est_slow   <- est_per_eval * 100 / 60   # 100 evals, 10 folds → scale by 10/5

# Adjust for different fold counts
est_fast <- est_fast * (3/5)
est_slow <- est_slow * (10/5)

cat("\n  Estimated runtimes PER STRATEGY:\n")
cat(sprintf("    fast:   %5.1f min  (10 evals, 3 folds)\n", est_fast))
cat(sprintf("    medium: %5.1f min  (30 evals, 5 folds)\n", est_medium))
cat(sprintf("    slow:   %5.1f min  (100 evals, 10 folds)\n", est_slow))

cat("\n  Estimated runtimes ALL 5 STRATEGIES:\n")
cat(sprintf("    fast:   %5.1f min  (%.1f hrs)\n", est_fast * 5, est_fast * 5 / 60))
cat(sprintf("    medium: %5.1f min  (%.1f hrs)\n", est_medium * 5, est_medium * 5 / 60))
cat(sprintf("    slow:   %5.1f min  (%.1f hrs)\n", est_slow * 5, est_slow * 5 / 60))

# ---- 4. Recommend mode ----
# Heuristic: recommend the most intensive mode that finishes 1 strategy in <30 min
cat("\n", strrep("-", 60), "\n")

if (est_slow <= 30) {
  recommended <- "slow"
  reason <- "Your machine is powerful enough for production-quality runs."
} else if (est_medium <= 30) {
  recommended <- "medium"
  reason <- "Good balance of quality and speed for your hardware."
} else {
  recommended <- "fast"
  reason <- "Limited resources — use fast mode for development, slow mode overnight."
}

cat(sprintf("  RECOMMENDED:  TRAIN_MODE <- \"%s\"\n", recommended))
cat(sprintf("  Reason:       %s\n", reason))
cat(strrep("-", 60), "\n")

cat(sprintf("\n  Copy this into main_RF.R:\n"))
cat(sprintf("  ───────────────────────────────────\n"))
cat(sprintf("  TRAIN_MODE <- \"%s\"\n", recommended))
cat(sprintf("  ───────────────────────────────────\n\n"))
```

</details>

---

## Table of Contents

1. [What Does This Pipeline Do?](#1-what-does-this-pipeline-do)
2. [How to Run It](#2-how-to-run-it)
3. [Configuration Knobs Explained](#3-configuration-knobs-explained)
4. [Data Pipeline (Sections 03–04)](#4-data-pipeline-sections-0304)
5. [Feature Engineering Strategies (Section 05)](#5-feature-engineering-strategies-section-05)
6. [Random Forest Model (Section 06)](#6-random-forest-model-section-06)
7. [Hyperparameter Optimization — MBO (Section 06)](#7-hyperparameter-optimization--mbo)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Output Charts & Tables (Sections 07–08)](#9-output-charts--tables-sections-0708)
10. [VAE / DAE Artifact Generator](#10-vae--dae-artifact-generator-generate_vae_dae_artifactsr)
11. [Helper Functions (RF_Subfunctions)](#11-helper-functions-rf_subfunctions)
12. [Troubleshooting](#12-troubleshooting)
13. [Mathematical Background](#13-mathematical-background)

---

## 1. What Does This Pipeline Do?

This pipeline builds a **Random Forest** classifier to predict corporate credit defaults (binary: default vs. non-default). It:

1. Loads and preprocesses financial data (`data.rda`)
2. Engineers features using up to **5 different strategies** (Base + A/B/C/D)
3. Tunes Random Forest hyperparameters using **Bayesian Optimization** (Model-Based Optimization)
4. Trains a final model per strategy and evaluates it on held-out test data
5. Produces **leaderboard charts and tables** comparing all strategies

**In plain English:** We try 5 different ways of preparing the input data, find the best Random Forest settings for each, and compare them all to see which data preparation helps the most.

---

## 2. How to Run It

### Prerequisites

| Requirement | Details |
|---|---|
| **R version** | >= 4.1 |
| **Data file** | `data/data.rda` at the project root |
| **Packages** | Auto-installed on first run (see Section 01 of the script) |
| **For Phase 1 (Base + D)** | Nothing else — runs immediately |
| **For Phase 2 (A + B + C)** | VAE/DAE pipeline must be run first to produce artifacts in `data/pipeline_artifacts/` (see Section 5) |

### Two-Phase Execution

The strategies are split into two groups with different prerequisites:

| Phase | Strategies | Prerequisites | What you get |
|---|---|---|---|
| **Phase 1** | Base + D | Just `data.rda` | Baseline RF + manual feature engineering comparison |
| **Phase 2** | A + B + C | VAE/DAE pipeline must be run first | Full strategy comparison including deep-learning-based features |

**Phase 1 runs immediately** — no extra setup needed. Phase 2 requires the VAE/DAE pipeline to have generated its artifact files (see Section 5 for details). If you run the script with all strategies enabled but without the VAE artifacts, Strategies A/B/C will print `[SKIP]` and only Base and D will execute. This is expected behavior, not an error.

#### Step 0: Run the hardware benchmark (once)

Before your first run, check what your machine can handle:

```r
source("CreditRisk_ML/01_Code/benchmark_hardware.R")
```

This takes ~30 seconds and outputs a recommended `TRAIN_MODE` (fast/medium/slow) with estimated runtimes. Copy the recommendation into `main_RF.R`.

#### Phase 1: Run Base + D (no dependencies)

```r
# Edit the config knob:
STRATEGIES_TO_RUN <- c("Base", "D")

# Then run:
setwd("/path/to/oenb_standalone")
source("CreditRisk_ML/01_Code/main_RF.R")
```

#### Phase 1.5: Generate VAE/DAE artifacts

This trains a VAE and DAE on the preprocessed features and saves the 6 `.rds` artifact files that Strategies A/B/C need. Requires the `torch` R package (auto-installed on first run).

```r
setwd("/path/to/oenb_standalone")
source("CreditRisk_ML/01_Code/generate_vae_dae_artifacts.R")
```

Takes ~2–5 minutes. Uses the **same seed and data pipeline** as `main_RF.R`, so train/test splits are identical. After this completes you will see 6 new files in `data/pipeline_artifacts/`.

#### Phase 2: Run A + B + C (after VAE/DAE artifacts)

```r
# Now you can enable all strategies:
STRATEGIES_TO_RUN <- c("Base", "A", "B", "C", "D")
source("CreditRisk_ML/01_Code/main_RF.R")
```

Or run only the VAE-dependent strategies (if Base+D results are already saved):

```r
STRATEGIES_TO_RUN <- c("A", "B", "C")
source("CreditRisk_ML/01_Code/main_RF.R")
```

#### Quick Test Run

To do a fast test, just set the training mode:

```r
TRAIN_MODE         <- "fast"          # 10 evals, 3 folds (~5 min/strategy)
STRATEGIES_TO_RUN  <- c("Base")      # Only run the base model
```

### Output Location

All charts and tables are saved to: `CreditRisk_ML/03_Charts/RF/`

---

## 3. Configuration Knobs Explained

All knobs are at the top of `main_RF.R` (lines 20–60). Change them before running.

### Training Speed

The single most important knob. Set `TRAIN_MODE` and everything else (HPO budget, CV folds) adjusts automatically:

| Mode | `N_EVALS` | `N_FOLDS` | `STAGNATION_ITERS` | Time per strategy | Use case |
|---|---|---|---|---|---|
| **`"fast"`** | 10 | 3 | 10 | ~5 min | Quick sanity check, debugging |
| **`"medium"`** | 30 | 5 | 50 | ~20 min | Standard development run |
| **`"slow"`** | 100 | 10 | 80 | ~2 hours | Final production results |

```r
TRAIN_MODE <- "medium"  # Just change this one line
```

**What the parameters mean:**
- `N_EVALS`: How many hyperparameter combinations to try. More = better but slower.
- `N_FOLDS`: Number of CV folds. More folds = more reliable AUC estimates but each evaluation takes longer.
- `STAGNATION_ITERS`: If no improvement after this many attempts, stop early. Saves time when the optimum is found quickly.

**Analogy:** Think of `TRAIN_MODE` like oven settings — "fast" is a quick reheat, "medium" is standard baking, "slow" is a slow roast. You don't need to set temperature and time separately.

You can still override individual parameters after `get_training_params()` if needed:

```r
TRAIN_MODE <- "medium"
TRAINING_PARAMS <- get_training_params(TRAIN_MODE)
N_EVALS <- 50L  # override just this one
```

### Other Settings

| Knob | Default | What it does |
|---|---|---|
| `TRAIN_PROP` | 0.7 | 70% of companies go to training, 30% to the final test set. |
| `RANDOM_SEED` | 123 | Ensures reproducibility — same seed = same data split = same results. |

### Chart Settings

| Knob | Default | What it does |
|---|---|---|
| `WIDTH_PX` | 3750 | Output chart width in pixels (high-res for presentations) |
| `HEIGHT_PX` | 1833 | Output chart height in pixels |
| `DPI` | 300 | Dots per inch — 300 is print quality |
| `BLUE/GREY/ORANGE/RED` | Hex colors | Standard project color palette |

### Parallelization

| Knob | Default | What it does |
|---|---|---|
| `N_CORES` | "auto" | Number of CPU cores. "auto" uses all available cores. Set to a number (e.g., 4) to limit. |
| `PARALLEL_BACKEND` | "multisession" | How R parallelizes. "multisession" works on all OS. "multicore" is faster on Linux/Mac but doesn't work on Windows. "sequential" disables parallelism. |

### Execution Control

| Knob | Default | What it does |
|---|---|---|
| `STRATEGIES_TO_RUN` | c("Base","A","B","C","D") | Which strategies to execute. Remove letters to skip strategies. |
| `DIVIDE_BY_TOTAL_ASSETS` | TRUE | Whether to normalize financial features by total assets (f1) before modeling. |

---

## 4. Data Pipeline (Sections 03–04)

The data pipeline mirrors `Main.R` sections 02–03 exactly, ensuring comparability with GLM and XGBoost results.

### Step 1: Load & Filter (Section 03)

```
data.rda → DataPreprocessing() → Drop ratio features (r1–r18)
```

`DataPreprocessing()` applies accounting consistency filters:
- `f10 >= -2` (no extreme negative values)
- `f2 + f3 <= f1 + 2` (liabilities can't massively exceed assets)
- `f4 + f5 <= f3 + 2` (sub-items can't exceed totals)
- `f6 + f11 <= f1 + 2` (same principle)

### Step 2: Train/Test Split (Section 03)

Uses **multivariate stratified sampling** at the company level:
- Groups all observations by company (`id`)
- Stratifies by `sector` and `y` (default status)
- Assigns 70% of *companies* to train, 30% to test
- All observations from a company go to the same set (no data leakage)

**Why company-level?** If the same company appears in both train and test, the model "remembers" it. This would inflate test scores. By splitting at the company level, we ensure the test set contains truly unseen companies.

### Step 3: Feature Engineering (Section 04)

#### 3a. Divide by Total Assets

Financial features `f2`–`f11` are divided by `f1` (total assets) to create **size-normalized ratios**. This makes a €1M debt for a large company comparable to a €10K debt for a small company.

#### 3b. Quantile Transformation

Each feature is transformed to follow a standard normal distribution using rank-based **Probability Integral Transform (PIT)**:

1. Rank all training values
2. Convert ranks to percentiles: `rank / n`
3. Apply the inverse normal CDF: `qnorm(percentile)`
4. For test data: use the training ECDF to map values, then apply `qnorm`

**Why?** Financial data is often heavily skewed (many small companies, few large ones). Quantile transformation makes every feature equally distributed, helping the model treat all features fairly.

#### 3c. Stratified CV Folds

Creates 5 cross-validation folds, stratified by company, sector, and default status. The same company never appears in both training and validation within a fold.

---

## 5. Feature Engineering Strategies (Section 05)

Each strategy prepares different input features. The RF model and HPO process are identical — only the features change.

### Phase 1 Strategies (no extra dependencies)

These run immediately with just `data.rda`:

| Strategy | Name | Features Used |
|---|---|---|
| **Base** | Base Model | Quantile-transformed `f1`–`f11` + `sector` + `size` |
| **D** | Manual Feature Eng. | Base features + 5 hand-crafted financial ratios |

### Phase 2 Strategies (require VAE/DAE artifacts)

These require running `generate_vae_dae_artifacts.R` first to create the artifact files in `data/pipeline_artifacts/`:

| Strategy | Name | Features Used | Required artifacts |
|---|---|---|---|
| **A** | Dim. Reduction | VAE latent features + `sector` + `size` | `vae_latent_train.rds`, `vae_latent_test.rds` |
| **B** | Anomaly Score | Base features + VAE reconstruction error | `vae_anomaly_train.rds`, `vae_anomaly_test.rds` |
| **C** | Feature Denoising | DAE-reconstructed features + `sector` + `size` | `dae_denoised_train.rds`, `dae_denoised_test.rds` |

> **First run?** Strategies A/B/C will print `[SKIP]` — this is expected. Run Phase 1 (Base + D) first, then generate the VAE/DAE artifacts, then re-run with all strategies.

### Strategy Details

#### Phase 1

**Base Model:** Uses the preprocessed financial features as-is. This is the baseline — all other strategies are compared to it via "uplift" percentages.

**Strategy D (Manual Feature Engineering):** Adds 5 hand-crafted financial ratios that credit analysts commonly use:

| Ratio | Formula | Interpretation |
|---|---|---|
| `leverage` | f3 / f1 | Debt-to-assets ratio |
| `liquidity` | f6 / f3 | Cash relative to debt |
| `profitability` | f8 / f1 | Profit margin on assets |
| `coverage` | f9 / f7 | Interest coverage |
| `asset_turnover` | f10 / f1 | Revenue per unit of assets |

#### Phase 2

**Strategy A (VAE Latent Features):** A Variational Autoencoder compresses the 11 financial features into a lower-dimensional "latent space" (e.g., 4–8 dimensions). These latent variables capture the essential structure of the data in fewer features. Think of it as a smart summary.

**Strategy B (Anomaly Score):** Adds a single extra feature — the VAE's reconstruction error. Companies that the VAE reconstructs poorly are "unusual" and may be higher risk. The higher the anomaly score, the more the company deviates from normal patterns.

**Strategy C (DAE Denoising):** A Denoising Autoencoder takes noisy financial data, removes the noise, and outputs "cleaned" features. If the original data contains measurement errors or one-off accounting anomalies, the DAE version may be more reliable.

### Execution Order

```
1. Run main_RF.R with STRATEGIES_TO_RUN = c("Base", "D")
   → Produces baseline results + manual feature eng. comparison

2. source("CreditRisk_ML/01_Code/generate_vae_dae_artifacts.R")
   → Trains VAE + DAE, saves 6 .rds files to data/pipeline_artifacts/

3. Re-run main_RF.R with STRATEGIES_TO_RUN = c("Base", "A", "B", "C", "D")
   → Full 5-strategy comparison with uplift percentages
```

---

## 6. Random Forest Model (Section 06)

### Implementation

The RF is implemented via the **ranger** package (fast C++ Random Forest) accessed through the **mlr3** machine learning framework. This combination provides:

- Fast training (C++ backend with multi-threading)
- Standardized interface for hyperparameter tuning
- Built-in cross-validation and metrics

### Hyperparameters Being Tuned

| Parameter | Search Range | What it controls |
|---|---|---|
| `mtry` | 1 to 2×√(n_features) | Number of features randomly considered at each split. Lower = more randomness between trees. |
| `min.node.size` | 1 to 50 | Minimum number of observations in a leaf node. Higher = simpler trees (less overfitting). |
| `sample.fraction` | 0.2 to 1.0 | Fraction of training data sampled for each tree. Lower = more diversity between trees. |
| `splitrule` | "gini" or "extratrees" | How to choose the best split. "gini" uses Gini impurity, "extratrees" picks random thresholds (faster, more regularized). |
| `num.trees` | 300 to 1,000 | Number of trees in the forest. More trees = more stable but diminishing returns. |
| `replace` | TRUE or FALSE | Whether to sample with replacement (TRUE = classic bagging, FALSE = subsampling). |

### Pipeline Flow per Strategy

```
Strategy Data → mlr3 Task → Custom CV → MBO Optimization → Best Params
                                                              ↓
                                            Train Final Model (all training data)
                                                              ↓
                                            Predict on Test Set → AUC, Brier, Accuracy
```

---

## 7. Hyperparameter Optimization — MBO

### What is Model-Based Optimization (MBO)?

Instead of trying random hyperparameter combinations (random search) or every possible combination (grid search), MBO builds a **surrogate model** that predicts which combinations are likely to work well.

**Step by step:**

1. **Warmup phase:** Try `n_dims × 5` random combinations (the "initial design"). This gives the surrogate model enough data to start learning.

2. **Bayesian Optimization phase:** For each remaining evaluation:
   - The surrogate model predicts the expected AUC for all untried combinations
   - An **acquisition function** balances exploitation (try what looks best) vs. exploration (try uncertain areas)
   - The most promising combination is evaluated
   - The surrogate model updates with the new result

3. **Early stopping:** If no improvement is found for `STAGNATION_ITERS` consecutive evaluations, optimization stops.

```
Random Design (warmup)          Bayesian Optimization (smart search)
├─ Eval 1: AUC = 0.72          ├─ Eval 31: AUC = 0.79  ← surrogate suggests
├─ Eval 2: AUC = 0.75          ├─ Eval 32: AUC = 0.78
├─ ...                         ├─ Eval 33: AUC = 0.80  ← new best!
└─ Eval 30: AUC = 0.77         └─ ... (stops if no improvement)
```

**Why MBO over grid search?** With 6 hyperparameters, a basic grid (10 values each) would require 1,000,000 evaluations. MBO typically finds near-optimal settings in 30–100 evaluations.

### Checkpointing

After each HPO run, a checkpoint file is saved (`checkpoint_<Strategy>.rds`). If the script crashes or is interrupted, it can resume from where it left off on the next run. Delete the checkpoint file to start fresh.

---

## 8. Evaluation Metrics

### AUC (Area Under the ROC Curve)

**What it measures:** How well the model distinguishes defaulters from non-defaulters, across all possible classification thresholds.

- **AUC = 0.5:** The model is no better than random guessing
- **AUC = 0.7–0.8:** Acceptable discrimination
- **AUC = 0.8–0.9:** Good discrimination
- **AUC > 0.9:** Excellent discrimination

**Intuition:** Pick a random defaulter and a random non-defaulter. AUC is the probability that the model assigns a higher default probability to the actual defaulter.

### Brier Score

**What it measures:** How well-calibrated the predicted probabilities are.

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

where $p_i$ is the predicted probability and $y_i$ is the actual outcome (0 or 1).

- **Brier = 0:** Perfect predictions
- **Brier = 0.25:** As bad as always predicting 50%
- **Lower is better**

**Difference from AUC:** AUC only cares about *ranking* (is the defaulter scored higher than the non-defaulter?). Brier also cares about *calibration* (if you predict 10% default probability, do roughly 10% of those companies actually default?).

### Uplift

Measures how much a strategy improves over the Base Model:

$$\text{Uplift}_\text{AUC} = \frac{\text{AUC}_\text{strategy} - \text{AUC}_\text{base}}{\text{AUC}_\text{base}} \times 100\%$$

$$\text{Uplift}_\text{Brier} = -\frac{\text{Brier}_\text{strategy} - \text{Brier}_\text{base}}{\text{Brier}_\text{base}} \times 100\%$$

Note the negative sign for Brier — lower Brier is better, so improvement is negative change.

---

## 9. Output Charts & Tables (Sections 07–08)

### Per-Strategy Charts (Section 07)

For each strategy that runs successfully, three charts are saved:

| Chart | Filename | What it shows |
|---|---|---|
| **Optimization History** | `01_OptimHistory_<Strategy>.png` | Scatter plot of AUC vs. iteration, showing warmup (random) vs. BO (smart) phases, with cumulative best line |
| **Feature Importance** | `02_FeatureImportance_<Strategy>.png` | Top 10 most important features, measured by Gini impurity reduction |
| **Calibration** | `03_Calibration_<Strategy>.png` | Predicted vs. observed default rates across 10 risk deciles |

**Reading the Optimization History chart:**
- Orange dots = warmup phase (random exploration)
- Blue triangles = BO phase (guided search)
- Green dashed line = best AUC found so far (cumulative max)
- Red star = overall best evaluation
- Ideally, the BO phase should show improvement (blue triangles above orange dots)

**Reading the Calibration chart:**
- 10 groups from lowest predicted risk (left) to highest (right)
- Blue bars = average predicted probability per group
- Grey bars = actual observed default rate per group
- Bars should match closely — if blue is much higher than grey, the model is overestimating risk for that group

### Strategy Comparison (Section 08)

| Output | Filename | What it shows |
|---|---|---|
| **AUC Leaderboard (Train)** | `04_AUC_Leaderboard_Train.png` | Bar chart comparing training AUC across all strategies, with uplift % labels |
| **AUC Leaderboard (Test)** | `05_AUC_Leaderboard_Test.png` | Same for test set — this is the primary result |
| **GT Table (Train)** | `06_Leaderboard_Train.png` | Formatted table with AUC, Brier, and uplift columns |
| **GT Table (Test)** | `07_Leaderboard_Test.png` | Same for test set |

**Reading the leaderboard charts:**
- Each bar = one strategy
- Label format: `78.5% (+2.3%)` means AUC is 78.5% with a 2.3% uplift over the Base Model
- Longer bars = better performance
- The test set leaderboard is what matters most — training performance can be inflated by overfitting

### Raw Results Files

| File | Contents |
|---|---|
| `rf_all_results.rds` | Complete results object (HPO archives, predictions, learners) |
| `rf_comparison_train.rds` | Training set comparison table as data.frame |
| `rf_comparison_test.rds` | Test set comparison table as data.frame |

Load in R with: `results <- readRDS("03_Charts/RF/rf_all_results.rds")`

---

## 10. VAE / DAE Artifact Generator (`generate_vae_dae_artifacts.R`)

This standalone script trains two neural networks and exports the features that Strategies A, B, and C consume. It lives at `01_Code/generate_vae_dae_artifacts.R`.

### Why is it a separate script?

The VAE/DAE artifacts only need to be generated **once** (for a given random seed and data split). Running them inside `main_RF.R` would add unnecessary training time every time you re-run the RF pipeline with different hyperparameters.

### What it trains

| Model | Architecture | Purpose |
|---|---|---|
| **VAE** (Variational Autoencoder) | 11 → 32 → 6 → 32 → 11 | Compresses 11 financial features into 6 latent dimensions, then reconstructs. Produces latent features (Strategy A) and anomaly scores (Strategy B). |
| **DAE** (Denoising Autoencoder) | 11 → 32 → 16 → 32 → 11 | Takes noisy financial features as input, learns to output clean features. Produces denoised features (Strategy C). |

### How it works

1. **Data pipeline**: Loads `data.rda` and applies the exact same preprocessing as `main_RF.R` (same seed, same stratified split, same quantile transformation). This guarantees the rows align perfectly.

2. **VAE training** (300 epochs): Minimizes β-VAE loss = MSE reconstruction + β × KL divergence. The KL term forces the latent space to be regular (close to a standard normal distribution).

3. **DAE training** (300 epochs): Adds Gaussian noise (σ = 0.3) to each input, trains the network to reconstruct the clean original. At inference, clean data goes in and "denoised" data comes out.

4. **Artifact extraction**:
   - **Strategy A** (`vae_latent_*.rds`): The encoder's `mu` output — a 6-column data.frame representing each observation's position in latent space.
   - **Strategy B** (`vae_anomaly_*.rds`): Per-observation MSE between the original and VAE-reconstructed features. High values = unusual companies.
   - **Strategy C** (`dae_denoised_*.rds`): The DAE's output — an 11-column data.frame with the same feature names as the original, but "cleaned."

### Configuration knobs

| Knob | Default | What it does |
|---|---|---|
| `VAE_LATENT_DIM` | 6 | Number of latent dimensions. Lower = more compression. |
| `VAE_HIDDEN_DIM` | 32 | Hidden layer width for VAE encoder/decoder. |
| `VAE_EPOCHS` | 300 | Training epochs for VAE. |
| `VAE_KL_WEIGHT` | 0.5 | β in β-VAE. Higher = more regular latent space. |
| `DAE_HIDDEN_DIM` | 32 | Hidden layer width for DAE. |
| `DAE_EPOCHS` | 300 | Training epochs for DAE. |
| `DAE_NOISE_SD` | 0.3 | Gaussian noise σ added during DAE training. |

### When to re-run it

You only need to re-run `generate_vae_dae_artifacts.R` if you:
- Change `RANDOM_SEED` or `TRAIN_PROP` (different data split)
- Change `DIVIDE_BY_TOTAL_ASSETS` (different feature values)
- Want to experiment with different VAE/DAE architectures

You do **not** need to re-run it when you change RF-specific settings (TRAIN_MODE, STRATEGIES_TO_RUN, etc.).

---

## 11. Helper Functions (RF_Subfunctions)

Located in `01_Code/RF_Subfunctions/`. Each file contains one function:

| File | Function | Purpose |
|---|---|---|
| `create_custom_cv.R` | `create_custom_cv(task, fold_ids)` | Converts a vector of fold assignments (1, 2, 3...) into an mlr3 resampling object. This ensures we use the same stratified folds as GLM/XGBoost. |
| `get_learner_config.R` | `get_learner_config(n_features)` | Returns the ranger learner and its hyperparameter search space. The `mtry` upper bound adapts to the number of features. |
| `run_hpo.R` | `run_hpo(task, cv, n_features, config)` | Runs the full MBO optimization loop. Handles initial design, BO iterations, early stopping, and checkpointing. |
| `train_and_eval.R` | `train_and_eval(params, train_task, test_task)` | Trains a final model with the best hyperparameters and evaluates on the test set. Returns AUC, accuracy, and Brier score. |
| `save_checkpoint.R` | `save_checkpoint(instance, filepath)` | Saves HPO progress to disk so it can be resumed later. |
| `load_checkpoint.R` | `load_checkpoint(filepath)` | Loads a previously saved checkpoint. Returns NULL if none exists. |
| `plot_optimization_history.R` | `plot_optimization_history(results)` | Creates the warmup-vs-BO scatter plot. |

---

## 12. Troubleshooting

### Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `Error in load(DATA_DIRECTORY)` | Data file not found | Check that `data/data.rda` exists at the project root |
| `Package 'mlr3mbo' not available` | R version too old | Upgrade to R >= 4.1; `mlr3mbo` needs recent R |
| `Strategy A/B/C skipped` | Expected on first run — VAE/DAE artifacts don't exist yet | This is normal. Run Phase 1 first (Base + D), then generate VAE/DAE artifacts, then re-run with all strategies (see Section 2 & 5) |
| `Killed` or out-of-memory | Too many cores × large data | Reduce `N_CORES` to 2–4 |
| Script takes very long | Large HPO budget | Reduce `N_EVALS` to 10–15 for testing |
| Results differ between runs | Different seed | Ensure `RANDOM_SEED` is the same. Note: `future` parallelization may introduce non-determinism. |

### How Long Does It Take?

Rough estimates (depends on data size and hardware):

| `TRAIN_MODE` | Per Strategy | All 5 Strategies |
|---|---|---|
| `"fast"` | ~5 min | ~25 min |
| `"medium"` | ~20 min | ~1.5 hours |
| `"slow"` | ~2 hours | ~10 hours |

---

## 13. Mathematical Background

### Random Forest — Formal Description

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where $\mathbf{x}_i \in \mathbb{R}^p$ and $y_i \in \{0, 1\}$:

1. For $b = 1, \ldots, B$ (number of trees):
   - Draw a bootstrap sample $\mathcal{D}_b$ of size $n' = \lfloor n \cdot \text{sample.fraction} \rfloor$
   - Grow a decision tree $T_b$ on $\mathcal{D}_b$:
     - At each node, randomly select $m = \text{mtry}$ features from the $p$ available
     - Find the best split among those $m$ features (using Gini impurity)
     - Split until nodes have $\leq \text{min.node.size}$ observations

2. Predict: $\hat{P}(y = 1 | \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$

### Gini Impurity

For a node with proportion $p$ of class 1:

$$G = 2p(1-p)$$

A split is chosen to maximize the **decrease in Gini impurity** (weighted by the number of observations going to each child node).

### Quantile Transformation (PIT)

For training data $\{x_1, \ldots, x_n\}$:

$$u_i = \frac{\text{rank}(x_i) - 0.5}{n}, \quad z_i = \Phi^{-1}(u_i)$$

where $\Phi^{-1}$ is the inverse standard normal CDF. For test data, the training ECDF $\hat{F}_n$ replaces the rank function:

$$u_{\text{test}} = \hat{F}_n(x_{\text{test}}), \quad z_{\text{test}} = \Phi^{-1}\left(\text{clip}(u_{\text{test}})\right)$$

### Model-Based Optimization

MBO approximates the expensive objective function $f(\boldsymbol{\theta}) = \text{CV-AUC}(\boldsymbol{\theta})$ with a surrogate model $\hat{f}$, then maximizes an acquisition function $\alpha$ to select the next evaluation point:

$$\boldsymbol{\theta}_{t+1} = \arg\max_{\boldsymbol{\theta}} \alpha(\boldsymbol{\theta} | \hat{f}, \mathcal{D}_{1:t})$$

Common acquisition functions include **Expected Improvement (EI)**:

$$\text{EI}(\boldsymbol{\theta}) = \mathbb{E}\left[\max(f(\boldsymbol{\theta}) - f^*, 0)\right]$$

where $f^*$ is the best observed value so far.

---

## File Structure Summary

```
CreditRisk_ML/
├── 01_Code/
│   ├── main_RF.R                          ← Main RF pipeline
│   ├── generate_vae_dae_artifacts.R       ← Prerequisite for Strategies A/B/C
│   ├── benchmark_hardware.R               ← Optional: recommends TRAIN_MODE
│   ├── Main.R                             ← Original pipeline (GLM + XGBoost)
│   ├── Subfunctions/                      ← Shared helpers (data prep, sampling)
│   │   ├── DataPreprocessing.R
│   │   ├── MVstratifiedsampling.R
│   │   ├── MVstratifiedsampling_CV.R
│   │   ├── QuantileTransformation.R
│   │   ├── BrierScore.R
│   │   └── ...
│   └── RF_Subfunctions/                   ← RF-specific helpers
│       ├── create_custom_cv.R
│       ├── get_learner_config.R
│       ├── run_hpo.R
│       ├── train_and_eval.R
│       ├── save_checkpoint.R
│       ├── load_checkpoint.R
│       └── plot_optimization_history.R
├── 03_Charts/
│   └── RF/                                ← All RF output goes here
│       ├── 01_OptimHistory_<Strategy>.png
│       ├── 02_FeatureImportance_<Strategy>.png
│       ├── 03_Calibration_<Strategy>.png
│       ├── 04_AUC_Leaderboard_Train.png
│       ├── 05_AUC_Leaderboard_Test.png
│       ├── 06_Leaderboard_Train.png
│       ├── 07_Leaderboard_Test.png
│       ├── rf_all_results.rds
│       ├── rf_comparison_train.rds
│       └── rf_comparison_test.rds
├── 06_Documentation/
│   └── Martin/
│       └── RF_Methodology.md              ← This document
└── data/
    ├── data.rda                           ← Input data
    └── pipeline_artifacts/                ← VAE/DAE outputs (for Strategies A–C)
        ├── vae_latent_train.rds
        ├── vae_latent_test.rds
        ├── vae_anomaly_train.rds
        ├── vae_anomaly_test.rds
        ├── dae_denoised_train.rds
        └── dae_denoised_test.rds
```
