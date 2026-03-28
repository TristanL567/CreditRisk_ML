# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Credit risk modeling pipeline for corporate default prediction at OeNB (Austrian National Bank). Combines penalized GLM, XGBoost (Bayesian HPO), Beta-VAE (dimensionality reduction), and AutoGluon ensemble across four feature configurations.

## Running the Pipeline

The pipeline is executed sequentially by stage. There is no single build command — each script must be run in order.

**R scripts** (run from within the R project `CreditRisk_ML.Rproj`):
```r
source("01_Code/1.Final Pipeline/config.R")       # Load config first — always
source("01_Code/1.Final Pipeline/01_Data.R")
source("01_Code/1.Final Pipeline/02_FeatureEngineering.R")
source("01_Code/1.Final Pipeline/02B_CV_Setup.R")
source("01_Code/1.Final Pipeline/04A_Train_GLM.R")
source("01_Code/1.Final Pipeline/04B_Train_XGBoost.R")
source("01_Code/1.Final Pipeline/06_Evaluation.R")
source("01_Code/1.Final Pipeline/07_Charts.R")
```

**Python scripts** (activate the Conda environment first):
```bash
conda activate ag150
python "01_Code/1.Final Pipeline/03_Autoencoder.py"
python "01_Code/1.Final Pipeline/05_AutoGluon.py"
```

**Conda environment setup:**
```bash
conda env create -f "01_Code/2.AutoML/ag150.yml"
```

See `00_Master.R` for the full pipeline orchestration overview and stage descriptions.

## Architecture

### Stage Order & Data Flow

```
config.R
  → 01_Data.R            (load raw .rda panel data)
  → 02_FeatureEngineering.R  (firm-level train/test split → transforms)
  → 02B_CV_Setup.R       (5-fold stratified CV groups)
  ↓
  ├─ 03_Autoencoder.py   (Beta-VAE on Normal[0,1] features → latent dims + recon error)
  ├─ 04A_Train_GLM.R     (penalized GLM with elastic net)
  └─ 04B_Train_XGBoost.R (XGBoost + Bayesian HPO)
  → 05_AutoGluon.py      (AutoML on M1–M4 feature configs)
  → 06_Evaluation.R      (test-set metrics + leaderboard)
  → 07_Charts.R          (PDP, importance, calibration)
```

### Central Configuration: `config.R`

**All paths, seeds, flags, and hyperparameters live here.** Always `source("config.R")` before running any other script. Key settings:

- `SPLIT_MODE`: `"OoS"` (stratified random, 70/30) or `"OoT"` (temporal, last N years to test)
- `SEED`: `123` — used in both R and Python for reproducibility
- `N_FOLDS`: `5` (cross-validation folds)
- `QUANTILE_TRANSFORM` / `TRANSFORM_BOUNDED01`: Feature transformation flags
- VAE hyperparameters (`beta`, `gamma`, `z_dim`, `n_epochs`)
- XGBoost Bayesian HPO settings (`Nrounds_bo`, `nrounds_final`)

### Feature Normalization Convention

Two parallel feature sets are maintained throughout:
- **Uniform[0,1]** → input to XGBoost, GLM, AutoGluon (tree-based models)
- **Normal[0,1]** (quantile-normalized) → VAE input only (`*_vae_*.rds` files)

### AutoGluon Feature Configurations (M1–M4)

| Config | Features |
|--------|----------|
| M1 | Raw Uniform[0,1] features (~508 dims) |
| M2 | VAE latent dims + reconstruction error |
| M3 | Reconstruction error only |
| M4 | Raw + latent + reconstruction combined |

Results land in `03_Output/AutoGluon/{M1..M4}_{split}/`.

### Leakage Prevention

All sector statistics, quantile transforms, and other data-derived statistics are **fit on Train only** and applied to Test. In `OoT` mode, all observations of test-period firms are excluded from Train (firm-level exclusion, not just observation-level).

### R–Python Bridge

`reticulate` is used to call Python from R. The Python executable path is set in `config.R`. Data is exchanged via `.rds` files (R → Python reads via `rpy2` or pandas), and `.parquet` files (Python → R reads via `arrow`).

## Key Files

| File | Purpose |
|------|---------|
| `01_Code/1.Final Pipeline/config.R` | Master config — single source of truth |
| `01_Code/1.Final Pipeline/00_Master.R` | Pipeline orchestration docs |
| `01_Code/2.AutoML/ag150.yml` | Conda env (Python 3.11, AutoGluon 1.5, PyTorch 2.6 CUDA 12.4) |
| `CreditRisk_ML.Rproj` | R project file |

## Directory Layout

- `01_Code/1.Final Pipeline/` — production scripts (numbered by stage)
- `01_Code/2.AutoML/` — AutoGluon configuration and environment
- `01_Code/3.TimeSeries/` — time series analysis scripts
- `02_Data/` — feature matrices (gitignored)
- `03_Output/` — model outputs and predictions (gitignored)
- `03_Charts/` / `04_Charts/` — visualization outputs
- `03_Research/` — research notebooks and methodology exploration
- `05_Documentation/` — methodology PDFs and meeting notes
- `06_archiv/` — archived experimental code (do not modify)
