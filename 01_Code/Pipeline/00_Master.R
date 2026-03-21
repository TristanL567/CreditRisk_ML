#==============================================================================#
#==== 00_Master.R =============================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Single entry point for the entire pipeline. Sources config.R first,
#   then loads all packages, sources all subfunctions, and runs each stage
#   in sequence. Every stage can also be sourced independently after
#   sourcing config.R.
#
# ── PIPELINE OVERVIEW ─────────────────────────────────────────────────────────
#
#   00_Master.R              Orchestrator (this file)
#   config.R                 Single source of truth — all parameters and paths
#
#   ── STAGE 1: Data & Feature Engineering (R) ──────────────────────────────
#   01_Data.R                Raw data load, structural preprocessing,
#                            column filtering (KEEP_FEATURES flag)
#
#   02_FeatureEngineering.R  Split (OoS or OoT), time-series dynamics,
#                            sector deviation features, quantile transform
#                            (Uniform), imputation, final cleanup
#
#   ── STAGE 2: Autoencoder (Python) ────────────────────────────────────────
#   03_Autoencoder.py        Beta-VAE on normal-scores features.
#                            Outputs latent dims + anomaly scores.
#
#   ── STAGE 3: Modelling (R) ───────────────────────────────────────────────
#   04A_Train_GLM.R          Penalised GLM (elastic net / lasso) baseline
#   04B_Train_XGBoost.R      XGBoost with Bayesian HPO
#
#   ── STAGE 4: AutoML (Python) ─────────────────────────────────────────────
#   05_AutoGluon.py          AutoGluon AutoML — four model configurations:
#                              M1: raw uniform features
#                              M2: VAE latent dims + recon error
#                              M3: recon error (anomaly score) only
#                              M4: raw + VAE latent combined
#
#   ── STAGE 5: Evaluation & Charts (R) ─────────────────────────────────────
#   06_Evaluate.R            Test-set evaluation, leaderboard, uplift table
#   07_Charts.R              Feature importance, PDP, calibration,
#                            SHAP, residual diagnostics
#
# ── KEY LOCATIONS ─────────────────────────────────────────────────────────────
#
#   INPUT DATA:
#     Raw data (.rda)        C:/Users/Tristan Leiter/Documents/Privat/ILAB/Data/WS2025/data.rda
#
#   R PIPELINE OUTPUTS  (→ {PATH_ROOT}/02_Data/):
#     02_train_final_OoS.rds          Uniform(0,1) train features  — XGBoost / GLM
#     02_test_final_OoS.rds           Uniform(0,1) test features   — XGBoost / GLM
#     02_train_final_vae_OoS.rds      N(0,1) train features        — VAE input
#     02_test_final_vae_OoS.rds       N(0,1) test features         — VAE input
#     02_train_id_vec_OoS.rds         Firm id vector (train rows)  — for joining
#     02_test_id_vec_OoS.rds          Firm id vector (test rows)   — for joining
#     (Replace _OoS with _OoT for the out-of-time split)
#
#   PYTHON / VAE OUTPUTS  (→ {PATH_ROOT}/03_Output/):
#     Latent/
#       latent_train_OoS.parquet      id, y, z1..z32, vae_recon_error (train)
#       latent_test_OoS.parquet       id, y, z1..z32, vae_recon_error (test)
#       anomaly_train_OoS.parquet     id, y, vae_recon_error only (train)
#       anomaly_test_OoS.parquet      id, y, vae_recon_error only (test)
#     Models/VAE/OoS/
#       vae_weights.pt                Full VAE model weights
#       encoder_weights.pt            Encoder weights only
#       vae_config.json               Architecture + training config
#     Figures/VAE/
#       training_curves_OoS.png
#       latent_space_OoS.png
#
#   AUTOGLUON OUTPUTS  (→ {PATH_ROOT}/03_Output/AutoGluon/{M1..M4}_{OoS}/):
#     ag_predictor/                   Full AutoGluon predictor (model weights)
#     predictions_test.parquet        id, y, p_default, split_mode, model_name, year
#     eval_summary.json               AUC-ROC, AP, Brier, BSS, Recall@FPR
#     feature_importance.csv          Permutation feature importance
#
#   R MODEL OUTPUTS  (→ {PATH_ROOT}/03_Output/Models/):
#     GLM/                            Elastic net model objects + metrics
#     XGBoost/                        XGBoost model objects + metrics
#
#   CHARTS  (→ {PATH_ROOT}/03_Charts/):
#     Feature importance, PDP, calibration, SHAP plots
#
# ── SPLIT CONTROL ─────────────────────────────────────────────────────────────
#   Set SPLIT_MODE in config.R before running:
#     "OoS"  →  stratified firm-level random split
#     "OoT"  →  firms whose last refdate falls in last OOT_N_YEARS → test
#
#   All outputs are suffixed _OoS or _OoT automatically.
#   Downstream Python scripts (03, 05) have their own SPLIT_MODE flag
#   at the top of the file — set them to match.
#
#==============================================================================#


#==============================================================================#
#==== 00 - Bootstrap: config + reticulate =====================================#
#==============================================================================#

## config.R must be sourced BEFORE any library() call because it sets
## RETICULATE_PYTHON via Sys.setenv() — reticulate reads this on first load.

source(file.path(here::here(""), "config.R"))

library(reticulate)
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)


#==============================================================================#
#==== 01 - Packages ===========================================================#
#==============================================================================#

packages <- c(
  ## Core data manipulation
  "here", "dplyr", "tidyr", "purrr", "tibble", "lubridate", "data.table",
  ## Modelling infrastructure
  "caret", "Matrix",
  ## Models
  "glmnet", "xgboost", "ranger",
  ## Bayesian optimisation
  "rBayesianOptimization",
  ## Evaluation
  "pROC",
  ## Parallelism
  "future", "future.apply", "parallel",
  ## Visualisation
  "ggplot2", "ggrepel", "scales", "gridExtra", "hexbin", "pdp",
  ## Output
  "openxlsx", "arrow"      ## arrow: read/write parquet from R
)

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
    message(sprintf("  Installed: %s", pkg))
  }
  library(pkg, character.only = TRUE)
}

message("All packages loaded.")


#==============================================================================#
#==== 02 - Source Subfunctions ================================================#
#==============================================================================#

source_dir <- function(path) {
  files <- list.files(path, pattern = "\\.R$", full.names = TRUE)
  invisible(lapply(files, function(f) {
    tryCatch(
      source(f, echo = FALSE, local = FALSE),
      error = function(e)
        stop(sprintf("Failed to source %s: %s", f, e$message))
    )
  }))
  message(sprintf("  Sourced %d file(s) from: %s", length(files), path))
}

source_dir(PATH_FN_GENERAL)
source_dir(PATH_FN_XGB)


#==============================================================================#
#==== 03 - Pipeline ===========================================================#
#==============================================================================#

pipeline_start <- proc.time()

## ── Stage 1: Data & Feature Engineering ─────────────────────────────────────
message("\n══ Stage 1/3: Data & Feature Engineering ════════════════════")
message(sprintf("   SPLIT_MODE = %s | KEEP_FEATURES = %s", SPLIT_MODE, KEEP_FEATURES))

source(file.path(PATH_ROOT, "01_Code", "01_Data.R"))
source(file.path(PATH_ROOT, "01_Code", "02_FeatureEngineering.R"))

message(sprintf(
  "   Outputs → %s/02_train_final_%s.rds  (+ vae, id vec variants)",
  PATH_DATA_OUT, SPLIT_MODE
))

## ── Stage 2: Autoencoder (Python — run manually) ─────────────────────────────
message("\n══ Stage 2: Autoencoder (Python) ════════════════════════════")
message("   Run manually: 03_Autoencoder.py")
message(sprintf("   Set SPLIT_MODE = '%s' at top of script", SPLIT_MODE))
message(sprintf(
  "   Outputs → 03_Output/Latent/latent_train_%s.parquet (+ anomaly, test)",
  SPLIT_MODE
))

## ── Stage 3a: GLM (R) ────────────────────────────────────────────────────────
message("\n══ Stage 3a: GLM ════════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "04A_Train_GLM.R"))

## ── Stage 3b: XGBoost (R) ────────────────────────────────────────────────────
message("\n══ Stage 3b: XGBoost ════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "04B_Train_XGBoost.R"))

## ── Stage 4: AutoGluon (Python — run manually) ───────────────────────────────
message("\n══ Stage 4: AutoGluon (Python) ══════════════════════════════")
message("   Run manually: 05_AutoGluon.py")
message(sprintf("   Set SPLIT_MODE = '%s' and MODEL = 'M1'/'M2'/'M3'/'M4'", SPLIT_MODE))
message(sprintf(
  "   Outputs → 03_Output/AutoGluon/M1_%s/ (+ M2, M3, M4)",
  SPLIT_MODE
))

## ── Stage 5: Evaluation & Charts (R) ─────────────────────────────────────────
message("\n══ Stage 5: Evaluation & Charts ═════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "06_Evaluate.R"))
source(file.path(PATH_ROOT, "01_Code", "07_Charts.R"))

## ── Pipeline Summary ─────────────────────────────────────────────────────────
elapsed <- proc.time() - pipeline_start
message(sprintf(
  "\n══ R pipeline complete — %.1f min ═══════════════════════════",
  elapsed["elapsed"] / 60
))
message("   Python stages (03_Autoencoder.py, 05_AutoGluon.py) run separately.")