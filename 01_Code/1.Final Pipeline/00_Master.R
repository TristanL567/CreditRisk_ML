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
#   ── STAGE 2: CV Setup (R) ────────────────────────────────────────────────
#   02B_CV_Setup.R           Stratified firm-level k-fold CV construction
#                            (sector × y_ever, N_FOLDS from config.R)
#                            Saves cv_folds_{split}.rds for 04B
#
#   ── STAGE 3: Autoencoder (Python) ────────────────────────────────────────
#   03_Autoencoder.py        Beta-VAE on normal-scores features.
#                            Outputs latent dims + anomaly scores.
#                            (Required for MODEL_GROUP 04 and 05 only)
#
#   ── STAGE 3: Modelling (R) ───────────────────────────────────────────────
#   04B_Train_XGBoost.R      XGBoost with Bayesian HPO
#
#   ── STAGE 4: AutoML (Python) ─────────────────────────────────────────────
#   05_AutoGluon.py          AutoGluon AutoML — five model groups (01–05):
#                              01: raw balance sheet + sector
#                              02: financial ratios + sector
#                              03: ratios + sector + time dynamics
#                              04: ratios + sector + time dynamics + latent
#                              05: latent features + categoricals only
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
#     Suffix format: _{KEEP_FEATURES}_{TD|noTD}_{SPLIT_MODE}
#     Examples:
#       02_train_final_f_noTD_OoS.rds     Uniform(0,1) train — XGBoost (group 01)
#       02_train_final_r_TD_OoS.rds       Uniform(0,1) train — XGBoost (groups 03-05)
#       02_train_final_vae_r_TD_OoS.rds   N(0,1) train — VAE input (groups 03-05)
#
#   PYTHON / VAE OUTPUTS  (→ {PATH_ROOT}/03_Output/Latent/):
#       latent_train_r_TD_OoS.parquet     id, y, l1..l8 (+ dae variants)
#       latent_test_r_TD_OoS.parquet
#
#   MODEL OUTPUTS  (→ {PATH_ROOT}/03_Output/Final/):
#     {group}{split}_{model}/             e.g. 01a_XGBoost_Manual/
#       xgb_model.rds                     Fitted XGBoost model
#       predictions_test.parquet          Predictions on test set
#       eval_summary.json                 AUC-ROC, AP, Brier, BSS
#       feature_importance.csv            Gain-based importance
#
#   CHARTS  (→ {PATH_ROOT}/03_Charts/):
#     Feature importance, PDP, calibration, SHAP plots
#
# ── SPLIT CONTROL ─────────────────────────────────────────────────────────────
#   Set SPLIT_MODE in config.R before running:
#     "OoS"  →  stratified firm-level random split
#     "OoT"  →  firms whose last refdate falls in last OOT_N_YEARS → test
#
#   All outputs are suffixed _{feat}_{TD|noTD}_{split} automatically.
#   Python scripts (03_Autoencoder.py, 05_AutoGluon.py) have their own
#   MODEL_GROUP / SPLIT_MODE flags at the top — set them to match config.R.
#
#==============================================================================#


#==============================================================================#
#==== 00 - Bootstrap: config + reticulate =====================================#
#==============================================================================#

## config.R must be sourced BEFORE any library() call because it sets
## RETICULATE_PYTHON via Sys.setenv() — reticulate reads this on first load.

file.path(here::here(""))
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
  "glmnet", "xgboost", "ranger", "PRROC",
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

## Only source the two subfunctions actually used by the pipeline.
## (The Subfunctions/ folder contains legacy scripts from earlier experiments
##  that have external package dependencies not needed here.)
for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
  message(sprintf("  Sourced: %s", .fn))
}
rm(.fn)


#==============================================================================#
#==== 03 - Pipeline ===========================================================#
#==============================================================================#

pipeline_start <- proc.time()

## ── Stage 1: Data & Feature Engineering ─────────────────────────────────────
message("\n══ Stage 1/3: Data & Feature Engineering ════════════════════")
message(sprintf("   SPLIT_MODE = %s | KEEP_FEATURES = %s", SPLIT_MODE, KEEP_FEATURES))

source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "01_Data.R"))
source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "02_FeatureEngineering.R"))

message(sprintf(
  "   Outputs → %s/02_train_final_%s_%s_%s.rds  (+ vae, id vec variants)",
  PATH_DATA_OUT, KEEP_FEATURES,
  ifelse(INCLUDE_TIME_DYNAMICS, "TD", "noTD"), SPLIT_MODE
))

## ── Stage 2: Autoencoder (Python — run manually) ─────────────────────────────
message("\n══ Stage 2: Autoencoder (Python) ════════════════════════════")
message("   Run manually: 03_Autoencoder.py")
message(sprintf("   Set SPLIT_MODE = '%s' at top of script", SPLIT_MODE))
message(sprintf(
  "   Outputs → 03_Output/Latent/latent_train_r_%s_%s.parquet (+ test)",
  ifelse(INCLUDE_TIME_DYNAMICS, "TD", "noTD"), SPLIT_MODE
))

## ── Stage 2b: CV Setup (R) ───────────────────────────────────────────────────
message("\n══ Stage 2b: CV Setup ═══════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "02B_CV_Setup.R"))

## ── Stage 3: XGBoost (R) ─────────────────────────────────────────────────────
message("\n══ Stage 3: XGBoost ═════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "04B_Train_XGBoost.R"))

## ── Stage 4: AutoGluon (Python — run manually) ───────────────────────────────
message("\n══ Stage 4: AutoGluon (Python) ══════════════════════════════")
message("   Run manually: 05_AutoGluon.py")
message(sprintf("   Set MODEL_GROUP = '%s' and SPLIT_MODE = '%s' at top of script",
                MODEL_GROUP, SPLIT_MODE))
message(sprintf(
  "   Outputs → 03_Output/Final/%s%s_AutoGluon/",
  MODEL_GROUP, ifelse(SPLIT_MODE == "OoS", "a", "b")
))

## ── Stage 5: Evaluation & Charts (R) ─────────────────────────────────────────
message("\n══ Stage 5: Evaluation & Charts ═════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "06_Evaluation.R"))
source(file.path(PATH_ROOT, "01_Code", "1.Final Pipeline", "07_Charts.R"))

## ── Pipeline Summary ─────────────────────────────────────────────────────────
elapsed <- proc.time() - pipeline_start
message(sprintf(
  "\n══ R pipeline complete — %.1f min ═══════════════════════════",
  elapsed["elapsed"] / 60
))
message("   Python stages (03_Autoencoder.py, 05_AutoGluon.py) run separately.")