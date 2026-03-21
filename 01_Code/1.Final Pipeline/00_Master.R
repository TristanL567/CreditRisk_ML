#==============================================================================#
#==== 00_Master.R =============================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   
# UPDATED:  2026
#
# PURPOSE:
#   Single entry point for the entire pipeline. Sources config.R first,
#   then installs/loads all packages, sources all subfunctions, and runs
#   each stage in sequence. Every stage can also be sourced independently —
#   all parameters are read from config.R.
#
# PIPELINE OVERVIEW:
#   00_Master.R              Orchestrator (this file)
#   config.R                 Single source of truth — all parameters and paths
#   ─────────────────────────────────────────────────────────────────────────
#   01_Data.R                Raw data load, preprocessing, column filtering
#   02_FeatureEngineering.R  Time-series dynamics, sector features,
#                            quantile transform, imputation, final cleanup
#   ─────────────────────────────────────────────────────────────────────────
#   03_Split.R               Stratified firm-level train/test split,
#                            leakage check, CV fold construction
#   ─────────────────────────────────────────────────────────────────────────
#   04_VAE_Prep.R            Column classification (continuous / bounded /
#                            binary), VAE input matrix, feat_dist
#   05_VAE_Train.R           Architecture derivation, manual training loop,
#                            KL warmup, early stopping, weight restore
#   06_Strategies.R          Strategy A (latent features), B (anomaly score),
#                            C (DAE denoising) — assemble train/test sets
#   ─────────────────────────────────────────────────────────────────────────
#   07_Train.R               XGBoost Bayesian HPO for all four strategies
#   08_Evaluate.R            Test-set evaluation, leaderboard, uplift table
#   09_Charts.R              Feature importance, PDP, calibration,
#                            hexbin interactions, residual diagnostics
#   ─────────────────────────────────────────────────────────────────────────
#   Subfunctions/
#     MVstratifiedsampling.R         Firm-level stratified train/test split
#     MVstratifiedsampling_CV.R      Stratified CV folds (full output)
#     MVstratifiedsampling_CV_ID.R   CV fold assignment at firm-ID level
#     MVstratifiedsampling_CV_Split.R Map firm-fold IDs to row indices
#     QuantileTransformation.R       Rank-based Gaussian normalisation
#     XGBoost_Training_revised.R     Full HPO + final model training
#     XGBoost_Test_revised.R         Test-set evaluation with Youden threshold
#
# STAGE GROUPINGS:
#   DATA          (01–02) : Load, preprocess, engineer features
#   SPLIT         (03)    : Train/test and CV split — run once, checkpoint
#   VAE           (04–06) : Encode, train, extract representations
#   MODELLING     (07–09) : Train, evaluate, visualise
#
#==============================================================================#


#==============================================================================#
#==== 00 - Python / Reticulate ================================================#
#==============================================================================#

## Must be sourced before library(reticulate) / library(keras) / library(autotab).
## config.R sets RETICULATE_PYTHON via Sys.setenv() as its first action —
## source it here before any library() call.

source(file.path(here::here(""), "config.R"))

library(reticulate)
use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)


#==============================================================================#
#==== 01 - Packages ===========================================================#
#==============================================================================#

packages <- c(
  ## Core data manipulation.
  "here", "dplyr", "tidyr", "purrr", "tibble", "lubridate",
  "data.table",
  ## Modelling infrastructure.
  "caret", "Matrix",
  ## Models.
  "glmnet", "xgboost", "ranger", "adabag",
  ## Bayesian optimisation.
  "rBayesianOptimization",
  ## mlr3 stack.
  "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo",
  "mlr3measures", "paradox",
  ## Evaluation.
  "pROC",
  ## Parallelism.
  "future", "future.apply", "parallel",
  ## VAE / deep learning.
  "autotab", "keras", "reticulate", "tensorflow",
  ## Clustering (soft interaction features).
  "mclust",
  ## Visualisation.
  "ggplot2", "ggrepel", "scales", "Ckmeans.1d.dp",
  "pdp", "gridExtra", "hexbin",
  ## Output.
  "openxlsx"
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
      error = function(e) stop(sprintf("Failed to source %s: %s", f, e$message))
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

## ── Stage 1: Data ───────────────────────────────────────────────────────────
message("\n══ Stage 1/5: Data ══════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "01_Data.R"))

## ── Stage 2: Feature Engineering ────────────────────────────────────────────
message("\n══ Stage 2/5: Feature Engineering ══════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "02_FeatureEngineering.R"))

## ── Stage 3: Split ──────────────────────────────────────────────────────────
message("\n══ Stage 3/5: Split ═════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "03_Split.R"))

## ── Stage 4: VAE ────────────────────────────────────────────────────────────
message("\n══ Stage 4/5: VAE ═══════════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "04_VAE_Prep.R"))
source(file.path(PATH_ROOT, "01_Code", "05_VAE_Train.R"))
source(file.path(PATH_ROOT, "01_Code", "06_Strategies.R"))

## ── Stage 5: Modelling ──────────────────────────────────────────────────────
message("\n══ Stage 5/5: Modelling ═════════════════════════════════════")
source(file.path(PATH_ROOT, "01_Code", "07_Train.R"))
source(file.path(PATH_ROOT, "01_Code", "08_Evaluate.R"))
source(file.path(PATH_ROOT, "01_Code", "09_Charts.R"))

## ── Pipeline Summary ────────────────────────────────────────────────────────
elapsed <- proc.time() - pipeline_start

message(sprintf(
  "\n══ Pipeline complete — %.1f min ═════════════════════════════",
  elapsed["elapsed"] / 60
))