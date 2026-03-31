#==============================================================================#
#==== run_xgb_single.R ========================================================#
#==============================================================================#
#
# PURPOSE:
#   Run XGBoost training for a single feature-set / split combination.
#   Assumes data prep (01_Data → 02_FeatureEngineering → 02B_CV_Setup) and
#   autoencoder outputs are already on disk.
#
#   Usage:
#     Rscript run_xgb_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>
#
#   Examples:
#     Rscript run_xgb_single.R 01 f FALSE OoS
#     Rscript run_xgb_single.R 04 r TRUE  OoT
#
#==============================================================================#

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L)
  stop("Usage: Rscript run_xgb_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>")

ARG_MODEL_GROUP           <- args[1L]
ARG_KEEP_FEATURES         <- args[2L]
ARG_INCLUDE_TIME_DYNAMICS <- as.logical(args[3L])
ARG_SPLIT_MODE            <- args[4L]

## ── 1. Source config ─────────────────────────────────────────────────────────
library(here)
PROJ_ROOT    <- here::here("")
PIPELINE_DIR <- file.path(PROJ_ROOT, "01_Code", "1.Final Pipeline")

source(file.path(PIPELINE_DIR, "config.R"))

## Override with run-specific values AFTER config.R sourced its defaults
MODEL_GROUP           <- ARG_MODEL_GROUP
KEEP_FEATURES         <- ARG_KEEP_FEATURES
INCLUDE_TIME_DYNAMICS <- ARG_INCLUDE_TIME_DYNAMICS
SPLIT_MODE            <- ARG_SPLIT_MODE


message(sprintf(
  "\n\n═══════════ XGB TRAIN: group=%s | feat=%s | TD=%s | split=%s ═══════════\n",
  MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE
))

## ── 2. Load packages ─────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr); library(tidyr); library(purrr); library(tibble)
  library(lubridate)
  library(caret); library(Matrix)
  library(pROC); library(PRROC)
  library(ggplot2); library(scales)
  library(openxlsx); library(arrow)
  library(xgboost)
  library(rBayesianOptimization)
})

## ── 3. Source subfunctions ───────────────────────────────────────────────────
for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
  message(sprintf("  Sourced: %s", .fn))
}
rm(.fn)

## ── 4. Regenerate CV folds for this group's training data ────────────────────
## cv_folds_{SPLIT_MODE}.rds is shared across groups but row indices are
## group-specific (different groups have different training set sizes after
## feature filtering). Regenerate here to ensure indices match the actual
## training data before xgb.cv runs.
## [ADDED 2026-03-30: fix for exit-139 segfault in xgb.cv for groups 02–05 OoS]
message("\n── Regenerating CV folds for this group ─────────────────────")
source(file.path(PIPELINE_DIR, "02B_CV_Setup.R"))

## ── 5. XGBoost training ──────────────────────────────────────────────────────
t0 <- proc.time()

source(file.path(PIPELINE_DIR, "04B_Train_XGBoost.R"))

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf(
  "\n═══ XGBoost complete [group=%s | feat=%s | TD=%s | split=%s] — %.1f min ═══\n",
  MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE, elapsed / 60
))
