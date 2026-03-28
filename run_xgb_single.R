#==============================================================================#
#==== run_xgb_single.R ========================================================#
#==============================================================================#
#
# PURPOSE:
#   Run one XGBoost configuration end-to-end (data → features → CV → XGBoost).
#   Called by run_xgb_all.R or directly via Rscript with four arguments:
#
#   Rscript run_xgb_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>
#
#   Examples:
#     Rscript run_xgb_single.R 01 f FALSE OoS
#     Rscript run_xgb_single.R 02 r FALSE OoT
#
#==============================================================================#

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L)
  stop("Usage: Rscript run_xgb_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>")

ARG_MODEL_GROUP           <- args[1L]
ARG_KEEP_FEATURES         <- args[2L]
ARG_INCLUDE_TIME_DYNAMICS <- as.logical(args[3L])
ARG_SPLIT_MODE            <- args[4L]

## ── 1. Source config (validates paths, creates output dirs) ──────────────────
library(here)
PROJ_ROOT    <- here::here("")
PIPELINE_DIR <- file.path(PROJ_ROOT, "01_Code", "1.Final Pipeline")

source(file.path(PIPELINE_DIR, "config.R"))

## Override with run-specific values AFTER config.R sourced its defaults
MODEL_GROUP           <- ARG_MODEL_GROUP
KEEP_FEATURES         <- ARG_KEEP_FEATURES
INCLUDE_TIME_DYNAMICS <- ARG_INCLUDE_TIME_DYNAMICS
SPLIT_MODE            <- ARG_SPLIT_MODE

RUN_LABEL <- sprintf(
  "%s%s_XGBoost_Manual",
  MODEL_GROUP, ifelse(SPLIT_MODE == "OoS", "a", "b")
)

message(sprintf(
  "\n\n═══════════ RUN: %s  [group=%s | feat=%s | TD=%s | split=%s] ═══════════\n",
  RUN_LABEL, MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE
))

## ── 2. Load packages ─────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr); library(tidyr); library(purrr); library(tibble)
  library(lubridate)
  library(caret); library(Matrix)
  library(xgboost)
  library(rBayesianOptimization)
  library(pROC); library(PRROC)
  library(future); library(future.apply); library(parallel)
  library(ggplot2); library(ggrepel); library(scales)
  library(gridExtra); library(hexbin)
  library(openxlsx); library(arrow)
})

## ── 3. Source subfunctions (only the two used by this pipeline) ──────────────
for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
  message(sprintf("  Sourced: %s", .fn))
}
rm(.fn)

## ── 4. Pipeline stages ───────────────────────────────────────────────────────
t0 <- proc.time()

message("\n── Stage 1: Data ────────────────────────────────────────────")
source(file.path(PIPELINE_DIR, "01_Data.R"))

message("\n── Stage 2: Feature Engineering ────────────────────────────")
source(file.path(PIPELINE_DIR, "02_FeatureEngineering.R"))

message("\n── Stage 3: CV Setup ────────────────────────────────────────")
source(file.path(PIPELINE_DIR, "02B_CV_Setup.R"))

message("\n── Stage 4: XGBoost HPO + Training ─────────────────────────")
source(file.path(PIPELINE_DIR, "04B_Train_XGBoost.R"))

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf(
  "\n═══ %s complete — %.1f min ═══\n",
  RUN_LABEL, elapsed / 60
))
