#==============================================================================#
#==== run_data_prep.R =========================================================#
#==============================================================================#
#
# PURPOSE:
#   Run data preparation stages only (no model training).
#   Stages: 01_Data → 02_FeatureEngineering → 02B_CV_Setup
#
#   Usage:
#     Rscript run_data_prep.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>
#
#   Examples:
#     Rscript run_data_prep.R 03 r TRUE OoT
#     Rscript run_data_prep.R 01 f FALSE OoS
#
#==============================================================================#

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L)
  stop("Usage: Rscript run_data_prep.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>")

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
  "\n\n═══════════ DATA PREP: group=%s | feat=%s | TD=%s | split=%s ═══════════\n",
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
})

## ── 3. Source subfunctions ───────────────────────────────────────────────────
for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
  message(sprintf("  Sourced: %s", .fn))
}
rm(.fn)

## ── 4. Data prep stages only ─────────────────────────────────────────────────
t0 <- proc.time()

message("\n── Stage 1: Data ────────────────────────────────────────────")
source(file.path(PIPELINE_DIR, "01_Data.R"))

message("\n── Stage 2: Feature Engineering ────────────────────────────")
source(file.path(PIPELINE_DIR, "02_FeatureEngineering.R"))

message("\n── Stage 3: CV Setup ────────────────────────────────────────")
source(file.path(PIPELINE_DIR, "02B_CV_Setup.R"))

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf(
  "\n═══ Data prep complete [feat=%s | TD=%s | split=%s] — %.1f min ═══\n",
  KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE, elapsed / 60
))
