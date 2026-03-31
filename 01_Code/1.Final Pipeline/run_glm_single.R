#==============================================================================#
#==== run_glm_single.R ========================================================#
#==============================================================================#
#
# PURPOSE:
#   Run GLM (elastic-net) training for a single feature-set / split combination.
#   Assumes data prep (01_Data → 02_FeatureEngineering) and, for Groups 04-05,
#   autoencoder outputs are already on disk.
#
#   Usage:
#     Rscript run_glm_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>
#
#   Examples:
#     Rscript run_glm_single.R 01 f FALSE OoS
#     Rscript run_glm_single.R 04 r TRUE  OoT
#
#==============================================================================#

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L)
  stop("Usage: Rscript run_glm_single.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>")

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
  "\n\n═══════════ GLM TRAIN: group=%s | feat=%s | TD=%s | split=%s ═══════════\n",
  MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE
))

## ── 2. Load packages ─────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr); library(tibble); library(purrr)
  library(glmnet)
  library(pROC)
  library(arrow)
  library(ggplot2); library(scales)
  library(openxlsx)
})

## ── 3. GLM training ──────────────────────────────────────────────────────────
t0 <- proc.time()

source(file.path(PIPELINE_DIR, "04A_Train_GLM.R"))

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf(
  "\n═══ GLM complete [group=%s | feat=%s | TD=%s | split=%s] — %.1f min ═══\n",
  MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE, elapsed / 60
))
