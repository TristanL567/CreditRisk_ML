#==============================================================================#
#==== run_data_only.R =========================================================#
#==============================================================================#
# Runs data prep + feature engineering only (no CV, no model training).
# Usage: Rscript run_data_only.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L)
  stop("Usage: Rscript run_data_only.R <MODEL_GROUP> <KEEP_FEATURES> <INCLUDE_TD> <SPLIT_MODE>")

library(here)
PROJ_ROOT    <- here::here("")
PIPELINE_DIR <- file.path(PROJ_ROOT, "01_Code", "1.Final Pipeline")

source(file.path(PIPELINE_DIR, "config.R"))

MODEL_GROUP           <- args[1L]
KEEP_FEATURES         <- args[2L]
INCLUDE_TIME_DYNAMICS <- as.logical(args[3L])
SPLIT_MODE            <- args[4L]

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr); library(tidyr); library(purrr); library(tibble)
  library(lubridate)
  library(caret); library(Matrix)
  library(pROC); library(PRROC)
  library(ggplot2); library(ggrepel); library(scales)
  library(gridExtra); library(hexbin)
  library(openxlsx); library(arrow)
})

for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
}
rm(.fn)

message(sprintf("\n── Data prep: group=%s feat=%s TD=%s split=%s ──",
                MODEL_GROUP, KEEP_FEATURES, INCLUDE_TIME_DYNAMICS, SPLIT_MODE))

source(file.path(PIPELINE_DIR, "01_Data.R"))
source(file.path(PIPELINE_DIR, "02_FeatureEngineering.R"))

message("── Data prep complete ──")
