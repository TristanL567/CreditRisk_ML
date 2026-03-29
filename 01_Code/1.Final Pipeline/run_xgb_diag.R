#==============================================================================#
#==== run_xgb_diag.R — XGBoost Diagnostic Test ================================#
#==============================================================================#
#
# PURPOSE:
#   Fast smoke-test for 04B_Train_XGBoost.R.
#   Runs group 01 / OoS with drastically reduced HPO budget
#   (2 init + 2 iter, 50 nrounds max) to verify the pipeline
#   executes end-to-end without errors in < 2 min.
#   Does NOT re-run data prep — loads existing .rds files directly.
#
# Usage:
#   Rscript run_xgb_diag.R
#
#==============================================================================#

library(here)
PROJ_ROOT    <- here::here("")
PIPELINE_DIR <- file.path(PROJ_ROOT, "01_Code", "1.Final Pipeline")

## 1. Source config (validates paths, creates output dirs)
source(file.path(PIPELINE_DIR, "config.R"))

## 2. Override to simplest combination
MODEL_GROUP           <- "01"
KEEP_FEATURES         <- "f"
INCLUDE_TIME_DYNAMICS <- FALSE
SPLIT_MODE            <- "OoS"

## 3. Reduce HPO budget drastically for speed
XGB_CONFIG$n_init_points    <- 2L
XGB_CONFIG$n_iter_bayes     <- 2L
XGB_CONFIG$nrounds_bo       <- 50L
XGB_CONFIG$nrounds_final    <- 100L
XGB_CONFIG$early_stop_bo    <- 10L
XGB_CONFIG$early_stop_final <- 20L

message("\n=== XGBoost DIAGNOSTIC RUN ===")
message("  Group: 01 | Split: OoS | HPO: 2+2 init/iter | nrounds: 50/100")
message("  (Reduced budget — for smoke-test only)\n")

## 4. Load packages
suppressPackageStartupMessages({
  library(data.table); library(dplyr); library(tibble)
  library(lubridate);  library(caret); library(Matrix)
  library(xgboost);    library(rBayesianOptimization)
  library(pROC);       library(PRROC)
  library(ggplot2);    library(openxlsx); library(arrow)
})

## 5. Source subfunctions
for (.fn in c("DataPreprocessing.R", "QuantileTransformation.R")) {
  source(file.path(PATH_FN_GENERAL, .fn), echo = FALSE, local = FALSE)
}
rm(.fn)

## 6. Verify input files exist before touching XGBoost
required_files <- c(
  get_split_path(SPLIT_OUT_TRAIN_FINAL),
  get_split_path(SPLIT_OUT_TEST_FINAL),
  get_split_path(SPLIT_OUT_TRAIN_IDS),
  get_split_path(SPLIT_OUT_TEST_IDS),
  file.path(PATH_DATA_OUT, sprintf("cv_folds_%s.rds", SPLIT_MODE))
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L)
  stop("Missing input files:\n", paste(" -", missing_files, collapse = "\n"))
message("  Input files: ALL PRESENT")
for (f in required_files) {
  sz <- round(file.size(f) / 1024^2, 1)
  message(sprintf("    %s  (%.1f MB)", basename(f), sz))
}

## 7. Run XGBoost stage only (data already prepared)
t0 <- proc.time()
message("\n── Running 04B_Train_XGBoost.R ──────────────────────────────")
source(file.path(PIPELINE_DIR, "04B_Train_XGBoost.R"))

elapsed <- (proc.time() - t0)["elapsed"]
message(sprintf("\n=== DIAGNOSTIC COMPLETE — %.1f sec ===", elapsed))
message("    If you see this line, the pipeline ran end-to-end without errors.")
