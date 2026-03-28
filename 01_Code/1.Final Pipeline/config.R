#==============================================================================#
#==== config.R ================================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Single source of truth for the entire pipeline.
#   All paths, seeds, flags, and model parameters live here.
#   Sourced first by 00_Master.R; can also be sourced independently.
#
#==============================================================================#


#==============================================================================#
#==== 00 - Python / Reticulate ================================================#
#==============================================================================#

Sys.setenv(RETICULATE_PYTHON = "C:/venvs/autotab_env/Scripts/python.exe")


#==============================================================================#
#==== 01 - Paths ==============================================================#
#==============================================================================#

PATH_ROOT      <- file.path(here::here(""))
PATH_DATA_RAW  <- "C:/Users/Tristan Leiter/Documents/Privat/ILAB/Data/WS2025"
PATH_DATA_FILE <- file.path(PATH_DATA_RAW, "data.rda")
PATH_DATA_OUT  <- file.path(PATH_ROOT, "02_Data")
PATH_CHARTS    <- file.path(PATH_ROOT, "03_Charts")
DIR_FINAL_OUT  <- file.path(PATH_ROOT, "03_Output", "Final")

PATH_CHARTS_XGB <- file.path(PATH_CHARTS, "VAE", "FinancialRatios", "XGBoost")

PATH_FN_GENERAL <- file.path(PATH_ROOT, "01_Code", "0.Before Final Presentation", "Subfunctions")


#==============================================================================#
#==== 02 - Reproducibility ====================================================#
#==============================================================================#

SEED <- 123L


#==============================================================================#
#==== 03 - Data Flags =========================================================#
#==============================================================================#

## Feature families to retain (see 01_Data.R).
## "r" = ratios only | "f" = raw financials only | "both" = all
KEEP_FEATURES <- "both"

## Whether to compute time-series dynamics in 02_FeatureEngineering.R (Stage B).
## Groups 01 and 02 require FALSE; groups 03, 04, 05 require TRUE.
INCLUDE_TIME_DYNAMICS <- TRUE

## Model group — controls feature set used by 04B and 05_AutoGluon.
## "01" Raw Balance Sheet + Sector
## "02" Financial Ratios + Sector
## "03" Financial Ratios + Sector + Time Dynamics
## "04" Financial Ratios + Sector + Time Dynamics + Latent Features (VAE)
## "05" Latent Features (VAE) Only
MODEL_GROUP <- "01"   ## <── change here: "01" | "02" | "03" | "04" | "05"

## Quantile-normalise continuous features before VAE input.
QUANTILE_TRANSFORM  <- TRUE

## If TRUE, also transform bounded [0,1] features.
TRANSFORM_BOUNDED01 <- FALSE

## Minimum firms per size x sector x year cell before a sparsity warning.
MIN_SS_FIRMS <- 5L

## Column name of the binary default target.
TARGET_COL <- "y"

## Stratification variables for the OoS firm-level split.
STRAT_VARS    <- c("sector", "y_ever")
STRAT_VARS_CV <- c("sector", "y")

## Train proportion for OoS split (remainder goes to Test).
TRAIN_SIZE <- 0.7


#==============================================================================#
#==== 04 - Split Configuration ================================================#
#==============================================================================#

## SPLIT_MODE: controls which split feeds stages 03 onwards.
##   "OoS" — stratified firm-level random split (TRAIN_SIZE above)
##   "OoT" — firms whose last refdate falls within the last OOT_N_YEARS
##            go entirely to Test (all their observations, no leakage)
##
SPLIT_MODE  <- "OoS"   ## <── change here: "OoS" or "OoT"
OOT_N_YEARS <- 1L      ## OoT only: number of trailing years assigned to Test

## Output paths — suffix encodes feature config + split mode so all 10
## feature/split combinations can coexist on disk simultaneously.
## Use get_split_path() / get_vae_path() in all downstream stages.
##
## Suffix format:  _{KEEP_FEATURES}_{TD|noTD}_{SPLIT_MODE}
## Examples:
##   KEEP_FEATURES="f", INCLUDE_TIME_DYNAMICS=FALSE, SPLIT_MODE="OoS"
##     → 02_train_final_f_noTD_OoS.rds
##   KEEP_FEATURES="r", INCLUDE_TIME_DYNAMICS=TRUE,  SPLIT_MODE="OoT"
##     → 02_train_final_r_TD_OoT.rds

.feat_suffix  <- function() {
  td <- ifelse(INCLUDE_TIME_DYNAMICS, "TD", "noTD")
  paste0("_", KEEP_FEATURES, "_", td)
}
.split_suffix <- function() paste0(.feat_suffix(), "_", SPLIT_MODE)

get_vae_path <- function(base_name) {
  ## Returns the normal-scores (.rds) path for VAE input.
  ## Example: get_vae_path("02_train_final")  [KEEP_FEATURES="r", TD=TRUE, OoS]
  ##   → <PATH_DATA_OUT>/02_train_final_vae_r_TD_OoS.rds
  file.path(PATH_DATA_OUT, paste0(base_name, "_vae", .split_suffix(), ".rds"))
}

get_split_path <- function(base_name) {
  ## Returns the correct .rds path for any split/feature-config-dependent output.
  ## Example: get_split_path("02_train_final")  [KEEP_FEATURES="f", TD=FALSE, OoS]
  ##   → <PATH_DATA_OUT>/02_train_final_f_noTD_OoS.rds
  file.path(PATH_DATA_OUT, paste0(base_name, .split_suffix(), ".rds"))
}

## Canonical split output names (pass to get_split_path()).
SPLIT_OUT_TRAIN_FINAL  <- "02_train_final"
SPLIT_OUT_TEST_FINAL   <- "02_test_final"
SPLIT_OUT_TRAIN_IDS    <- "02_train_id_vec"
SPLIT_OUT_TEST_IDS     <- "02_test_id_vec"


#==============================================================================#
#==== 05 - Cross-Validation ===================================================#
#==============================================================================#

N_FOLDS <- 5L


#==============================================================================#
#==== 06 - VAE Configuration ==================================================#
#==============================================================================#

VAE_CONFIG <- list(
  encoder_dims     = NULL,   ## Derived in 05_VAE_Train.R
  decoder_dims     = NULL,
  latent_dim       = NULL,
  epochs           = 150L,
  batch_size       = 256L,
  beta             = 3.0,
  lr               = 1e-3,
  temperature      = 0.5,
  patience         = 10L,
  kl_warmup        = TRUE,
  kl_warmup_epochs = 20L
)

## TRUE  → downstream models see VAE-derived features + target only.
## FALSE → downstream models see original + VAE-derived features.
USE_VAE_ONLY <- TRUE


#==============================================================================#
#==== 07 - XGBoost Configuration ==============================================#
#==============================================================================#

XGB_CONFIG <- list(
  n_init_points    = 10L,
  n_iter_bayes     = 20L,
  early_stop_bo    = 20L,
  early_stop_final = 50L,
  nrounds_bo       = 1000L,
  nrounds_final    = 2000L,
  nthread          = parallel::detectCores() - 1L,
  eval_metric      = "auc"
)


#==============================================================================#
#==== 08 - Chart Parameters ===================================================#
#==============================================================================#

CHART_WIDTH  <- 3750L
CHART_HEIGHT <- 1833L

COL_BLUE   <- "#004890"
COL_GREY   <- "#708090"
COL_ORANGE <- "#F37021"
COL_RED    <- "#B22222"

FEATURE_MAP <- c(
  "r1"  = "Total Assets",          "r2"  = "Total Equity",
  "r3"  = "Total Liabilities",     "r4"  = "Total Debt",
  "r5"  = "Asset Coverage Ratio",  "r6"  = "Equity Ratio",
  "r7"  = "Debt Ratio",            "r8"  = "Net Debt Ratio",
  "r9"  = "Self-Financing Ratio",  "r10" = "Short-term Asset Structure",
  "r11" = "Inventory Intensity",   "r12" = "Cash to Current Assets",
  "r13" = "Cash to Total Assets",  "r14" = "Return on Assets (ROA)",
  "r15" = "Return on Equity (ROE)","r16" = "Return on Fixed Assets",
  "r17" = "Debt Service Coverage", "r18" = "Net Debt Service Coverage",
  "l1"  = "Latent Dim 1", "l2" = "Latent Dim 2",
  "l3"  = "Latent Dim 3", "l4" = "Latent Dim 4",
  "l5"  = "Latent Dim 5", "l6" = "Latent Dim 6",
  "l7"  = "Latent Dim 7", "l8" = "Latent Dim 8",
  "dae_l1" = "Robust Latent 1", "dae_l2" = "Robust Latent 2",
  "dae_l3" = "Robust Latent 3", "dae_l4" = "Robust Latent 4",
  "dae_l5" = "Robust Latent 5", "dae_l6" = "Robust Latent 6",
  "dae_l7" = "Robust Latent 7", "dae_l8" = "Robust Latent 8"
)


#==============================================================================#
#==== 09 - Directory Setup & Validation =======================================#
#==============================================================================#

stopifnot(
  "PATH_DATA_FILE not found — set PATH_DATA_RAW in config.R" =
    file.exists(PATH_DATA_FILE)
)

if (!SPLIT_MODE %in% c("OoS", "OoT"))
  stop("SPLIT_MODE must be 'OoS' or 'OoT'. Got: ", SPLIT_MODE)

if (!MODEL_GROUP %in% c("01", "02", "03", "04", "05"))
  stop("MODEL_GROUP must be one of '01'–'05'. Got: ", MODEL_GROUP)

for (.dir in c(PATH_DATA_OUT, PATH_CHARTS, DIR_FINAL_OUT)) {
  dir.create(.dir, recursive = TRUE, showWarnings = FALSE)
}
rm(.dir)

message(sprintf(
  "config.R loaded  |  root: %s  |  split: %s  |  data out: %s",
  PATH_ROOT, SPLIT_MODE, PATH_DATA_OUT
))