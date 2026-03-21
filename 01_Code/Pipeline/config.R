#==============================================================================#
#==== config.R ================================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   
# UPDATED:  2026
#
# PURPOSE:
#   Single source of truth for the entire pipeline.
#   All paths, seeds, flags, and model parameters live here.
#   No hardcoded values anywhere else in the pipeline.
#
#   Sourced first by 00_Master.R; can also be sourced independently
#   before running any stage in isolation.
#
#==============================================================================#


#==============================================================================#
#==== 00 - Python / Reticulate (must be set before any library() call) ========#
#==============================================================================#

Sys.setenv(RETICULATE_PYTHON = "C:/venvs/autotab_env/Scripts/python.exe")


#==============================================================================#
#==== 01 - Paths ==============================================================#
#==============================================================================#

## Root — resolved automatically via here::here(); no manual editing needed.
PATH_ROOT <- file.path(here::here(""))

## Input data.
PATH_DATA_RAW  <- "C:/Users/Tristan Leiter/Documents/Privat/ILAB/Data/WS2025"
PATH_DATA_FILE <- file.path(PATH_DATA_RAW, "data.rda")

## Pipeline outputs.
PATH_DATA_OUT   <- file.path(PATH_ROOT, "02_Data")
PATH_CHARTS     <- file.path(PATH_ROOT, "03_Charts")

## Chart sub-directories (created on demand in 09_Charts.R).
PATH_CHARTS_XGB <- file.path(PATH_CHARTS, "VAE", "FinancialRatios", "XGBoost")

## Subfunction directories.
PATH_FN_GENERAL <- file.path(PATH_ROOT, "01_Code", "Subfunctions")
PATH_FN_XGB     <- file.path(PATH_ROOT, "01_Code", "XGBoost_Subfunctions")


#==============================================================================#
#==== 02 - Reproducibility ====================================================#
#==============================================================================#

SEED <- 123L


#==============================================================================#
#==== 03 - Data Flags =========================================================#
#==============================================================================#

## Quantile-normalise continuous features before VAE input.
QUANTILE_TRANSFORM <- TRUE

## If TRUE, transform bounded [0,1] features alongside continuous ones.
## Set FALSE to leave rank/proportion features on their natural scale.
TRANSFORM_BOUNDED01 <- FALSE

## Minimum number of firms per size × sector × year cell.
## Cells below this threshold trigger a warning in 02_FeatureEngineering.R.
MIN_SS_FIRMS <- 5L

## Column name of the binary default target.
TARGET_COL <- "y"

## Stratification variables for train/test and CV splits.
STRAT_VARS    <- c("sector", "y_ever")
STRAT_VARS_CV <- c("sector", "y")

## Train proportion (remainder goes to test).
TRAIN_SIZE <- 0.7


#==============================================================================#
#==== 04 - Cross-Validation ===================================================#
#==============================================================================#

N_FOLDS <- 5L


#==============================================================================#
#==== 05 - VAE Configuration ==================================================#
#==============================================================================#

## Set to NULL to let 04_VAE_Prep.R derive dimensions automatically.
VAE_CONFIG <- list(
  encoder_dims     = NULL,   ## Derived in 05_VAE_Train.R
  decoder_dims     = NULL,   ## Derived in 05_VAE_Train.R
  latent_dim       = NULL,   ## Derived in 05_VAE_Train.R: max(4, floor(sqrt(n_features)))
  epochs           = 150L,
  batch_size       = 256L,
  beta             = 3.0,    ## KL weight (β-VAE)
  lr               = 1e-3,
  temperature      = 0.5,
  patience         = 10L,    ## Early-stopping patience (epochs)
  kl_warmup        = TRUE,
  kl_warmup_epochs = 20L
)

## Strategy flag: TRUE  → model sees only VAE-derived features + target.
##                FALSE → model sees original features + VAE-derived features.
USE_VAE_ONLY <- TRUE


#==============================================================================#
#==== 06 - XGBoost Configuration ==============================================#
#==============================================================================#

XGB_CONFIG <- list(
  ## Bayesian optimisation budget.
  n_init_points    = 10L,
  n_iter_bayes     = 20L,
  
  ## Early-stopping rounds (BO phase vs final-model phase).
  early_stop_bo    = 20L,
  early_stop_final = 50L,
  
  ## Maximum rounds searched during BO; final model trained to optimal.
  nrounds_bo       = 1000L,
  nrounds_final    = 2000L,
  
  ## Parallelism.
  nthread          = parallel::detectCores() - 1L,
  
  ## Optimisation metric (passed to xgb.cv / xgb.train).
  eval_metric      = "auc"
)


#==============================================================================#
#==== 07 - Chart Parameters ===================================================#
#==============================================================================#

CHART_WIDTH  <- 3750L
CHART_HEIGHT <- 1833L

## Brand palette.
COL_BLUE   <- "#004890"
COL_GREY   <- "#708090"
COL_ORANGE <- "#F37021"
COL_RED    <- "#B22222"

## Feature label map — used in all chart scripts.
FEATURE_MAP <- c(
  "r1"  = "Total Assets",
  "r2"  = "Total Equity",
  "r3"  = "Total Liabilities",
  "r4"  = "Total Debt",
  "r5"  = "Asset Coverage Ratio (I)",
  "r6"  = "Equity Ratio",
  "r7"  = "Debt Ratio",
  "r8"  = "Net Debt Ratio",
  "r9"  = "Self-Financing Ratio",
  "r10" = "Short-term Asset Structure",
  "r11" = "Inventory Intensity",
  "r12" = "Cash to Current Assets",
  "r13" = "Cash to Total Assets",
  "r14" = "Return on Assets (ROA)",
  "r15" = "Return on Equity (ROE)",
  "r16" = "Return on Fixed Assets",
  "r17" = "Debt Service Coverage",
  "r18" = "Net Debt Service Coverage",
  ## VAE latent dimensions.
  "l1" = "Latent Dim 1", "l2" = "Latent Dim 2",
  "l3" = "Latent Dim 3", "l4" = "Latent Dim 4",
  "l5" = "Latent Dim 5", "l6" = "Latent Dim 6",
  "l7" = "Latent Dim 7", "l8" = "Latent Dim 8",
  ## DAE latent dimensions.
  "dae_l1" = "Robust Latent 1", "dae_l2" = "Robust Latent 2",
  "dae_l3" = "Robust Latent 3", "dae_l4" = "Robust Latent 4",
  "dae_l5" = "Robust Latent 5", "dae_l6" = "Robust Latent 6",
  "dae_l7" = "Robust Latent 7", "dae_l8" = "Robust Latent 8"
)


#==============================================================================#
#==== 08 - Validation =========================================================#
#==============================================================================#

## Fail fast if any required path does not exist.
stopifnot(
  "PATH_DATA_FILE not found — set PATH_DATA_RAW in config.R" =
    file.exists(PATH_DATA_FILE)
)

message("config.R loaded — root: ", PATH_ROOT)