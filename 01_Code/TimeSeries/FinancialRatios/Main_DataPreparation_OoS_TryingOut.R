#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

## Last changed: 2026-02-04 | Tristan Leiter
## Data leakage might be an issue since the VAE is not trained separately for all folds
## but on all the training data.

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("dplyr", "caret", "lubridate", "purrr", "tidyr",
              "Matrix", "pROC",            ## Sparse Matrices and efficient AUC computation.
              "glmnet",                    ## GLM library.
              "xgboost",                   ## XGBoost library.
              "rBayesianOptimization",     ## Bayesian Optimization.
              "ggplot2", "Ckmeans.1d.dp",  ## Plotting & Charts | XG-Charts / Feature Importance.
              "scales",                    ## ggplot2 extension for nice charts.
              "ggrepel",                   ## Non-overlapping ggplot2 text labels.
              "ranger",                    ## Random Forest library.
              "mlr3", "mlr3learners", "mlr3tuning",
              "mlr3mbo", "mlr3measures", "data.table",
              "paradox", "future", 
              "future.apply", "parallel",
              "adabag",
              "purrr", "tibble",
              "autotab", "keras",           ## VAE.
              "reticulate", "tensorflow",
              "mclust",                     ## Finding soft interaction features.
              "pdp", "gridExtra", "hexbin", "openxlsx",
              "tensorflow"
)

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

#==== 1A - Python & Tensorflow dependencies ===================================#

# reticulate::install_python(version = "3.10")
# install_tensorflow(version = "2.10.0", envname = "r-reticulate")
# install_miniconda()

# Check if R can find the "real" python version
# py_versions <- py_discover_config()
# print(py_versions)

# use_python("C:/Users/TristanLeiter/OneDrive - Hinder Asset Management AG/Dokumente/.virtualenvs/r-reticulate/Scripts/python.exe", required = TRUE)

#==== 1B - Functions ==========================================================#

sourceFunctions <- function (functionDirectory)  {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R", 
                              full.names = T)
  ssource <- function(path) {
    try(source(path, echo = F, verbose = F, print.eval = T, 
               local = F))
  }
  sapply(functionFiles, ssource)
}

#==== 1C - Parameters =========================================================#

## Directories.
Data_Path <- "C:/Users/Tristan Leiter/Documents/Privat/ILAB/Data/WS2025" ## Needs to be set manually.
Data_Directory <- file.path(Data_Path, "data.rda")
Data_Directory_write <- file.path(Path, "02_Data")
Charts_Directory <- file.path(Path, "03_Charts")
Data_RF <- Path

## Charts Directories.
Charts_GLM_Directory <- file.path(Charts_Directory, "GLM")
Charts_RF_Directory <- file.path(Charts_Directory, "RF")
Charts_XGBoost_Directory <- file.path(Charts_Directory, "XGBoost")
Charts_TestSet_Directory <- file.path(Charts_Directory, "TestSet")

Functions_Directory <- file.path(Path, "01_Code/Subfunctions")
Functions_RF_Directory <- file.path(Path, "01_Code/RF_Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)
sourceFunctions(Functions_RF_Directory)

## Data Sampling.
set.seed(123)              ## Check seed.

## Charts.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"


width <- 3750
heigth <- 1833

## General Parameters for modeling.
N_folds <- 5

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02A - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

#==== 02B - Pre-process checks ================================================#

## Apply the simple methodology to exclude some observations (outlined by the Leitner Notes).
## Remove all ratios.

Data <- DataPreprocessing(Data, Tolerance = 2)

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .))) %>%
  drop_na()

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

MVstratifiedsampling <- function(Data,
                                 strat_vars = c("sector", "y_ever"),
                                 Train_size = 0.7) {
  
  ## ── 1. Firm-Level Profile ─────────────────────────────────────────────────
  ## Aggregate panel to one row per firm.
  ## Use y_ever (not y) to avoid dplyr naming collision with summarise().
  
  firm_profile <- Data %>%
    group_by(id) %>%
    summarise(
      y_ever = as.integer(any(y == 1)),
      sector = first(sector),
      size   = first(size),
      .groups = "drop"
    )
  
  ## ── 2. Build Stratification Key ───────────────────────────────────────────
  ## Use across() instead of select(.) inside mutate() — 
  ## the latter passes the full tibble to interaction() and breaks.
  
  firm_profile <- firm_profile %>%
    mutate(
      Strat_Key = interaction(
        across(all_of(strat_vars)),
        drop = TRUE
      )
    )
  
  ## ── 3. Guard Against Singleton Strata ─────────────────────────────────────
  ## createDataPartition fails on strata with <2 members because it cannot
  ## allocate at least one observation to both Train and Test.
  ## Drop these firms and warn — they will be excluded from both splits.
  
  strat_counts <- table(firm_profile$Strat_Key)
  small_strata <- names(strat_counts[strat_counts < 2])
  
  if (length(small_strata) > 0) {
    n_dropped <- sum(firm_profile$Strat_Key %in% small_strata)
    warning(paste0(
      "Dropping ", length(small_strata), " strata with <2 firms (",
      n_dropped, " firms total): ",
      paste(small_strata, collapse = ", ")
    ))
    firm_profile <- firm_profile %>%
      filter(!Strat_Key %in% small_strata) %>%
      mutate(Strat_Key = droplevels(Strat_Key))
  }
  
  ## ── 4. Stratified Split ───────────────────────────────────────────────────
  set.seed(123)
  train_index <- createDataPartition(
    y     = firm_profile$Strat_Key,
    p     = Train_size,
    list  = FALSE,
    times = 1
  )
  
  train_ids <- firm_profile$id[train_index]
  test_ids  <- firm_profile$id[-train_index]
  
  ## ── 5. Return Splits of the Original Panel Data ───────────────────────────
  Train <- Data %>% filter(id %in% train_ids)
  Test  <- Data %>% filter(id %in% test_ids)
  
  return(list(Train = Train, Test = Test))
}

Quantile_Transform <- TRUE

#==== 03A - Incorporate time-series dynamics ==================================#

tryCatch({
  
  message("--- Starting 03A: Time-Series Dynamics ---")
  
  ## ── A. Setup ───────────────────────────────────────────────────────────────
  Data_Eng <- as.data.table(Data)
  setorder(Data_Eng, id, refdate)
  
  ## Define ratio groups by economic meaning
  ## Levels (absolute balance sheet aggregates) - r1 to r4
  level_ratios <- paste0("r", 1:4)
  
  ## Rate/margin ratios - r5 to r18
  rate_ratios  <- paste0("r", 5:18)
  
  ## All ratios combined
  all_ratios   <- paste0("r", 1:18)
  
  ## Ratios where "higher is better" — peak drop is meaningful
  ## r6  = profitability, r9 = liquidity, r10 = coverage,
  ## r14 = solvency, r15 = another coverage, r16/r17/r18 = return metrics
  peak_ratios  <- c("r6", "r9", "r10", "r14", "r15", "r16", "r17", "r18")
  
  ## Ratios where "lower is better" — trough rise is meaningful  
  ## r7/r8 = leverage (debt ratios, lower = safer)
  trough_ratios <- c("r7", "r8")
  
  ## Ratios suitable for consecutive decline tracking (profitability/solvency)
  consec_ratios <- c("r6", "r9", "r10", "r13", "r14", "r16", "r17", "r18")
  
  ## ── B. Base Tracking ───────────────────────────────────────────────────────
  Data_Eng[, time_index     := seq_len(.N), by = id]
  Data_Eng[, is_mature      := fifelse(time_index >= 3, 1L, 0L)]
  Data_Eng[, history_length := .N, by = id]
  Data_Eng[, has_history    := fifelse(time_index > 1, 1L, 0L)]
  
  ## ── C. Year-over-Year Changes (1st Differences) ───────────────────────────
  ## Trajectory of change is a leading default indicator even when levels
  ## are not yet distressed. Sudden deterioration in profitability or
  ## liquidity ratios precedes default by 1–3 years in empirical literature.
  
  message("  Computing YoY changes...")
  
  Data_Eng[, paste0("yoy_", all_ratios) :=
             lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
           by = id, .SDcols = all_ratios]
  
  ## ── D. Acceleration (2nd Differences) ─────────────────────────────────────
  ## Rate of change of change — is deterioration speeding up?
  ## Firms with accelerating ratio decline are materially higher risk
  ## than firms with a flat but negative trajectory.
  
  message("  Computing acceleration (2nd differences)...")
  
  yoy_cols <- paste0("yoy_", all_ratios)
  
  Data_Eng[, paste0("accel_", all_ratios) :=
             lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
           by = id, .SDcols = yoy_cols]
  
  ## ── E. Expanding Mean ─────────────────────────────────────────────────────
  ## Firm's own historical baseline. Used directly and as denominator
  ## for deviation features below. Controls for cross-sectional heterogeneity
  ## — a capital-heavy firm should be benchmarked against itself, not peers
  ## (peer benchmarking is handled separately in 03B sector Z-scores).
  
  message("  Computing expanding means...")
  
  Data_Eng[, paste0("expmean_", all_ratios) :=
             lapply(.SD, cummean),
           by = id, .SDcols = all_ratios]
  
  ## ── F. Deviation from Expanding Mean ──────────────────────────────────────
  ## "Is this firm currently underperforming its own historical norm?"
  ## This is the feature you originally intended: (current - own mean).
  ## Negative values = below own baseline = firm-specific deterioration signal.
  ## Normalised version guards against scale differences across ratio types.
  
  message("  Computing deviations from expanding mean...")
  
  expmean_cols <- paste0("expmean_", all_ratios)
  
  for (i in seq_along(all_ratios)) {
    raw_col  <- all_ratios[i]
    mean_col <- expmean_cols[i]
    out_col  <- paste0("dev_expmean_", raw_col)
    
    Data_Eng[, (out_col) := get(raw_col) - get(mean_col)]
  }
  
  ## ── G. Expanding Volatility ────────────────────────────────────────────────
  ## Ratio instability = uncertain cash flows = higher credit risk.
  ## Analogous to asset volatility in Merton's structural model.
  ## Vectorised expanding SD via cumulative variance formula — avoids sapply.
  ## NA on first observation is structural (can't compute SD from 1 point).
  
  message("  Computing expanding volatility...")
  
  Data_Eng[, paste0("expvol_", all_ratios) :=
             lapply(.SD, function(x) {
               n      <- seq_along(x)
               mu     <- cummean(x)
               expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
               fifelse(n < 2L, NA_real_, sqrt(pmax(0, expvar)))
             }),
           by = id, .SDcols = all_ratios]
  
  ## ── H. Peak Deterioration ("Higher is Better" Ratios) ─────────────────────
  ## Distance from historical best — firms rarely default at their peak,
  ## they default after sustained decline from it.
  ## Applied only to ratios where higher values indicate health.
  ## NA on first observation (no prior peak to compare against).
  
  message("  Computing peak deterioration...")
  
  Data_Eng[, paste0("peak_drop_", peak_ratios) :=
             lapply(.SD, function(x) x - cummax(x)),
           by = id, .SDcols = peak_ratios]
  
  Data_Eng[time_index == 1L,
           paste0("peak_drop_", peak_ratios) := lapply(peak_ratios, function(x) NA_real_)]
  
  ## ── I. Trough Rise ("Lower is Better" Ratios) ─────────────────────────────
  ## Mirror of peak_drop for leverage/debt ratios.
  ## Rising from historical low = leverage is increasing = deterioration signal.
  
  message("  Computing trough rise (leverage ratios)...")
  
  Data_Eng[, paste0("trough_rise_", trough_ratios) :=
             lapply(.SD, function(x) x - cummin(x)),
           by = id, .SDcols = trough_ratios]
  
  Data_Eng[time_index == 1L,
           paste0("trough_rise_", trough_ratios) := lapply(trough_ratios, function(x) NA_real_)]
  
  ## ── J. Momentum (Recent vs Long-Term Mean) ─────────────────────────────────
  ## Short-run vs long-run performance divergence.
  ## Positive = recent 2Y better than full history = recovery momentum.
  ## Negative = recent 2Y worse than full history = deterioration momentum.
  ## Meaningful from time_index >= 2 (need at least 2 points for frollmean).
  
  message("  Computing momentum (2Y vs expanding mean)...")
  
  Data_Eng[, paste0("momentum_", all_ratios) :=
             lapply(.SD, function(x) {
               recent    <- frollmean(x, n = 2L, align = "right", fill = NA)
               long_term <- cummean(x)
               recent - long_term
             }),
           by = id, .SDcols = all_ratios]
  
  ## ── K. Consecutive Decline Counter ────────────────────────────────────────
  ## Non-linear distress signal: 3 consecutive years of declining profitability
  ## is qualitatively different from an erratic pattern with the same mean YoY.
  ## Applied to profitability, liquidity, and solvency ratios only —
  ## consecutive changes in leverage/balance sheet levels are less meaningful.
  
  message("  Computing consecutive decline counters...")
  
  for (col in consec_ratios) {
    yoy_col <- paste0("yoy_", col)
    out_col <- paste0("consec_decline_", col)
    
    Data_Eng[, (out_col) := {
      yoy     <- get(yoy_col)
      counter <- integer(.N)
      for (i in seq_len(.N)) {
        if (i == 1L || is.na(yoy[i])) {
          counter[i] <- 0L
        } else if (yoy[i] < 0) {
          counter[i] <- counter[i - 1L] + 1L
        } else {
          counter[i] <- 0L
        }
      }
      counter
    }, by = id]
  }
  
  ## ── L. Sanity Check ────────────────────────────────────────────────────────
  n_new_features <- ncol(Data_Eng) - ncol(Data)
  
  message(paste0("03A Complete."))
  message(paste0("  Original columns : ", ncol(Data)))
  message(paste0("  New TS features  : ", n_new_features))
  message(paste0("  Total columns    : ", ncol(Data_Eng)))
  message(paste0("  Rows             : ", nrow(Data_Eng)))
  message(paste0("  NAs introduced   : ", sum(is.na(Data_Eng)) - sum(is.na(Data))))
  
  ## Quick NA audit per feature family
  na_audit <- data.frame(
    family  = c("yoy", "accel", "expmean", "dev_expmean", "expvol", 
                "peak_drop", "trough_rise", "momentum", "consec_decline"),
    na_count = c(
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^yoy_",          names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^accel_",        names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^expmean_",      names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^dev_expmean_",  names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^expvol_",       names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^peak_drop_",    names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^trough_rise_",  names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^momentum_",     names(Data_Eng), value=TRUE)])),
      sum(is.na(Data_Eng[, .SD, .SDcols = grep("^consec_decline_", names(Data_Eng), value=TRUE)]))
    )
  )
  print(na_audit)
  
}, error = function(e) stop("03A Failed: ", e$message))

#==== 03B - Data Sampling Out-of-Sample =======================================#

tryCatch({
  
  message("--- Starting 03B: Train/Test Split ---")
  set.seed(123)
  
  ## ── A. Pre-Split Validation ───────────────────────────────────────────────
  ## Confirm no Test-dependent features exist yet.
  ## At this point Data_Eng should only contain:
  ## raw ratios, time-series features from 03A, and metadata.
  ## Sector features (secZ_, secRank_, etc.) must NOT exist yet.
  
  forbidden_prefixes <- c("secZ_", "secRank_", "secDev_", "secVol_",
                          "secTrend_", "secDiverg_", "secSizeZ_")
  premature_cols <- names(Data_Eng)[
    sapply(names(Data_Eng), function(n)
      any(startsWith(n, forbidden_prefixes)))
  ]
  
  if (length(premature_cols) > 0) {
    stop(paste(
      "CRITICAL: Sector features found before split — leakage risk.",
      "The following columns should not exist yet:",
      paste(premature_cols, collapse = ", ")
    ))
  }
  
  message("  Pre-split validation passed — no sector features present.")
  
  ## ── B. Class & Sector Distribution Before Split ───────────────────────────
  cat(sprintf("  Full dataset     : %d rows x %d cols\n",
              nrow(Data_Eng), ncol(Data_Eng)))
  cat(sprintf("  Overall default rate: %.4f%% (%d defaults)\n",
              100 * mean(Data_Eng$y), sum(Data_Eng$y)))
  
  ## Sector-level default rates — useful diagnostic
  sector_dist <- Data_Eng[, .(
    n         = .N,
    n_default = sum(y),
    rate      = round(mean(y) * 100, 3)
  ), by = sector][order(-rate)]
  
  message("  Sector default rates:")
  print(sector_dist)
  
  ## ── C. ID-Based Stratified Split ──────────────────────────────────────────
  ## CRITICAL: Split must be by FIRM ID, not by row.
  ## A single firm's observations must be entirely in Train OR Test —
  ## never split across both. Splitting by row would cause:
  ## (1) time-series feature leakage (a firm's t=3 in Test informed by t=1,2 in Train)
  ## (2) data snooping in expanding features (expmean, expvol computed on full history)
  ##
  ## Stratification: by sector AND default status — ensures both Train and Test
  ## preserve the marginal distributions of sector and y simultaneously.
  ##
  ## NOTE: MVstratifiedsampling receives the FULL panel (Data_Eng) and
  ## handles firm-level aggregation internally. Do NOT pre-aggregate.
  
  if (exists("MVstratifiedsampling")) {
    
    message("  Using MVstratifiedsampling for firm-level stratified split...")
    
    ## Quick diagnostics before split
    firm_level_diag <- Data_Eng[, .(
      sector = sector[1],
      y_ever = as.integer(any(y == 1))
    ), by = id]
    
    message(paste("  Unique firms:", nrow(firm_level_diag)))
    message(paste("  Ever-defaulted firms:", sum(firm_level_diag$y_ever)))
    
    ## Pass the FULL panel — function aggregates internally
    Data_Sampled <- MVstratifiedsampling(
      Data       = as.data.frame(Data_Eng),
      strat_vars = c("sector", "y_ever"),
      Train_size = 0.7
    )
    
    Train <- as.data.table(Data_Sampled[["Train"]])
    Test  <- as.data.table(Data_Sampled[["Test"]])
    
    train_ids <- unique(Train$id)
    test_ids  <- unique(Test$id)
    
    ## Use diagnostic frame for downstream reporting
    firm_level <- firm_level_diag
    
  } else {
    
    warning("MVstratifiedsampling not found. Using manual firm-level stratified split.")
    
    ## Manual firm-level stratified split
    ## Stratify by sector x default_ever combinations
    firm_level <- Data_Eng[, .(
      sector = sector[1],
      y_ever = as.integer(any(y == 1))
    ), by = id]
    
    firm_level[, strat_key := paste(sector, y_ever, sep = "_")]
    
    set.seed(123)
    train_ids <- firm_level[, {
      n_train <- max(1L, round(.N * 0.7))
      idx     <- sample(.N, n_train)
      .(id = id[idx])
    }, by = strat_key]$id
    
    test_ids <- firm_level[!id %in% train_ids]$id
    
    ## Apply ID split to full panel
    Train <- Data_Eng[id %in% train_ids]
    Test  <- Data_Eng[id %in% test_ids]
  }
  
  ## ── E. Split Validation ───────────────────────────────────────────────────
  
  ## 1. No ID overlap
  id_overlap <- intersect(Train$id, Test$id)
  if (length(id_overlap) > 0) {
    stop(paste("CRITICAL: ID overlap between Train and Test —",
               length(id_overlap), "firms appear in both sets."))
  }
  
  ## 2. Row conservation
  if (nrow(Train) + nrow(Test) != nrow(Data_Eng)) {
    ## Some firms may have been dropped due to singleton strata —
    ## warn instead of hard-failing
    n_lost <- nrow(Data_Eng) - (nrow(Train) + nrow(Test))
    if (n_lost > 0 && n_lost < nrow(Data_Eng) * 0.01) {
      warning(paste("Minor row loss after split:", n_lost, "rows (",
                    round(100 * n_lost / nrow(Data_Eng), 3),
                    "%) — likely from singleton strata removal."))
    } else {
      stop(paste("CRITICAL: Row count mismatch after split.",
                 "Expected:", nrow(Data_Eng),
                 "Got:", nrow(Train) + nrow(Test),
                 "Lost:", n_lost))
    }
  }
  
  ## 3. Sector coverage — Test must not contain sectors absent from Train
  train_sectors <- unique(Train$sector)
  test_sectors  <- unique(Test$sector)
  unseen_sectors <- setdiff(test_sectors, train_sectors)
  
  if (length(unseen_sectors) > 0) {
    warning(paste("Test contains sectors not in Train:",
                  paste(unseen_sectors, collapse = ", "),
                  "— sector features will be NA for these."))
  }
  
  ## 4. Default rate preservation
  cat(sprintf("\n  %-12s | %8s | %8s | %12s | %10s\n",
              "Set", "Rows", "Firms", "Default Rate", "Defaults"))
  cat(sprintf("  %-12s | %8d | %8d | %11.4f%% | %10d\n",
              "Train", nrow(Train), length(train_ids),
              100 * mean(Train$y), sum(Train$y)))
  cat(sprintf("  %-12s | %8d | %8d | %11.4f%% | %10d\n",
              "Test", nrow(Test), length(test_ids),
              100 * mean(Test$y), sum(Test$y)))
  cat(sprintf("  %-12s | %8d | %8d | %11.4f%% | %10d\n",
              "Full", nrow(Data_Eng), nrow(firm_level),
              100 * mean(Data_Eng$y), sum(Data_Eng$y)))
  
  ## 5. Sector x default rate breakdown per split
  train_sector_dist <- Train[, .(
    n = .N, rate = round(mean(y) * 100, 3)
  ), by = sector][order(sector)]
  
  test_sector_dist <- Test[, .(
    n = .N, rate = round(mean(y) * 100, 3)
  ), by = sector][order(sector)]
  
  sector_check <- merge(
    train_sector_dist, test_sector_dist,
    by = "sector", suffixes = c("_train", "_test")
  )
  sector_check[, rate_delta := round(rate_train - rate_test, 3)]
  
  message("\n  Sector distribution check (default rate delta = Train - Test):")
  print(sector_check)
  
  ## Warn if any sector has a large default rate discrepancy
  large_delta <- sector_check[abs(rate_delta) > 2]
  if (nrow(large_delta) > 0) {
    warning(paste("  Large default rate discrepancy (>2pp) in sectors:",
                  paste(large_delta$sector, collapse = ", "),
                  "— consider increasing stratification granularity."))
  }
  
  message("03B Complete. Train and Test sets validated.")
  
}, error = function(e) stop("03B Failed: ", e$message))

#==== 03C - Sector Specific Deviation Features ================================#

tryCatch({
  
  message("--- Starting 03C: Sector Deviation Features (Leakage-Free) ---")
  
  ## ── A. Setup ───────────────────────────────────────────────────────────────
  if (!"year" %in% names(Train)) Train[, year := year(refdate)]
  if (!"year" %in% names(Test))  Test[,  year := year(refdate)]
  
  sector_ratios      <- paste0("r", 5:18)
  size_sector_ratios <- c("r5", "r6", "r7", "r9", "r10", "r14")
  sector_levels_ref  <- c("real estate", "wholesale", "manufacture",
                          "construction", "retail", "energy")  ## "service" = reference
  
  ## Store initial row counts for validation at the end
  n_train_init <- nrow(Train)
  n_test_init  <- nrow(Test)
  
  ## ── B. Fit Sector Statistics on TRAIN Only ────────────────────────────────
  ## ALL sector statistics are derived exclusively from Train.
  ## Test observations are mapped onto these Train-derived parameters.
  
  message("  Fitting sector-year statistics on Train only...")
  
  sector_stats_train <- Train[,
                              c(
                                lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                       function(col) mean(get(col), na.rm = TRUE)),
                                lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                       function(col) sd(get(col),   na.rm = TRUE)),
                                lapply(setNames(sector_ratios, paste0("med_",  sector_ratios)),
                                       function(col) median(get(col), na.rm = TRUE)),
                                lapply(setNames(sector_ratios, paste0("p25_",  sector_ratios)),
                                       function(col) quantile(get(col), 0.25, na.rm = TRUE)),
                                lapply(setNames(sector_ratios, paste0("p75_",  sector_ratios)),
                                       function(col) quantile(get(col), 0.75, na.rm = TRUE)),
                                list(n_firms_train = .N)
                              ),
                              by = .(sector, year)
  ]
  
  cat(sprintf("  Sector-year combinations in Train: %d\n", nrow(sector_stats_train)))
  
  ## ── C. Fallback Statistics for Unseen Sector-Years ────────────────────────
  ## Sector-level fallback (year-agnostic) — fit on Train
  sector_stats_fallback <- Train[,
                                 c(
                                   lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                          function(col) mean(get(col), na.rm = TRUE)),
                                   lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                          function(col) sd(get(col),   na.rm = TRUE))
                                 ),
                                 by = sector
  ]
  
  ## Global fallback — fit on Train
  global_stats_train <- Train[,
                              c(
                                lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                       function(col) mean(get(col), na.rm = TRUE)),
                                lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                       function(col) sd(get(col),   na.rm = TRUE))
                              )
  ]
  
  message("  Sector stats fitted on Train. Applying to Train and Test...")
  
  ## ── D. Join Train-Fitted Stats to Both Splits ─────────────────────────────
  ## FIX: Use on= indexed join with := instead of merge() to prevent
  ## cartesian row multiplication from non-unique keys.
  
  message("  Joining sector stats to Train and Test...")
  
  sector_stats_train <- unique(sector_stats_train, by = c("sector", "year"))
  setkeyv(sector_stats_train, c("sector", "year"))
  
  stat_join_cols <- setdiff(names(sector_stats_train), c("sector", "year"))
  
  for (sc in stat_join_cols) {
    Train[sector_stats_train, (sc) := get(paste0("i.", sc)),
          on = .(sector, year)]
  }
  for (sc in stat_join_cols) {
    Test[sector_stats_train, (sc) := get(paste0("i.", sc)),
         on = .(sector, year)]
  }
  
  ## Fallback for Test sector-years not in Train
  test_missing_mask <- is.na(Test[[paste0("mean_", sector_ratios[1])]])
  n_missing <- sum(test_missing_mask)
  
  if (n_missing > 0) {
    warning(paste(n_missing, "Test rows have sector-year combos absent from Train.",
                  "Applying sector-level fallback stats."))
    
    setkeyv(sector_stats_fallback, "sector")
    
    for (col in sector_ratios) {
      mean_col <- paste0("mean_", col)
      sd_col   <- paste0("sd_",   col)
      
      fb_means <- sector_stats_fallback[Test[test_missing_mask], get(mean_col),
                                        on = .(sector)]
      fb_sds   <- sector_stats_fallback[Test[test_missing_mask], get(sd_col),
                                        on = .(sector)]
      
      Test[test_missing_mask, (mean_col) := fb_means]
      Test[test_missing_mask, (sd_col)   := fb_sds]
    }
    
    ## Any still-missing: apply global Train stats
    still_missing <- is.na(Test[[paste0("mean_", sector_ratios[1])]])
    if (any(still_missing)) {
      warning(paste(sum(still_missing),
                    "Test rows still missing stats after sector fallback.",
                    "Applying global Train means."))
      for (col in sector_ratios) {
        mean_col <- paste0("mean_", col)
        sd_col   <- paste0("sd_",   col)
        Test[still_missing, (mean_col) := global_stats_train[[mean_col]]]
        Test[still_missing, (sd_col)   := global_stats_train[[sd_col]]]
      }
    }
  }
  
  ## ── E. Sector Z-Score ─────────────────────────────────────────────────────
  ## (firm - Train sector mean) / Train sector SD
  
  message("  Computing sector Z-scores...")
  
  for (col in sector_ratios) {
    mean_col <- paste0("mean_", col)
    sd_col   <- paste0("sd_",   col)
    out_col  <- paste0("secZ_", col)
    
    Train[, (out_col) := {
      z <- (get(col) - get(mean_col)) / get(sd_col)
      fifelse(is.na(z) | is.infinite(z), 0, z)
    }]
    Test[, (out_col) := {
      z <- (get(col) - get(mean_col)) / get(sd_col)
      fifelse(is.na(z) | is.infinite(z), 0, z)
    }]
  }
  
  ## ── F. Raw Sector Deviation ───────────────────────────────────────────────
  message("  Computing raw sector deviations...")
  
  for (col in sector_ratios) {
    mean_col <- paste0("mean_", col)
    out_col  <- paste0("secDev_", col)
    
    Train[, (out_col) := get(col) - get(mean_col)]
    Test[,  (out_col) := get(col) - get(mean_col)]
  }
  
  ## ── G. Sector Percentile Rank — Leakage-Free via Train ECDF ──────────────
  ## FIX: Use direct := assignment instead of merge-back.
  ## The original merge on (sector, year, ratio_value) caused cartesian
  ## explosion when multiple firms share the same ratio value.
  
  message("  Computing sector percentile ranks (Train ECDF applied to Test)...")
  
  for (col in sector_ratios) {
    out_col <- paste0("secRank_", col)
    
    ## Train: standard within-group rank
    Train[, (out_col) :=
            frank(get(col), ties.method = "average", na.last = "keep") / .N,
          by = .(sector, year)]
    
    ## Test: build Train ECDFs per sector-year, then apply directly
    train_ecdf_list <- Train[,
                             .(ecdf_fn = list(ecdf(get(col)[!is.na(get(col))]))),
                             by = .(sector, year)
    ]
    
    ## Initialise column
    Test[, (out_col) := NA_real_]
    
    ## Apply ECDF per sector-year group — no merge, no row duplication
    for (i in seq_len(nrow(train_ecdf_list))) {
      s   <- train_ecdf_list$sector[i]
      yr  <- train_ecdf_list$year[i]
      fn  <- train_ecdf_list$ecdf_fn[[i]]
      
      mask <- Test$sector == s & Test$year == yr & !is.na(Test[[col]])
      if (any(mask)) {
        Test[mask, (out_col) := fn(get(col))]
      }
    }
    
    ## Fallback for unseen sector-year
    Test[is.na(get(out_col)), (out_col) := 0.5]
  }
  
  ## ── H. Sector Volatility as Feature ───────────────────────────────────────
  ## Sector SD from Train stats — already joined in step D
  
  message("  Extracting sector volatility features...")
  
  for (col in sector_ratios) {
    sd_col  <- paste0("sd_",    col)
    out_col <- paste0("secVol_", col)
    
    Train[, (out_col) := get(sd_col)]
    Test[,  (out_col) := get(sd_col)]
  }
  
  ## ── I. Sector Trend ────────────────────────────────────────────────────────
  ## YoY change in sector mean — computed from Train sector_stats only.
  ## FIX: Use indexed join with := instead of merge().
  
  message("  Computing sector trend from Train statistics...")
  
  sector_mean_cols <- paste0("mean_", sector_ratios)
  
  setorder(sector_stats_train, sector, year)
  
  sector_trend_train <- sector_stats_train[, c(
    list(sector = sector, year = year),
    lapply(
      setNames(sector_mean_cols, paste0("secTrend_", sector_ratios)),
      function(col_name) get(col_name) - shift(get(col_name), n = 1L, type = "lag")
    )
  ), by = sector]
  
  trend_cols <- paste0("secTrend_", sector_ratios)
  
  sector_trend_train <- unique(sector_trend_train, by = c("sector", "year"))
  setkeyv(sector_trend_train, c("sector", "year"))
  
  for (tc in trend_cols) {
    if (!tc %in% names(Train)) Train[, (tc) := NA_real_]
    if (!tc %in% names(Test))  Test[,  (tc) := NA_real_]
    
    Train[sector_trend_train, (tc) := get(paste0("i.", tc)),
          on = .(sector, year)]
    Test[sector_trend_train,  (tc) := get(paste0("i.", tc)),
         on = .(sector, year)]
  }
  
  ## ── J. Firm vs Sector Divergence ──────────────────────────────────────────
  ## Firm's own YoY change minus sector YoY trend.
  
  message("  Computing firm vs sector trajectory divergence...")
  
  for (col in sector_ratios) {
    yoy_col   <- paste0("yoy_",       col)
    trend_col <- paste0("secTrend_",  col)
    out_col   <- paste0("secDiverg_", col)
    
    if (yoy_col %in% names(Train) && trend_col %in% names(Train)) {
      Train[, (out_col) := get(yoy_col) - get(trend_col)]
      Test[,  (out_col) := get(yoy_col) - get(trend_col)]
    } else {
      warning(paste("  Skipping secDiverg for", col,
                    "— yoy_ or secTrend_ column missing. Ensure 03A ran first."))
    }
  }
  
  ## ── K. Size x Sector Z-Score ──────────────────────────────────────────────
  ## FIX: Use indexed join with := instead of merge().
  
  message("  Computing size x sector Z-scores...")
  
  size_sector_stats <- Train[,
                             c(
                               lapply(setNames(size_sector_ratios, paste0("ss_mean_", size_sector_ratios)),
                                      function(col) mean(get(col), na.rm = TRUE)),
                               lapply(setNames(size_sector_ratios, paste0("ss_sd_",   size_sector_ratios)),
                                      function(col) sd(get(col),   na.rm = TRUE))
                             ),
                             by = .(sector, size, year)
  ]
  
  size_sector_stats <- unique(size_sector_stats, by = c("sector", "size", "year"))
  setkeyv(size_sector_stats, c("sector", "size", "year"))
  
  ss_join_cols <- setdiff(names(size_sector_stats), c("sector", "size", "year"))
  
  for (sc in ss_join_cols) {
    Train[size_sector_stats, (sc) := get(paste0("i.", sc)),
          on = .(sector, size, year)]
    Test[size_sector_stats,  (sc) := get(paste0("i.", sc)),
         on = .(sector, size, year)]
  }
  
  for (col in size_sector_ratios) {
    ss_mean <- paste0("ss_mean_", col)
    ss_sd   <- paste0("ss_sd_",   col)
    out_col <- paste0("secSizeZ_", col)
    
    Train[, (out_col) := {
      z <- (get(col) - get(ss_mean)) / get(ss_sd)
      fifelse(is.na(z) | is.infinite(z), 0, z)
    }]
    Test[, (out_col) := {
      z <- (get(col) - get(ss_mean)) / get(ss_sd)
      fifelse(is.na(z) | is.infinite(z), 0, z)
    }]
  }
  
  ## Drop intermediate size-sector stat columns
  ss_stat_cols <- c(paste0("ss_mean_", size_sector_ratios),
                    paste0("ss_sd_",   size_sector_ratios))
  Train[, (intersect(ss_stat_cols, names(Train))) := NULL]
  Test[,  (intersect(ss_stat_cols, names(Test)))  := NULL]
  
  ## ── L. Sector Dummy Encoding ───────────────────────────────────────────────
  message("  Creating sector dummy variables...")
  
  for (s in sector_levels_ref) {
    out_col <- paste0("sector_", gsub(" ", "_", s))
    Train[, (out_col) := fifelse(sector == s, 1L, 0L)]
    Test[,  (out_col) := fifelse(sector == s, 1L, 0L)]
  }
  
  ## ── M. Cleanup — Remove Intermediate Stat Columns ─────────────────────────
  stat_cols_to_drop <- c(
    paste0("mean_", sector_ratios),
    paste0("sd_",   sector_ratios),
    paste0("med_",  sector_ratios),
    paste0("p25_",  sector_ratios),
    paste0("p75_",  sector_ratios),
    "n_firms_train"
  )
  
  drop_train <- intersect(stat_cols_to_drop, names(Train))
  drop_test  <- intersect(stat_cols_to_drop, names(Test))
  if (length(drop_train) > 0) Train[, (drop_train) := NULL]
  if (length(drop_test)  > 0) Test[,  (drop_test)  := NULL]
  
  ## Remove raw balance sheet columns (f1-f18)
  f_cols <- paste0("f", 1:18)
  drop_train_f <- intersect(f_cols, names(Train))
  drop_test_f  <- intersect(f_cols, names(Test))
  if (length(drop_train_f) > 0) Train[, (drop_train_f) := NULL]
  if (length(drop_test_f)  > 0) Test[,  (drop_test_f)  := NULL]
  
  ## ── N. Sanity Check ────────────────────────────────────────────────────────
  ## CRITICAL: Verify row counts haven't changed
  stopifnot(
    "Train row count changed during 03C!" = nrow(Train) == n_train_init,
    "Test row count changed during 03C!"  = nrow(Test)  == n_test_init
  )
  
  families <- c("secZ_", "secDev_", "secRank_", "secVol_",
                "secTrend_", "secDiverg_", "secSizeZ_", "sector_")
  
  na_audit_C <- data.frame(
    family   = families,
    n_cols   = sapply(families, function(f)
      sum(grepl(paste0("^", f), names(Train)))),
    na_train = sapply(families, function(f) {
      cols <- grep(paste0("^", f), names(Train), value = TRUE)
      if (length(cols) == 0L) return(0L)
      sum(is.na(Train[, .SD, .SDcols = cols]))
    }),
    na_test  = sapply(families, function(f) {
      cols <- grep(paste0("^", f), names(Test), value = TRUE)
      if (length(cols) == 0L) return(0L)
      sum(is.na(Test[, .SD, .SDcols = cols]))
    })
  )
  
  message("03C Complete.")
  cat(sprintf("  Train: %d rows x %d cols\n", nrow(Train), ncol(Train)))
  cat(sprintf("  Test : %d rows x %d cols\n", nrow(Test),  ncol(Test)))
  message("  NA audit by feature family:")
  print(na_audit_C)
  
}, error = function(e) stop("03C Failed: ", e$message))

#==== 03D - Quantile Transformation ===========================================#

tryCatch({
  
  message("--- Starting 03D: Quantile Transformation ---")
  
  if (exists("Quantile_Transform") && Quantile_Transform) {
    
    ## ── A. Define Exclusion Groups ─────────────────────────────────────────
    ## Explicit by-family exclusions — maintainable as features grow.
    ## Each group has a documented reason.
    
    ## 1. Metadata & identifiers — not model inputs
    meta_cols <- c("y", "id", "refdate", "sector", "size", "year")
    
    ## 2. Binary flags — transformation is meaningless on {0, 1}
    binary_cols <- c(
      "groupmember", "public",
      "is_mature", "has_history",
      grep("^sector_", names(Train), value = TRUE)   ## Sector dummies from 03B
    )
    
    ## 3. Ordinal / count features — meaningful integer scale
    ordinal_cols <- c(
      "time_index", "history_length",
      grep("^consec_decline_", names(Train), value = TRUE)
    )
    
    ## 4. Already-standardised features — retransforming destroys the scale
    already_scaled_cols <- c(
      grep("^secZ_",     names(Train), value = TRUE),   ## Sector Z-scores
      grep("^secSizeZ_", names(Train), value = TRUE)    ## Size x Sector Z-scores
    )
    
    ## 5. Already-bounded [0,1] features — quantile transform is near-identity
    bounded_cols <- c(
      grep("^secRank_", names(Train), value = TRUE)    ## Sector percentile ranks
    )
    
    ## Combine all exclusions
    exclude_cols <- unique(c(
      meta_cols,
      binary_cols,
      ordinal_cols,
      already_scaled_cols,
      bounded_cols
    ))
    
    ## ── B. Identify Columns to Transform ──────────────────────────────────
    numeric_cols      <- names(Train)[sapply(Train, is.numeric)]
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    ## Verify all expected families are represented
    families_present <- c("^r\\d+$", "^yoy_", "^accel_", "^expmean_",
                          "^dev_expmean_", "^expvol_", "^peak_drop_",
                          "^trough_rise_", "^momentum_", "^secDev_",
                          "^secVol_", "^secTrend_", "^secDiverg_")
    
    family_audit <- data.frame(
      family = families_present,
      n_cols = sapply(families_present, function(p)
        sum(grepl(p, cols_to_transform)))
    )
    
    message(paste("  Total features to transform:", length(cols_to_transform)))
    message("  Feature family breakdown:")
    print(family_audit)
    
    ## Sanity check — warn if any family has 0 matches
    ## (indicates a naming mismatch between 03A/03B and this block)
    empty_families <- family_audit$family[family_audit$n_cols == 0]
    if (length(empty_families) > 0) {
      warning("  The following families had 0 columns matched — check 03A/03B ran correctly:\n  ",
              paste(empty_families, collapse = ", "))
    }
    
    ## ── C. Guard: QuantileTransformation must exist ────────────────────────
    if (!exists("QuantileTransformation")) {
      stop("QuantileTransformation() function not found. Check Functions_Directory was sourced correctly.")
    }
    
    ## ── D. Transform Loop ─────────────────────────────────────────────────
    ## Fit transformation parameters on Train only.
    ## Apply the same mapping to Test — no refitting on Test.
    
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    
    failed_cols <- character(0)
    
    for (col in cols_to_transform) {
      
      ## Guard: column must exist in both sets
      if (!col %in% names(Train) || !col %in% names(Test)) {
        warning(paste("  Column not found in Train or Test, skipping:", col))
        next
      }
      
      ## Guard: skip if column is entirely NA in Train
      if (all(is.na(Train[[col]]))) {
        warning(paste("  Column entirely NA in Train, skipping:", col))
        next
      }
      
      res <- tryCatch({
        QuantileTransformation(Train[[col]], Test[[col]])
      }, error = function(e) {
        message(paste("  Transform failed for:", col, "—", e$message))
        failed_cols <<- c(failed_cols, col)
        NULL
      })
      
      if (!is.null(res)) {
        Train_Transformed[[col]] <- res$train
        Test_Transformed[[col]]  <- res$test
      }
    }
    
    ## ── E. Post-Transform Audit ────────────────────────────────────────────
    message("  Transformation complete.")
    message(paste("  Columns successfully transformed:", length(cols_to_transform) - length(failed_cols)))
    
    if (length(failed_cols) > 0) {
      message(paste("  Columns that failed transformation:", length(failed_cols)))
      message(paste("   ", paste(failed_cols, collapse = ", ")))
    }
    
    ## NA check — transformation should not introduce new NAs
    ## (structural NAs from 03A are already present and expected)
    na_before_train <- sum(is.na(Train[,  .SD, .SDcols = cols_to_transform]))
    na_after_train  <- sum(is.na(Train_Transformed[, .SD, .SDcols = cols_to_transform]))
    na_before_test  <- sum(is.na(Test[,   .SD, .SDcols = cols_to_transform]))
    na_after_test   <- sum(is.na(Test_Transformed[,  .SD, .SDcols = cols_to_transform]))
    
    if (na_after_train > na_before_train) {
      warning(paste("  Transformation introduced", na_after_train - na_before_train,
                    "new NAs in Train — inspect QuantileTransformation() for edge cases."))
    }
    if (na_after_test > na_before_test) {
      warning(paste("  Transformation introduced", na_after_test - na_before_test,
                    "new NAs in Test — inspect QuantileTransformation() for edge cases."))
    }
    
    message(paste0("  Train NA delta: ", na_after_train - na_before_train,
                   " | Test NA delta: ", na_after_test - na_before_test))
    
  } else {
    
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    message("  Skipping Quantile Transformation (Quantile_Transform = FALSE).")
    
  }
  
}, error = function(e) stop("03D Failed: ", e$message))

#==== 03E - Cleanup and factors ===============================================#

tryCatch({
  
  message("--- Starting 03E: Final Cleanup, Imputation & Validation ---")
  
  ## ── A. Ensure data.table ──────────────────────────────────────────────────
  if (!is.data.table(Train_Transformed)) setDT(Train_Transformed)
  if (!is.data.table(Test_Transformed))  setDT(Test_Transformed)
  
  ## ── B. Structural NA Imputation ───────────────────────────────────────────
  ## NAs here are not missing data — they are structural cold-start artefacts
  ## from lag/diff operations in 03A. Each family has a defensible fill value.
  ## Strategy: fit imputation values on Train only, apply same values to Test.
  
  message("  Imputing structural cold-start NAs...")
  
  ## Helper: impute a vector with a fixed value, both Train and Test
  impute_fixed <- function(col, fill_value) {
    if (col %in% names(Train_Transformed)) {
      Train_Transformed[is.na(get(col)), (col) := fill_value]
      Test_Transformed[is.na(get(col)),  (col) := fill_value]
    }
  }
  
  ## Helper: impute with Train median (for features where 0 is misleading)
  impute_train_median <- function(col) {
    if (col %in% names(Train_Transformed)) {
      fill_val <- median(Train_Transformed[[col]], na.rm = TRUE)
      Train_Transformed[is.na(get(col)), (col) := fill_val]
      Test_Transformed[is.na(get(col)),  (col) := fill_val]
    }
  }
  
  ## 1. YoY (1st diff) — NA at t=1: "no change observed" → 0
  yoy_cols <- grep("^yoy_", names(Train_Transformed), value = TRUE)
  for (col in yoy_cols) impute_fixed(col, 0)
  
  ## 2. Acceleration (2nd diff) — NA at t=1,2: "no acceleration observed" → 0
  accel_cols <- grep("^accel_", names(Train_Transformed), value = TRUE)
  for (col in accel_cols) impute_fixed(col, 0)
  
  ## 3. Momentum — NA at t=1: "no momentum signal" → 0
  momentum_cols <- grep("^momentum_", names(Train_Transformed), value = TRUE)
  for (col in momentum_cols) impute_fixed(col, 0)
  
  ## 4. Peak drop — NA at t=1: firm is at its own peak → 0 is exactly correct
  peak_cols <- grep("^peak_drop_", names(Train_Transformed), value = TRUE)
  for (col in peak_cols) impute_fixed(col, 0)
  
  ## 5. Trough rise — NA at t=1: firm is at its own trough → 0 is exactly correct
  trough_cols <- grep("^trough_rise_", names(Train_Transformed), value = TRUE)
  for (col in trough_cols) impute_fixed(col, 0)
  
  ## 6. Expanding volatility — NA at t=1: DO NOT use 0
  ##    0 implies "perfectly stable" which is a strong positive credit signal
  ##    Use Train median volatility — "typical firm uncertainty" is neutral
  expvol_cols <- grep("^expvol_", names(Train_Transformed), value = TRUE)
  for (col in expvol_cols) impute_train_median(col)
  
  ## 7. Sector trend — NA for earliest year in dataset per sector
  ##    Use 0: "no trend observed for this sector yet"
  sectrend_cols <- grep("^secTrend_", names(Train_Transformed), value = TRUE)
  for (col in sectrend_cols) impute_fixed(col, 0)
  
  ## 8. Sector divergence — depends on yoy (already imputed above) and
  ##    secTrend (just imputed) — recompute rather than impute if NA remains
  secdiverg_cols <- grep("^secDiverg_", names(Train_Transformed), value = TRUE)
  for (col in secdiverg_cols) impute_fixed(col, 0)
  
  ## Post-imputation NA check — only structural NAs should remain
  ## (secRank_ may have NAs for singleton sector-year groups — acceptable)
  remaining_na_train <- sum(is.na(Train_Transformed))
  remaining_na_test  <- sum(is.na(Test_Transformed))
  message(paste0("  Remaining NAs after imputation — Train: ", remaining_na_train,
                 " | Test: ", remaining_na_test))
  
  if (remaining_na_train > 0) {
    ## Identify which columns still have NAs — helps diagnose unexpected issues
    na_cols <- names(Train_Transformed)[sapply(Train_Transformed, anyNA)]
    message("  Columns with remaining NAs in Train:")
    na_summary <- sapply(na_cols, function(col) sum(is.na(Train_Transformed[[col]])))
    print(sort(na_summary, decreasing = TRUE))
  }
  
  ## ── C. Metadata Columns to Drop ───────────────────────────────────────────
  ## Pure identifiers and leakage columns.
  ## NOTE: time_index and history_length are KEPT — they carry credit signal
  ## (young firms default more; firm age is a proxy for survivorship/stability)
  
  cols_to_drop <- c(
    "id", "company_id", "row_id",   ## Identifiers
    "refdate",                       ## Date — year is already extracted
    "year"                           ## Encoded in sector_year stats already
    ## time_index: KEPT (maturity signal)
    ## history_length: KEPT (firm age signal)
  )
  cols_to_drop <- intersect(cols_to_drop, names(Train_Transformed))
  
  ## ── D. Align Categorical Columns ──────────────────────────────────────────
  ## Enforce Train factor levels on Test strictly.
  ## Any Test category not seen in Train is coerced to NA then imputed.
  ## This prevents silent level mismatch errors in GLM/sparse.model.matrix.
  
  message("  Aligning categorical levels...")
  
  cat_cols <- names(Train_Transformed)[
    sapply(Train_Transformed, function(x) is.character(x) || is.factor(x))
  ]
  cat_cols <- setdiff(cat_cols, c("y", cols_to_drop))
  
  unseen_levels <- list()
  
  for (col in cat_cols) {
    Train_Transformed[, (col) := as.factor(get(col))]
    train_levels <- levels(Train_Transformed[[col]])
    
    ## Check for Test levels not in Train before coercing
    test_vals    <- unique(Test_Transformed[[col]])
    new_in_test  <- setdiff(as.character(test_vals), train_levels)
    
    if (length(new_in_test) > 0) {
      unseen_levels[[col]] <- new_in_test
      warning(paste0("  Column '", col, "' has ", length(new_in_test),
                     " Test level(s) not seen in Train — will become NA: ",
                     paste(new_in_test, collapse = ", ")))
    }
    
    Test_Transformed[, (col) := factor(get(col), levels = train_levels)]
    
    ## Impute unseen Test levels with Train mode
    if (length(new_in_test) > 0) {
      mode_val <- train_levels[which.max(tabulate(match(Train_Transformed[[col]], train_levels)))]
      Test_Transformed[is.na(get(col)), (col) := mode_val]
    }
  }
  
  if (length(unseen_levels) == 0) {
    message("  All categorical levels aligned cleanly.")
  }
  
  ## ── E. Target Variable Enforcement ────────────────────────────────────────
  ## XGBoost binary:logistic requires numeric 0/1.
  ## GLM/LASSO works with both but numeric is safer.
  ## Explicit conversion guards against factor y from upstream processing.
  
  for (dt in list(Train_Transformed, Test_Transformed)) {
    if ("y" %in% names(dt)) {
      dt[, y := as.integer(as.character(y))]
    }
  }
  
  ## ── F. Build Final Modelling Sets ─────────────────────────────────────────
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  
  Train_Final <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final  <- copy(Test_Transformed[,  ..cols_to_keep])
  
  ## ── G. Final Validation Report ────────────────────────────────────────────
  message("--- 03E Validation Report ---")
  
  ## Dimensions
  cat(sprintf("  Train_Final : %d rows x %d cols\n", nrow(Train_Final), ncol(Train_Final)))
  cat(sprintf("  Test_Final  : %d rows x %d cols\n", nrow(Test_Final),  ncol(Test_Final)))
  
  ## Class balance
  cat(sprintf("  Train default rate : %.3f%% (%d defaults)\n",
              100 * mean(Train_Final$y), sum(Train_Final$y)))
  cat(sprintf("  Test  default rate : %.3f%% (%d defaults)\n",
              100 * mean(Test_Final$y),  sum(Test_Final$y)))
  
  ## NA summary
  cat(sprintf("  Train NAs : %d\n", sum(is.na(Train_Final))))
  cat(sprintf("  Test  NAs : %d\n", sum(is.na(Test_Final))))
  
  ## Column alignment check — Train and Test must have identical columns
  col_mismatch <- setdiff(names(Train_Final), names(Test_Final))
  if (length(col_mismatch) > 0) {
    stop(paste("CRITICAL: Column mismatch between Train_Final and Test_Final:",
               paste(col_mismatch, collapse = ", ")))
  } else {
    message("  Column alignment: OK")
  }
  
  ## Feature family summary
  families <- c("^r\\d+$", "^yoy_", "^accel_", "^expmean_", "^dev_expmean_",
                "^expvol_", "^peak_drop_", "^trough_rise_", "^momentum_",
                "^consec_decline_", "^secZ_", "^secDev_", "^secRank_",
                "^secVol_", "^secTrend_", "^secDiverg_", "^secSizeZ_", "^sector_")
  
  family_summary <- data.frame(
    family  = families,
    n_cols  = sapply(families, function(p) sum(grepl(p, names(Train_Final))))
  )
  family_summary <- family_summary[family_summary$n_cols > 0, ]
  
  message("  Feature family breakdown:")
  print(family_summary)
  cat(sprintf("  Total modelling features (excl. y): %d\n", ncol(Train_Final) - 1))
  
  message("03E Complete. Train_Final and Test_Final ready for modelling.")
  
}, error = function(e) stop("03E Failed: ", e$message))
  
#==============================================================================#
#==== 04 - VAE Setup & Data Preparation =======================================#
#==============================================================================#

#==== 04A - Data preparation ==================================================#

tryCatch({
  message("--- Starting VAE Data Prep (with Label Encoding) ---")
  
  if(!is.data.table(Train_Final)) setDT(Train_Final)
  
  # 1. Safely identify metadata
  meta_cols <- c("y", "id", "refdate", "year", "time_index", "company_id", "row_id")
  actual_meta <- intersect(meta_cols, names(Train_Final))
  
  cols_to_keep <- setdiff(names(Train_Final), actual_meta)
  features_only <- Train_Final[, ..cols_to_keep]
  
  # 2. Define column types
  expected_bin_cols <- c("groupmember", "public", "is_new_company", "Profit_Trend_Consistent", "is_mature")
  bin_cols <- intersect(expected_bin_cols, names(features_only))
  
  cat_cols <- names(features_only)[sapply(features_only, is.factor)]
  num_cols <- names(features_only)[sapply(features_only, is.numeric)]
  cont_cols <- setdiff(num_cols, bin_cols)
  
  # 3. Extract Continuous and Binary natively
  data_cont <- features_only[, ..cont_cols]
  data_bin  <- features_only[, ..bin_cols]
  
  # 4. LABEL ENCODE Categorical Features for Keras
  data_cat <- features_only[, ..cat_cols]
  for (col in cat_cols) {
    # as.numeric() on a factor returns 1, 2, 3... 
    # We subtract 1 to make it 0-indexed, which Keras strongly prefers
    data_cat[, (col) := as.numeric(get(col)) - 1]
  }
  
  # 5. Combine into a standard data.frame (all numeric now)
  vae_input_data <- as.data.frame(cbind(data_cont, data_bin, data_cat))
  
  if(any(is.na(vae_input_data))) stop("Input data contains NAs! The VAE will crash.")
  
  # 6. Extract and set distribution metadata for autotab
  if(exists("extracting_distribution")) {
    # Because they are integers now, autotab correctly flags them as categorical
    feat_dist <- extracting_distribution(vae_input_data)
    set_feat_dist(feat_dist)
  }
  
  message("VAE Prep Complete. Ready for VAE_train().")
  
}, error = function(e) message("VAE Prep Error: ", e))

#==== 04B - Train the Baseline VAE ============================================#

tryCatch({
  
vae_fit <- VAE_train(
  data = vae_input_data,
  encoder_info = encoder_config,
  decoder_info = decoder_config,
  latent_dim = 8,
  epoch = 100,
  batchsize = 256,
  beta = 0.5,
  lr = 0.001,
  temperature = 0.5,
  wait = 10,
  kl_warm = TRUE,
  beta_epoch = 10,
  Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0
)

}, error = function(e) message(e))

#==============================================================================#
#==== 05 - VAE Modeling =======================================================#
#==============================================================================#

Use_VAE_Only <- TRUE

tryCatch({

#==== 05A - Strategy A: Latent features (Dimensional reduction) ===============#

## Training Set.
tryCatch({
    enc_weights <- Encoder_weights(
      encoder_layers = 2, 
      trained_model = vae_fit$trained_model,
      lip_enc = 0, pi_enc = 0, BNenc_layers = 0, learn_BN = 0
    )
    
    enc_model <- encoder_latent(
      encoder_input = vae_input_data, 
      encoder_info = encoder_config, 
      latent_dim = 8,
      Lip_en = 0, power_iterations = 0
    ) %>% keras::set_weights(enc_weights)
    
    latent_output <- predict(enc_model, as.matrix(vae_input_data))
    Strategy_A_LF <- as.data.frame(latent_output[[1]])
    colnames(Strategy_A_LF) <- paste0("l", 1:8)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_A_Train <- cbind(Train_Final[, .(y)], Strategy_A_LF)
    } else {
      Strategy_A_Train <- cbind(Train_Final, Strategy_A_LF)
    }
    
  }, error = function(e) message("Strategy A Train Error: ", e))
  
## Test set.
tryCatch({
    message("--- Starting Strategy A Test Preparation ---")
    
    # 1. Select exact columns used in training
    target_cols <- colnames(vae_input_data)
    
    # Keep as data.table temporarily so we can apply the encoding loop
    test_vae_input_dt <- as.data.table(Test_Final[, ..target_cols])
    
    # 2. THE FIX: Apply Integer Encoding to Test Set Categoricals
    # (Relies on 'cat_cols' from the VAE Prep block)
    if(exists("cat_cols") && length(cat_cols) > 0) {
      for (col in cat_cols) {
        test_vae_input_dt[, (col) := as.numeric(get(col)) - 1]
      }
    }
    
    # 3. Convert to matrix format required by Keras predict
    # Because all columns are now numeric, as.matrix() will create a numeric matrix
    x_test_matrix <- as.matrix(as.data.frame(test_vae_input_dt))
    
    # 4. Predict Latent Features
    latent_output_raw <- predict(enc_model, x_test_matrix)
    latent_values <- if(is.list(latent_output_raw)) latent_output_raw[[1]] else latent_output_raw
    
    Strategy_A_LF_Test <- as.data.frame(latent_values)
    colnames(Strategy_A_LF_Test) <- paste0("l", 1:8)
    
    # 5. Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_A_Test <- cbind(Test_Final[, .(y)], Strategy_A_LF_Test)
    } else {
      Strategy_A_Test <- cbind(Test_Final, Strategy_A_LF_Test)
    }
    
    message("Success: Strategy_A_Test created.")
    
  }, error = function(e) message("Strategy A Test Error: ", e))
  
#==== 05B - Strategy B: Anomaly Score =========================================#

## Training set.
tryCatch({
    message("--- Calculating Training Anomaly Scores ---")
    
    reconstructed_list <- predict(vae_fit$trained_model, as.matrix(vae_input_data))
    reconstructed_data <- as.matrix(if(is.list(reconstructed_list)) reconstructed_list[[1]] else reconstructed_list)
    
    # Safely extract continuous dimensions for MSE
    n_cont_cols <- length(cont_cols)
    
    if (n_cont_cols > 0) {
      input_cont <- as.matrix(vae_input_data[, 1:n_cont_cols, drop = FALSE])
      recon_cont <- as.matrix(reconstructed_data[, 1:n_cont_cols, drop = FALSE])
      
      mse_raw <- (input_cont - recon_cont)^2
      mse_raw[is.infinite(mse_raw)] <- 1e9 
      mse_normalized <- rowSums(mse_raw, na.rm = TRUE) / n_cont_cols
    } else {
      mse_normalized <- numeric(nrow(vae_input_data))
    }
    
    # Note: Skipping BCE for categorical here because autotab expands factors 
    # into multiple output nodes (softmax), causing dimension mismatch with raw input.
    # MSE on continuous features is the most stable anomaly score.
    final_score <- mse_normalized 
    
    Strategy_B_AS <- data.table(anomaly_score = final_score)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_B_Train <- cbind(Train_Final[, .(y)], Strategy_B_AS)
    } else {
      Strategy_B_Train <- cbind(Train_Final, Strategy_B_AS)
    }
    
  }, error = function(e) message("Error in Strategy B Train: ", e))
  
## Test set.
tryCatch({
    message("--- Calculating Test Anomaly Scores ---")
    
    # We already built x_test_matrix in Strategy A
    reconstructed_list <- predict(vae_fit$trained_model, x_test_matrix)
    test_recon_matrix <- as.matrix(if(is.list(reconstructed_list)) reconstructed_list[[1]] else reconstructed_list)
    
    test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix > 0] <- 1e9
    test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix < 0] <- -1e9
    test_recon_matrix[is.na(test_recon_matrix)] <- 0 
    
    if (n_cont_cols > 0) {
      input_cont_test <- as.matrix(x_test_matrix[, 1:n_cont_cols, drop = FALSE])
      recon_cont_test <- as.matrix(test_recon_matrix[, 1:n_cont_cols, drop = FALSE])
      
      input_cont_test[is.infinite(input_cont_test)] <- 1e9 
      sq_error <- (input_cont_test - recon_cont_test)^2
      sq_error[is.infinite(sq_error)] <- 1e9 
      
      mse_norm_test <- rowSums(sq_error, na.rm = TRUE) / n_cont_cols
    } else {
      mse_norm_test <- numeric(nrow(x_test_matrix)) 
    }
    
    Strategy_B_AS_Test <- data.table(anomaly_score = mse_norm_test)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_B_Test <- cbind(Test_Final[, .(y)], Strategy_B_AS_Test)
    } else {
      Strategy_B_Test <- cbind(Test_Final, Strategy_B_AS_Test)
    }
    
  }, error = function(e) message("Error in Test Strategy B: ", e))
  
#==== 05C - Strategy C: Feature Denoising =====================================#

## Training set.
tryCatch({
    message("--- Strategy C: Training DAE ---")
    
    # 1. Keras Dense requires a strictly numeric matrix. We MUST one-hot encode here.
    # Create a temporary dataframe without 'y'
    dae_train_df <- as.data.frame(Train_Final[, .SD, .SDcols = !c("y")])
    x_train_clean <- model.matrix(~ . - 1, data = dae_train_df)
    input_dim <- ncol(x_train_clean)
    
    # 2. Build DAE
    input_layer <- layer_input(shape = c(input_dim))
    
    encoded <- input_layer %>%
      layer_gaussian_noise(stddev = 0.1) %>% 
      layer_dense(units = 128, activation = "relu") %>% 
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 8, activation = "relu", name = "bottleneck") 
    
    decoded <- encoded %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 128, activation = "relu") %>%
      layer_dense(units = input_dim, activation = "linear") 
    
    dae_autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
    dae_autoencoder %>% compile(optimizer = "adam", loss = "mse")
    
    # 3. Train
    history <- dae_autoencoder %>% fit(
      x = x_train_clean, y = x_train_clean, 
      epochs = 50, batch_size = 256, shuffle = TRUE, validation_split = 0.1, verbose = 0
    )
    
    # 4. Extract Latent Features
    encoder_only <- keras_model(inputs = dae_autoencoder$input, 
                                outputs = get_layer(dae_autoencoder, "bottleneck")$output)
    
    latent_train <- predict(encoder_only, x_train_clean)
    Strategy_C_LF <- as.data.frame(latent_train)
    colnames(Strategy_C_LF) <- paste0("dae_l", 1:8)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_C_Train <- cbind(Train_Final[, .(y)], Strategy_C_LF)
    } else {
      Strategy_C_Train <- cbind(Train_Final, Strategy_C_LF)
    }
    
  }, error = function(e) message("Strategy C Training Error: ", e))
  
## Test set.
tryCatch({
    message("--- Preparing Test Set for Strategy C ---")
    
    # 1. Prepare numeric matrix via model.matrix
    dae_test_df <- as.data.frame(Test_Final[, .SD, .SDcols = !c("y")])
    test_dae_input_raw <- model.matrix(~ . - 1, data = dae_test_df)
    
    # 2. MATRIX ALIGNMENT (Crucial for DAE Keras predict)
    target_cols_dae <- colnames(x_train_clean)
    
    missing_cols <- setdiff(target_cols_dae, colnames(test_dae_input_raw))
    if(length(missing_cols) > 0) {
      missing_mat <- matrix(0, nrow = nrow(test_dae_input_raw), ncol = length(missing_cols))
      colnames(missing_mat) <- missing_cols
      test_dae_input_raw <- cbind(test_dae_input_raw, missing_mat)
    }
    
    extra_cols <- setdiff(colnames(test_dae_input_raw), target_cols_dae)
    if(length(extra_cols) > 0) {
      test_dae_input_raw <- test_dae_input_raw[, !colnames(test_dae_input_raw) %in% extra_cols, drop = FALSE]
    }
    
    x_test_dae_matrix <- test_dae_input_raw[, target_cols_dae]
    
    # 3. Predict Latent Features
    test_latent_dae <- predict(encoder_only, x_test_dae_matrix)
    Strategy_C_LF_Test <- as.data.frame(test_latent_dae)
    colnames(Strategy_C_LF_Test) <- paste0("dae_l", 1:8)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_C_Test <- cbind(Test_Final[, .(y)], Strategy_C_LF_Test)
    } else {
      Strategy_C_Test <- cbind(Test_Final, Strategy_C_LF_Test)
    }
    
  }, error = function(e) message("DAE Test Set Error: ", e))
  
#==============================================================================#

}, error = function(e) message(e))

#==============================================================================#
#==== 00 - END ================================================================#
#==============================================================================#