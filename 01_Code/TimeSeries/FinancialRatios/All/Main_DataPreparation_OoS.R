#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

## Last changed: 2026-02-04 | Tristan Leiter

### Just for Tristan. Special config bc of wrong python installation.

Sys.setenv(RETICULATE_PYTHON = "C:/venvs/autotab_env/Scripts/python.exe")
# Sys.setenv(TF_USE_LEGACY_KERAS = "1")
# Sys.setenv(USE_LEGACY_KERAS = "1")

library(reticulate)
use_python("C:/venvs/autotab_env/Scripts/python.exe", required = TRUE)

# reticulate::py_run_string("import tensorflow as tf; import keras; print(keras.__version__)")

library(keras)
library(autotab)

####################################

# 1. Lock in the environment
# library(tensorflow)
# 
# install_tensorflow(
#   method = "virtualenv", 
#   envname = "C:/venvs/autotab_env", 
#   version = "2.15.0",
#   extra_packages = c("keras==2.15.0")
# )
# 
# reticulate::py_install(
#   packages = "tensorflow-probability==0.23.0", 
#   envname = "C:/venvs/autotab_env", 
#   pip = TRUE
# )

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

Data <- DataPreprocessing(Data, Tolerance = 2)

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .))) %>%
  drop_na()

### Remove all financial raw-positions.
Data <- Data[, !grepl("^f", colnames(Data))]

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

MVstratifiedsampling <- function(Data,
                                 strat_vars = c("sector", "y"),
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
  Data <- as.data.table(Data)
  setorder(Data, id, refdate)
  
  ## ── B. Detect Ratio Columns Programmatically ───────────────────────────────
  ## Ratios are all numeric columns excluding known metadata/target columns.
  ## No hardcoded "r1:r18" — works regardless of how many ratios exist.
  
  meta_cols <- c("id", "refdate", "sector", "size", "groupmember",
                 "public", "y", "year")
  
  all_ratios <- setdiff(
    names(Data)[sapply(Data, is.numeric)],
    meta_cols
  )
  
  ## ── C. Classify Ratios by Economic Direction ───────────────────────────────
  ## Rather than hardcoding lists, classify by statistical property:
  ##   "higher is better" → ratios where high values are healthy
  ##   "lower is better"  → ratios where low values are healthy
  ##
  ## Heuristic: if the Train-set median is > 0.5 AND the ratio is bounded
  ## between 0 and 1, it is likely a leverage/debt ratio (lower = safer).
  ## Override by setting PEAK_RATIOS / TROUGH_RATIOS / CONSEC_RATIOS upstream
  ## if domain knowledge should take precedence over the heuristic.
  
  if (exists("PEAK_RATIOS") && exists("TROUGH_RATIOS")) {
    peak_ratios   <- intersect(PEAK_RATIOS,   all_ratios)
    trough_ratios <- intersect(TROUGH_RATIOS, all_ratios)
  } else {
    ## Heuristic classification
    ratio_medians  <- sapply(all_ratios, function(r) median(Data[[r]], na.rm = TRUE))
    ratio_mins     <- sapply(all_ratios, function(r) min(Data[[r]],    na.rm = TRUE))
    ratio_maxs     <- sapply(all_ratios, function(r) max(Data[[r]],    na.rm = TRUE))
    
    is_bounded01   <- ratio_mins >= 0 & ratio_maxs <= 1
    is_high_median <- ratio_medians > 0.5
    
    ## High median + bounded [0,1] → likely a debt/leverage ratio → lower is better
    trough_ratios  <- all_ratios[is_bounded01 &  is_high_median]
    peak_ratios    <- all_ratios[!(is_bounded01 & is_high_median)]
  }
  
  ## Consecutive decline: apply to all non-leverage ratios by default
  ## (override with CONSEC_RATIOS upstream if needed)
  consec_ratios <- if (exists("CONSEC_RATIOS")) {
    intersect(CONSEC_RATIOS, all_ratios)
  } else {
    peak_ratios   ## decline in "higher is better" ratios is the meaningful signal
  }
  
  message(sprintf("  Detected %d ratio columns | %d peak | %d trough | %d consec",
                  length(all_ratios), length(peak_ratios),
                  length(trough_ratios), length(consec_ratios)))
  
  ## ── D. Base Tracking ───────────────────────────────────────────────────────
  Data[, time_index     := seq_len(.N),              by = id]
  Data[, is_mature      := fifelse(time_index >= 3, 1L, 0L)]
  Data[, history_length := .N,                       by = id]
  Data[, has_history    := fifelse(time_index > 1,  1L, 0L)]
  
  ## ── E. Year-over-Year Changes (1st Differences) ───────────────────────────
  message("  Computing YoY changes...")
  
  Data[, paste0("yoy_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
       by = id, .SDcols = all_ratios]
  
  ## ── F. Acceleration (2nd Differences) ─────────────────────────────────────
  message("  Computing acceleration (2nd differences)...")
  
  yoy_cols <- paste0("yoy_", all_ratios)
  
  Data[, paste0("accel_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
       by = id, .SDcols = yoy_cols]
  
  ## ── G. Expanding Mean ─────────────────────────────────────────────────────
  message("  Computing expanding means...")
  
  Data[, paste0("expmean_", all_ratios) :=
         lapply(.SD, cummean),
       by = id, .SDcols = all_ratios]
  
  ## ── H. Deviation from Expanding Mean ──────────────────────────────────────
  message("  Computing deviations from expanding mean...")
  
  expmean_cols <- paste0("expmean_", all_ratios)
  
  for (i in seq_along(all_ratios)) {
    Data[, paste0("dev_expmean_", all_ratios[i]) :=
           get(all_ratios[i]) - get(expmean_cols[i])]
  }
  
  ## ── I. Expanding Volatility ────────────────────────────────────────────────
  message("  Computing expanding volatility...")
  
  Data[, paste0("expvol_", all_ratios) :=
         lapply(.SD, function(x) {
           n      <- seq_along(x)
           mu     <- cummean(x)
           expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
           fifelse(n < 2L, NA_real_, sqrt(pmax(0, expvar)))
         }),
       by = id, .SDcols = all_ratios]
  
  ## ── J. Peak Deterioration ─────────────────────────────────────────────────
  message("  Computing peak deterioration...")
  
  if (length(peak_ratios) > 0) {
    Data[, paste0("peak_drop_", peak_ratios) :=
           lapply(.SD, function(x) x - cummax(x)),
         by = id, .SDcols = peak_ratios]
    Data[time_index == 1L,
         paste0("peak_drop_", peak_ratios) := lapply(peak_ratios, function(x) NA_real_)]
  }
  
  ## ── K. Trough Rise ────────────────────────────────────────────────────────
  message("  Computing trough rise...")
  
  if (length(trough_ratios) > 0) {
    Data[, paste0("trough_rise_", trough_ratios) :=
           lapply(.SD, function(x) x - cummin(x)),
         by = id, .SDcols = trough_ratios]
    Data[time_index == 1L,
         paste0("trough_rise_", trough_ratios) := lapply(trough_ratios, function(x) NA_real_)]
  }
  
  ## ── L. Momentum (2Y vs Expanding Mean) ────────────────────────────────────
  message("  Computing momentum...")
  
  Data[, paste0("momentum_", all_ratios) :=
         lapply(.SD, function(x) {
           frollmean(x, n = 2L, align = "right", fill = NA) - cummean(x)
         }),
       by = id, .SDcols = all_ratios]
  
  ## ── M. Consecutive Decline Counter ────────────────────────────────────────
  message("  Computing consecutive decline counters...")
  
  for (col in consec_ratios) {
    yoy_col <- paste0("yoy_", col)
    out_col <- paste0("consec_decline_", col)
    
    Data[, (out_col) := {
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
  
  ## ── N. Sanity Check ────────────────────────────────────────────────────────
  ## Family audit is fully dynamic — no hardcoded family list.
  
  ts_prefixes <- c("yoy_", "accel_", "expmean_", "dev_expmean_", "expvol_",
                   "peak_drop_", "trough_rise_", "momentum_", "consec_decline_")
  
  na_audit <- data.frame(
    family   = ts_prefixes,
    n_cols   = sapply(ts_prefixes, function(p)
      length(grep(paste0("^", p), names(Data), value = TRUE))),
    na_count = sapply(ts_prefixes, function(p) {
      cols <- grep(paste0("^", p), names(Data), value = TRUE)
      if (length(cols) == 0) return(0L)
      sum(is.na(Data[, .SD, .SDcols = cols]))
    })
  )
  
  message("03A Complete.")
  message(sprintf("  Ratio columns detected : %d", length(all_ratios)))
  message(sprintf("  New TS features        : %d", sum(na_audit$n_cols)))
  message(sprintf("  Total columns          : %d", ncol(Data)))
  message(sprintf("  Rows                   : %d", nrow(Data)))
  print(na_audit, row.names = FALSE)
  
}, error = function(e) stop("03A Failed: ", e$message))

#==== 03B - Data Sampling Out-of-Sample =======================================#

tryCatch({
  
  message("--- Starting 03B: Train/Test Split ---")
  
  Data_Sampled <- MVstratifiedsampling(
    Data       = as.data.frame(Data),
    strat_vars = c("sector", "y_ever"),
    Train_size = 0.7
  )
  
  Train <- as.data.table(Data_Sampled[["Train"]])
  Test  <- as.data.table(Data_Sampled[["Test"]])
  
  train_ids <- unique(Train$id)
  test_ids  <- unique(Test$id)
  
  stopifnot("Firm-level leakage detected!" = length(intersect(train_ids, test_ids)) == 0)
  stopifnot("Row loss detected!"           = nrow(Train) + nrow(Test) == nrow(Data))
  
  message(sprintf("Train: %d firms | %d rows | %.1f%% default rate",
                  length(train_ids), nrow(Train), 100 * mean(Train$y)))
  message(sprintf("Test:  %d firms | %d rows | %.1f%% default rate",
                  length(test_ids),  nrow(Test),  100 * mean(Test$y)))
  
}, error = function(e) stop("03B Failed: ", e$message))

#==== 03C - Sector Specific Deviation Features ================================#

tryCatch({
  
  message("--- Starting 03C: Sector Deviation Features (Leakage-Free) ---")
  
  ## ── A. Setup ───────────────────────────────────────────────────────────────
  if (!is.data.table(Train)) setDT(Train)
  if (!is.data.table(Test))  setDT(Test)
  if (!"year" %in% names(Train)) Train[, year := year(refdate)]
  if (!"year" %in% names(Test))  Test[,  year := year(refdate)]
  
  n_train_init <- nrow(Train)
  n_test_init  <- nrow(Test)
  
  ## ── B. Detect Sector Ratios ───────────────────────────────────────────────
  ## "public" removed from meta_cols — it is a continuous % ownership variable
  ## and should receive sector deviation features like any other ratio.
  meta_cols   <- c("id", "refdate", "sector", "size", "groupmember", "y", "year")
  ts_prefixes <- c("^yoy_", "^accel_", "^expmean_", "^dev_expmean_",
                   "^expvol_", "^peak_drop_", "^trough_rise_",
                   "^momentum_", "^consec_decline_",
                   "^time_", "^is_", "^has_", "^history_")
  ts_cols     <- grep(paste(ts_prefixes, collapse = "|"), names(Train), value = TRUE)
  
  sector_ratios <- setdiff(
    names(Train)[sapply(Train, is.numeric)],
    c(meta_cols, ts_cols)
  )
  
  ## ── C. Coerce Sector Ratios to Double ─────────────────────────────────────
  int_cols <- sector_ratios[vapply(sector_ratios, function(nm)
    is.integer(Train[[nm]]), logical(1))]
  
  if (length(int_cols) > 0) {
    message(sprintf("  Coercing %d integer ratio columns to double: %s",
                    length(int_cols), paste(int_cols, collapse = ", ")))
    for (col in int_cols) {
      Train[, (col) := as.double(get(col))]
      Test[,  (col) := as.double(get(col))]
    }
  }
  
  MIN_SS_FIRMS       <- if (exists("MIN_SS_FIRMS")) MIN_SS_FIRMS else 5L
  ss_coverage        <- Train[, .(n = .N), by = .(sector, size, year)]
  size_sector_ratios <- sector_ratios
  if (mean(ss_coverage$n) < MIN_SS_FIRMS)
    warning("  Low size-sector cell coverage — secSizeZ features may be sparse.")
  
  message(sprintf("  Detected %d sector ratios (%d integer cols coerced)",
                  length(sector_ratios), length(int_cols)))
  
  ## ── D. Fit Sector Statistics on Train Only ────────────────────────────────
  message("  Fitting sector-year statistics on Train only...")
  
  sector_stats <- Train[,
                        c(lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                 function(col) mean(get(col),          na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                 function(col) sd(get(col),            na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("med_",  sector_ratios)),
                                 function(col) median(get(col),        na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("p25_",  sector_ratios)),
                                 function(col) quantile(get(col), 0.25, na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("p75_",  sector_ratios)),
                                 function(col) quantile(get(col), 0.75, na.rm = TRUE)),
                          list(n_firms = .N)),
                        by = .(sector, year)
  ]
  
  sector_fallback <- Train[,
                           c(lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                    function(col) mean(get(col), na.rm = TRUE)),
                             lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                    function(col) sd(get(col),   na.rm = TRUE))),
                           by = sector
  ]
  global_fallback <- Train[,
                           c(lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                    function(col) mean(get(col), na.rm = TRUE)),
                             lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                    function(col) sd(get(col),   na.rm = TRUE)))
  ]
  
  ## ── E. Join Stats + Fallback Cascade ──────────────────────────────────────
  message("  Joining sector stats with fallback cascade...")
  
  join_stats <- function(DT, stats, by_cols) {
    stats_u   <- unique(stats, by = by_cols)
    setkeyv(stats_u, by_cols)
    join_cols <- setdiff(names(stats_u), by_cols)
    for (sc in join_cols)
      DT[stats_u, (sc) := get(paste0("i.", sc)), on = by_cols]
  }
  
  join_stats(Train, sector_stats,    c("sector", "year"))
  join_stats(Test,  sector_stats,    c("sector", "year"))
  
  probe_col    <- paste0("mean_", sector_ratios[1])
  missing_mask <- is.na(Test[[probe_col]])
  if (any(missing_mask)) {
    warning(sprintf("  %d Test rows with unseen sector-year → sector fallback",
                    sum(missing_mask)))
    join_stats(Test[missing_mask], sector_fallback, "sector")
  }
  still_missing <- is.na(Test[[probe_col]])
  if (any(still_missing)) {
    warning(sprintf("  %d Test rows still missing → global fallback", sum(still_missing)))
    for (col in c(paste0("mean_", sector_ratios), paste0("sd_", sector_ratios)))
      Test[still_missing, (col) := global_fallback[[col]]]
  }
  
  ## ── F. Sector Z-Score ─────────────────────────────────────────────────────
  message("  Computing sector Z-scores...")
  
  z_score <- function(DT, col) {
    z <- (DT[[col]] - DT[[paste0("mean_", col)]]) / DT[[paste0("sd_", col)]]
    fifelse(is.na(z) | is.infinite(z), 0, z)
  }
  for (col in sector_ratios) {
    Train[, paste0("secZ_", col) := z_score(Train, col)]
    Test[,  paste0("secZ_", col) := z_score(Test,  col)]
  }
  
  ## ── G. Raw Sector Deviation ───────────────────────────────────────────────
  message("  Computing raw sector deviations...")
  
  for (col in sector_ratios) {
    Train[, paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
    Test[,  paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
  }
  
  ## ── H. Sector Percentile Rank ─────────────────────────────────────────────
  message("  Computing sector percentile ranks...")
  
  for (col in sector_ratios) {
    out <- paste0("secRank_", col)
    Train[, (out) := frank(get(col), ties.method = "average",
                           na.last = "keep") / .N, by = .(sector, year)]
    
    ecdf_list <- Train[, .(ecdf_fn = list(ecdf(get(col)[!is.na(get(col))]))),
                       by = .(sector, year)]
    Test[, (out) := NA_real_]
    for (i in seq_len(nrow(ecdf_list))) {
      mask <- Test$sector == ecdf_list$sector[i] &
        Test$year   == ecdf_list$year[i]   &
        !is.na(Test[[col]])
      if (any(mask))
        Test[mask, (out) := ecdf_list$ecdf_fn[[i]](get(col))]
    }
    Test[is.na(get(out)), (out) := 0.5]
  }
  
  ## ── I. Sector Volatility ──────────────────────────────────────────────────
  message("  Extracting sector volatility features...")
  
  for (col in sector_ratios) {
    Train[, paste0("secVol_", col) := get(paste0("sd_", col))]
    Test[,  paste0("secVol_", col) := get(paste0("sd_", col))]
  }
  
  ## ── J. Sector Trend ───────────────────────────────────────────────────────
  message("  Computing sector trend...")
  
  setorder(sector_stats, sector, year)
  sector_trend <- sector_stats[,
                               c(list(sector = sector, year = year),
                                 lapply(setNames(paste0("mean_", sector_ratios),
                                                 paste0("secTrend_", sector_ratios)),
                                        function(col) get(col) - shift(get(col), n = 1L, type = "lag"))),
                               by = sector
  ]
  join_stats(Train, sector_trend, c("sector", "year"))
  join_stats(Test,  sector_trend, c("sector", "year"))
  
  ## ── K. Firm vs Sector Divergence ──────────────────────────────────────────
  message("  Computing firm vs sector divergence...")
  
  for (col in sector_ratios) {
    yoy_col   <- paste0("yoy_",      col)
    trend_col <- paste0("secTrend_", col)
    if (yoy_col %in% names(Train) && trend_col %in% names(Train)) {
      Train[, paste0("secDiverg_", col) := get(yoy_col) - get(trend_col)]
      Test[,  paste0("secDiverg_", col) := get(yoy_col) - get(trend_col)]
    }
  }
  
  ## ── L. Size x Sector Z-Score ──────────────────────────────────────────────
  message("  Computing size x sector Z-scores...")
  
  ss_stats <- Train[,
                    c(lapply(setNames(size_sector_ratios, paste0("ss_mean_", size_sector_ratios)),
                             function(col) mean(get(col), na.rm = TRUE)),
                      lapply(setNames(size_sector_ratios, paste0("ss_sd_",   size_sector_ratios)),
                             function(col) sd(get(col),   na.rm = TRUE))),
                    by = .(sector, size, year)
  ]
  join_stats(Train, ss_stats, c("sector", "size", "year"))
  join_stats(Test,  ss_stats, c("sector", "size", "year"))
  
  for (col in size_sector_ratios) {
    Train[, paste0("secSizeZ_", col) := z_score(
      data.table(x    = Train[[col]],
                 mean = Train[[paste0("ss_mean_", col)]],
                 sd   = Train[[paste0("ss_sd_",   col)]]), "x")]
    Test[,  paste0("secSizeZ_", col) := z_score(
      data.table(x    = Test[[col]],
                 mean = Test[[paste0("ss_mean_", col)]],
                 sd   = Test[[paste0("ss_sd_",   col)]]), "x")]
  }
  
  ## ── M. Sector Dummies ─────────────────────────────────────────────────────
  message("  Creating sector dummies...")
  
  sector_levels <- setdiff(unique(Train$sector), NA)
  ref_level     <- sector_levels[which.max(tabulate(match(Train$sector, sector_levels)))]
  dummy_levels  <- setdiff(sector_levels, ref_level)
  
  for (s in dummy_levels) {
    out_col <- paste0("sector_", gsub("[^a-zA-Z0-9]", "_", s))
    Train[, (out_col) := fifelse(sector == s, 1L, 0L)]
    Test[,  (out_col) := fifelse(sector == s, 1L, 0L)]
  }
  
  ## ── N. Cleanup ────────────────────────────────────────────────────────────
  stat_prefixes <- c("mean_", "sd_", "med_", "p25_", "p75_",
                     "ss_mean_", "ss_sd_", "n_firms")
  drop_cols <- unique(unlist(lapply(stat_prefixes, function(p)
    grep(paste0("^", p), names(Train), value = TRUE)
  )))
  Train[, (intersect(drop_cols, names(Train))) := NULL]
  Test[,  (intersect(drop_cols, names(Test)))  := NULL]
  
  ## ── O. Sanity Check ───────────────────────────────────────────────────────
  stopifnot(
    "Train row count changed!" = nrow(Train) == n_train_init,
    "Test row count changed!"  = nrow(Test)  == n_test_init
  )
  
  families <- c("secZ_", "secDev_", "secRank_", "secVol_",
                "secTrend_", "secDiverg_", "secSizeZ_", "sector_")
  na_audit <- data.frame(
    family   = families,
    n_cols   = sapply(families, function(f) sum(startsWith(names(Train), f))),
    na_train = sapply(families, function(f) {
      cols <- grep(paste0("^", f), names(Train), value = TRUE)
      if (!length(cols)) 0L else sum(is.na(Train[, .SD, .SDcols = cols]))
    }),
    na_test  = sapply(families, function(f) {
      cols <- grep(paste0("^", f), names(Test), value = TRUE)
      if (!length(cols)) 0L else sum(is.na(Test[, .SD, .SDcols = cols]))
    })
  )
  message("03C Complete.")
  cat(sprintf("  Train: %d rows x %d cols\n", nrow(Train), ncol(Train)))
  cat(sprintf("  Test : %d rows x %d cols\n", nrow(Test),  ncol(Test)))
  print(na_audit, row.names = FALSE)
  
}, error = function(e) stop("03C Failed: ", e$message))

#==== 03D - Quantile Transformation ===========================================#

tryCatch({
  
  message("--- Starting 03D: Quantile Transformation ---")
  
  if (exists("Quantile_Transform") && Quantile_Transform) {
    
    ## ── A. Exclusion Logic ────────────────────────────────────────────────
    exclude_patterns <- c(
      "^secZ_", "^secSizeZ_", "^secRank_", "^consec_",
      "^sector_", "^time_", "^history_", "^is_", "^has_"
    )
    
    ## Semantic exclusions — "public" removed: confirmed continuous (% ownership)
    semantic_exclude <- c("y", "id", "refdate", "sector", "size", "year", "groupmember")
    
    ## ── B. Build Exclusion Set ────────────────────────────────────────────
    all_cols     <- names(Train)
    numeric_cols <- all_cols[sapply(Train, is.numeric)]
    
    pattern_excluded <- all_cols[vapply(all_cols, function(nm)
      any(vapply(exclude_patterns, function(p) grepl(p, nm), logical(1))),
      logical(1))]
    
    detect_binary <- function(nm) {
      x_obs <- Train[[nm]][!is.na(Train[[nm]])]
      length(x_obs) > 0 && all(x_obs %in% c(0, 1))
    }
    binary_excluded <- all_cols[vapply(all_cols, detect_binary, logical(1))]
    
    detect_bounded01 <- function(nm) {
      x_obs <- Train[[nm]][!is.na(Train[[nm]])]
      length(x_obs) > 0 && !all(x_obs %in% c(0, 1)) &&
        min(x_obs) >= 0 && max(x_obs) <= 1
    }
    bounded_excluded <- if (!exists("TRANSFORM_BOUNDED01") || !TRANSFORM_BOUNDED01) {
      numeric_cols[vapply(numeric_cols, detect_bounded01, logical(1))]
    } else character(0)
    
    exclude_cols      <- unique(c(semantic_exclude, pattern_excluded,
                                  binary_excluded,  bounded_excluded))
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    ## ── C. Exclusion Audit ────────────────────────────────────────────────
    message(sprintf("  Numeric columns        : %d", length(numeric_cols)))
    message(sprintf("  Semantic exclusions    : %d", sum(all_cols %in% semantic_exclude)))
    message(sprintf("  Pattern exclusions     : %d", length(pattern_excluded)))
    message(sprintf("  Binary exclusions      : %d", length(binary_excluded)))
    message(sprintf("  Bounded [0,1] excl.    : %d", length(bounded_excluded)))
    message(sprintf("  Columns to transform   : %d", length(cols_to_transform)))
    message("  Sample TO transform     : ",
            paste(head(cols_to_transform, 8), collapse = ", "))
    message("  Sample NOT transforming : ",
            paste(head(exclude_cols[exclude_cols %in% all_cols], 8), collapse = ", "))
    
    semantic_in_transform <- intersect(semantic_exclude, cols_to_transform)
    if (length(semantic_in_transform) > 0)
      stop("Semantic exclusion cols in transform list: ",
           paste(semantic_in_transform, collapse = ", "))
    
    binary_in_transform <- cols_to_transform[vapply(cols_to_transform,
                                                    detect_binary, logical(1))]
    if (length(binary_in_transform) > 0)
      stop("Binary columns in transform list: ",
           paste(binary_in_transform, collapse = ", "))
    
    ## ── D. Guard ──────────────────────────────────────────────────────────
    if (!exists("QuantileTransformation"))
      stop("QuantileTransformation() not found.")
    if (length(cols_to_transform) == 0)
      stop("No columns selected for transformation.")
    
    ## ── E. Transform Loop ─────────────────────────────────────────────────
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    failed_cols       <- character(0)
    
    for (col in cols_to_transform) {
      if (!col %in% names(Test)) {
        warning("Column missing from Test, skipping: ", col); next
      }
      if (all(is.na(Train[[col]]))) {
        warning("Column entirely NA in Train, skipping: ", col); next
      }
      res <- tryCatch(
        QuantileTransformation(Train[[col]], Test[[col]]),
        error = function(e) {
          message("  Transform failed: ", col, " — ", e$message)
          failed_cols <<- c(failed_cols, col)
          NULL
        }
      )
      if (!is.null(res)) {
        Train_Transformed[[col]] <- res$train
        Test_Transformed[[col]]  <- res$test
      }
    }
    
    ## ── F. Post-Transform Audit ───────────────────────────────────────────
    n_ok <- length(cols_to_transform) - length(failed_cols)
    message(sprintf("  Successfully transformed : %d", n_ok))
    if (length(failed_cols) > 0)
      message("  Failed columns: ", paste(failed_cols, collapse = ", "))
    
    na_delta <- function(before, after, label) {
      d <- sum(is.na(after[,  .SD, .SDcols = cols_to_transform])) -
        sum(is.na(before[, .SD, .SDcols = cols_to_transform]))
      if (d > 0) warning(sprintf("  %s: introduced %d new NAs", label, d))
      d
    }
    message(sprintf("  NA delta — Train: %d | Test: %d",
                    na_delta(Train, Train_Transformed, "Train"),
                    na_delta(Test,  Test_Transformed,  "Test")))
    
    ## ── G. Binary Integrity Check ─────────────────────────────────────────
    ## Check data-driven binary cols + groupmember only.
    ## "public" removed — confirmed continuous, not a binary flag.
    binary_check_cols <- intersect(binary_excluded, names(Train_Transformed))
    structural_binaries <- intersect(c("groupmember"), names(Train_Transformed))
    binary_check_cols <- unique(c(binary_check_cols, structural_binaries))
    binary_check_cols <- binary_check_cols[vapply(binary_check_cols, function(nm) {
      x <- Train_Transformed[[nm]]
      is.numeric(x) || is.integer(x)
    }, logical(1))]
    
    corrupted <- binary_check_cols[vapply(binary_check_cols, function(nm) {
      x <- Train_Transformed[[nm]]
      !all(x[!is.na(x)] %in% c(0, 1))
    }, logical(1))]
    
    if (length(corrupted) > 0)
      stop("CRITICAL: Binary columns corrupted: ", paste(corrupted, collapse = ", "))
    message(sprintf("  Binary integrity check : OK (%d cols verified)",
                    length(binary_check_cols)))
    
  } else {
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    message("  Skipping Quantile Transformation (Quantile_Transform = FALSE).")
  }
  
}, error = function(e) stop("03D Failed: ", e$message))

#==== 03E - Cleanup and Factors ===============================================#

tryCatch({
  
  message("--- Starting 03E: Final Cleanup, Imputation & Validation ---")
  
  if (!is.data.table(Train_Transformed)) setDT(Train_Transformed)
  if (!is.data.table(Test_Transformed))  setDT(Test_Transformed)
  
  ## ── B. Structural NA Imputation ───────────────────────────────────────────
  message("  Imputing structural cold-start NAs...")
  
  imputation_rules <- list(
    list(pattern = "^yoy_",         strategy = "zero"),
    list(pattern = "^accel_",       strategy = "zero"),
    list(pattern = "^momentum_",    strategy = "zero"),
    list(pattern = "^peak_drop_",   strategy = "zero"),
    list(pattern = "^trough_rise_", strategy = "zero"),
    list(pattern = "^expvol_",      strategy = "median"),
    list(pattern = "^secTrend_",    strategy = "zero"),
    list(pattern = "^secDiverg_",   strategy = "zero"),
    list(pattern = "^secDev_",      strategy = "zero"),
    list(pattern = "^secVol_",      strategy = "median"),
    list(pattern = "^expmean_",     strategy = "median"),
    list(pattern = "^dev_expmean_", strategy = "zero")
  )
  
  impute_fixed <- function(col, fill_value) {
    Train_Transformed[is.na(get(col)), (col) := fill_value]
    Test_Transformed[is.na(get(col)),  (col) := fill_value]
  }
  impute_median <- function(col) {
    fill_val <- median(Train_Transformed[[col]], na.rm = TRUE)
    if (is.na(fill_val)) fill_val <- 0
    Train_Transformed[is.na(get(col)), (col) := fill_val]
    Test_Transformed[is.na(get(col)),  (col) := fill_val]
  }
  
  for (rule in imputation_rules) {
    matched_cols <- grep(rule$pattern, names(Train_Transformed), value = TRUE)
    for (col in matched_cols) {
      if (rule$strategy == "zero")   impute_fixed(col, 0)
      if (rule$strategy == "median") impute_median(col)
    }
  }
  
  ## Catch-all for any remaining NAs not covered by rules
  remaining_na_cols <- names(Train_Transformed)[
    vapply(names(Train_Transformed), function(nm)
      is.numeric(Train_Transformed[[nm]]) &&
        (anyNA(Train_Transformed[[nm]]) || anyNA(Test_Transformed[[nm]])),
      logical(1))
  ]
  catchall_cols <- setdiff(
    remaining_na_cols,
    unlist(lapply(imputation_rules, function(r)
      grep(r$pattern, names(Train_Transformed), value = TRUE)))
  )
  if (length(catchall_cols) > 0) {
    message(sprintf("  Catch-all median imputation: %d cols — %s",
                    length(catchall_cols),
                    paste(head(catchall_cols, 6), collapse = ", ")))
    for (col in catchall_cols) impute_median(col)
  }
  
  remaining_na_train <- sum(is.na(Train_Transformed))
  remaining_na_test  <- sum(is.na(Test_Transformed))
  message(sprintf("  Remaining NAs after imputation — Train: %d | Test: %d",
                  remaining_na_train, remaining_na_test))
  
  if (remaining_na_train > 0) {
    na_cols    <- names(Train_Transformed)[sapply(Train_Transformed, anyNA)]
    na_summary <- sort(sapply(na_cols, function(c)
      sum(is.na(Train_Transformed[[c]]))), decreasing = TRUE)
    message("  Columns with remaining NAs:")
    print(na_summary)
  }
  
  ## ── C. Drop Metadata / Leakage / Zero-Variance Columns ───────────────────
  drop_patterns <- c("^id$", "^company_id$", "^row_id$", "refdate", "^year$")
  
  pattern_drop <- names(Train_Transformed)[vapply(names(Train_Transformed), function(nm)
    any(vapply(drop_patterns, function(p) grepl(p, nm), logical(1))),
    logical(1))]
  
  zerovar_drop <- names(Train_Transformed)[vapply(names(Train_Transformed), function(nm) {
    x <- Train_Transformed[[nm]]
    is.numeric(x) && !anyNA(x) && var(x) == 0
  }, logical(1))]
  
  if (length(zerovar_drop) > 0)
    message("  Zero-variance columns dropped: ", paste(zerovar_drop, collapse = ", "))
  
  cols_to_drop <- setdiff(unique(c(pattern_drop, zerovar_drop)), "y")
  
  ## ── D. Align Categorical Columns ──────────────────────────────────────────
  message("  Aligning categorical levels...")
  cat_cols <- names(Train_Transformed)[
    sapply(Train_Transformed, function(x) is.character(x) || is.factor(x))
  ]
  cat_cols <- setdiff(cat_cols, c("y", cols_to_drop))
  
  unseen_levels <- list()
  for (col in cat_cols) {
    Train_Transformed[, (col) := as.factor(get(col))]
    train_levels <- levels(Train_Transformed[[col]])
    new_in_test  <- setdiff(as.character(unique(Test_Transformed[[col]])), train_levels)
    if (length(new_in_test) > 0) {
      unseen_levels[[col]] <- new_in_test
      warning(sprintf("  '%s': %d unseen Test level(s) → mode imputed",
                      col, length(new_in_test)))
    }
    Test_Transformed[, (col) := factor(get(col), levels = train_levels)]
    if (length(new_in_test) > 0) {
      mode_val <- train_levels[which.max(tabulate(
        match(Train_Transformed[[col]], train_levels)))]
      Test_Transformed[is.na(get(col)), (col) := mode_val]
    }
  }
  if (length(unseen_levels) == 0) message("  Categorical levels aligned cleanly.")
  
  ## ── E. Target Enforcement ─────────────────────────────────────────────────
  target_col <- if (exists("TARGET_COL")) TARGET_COL else "y"
  for (dt in list(Train_Transformed, Test_Transformed)) {
    if (target_col %in% names(dt))
      dt[, (target_col) := as.integer(as.character(get(target_col)))]
  }
  
  ## ── F. Build Final Sets ───────────────────────────────────────────────────
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  Train_Final  <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final   <- copy(Test_Transformed[,  ..cols_to_keep])
  
  ## ── G. Validation Report ──────────────────────────────────────────────────
  message("--- 03E Validation Report ---")
  cat(sprintf("  Train_Final : %d rows x %d cols\n", nrow(Train_Final), ncol(Train_Final)))
  cat(sprintf("  Test_Final  : %d rows x %d cols\n", nrow(Test_Final),  ncol(Test_Final)))
  cat(sprintf("  Train default rate : %.3f%% (%d)\n",
              100 * mean(Train_Final[[target_col]]), sum(Train_Final[[target_col]])))
  cat(sprintf("  Test  default rate : %.3f%% (%d)\n",
              100 * mean(Test_Final[[target_col]]),  sum(Test_Final[[target_col]])))
  cat(sprintf("  Train NAs : %d | Test NAs : %d\n",
              sum(is.na(Train_Final)), sum(is.na(Test_Final))))
  
  col_mismatch <- setdiff(names(Train_Final), names(Test_Final))
  if (length(col_mismatch) > 0)
    stop("CRITICAL: Column mismatch: ", paste(col_mismatch, collapse = ", "))
  message("  Column alignment: OK")
  
  all_feature_cols <- setdiff(names(Train_Final), target_col)
  prefixes <- unique(sub("^(([a-zA-Z]+[0-9]*_?[a-zA-Z]*_?)|([a-zA-Z]+)).*", "\\1",
                         all_feature_cols))
  prefixes <- prefixes[nchar(prefixes) > 1]
  family_summary <- data.frame(
    prefix = prefixes,
    n_cols = sapply(prefixes, function(p) sum(startsWith(all_feature_cols, p)))
  )
  family_summary <- family_summary[order(-family_summary$n_cols), ]
  family_summary <- family_summary[family_summary$n_cols > 0, ]
  message("  Feature family breakdown:")
  print(family_summary, row.names = FALSE)
  cat(sprintf("  Total modelling features: %d\n", length(all_feature_cols)))
  message("03E Complete.")
  
}, error = function(e) stop("03E Failed: ", e$message))

#==============================================================================#
#==== 04 - VAE SetUp ==========================================================#
#==============================================================================#

#==== 04A - Data Preparation ==================================================#

tryCatch({
  message("--- Starting 04A: VAE Data Preparation ---")
  
  if (!is.data.table(Train_Final)) setDT(Train_Final)
  if (!is.data.table(Test_Final))  setDT(Test_Final)
  
  ## ── 0. Utility ────────────────────────────────────────────────────────────
  `%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b
  
  ## ── A. Detect Column Roles ────────────────────────────────────────────────
  target_col <- if (exists("TARGET_COL")) TARGET_COL else "y"
  
  meta_patterns <- c("^id$", "^company_id$", "^row_id$", "refdate",
                     "^year$", "^time_index$", "^size$", "^sector$",
                     paste0("^", target_col, "$"))
  
  is_meta       <- function(nm) any(sapply(meta_patterns, function(p) grepl(p, nm)))
  meta_cols_vae <- names(Train_Final)[sapply(names(Train_Final), is_meta)]
  feature_cols  <- setdiff(names(Train_Final), meta_cols_vae)
  
  non_numeric <- feature_cols[!sapply(Train_Final[, ..feature_cols], is.numeric)]
  if (length(non_numeric) > 0)
    stop("Non-numeric feature columns remain: ", paste(non_numeric, collapse = ", "))
  
  message(sprintf("  Excluded meta/raw-cat cols : %d", length(meta_cols_vae)))
  message(sprintf("  Feature cols for VAE       : %d", length(feature_cols)))
  
  ## ── B. Classify Features by Distribution Type ─────────────────────────────
  binary_patterns  <- c("^groupmember$", "^is_", "^has_", "^sector_")
  bounded_patterns <- c("^secRank_")
  
  bin_cols_pattern  <- feature_cols[sapply(feature_cols, function(nm)
    any(sapply(binary_patterns,  function(p) grepl(p, nm))))]
  rank_cols_pattern <- feature_cols[sapply(feature_cols, function(nm)
    any(sapply(bounded_patterns, function(p) grepl(p, nm))))]
  
  remaining_cols <- setdiff(feature_cols, c(bin_cols_pattern, rank_cols_pattern))
  bin_cols_data  <- remaining_cols[sapply(remaining_cols, function(nm) {
    x_obs <- Train_Final[[nm]][!is.na(Train_Final[[nm]])]
    length(x_obs) > 0 && all(x_obs %in% c(0, 1))
  })]
  
  bin_cols  <- as.character(unique(c(bin_cols_pattern, bin_cols_data)) %||% character(0))
  rank_cols <- as.character(rank_cols_pattern %||% character(0))
  cont_cols <- as.character(setdiff(feature_cols, c(bin_cols, rank_cols)) %||% character(0))
  
  classified   <- c(cont_cols, rank_cols, bin_cols)
  unclassified <- setdiff(feature_cols, classified)
  if (length(unclassified) > 0) {
    warning(sprintf("  %d unclassified cols → defaulting to continuous: %s",
                    length(unclassified),
                    paste(head(unclassified, 6), collapse = ", ")))
    cont_cols <- c(cont_cols, as.character(unclassified))
  }
  stopifnot(
    "Feature cols not fully partitioned" =
      length(c(cont_cols, rank_cols, bin_cols)) == length(feature_cols),
    "Overlap: binary ∩ bounded" = length(intersect(bin_cols,  rank_cols)) == 0,
    "Overlap: binary ∩ cont"    = length(intersect(bin_cols,  cont_cols)) == 0,
    "Overlap: bounded ∩ cont"   = length(intersect(rank_cols, cont_cols)) == 0
  )
  
  message(sprintf("  Features: %d total | %d continuous | %d bounded[0,1] | %d binary",
                  length(feature_cols), length(cont_cols),
                  length(rank_cols), length(bin_cols)))
  message(sprintf("  Binary — pattern: %d | data-driven: %d",
                  length(bin_cols_pattern), length(bin_cols_data)))
  
  ## ── C. Repair Corrupted Binary Columns ────────────────────────────────────
  ## Safety net — after 03D fixes this should always report 0 repaired cols.
  repaired <- character(0)
  for (col in bin_cols) {
    train_corrupt <- !all(Train_Final[[col]][!is.na(Train_Final[[col]])] %in% c(0, 1))
    test_corrupt  <- !all(Test_Final[[col]][!is.na(Test_Final[[col]])]   %in% c(0, 1))  # fix: [[col]][...] not [[col][...]]
    if (train_corrupt || test_corrupt) {
      Train_Final[, (col) := as.integer(round(get(col)))]
      Test_Final[,  (col) := as.integer(round(get(col)))]
      repaired <- c(repaired, col)
    }
  }
  if (length(repaired) > 0) {
    message(sprintf("  Repaired %d corrupted binary cols:", length(repaired)))
    message("    ", paste(head(repaired, 10), collapse = ", "),
            if (length(repaired) > 10) " [...]" else "")
  } else {
    message("  Binary integrity: OK — no corrupted cols")
  }
  
  ## ── D. NA Imputation ──────────────────────────────────────────────────────
  residual_rules <- list(
    list(pattern = "^expmean_",     strategy = "median"),
    list(pattern = "^dev_expmean_", strategy = "zero"),
    list(pattern = "^secZ_",        strategy = "zero"),
    list(pattern = "^secSizeZ_",    strategy = "zero"),
    list(pattern = "^secRank_",     strategy = "value", value = 0.5),
    list(pattern = "^secTrend_",    strategy = "zero"),
    list(pattern = "^secDiverg_",   strategy = "zero"),
    list(pattern = "^secDev_",      strategy = "zero"),
    list(pattern = "^secVol_",      strategy = "median")
  )
  
  impute_col <- function(col, strategy, value = NULL) {
    fill <- switch(strategy,
                   "zero"   = 0,
                   "value"  = value,
                   "median" = median(Train_Final[[col]], na.rm = TRUE))
    if (is.na(fill)) fill <- 0
    Train_Final[is.na(get(col)), (col) := fill]
    Test_Final[is.na(get(col)),  (col) := fill]
  }
  
  matched_by_rule <- character(0)
  for (rule in residual_rules) {
    cols <- grep(rule$pattern, feature_cols, value = TRUE)
    cols <- cols[vapply(cols, function(nm)
      anyNA(Train_Final[[nm]]) || anyNA(Test_Final[[nm]]), logical(1))]
    for (col in cols) impute_col(col, rule$strategy, rule$value)
    matched_by_rule <- c(matched_by_rule, cols)
  }
  
  catchall_cols <- setdiff(
    feature_cols[vapply(feature_cols, function(nm)
      anyNA(Train_Final[[nm]]) || anyNA(Test_Final[[nm]]), logical(1))],
    matched_by_rule
  )
  if (length(catchall_cols) > 0) {
    message(sprintf("  Catch-all median imputation: %d cols", length(catchall_cols)))
    for (col in catchall_cols) impute_col(col, "median")
  }
  
  na_post_train <- sum(vapply(feature_cols, function(nm)
    sum(is.na(Train_Final[[nm]])), integer(1)))
  na_post_test  <- sum(vapply(feature_cols, function(nm)
    sum(is.na(Test_Final[[nm]])),  integer(1)))
  if (na_post_train > 0 || na_post_test > 0)
    stop(sprintf("NAs remain — Train: %d | Test: %d", na_post_train, na_post_test))
  message("  NAs after imputation — Train: 0 | Test: 0")
  
  ## ── E. Build VAE Input Matrix ─────────────────────────────────────────────
  col_order  <- c(cont_cols, rank_cols, bin_cols)
  n_cont     <- length(cont_cols)
  n_bounded  <- length(rank_cols)
  n_binary   <- length(bin_cols)
  n_features <- length(col_order)
  
  stopifnot(
    "Duplicate cols in col_order"             = !anyDuplicated(col_order),
    "col_order cols missing from Train_Final" = all(col_order %in% names(Train_Final)),
    "col_order length != feature_cols length" = n_features == length(feature_cols)
  )
  
  vae_train_mat <- as.matrix(Train_Final[, ..col_order])
  vae_test_mat  <- as.matrix(Test_Final[,  ..col_order])
  
  if (n_binary > 0) {
    bin_idx <- (n_cont + n_bounded + 1):n_features
    if (!all(vae_train_mat[, bin_idx] %in% c(0, 1)))
      warning("  Binary block contains non-{0,1} values — check 03D exclusions.")
    else
      message(sprintf("  Binary block verified: %d cols all in {0,1}", n_binary))
  }
  if (sum(is.na(vae_train_mat)) > 0) stop("NAs in final VAE matrix.")
  
  ## ── F. VAE Config ─────────────────────────────────────────────────────────
  vae_config <- if (exists("VAE_CONFIG")) VAE_CONFIG else list(
    encoder_dims     = NULL,
    decoder_dims     = NULL,
    latent_dim       = NULL,
    epochs           = 100L,
    batch_size       = 256L,
    beta             = 1.0,
    lr               = 1e-3,
    temperature      = 0.5,
    patience         = 10L,
    kl_warmup        = TRUE,
    kl_warmup_epochs = 10L
  )
  
  message(sprintf("  β (KL weight) : %.2f", vae_config$beta))
  message(sprintf("  Column split  : %d cont | %d bounded | %d binary",
                  n_cont, n_bounded, n_binary))
  
  ## ── G. Distribution Metadata ──────────────────────────────────────────────
  if (!exists("extracting_distribution"))
    stop("extracting_distribution() not found — check autotab is loaded.")
  if (!exists("set_feat_dist"))
    stop("set_feat_dist() not found — check autotab is loaded.")
  
  ## Pass data.frame with explicit colnames matching col_order — critical for
  ## correct loss tensor graph wiring in VAE_train (missing deps error if wrong)
  feat_dist_input <- as.data.frame(vae_train_mat)
  colnames(feat_dist_input) <- col_order
  
  feat_dist <- extracting_distribution(feat_dist_input)
  set_feat_dist(feat_dist)
  
  ## Sanity checks
  total_params <- sum(feat_dist$num_params)
  message(sprintf("  feat_dist rows        : %d (should == %d features)",
                  nrow(feat_dist), length(col_order)))
  message(sprintf("  total decoder params  : %d", total_params))
  message(sprintf("  feat_dist dist types  : %s",
                  paste(names(sort(table(feat_dist$distribution), decreasing = TRUE)),
                        collapse = ", ")))
  
  if (!is.null(feat_dist$name)) {
    misaligned <- sum(feat_dist$name != col_order)
    if (misaligned > 0)
      stop(sprintf("feat_dist column order misaligned: %d mismatches.", misaligned))
    else
      message("  feat_dist column alignment: OK")
  }
  
  message(sprintf("  feat_dist class  : %s", paste(class(feat_dist), collapse = ", ")))
  message(sprintf("  feat_dist length : %d", length(feat_dist)))
  message("04A Complete.")
  
}, error = function(e) stop("04A Failed: ", e$message))

#==== 04B - VAE Training ======================================================#

tryCatch({
  message("--- Starting 04B: VAE Training ---")
  
  if (!exists("vae_train_mat")) stop("vae_train_mat missing — run 04A first.")
  if (!exists("vae_config"))    stop("vae_config missing — run 04A first.")
  if (!exists("feat_dist"))     stop("feat_dist missing — run 04A first.")
  
  ## ── A. Derive Architecture from feat_dist ─────────────────────────────────
  n_features          <- ncol(vae_train_mat)
  total_output_params <- sum(feat_dist$num_params)
  
  message(sprintf("  total_output_params : %d", total_output_params))
  
  latent_dim  <- max(4L, as.integer(sqrt(n_features)))
  min_last    <- as.integer(ceiling(total_output_params * 1.1))
  
  enc_dims <- c(
    max(as.integer(n_features * 0.75), min_last + 200L),
    max(as.integer(n_features * 0.50), min_last + 100L),
    min_last
  )
  dec_dims <- rev(enc_dims)
  
  make_layer_spec <- function(dims, activation = "relu") {
    lapply(dims, function(u) list("dense", as.integer(u), activation))
  }
  
  encoder_spec <- make_layer_spec(enc_dims)
  decoder_spec <- make_layer_spec(dec_dims)
  
  last_dec_units <- decoder_spec[[length(decoder_spec)]][[2]]
  if (last_dec_units < total_output_params)
    stop(sprintf(
      "Last decoder layer %d < total_output_params %d — increase dec_dims.",
      last_dec_units, total_output_params))
  
  message(sprintf("  Input dims    : %d x %d", nrow(vae_train_mat), n_features))
  message(sprintf("  Encoder dims  : %s → %d (latent)",
                  paste(enc_dims, collapse = " → "), latent_dim))
  message(sprintf("  Decoder dims  : %d → %s",
                  latent_dim, paste(dec_dims, collapse = " → ")))
  message(sprintf("  Last dec layer: %d >= %d required : OK",
                  last_dec_units, total_output_params))
  message(sprintf("  Epochs: %d | Batch: %d | β: %.2f",
                  vae_config$epochs, vae_config$batch_size, vae_config$beta))
  
  ### ── B. Validate feat_dist — DO NOT rebuild, DO NOT clear session ──────────
  ## VAE_train() calls get_feat_dist() internally — set_feat_dist() in 04A
  ## is sufficient. Rebuilding feat_dist here creates a second set of loss
  ## tensors disconnected from the model graph → "missing dependencies" crash.
  ## k_clear_session() would invalidate the tensors registered in 04A → same crash.
  
  stopifnot(
    "feat_dist not set"    = exists("feat_dist"),
    "feat_dist wrong rows" = nrow(feat_dist) == ncol(vae_train_mat)
  )
  message(sprintf("  feat_dist validated: %d rows | %d total params",
                  nrow(feat_dist), sum(feat_dist$num_params)))
  
  ## ── C. Store Resolved Architecture ────────────────────────────────────────
  vae_config$latent_dim   <- latent_dim
  vae_config$encoder_dims <- enc_dims
  vae_config$decoder_dims <- dec_dims
  
  ## ── D. Train ──────────────────────────────────────────────────────────────
  vae_fit <- VAE_train(
    data         = vae_train_mat,
    encoder_info = encoder_spec,
    decoder_info = decoder_spec,
    latent_dim   = latent_dim,
    epoch        = vae_config$epochs,
    batchsize    = vae_config$batch_size,
    beta         = vae_config$beta,
    lr           = vae_config$lr,
    temperature  = vae_config$temperature,
    wait         = vae_config$patience,
    kl_warm      = vae_config$kl_warmup,
    beta_epoch   = vae_config$kl_warmup_epochs,
    Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0
  )
  
  if (is.null(vae_fit) || is.null(vae_fit$trained_model))
    stop("VAE_train() returned NULL.")
  
  ## ── E. Training Diagnostics ───────────────────────────────────────────────
  if (!is.null(vae_fit$history)) {
    history_names <- names(vae_fit$history)
    get_metric    <- function(candidates) {
      nm <- intersect(candidates, history_names)[1]
      if (is.na(nm)) return(NA_real_)
      tail(vae_fit$history[[nm]], 1)
    }
    final_loss  <- get_metric(c("loss", "total_loss"))
    final_recon <- get_metric(c("recon_loss", "reconstruction_loss", "mse_loss"))
    final_kl    <- get_metric(c("kl_loss", "kl_divergence", "kl"))
    
    message(sprintf("  Final total loss : %.4f", final_loss))
    message(sprintf("  Final recon loss : %.4f", final_recon))
    message(sprintf("  Final KL loss    : %.4f", final_kl))
    message(sprintf("  History fields   : %s",   paste(history_names, collapse = ", ")))
    
    if (!is.na(final_kl) && final_kl < 0.01)
      warning("  KL near zero — possible posterior collapse. ",
              "Try increasing β or kl_warmup_epochs.")
  } else {
    message("  No training history returned.")
  }
  
  message("04B Complete.")
  
}, error = function(e) stop("04B Failed: ", e$message))

#==============================================================================#
#==== 05 - VAE Modeling =======================================================#
#==============================================================================#

Use_VAE_Only <- TRUE

tryCatch({

#==== 05A - Strategy A: Latent features (Dimensional reduction) ===============#

tryCatch({
    message("--- Starting 05A: Strategy A - Latent Feature Extraction ---")
    
    ## ── A. Guards ─────────────────────────────────────────────────────────────
    if (!exists("vae_fit"))       stop("vae_fit not found — run 04B first.")
    if (!exists("vae_train_mat")) stop("vae_train_mat not found — run 04A first.")
    if (!exists("vae_test_mat"))  stop("vae_test_mat not found — run 04A first.")
    if (!exists("vae_config"))    stop("vae_config not found — run 04A first.")
    
    latent_dim   <- vae_config$latent_dim
    latent_names <- paste0("l", seq_len(latent_dim))
    target_col   <- if (exists("TARGET_COL")) TARGET_COL else "y"
    Use_VAE_Only <- if (exists("Use_VAE_Only")) Use_VAE_Only else FALSE
    
    ## ── B. Extract Encoder Sub-model from Trained VAE ─────────────────────────
    ## Do NOT call encoder_latent() — its layers are already owned by the trained
    ## TF graph and cannot be added to a new model (missing deps error).
    ## Instead, derive the encoder directly from vae_fit$trained_model.
    
    layer_names <- sapply(vae_fit$trained_model$layers, function(l) l$name)
    message("  Available layers: ", paste(layer_names, collapse = ", "))
    
    mu_name <- layer_names[grepl("z_mean|mu|mean", layer_names, ignore.case = TRUE)][1]
    
    if (!is.na(mu_name)) {
      message(sprintf("  Using mu layer   : %s", mu_name))
      mu_layer  <- vae_fit$trained_model$get_layer(mu_name)
      enc_model <- keras::keras_model(
        inputs  = vae_fit$trained_model$input,
        outputs = mu_layer$output
      )
    } else {
      message("  No named mu layer found — will use full model output and slice.")
      enc_model <- NULL
    }
    
    ## ── C. Helper: encode matrix → latent data.frame ──────────────────────────
    encode_data <- function(model, mat, latent_dim) {
      raw <- predict(model, mat)
      if (is.list(raw)) raw <- raw[[1]]
      if (ncol(raw) > latent_dim) raw <- raw[, seq_len(latent_dim), drop = FALSE]
      as.data.frame(raw)
    }
    
    ## ── D. Encode Train & Test ────────────────────────────────────────────────
    if (!is.null(enc_model)) {
      z_train <- encode_data(enc_model, vae_train_mat, latent_dim)
      z_test  <- encode_data(enc_model, vae_test_mat,  latent_dim)
    } else {
      ## Fallback: full model predict — mu is typically first latent_dim cols
      raw_train <- predict(vae_fit$trained_model, vae_train_mat)
      raw_test  <- predict(vae_fit$trained_model, vae_test_mat)
      z_train <- as.data.frame(
        if (is.list(raw_train)) raw_train[[1]] else raw_train[, seq_len(latent_dim), drop = FALSE])
      z_test  <- as.data.frame(
        if (is.list(raw_test))  raw_test[[1]]  else raw_test[,  seq_len(latent_dim), drop = FALSE])
    }
    
    colnames(z_train) <- latent_names
    colnames(z_test)  <- latent_names
    
    ## ── E. Reconstruction Error as Anomaly Feature ────────────────────────────
    ## VAE_reconstruct() does not exist in autotab — reconstruct via full model.
    ## The full trained model output is the reconstruction; take first element
    ## if multi-output, then clip to ncol(vae_train_mat) to drop parameter cols.
    
    reconstruct_vae <- function(vae_fit, input_mat) {
      recon_raw <- predict(vae_fit$trained_model, input_mat)
      if (is.list(recon_raw)) recon_raw[[1]] else recon_raw
    }
    
    recon_train <- reconstruct_vae(vae_fit, vae_train_mat)
    recon_test  <- reconstruct_vae(vae_fit, vae_test_mat)
    
    ## Clip to input feature count (decoder may output extra distribution params)
    n_cols      <- ncol(vae_train_mat)
    recon_train <- recon_train[, seq_len(n_cols), drop = FALSE]
    recon_test  <- recon_test[,  seq_len(n_cols), drop = FALSE]
    
    recon_err_train <- rowMeans((vae_train_mat - recon_train)^2)
    recon_err_test  <- rowMeans((vae_test_mat  - recon_test)^2)
    
    ## ── F. Latent Space Diagnostics ───────────────────────────────────────────
    dim_variance <- apply(z_train, 2, var)
    collapsed    <- sum(dim_variance < 1e-3)
    if (collapsed > 0)
      warning(sprintf("  %d latent dim(s) may have collapsed (var < 1e-3).", collapsed),
              " Consider increasing β or reducing latent_dim.")
    
    message(sprintf("  Latent dim variances — min: %.4f | mean: %.4f | max: %.4f",
                    min(dim_variance), mean(dim_variance), max(dim_variance)))
    message(sprintf("  Recon error Train — mean: %.4f | p95: %.4f",
                    mean(recon_err_train), quantile(recon_err_train, 0.95)))
    message(sprintf("  Recon error Test  — mean: %.4f | p95: %.4f",
                    mean(recon_err_test),  quantile(recon_err_test,  0.95)))
    
    ## ── G. Assemble Modelling Sets ────────────────────────────────────────────
    base_train <- if (Use_VAE_Only) Train_Final[, .SD, .SDcols = target_col] else copy(Train_Final)
    base_test  <- if (Use_VAE_Only) Test_Final[,  .SD, .SDcols = target_col] else copy(Test_Final)
    
    Strategy_A_Train <- cbind(
      base_train,
      z_train,
      vae_recon_error = recon_err_train
    )
    Strategy_A_Test  <- cbind(
      base_test,
      z_test,
      vae_recon_error = recon_err_test
    )
    
    ## ── H. Sanity Checks ──────────────────────────────────────────────────────
    stopifnot(
      "Row mismatch Train" = nrow(Strategy_A_Train) == nrow(Train_Final),
      "Row mismatch Test"  = nrow(Strategy_A_Test)  == nrow(Test_Final),
      "NAs in Train"       = sum(is.na(Strategy_A_Train)) == 0,
      "NAs in Test"        = sum(is.na(Strategy_A_Test))  == 0,
      "Col mismatch"       = all(colnames(Strategy_A_Train) == colnames(Strategy_A_Test))
    )
    
    message(sprintf("  Mode            : %s", ifelse(Use_VAE_Only, "latent only", "augmented")))
    message(sprintf("  Strategy_A_Train: %d rows x %d cols", nrow(Strategy_A_Train), ncol(Strategy_A_Train)))
    message(sprintf("  Strategy_A_Test : %d rows x %d cols", nrow(Strategy_A_Test),  ncol(Strategy_A_Test)))
    message("05A Complete.")
    
  }, error = function(e) stop("05A Failed: ", e$message))
  
#==== 05B - Strategy B: Anomaly Score =========================================#

tryCatch({
    message("--- Starting 05B: Strategy B - Anomaly Score ---")
    
    ## ── A. Guards ─────────────────────────────────────────────────────────────
    if (!exists("vae_fit"))       stop("vae_fit not found — run 04B first.")
    if (!exists("vae_train_mat")) stop("vae_train_mat not found — run 04A first.")
    if (!exists("vae_test_mat"))  stop("vae_test_mat not found — run 04A first.")
    if (!exists("vae_config"))    stop("vae_config not found — run 04A first.")
    if (!exists("n_cont"))        stop("n_cont not found — run 04A first.")
    if (!exists("n_bounded"))     stop("n_bounded not found — run 04A first.")
    if (!exists("n_binary"))      stop("n_binary not found — run 04A first.")
    
    target_col   <- if (exists("TARGET_COL")) TARGET_COL else "y"
    Use_VAE_Only <- if (exists("Use_VAE_Only")) Use_VAE_Only else FALSE
    
    ## ── B. Reconstruct via Full VAE ───────────────────────────────────────────
    ## Predict returns a list when the decoder has multiple output heads
    ## (one per variable type in mixed VAEs). We handle both cases.
    
    reconstruct <- function(input_mat) {
      raw <- predict(vae_fit$trained_model, input_mat)
      as.matrix(if (is.list(raw)) raw[[1]] else raw)     ## take first head (continuous)
    }
    
    recon_train <- reconstruct(vae_train_mat)
    recon_test  <- reconstruct(vae_test_mat)
    
    ## ── C. Composite Anomaly Score ────────────────────────────────────────────
    ## Scores are computed separately per variable type and then combined.
    ## This avoids scale dominance by any single variable family.
    ##
    ## Continuous  → MSE  (Gaussian reconstruction loss)
    ## Bounded[0,1]→ MSE  (same; Beta loss not available in base autotab)
    ## Binary      → BCE  (Bernoulli reconstruction loss)
    ##
    ## Column layout in vae_*_mat (set in 04A):
    ##   [1 : n_cont]                        → continuous
    ##   [n_cont+1 : n_cont+n_bounded]       → bounded [0,1]
    ##   [n_cont+n_bounded+1 : n_features]   → binary
    
    anomaly_score <- function(input_mat, recon_mat) {
      
      score <- numeric(nrow(input_mat))
      
      ## Continuous + bounded: MSE — averaged within family then summed
      n_cb <- n_cont + n_bounded
      if (n_cb > 0) {
        idx        <- seq_len(n_cb)
        sq_err     <- (input_mat[, idx, drop = FALSE] -
                         recon_mat[, idx, drop = FALSE])^2
        sq_err     <- pmin(sq_err, 1e6)                  ## cap outliers, not Inf replacement
        score      <- score + rowMeans(sq_err)            ## mean across features, not sum
      }
      
      ## Binary: BCE — averaged across binary features
      if (n_binary > 0) {
        idx        <- (n_cb + 1):(n_cb + n_binary)
        y_true     <- input_mat[, idx, drop = FALSE]
        y_pred     <- pmax(pmin(recon_mat[, idx, drop = FALSE],
                                1 - 1e-7), 1e-7)          ## clip for log stability
        bce        <- -(y_true * log(y_pred) +
                          (1 - y_true) * log(1 - y_pred))
        score      <- score + rowMeans(bce)
      }
      
      score
    }
    
    score_train <- anomaly_score(vae_train_mat, recon_train)
    score_test  <- anomaly_score(vae_test_mat,  recon_test)
    
    ## ── D. Score Diagnostics ──────────────────────────────────────────────────
    ## Compare score distributions between defaulters and non-defaulters.
    ## A well-trained VAE should assign higher anomaly scores to defaulters
    ## — if not, the VAE has not learned a useful manifold for credit risk.
    
    y_train <- Train_Final[[target_col]]
    
    score_diag <- data.frame(
      group     = c("Non-default", "Default"),
      mean      = c(mean(score_train[y_train == 0]), mean(score_train[y_train == 1])),
      median    = c(median(score_train[y_train == 0]), median(score_train[y_train == 1])),
      p95       = c(quantile(score_train[y_train == 0], 0.95),
                    quantile(score_train[y_train == 1], 0.95))
    )
    
    message("  Anomaly score distribution by label:")
    print(score_diag, row.names = FALSE, digits = 4)
    
    ## Rank-order check: defaulters should have higher mean score
    if (score_diag$mean[2] <= score_diag$mean[1])
      warning("  Default mean score <= non-default mean score. ",
              "VAE may not have learned a default-informative manifold. ",
              "Consider re-tuning β or latent_dim.")
    
    ## Score AUC as a quick standalone signal check
    if (requireNamespace("pROC", quietly = TRUE)) {
      auc_val <- pROC::auc(pROC::roc(y_train, score_train, quiet = TRUE))
      message(sprintf("  Anomaly score AUC (Train) : %.4f", auc_val))
      if (auc_val < 0.55)
        warning("  Anomaly score AUC < 0.55 — score carries minimal credit signal.")
    }
    
    ## ── E. Assemble Modelling Sets ────────────────────────────────────────────
    ## Use_VAE_Only = TRUE  → target + anomaly score only
    ## Use_VAE_Only = FALSE → original features + anomaly score appended
    
    base_train <- if (Use_VAE_Only) Train_Final[, .SD, .SDcols = target_col] else copy(Train_Final)
    base_test  <- if (Use_VAE_Only) Test_Final[,  .SD, .SDcols = target_col] else copy(Test_Final)
    
    Strategy_B_Train <- cbind(base_train, anomaly_score = score_train)
    Strategy_B_Test  <- cbind(base_test,  anomaly_score = score_test)
    
    ## ── F. Sanity Checks ──────────────────────────────────────────────────────
    stopifnot(
      "Row mismatch Train" = nrow(Strategy_B_Train) == nrow(Train_Final),
      "Row mismatch Test"  = nrow(Strategy_B_Test)  == nrow(Test_Final),
      "NAs in Train score" = sum(is.na(Strategy_B_Train$anomaly_score)) == 0,
      "NAs in Test score"  = sum(is.na(Strategy_B_Test$anomaly_score))  == 0,
      "Inf in Train score" = sum(is.infinite(Strategy_B_Train$anomaly_score)) == 0,
      "Inf in Test score"  = sum(is.infinite(Strategy_B_Test$anomaly_score))  == 0
    )
    
    message(sprintf("  Mode             : %s", ifelse(Use_VAE_Only, "anomaly score only", "augmented")))
    message(sprintf("  Strategy_B_Train : %d rows x %d cols", nrow(Strategy_B_Train), ncol(Strategy_B_Train)))
    message(sprintf("  Strategy_B_Test  : %d rows x %d cols", nrow(Strategy_B_Test),  ncol(Strategy_B_Test)))
    message("05B Complete.")
    
  }, error = function(e) stop("05B Failed: ", e$message))
  
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