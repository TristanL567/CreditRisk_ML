#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

## Last changed: 2026-02-04 | Tristan Leiter

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
  
  # ── Step 2: Sanity checks ────────────────────────────────────────────────────
  stopifnot("Firm-level leakage detected!" = length(intersect(train_ids, test_ids)) == 0)
  stopifnot("Row loss detected!"           = nrow(Train) + nrow(Test) == nrow(Data))
  
  message(sprintf("Train: %d firms | %d rows | %.1f%% default rate",
                  length(train_ids), nrow(Train), 100 * mean(Train$y)))
  message(sprintf("Test:  %d firms | %d rows | %.1f%% default rate",
                  length(test_ids),  nrow(Test),  100 * mean(Test$y)))
  
}, error = function(e) stop("03B Failed: ", e$message))

#==== 03D - Quantile Transformation ===========================================#

tryCatch({
  
  message("--- Starting 03D: Quantile Transformation ---")
  
  if (exists("Quantile_Transform") && Quantile_Transform) {
    
    ## ── A. Pattern-Based Exclusion Logic ──────────────────────────────────
    ## Instead of hardcoding column names, define exclusion rules by:
    ##   (1) exact semantic role   → detected programmatically
    ##   (2) naming convention     → regex patterns
    ##   (3) statistical property  → detected from data
    
    detect_binary    <- function(x) is.numeric(x) && all(x %in% c(0, 1, NA))
    detect_bounded01 <- function(x) is.numeric(x) && !all(x %in% c(0, 1, NA)) &&
      min(x, na.rm = TRUE) >= 0 &&
      max(x, na.rm = TRUE) <= 1
    
    ## Patterns that mark already-scaled / ordinal / meta columns
    ## Add new prefixes here as pipeline grows — no other edits needed.
    exclude_patterns <- c(
      "^secZ_",        ## Sector Z-scores          (already standardised)
      "^secSizeZ_",    ## Size x Sector Z-scores   (already standardised)
      "^secRank_",     ## Sector percentile ranks  (bounded [0,1])
      "^consec_",      ## Consecutive decline counters (ordinal/count)
      "^sector_",      ## Sector dummies            (binary)
      "^time_",        ## Time index / metadata
      "^history_"      ## History length            (count/ordinal)
    )
    
    ## Fixed semantic exclusions — always exclude regardless of naming
    semantic_exclude <- c("y", "id", "refdate", "sector", "size", "year")
    
    ## ── B. Build Exclusion Set ─────────────────────────────────────────────
    all_cols     <- names(Train)
    numeric_cols <- all_cols[sapply(Train, is.numeric)]
    
    ## Pattern matches
    pattern_excluded <- all_cols[sapply(all_cols, function(nm)
      any(sapply(exclude_patterns, function(p) grepl(p, nm)))
    )]
    
    ## Data-driven: binary columns ({0,1} only)
    binary_excluded <- numeric_cols[sapply(numeric_cols, function(nm)
      detect_binary(Train[[nm]])
    )]
    
    ## Data-driven: already bounded [0,1] but not binary → near-identity transform
    ## Keep these out by default; set TRANSFORM_BOUNDED01 = TRUE to include them.
    bounded_excluded <- if (!exists("TRANSFORM_BOUNDED01") || !TRANSFORM_BOUNDED01) {
      numeric_cols[sapply(numeric_cols, function(nm) detect_bounded01(Train[[nm]]))]
    } else character(0)
    
    exclude_cols <- unique(c(semantic_exclude, pattern_excluded,
                             binary_excluded, bounded_excluded))
    
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    ## ── C. Exclusion Audit ─────────────────────────────────────────────────
    message(sprintf("  Numeric columns        : %d", length(numeric_cols)))
    message(sprintf("  Semantic exclusions    : %d", sum(numeric_cols %in% semantic_exclude)))
    message(sprintf("  Pattern exclusions     : %d", sum(numeric_cols %in% pattern_excluded)))
    message(sprintf("  Binary exclusions      : %d", length(binary_excluded)))
    message(sprintf("  Bounded [0,1] excl.    : %d", length(bounded_excluded)))
    message(sprintf("  Columns to transform   : %d", length(cols_to_transform)))
    
    ## Spot-check: show a sample of what IS and IS NOT being transformed
    message("  Sample cols TO transform     : ",
            paste(head(cols_to_transform, 8), collapse = ", "))
    message("  Sample cols NOT transforming : ",
            paste(head(exclude_cols[exclude_cols %in% numeric_cols], 8), collapse = ", "))
    
    ## ── D. Guard ──────────────────────────────────────────────────────────
    if (!exists("QuantileTransformation"))
      stop("QuantileTransformation() not found. Check Functions_Directory was sourced.")
    
    if (length(cols_to_transform) == 0)
      stop("No columns selected for transformation — check exclusion logic.")
    
    ## ── E. Transform Loop ─────────────────────────────────────────────────
    ## Fit on Train only; apply same mapping to Test.
    
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
    
    ## ── F. Post-Transform Audit ────────────────────────────────────────────
    n_ok <- length(cols_to_transform) - length(failed_cols)
    message(sprintf("  Successfully transformed : %d", n_ok))
    if (length(failed_cols) > 0)
      message("  Failed columns: ", paste(failed_cols, collapse = ", "))
    
    ## NA delta — transformation must not introduce new NAs
    na_delta <- function(before, after, label) {
      d <- sum(is.na(after[, .SD, .SDcols = cols_to_transform])) -
        sum(is.na(before[, .SD, .SDcols = cols_to_transform]))
      if (d > 0) warning(sprintf("  %s: transformation introduced %d new NAs", label, d))
      d
    }
    
    d_train <- na_delta(Train, Train_Transformed, "Train")
    d_test  <- na_delta(Test,  Test_Transformed,  "Test")
    message(sprintf("  NA delta — Train: %d | Test: %d", d_train, d_test))
    
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
  ## Rules defined as a list of (pattern → strategy) pairs.
  ## strategy = "zero"   : fill with 0  (no-change / no-signal semantics)
  ## strategy = "median" : fill with Train median (neutral / typical-firm)
  ## To add a new feature family: add one entry here, nothing else changes.
  
  message("  Imputing structural cold-start NAs...")
  
  imputation_rules <- list(
    list(pattern = "^yoy_",        strategy = "zero"),    ## 1st diff cold-start
    list(pattern = "^accel_",      strategy = "zero"),    ## 2nd diff cold-start
    list(pattern = "^momentum_",   strategy = "zero"),    ## no signal at t=1
    list(pattern = "^peak_drop_",  strategy = "zero"),    ## firm is at own peak
    list(pattern = "^trough_rise_",strategy = "zero"),    ## firm is at own trough
    list(pattern = "^expvol_",     strategy = "median"),  ## 0 = "perfectly stable" → misleading
    list(pattern = "^secTrend_",   strategy = "zero"),    ## no sector trend yet
    list(pattern = "^secDiverg_",  strategy = "zero"),    ## depends on yoy + secTrend
    list(pattern = "^secDev_",     strategy = "zero"),    ## sector deviation
    list(pattern = "^secVol_",     strategy = "median")   ## sector vol: same reasoning as expvol
  )
  
  impute_fixed <- function(col, fill_value) {
    Train_Transformed[is.na(get(col)), (col) := fill_value]
    Test_Transformed[is.na(get(col)),  (col) := fill_value]
  }
  
  impute_median <- function(col) {
    fill_val <- median(Train_Transformed[[col]], na.rm = TRUE)
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
  
  ## Post-imputation NA summary
  remaining_na_train <- sum(is.na(Train_Transformed))
  remaining_na_test  <- sum(is.na(Test_Transformed))
  message(sprintf("  Remaining NAs after imputation — Train: %d | Test: %d",
                  remaining_na_train, remaining_na_test))
  
  if (remaining_na_train > 0) {
    na_cols     <- names(Train_Transformed)[sapply(Train_Transformed, anyNA)]
    na_summary  <- sort(sapply(na_cols, function(c) sum(is.na(Train_Transformed[[c]]))),
                        decreasing = TRUE)
    message("  Columns with remaining NAs in Train:")
    print(na_summary)
  }
  
  ## ── C. Identifier / Leakage Column Detection ──────────────────────────────
  ## Drop columns that match identifier patterns OR are constant (zero-variance).
  ## Explicit list kept minimal — pattern + variance test covers the rest.
  
  ## Patterns that mark pure metadata / leakage
  drop_patterns <- c(
    "^id$", "^company_id$", "^row_id$",   ## Identifiers (exact match)
    "refdate",                              ## Date columns
    "^year$"                               ## Year already encoded in sector stats
  )
  
  pattern_drop <- names(Train_Transformed)[sapply(names(Train_Transformed), function(nm)
    any(sapply(drop_patterns, function(p) grepl(p, nm)))
  )]
  
  ## Zero-variance columns — carry no information, drop automatically
  zerovar_drop <- names(Train_Transformed)[sapply(names(Train_Transformed), function(nm) {
    x <- Train_Transformed[[nm]]
    is.numeric(x) && !anyNA(x) && var(x) == 0
  })]
  
  if (length(zerovar_drop) > 0)
    message("  Zero-variance columns dropped: ", paste(zerovar_drop, collapse = ", "))
  
  cols_to_drop <- unique(c(pattern_drop, zerovar_drop))
  cols_to_drop <- setdiff(cols_to_drop, "y")   ## Never drop target
  
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
    
    new_in_test <- setdiff(as.character(unique(Test_Transformed[[col]])), train_levels)
    
    if (length(new_in_test) > 0) {
      unseen_levels[[col]] <- new_in_test
      warning(sprintf("  '%s': %d Test level(s) unseen in Train → imputed with mode: %s",
                      col, length(new_in_test), paste(new_in_test, collapse = ", ")))
    }
    
    Test_Transformed[, (col) := factor(get(col), levels = train_levels)]
    
    if (length(new_in_test) > 0) {
      mode_val <- train_levels[which.max(tabulate(match(Train_Transformed[[col]], train_levels)))]
      Test_Transformed[is.na(get(col)), (col) := mode_val]
    }
  }
  
  if (length(unseen_levels) == 0) message("  All categorical levels aligned cleanly.")
  
  ## ── E. Target Variable Enforcement ────────────────────────────────────────
  ## Detect target column dynamically — defaults to "y" but respects TARGET_COL
  ## if defined upstream (allows pipeline reuse on different datasets).
  
  target_col <- if (exists("TARGET_COL")) TARGET_COL else "y"
  
  for (dt in list(Train_Transformed, Test_Transformed)) {
    if (target_col %in% names(dt)) {
      dt[, (target_col) := as.integer(as.character(get(target_col)))]
    }
  }
  
  ## ── F. Build Final Modelling Sets ─────────────────────────────────────────
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  
  Train_Final <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final  <- copy(Test_Transformed[,  ..cols_to_keep])
  
  ## ── G. Final Validation Report ────────────────────────────────────────────
  message("--- 03E Validation Report ---")
  
  cat(sprintf("  Train_Final : %d rows x %d cols\n", nrow(Train_Final), ncol(Train_Final)))
  cat(sprintf("  Test_Final  : %d rows x %d cols\n", nrow(Test_Final),  ncol(Test_Final)))
  
  cat(sprintf("  Train default rate : %.3f%% (%d defaults)\n",
              100 * mean(Train_Final[[target_col]]), sum(Train_Final[[target_col]])))
  cat(sprintf("  Test  default rate : %.3f%% (%d defaults)\n",
              100 * mean(Test_Final[[target_col]]),  sum(Test_Final[[target_col]])))
  
  cat(sprintf("  Train NAs : %d\n", sum(is.na(Train_Final))))
  cat(sprintf("  Test  NAs : %d\n", sum(is.na(Test_Final))))
  
  ## Column alignment
  col_mismatch <- setdiff(names(Train_Final), names(Test_Final))
  if (length(col_mismatch) > 0) {
    stop("CRITICAL: Column mismatch Train_Final vs Test_Final: ",
         paste(col_mismatch, collapse = ", "))
  } else {
    message("  Column alignment: OK")
  }
  
  ## Feature family summary — fully dynamic, no hardcoded family list
  ## Groups columns by shared prefix (up to first "_" or digit boundary)
  all_feature_cols <- setdiff(names(Train_Final), target_col)
  
  prefixes <- unique(sub("^(([a-zA-Z]+[0-9]*_?[a-zA-Z]*_?)|([a-zA-Z]+)).*", "\\1",
                         all_feature_cols))
  prefixes <- prefixes[nchar(prefixes) > 1]   ## drop trivial single-char matches
  
  family_summary <- data.frame(
    prefix = prefixes,
    n_cols = sapply(prefixes, function(p)
      sum(startsWith(all_feature_cols, p)))
  )
  family_summary <- family_summary[order(-family_summary$n_cols), ]
  family_summary <- family_summary[family_summary$n_cols > 0, ]
  
  message("  Feature family breakdown:")
  print(family_summary, row.names = FALSE)
  cat(sprintf("  Total modelling features (excl. target): %d\n", length(all_feature_cols)))
  
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