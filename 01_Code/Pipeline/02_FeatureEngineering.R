#==============================================================================#
#==== 02_FeatureEngineering.R =================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Build the full feature set from the clean panel produced by 01_Data.R.
#   Stage order (leakage-free):
#     A  Firm-level split        (OoS or OoT, controlled by SPLIT_MODE)
#     B  Time-series dynamics    (computed separately on Train and Test)
#     C  Sector deviation features (fit on Train, applied to Test)
#     D  Quantile transformation  (fit on Train, applied to Test)
#     E  Final cleanup, imputation, validation, and .rds save
#
# INPUTS  (from config.R / 01_Data.R):
#   Data                          clean panel from 01_Data.R
#   SPLIT_MODE, OOT_N_YEARS       split configuration
#   SEED, TARGET_COL              reproducibility and target
#   STRAT_VARS, TRAIN_SIZE        OoS split parameters
#   QUANTILE_TRANSFORM,
#   TRANSFORM_BOUNDED01           transformation flags
#   MIN_SS_FIRMS                  sparsity warning threshold
#
# OUTPUTS (saved via get_split_path(), suffixed _OoS or _OoT):
#   Train_Final, Test_Final       modelling-ready data.tables (no id, no NAs)
#   train_id_vec, test_id_vec     firm-id vectors aligned to final row order
#
# KEY DESIGN DECISIONS:
#   - Split comes FIRST (02A) so TS features and sector stats are computed
#     strictly on Train-only data — no information from Test firms leaks in.
#   - OoT excludes ALL observations of a test-period firm from Train, not
#     just the last-year rows (firm-level exclusion, strictest setup).
#   - consec_decline_ uses vectorised rle() instead of a row-level for-loop.
#   - Target enforcement operates directly on each data.table, not via a list
#     copy (bug fix: list iteration with := modifies local copies only).
#
#==============================================================================#


#==============================================================================#
#==== A - Firm-Level Split (OoS or OoT) =======================================#
#==============================================================================#

tryCatch({
  
  message(sprintf("--- Starting 02A: Firm-Level Split  [mode = %s] ---", SPLIT_MODE))
  
  Data <- as.data.table(Data)
  setorder(Data, id, refdate)
  
  ## year column — used by sector stats in 02C; derived here once.
  if (!"year" %in% names(Data))
    Data[, year := data.table::year(refdate)]
  
  ## ── Out-of-Sample split ────────────────────────────────────────────────────
  if (SPLIT_MODE == "OoS") {
    
    ## Firm-level profile for stratification.
    ## y_ever = 1 if the firm ever defaults (correct panel-level label).
    firm_profile <- Data[,
                         .(y_ever = as.integer(any(get(TARGET_COL) == 1L)),
                           sector = first(sector),
                           size   = first(size)),
                         by = id
    ]
    firm_profile[, Strat_Key := interaction(
      firm_profile[, .SD, .SDcols = STRAT_VARS], drop = TRUE
    )]
    
    ## Guard: createDataPartition requires >= 2 members per stratum.
    strat_counts <- firm_profile[, .N, by = Strat_Key]
    small_strata <- strat_counts[N < 2L, Strat_Key]
    
    if (length(small_strata) > 0L) {
      n_dropped <- firm_profile[Strat_Key %in% small_strata, .N]
      warning(sprintf(
        "Dropping %d singleton stratum/a (%d firm(s)): %s",
        length(small_strata), n_dropped,
        paste(as.character(small_strata), collapse = ", ")
      ))
      firm_profile <- firm_profile[!Strat_Key %in% small_strata]
      firm_profile[, Strat_Key := droplevels(Strat_Key)]
    }
    
    set.seed(SEED)
    train_idx  <- caret::createDataPartition(
      y = firm_profile$Strat_Key, p = TRAIN_SIZE, list = FALSE, times = 1L
    )
    train_ids  <- firm_profile$id[ train_idx]
    test_ids   <- firm_profile$id[-train_idx]
    
    ## ── Out-of-Time split ──────────────────────────────────────────────────────
  } else {
    
    ## Cutoff: the earliest refdate in the last OOT_N_YEARS of the panel.
    ## Firms whose LAST refdate >= cutoff go entirely to Test.
    max_date <- max(Data$refdate)
    cutoff   <- max_date - lubridate::years(OOT_N_YEARS)
    
    message(sprintf("  OoT cutoff: %s  (last %d year(s))",
                    format(cutoff, "%Y-%m-%d"), OOT_N_YEARS))
    
    firm_last_date <- Data[, .(last_refdate = max(refdate)), by = id]
    test_ids  <- firm_last_date[last_refdate >= cutoff, id]
    train_ids <- firm_last_date[last_refdate <  cutoff, id]
    
    if (length(test_ids) == 0L)
      stop("OoT split produced an empty Test set — increase OOT_N_YEARS.")
    if (length(train_ids) == 0L)
      stop("OoT split produced an empty Train set — decrease OOT_N_YEARS.")
  }
  
  ## ── Leakage check ──────────────────────────────────────────────────────────
  stopifnot(
    "Firm-level leakage: id appears in both Train and Test!" =
      length(intersect(train_ids, test_ids)) == 0L
  )
  
  Train <- Data[id %in% train_ids]
  Test  <- Data[id %in% test_ids]
  
  ## Row conservation check (only rows belonging to assigned firms).
  rows_assigned <- nrow(Data[id %in% c(train_ids, test_ids)])
  stopifnot("Row loss in split!" = nrow(Train) + nrow(Test) == rows_assigned)
  
  message(sprintf("  Train: %d firms | %d rows | %.3f%% default rate",
                  length(train_ids), nrow(Train), 100 * mean(Train[[TARGET_COL]])))
  message(sprintf("  Test : %d firms | %d rows | %.3f%% default rate",
                  length(test_ids),  nrow(Test),  100 * mean(Test[[TARGET_COL]])))
  message("--- 02A complete ---")
  
}, error = function(e) stop("02A failed: ", e$message))


#==============================================================================#
#==== B - Time-Series Dynamics ================================================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02B: Time-Series Dynamics ---")
  
  ## ── Helper: build all TS features for one data.table in-place ─────────────
  build_ts_features <- function(DT, label = "") {
    
    setorder(DT, id, refdate)
    
    ## Detect base ratio columns (numeric, not metadata).
    meta_cols <- c("id", "refdate", "sector", "size", "groupmember",
                   "public", "y", "year")
    all_ratios <- setdiff(names(DT)[sapply(DT, is.numeric)], meta_cols)
    
    message(sprintf("  [%s] Base ratio columns: %d", label, length(all_ratios)))
    
    ## ── Direction classification (higher=better vs lower=better) ────────────
    if (exists("PEAK_RATIOS") && exists("TROUGH_RATIOS")) {
      peak_ratios   <- intersect(PEAK_RATIOS,   all_ratios)
      trough_ratios <- intersect(TROUGH_RATIOS, all_ratios)
    } else {
      r_med  <- sapply(all_ratios, function(r) median(DT[[r]], na.rm = TRUE))
      r_min  <- sapply(all_ratios, function(r) min(DT[[r]],    na.rm = TRUE))
      r_max  <- sapply(all_ratios, function(r) max(DT[[r]],    na.rm = TRUE))
      b01    <- r_min >= 0 & r_max <= 1
      trough_ratios <- all_ratios[ b01 & r_med > 0.5]
      peak_ratios   <- all_ratios[!(b01 & r_med > 0.5)]
    }
    consec_ratios <- if (exists("CONSEC_RATIOS")) intersect(CONSEC_RATIOS, all_ratios) else peak_ratios
    
    ## ── Panel tracking ────────────────────────────────────────────────────────
    DT[, time_index     := seq_len(.N),            by = id]
    DT[, is_mature      := fifelse(time_index >= 3, 1L, 0L)]
    DT[, history_length := .N,                     by = id]
    DT[, has_history    := fifelse(time_index > 1,  1L, 0L)]
    
    ## ── YoY (period-over-period) ──────────────────────────────────────────────
    DT[, paste0("yoy_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, 1L, type = "lag")),
       by = id, .SDcols = all_ratios]
    
    ## ── Acceleration (2nd difference) ────────────────────────────────────────
    yoy_cols <- paste0("yoy_", all_ratios)
    DT[, paste0("accel_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, 1L, type = "lag")),
       by = id, .SDcols = yoy_cols]
    
    ## ── Expanding mean ────────────────────────────────────────────────────────
    DT[, paste0("expmean_", all_ratios) := lapply(.SD, cummean),
       by = id, .SDcols = all_ratios]
    
    ## ── Deviation from expanding mean ─────────────────────────────────────────
    expmean_cols <- paste0("expmean_", all_ratios)
    for (i in seq_along(all_ratios))
      DT[, paste0("dev_expmean_", all_ratios[i]) :=
           get(all_ratios[i]) - get(expmean_cols[i])]
    
    ## ── Expanding volatility ──────────────────────────────────────────────────
    DT[, paste0("expvol_", all_ratios) :=
         lapply(.SD, function(x) {
           n   <- seq_along(x)
           mu  <- cummean(x)
           v   <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
           fifelse(n < 2L, NA_real_, sqrt(pmax(0, v)))
         }),
       by = id, .SDcols = all_ratios]
    
    ## ── Peak deterioration (peak ratios) ─────────────────────────────────────
    if (length(peak_ratios) > 0L) {
      DT[, paste0("peak_drop_", peak_ratios) :=
           lapply(.SD, function(x) x - cummax(x)),
         by = id, .SDcols = peak_ratios]
      DT[time_index == 1L,
         paste0("peak_drop_", peak_ratios) :=
           lapply(seq_along(peak_ratios), function(i) NA_real_)]
    }
    
    ## ── Trough rise (trough ratios) ───────────────────────────────────────────
    if (length(trough_ratios) > 0L) {
      DT[, paste0("trough_rise_", trough_ratios) :=
           lapply(.SD, function(x) x - cummin(x)),
         by = id, .SDcols = trough_ratios]
      DT[time_index == 1L,
         paste0("trough_rise_", trough_ratios) :=
           lapply(seq_along(trough_ratios), function(i) NA_real_)]
    }
    
    ## ── Momentum (2-period rolling mean vs expanding mean) ───────────────────
    DT[, paste0("momentum_", all_ratios) :=
         lapply(.SD, function(x)
           frollmean(x, n = 2L, align = "right", fill = NA) - cummean(x)),
       by = id, .SDcols = all_ratios]
    
    ## ── Consecutive decline counter (vectorised rle) ──────────────────────────
    ## For each firm, count how many consecutive periods each peak ratio
    ## has been declining (yoy < 0). Uses rle() for vectorised computation.
    for (col in consec_ratios) {
      yoy_col <- paste0("yoy_", col)
      out_col <- paste0("consec_decline_", col)
      DT[, (out_col) := {
        yoy     <- get(yoy_col)
        decline <- (!is.na(yoy)) & (yoy < 0)
        ## rle() over the decline flag; expand back to original length
        r       <- rle(decline)
        counts  <- sequence(r$lengths)
        ## Reset counter to 0 where decline == FALSE or yoy is NA
        result  <- ifelse(decline, counts, 0L)
        result[is.na(yoy)] <- 0L
        as.integer(result)
      }, by = id]
    }
    
    invisible(DT)
  }
  
  ## Build TS features on Train and Test independently (no cross-contamination).
  build_ts_features(Train, "Train")
  build_ts_features(Test,  "Test")
  
  ## Audit
  ts_prefixes <- c("yoy_", "accel_", "expmean_", "dev_expmean_", "expvol_",
                   "peak_drop_", "trough_rise_", "momentum_", "consec_decline_")
  message("--- 02B complete ---")
  message(sprintf("  Train: %d rows x %d cols  |  Test: %d rows x %d cols",
                  nrow(Train), ncol(Train), nrow(Test), ncol(Test)))
  message(sprintf("  TS feature families: %d", length(ts_prefixes)))
  
}, error = function(e) stop("02B failed: ", e$message))


#==============================================================================#
#==== C - Sector Deviation Features (Leakage-Free) ============================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02C: Sector Deviation Features ---")
  
  n_train_init <- nrow(Train)
  n_test_init  <- nrow(Test)
  
  ## ── Identify base ratio columns ────────────────────────────────────────────
  meta_cols_c <- c("id", "refdate", "sector", "size", "groupmember",
                   "public", "y", "year")
  ts_pattern  <- paste(c("^yoy_", "^accel_", "^expmean_", "^dev_expmean_",
                         "^expvol_", "^peak_drop_", "^trough_rise_",
                         "^momentum_", "^consec_decline_",
                         "^time_", "^is_", "^has_", "^history_"),
                       collapse = "|")
  ts_cols       <- grep(ts_pattern, names(Train), value = TRUE)
  sector_ratios <- setdiff(
    names(Train)[sapply(Train, is.numeric)],
    c(meta_cols_c, ts_cols)
  )
  
  ## Coerce any integer ratio cols to double for aggregation.
  int_cols <- sector_ratios[vapply(sector_ratios,
                                   function(nm) is.integer(Train[[nm]]), logical(1L))]
  if (length(int_cols) > 0L) {
    for (col in int_cols) {
      Train[, (col) := as.double(get(col))]
      Test[,  (col) := as.double(get(col))]
    }
    message(sprintf("  Coerced %d integer col(s) to double", length(int_cols)))
  }
  
  message(sprintf("  Base sector ratios: %d", length(sector_ratios)))
  
  ## ── Fit sector statistics on Train only ────────────────────────────────────
  sector_stats <- Train[,
                        c(lapply(setNames(sector_ratios, paste0("mean_", sector_ratios)),
                                 function(col) mean(get(col),           na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("sd_",   sector_ratios)),
                                 function(col) sd(get(col),             na.rm = TRUE)),
                          lapply(setNames(sector_ratios, paste0("med_",  sector_ratios)),
                                 function(col) median(get(col),         na.rm = TRUE)),
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
  
  ## ── Join stats (sector-year → sector → global fallback) ───────────────────
  join_stats <- function(DT, stats, by_cols) {
    stats_u <- unique(stats, by = by_cols)
    setkeyv(stats_u, by_cols)
    for (sc in setdiff(names(stats_u), by_cols))
      DT[stats_u, (sc) := get(paste0("i.", sc)), on = by_cols]
  }
  
  join_stats(Train, sector_stats, c("sector", "year"))
  join_stats(Test,  sector_stats, c("sector", "year"))
  
  probe_col <- paste0("mean_", sector_ratios[1L])
  missing_mask <- is.na(Test[[probe_col]])
  if (any(missing_mask)) {
    warning(sprintf("  %d Test row(s) with unseen sector-year → sector fallback",
                    sum(missing_mask)))
    join_stats(Test[missing_mask], sector_fallback, "sector")
    missing_mask <- is.na(Test[[probe_col]])
  }
  if (any(missing_mask)) {
    warning(sprintf("  %d Test row(s) still missing → global fallback",
                    sum(missing_mask)))
    for (col in c(paste0("mean_", sector_ratios), paste0("sd_", sector_ratios)))
      Test[missing_mask, (col) := global_fallback[[col]]]
  }
  
  ## ── Sector Z-score ─────────────────────────────────────────────────────────
  z_score <- function(DT, col) {
    z <- (DT[[col]] - DT[[paste0("mean_", col)]]) / DT[[paste0("sd_", col)]]
    fifelse(is.na(z) | is.infinite(z), 0, z)
  }
  for (col in sector_ratios) {
    Train[, paste0("secZ_",   col) := z_score(Train, col)]
    Test[,  paste0("secZ_",   col) := z_score(Test,  col)]
    Train[, paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
    Test[,  paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
  }
  
  ## ── Sector percentile rank ─────────────────────────────────────────────────
  for (col in sector_ratios) {
    out <- paste0("secRank_", col)
    Train[, (out) :=
            frank(get(col), ties.method = "average", na.last = "keep") / .N,
          by = .(sector, year)]
    ecdf_list <- Train[,
                       .(ecdf_fn = list(ecdf(get(col)[!is.na(get(col))]))),
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
  
  ## ── Sector volatility ─────────────────────────────────────────────────────
  for (col in sector_ratios) {
    Train[, paste0("secVol_", col) := get(paste0("sd_", col))]
    Test[,  paste0("secVol_", col) := get(paste0("sd_", col))]
  }
  
  ## ── Sector trend (YoY in sector mean) ─────────────────────────────────────
  setorder(sector_stats, sector, year)
  sector_trend <- sector_stats[,
                               c(list(sector = sector, year = year),
                                 lapply(
                                   setNames(paste0("mean_", sector_ratios),
                                            paste0("secTrend_", sector_ratios)),
                                   function(col) get(col) - shift(get(col), 1L, type = "lag")
                                 )),
                               by = sector
  ]
  join_stats(Train, sector_trend, c("sector", "year"))
  join_stats(Test,  sector_trend, c("sector", "year"))
  
  ## ── Firm vs sector divergence ──────────────────────────────────────────────
  for (col in sector_ratios) {
    yc <- paste0("yoy_",      col)
    tc <- paste0("secTrend_", col)
    if (yc %in% names(Train) && tc %in% names(Train)) {
      Train[, paste0("secDiverg_", col) := get(yc) - get(tc)]
      Test[,  paste0("secDiverg_", col) := get(yc) - get(tc)]
    }
  }
  
  ## ── Size x sector Z-score (bug-fixed: uses ss_mean_/ss_sd_ prefixes) ──────
  ss_stats <- Train[,
                    c(lapply(setNames(sector_ratios, paste0("ss_mean_", sector_ratios)),
                             function(col) mean(get(col), na.rm = TRUE)),
                      lapply(setNames(sector_ratios, paste0("ss_sd_",   sector_ratios)),
                             function(col) sd(get(col),   na.rm = TRUE))),
                    by = .(sector, size, year)
  ]
  join_stats(Train, ss_stats, c("sector", "size", "year"))
  join_stats(Test,  ss_stats, c("sector", "size", "year"))
  
  ss_z_score <- function(DT, col) {
    z <- (DT[[col]] - DT[[paste0("ss_mean_", col)]]) / DT[[paste0("ss_sd_", col)]]
    fifelse(is.na(z) | is.infinite(z), 0, z)
  }
  for (col in sector_ratios) {
    Train[, paste0("secSizeZ_", col) := ss_z_score(Train, col)]
    Test[,  paste0("secSizeZ_", col) := ss_z_score(Test,  col)]
  }
  
  ## ── Sector and size dummies ────────────────────────────────────────────────
  make_dummies <- function(Train, Test, col, prefix) {
    lvls    <- setdiff(unique(Train[[col]]), NA_character_)
    ref_lvl <- lvls[which.max(tabulate(match(Train[[col]], lvls)))]
    for (lvl in setdiff(lvls, ref_lvl)) {
      nm <- paste0(prefix, gsub("[^a-zA-Z0-9]", "_", lvl))
      Train[, (nm) := fifelse(get(col) == lvl, 1L, 0L)]
      Test[,  (nm) := fifelse(get(col) == lvl, 1L, 0L)]
    }
    message(sprintf("  %s dummies: %d created  (ref: '%s')",
                    prefix, length(lvls) - 1L, ref_lvl))
  }
  make_dummies(Train, Test, "sector", "sector_")
  make_dummies(Train, Test, "size",   "size_")
  
  ## ── Drop intermediate stat columns ────────────────────────────────────────
  stat_pfx   <- c("^mean_", "^sd_", "^med_", "^p25_", "^p75_",
                  "^ss_mean_", "^ss_sd_", "^n_firms$")
  drop_stats <- unique(unlist(lapply(stat_pfx, function(p)
    grep(p, names(Train), value = TRUE))))
  Train[, (intersect(drop_stats, names(Train))) := NULL]
  Test[,  (intersect(drop_stats, names(Test)))  := NULL]
  
  stopifnot(
    "Train row count changed in 02C!" = nrow(Train) == n_train_init,
    "Test row count changed in 02C!"  = nrow(Test)  == n_test_init
  )
  
  message("--- 02C complete ---")
  message(sprintf("  Train: %d rows x %d cols  |  Test: %d rows x %d cols",
                  nrow(Train), ncol(Train), nrow(Test), ncol(Test)))
  
}, error = function(e) stop("02C failed: ", e$message))


#==============================================================================#
#==== D - Quantile Transformation =============================================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02D: Quantile Transformation ---")
  
  if (QUANTILE_TRANSFORM) {
    
    exclude_patterns <- c(
      "^secZ_", "^secSizeZ_", "^secRank_", "^consec_",
      "^sector_", "^size_", "^time_", "^history_", "^is_", "^has_"
    )
    semantic_exclude <- c("y", "id", "refdate", "sector", "size",
                          "year", "groupmember", "public")
    
    numeric_cols <- names(Train)[sapply(Train, is.numeric)]
    
    pattern_excluded <- names(Train)[vapply(names(Train), function(nm)
      any(vapply(exclude_patterns, function(p) grepl(p, nm), logical(1L))),
      logical(1L))]
    
    detect_binary    <- function(nm) {
      x <- Train[[nm]][!is.na(Train[[nm]])]
      length(x) > 0L && all(x %in% c(0, 1))
    }
    detect_bounded01 <- function(nm) {
      x <- Train[[nm]][!is.na(Train[[nm]])]
      length(x) > 0L && !all(x %in% c(0, 1)) && min(x) >= 0 && max(x) <= 1
    }
    
    binary_excluded  <- names(Train)[vapply(names(Train), detect_binary,  logical(1L))]
    bounded_excluded <- if (!TRANSFORM_BOUNDED01)
      numeric_cols[vapply(numeric_cols, detect_bounded01, logical(1L))]
    else character(0L)
    
    exclude_cols      <- unique(c(semantic_exclude, pattern_excluded,
                                  binary_excluded, bounded_excluded))
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    ## Safety guards
    bad_semantic <- intersect(semantic_exclude, cols_to_transform)
    if (length(bad_semantic) > 0L)
      stop("Semantic cols in transform list: ", paste(bad_semantic, collapse = ", "))
    bad_binary <- cols_to_transform[vapply(cols_to_transform, detect_binary, logical(1L))]
    if (length(bad_binary) > 0L)
      stop("Binary cols in transform list: ", paste(bad_binary, collapse = ", "))
    if (!exists("QuantileTransformation"))
      stop("QuantileTransformation() not found — check Subfunctions/ was sourced.")
    if (length(cols_to_transform) == 0L)
      stop("No columns selected for transformation — check exclusion logic.")
    
    message(sprintf("  Columns to transform: %d  |  excluded: %d",
                    length(cols_to_transform),
                    length(numeric_cols) - length(cols_to_transform)))
    
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    failed_cols       <- character(0L)
    
    for (col in cols_to_transform) {
      if (!col %in% names(Test_Transformed)) {
        warning("Column missing from Test, skipping: ", col); next
      }
      if (all(is.na(Train_Transformed[[col]]))) {
        warning("Column entirely NA in Train, skipping: ", col); next
      }
      res <- tryCatch(
        QuantileTransformation(Train_Transformed[[col]], Test_Transformed[[col]]),
        error = function(e) {
          failed_cols <<- c(failed_cols, col)
          message("  Transform failed: ", col, " — ", e$message)
          NULL
        }
      )
      if (!is.null(res)) {
        Train_Transformed[[col]] <- res$train
        Test_Transformed[[col]]  <- res$test
      }
    }
    
    message(sprintf("  Transformed: %d  |  Failed: %d",
                    length(cols_to_transform) - length(failed_cols),
                    length(failed_cols)))
    
    ## NA delta check
    na_delta_train <- sum(is.na(Train_Transformed[, .SD, .SDcols = cols_to_transform])) -
      sum(is.na(Train[,             .SD, .SDcols = cols_to_transform]))
    na_delta_test  <- sum(is.na(Test_Transformed[,  .SD, .SDcols = cols_to_transform])) -
      sum(is.na(Test[,              .SD, .SDcols = cols_to_transform]))
    if (na_delta_train != 0L || na_delta_test != 0L)
      warning(sprintf("NA delta after transform — Train: %+d  Test: %+d",
                      na_delta_train, na_delta_test))
    message(sprintf("  NA delta — Train: %+d  Test: %+d", na_delta_train, na_delta_test))
    
  } else {
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    message("  QUANTILE_TRANSFORM = FALSE — skipping.")
  }
  
  message("--- 02D complete ---")
  
}, error = function(e) stop("02D failed: ", e$message))


#==============================================================================#
#==== E - Final Cleanup, Imputation, Validation & Save ========================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02E: Final Cleanup, Imputation & Validation ---")
  
  ## ── Structural NA imputation ───────────────────────────────────────────────
  ## Cold-start NAs (first obs per firm has no prior period) are expected in
  ## all lag-derived features. Strategy: zero for differences/deviations,
  ## train median for level/scale features.
  
  imputation_rules <- list(
    list(pattern = "^yoy_",         strategy = "zero"),
    list(pattern = "^accel_",       strategy = "zero"),
    list(pattern = "^momentum_",    strategy = "zero"),
    list(pattern = "^peak_drop_",   strategy = "zero"),
    list(pattern = "^trough_rise_", strategy = "zero"),
    list(pattern = "^secTrend_",    strategy = "zero"),
    list(pattern = "^secDiverg_",   strategy = "zero"),
    list(pattern = "^secDev_",      strategy = "zero"),
    list(pattern = "^expvol_",      strategy = "median"),
    list(pattern = "^secVol_",      strategy = "median"),
    list(pattern = "^expmean_",     strategy = "median"),
    list(pattern = "^dev_expmean_", strategy = "zero")
  )
  
  impute_fixed  <- function(col, val) {
    Train_Transformed[is.na(get(col)), (col) := val]
    Test_Transformed[ is.na(get(col)), (col) := val]
  }
  impute_median <- function(col) {
    fill <- median(Train_Transformed[[col]], na.rm = TRUE)
    if (is.na(fill)) fill <- 0
    Train_Transformed[is.na(get(col)), (col) := fill]
    Test_Transformed[ is.na(get(col)), (col) := fill]
  }
  
  handled_cols <- character(0L)
  for (rule in imputation_rules) {
    matched <- grep(rule$pattern, names(Train_Transformed), value = TRUE)
    for (col in matched) {
      if (rule$strategy == "zero")   impute_fixed(col, 0)
      if (rule$strategy == "median") impute_median(col)
    }
    handled_cols <- c(handled_cols, matched)
  }
  
  ## Catch-all for any remaining numeric NAs not matched above
  remaining_na_cols <- names(Train_Transformed)[
    vapply(names(Train_Transformed), function(nm)
      is.numeric(Train_Transformed[[nm]]) &&
        (anyNA(Train_Transformed[[nm]]) || anyNA(Test_Transformed[[nm]])),
      logical(1L))
  ]
  catchall_cols <- setdiff(remaining_na_cols, handled_cols)
  if (length(catchall_cols) > 0L) {
    message(sprintf("  Catch-all median imputation (%d col(s)): %s",
                    length(catchall_cols), paste(catchall_cols, collapse = ", ")))
    for (col in catchall_cols) impute_median(col)
  }
  
  message(sprintf("  NAs after imputation — Train: %d  Test: %d",
                  sum(is.na(Train_Transformed)), sum(is.na(Test_Transformed))))
  
  ## ── Target enforcement (direct assignment — no list copy) ─────────────────
  ## Bug fix: original iterated over list(Train_T, Test_T) with :=, which
  ## modifies local list-element copies, not the original data.tables.
  ## Fixed by assigning directly to each data.table.
  Train_Transformed[, (TARGET_COL) := as.integer(as.character(get(TARGET_COL)))]
  Test_Transformed[,  (TARGET_COL) := as.integer(as.character(get(TARGET_COL)))]
  
  ## ── Export id vectors BEFORE dropping id ──────────────────────────────────
  if (!"id" %in% names(Train_Transformed))
    stop("'id' column missing from Train_Transformed before export.")
  train_id_vec <- Train_Transformed$id
  test_id_vec  <- Test_Transformed$id
  message(sprintf("  Exported id vectors — Train: %d  Test: %d",
                  length(train_id_vec), length(test_id_vec)))
  
  ## ── Drop metadata and zero-variance columns ────────────────────────────────
  drop_patterns <- c("^id$", "^company_id$", "^row_id$",
                     "refdate", "^year$", "^sector$", "^size$")
  pattern_drop  <- names(Train_Transformed)[vapply(names(Train_Transformed),
                                                   function(nm) any(vapply(drop_patterns, function(p) grepl(p, nm), logical(1L))),
                                                   logical(1L))]
  
  ## Use sd() not var() — var() returns near-zero floats on transformed cols,
  ## causing false zero-variance positives.
  zerovar_drop <- names(Train_Transformed)[vapply(names(Train_Transformed),
                                                  function(nm) {
                                                    x <- Train_Transformed[[nm]]
                                                    is.numeric(x) && !anyNA(x) && sd(x) == 0
                                                  }, logical(1L))]
  if (length(zerovar_drop) > 0L)
    message("  Zero-variance cols dropped: ", paste(zerovar_drop, collapse = ", "))
  
  cols_to_drop <- setdiff(unique(c(pattern_drop, zerovar_drop)), TARGET_COL)
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  
  Train_Final <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final  <- copy(Test_Transformed[,  ..cols_to_keep])
  
  ## ── Final validation ───────────────────────────────────────────────────────
  if (!setequal(names(Train_Final), names(Test_Final))) {
    only_train <- setdiff(names(Train_Final), names(Test_Final))
    only_test  <- setdiff(names(Test_Final),  names(Train_Final))
    stop(sprintf("Column mismatch — Train only: [%s]  Test only: [%s]",
                 paste(only_train, collapse = ", "),
                 paste(only_test,  collapse = ", ")))
  }
  
  all_feature_cols <- setdiff(names(Train_Final), TARGET_COL)
  non_numeric <- all_feature_cols[
    !vapply(Train_Final[, .SD, .SDcols = all_feature_cols], is.numeric, logical(1L))]
  if (length(non_numeric) > 0L)
    stop("Non-numeric columns remain (will break VAE): ",
         paste(non_numeric, collapse = ", "))
  
  stopifnot(
    "NAs in Train_Final" = sum(is.na(Train_Final)) == 0L,
    "NAs in Test_Final"  = sum(is.na(Test_Final))  == 0L
  )
  
  message("--- 02E Validation Report ---")
  message(sprintf("  Split mode          : %s", SPLIT_MODE))
  message(sprintf("  Train_Final         : %d rows x %d cols", nrow(Train_Final), ncol(Train_Final)))
  message(sprintf("  Test_Final          : %d rows x %d cols", nrow(Test_Final),  ncol(Test_Final)))
  message(sprintf("  Train default rate  : %.3f%%  (%d events)",
                  100 * mean(Train_Final[[TARGET_COL]]), sum(Train_Final[[TARGET_COL]])))
  message(sprintf("  Test  default rate  : %.3f%%  (%d events)",
                  100 * mean(Test_Final[[TARGET_COL]]),  sum(Test_Final[[TARGET_COL]])))
  message(sprintf("  Total model features: %d", length(all_feature_cols)))
  message("  Column alignment    : OK")
  message("  All features numeric: OK")
  
  ## ── Save .rds (suffixed by SPLIT_MODE for downstream auto-resolution) ─────
  saveRDS(Train_Final,  get_split_path(SPLIT_OUT_TRAIN_FINAL))
  saveRDS(Test_Final,   get_split_path(SPLIT_OUT_TEST_FINAL))
  saveRDS(train_id_vec, get_split_path(SPLIT_OUT_TRAIN_IDS))
  saveRDS(test_id_vec,  get_split_path(SPLIT_OUT_TEST_IDS))
  
  message(sprintf("  Saved: %s", get_split_path(SPLIT_OUT_TRAIN_FINAL)))
  message(sprintf("  Saved: %s", get_split_path(SPLIT_OUT_TEST_FINAL)))
  message(sprintf("  Saved: %s", get_split_path(SPLIT_OUT_TRAIN_IDS)))
  message(sprintf("  Saved: %s", get_split_path(SPLIT_OUT_TEST_IDS)))
  message("--- 02_FeatureEngineering complete ---")
  
}, error = function(e) stop("02E failed: ", e$message))