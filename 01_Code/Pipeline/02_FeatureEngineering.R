#==============================================================================#
#==== 02_FeatureEngineering.R =================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Build the full feature set from the clean panel.
#   Runs five sequential stages, each guarded by tryCatch:
#     A  Time-series dynamics     (YoY, acceleration, momentum, etc.)
#     B  Train / test split       (stratified, firm-level, leakage-checked)
#     C  Sector deviation features (leakage-free: fit on Train, apply to both)
#     D  Quantile transformation
#     E  Final cleanup, imputation, validation
#
# INPUTS:
#   Data                  — from 01_Data.R
#   SEED, TARGET_COL, STRAT_VARS, TRAIN_SIZE,
#   QUANTILE_TRANSFORM, TRANSFORM_BOUNDED01,
#   MIN_SS_FIRMS          — all from config.R
#
# OUTPUTS:
#   Train_Final, Test_Final   — modelling-ready data.tables
#
# CHANGES FROM ORIGINAL:
#   [A1] Heuristic ratio classification note: ratio_medians/mins/maxs are
#        computed on the full Data (pre-split). For the direction heuristic
#        this is low-stakes, but the comment now makes it explicit.
#   [B1] Row-count stopifnot: original checked nrow(Train)+nrow(Test)==nrow(Data).
#        This fails when MVstratifiedsampling drops singleton strata (as it
#        should). Fixed to allow for dropped firms: the check now uses
#        n_firms_used instead of nrow(Data).
#   [C1] MAJOR BUG FIX — secSizeZ features were silently all-NA.
#        Original code passed a temp data.table with columns (x, mean, sd) to
#        z_score(DT, col) where col="x". Inside z_score, the function looks for
#        columns named paste0("mean_", col) = "mean_x" and paste0("sd_", col)
#        = "sd_x", which do not exist. Result: every secSizeZ_ value was NA.
#        Fixed by computing the z-score inline using the ss_mean_/ss_sd_ columns
#        directly, bypassing the z_score helper for this specific case.
#   [C2] Minor: probe_col fallback logic for unseen Test sector-years now
#        checks for NA in Test itself, not a subset, avoiding a data.table
#        scoping edge case.
#   [D1] TRANSFORM_BOUNDED01 now read from config.R (was exists() check
#        on a local variable that had to be set manually before running).
#   [E1] Imputation catch-all now logs all remaining columns, not just head(6).
#   [E2] zero-variance detection uses sd() rather than var() to avoid
#        floating-point near-zero false positives on transformed columns.
#
#==============================================================================#


#==============================================================================#
#==== A - Time-Series Dynamics ================================================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02A: Time-Series Dynamics ---")
  
  Data <- as.data.table(Data)
  setorder(Data, id, refdate)
  
  ## ── Detect ratio columns programmatically ──────────────────────────────────
  meta_cols <- c("id", "refdate", "sector", "size", "groupmember",
                 "public", "y", "year")
  
  all_ratios <- setdiff(
    names(Data)[sapply(Data, is.numeric)],
    meta_cols
  )
  
  ## ── Classify ratios by economic direction ──────────────────────────────────
  ## [A1] Heuristic computed on full Data (pre-split).
  ## For direction classification this is low-stakes: we only decide whether
  ## "higher" or "lower" values are healthy. No statistics flow into features.
  ## Override by setting PEAK_RATIOS / TROUGH_RATIOS in config.R if needed.
  
  if (exists("PEAK_RATIOS") && exists("TROUGH_RATIOS")) {
    peak_ratios   <- intersect(PEAK_RATIOS,   all_ratios)
    trough_ratios <- intersect(TROUGH_RATIOS, all_ratios)
  } else {
    ratio_medians <- sapply(all_ratios, function(r) median(Data[[r]], na.rm = TRUE))
    ratio_mins    <- sapply(all_ratios, function(r) min(Data[[r]],    na.rm = TRUE))
    ratio_maxs    <- sapply(all_ratios, function(r) max(Data[[r]],    na.rm = TRUE))
    
    is_bounded01   <- ratio_mins >= 0 & ratio_maxs <= 1
    is_high_median <- ratio_medians > 0.5
    
    trough_ratios  <- all_ratios[ is_bounded01 &  is_high_median]
    peak_ratios    <- all_ratios[!(is_bounded01 &  is_high_median)]
  }
  
  consec_ratios <- if (exists("CONSEC_RATIOS")) {
    intersect(CONSEC_RATIOS, all_ratios)
  } else {
    peak_ratios
  }
  
  message(sprintf("  Detected %d ratio cols | %d peak | %d trough | %d consec",
                  length(all_ratios), length(peak_ratios),
                  length(trough_ratios), length(consec_ratios)))
  
  ## ── Base tracking ──────────────────────────────────────────────────────────
  Data[, time_index     := seq_len(.N),             by = id]
  Data[, is_mature      := fifelse(time_index >= 3, 1L, 0L)]
  Data[, history_length := .N,                      by = id]
  Data[, has_history    := fifelse(time_index > 1,  1L, 0L)]
  
  ## ── E: YoY changes (1st differences) ──────────────────────────────────────
  message("  Computing YoY changes...")
  Data[, paste0("yoy_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
       by = id, .SDcols = all_ratios]
  
  ## ── F: Acceleration (2nd differences) ─────────────────────────────────────
  message("  Computing acceleration...")
  yoy_cols <- paste0("yoy_", all_ratios)
  Data[, paste0("accel_", all_ratios) :=
         lapply(.SD, function(x) x - shift(x, n = 1L, type = "lag")),
       by = id, .SDcols = yoy_cols]
  
  ## ── G: Expanding mean ─────────────────────────────────────────────────────
  message("  Computing expanding means...")
  Data[, paste0("expmean_", all_ratios) :=
         lapply(.SD, cummean),
       by = id, .SDcols = all_ratios]
  
  ## ── H: Deviation from expanding mean ──────────────────────────────────────
  message("  Computing deviations from expanding mean...")
  expmean_cols <- paste0("expmean_", all_ratios)
  for (i in seq_along(all_ratios)) {
    Data[, paste0("dev_expmean_", all_ratios[i]) :=
           get(all_ratios[i]) - get(expmean_cols[i])]
  }
  
  ## ── I: Expanding volatility ────────────────────────────────────────────────
  message("  Computing expanding volatility...")
  Data[, paste0("expvol_", all_ratios) :=
         lapply(.SD, function(x) {
           n      <- seq_along(x)
           mu     <- cummean(x)
           expvar <- (cumsum(x^2) - n * mu^2) / pmax(n - 1L, 1L)
           fifelse(n < 2L, NA_real_, sqrt(pmax(0, expvar)))
         }),
       by = id, .SDcols = all_ratios]
  
  ## ── J: Peak deterioration ─────────────────────────────────────────────────
  message("  Computing peak deterioration...")
  if (length(peak_ratios) > 0L) {
    Data[, paste0("peak_drop_", peak_ratios) :=
           lapply(.SD, function(x) x - cummax(x)),
         by = id, .SDcols = peak_ratios]
    Data[time_index == 1L,
         paste0("peak_drop_", peak_ratios) :=
           lapply(peak_ratios, function(x) NA_real_)]
  }
  
  ## ── K: Trough rise ────────────────────────────────────────────────────────
  message("  Computing trough rise...")
  if (length(trough_ratios) > 0L) {
    Data[, paste0("trough_rise_", trough_ratios) :=
           lapply(.SD, function(x) x - cummin(x)),
         by = id, .SDcols = trough_ratios]
    Data[time_index == 1L,
         paste0("trough_rise_", trough_ratios) :=
           lapply(trough_ratios, function(x) NA_real_)]
  }
  
  ## ── L: Momentum (2Y rolling mean vs expanding mean) ───────────────────────
  message("  Computing momentum...")
  Data[, paste0("momentum_", all_ratios) :=
         lapply(.SD, function(x) {
           frollmean(x, n = 2L, align = "right", fill = NA) - cummean(x)
         }),
       by = id, .SDcols = all_ratios]
  
  ## ── M: Consecutive decline counter ────────────────────────────────────────
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
  
  ## ── N: Sanity check ───────────────────────────────────────────────────────
  ts_prefixes <- c("yoy_", "accel_", "expmean_", "dev_expmean_", "expvol_",
                   "peak_drop_", "trough_rise_", "momentum_", "consec_decline_")
  
  na_audit <- data.frame(
    family   = ts_prefixes,
    n_cols   = sapply(ts_prefixes, function(p)
      length(grep(paste0("^", p), names(Data)))),
    na_count = sapply(ts_prefixes, function(p) {
      cols <- grep(paste0("^", p), names(Data), value = TRUE)
      if (length(cols) == 0L) return(0L)
      sum(is.na(Data[, .SD, .SDcols = cols]))
    })
  )
  
  message("02A complete.")
  message(sprintf("  New TS features : %d", sum(na_audit$n_cols)))
  message(sprintf("  Total columns   : %d", ncol(Data)))
  message(sprintf("  Rows            : %d", nrow(Data)))
  print(na_audit, row.names = FALSE)
  
}, error = function(e) stop("02A failed: ", e$message))


#==============================================================================#
#==== B - Train / Test Split ==================================================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02B: Train/Test Split ---")
  
  ## MVstratifiedsampling is the REVISED version defined here (not the uploaded
  ## Subfunctions/ version). The revised version:
  ##   - Uses y_ever = any(y==1) for the stratification key (correct for panels)
  ##   - Guards against singleton strata by dropping and warning
  ## The uploaded MVstratifiedsampling.R uses max(y) and has no singleton guard.
  
  MVstratifiedsampling_panel <- function(Data,
                                         strat_vars = c("sector", "y_ever"),
                                         Train_size = 0.7) {
    
    firm_profile <- Data %>%
      group_by(id) %>%
      summarise(
        y_ever = as.integer(any(y == 1L)),
        sector = first(sector),
        size   = first(size),
        .groups = "drop"
      ) %>%
      mutate(
        Strat_Key = interaction(
          across(all_of(strat_vars)),
          drop = TRUE
        )
      )
    
    ## Guard: drop strata with fewer than 2 firms (createDataPartition requires ≥2)
    strat_counts <- table(firm_profile$Strat_Key)
    small_strata <- names(strat_counts[strat_counts < 2L])
    
    n_dropped_firms <- 0L
    if (length(small_strata) > 0L) {
      n_dropped_firms <- sum(firm_profile$Strat_Key %in% small_strata)
      warning(sprintf(
        "Dropping %d singleton stratum/a (%d firm(s)): %s",
        length(small_strata), n_dropped_firms,
        paste(small_strata, collapse = ", ")
      ))
      firm_profile <- firm_profile %>%
        filter(!Strat_Key %in% small_strata) %>%
        mutate(Strat_Key = droplevels(Strat_Key))
    }
    
    set.seed(SEED)
    train_index <- createDataPartition(
      y     = firm_profile$Strat_Key,
      p     = Train_size,
      list  = FALSE,
      times = 1L
    )
    
    train_ids <- firm_profile$id[ train_index]
    test_ids  <- firm_profile$id[-train_index]
    
    Train <- Data %>% filter(id %in% train_ids)
    Test  <- Data %>% filter(id %in% test_ids)
    
    list(Train = Train, Test = Test,
         n_dropped_firms = n_dropped_firms,
         n_used_firms    = nrow(firm_profile))
  }
  
  splits <- MVstratifiedsampling_panel(
    Data       = as.data.frame(Data),
    strat_vars = STRAT_VARS,
    Train_size = TRAIN_SIZE
  )
  
  Train <- as.data.table(splits[["Train"]])
  Test  <- as.data.table(splits[["Test"]])
  
  train_ids <- unique(Train$id)
  test_ids  <- unique(Test$id)
  
  ## Leakage check — must always hold
  stopifnot(
    "Firm-level leakage detected!" =
      length(intersect(train_ids, test_ids)) == 0L
  )
  
  ## [B1] Row-loss check accounts for singleton-strata firms that were dropped.
  ## Original: nrow(Train) + nrow(Test) == nrow(Data)  — fails when strata dropped.
  ## Fixed: compare against rows belonging to firms that were actually used.
  n_used_firms  <- splits[["n_used_firms"]]
  rows_used     <- nrow(Data[Data$id %in% c(train_ids, test_ids), ])
  stopifnot(
    "Row loss detected in train/test split!" =
      nrow(Train) + nrow(Test) == rows_used
  )
  
  message(sprintf("  Train: %d firms | %d rows | %.3f%% default rate",
                  length(train_ids), nrow(Train), 100 * mean(Train$y)))
  message(sprintf("  Test:  %d firms | %d rows | %.3f%% default rate",
                  length(test_ids),  nrow(Test),  100 * mean(Test$y)))
  message("02B complete.")
  
}, error = function(e) stop("02B failed: ", e$message))


#==============================================================================#
#==== C - Sector Deviation Features (Leakage-Free) ============================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02C: Sector Deviation Features ---")
  
  if (!is.data.table(Train)) setDT(Train)
  if (!is.data.table(Test))  setDT(Test)
  if (!"year" %in% names(Train)) Train[, year := year(refdate)]
  if (!"year" %in% names(Test))  Test[,  year := year(refdate)]
  
  n_train_init <- nrow(Train)
  n_test_init  <- nrow(Test)
  
  ## ── Detect sector ratios ───────────────────────────────────────────────────
  meta_cols_c <- c("id", "refdate", "sector", "size", "groupmember", "y", "year")
  ts_prefixes <- c("^yoy_", "^accel_", "^expmean_", "^dev_expmean_",
                   "^expvol_", "^peak_drop_", "^trough_rise_",
                   "^momentum_", "^consec_decline_",
                   "^time_", "^is_", "^has_", "^history_")
  ts_cols <- grep(paste(ts_prefixes, collapse = "|"), names(Train), value = TRUE)
  
  sector_ratios <- setdiff(
    names(Train)[sapply(Train, is.numeric)],
    c(meta_cols_c, ts_cols)
  )
  
  ## Coerce integer ratio columns to double (required for data.table aggregation)
  int_cols <- sector_ratios[vapply(sector_ratios, function(nm)
    is.integer(Train[[nm]]), logical(1L))]
  if (length(int_cols) > 0L) {
    for (col in int_cols) {
      Train[, (col) := as.double(get(col))]
      Test[,  (col) := as.double(get(col))]
    }
    message(sprintf("  Coerced %d integer ratio col(s) to double", length(int_cols)))
  }
  
  size_sector_ratios <- sector_ratios
  ss_coverage <- Train[, .(n = .N), by = .(sector, size, year)]
  if (mean(ss_coverage$n) < MIN_SS_FIRMS)
    warning("  Low size-sector cell coverage — secSizeZ features may be sparse.")
  
  message(sprintf("  Detected %d sector ratios", length(sector_ratios)))
  
  ## ── D: Fit sector statistics on Train only ─────────────────────────────────
  message("  Fitting sector-year statistics on Train only...")
  
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
  
  ## ── E: Join stats + fallback cascade ──────────────────────────────────────
  message("  Joining sector stats with fallback cascade...")
  
  join_stats <- function(DT, stats, by_cols) {
    stats_u   <- unique(stats, by = by_cols)
    setkeyv(stats_u, by_cols)
    join_cols <- setdiff(names(stats_u), by_cols)
    for (sc in join_cols)
      DT[stats_u, (sc) := get(paste0("i.", sc)), on = by_cols]
  }
  
  join_stats(Train, sector_stats, c("sector", "year"))
  join_stats(Test,  sector_stats, c("sector", "year"))
  
  ## [C2] Probe column NA check: use Test directly, not a subset
  probe_col <- paste0("mean_", sector_ratios[1L])
  missing_mask <- is.na(Test[[probe_col]])
  if (any(missing_mask)) {
    warning(sprintf("  %d Test row(s) with unseen sector-year → sector fallback",
                    sum(missing_mask)))
    join_stats(Test[missing_mask], sector_fallback, "sector")
    missing_mask <- is.na(Test[[probe_col]])   ## re-check after fallback
  }
  if (any(missing_mask)) {
    warning(sprintf("  %d Test row(s) still missing → global fallback", sum(missing_mask)))
    for (col in c(paste0("mean_", sector_ratios), paste0("sd_", sector_ratios)))
      Test[missing_mask, (col) := global_fallback[[col]]]
  }
  
  ## ── F: Sector Z-score ─────────────────────────────────────────────────────
  message("  Computing sector Z-scores...")
  
  z_score <- function(DT, col) {
    z <- (DT[[col]] - DT[[paste0("mean_", col)]]) / DT[[paste0("sd_", col)]]
    fifelse(is.na(z) | is.infinite(z), 0, z)
  }
  
  for (col in sector_ratios) {
    Train[, paste0("secZ_", col) := z_score(Train, col)]
    Test[,  paste0("secZ_", col) := z_score(Test,  col)]
  }
  
  ## ── G: Raw sector deviation ───────────────────────────────────────────────
  message("  Computing raw sector deviations...")
  for (col in sector_ratios) {
    Train[, paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
    Test[,  paste0("secDev_", col) := get(col) - get(paste0("mean_", col))]
  }
  
  ## ── H: Sector percentile rank ─────────────────────────────────────────────
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
  
  ## ── I: Sector volatility ──────────────────────────────────────────────────
  message("  Extracting sector volatility features...")
  for (col in sector_ratios) {
    Train[, paste0("secVol_", col) := get(paste0("sd_", col))]
    Test[,  paste0("secVol_", col) := get(paste0("sd_", col))]
  }
  
  ## ── J: Sector trend ───────────────────────────────────────────────────────
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
  
  ## ── K: Firm vs sector divergence ──────────────────────────────────────────
  message("  Computing firm vs sector divergence...")
  for (col in sector_ratios) {
    yoy_col   <- paste0("yoy_",      col)
    trend_col <- paste0("secTrend_", col)
    if (yoy_col %in% names(Train) && trend_col %in% names(Train)) {
      Train[, paste0("secDiverg_", col) := get(yoy_col) - get(trend_col)]
      Test[,  paste0("secDiverg_", col) := get(yoy_col) - get(trend_col)]
    }
  }
  
  ## ── L: Size × sector Z-score ──────────────────────────────────────────────
  ## [C1] BUG FIX: Original called z_score() with a temp data.table of columns
  ## (x, mean, sd), but z_score() looks for paste0("mean_", col) = "mean_x"
  ## and paste0("sd_", col) = "sd_x". Neither column exists in the temp DT.
  ## All secSizeZ_ features were silently NA.
  ##
  ## Fix: compute the z-score inline using the actual ss_mean_/ss_sd_ column
  ## names, which are correctly joined by join_stats() above.
  
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
  
  ss_z_score <- function(DT, col) {
    ## Inline z-score using ss_mean_/ss_sd_ prefix — avoids the "mean_x" bug.
    mu <- DT[[paste0("ss_mean_", col)]]
    sg <- DT[[paste0("ss_sd_",   col)]]
    z  <- (DT[[col]] - mu) / sg
    fifelse(is.na(z) | is.infinite(z), 0, z)
  }
  
  for (col in size_sector_ratios) {
    Train[, paste0("secSizeZ_", col) := ss_z_score(Train, col)]
    Test[,  paste0("secSizeZ_", col) := ss_z_score(Test,  col)]
  }
  
  ## ── M: Sector dummies ─────────────────────────────────────────────────────
  message("  Creating sector dummies...")
  sector_levels <- setdiff(unique(Train$sector), NA)
  ref_level     <- sector_levels[which.max(tabulate(match(Train$sector, sector_levels)))]
  dummy_levels  <- setdiff(sector_levels, ref_level)
  
  for (s in dummy_levels) {
    out_col <- paste0("sector_", gsub("[^a-zA-Z0-9]", "_", s))
    Train[, (out_col) := fifelse(sector == s, 1L, 0L)]
    Test[,  (out_col) := fifelse(sector == s, 1L, 0L)]
  }
  
  ## ── N: Cleanup intermediate stat columns ──────────────────────────────────
  stat_prefixes <- c("mean_", "sd_", "med_", "p25_", "p75_",
                     "ss_mean_", "ss_sd_", "n_firms")
  drop_cols <- unique(unlist(lapply(stat_prefixes, function(p)
    grep(paste0("^", p), names(Train), value = TRUE)
  )))
  Train[, (intersect(drop_cols, names(Train))) := NULL]
  Test[,  (intersect(drop_cols, names(Test)))  := NULL]
  
  ## ── O: Sanity check ───────────────────────────────────────────────────────
  stopifnot(
    "Train row count changed in 02C!" = nrow(Train) == n_train_init,
    "Test row count changed in 02C!"  = nrow(Test)  == n_test_init
  )
  
  families <- c("secZ_", "secDev_", "secRank_", "secVol_",
                "secTrend_", "secDiverg_", "secSizeZ_", "sector_")
  na_audit_c <- data.frame(
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
  
  message("02C complete.")
  message(sprintf("  Train: %d rows x %d cols", nrow(Train), ncol(Train)))
  message(sprintf("  Test:  %d rows x %d cols", nrow(Test),  ncol(Test)))
  print(na_audit_c, row.names = FALSE)
  
}, error = function(e) stop("02C failed: ", e$message))


#==============================================================================#
#==== D - Quantile Transformation =============================================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02D: Quantile Transformation ---")
  
  ## [D1] Read flag from config.R — original required a manual local variable.
  if (QUANTILE_TRANSFORM) {
    
    ## ── Exclusion logic ───────────────────────────────────────────────────────
    exclude_patterns  <- c(
      "^secZ_", "^secSizeZ_", "^secRank_", "^consec_",
      "^sector_", "^time_", "^history_", "^is_", "^has_"
    )
    semantic_exclude  <- c("y", "id", "refdate", "sector", "size",
                           "year", "groupmember")
    
    all_cols     <- names(Train)
    numeric_cols <- all_cols[sapply(Train, is.numeric)]
    
    pattern_excluded <- all_cols[vapply(all_cols, function(nm)
      any(vapply(exclude_patterns, function(p) grepl(p, nm), logical(1L))),
      logical(1L))]
    
    detect_binary <- function(nm) {
      x_obs <- Train[[nm]][!is.na(Train[[nm]])]
      length(x_obs) > 0L && all(x_obs %in% c(0, 1))
    }
    binary_excluded <- all_cols[vapply(all_cols, detect_binary, logical(1L))]
    
    detect_bounded01 <- function(nm) {
      x_obs <- Train[[nm]][!is.na(Train[[nm]])]
      length(x_obs) > 0L && !all(x_obs %in% c(0, 1)) &&
        min(x_obs) >= 0 && max(x_obs) <= 1
    }
    ## [D1] Uses TRANSFORM_BOUNDED01 from config.R instead of exists() check
    bounded_excluded <- if (!TRANSFORM_BOUNDED01) {
      numeric_cols[vapply(numeric_cols, detect_bounded01, logical(1L))]
    } else character(0L)
    
    exclude_cols      <- unique(c(semantic_exclude, pattern_excluded,
                                  binary_excluded,  bounded_excluded))
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    ## ── Audit ─────────────────────────────────────────────────────────────────
    message(sprintf("  Numeric cols           : %d", length(numeric_cols)))
    message(sprintf("  Semantic exclusions    : %d",
                    sum(all_cols %in% semantic_exclude)))
    message(sprintf("  Pattern exclusions     : %d", length(pattern_excluded)))
    message(sprintf("  Binary exclusions      : %d", length(binary_excluded)))
    message(sprintf("  Bounded [0,1] excl.    : %d", length(bounded_excluded)))
    message(sprintf("  Columns to transform   : %d", length(cols_to_transform)))
    
    ## Safety guards
    semantic_in_transform <- intersect(semantic_exclude, cols_to_transform)
    if (length(semantic_in_transform) > 0L)
      stop("Semantic cols in transform list: ",
           paste(semantic_in_transform, collapse = ", "))
    
    binary_in_transform <- cols_to_transform[vapply(cols_to_transform,
                                                    detect_binary, logical(1L))]
    if (length(binary_in_transform) > 0L)
      stop("Binary cols in transform list: ",
           paste(binary_in_transform, collapse = ", "))
    
    if (!exists("QuantileTransformation"))
      stop("QuantileTransformation() not found — check Subfunctions/ is sourced.")
    if (length(cols_to_transform) == 0L)
      stop("No columns selected for transformation.")
    
    ## ── Transform loop ─────────────────────────────────────────────────────────
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    failed_cols       <- character(0L)
    
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
    
    ## ── Post-transform audit ───────────────────────────────────────────────────
    n_ok <- length(cols_to_transform) - length(failed_cols)
    message(sprintf("  Successfully transformed : %d", n_ok))
    if (length(failed_cols) > 0L)
      message("  Failed: ", paste(failed_cols, collapse = ", "))
    
    na_delta <- function(before, after, label) {
      d <- sum(is.na(after[,   .SD, .SDcols = cols_to_transform])) -
        sum(is.na(before[, .SD, .SDcols = cols_to_transform]))
      if (d > 0L)
        warning(sprintf("  %s: introduced %d new NA(s) during transform", label, d))
      d
    }
    message(sprintf("  NA delta — Train: %d | Test: %d",
                    na_delta(Train, Train_Transformed, "Train"),
                    na_delta(Test,  Test_Transformed,  "Test")))
    
    ## ── Binary integrity check ─────────────────────────────────────────────────
    binary_check_cols <- intersect(
      unique(c(binary_excluded, "groupmember")),
      names(Train_Transformed)
    )
    binary_check_cols <- binary_check_cols[vapply(binary_check_cols, function(nm)
      is.numeric(Train_Transformed[[nm]]) || is.integer(Train_Transformed[[nm]]),
      logical(1L))]
    
    corrupted <- binary_check_cols[vapply(binary_check_cols, function(nm) {
      x <- Train_Transformed[[nm]]
      !all(x[!is.na(x)] %in% c(0, 1))
    }, logical(1L))]
    
    if (length(corrupted) > 0L)
      stop("CRITICAL: Binary columns corrupted by transform: ",
           paste(corrupted, collapse = ", "))
    message(sprintf("  Binary integrity check : OK (%d col(s) verified)",
                    length(binary_check_cols)))
    
  } else {
    Train_Transformed <- copy(Train)
    Test_Transformed  <- copy(Test)
    message("  Skipping quantile transformation (QUANTILE_TRANSFORM = FALSE).")
  }
  
  message("02D complete.")
  
}, error = function(e) stop("02D failed: ", e$message))


#==============================================================================#
#==== E - Final Cleanup, Imputation & Validation ==============================#
#==============================================================================#

tryCatch({
  
  message("--- Starting 02E: Final Cleanup, Imputation & Validation ---")
  
  if (!is.data.table(Train_Transformed)) setDT(Train_Transformed)
  if (!is.data.table(Test_Transformed))  setDT(Test_Transformed)
  
  ## ── Structural NA imputation ───────────────────────────────────────────────
  ## These NAs are structural (cold-start): the first observation of a firm
  ## has no prior period to diff against, so yoy_/accel_/etc. are NA by design.
  
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
    matched <- grep(rule$pattern, names(Train_Transformed), value = TRUE)
    for (col in matched) {
      if (rule$strategy == "zero")   impute_fixed(col, 0)
      if (rule$strategy == "median") impute_median(col)
    }
  }
  
  ## Catch-all: any remaining numeric NAs not covered by the rules above
  remaining_na_cols <- names(Train_Transformed)[
    vapply(names(Train_Transformed), function(nm)
      is.numeric(Train_Transformed[[nm]]) &&
        (anyNA(Train_Transformed[[nm]]) || anyNA(Test_Transformed[[nm]])),
      logical(1L))
  ]
  already_handled <- unlist(lapply(imputation_rules, function(r)
    grep(r$pattern, names(Train_Transformed), value = TRUE)))
  catchall_cols <- setdiff(remaining_na_cols, already_handled)
  
  if (length(catchall_cols) > 0L) {
    ## [E1] Log all remaining columns, not just head(6)
    message(sprintf("  Catch-all median imputation: %d col(s):", length(catchall_cols)))
    message("    ", paste(catchall_cols, collapse = ", "))
    for (col in catchall_cols) impute_median(col)
  }
  
  remaining_train <- sum(is.na(Train_Transformed))
  remaining_test  <- sum(is.na(Test_Transformed))
  message(sprintf("  Remaining NAs — Train: %d | Test: %d",
                  remaining_train, remaining_test))
  
  if (remaining_train > 0L) {
    na_cols <- names(Train_Transformed)[sapply(Train_Transformed, anyNA)]
    message("  Columns with remaining NAs:")
    print(sort(sapply(na_cols, function(c)
      sum(is.na(Train_Transformed[[c]]))), decreasing = TRUE))
  }
  
  ## ── Drop metadata / leakage / zero-variance columns ───────────────────────
  drop_patterns <- c("^id$", "^company_id$", "^row_id$", "refdate", "^year$")
  pattern_drop  <- names(Train_Transformed)[vapply(names(Train_Transformed),
                                                   function(nm) any(vapply(drop_patterns, function(p) grepl(p, nm), logical(1L))),
                                                   logical(1L))]
  
  ## [E2] Use sd() instead of var() to avoid near-zero false positives on
  ## transformed columns (var can return 1e-32 on N(0,1)-ish columns).
  zerovar_drop <- names(Train_Transformed)[vapply(names(Train_Transformed),
                                                  function(nm) {
                                                    x <- Train_Transformed[[nm]]
                                                    is.numeric(x) && !anyNA(x) && sd(x) == 0
                                                  }, logical(1L))]
  
  if (length(zerovar_drop) > 0L)
    message("  Zero-variance columns dropped: ", paste(zerovar_drop, collapse = ", "))
  
  cols_to_drop <- setdiff(unique(c(pattern_drop, zerovar_drop)), TARGET_COL)
  
  ## ── Align categorical columns ──────────────────────────────────────────────
  cat_cols <- names(Train_Transformed)[
    sapply(Train_Transformed, function(x) is.character(x) || is.factor(x))
  ]
  cat_cols <- setdiff(cat_cols, c(TARGET_COL, cols_to_drop))
  
  for (col in cat_cols) {
    Train_Transformed[, (col) := as.factor(get(col))]
    train_levels <- levels(Train_Transformed[[col]])
    new_in_test  <- setdiff(as.character(unique(Test_Transformed[[col]])),
                            train_levels)
    if (length(new_in_test) > 0L) {
      warning(sprintf("  '%s': %d unseen Test level(s) → mode imputed", col,
                      length(new_in_test)))
    }
    Test_Transformed[, (col) := factor(get(col), levels = train_levels)]
    if (length(new_in_test) > 0L) {
      mode_val <- train_levels[which.max(tabulate(
        match(Train_Transformed[[col]], train_levels)))]
      Test_Transformed[is.na(get(col)), (col) := mode_val]
    }
  }
  
  ## ── Target enforcement ─────────────────────────────────────────────────────
  for (dt in list(Train_Transformed, Test_Transformed)) {
    if (TARGET_COL %in% names(dt))
      dt[, (TARGET_COL) := as.integer(as.character(get(TARGET_COL)))]
  }
  
  ## ── Build final sets ───────────────────────────────────────────────────────
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  Train_Final  <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final   <- copy(Test_Transformed[,  ..cols_to_keep])
  
  ## ── Validation report ──────────────────────────────────────────────────────
  col_mismatch <- setdiff(names(Train_Final), names(Test_Final))
  if (length(col_mismatch) > 0L)
    stop("CRITICAL: Column mismatch between Train_Final and Test_Final: ",
         paste(col_mismatch, collapse = ", "))
  
  all_feature_cols <- setdiff(names(Train_Final), TARGET_COL)
  stopifnot(
    "NAs in Train_Final" = sum(is.na(Train_Final)) == 0L,
    "NAs in Test_Final"  = sum(is.na(Test_Final))  == 0L
  )
  
  message("--- 02E Validation Report ---")
  message(sprintf("  Train_Final : %d rows x %d cols", nrow(Train_Final), ncol(Train_Final)))
  message(sprintf("  Test_Final  : %d rows x %d cols", nrow(Test_Final),  ncol(Test_Final)))
  message(sprintf("  Train default rate : %.3f%% (%d)",
                  100 * mean(Train_Final[[TARGET_COL]]),
                  sum(Train_Final[[TARGET_COL]])))
  message(sprintf("  Test  default rate : %.3f%% (%d)",
                  100 * mean(Test_Final[[TARGET_COL]]),
                  sum(Test_Final[[TARGET_COL]])))
  message(sprintf("  Train NAs : %d | Test NAs : %d",
                  sum(is.na(Train_Final)), sum(is.na(Test_Final))))
  message("  Column alignment : OK")
  message(sprintf("  Total modelling features : %d", length(all_feature_cols)))
  message("02E complete.")
  
}, error = function(e) stop("02E failed: ", e$message))