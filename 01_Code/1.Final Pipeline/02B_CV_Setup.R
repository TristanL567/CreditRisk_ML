#==============================================================================#
#==== 03_CV_Setup.R ===========================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Build stratified firm-level CV folds for use in 04A_Train_GLM.R and
#   04B_Train_XGBoost.R. Run once per SPLIT_MODE after 02_FeatureEngineering.R.
#
# DESIGN:
#   Stratification key: sector × y_ever  (mirrors the train/test split in 02A)
#   Unit of stratification: FIRM — all observations of a firm stay in one fold.
#   N_FOLDS: read from config.R (default 5).
#
#   This guarantees:
#     (a) No firm-level leakage between train and validation within any fold
#     (b) Default rate and sector distribution are balanced across folds
#     (c) CV folds are consistent with the train/test split philosophy
#
# INPUTS (from config.R / 02_FeatureEngineering.R):
#   02_train_final_{SPLIT_MODE}.rds     uniform features + y (Train_Final)
#   02_train_id_vec_{SPLIT_MODE}.rds    firm id vector aligned to Train_Final rows
#
# OUTPUTS:
#   cv_folds_{SPLIT_MODE}.rds           xgb.cv-compatible fold list
#                                       also contains firm-level fold assignment
#
# VARIABLES PRODUCED (in memory):
#   cv_folds       — list of N_FOLDS elements, each a vector of val row indices
#                    compatible with xgb.cv(folds = cv_folds)
#   cv_fold_info   — data.table: id, fold, y_ever, sector (for diagnostics)
#
#==============================================================================#

message(sprintf("\n══ 03_CV_Setup [SPLIT_MODE = %s | N_FOLDS = %d] ══",
                SPLIT_MODE, N_FOLDS))

tryCatch({
  
  #==============================================================================#
  #==== A - Load Inputs =========================================================#
  #==============================================================================#
  
  path_train <- get_split_path(SPLIT_OUT_TRAIN_FINAL)
  path_ids   <- get_split_path(SPLIT_OUT_TRAIN_IDS)
  
  stopifnot(
    "Train_Final .rds not found — run 02_FeatureEngineering.R first" =
      file.exists(path_train),
    "train_id_vec .rds not found — run 02_FeatureEngineering.R first" =
      file.exists(path_ids)
  )
  
  Train_Final  <- readRDS(path_train)
  train_id_vec <- readRDS(path_ids)
  
  if (!is.data.table(Train_Final)) setDT(Train_Final)
  
  stopifnot(
    "Row count mismatch: Train_Final vs train_id_vec" =
      nrow(Train_Final) == length(train_id_vec)
  )
  
  message(sprintf("  Loaded Train_Final : %d rows x %d cols", nrow(Train_Final), ncol(Train_Final)))
  message(sprintf("  train_id_vec       : %d values", length(train_id_vec)))
  
  
  #==============================================================================#
  #==== B - Reconstruct Firm-Level Metadata =====================================#
  #==============================================================================#
  
  ## Attach id to Train_Final (row-aligned, not a model feature)
  Train_Final[, .id := train_id_vec]
  
  ## ── y_ever ────────────────────────────────────────────────────────────────
  ## Recompute: a firm ever defaulted if any of its observations has y == 1.
  firm_y_ever <- Train_Final[, .(y_ever = as.integer(any(y == 1L))), by = .id]
  
  ## ── sector ────────────────────────────────────────────────────────────────
  ## Reconstruct from one-hot sector dummies (sector_*).
  ## Each row has exactly one sector_* == 1; the reference sector has all zeros
  ## and is recovered as "sector_ref".
  sector_cols <- grep("^sector_", names(Train_Final), value = TRUE)
  
  if (length(sector_cols) > 0L) {
    
    ## For each row, find the dummy column that equals 1.
    ## If no dummy is 1 → reference sector (labelled "sector_ref").
    sector_vec <- apply(
      Train_Final[, .SD, .SDcols = sector_cols], 1L,
      function(row) {
        hit <- which(row == 1L)
        if (length(hit) == 0L) "sector_ref"
        else sub("^sector_", "", sector_cols[hit[1L]])
      }
    )
    
    ## Take first (modal) sector per firm — firms don't change sector.
    firm_sector <- Train_Final[, .(sector = sector_vec[1L]), by = .id]
    
  } else {
    ## No sector dummies present (e.g. KEEP_FEATURES = "f") — use constant.
    warning("No sector_* columns found in Train_Final — CV stratification will use y_ever only.")
    firm_sector <- Train_Final[, .(sector = "unknown"), by = .id]
  }
  
  ## ── Firm profile ──────────────────────────────────────────────────────────
  firm_profile <- merge(firm_y_ever, firm_sector, by = ".id")
  setnames(firm_profile, ".id", "id")
  
  n_firms    <- nrow(firm_profile)
  n_defaults <- sum(firm_profile$y_ever)
  message(sprintf("  Firm profile       : %d firms | %d ever-defaulted (%.2f%%)",
                  n_firms, n_defaults, 100 * n_defaults / n_firms))
  
  
  #==============================================================================#
  #==== C - Stratified Firm-Level Fold Assignment ================================#
  #==============================================================================#
  
  ## Stratification key: sector × y_ever (same logic as 02A OoS split).
  firm_profile[, strat_key := interaction(sector, y_ever, drop = TRUE)]
  
  ## Guard: strata with < N_FOLDS firms cannot be split N ways.
  ## Merge small strata into an "other" stratum rather than dropping firms.
  strat_counts <- firm_profile[, .N, by = strat_key]
  small_strata <- strat_counts[N < N_FOLDS, strat_key]
  
  if (length(small_strata) > 0L) {
    n_affected <- firm_profile[strat_key %in% small_strata, .N]
    warning(sprintf(
      "  %d small stratum/a (< %d firms) merged into 'other': %s",
      length(small_strata), N_FOLDS,
      paste(as.character(small_strata), collapse = ", ")
    ))
    firm_profile[strat_key %in% small_strata, strat_key := factor("other")]
  }
  
  ## Assign folds within each stratum using round-robin to balance sizes.
  set.seed(SEED)
  firm_profile[, fold := {
    n   <- .N
    idx <- sample(n)           ## randomise within stratum
    ((idx - 1L) %% N_FOLDS) + 1L
  }, by = strat_key]
  
  ## Verify fold balance
  fold_summary <- firm_profile[, .(
    n_firms   = .N,
    n_default = sum(y_ever),
    pct_def   = round(100 * mean(y_ever), 2)
  ), by = fold][order(fold)]
  
  message("  Fold balance:")
  for (i in seq_len(nrow(fold_summary)))
    message(sprintf("    Fold %d : %d firms | %d defaults (%.2f%%)",
                    fold_summary$fold[i],
                    fold_summary$n_firms[i],
                    fold_summary$n_default[i],
                    fold_summary$pct_def[i]))
  
  
  #==============================================================================#
  #==== D - Map Firm Folds to Row Indices =======================================#
  #==============================================================================#
  
  ## Attach fold assignment to every row of Train_Final.
  row_folds <- firm_profile[, .(id, fold)]
  Train_Final[, .fold := row_folds[match(Train_Final$.id, row_folds$id), fold]]
  
  ## Verify no unmatched rows
  n_unmatched <- sum(is.na(Train_Final$.fold))
  if (n_unmatched > 0L)
    stop(sprintf("%d Train_Final rows could not be assigned a fold — check id alignment.", n_unmatched))
  
  ## Build xgb.cv-compatible fold list:
  ## Each element = integer vector of VALIDATION row indices for that fold.
  ## xgb.cv trains on all other rows and validates on these.
  cv_folds <- lapply(seq_len(N_FOLDS), function(k)
    which(Train_Final$.fold == k)
  )
  
  ## Sanity checks
  all_row_indices <- unlist(cv_folds)
  stopifnot(
    "Fold row indices don't cover all rows"    = length(all_row_indices) == nrow(Train_Final),
    "Duplicate row indices across folds"       = !anyDuplicated(all_row_indices),
    "Some folds are empty"                     = all(lengths(cv_folds) > 0L)
  )
  
  message(sprintf("  cv_folds built     : %d folds | row counts: %s",
                  N_FOLDS,
                  paste(lengths(cv_folds), collapse = " / ")))
  
  
  #==============================================================================#
  #==== E - Export & Save =======================================================#
  #==============================================================================#
  
  ## cv_fold_info: firm-level fold assignment for diagnostics / GLM CV
  cv_fold_info <- firm_profile[, .(id, fold, y_ever, sector,
                                   strat_key = as.character(strat_key))]
  
  ## Clean up temporary columns from Train_Final
  Train_Final[, c(".id", ".fold") := NULL]
  
  ## Save
  path_cv <- file.path(PATH_DATA_OUT,
                       sprintf("cv_folds_%s.rds", SPLIT_MODE))
  saveRDS(
    list(cv_folds    = cv_folds,
         cv_fold_info = cv_fold_info,
         n_folds      = N_FOLDS,
         split_mode   = SPLIT_MODE,
         seed         = SEED),
    path_cv
  )
  message(sprintf("  Saved: %s", basename(path_cv)))
  
  message("--- 03_CV_Setup complete ---")
  message(sprintf("  cv_folds       : list of %d fold index vectors", N_FOLDS))
  message(sprintf("  cv_fold_info   : %d firms with fold + sector + y_ever", nrow(cv_fold_info)))
  
}, error = function(e) stop("03_CV_Setup failed: ", e$message))