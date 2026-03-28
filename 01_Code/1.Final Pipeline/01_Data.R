#==============================================================================#
#==== 01_Data.R ===============================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Load the raw panel dataset, run structural preprocessing, and produce
#   a clean Data object ready for feature engineering in 02_FeatureEngineering.R.
#
# INPUTS  (all from config.R):
#   PATH_DATA_FILE   — path to data.rda
#   TARGET_COL       — name of the binary default target column
#
# OUTPUT:
#   Data             — clean data.table, passed in memory to next stage.
#                      No NAs, no duplicates.
#
#==============================================================================#

## KEEP_FEATURES is set in config.R:
##   "r"    — keep only ratio features (r1–r18); drop raw financials (f*)
##   "f"    — keep only raw financial position features (f*); drop ratios (r*)
##   "both" — keep all features
if (!exists("KEEP_FEATURES"))
  stop("KEEP_FEATURES not found — source config.R before running 01_Data.R.")

message("--- Starting 01_Data: Load & Preprocess ---")

tryCatch({
  
  #==============================================================================#
  #==== A - Load =================================================================#
  #==============================================================================#
  
  ## Safe load() wrapper — retrieves first object from .rda without assuming name.
  load_rda <- function(path) {
    tmp <- new.env(parent = emptyenv())
    nms <- load(path, envir = tmp)
    if (length(nms) == 0L)
      stop("No objects found in: ", path)
    if (length(nms) > 1L)
      message(sprintf("  Note: .rda contains %d objects. Using first: '%s'.",
                      length(nms), nms[1L]))
    get(nms[1L], envir = tmp)
  }
  
  Data <- load_rda(PATH_DATA_FILE)
  if (!is.data.table(Data)) setDT(Data)
  
  message(sprintf("  Loaded: %d rows x %d cols  [%s]",
                  nrow(Data), ncol(Data), basename(PATH_DATA_FILE)))
  
  ### Add the manually adjusted ratios (correct ones, see documentation).
  Data$r19<-Data$f6/Data$f1
  Data$r20<-(Data$f3-Data$f11)/Data$f1
  setcolorder(Data, c(setdiff(names(Data), "y"), "y"))
  
  #==============================================================================#
  #==== B - DataPreprocessing ====================================================#
  #==============================================================================#
  
  required_cols <- c("id", "refdate", TARGET_COL, "sector", "size")
  ncol_before   <- ncol(Data)
  
  Data <- DataPreprocessing(Data, Tolerance = 2)
  if (!is.data.table(Data)) setDT(Data)
  
  message(sprintf("  DataPreprocessing(): %d -> %d cols  (%d removed)",
                  ncol_before, ncol(Data), ncol_before - ncol(Data)))
  
  missing_req <- setdiff(required_cols, names(Data))
  if (length(missing_req) > 0L)
    stop("DataPreprocessing() dropped required columns: ",
         paste(missing_req, collapse = ", "))
  
  
  #==============================================================================#
  #==== C - Structural Cleaning ==================================================#
  #==============================================================================#
  
  ## Step 1: Drop feature families according to KEEP_FEATURES.
  ##         Must happen before NA handling — dropped cols may contain NAs
  ##         that would otherwise cause rows to be dropped unnecessarily.
  if (!KEEP_FEATURES %in% c("r", "f", "both"))
    stop('KEEP_FEATURES must be one of: "r", "f", "both".')
  
  f_cols <- grep("^f[0-9]", names(Data), value = TRUE)
  r_cols <- grep("^r[0-9]", names(Data), value = TRUE)
  
  drop_cols <- switch(KEEP_FEATURES,
                      "r"    = f_cols,
                      "f"    = r_cols,
                      "both" = character(0L)
  )
  
  if (length(drop_cols) > 0L) {
    Data[, (drop_cols) := NULL]
    message(sprintf("  KEEP_FEATURES = '%s' — removed %d col(s): %s%s",
                    KEEP_FEATURES,
                    length(drop_cols),
                    paste(head(drop_cols, 6L), collapse = ", "),
                    if (length(drop_cols) > 6L) ", ..." else ""))
  } else {
    message("  KEEP_FEATURES = 'both' — retaining all f* and r* cols.")
  }
  
  ## Step 2: Replace Inf/-Inf with NA.
  inf_cols <- names(Data)[sapply(Data, function(x)
    is.numeric(x) && any(is.infinite(x)))]
  if (length(inf_cols) > 0L) {
    Data[, (inf_cols) := lapply(.SD, function(x)
      replace(x, is.infinite(x), NA_real_)), .SDcols = inf_cols]
    message(sprintf("  Replaced Inf in %d col(s): %s",
                    length(inf_cols), paste(inf_cols, collapse = ", ")))
  }
  
  ## Step 3: NA diagnostic.
  na_counts <- sort(sapply(names(Data), function(nm) sum(is.na(Data[[nm]]))),
                    decreasing = TRUE)
  na_counts <- na_counts[na_counts > 0L]
  
  if (length(na_counts) > 0L) {
    n_rows_affected <- nrow(Data) - nrow(na.omit(Data))
    message(sprintf("  NA report (%d col(s) affected, %d row(s) will be dropped):",
                    length(na_counts), n_rows_affected))
    for (nm in names(na_counts))
      message(sprintf("    %-28s: %d NA(s)  (%.2f%%)",
                      nm, na_counts[[nm]], 100 * na_counts[[nm]] / nrow(Data)))
  }
  
  ## Step 4: Drop rows with remaining NAs.
  n_before  <- nrow(Data)
  Data      <- na.omit(Data)
  n_dropped <- n_before - nrow(Data)
  
  if (n_dropped > 0L)
    message(sprintf("  na.omit(): removed %d row(s)  (%.2f%%)",
                    n_dropped, 100 * n_dropped / n_before))
  else
    message("  na.omit(): no rows removed.")
  
  
  #==============================================================================#
  #==== D - Panel Validation =====================================================#
  #==============================================================================#
  
  setorder(Data, id, refdate)
  
  ## Assert (id, refdate) uniqueness — duplicates cause silent errors in shift().
  n_dup <- anyDuplicated(Data[, .(id, refdate)])
  if (n_dup > 0L) {
    dup_sample <- Data[
      duplicated(Data[, .(id, refdate)]) |
        duplicated(Data[, .(id, refdate)], fromLast = TRUE),
      .(id, refdate)
    ]
    message("  Sample duplicate keys:")
    print(head(dup_sample, 10L))
    stop(sprintf("Panel key violation: %d duplicate (id, refdate) pair(s).", n_dup))
  }
  
  ## Panel summary.
  obs_per_firm <- Data[, .N, by = id]
  message(sprintf(
    "  Panel summary: %d firms | obs/firm — median: %.1f  min: %d  max: %d",
    uniqueN(Data$id), median(obs_per_firm$N),
    min(obs_per_firm$N), max(obs_per_firm$N)
  ))
  
  ## Mixed-frequency check.
  Data[, .obs_gap := as.numeric(
    difftime(refdate, shift(refdate, 1L, type = "lag"), units = "days")
  ), by = id]
  
  gap_vals <- Data[!is.na(.obs_gap), .obs_gap]
  message(sprintf("  Obs gap (days) — median: %.0f  min: %.0f  max: %.0f",
                  median(gap_vals), min(gap_vals), max(gap_vals)))
  
  if (min(gap_vals) < 300L && max(gap_vals) >= 300L)
    warning(
      "Mixed reporting frequencies detected. ",
      "Lag-1 features in 02_FeatureEngineering will be period-over-period ",
      "for sub-annual reporters, NOT year-over-year."
    )
  
  Data[, .obs_gap := NULL]
  
  
  #==============================================================================#
  #==== E - Exit Validation ======================================================#
  #==============================================================================#
  
  stopifnot(
    "TARGET_COL missing after 01_Data" = TARGET_COL %in% names(Data),
    "NAs remain after 01_Data"         = sum(is.na(Data)) == 0L,
    "Data is empty after 01_Data"      = nrow(Data) > 0L
  )
  
  n_defaults <- sum(Data[[TARGET_COL]] == 1L, na.rm = TRUE)
  
  message("--- 01_Data complete ---")
  message(sprintf("  Rows         : %d", nrow(Data)))
  message(sprintf("  Columns      : %d", ncol(Data)))
  message(sprintf("  Default rate : %.3f%%  (%d events)",
                  100 * n_defaults / nrow(Data), n_defaults))
  
}, error = function(e) stop("01_Data failed: ", e$message))