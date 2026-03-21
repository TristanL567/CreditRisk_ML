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
#   a clean Data object ready for feature engineering.
#
# INPUTS:
#   PATH_DATA_FILE          — path to data.rda         (from config.R)
#   TARGET_COL              — name of the target col   (from config.R)
#
# OUTPUT:
#   Data                    — cleaned data.frame / data.table
#
# CHANGES FROM ORIGINAL:
#   [01] load() wrapper: original used Data <- load(...); Data <- d.
#        load() returns a character vector of object names, not the object.
#        The two-line idiom works but silently assumes the object inside the
#        .rda is always named "d". Replaced with a safe wrapper that loads
#        into a temporary environment and retrieves the first object by name,
#        making the assumption explicit and failure-safe.
#   [02] Column removal before drop_na: original called drop_na() first, then
#        removed "f*" columns. Any NA introduced by the raw financial position
#        columns ("f*") would cause rows to be dropped unnecessarily before
#        those columns were removed. Reordered to: drop "f*" → replace Inf →
#        drop_na(). This preserves more rows.
#   [03] Validation report: added a brief summary (rows, cols, default rate,
#        remaining NAs) so the stage is self-documenting in the console log.
#
#==============================================================================#


tryCatch({
  
  message("--- Starting 01_Data: Load & Preprocess ---")
  
  #============================================================================#
  #==== A - Load Raw Data ======================================================#
  #============================================================================#
  
  ## [CHANGE 01] Safe load wrapper.
  ## load() assigns objects into an environment and returns their names as a
  ## character vector. The original code relied on the object inside the .rda
  ## being named "d" (Data <- load(...) then Data <- d).
  ## This wrapper is explicit: it loads into a temporary env, retrieves the
  ## first object by name, and fails with a clear message if the file is empty.
  
  load_rda <- function(path) {
    tmp  <- new.env(parent = emptyenv())
    nms  <- load(path, envir = tmp)
    if (length(nms) == 0L)
      stop("No objects found in: ", path)
    if (length(nms) > 1L)
      message(sprintf(
        "  Note: .rda contains %d objects (%s). Using first: '%s'.",
        length(nms), paste(nms, collapse = ", "), nms[1L]
      ))
    get(nms[1L], envir = tmp)
  }
  
  Data <- load_rda(PATH_DATA_FILE)
  message(sprintf("  Loaded: %d rows x %d cols from '%s'",
                  nrow(Data), ncol(Data), basename(PATH_DATA_FILE)))
  
  
  #============================================================================#
  #==== B - Structural Preprocessing ===========================================#
  #============================================================================#
  
  ## External preprocessing function (sourced from Subfunctions/).
  ## Tolerance = 2 removes columns where more than 2% of values are missing
  ## or structurally problematic (exact behaviour defined inside the function).
  Data <- DataPreprocessing(Data, Tolerance = 2)
  
  ## [CHANGE 02] Column removal BEFORE Inf → NA and drop_na.
  ## Original order: replace Inf → drop_na → remove "f*" columns.
  ## Problem: "f*" (raw financial position) columns may contain NAs. Calling
  ## drop_na() before removing them discards rows that would have been fine
  ## after column removal. Correct order: remove columns → replace Inf → drop rows.
  
  ## Step 1: Drop raw financial position columns (prefix "f").
  f_cols <- grep("^f", colnames(Data), value = TRUE)
  if (length(f_cols) > 0L) {
    Data <- Data[, !colnames(Data) %in% f_cols, drop = FALSE]
    message(sprintf("  Dropped %d raw financial position column(s): %s%s",
                    length(f_cols),
                    paste(head(f_cols, 5L), collapse = ", "),
                    if (length(f_cols) > 5L) ", ..." else ""))
  }
  
  ## Step 2: Replace infinite values with NA (can arise from ratio calculations).
  Data <- Data %>%
    mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA_real_, .)))
  
  ## Step 3: Drop rows that still contain any NA.
  n_before <- nrow(Data)
  Data     <- tidyr::drop_na(Data)
  n_dropped <- n_before - nrow(Data)
  if (n_dropped > 0L)
    message(sprintf("  drop_na(): removed %d row(s) (%.2f%% of pre-drop rows)",
                    n_dropped, 100 * n_dropped / n_before))
  
  
  #============================================================================#
  #==== C - Validation Report ==================================================#
  #============================================================================#
  
  ## [CHANGE 03] Self-documenting summary — not present in original.
  n_defaults <- sum(Data[[TARGET_COL]] == 1L, na.rm = TRUE)
  message("--- 01_Data complete ---")
  message(sprintf("  Rows            : %d", nrow(Data)))
  message(sprintf("  Columns         : %d", ncol(Data)))
  message(sprintf("  Default rate    : %.3f%% (%d events)",
                  100 * n_defaults / nrow(Data), n_defaults))
  message(sprintf("  Remaining NAs   : %d", sum(is.na(Data))))
  
  stopifnot(
    "Target column missing from Data"  = TARGET_COL %in% colnames(Data),
    "NAs remain in Data after 01_Data" = sum(is.na(Data)) == 0L
  )
  
}, error = function(e) stop("01_Data failed: ", e$message))
