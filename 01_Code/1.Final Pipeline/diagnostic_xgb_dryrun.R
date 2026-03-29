#==============================================================================#
#==== diagnostic_xgb_dryrun.R =================================================#
#==============================================================================#
#
# PURPOSE:
#   Minimal dry-run to trigger and diagnose the memory alignment bug.
#
#==============================================================================#

message("\n[DIAGNOSTIC] Starting XGBoost Memory Alignment Check...")

## 1. Load config and library
tryCatch({
  source("config.R")
  library(xgboost)
  library(data.table)
}, error = function(e) {
  stop("Failed to load environment: ", e$message)
})

## 2. Load minimal data
tryCatch({
  data_path <- file.path("..", "..", "02_Data", "02_train_final_f_noTD_OoS.rds")
  message("  Loading: ", data_path)
  df <- readRDS(data_path)
  if (!is.data.table(df)) setDT(df)
  
  # Use full dataset
  message("  Data loaded. Full dataset used (", nrow(df), " rows).")
}, error = function(e) {
  stop("Data loading failed: ", e$message)
})

## 3. Data Formatting (The likely failure point)
tryCatch({
  message("\n[DIAGNOSTIC] Step 1: Formatting Matrix...")
  
  # Simulate potential non-contiguity by re-ordering columns
  feature_cols <- setdiff(names(df), TARGET_COL)
  y <- as.numeric(as.character(df[[TARGET_COL]]))
  # Randomly shuffle columns
  shuffled_cols <- sample(feature_cols)
  
  message("  Converting to matrix with TRANSPOSE (forcing non-contiguity)...")
  base_mat <- data.matrix(as.data.frame(df)[, feature_cols, drop = FALSE])
  # Transpose of a transpose is non-contiguous in R if not handled carefully, 
  # but actually t() always returns a non-contiguous matrix in terms of column-major order.
  train_mat <- t(t(base_mat)) 
  storage.mode(train_mat) <- "double"
  
  # Check for NAs in target
  if (any(is.na(y))) {
    message("  !!! WARNING: NAs found in target column. Imputing to 0 for diagnostic.")
    y[is.na(y)] <- 0
  }
  
  message("\n[DIAGNOSTIC] Step 2: Creating xgb.DMatrix...")
  dtrain <- xgb.DMatrix(data = train_mat, label = y)
  message("  DMatrix created successfully.")
  
}, error = function(e) {
  message("\n!!! ERROR CAPTURED IN MATRIX CONSTRUCTOR !!!")
  message("Message: ", e$message)
  message("Call trace:")
  print(traceback())
})

## 4. Minimal Training
tryCatch({
  if (exists("dtrain")) {
    message("\n[DIAGNOSTIC] Step 3: Minimal Training (nrounds=2)...")
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 3,
      eta = 0.1,
      nthread = parallel::detectCores() - 1L
    )
    model <- xgb.train(params = params, data = dtrain, nrounds = 2)
    message("  Training completed successfully.")
  }
}, error = function(e) {
  message("\n!!! ERROR CAPTURED IN TRAINING !!!")
  message("Message: ", e$message)
  message("Call trace:")
  print(traceback())
})

message("\n[DIAGNOSTIC] Dry-run complete.")
