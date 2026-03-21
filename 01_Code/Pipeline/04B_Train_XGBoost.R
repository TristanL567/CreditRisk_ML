#==============================================================================#
#==== 04B_Train_XGBoost.R =====================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Train XGBoost with Bayesian HPO on four feature configurations,
#   evaluate on the held-out test set, and save a combined leaderboard.
#
# MODEL SELECTION (set at top):
#   M1 — "raw"       : Uniform(0,1) features from R pipeline (~508 features)
#   M2 — "latent"    : VAE latent dims + reconstruction error (z1..z32 + recon_error)
#   M3 — "anomaly"   : Reconstruction error only (vae_recon_error)
#   M4 — "augmented" : Raw uniform + VAE latent dims + recon error combined
#
# INPUTS (from config.R / prior stages):
#   02_train_final_{split}.rds          uniform train features
#   02_test_final_{split}.rds           uniform test features
#   02_train_id_vec_{split}.rds         firm id vector (train)
#   02_test_id_vec_{split}.rds          firm id vector (test)
#   cv_folds_{split}.rds                CV fold list from 03_CV_Setup.R
#   03_Output/Latent/latent_train_{split}.parquet
#   03_Output/Latent/latent_test_{split}.parquet
#   03_Output/Latent/anomaly_train_{split}.parquet
#   03_Output/Latent/anomaly_test_{split}.parquet
#
# OUTPUTS (→ 03_Output/Models/XGBoost/{MODEL}_{SPLIT_MODE}/):
#   xgb_model.rds          full result object (model + params + preds + metrics)
#   predictions_test.rds   data.table: id, y, p_default, model_name, split_mode
#   eval_summary.rds        metrics data.frame
#
# LEADERBOARD (→ 03_Output/Models/XGBoost/):
#   leaderboard_{split}.rds    combined metrics across all saved models
#
#==============================================================================#


#==============================================================================#
#==== 0 - Configuration  ← CHANGE HERE =======================================#
#==============================================================================#

## Select feature configuration
MODEL <- "M1"    ## "M1" | "M2" | "M3" | "M4"

## Select split (must match 02_FeatureEngineering.R and 03_CV_Setup.R)
## Note: SPLIT_MODE is already set in config.R — override here only if needed.
## SPLIT_MODE <- "OoS"

## XGBoost HPO budget — increase for production runs
XGB_N_INIT   <- XGB_CONFIG$n_init_points   ## from config.R (default 10)
XGB_N_ITER   <- XGB_CONFIG$n_iter_bayes    ## from config.R (default 20)
XGB_NROUNDS_BO    <- XGB_CONFIG$nrounds_bo
XGB_NROUNDS_FINAL <- XGB_CONFIG$nrounds_final
XGB_EARLY_BO      <- XGB_CONFIG$early_stop_bo
XGB_EARLY_FINAL   <- XGB_CONFIG$early_stop_final
XGB_NTHREAD       <- XGB_CONFIG$nthread
XGB_EVAL_METRIC   <- XGB_CONFIG$eval_metric   ## "auc"

assert_model <- function() {
  if (!MODEL %in% c("M1", "M2", "M3", "M4"))
    stop(sprintf("MODEL must be one of M1/M2/M3/M4, got: '%s'", MODEL))
}
assert_model()

MODEL_DESCRIPTIONS <- c(
  M1 = "Raw uniform(0,1) features (~508)",
  M2 = "VAE latent dims + reconstruction error",
  M3 = "VAE reconstruction error (anomaly score) only",
  M4 = "Raw features + VAE latent dims + reconstruction error (augmented)"
)

message(sprintf("\n══ 04B_Train_XGBoost [MODEL = %s | SPLIT_MODE = %s] ══",
                MODEL, SPLIT_MODE))
message(sprintf("   %s", MODEL_DESCRIPTIONS[MODEL]))


#==============================================================================#
#==== 1 - Paths & Directories =================================================#
#==============================================================================#

DIR_XGB_ROOT <- file.path(PATH_ROOT, "03_Output", "Models", "XGBoost")
DIR_LAT      <- file.path(PATH_ROOT, "03_Output", "Latent")
RUN_NAME     <- sprintf("%s_%s", MODEL, SPLIT_MODE)
DIR_RUN      <- file.path(DIR_XGB_ROOT, RUN_NAME)

dir.create(DIR_RUN,      recursive = TRUE, showWarnings = FALSE)
dir.create(DIR_XGB_ROOT, recursive = TRUE, showWarnings = FALSE)

## Helper: build latent file path
lat_path <- function(prefix)
  file.path(DIR_LAT, sprintf("%s_%s.parquet", prefix, SPLIT_MODE))


#==============================================================================#
#==== 2 - Load Data ===========================================================#
#==============================================================================#

message("  Loading data...")

## ── Always load raw train/test ────────────────────────────────────────────────
stopifnot(file.exists(get_split_path(SPLIT_OUT_TRAIN_FINAL)),
          file.exists(get_split_path(SPLIT_OUT_TEST_FINAL)))

Train_Final  <- readRDS(get_split_path(SPLIT_OUT_TRAIN_FINAL))
Test_Final   <- readRDS(get_split_path(SPLIT_OUT_TEST_FINAL))
train_id_vec <- readRDS(get_split_path(SPLIT_OUT_TRAIN_IDS))
test_id_vec  <- readRDS(get_split_path(SPLIT_OUT_TEST_IDS))

if (!is.data.table(Train_Final)) setDT(Train_Final)
if (!is.data.table(Test_Final))  setDT(Test_Final)

message(sprintf("  Train_Final : %d rows x %d cols", nrow(Train_Final), ncol(Train_Final)))
message(sprintf("  Test_Final  : %d rows x %d cols", nrow(Test_Final),  ncol(Test_Final)))

## ── Load CV folds ─────────────────────────────────────────────────────────────
path_cv <- file.path(PATH_DATA_OUT, sprintf("cv_folds_%s.rds", SPLIT_MODE))
stopifnot(
  "cv_folds not found — run 03_CV_Setup.R first" = file.exists(path_cv)
)
cv_obj   <- readRDS(path_cv)
cv_folds <- cv_obj$cv_folds
message(sprintf("  CV folds    : %d folds loaded", length(cv_folds)))

## ── Load latent / anomaly files (M2/M3/M4) ───────────────────────────────────
LAT_META <- c("id", "y")   ## columns to exclude from features

load_latent <- function(prefix) {
  p <- lat_path(prefix)
  stopifnot(sprintf("Latent file not found: %s\nRun 03_Autoencoder.py first.", p) = file.exists(p))
  dt <- as.data.table(arrow::read_parquet(p))
  message(sprintf("  %-30s: %d rows x %d cols", basename(p), nrow(dt), ncol(dt)))
  dt
}

if (MODEL %in% c("M2", "M4")) {
  lat_train <- load_latent("latent_train")
  lat_test  <- load_latent("latent_test")
}
if (MODEL %in% c("M3")) {
  ano_train <- load_latent("anomaly_train")
  ano_test  <- load_latent("anomaly_test")
}


#==============================================================================#
#==== 3 - Assemble Feature Matrices ===========================================#
#==============================================================================#

message("  Assembling feature matrices...")

assemble_data <- function(base_dt, id_vec, latent_dt = NULL,
                          use_base = TRUE, label = "") {
  ## Returns a data.table with features + y, no id column.
  dt <- copy(base_dt)
  
  if (!use_base) {
    ## M2/M3: features come entirely from the latent/anomaly file
    feat_cols <- setdiff(names(latent_dt), LAT_META)
    out <- latent_dt[, c("y", feat_cols), with = FALSE]
    message(sprintf("  [%s] %d feature cols (latent only)", label, length(feat_cols)))
    return(out)
  }
  
  if (!is.null(latent_dt)) {
    ## M4: join latent features onto raw by row-aligned id
    ## Add id to base for joining, then drop it
    dt[, .join_id := id_vec]
    latent_feats <- setdiff(names(latent_dt), LAT_META)
    latent_join  <- latent_dt[, c("id", latent_feats), with = FALSE]
    dt <- merge(dt, latent_join, by.x = ".join_id", by.y = "id", all.x = TRUE)
    dt[, .join_id := NULL]
    n_unmatched <- sum(is.na(dt[[latent_feats[1L]]]))
    if (n_unmatched > 0L)
      warning(sprintf("  [%s] %d unmatched rows after join — check id alignment",
                      label, n_unmatched))
    message(sprintf("  [%s] %d feature cols (raw + latent)", label,
                    ncol(dt) - 1L))
  } else {
    message(sprintf("  [%s] %d feature cols (raw only)", label, ncol(dt) - 1L))
  }
  
  dt
}

train_df <- switch(MODEL,
                   M1 = assemble_data(Train_Final, train_id_vec, label = "M1 train"),
                   M2 = assemble_data(Train_Final, train_id_vec, lat_train, use_base = FALSE, label = "M2 train"),
                   M3 = assemble_data(Train_Final, train_id_vec, ano_train, use_base = FALSE, label = "M3 train"),
                   M4 = assemble_data(Train_Final, train_id_vec, lat_train, use_base = TRUE,  label = "M4 train")
)

test_df <- switch(MODEL,
                  M1 = assemble_data(Test_Final, test_id_vec, label = "M1 test"),
                  M2 = assemble_data(Test_Final, test_id_vec, lat_test,  use_base = FALSE, label = "M2 test"),
                  M3 = assemble_data(Test_Final, test_id_vec, ano_test,  use_base = FALSE, label = "M3 test"),
                  M4 = assemble_data(Test_Final, test_id_vec, lat_test,  use_base = TRUE,  label = "M4 test")
)

## ── Build DMatrix ─────────────────────────────────────────────────────────────
feature_cols <- setdiff(names(train_df), TARGET_COL)

train_y <- as.integer(as.character(train_df[[TARGET_COL]]))
test_y  <- as.integer(as.character(test_df[[TARGET_COL]]))

options(na.action = "na.pass")
train_mat <- sparse.model.matrix(as.formula(paste(TARGET_COL, "~ . - 1")),
                                 data = train_df)
test_mat  <- sparse.model.matrix(as.formula(paste(TARGET_COL, "~ . - 1")),
                                 data = test_df)
options(na.action = "na.omit")

dtrain <- xgb.DMatrix(data = train_mat, label = train_y)
dtest  <- xgb.DMatrix(data = test_mat,  label = test_y)

## Class imbalance weight
n_neg            <- sum(train_y == 0L)
n_pos            <- sum(train_y == 1L)
scale_pos_weight <- n_neg / n_pos

message(sprintf("  Features    : %d", ncol(train_mat)))
message(sprintf("  Train rows  : %d | defaults: %d (%.3f%%)",
                length(train_y), n_pos, 100 * n_pos / length(train_y)))
message(sprintf("  Test rows   : %d | defaults: %d (%.3f%%)",
                length(test_y), sum(test_y), 100 * mean(test_y)))
message(sprintf("  scale_pos_weight: %.2f", scale_pos_weight))


#==============================================================================#
#==== 4 - Evaluation Helpers ==================================================#
#==============================================================================#

recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj  <- pROC::roc(y_true, y_pred, quiet = TRUE)
  coords   <- pROC::coords(roc_obj, "all",
                           ret = c("fpr", "tpr"), transpose = FALSE)
  eligible <- coords$fpr <= fpr_target
  if (!any(eligible)) return(0)
  max(coords$tpr[eligible])
}

brier_skill_score <- function(y_true, y_pred) {
  bs      <- mean((y_pred - y_true)^2)
  bs_clim <- mean(y_true) * (1 - mean(y_true))
  if (bs_clim == 0) return(NA_real_)
  round(1 - bs / bs_clim, 4)
}

compute_metrics <- function(y_true, y_pred, set_name) {
  valid <- !is.na(y_true) & !is.na(y_pred)
  yt <- y_true[valid]; yp <- y_pred[valid]
  if (length(unique(yt)) < 2L) return(NULL)
  
  roc_obj   <- pROC::roc(yt, yp, quiet = TRUE)
  auc_val   <- as.numeric(pROC::auc(roc_obj))
  brier_val <- mean((yp - yt)^2)
  
  ## Youden-optimal threshold for classification metrics
  opt_coords <- pROC::coords(roc_obj, "best",
                             ret = c("threshold", "sensitivity", "specificity"),
                             best.method = "youden")
  threshold  <- opt_coords$threshold[1L]
  pred_class <- as.integer(yp >= threshold)
  tp <- sum(yt == 1L & pred_class == 1L)
  fp <- sum(yt == 0L & pred_class == 1L)
  fn <- sum(yt == 1L & pred_class == 0L)
  precision <- tp / max(tp + fp, 1L)
  recall    <- tp / max(tp + fn, 1L)
  f1        <- if (precision + recall > 0)
    2 * precision * recall / (precision + recall) else 0
  
  data.frame(
    set            = set_name,
    model          = MODEL,
    split_mode     = SPLIT_MODE,
    n_obs          = length(yt),
    n_defaults     = sum(yt),
    prevalence     = round(mean(yt), 5),
    auc_roc        = round(auc_val, 4),
    avg_precision  = round(
      as.numeric(PRROC::pr.curve(yp[yt==1], yp[yt==0])$auc.integral), 4),
    brier          = round(brier_val, 5),
    bss            = brier_skill_score(yt, yp),
    recall_fpr1    = round(recall_at_fpr(yt, yp, 0.01), 4),
    recall_fpr3    = round(recall_at_fpr(yt, yp, 0.03), 4),
    recall_fpr5    = round(recall_at_fpr(yt, yp, 0.05), 4),
    recall_fpr10   = round(recall_at_fpr(yt, yp, 0.10), 4),
    threshold_opt  = round(threshold, 4),
    sensitivity    = round(opt_coords$sensitivity[1L], 4),
    specificity    = round(opt_coords$specificity[1L], 4),
    precision      = round(precision, 4),
    f1             = round(f1, 4),
    stringsAsFactors = FALSE
  )
}


#==============================================================================#
#==== 5 - Bayesian HPO + Final Model ==========================================#
#==============================================================================#

message(sprintf("\n  Starting Bayesian HPO (%d init + %d iter)...",
                XGB_N_INIT, XGB_N_ITER))

bo_iter <- 0L
total_iter <- XGB_N_INIT + XGB_N_ITER

time_bo <- system.time({
  
  xgb_bo_fn <- function(eta, max_depth, subsample, colsample_bytree,
                        min_child_weight, gamma, lambda, alpha) {
    bo_iter <<- bo_iter + 1L
    
    params <- list(
      booster          = "gbtree",
      objective        = "binary:logistic",
      eval_metric      = XGB_EVAL_METRIC,
      eta              = eta,
      max_depth        = as.integer(round(max_depth)),
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = as.integer(round(min_child_weight)),
      gamma            = gamma,
      lambda           = lambda,
      alpha            = alpha,
      scale_pos_weight = scale_pos_weight,
      max_delta_step   = 1L,
      nthread          = XGB_NTHREAD
    )
    
    cv_res <- tryCatch(
      xgb.cv(
        params                = params,
        data                  = dtrain,
        nrounds               = XGB_NROUNDS_BO,
        folds                 = cv_folds,
        early_stopping_rounds = XGB_EARLY_BO,
        verbose               = 0,
        maximize              = TRUE
      ),
      error = function(e) {
        message(sprintf("  [%02d/%02d] CV failed: %s",
                        bo_iter, total_iter, e$message))
        NULL
      }
    )
    
    if (is.null(cv_res)) return(list(Score = -Inf, Pred = 0))
    
    best_score <- max(cv_res$evaluation_log[[
      paste0("test_", XGB_EVAL_METRIC, "_mean")]])
    
    message(sprintf(
      "  [%02d/%02d] eta=%.3f depth=%d sub=%.2f col=%.2f mcw=%d "
      "gam=%.2f L2=%.2f L1=%.2f → %s=%.4f",
      bo_iter, total_iter, eta, as.integer(round(max_depth)),
      subsample, colsample_bytree, as.integer(round(min_child_weight)),
      gamma, lambda, alpha, XGB_EVAL_METRIC, best_score
    ))
    
    list(Score = best_score, Pred = 0)
  }
  
  bo_bounds <- list(
    eta              = c(0.01, 0.30),
    max_depth        = c(3L,   8L),
    subsample        = c(0.50, 1.00),
    colsample_bytree = c(0.50, 1.00),
    min_child_weight = c(1L,   20L),
    gamma            = c(0.00, 5.00),
    lambda           = c(0.00, 5.00),
    alpha            = c(0.00, 5.00)
  )
  
  bo_result <- tryCatch(
    rBayesianOptimization::BayesianOptimization(
      FUN         = xgb_bo_fn,
      bounds      = bo_bounds,
      init_points = XGB_N_INIT,
      n_iter      = XGB_N_ITER,
      acq         = "ucb",
      kappa       = 2.576,
      verbose     = FALSE
    ),
    error = function(e) stop("Bayesian Optimisation failed: ", e$message)
  )
  
}) ## end system.time

## ── Parse BO results ──────────────────────────────────────────────────────────
bo_history <- bo_result$History %>%
  dplyr::rename(Score = Value) %>%
  dplyr::mutate(
    max_depth        = as.integer(round(max_depth)),
    min_child_weight = as.integer(round(min_child_weight))
  ) %>%
  dplyr::arrange(dplyr::desc(Score))

message(sprintf("\n  BO complete (%.1fs) | Best %s: %.4f",
                time_bo["elapsed"], XGB_EVAL_METRIC, bo_history$Score[1L]))
message("  Top 3 configurations:")
print(head(bo_history[, c("Score", "eta", "max_depth", "subsample",
                          "colsample_bytree", "min_child_weight",
                          "gamma", "lambda", "alpha")], 3L),
      row.names = FALSE)

## ── Final model: optimal params + optimal rounds ──────────────────────────────
best <- bo_history[1L, ]

final_params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = XGB_EVAL_METRIC,
  eta              = best$eta,
  max_depth        = best$max_depth,
  subsample        = best$subsample,
  colsample_bytree = best$colsample_bytree,
  min_child_weight = best$min_child_weight,
  gamma            = best$gamma,
  lambda           = best$lambda,
  alpha            = best$alpha,
  scale_pos_weight = scale_pos_weight,
  max_delta_step   = 1L,
  nthread          = XGB_NTHREAD
)

message("\n  Determining optimal nrounds with final params...")
cv_final <- xgb.cv(
  params                = final_params,
  data                  = dtrain,
  nrounds               = XGB_NROUNDS_FINAL,
  folds                 = cv_folds,
  early_stopping_rounds = XGB_EARLY_FINAL,
  verbose               = 0,
  maximize              = TRUE
)

metric_col    <- paste0("test_", XGB_EVAL_METRIC, "_mean")
optimal_round <- cv_final$evaluation_log[
  which.max(cv_final$evaluation_log[[metric_col]]), iter]
cv_score_mean <- max(cv_final$evaluation_log[[metric_col]])
cv_score_sd   <- cv_final$evaluation_log[
  which.max(cv_final$evaluation_log[[metric_col]]),
  get(paste0("test_", XGB_EVAL_METRIC, "_std"))]

message(sprintf("  Optimal rounds : %d | CV %s: %.4f (+/- %.4f)",
                optimal_round, XGB_EVAL_METRIC, cv_score_mean, cv_score_sd))

## Train final model on full training set
model_final <- xgb.train(
  params  = final_params,
  data    = dtrain,
  nrounds = optimal_round,
  verbose = 0
)


#==============================================================================#
#==== 6 - Test Set Evaluation =================================================#
#==============================================================================#

message("\n  Evaluating on test set...")

## ── Column alignment safety ───────────────────────────────────────────────────
## Test matrix may have different dummy columns than train if factor levels differ.
train_features <- model_final$feature_names

missing_cols <- setdiff(train_features, colnames(test_mat))
if (length(missing_cols) > 0L) {
  message(sprintf("  Adding %d missing cols to test matrix (zero-filled)", length(missing_cols)))
  zero_mat <- Matrix::sparseMatrix(
    i = integer(0), j = integer(0),
    dims     = c(nrow(test_mat), length(missing_cols)),
    dimnames = list(NULL, missing_cols)
  )
  test_mat <- cbind(test_mat, zero_mat)
}
extra_cols <- setdiff(colnames(test_mat), train_features)
if (length(extra_cols) > 0L) {
  message(sprintf("  Dropping %d extra cols from test matrix", length(extra_cols)))
  test_mat <- test_mat[, !colnames(test_mat) %in% extra_cols, drop = FALSE]
}
test_mat <- test_mat[, train_features, drop = FALSE]
dtest    <- xgb.DMatrix(data = test_mat, label = test_y)

preds_train <- predict(model_final, dtrain)
preds_test  <- predict(model_final, dtest)

metrics_cv <- data.frame(
  set = "cv_train", model = MODEL, split_mode = SPLIT_MODE,
  auc_roc = NA_real_, avg_precision = round(cv_score_mean, 4),
  bss = NA_real_, brier = NA_real_,
  recall_fpr1 = NA_real_, recall_fpr3 = NA_real_,
  recall_fpr5 = NA_real_, recall_fpr10 = NA_real_,
  stringsAsFactors = FALSE
)

metrics_train <- compute_metrics(train_y, preds_train, "train_insample")
metrics_test  <- compute_metrics(test_y,  preds_test,  "test")

eval_table <- dplyr::bind_rows(metrics_cv, metrics_train, metrics_test)

message("\n  ── Evaluation Summary ──────────────────────────────────────")
message(sprintf("  CV %s (train)  : %.4f (+/- %.4f)",
                XGB_EVAL_METRIC, cv_score_mean, cv_score_sd))
if (!is.null(metrics_test)) {
  message(sprintf("  Test AUC-ROC   : %.4f", metrics_test$auc_roc))
  message(sprintf("  Test Avg Prec  : %.4f", metrics_test$avg_precision))
  message(sprintf("  Test Brier     : %.5f", metrics_test$brier))
  message(sprintf("  Test BSS       : %.4f", metrics_test$bss))
  message(sprintf("  Test R@FPR1%%   : %.4f", metrics_test$recall_fpr1))
  message(sprintf("  Test R@FPR3%%   : %.4f", metrics_test$recall_fpr3))
  message(sprintf("  Test R@FPR5%%   : %.4f", metrics_test$recall_fpr5))
  message(sprintf("  Test R@FPR10%%  : %.4f", metrics_test$recall_fpr10))
  message(sprintf("  Youden thresh  : %.4f  (Sens: %.3f | Spec: %.3f)",
                  metrics_test$threshold_opt,
                  metrics_test$sensitivity,
                  metrics_test$specificity))
}


#==============================================================================#
#==== 7 - Feature Importance ==================================================#
#==============================================================================#

importance_mat <- xgb.importance(
  feature_names = model_final$feature_names,
  model         = model_final
)

message("\n  Top 10 features by Gain:")
top10 <- head(importance_mat[order(-importance_mat$Gain), ], 10L)

## Apply human-readable labels from config.R FEATURE_MAP where available
top10$Label <- ifelse(top10$Feature %in% names(FEATURE_MAP),
                      FEATURE_MAP[top10$Feature],
                      top10$Feature)
print(top10[, c("Label", "Gain", "Cover", "Frequency")], row.names = FALSE)


#==============================================================================#
#==== 8 - Assemble Predictions Output =========================================#
#==============================================================================#

preds_out <- data.table(
  id         = test_id_vec,
  y          = test_y,
  p_default  = preds_test,
  model_name = MODEL,
  split_mode = SPLIT_MODE
)

## Add year if available (not in feature matrix — check raw Test_Final)
if ("year" %in% names(Test_Final)) {
  preds_out[, year := Test_Final$year]
} else {
  preds_out[, year := NA_integer_]
}

stopifnot(
  "NAs in predictions" = !anyNA(preds_out$p_default),
  "Predictions out of [0,1]" = all(between(preds_out$p_default, 0, 1))
)


#==============================================================================#
#==== 9 - Save ================================================================#
#==============================================================================#

message(sprintf("\n  Saving outputs → %s", DIR_RUN))

## Full result object
result_obj <- list(
  model            = model_final,
  optimal_rounds   = optimal_round,
  params           = final_params,
  bo_history       = bo_history,
  cv_score_mean    = cv_score_mean,
  cv_score_sd      = cv_score_sd,
  cv_metric        = XGB_EVAL_METRIC,
  eval_table       = eval_table,
  importance       = importance_mat,
  preds_train      = data.table(id = train_id_vec, y = train_y, p_default = preds_train),
  preds_test       = preds_out,
  scale_pos_weight = scale_pos_weight,
  model_name       = MODEL,
  split_mode       = SPLIT_MODE,
  n_features       = ncol(train_mat),
  time_bo          = time_bo
)

saveRDS(result_obj, file.path(DIR_RUN, "xgb_model.rds"))
saveRDS(preds_out,  file.path(DIR_RUN, "predictions_test.rds"))
saveRDS(eval_table, file.path(DIR_RUN, "eval_summary.rds"))

message(sprintf("  xgb_model.rds"))
message(sprintf("  predictions_test.rds  (%d rows)", nrow(preds_out)))
message(sprintf("  eval_summary.rds"))


#==============================================================================#
#==== 10 - Leaderboard (all saved models) =====================================#
#==============================================================================#

## Load eval tables from all saved model runs for this SPLIT_MODE
all_models  <- c("M1", "M2", "M3", "M4")
saved_evals <- lapply(all_models, function(m) {
  p <- file.path(DIR_XGB_ROOT, sprintf("%s_%s", m, SPLIT_MODE), "eval_summary.rds")
  if (file.exists(p)) readRDS(p) else NULL
})
saved_evals <- dplyr::bind_rows(Filter(Negate(is.null), saved_evals))

if (nrow(saved_evals) > 0L) {
  test_rows <- saved_evals[saved_evals$set == "test", ]
  base_auc  <- test_rows$auc_roc[test_rows$model == "M1"]
  
  if (length(base_auc) == 1L) {
    test_rows$uplift_auc_pct <- round(
      (test_rows$auc_roc - base_auc) / base_auc * 100, 2)
  }
  
  test_rows <- test_rows[order(-test_rows$auc_roc), ]
  
  message("\n══ XGBoost Leaderboard ════════════════════════════════════════")
  message(sprintf("  %-12s | %-8s | %-8s | %-8s | %-7s | %-7s | %-7s",
                  "Model", "AUC-ROC", "Avg Prec", "Brier", "BSS", "R@FPR3", "Uplift%"))
  message(sprintf("  %s", strrep("-", 75)))
  for (i in seq_len(nrow(test_rows))) {
    r <- test_rows[i, ]
    uplift_str <- if (!is.null(r$uplift_auc_pct) && !is.na(r$uplift_auc_pct))
      sprintf("%+.2f%%", r$uplift_auc_pct) else "  base"
    message(sprintf("  %-12s | %-8.4f | %-8.4f | %-8.5f | %-7.4f | %-7.4f | %s",
                    r$model, r$auc_roc, r$avg_precision,
                    r$brier, r$bss, r$recall_fpr3, uplift_str))
  }
  
  saveRDS(saved_evals,
          file.path(DIR_XGB_ROOT,
                    sprintf("leaderboard_%s.rds", SPLIT_MODE)))
  message(sprintf("\n  Leaderboard saved: leaderboard_%s.rds", SPLIT_MODE))
}

message(sprintf("\n══ 04B_Train_XGBoost complete [%s / %s] ══", MODEL, SPLIT_MODE))
message(sprintf("   CV %s  : %.4f (+/- %.4f)",
                XGB_EVAL_METRIC, cv_score_mean, cv_score_sd))
if (!is.null(metrics_test))
  message(sprintf("   Test AUC : %.4f | AP: %.4f | BSS: %.4f",
                  metrics_test$auc_roc, metrics_test$avg_precision, metrics_test$bss))