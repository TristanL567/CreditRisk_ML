#==============================================================================#
#==== 10_GLM.R ================================================================#
#==============================================================================#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Anastasia Pryshchepa
# UPDATED:  2026
#
# PURPOSE:
#   Regularised GLM (glmnet, elastic-net) for all five strategies.
#   Fully standalone — loads all data from disk, all helper functions
#   are defined inline. No other pipeline script needs to be in memory.
#
# FOLDER STRUCTURE EXPECTED:
#   /Users/admin/Desktop/CreditRisk_ML/
#   └── 02_Pipeline Datasets and Results/
#       ├── 02_Data/
#       │   ├── 02_train_final_OoS.rds
#       │   ├── 02_test_final_OoS.rds
#       │   ├── 02_train_id_vec_OoS.rds
#       │   └── 02_test_id_vec_OoS.rds
#       └── 03_Output/
#           └── Latent/
#               ├── latent_train_OoS.parquet
#               ├── latent_test_OoS.parquet
#               ├── anomaly_train_OoS.parquet
#               └── anomaly_test_OoS.parquet
#
# OUTPUTS (saved to DIR_DATA):
#   GLM_Leaderboard_OoS.xlsx
#   GLM_Variable_Importance_OoS.xlsx
#
#==============================================================================#


#==============================================================================#
#==== 00 - Parameters & Paths =================================================#
#==============================================================================#

## ── Parameters: guarded so runner overrides are not clobbered ────────────────
## [FIX 2026-03-30: unconditional assignments overwrote values set by runner]
if (!exists("SPLIT_MODE"))            SPLIT_MODE            <- "OoS"
if (!exists("SEED"))                  SEED                  <- 123L
if (!exists("TARGET_COL"))            TARGET_COL            <- "y"
if (!exists("MODEL_GROUP"))           MODEL_GROUP           <- "01"
if (!exists("KEEP_FEATURES"))         KEEP_FEATURES         <- "f"
if (!exists("INCLUDE_TIME_DYNAMICS")) INCLUDE_TIME_DYNAMICS <- FALSE

## ── Paths: use runner-supplied config.R values if present ─────────────────────
## [FIX 2026-03-30: replaced hardcoded Mac path "/Users/admin/Desktop/..."]
if (!exists("PATH_DATA_OUT")) {
  suppressPackageStartupMessages(library(here))
  PATH_ROOT     <- here::here("")
  PATH_DATA_OUT <- file.path(PATH_ROOT, "02_Data")
  DIR_FINAL_OUT <- file.path(PATH_ROOT, "03_Output", "Final")
}
DIR_DATA   <- PATH_DATA_OUT
DIR_LATENT <- file.path(dirname(DIR_FINAL_OUT), "Latent")   ## 03_Output/Latent/

## ── Output folder: 03_Output/Final/<GROUP><a|b>_GLM/ ─────────────────────────
.split_abbrev <- ifelse(SPLIT_MODE == "OoS", "a", "b")
DIR_GLM_OUT   <- file.path(DIR_FINAL_OUT,
                            paste0(MODEL_GROUP, .split_abbrev, "_GLM"))
dir.create(DIR_GLM_OUT, recursive = TRUE, showWarnings = FALSE)
rm(.split_abbrev)

## ── GLM tuning parameters ────────────────────────────────────────────────────
N_FOLDS_GLM   <- 5L    ## CV folds
N_INIT_POINTS <- 1L    ## reduced for speed
N_ITER_BAYES  <- 2L    ## reduced for speed
TOP_N_COEF    <- 20L   ## top-N coefficients to report per strategy

message("\n══ 10_GLM: Regularised GLM Pipeline ════════════════════════")


#==============================================================================#
#==== 01 - Packages ===========================================================#
#==============================================================================#

glm_packages <- c(
  "data.table", "dplyr", "tibble", "purrr",
  "glmnet", "pROC",
  "arrow",        ## read/write parquet files
  "ggplot2",      ## leaderboard chart
  "scales",       ## axis formatting
  "openxlsx"
)

for (pkg in glm_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
message("  Packages loaded.")


#==============================================================================#
#==== 02 - Load Data from Disk ================================================#
#==============================================================================#

message("\n── Loading data from disk ───────────────────────────────────")

## ── Helper: file path with split suffix ──────────────────────────────────────
## [FIX 2026-03-30: was paste0(base_name, "_", SPLIT_MODE, ".rds") — missing
##  KEEP_FEATURES and INCLUDE_TIME_DYNAMICS components, so files like
##  02_train_final_f_noTD_OoS.rds were never found]
split_path <- function(dir, base_name) {
  td <- ifelse(INCLUDE_TIME_DYNAMICS, "TD", "noTD")
  file.path(dir, paste0(base_name, "_", KEEP_FEATURES, "_", td, "_", SPLIT_MODE, ".rds"))
}

## ── 02A: RDS files from 02_Data/ ─────────────────────────────────────────────
stopifnot(
  "02_Data folder not found — check DIR_DATA path" = dir.exists(DIR_DATA),
  "02_train_final not found"   = file.exists(split_path(DIR_DATA, "02_train_final")),
  "02_test_final not found"    = file.exists(split_path(DIR_DATA, "02_test_final")),
  "02_train_id_vec not found"  = file.exists(split_path(DIR_DATA, "02_train_id_vec")),
  "02_test_id_vec not found"   = file.exists(split_path(DIR_DATA, "02_test_id_vec"))
)

Train_Final  <- readRDS(split_path(DIR_DATA, "02_train_final"))
Test_Final   <- readRDS(split_path(DIR_DATA, "02_test_final"))
train_id_vec <- readRDS(split_path(DIR_DATA, "02_train_id_vec"))
test_id_vec  <- readRDS(split_path(DIR_DATA, "02_test_id_vec"))

message(sprintf("  Train_Final  : %d rows x %d cols", nrow(Train_Final), ncol(Train_Final)))
message(sprintf("  Test_Final   : %d rows x %d cols", nrow(Test_Final),  ncol(Test_Final)))

## ── 02B: Parquet files from 03_Output/Latent/ (Groups 04 and 05 only) ────────
## ── 02B: For Groups 04/05, load latent parquet and build XGBoost-equivalent ───
##        feature matrix. Groups 01-03: no augmentation needed.
##
## [FIX 2026-03-30: Replaced multi-strategy approach (Base/A/B/C) with single
##  pre-assembled feature matrix that matches XGBoost/AutoGluon exactly:
##    Group 04 → base features + VAE latent features (cbind)
##    Group 05 → VAE latent features + categorical variables only
##  The _vae_ files in 02_Data/ are quantile-normalised VAE inputs, NOT merged
##  outputs; assembly from base + latent parquet (same as 04B_Train_XGBoost.R)
##  is the architecturally correct approach.]
HAS_LATENT <- MODEL_GROUP %in% c("04", "05")

if (HAS_LATENT) {
  .td         <- ifelse(INCLUDE_TIME_DYNAMICS, "TD", "noTD")
  .par_suffix <- paste0("_", KEEP_FEATURES, "_", .td, "_", SPLIT_MODE)

  stopifnot(
    "03_Output/Latent folder not found — check DIR_LATENT path" = dir.exists(DIR_LATENT),
    "latent_train parquet not found" = file.exists(
      file.path(DIR_LATENT, paste0("latent_train", .par_suffix, ".parquet"))),
    "latent_test parquet not found"  = file.exists(
      file.path(DIR_LATENT, paste0("latent_test",  .par_suffix, ".parquet")))
  )

  latent_train <- as.data.frame(arrow::read_parquet(
    file.path(DIR_LATENT, paste0("latent_train", .par_suffix, ".parquet"))))
  latent_test  <- as.data.frame(arrow::read_parquet(
    file.path(DIR_LATENT, paste0("latent_test",  .par_suffix, ".parquet"))))

  message(sprintf("  latent_train : %d rows x %d cols", nrow(latent_train), ncol(latent_train)))
  message(sprintf("  latent_test  : %d rows x %d cols", nrow(latent_test),  ncol(latent_test)))

  .lat_feats <- setdiff(names(latent_train), c("id", "y"))

  if (MODEL_GROUP == "04") {
    ## Group 04: base features + VAE latent features (mirrors XGBoost Group 04)
    Train_Final <- cbind(as.data.frame(Train_Final),
                         latent_train[, .lat_feats, drop = FALSE])
    Test_Final  <- cbind(as.data.frame(Test_Final),
                         latent_test[,  .lat_feats, drop = FALSE])
    message(sprintf("  Group 04 assembled : %d cols (base + %d latent)",
                    ncol(Train_Final), length(.lat_feats)))

  } else {  ## MODEL_GROUP == "05"
    ## Group 05: VAE latent features + categorical variables (mirrors XGBoost Group 05)
    .cat_cols <- grep("^sector_|^size_|^groupmember$|^public$",
                      names(Train_Final), value = TRUE)
    Train_Final <- cbind(
      setNames(data.frame(as.data.frame(Train_Final)[[TARGET_COL]]), TARGET_COL),
      latent_train[, .lat_feats, drop = FALSE],
      as.data.frame(Train_Final)[, .cat_cols, drop = FALSE]
    )
    Test_Final <- cbind(
      setNames(data.frame(as.data.frame(Test_Final)[[TARGET_COL]]), TARGET_COL),
      latent_test[,  .lat_feats, drop = FALSE],
      as.data.frame(Test_Final)[,  .cat_cols, drop = FALSE]
    )
    message(sprintf("  Group 05 assembled : %d cols (%d latent + %d categoricals)",
                    ncol(Train_Final), length(.lat_feats), length(.cat_cols)))
    rm(.cat_cols)
  }
  rm(.td, .par_suffix, .lat_feats)

} else {
  message(sprintf("  MODEL_GROUP=%s — base features only, no latent augmentation.",
                  MODEL_GROUP))
  latent_train <- NULL
  latent_test  <- NULL
}


## ── 02C: Single feature matrix — no separate strategies ──────────────────────
## [FIX 2026-03-30: All groups now train a single GLM on the assembled feature
##  matrix (same methodology as XGBoost/AutoGluon). Strategies A/B/C removed.]

Train_Final_df <- as.data.frame(Train_Final)
Test_Final_df  <- as.data.frame(Test_Final)

Train_Data_Base_Model <- Train_Final_df
Test_Data_Base_Model  <- Test_Final_df
message(sprintf("  Feature matrix : %d rows x %d cols", nrow(Train_Data_Base_Model),
                ncol(Train_Data_Base_Model)))

## Strategies A/B/C are unused — set to NULL so all downstream guards pass cleanly
Strategy_A_train <- NULL;  Strategy_A_test  <- NULL
Strategy_B_train <- NULL;  Strategy_B_test  <- NULL
Strategy_C_train <- NULL;  Strategy_C_test  <- NULL

message("── Data loading complete ─────────────────────────────────────")


#==============================================================================#
#==== 01 - Helper Functions ===================================================#
#==============================================================================#

## ── 01A: Align test columns/factors to train ─────────────────────────────────
##   Prevents model.matrix / factor-level mismatches when scoring.
align_test_to_train <- function(train_df, test_df) {
  ## Add any columns present in train but missing from test (fill with NA)
  miss <- setdiff(names(train_df), names(test_df))
  if (length(miss) > 0L)
    for (cc in miss) test_df[[cc]] <- NA

  ## Drop columns present in test but absent from train
  extra <- setdiff(names(test_df), names(train_df))
  if (length(extra) > 0L)
    test_df <- test_df[, setdiff(names(test_df), extra), drop = FALSE]

  ## Re-level factors to match train levels exactly
  fac_cols <- names(train_df)[vapply(train_df, is.factor, logical(1L))]
  for (cc in fac_cols) {
    if (cc %in% names(test_df))
      test_df[[cc]] <- factor(test_df[[cc]], levels = levels(train_df[[cc]]))
  }

  ## Enforce identical column order
  test_df[, names(train_df), drop = FALSE]
}


## ── 01B: GLM Training with cv.glmnet ─────────────────────────────────────────
##   Uses glmnet's built-in cross-validation (C-level, very fast) to find
##   the optimal lambda for each alpha. Then picks the best (alpha, lambda)
##   pair by CV AUC across a small alpha grid.
##   ~10-20x faster than manual Bayesian optimisation on large datasets.
##
##   Arguments:
##     Data_Train_CV_Vector  — integer fold vector (length = nrow(Train_Data))
##     Train_Data            — data.frame with features + TARGET_COL ("y")
##     alpha_grid            — vector of alpha values to search over
##     n_init_points         — ignored (kept for API compatibility)
##     n_iter_bayes          — ignored (kept for API compatibility)

GLM_Training <- function(Data_Train_CV_Vector,
                         Train_Data,
                         alpha_grid    = c(0, 0.25, 0.5, 0.75, 1),
                         n_init_points = NULL,
                         n_iter_bayes  = NULL) {

  ## ── Build model matrix ───────────────────────────────────────────────────
  X_full <- tryCatch({
    mm <- stats::model.matrix(y ~ ., data = Train_Data)
    mm <- mm[, -1L, drop = FALSE]   ## drop intercept
    if (ncol(mm) == 0L)
      stop("model.matrix produced zero feature columns after dropping intercept.")
    mm
  }, error = function(e) stop("model.matrix failed: ", e$message))

  y_full <- as.numeric(as.character(Train_Data[[TARGET_COL]]))

  ## ── Search over alpha grid using cv.glmnet ───────────────────────────────
  ## cv.glmnet runs k-fold CV internally in C — much faster than manual loops
  ## We reuse Data_Train_CV_Vector as the foldid so CV is consistent across
  ## all strategies (same firm-level fold assignment)

  best_auc    <- -Inf
  best_alpha  <- 0.5
  best_lambda <- 0.01
  best_cvfit  <- NULL

  ## ── Special case: single feature column (e.g. Strategy B: anomaly only) ──
  ## [FIX 2026-03-30: old code used glmnet() for both CV folds and final model,
  ##  but glmnet() requires >= 2 columns so the final fit always crashed with
  ##  "x should be a matrix with 2 or more columns".
  ##  GLM_Test() and get_top_coeffs_glmnet() already contain glm() branches for
  ##  exactly this case, so the architecturally correct fix is to use stats::glm()
  ##  (plain logistic regression) here and return early, bypassing the shared
  ##  glmnet final-model block below.]
  if (ncol(X_full) < 2L) {

    message("  Single-feature strategy detected — fitting plain logistic regression (glm).")

    ## Build a data.frame for glm() with the single predictor + response
    single_df        <- as.data.frame(X_full)
    single_df[[TARGET_COL]] <- y_full

    ## ── CV AUC via plain logistic regression ─────────────────────────────────
    n_folds   <- max(Data_Train_CV_Vector, na.rm = TRUE)
    fold_aucs <- numeric(n_folds)
    for (fold in seq_len(n_folds)) {
      idx_val   <- which(Data_Train_CV_Vector == fold)
      idx_train <- which(Data_Train_CV_Vector != fold)
      fit_cv <- tryCatch(
        stats::glm(stats::as.formula(paste(TARGET_COL, "~ .")),
                   data   = single_df[idx_train, , drop = FALSE],
                   family = binomial()),
        error = function(e) NULL)
      if (is.null(fit_cv)) { fold_aucs[fold] <- 0.5; next }
      preds_cv <- tryCatch(
        as.numeric(predict(fit_cv,
                           newdata = single_df[idx_val, , drop = FALSE],
                           type    = "response")),
        error = function(e) NULL)
      if (is.null(preds_cv) || all(is.na(preds_cv))) { fold_aucs[fold] <- 0.5; next }
      roc_cv <- tryCatch(pROC::roc(y_full[idx_val], preds_cv, quiet = TRUE),
                         error = function(e) NULL)
      fold_aucs[fold] <- if (!is.null(roc_cv)) as.numeric(pROC::auc(roc_cv)) else 0.5
    }
    best_auc <- mean(fold_aucs, na.rm = TRUE)
    message(sprintf("  >> Single-feature glm()  CV-AUC=%.4f", best_auc))

    ## ── Final model on full training data ────────────────────────────────────
    final_model <- tryCatch(
      stats::glm(stats::as.formula(paste(TARGET_COL, "~ .")),
                 data   = single_df,
                 family = binomial()),
      error = function(e) stop("Single-feature glm() failed: ", e$message)
    )

    ## ── Training-set Brier score & penalised Brier ────────────────────────────
    train_preds     <- as.numeric(predict(final_model, newdata = single_df,
                                          type = "response"))
    brier           <- mean((train_preds - y_full)^2, na.rm = TRUE)
    n_nonzero       <- sum(!is.na(coef(final_model)))
    penalised_brier <- brier * (1 + n_nonzero / length(y_full))

    return(list(
      optimal_model         = final_model,
      optimal_parameters    = list(alpha = NA_real_, lambda = NA_real_),
      results               = data.frame(AUC = best_auc),
      Brier_Score           = brier,
      Penalized_Brier_Score = penalised_brier,
      train_predictions     = train_preds
    ))

  } else {

    ## ── Standard case: use cv.glmnet over alpha grid ──────────────────────
    for (a in alpha_grid) {

      cvfit <- tryCatch(
        glmnet::cv.glmnet(
          x        = X_full,
          y        = y_full,
          family   = "binomial",
          alpha    = a,
          foldid   = Data_Train_CV_Vector,
          type.measure = "auc",
          nfolds   = max(Data_Train_CV_Vector, na.rm = TRUE),
          parallel = FALSE
        ),
        error = function(e) {
          message(sprintf("  cv.glmnet failed for alpha=%.2f: %s", a, e$message))
          NULL
        }
      )

      if (is.null(cvfit)) next

      cv_auc <- max(cvfit$cvm, na.rm = TRUE)
      message(sprintf("    alpha=%.2f  best_lambda=%.6f  CV-AUC=%.4f",
                      a, cvfit$lambda.min, cv_auc))

      if (cv_auc > best_auc) {
        best_auc    <- cv_auc
        best_alpha  <- a
        best_lambda <- cvfit$lambda.min
        best_cvfit  <- cvfit
      }
    }

    message(sprintf("  >> Best: alpha=%.2f  lambda=%.6f  CV-AUC=%.4f",
                    best_alpha, best_lambda, best_auc))
  }

  ## ── Final model on full training data ────────────────────────────────────
  final_model <- tryCatch(
    glmnet::glmnet(
      x      = X_full,
      y      = y_full,
      family = "binomial",
      alpha  = best_alpha,
      lambda = best_lambda
    ),
    error = function(e) stop("Final glmnet fit failed: ", e$message)
  )

  ## ── Training-set Brier score & penalised Brier ────────────────────────────
  train_preds     <- as.numeric(predict(final_model, newx = X_full,
                                        type = "response"))
  brier           <- mean((train_preds - y_full)^2, na.rm = TRUE)
  n_nonzero       <- sum(abs(as.numeric(coef(final_model))) > 0, na.rm = TRUE)
  penalised_brier <- brier * (1 + n_nonzero / length(y_full))

  list(
    optimal_model         = final_model,
    optimal_parameters    = list(alpha = best_alpha, lambda = best_lambda),
    results               = data.frame(AUC = best_auc),
    Brier_Score           = brier,
    Penalized_Brier_Score = penalised_brier,
    train_predictions     = train_preds
  )
}


## ── 01C: GLM Test-set Evaluation ─────────────────────────────────────────────
##   Handles both glmnet models (Base, A, C) and glm model (Strategy B).

GLM_Test <- function(Model, Test_Data) {
  if (is.null(Model))
    stop("GLM_Test: Model is NULL.")

  y_test <- as.numeric(as.character(Test_Data[[TARGET_COL]]))

  ## ── Detect model type and predict accordingly ─────────────────────────────
  if (inherits(Model, "glm")) {
    ## Strategy B: plain logistic regression fitted with glm()
    preds <- tryCatch(
      as.numeric(predict(Model, newdata = Test_Data, type = "response")),
      error = function(e) stop("GLM_Test predict.glm failed: ", e$message)
    )
    n_nonzero <- sum(!is.na(coef(Model)))
  } else {
    ## Base, A, C: elastic-net fitted with glmnet()
    X_test <- tryCatch(
      stats::model.matrix(y ~ ., data = Test_Data)[, -1L, drop = FALSE],
      error = function(e) stop("GLM_Test model.matrix failed: ", e$message)
    )
    preds <- tryCatch(
      as.numeric(predict(Model, newx = X_test, type = "response")),
      error = function(e) stop("GLM_Test predict.glmnet failed: ", e$message)
    )
    n_nonzero <- sum(abs(as.numeric(coef(Model))) > 0, na.rm = TRUE)
  }

  roc_obj <- pROC::roc(y_test, preds, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  brier   <- mean((preds - y_test)^2, na.rm = TRUE)
  penalised_brier <- brier * (1 + n_nonzero / length(y_test))

  list(
    AUC                   = auc_val,
    Brier_Score           = brier,
    Penalized_Brier_Score = penalised_brier,
    predictions           = preds,
    actuals               = y_test
  )
}


## ── 01D: Extract training-set metrics for leaderboard ────────────────────────
extract_metrics <- function(model_obj, model_name) {
  if (is.null(model_obj) || is.null(model_obj$results)) {
    warning(paste("Model", model_name, "missing — returning NA row."))
    return(data.frame(Model = model_name, Type = "Error",
                      AUC = NA_real_, Brier_Score = NA_real_,
                      Penalized_Brier = NA_real_,
                      Alpha = NA_real_, Lambda = NA_real_))
  }
  params <- model_obj$optimal_parameters
  data.frame(
    Model           = model_name,
    Type            = "GLM",
    AUC             = model_obj$results$AUC[1L],
    Brier_Score     = model_obj$Brier_Score,
    Penalized_Brier = model_obj$Penalized_Brier_Score,
    Alpha           = params$alpha,
    Lambda          = params$lambda
  )
}


## ── 01E: Extract test-set metrics for leaderboard ────────────────────────────
extract_test_metrics <- function(test_obj, test_df, train_obj, model_name) {
  if (is.null(test_obj)) {
    warning(paste("Test result for", model_name, "is NULL — returning NA row."))
    return(data.frame(Model = model_name, Type = "Error",
                      AUC = NA_real_, Brier_Score = NA_real_,
                      Penalized_Brier = NA_real_,
                      Alpha = NA_real_, Lambda = NA_real_))
  }
  params <- if (!is.null(train_obj)) train_obj$optimal_parameters else list(alpha = NA, lambda = NA)
  data.frame(
    Model           = model_name,
    Type            = "GLM",
    AUC             = test_obj$AUC,
    Brier_Score     = test_obj$Brier_Score,
    Penalized_Brier = test_obj$Penalized_Brier_Score,
    Alpha           = params$alpha,
    Lambda          = params$lambda
  )
}


## ── 01F: Safe uplift % relative to base ──────────────────────────────────────
safe_uplift_pct <- function(x, base, higher_is_better = TRUE) {
  ok  <- is.finite(x) & is.finite(base) & base != 0
  out <- rep(0, length(x))
  out[ok] <- (x[ok] - base) / base * 100
  if (!higher_is_better) out[ok] <- -out[ok]
  out
}


## ── 01G: Extract & rank coefficients (train importance) ──────────────────────
##   Works for both glmnet and glm models.
get_top_coeffs_glmnet <- function(model, model_name = "Model", top_n = TOP_N_COEF) {
  if (is.null(model)) {
    warning(paste(model_name, "model is NULL"))
    return(tibble::tibble(Strategy = model_name, Feature = character(),
                          Coef = numeric(), AbsCoef = numeric(),
                          OddsRatio = numeric()))
  }
  cc <- tryCatch({
    if (inherits(model, "glm")) {
      ## glm: coef() returns a named vector
      cv <- coef(model)
      matrix(cv, ncol = 1L, dimnames = list(names(cv), "s0"))
    } else {
      ## glmnet: coef() returns a sparse matrix
      as.matrix(stats::coef(model))
    }
  }, error = function(e) NULL)

  if (is.null(cc)) {
    warning(paste("Could not extract coefficients for", model_name))
    return(tibble::tibble(Strategy = model_name, Feature = character(),
                          Coef = numeric(), AbsCoef = numeric(),
                          OddsRatio = numeric()))
  }
  tibble::tibble(
    Feature = rownames(cc),
    Coef    = as.numeric(cc[, 1L])
  ) %>%
    dplyr::filter(!is.na(Coef), Feature != "(Intercept)") %>%
    dplyr::mutate(
      Strategy  = model_name,
      AbsCoef   = abs(Coef),
      OddsRatio = exp(Coef)
    ) %>%
    dplyr::arrange(dplyr::desc(AbsCoef)) %>%
    dplyr::slice_head(n = top_n) %>%
    dplyr::select(Strategy, Feature, Coef, AbsCoef, OddsRatio)
}


## ── 01H: Test-set variable contribution (mean |x * beta|) ───────────────────
get_test_contrib <- function(model, test_df, strategy_name = "Model",
                             top_n = TOP_N_COEF) {
  empty <- tibble::tibble(Strategy = strategy_name, Feature = character(),
                          Coef = numeric(), MeanAbsContrib = numeric(),
                          MeanContrib = numeric())
  if (is.null(model) || is.null(test_df)) {
    warning(paste("Missing model or test_df for", strategy_name)); return(empty)
  }
  beta <- tryCatch({
    cc <- as.matrix(stats::coef(model))
    stats::setNames(as.numeric(cc[, 1L]), rownames(cc))
  }, error = function(e) NULL)
  if (is.null(beta)) { warning(paste("coef() failed for", strategy_name)); return(empty) }

  X <- tryCatch(
    stats::model.matrix(y ~ ., data = test_df),
    error = function(e) NULL
  )
  if (is.null(X)) { warning(paste("model.matrix failed for", strategy_name)); return(empty) }

  common <- intersect(colnames(X), names(beta))
  Xc     <- X[, common, drop = FALSE]
  bc     <- beta[common]

  contrib <- sweep(Xc, 2L, bc, `*`)

  ## Remove intercept column from contribution table
  if ("(Intercept)" %in% colnames(contrib)) {
    contrib <- contrib[, setdiff(colnames(contrib), "(Intercept)"), drop = FALSE]
    bc      <- bc[setdiff(names(bc), "(Intercept)")]
  }

  tibble::tibble(
    Strategy       = strategy_name,
    Feature        = colnames(contrib),
    Coef           = as.numeric(bc[colnames(contrib)]),
    MeanAbsContrib = colMeans(abs(contrib), na.rm = TRUE),
    MeanContrib    = colMeans(contrib,      na.rm = TRUE)
  ) %>%
    dplyr::arrange(dplyr::desc(MeanAbsContrib)) %>%
    dplyr::slice_head(n = top_n)
}




## ── 01I: Stratified CV fold construction (firm-level) ────────────────────────
##   Stratifies by y_ever (ever-defaulted flag) only.
##   sector is NOT used — it was dropped by 02_FeatureEngineering.R before saving.
MVstratifiedsampling_CV_ID <- function(data, num_folds = 5L) {
  dt <- data.table::as.data.table(data)

  ## Firm-level profile: does this firm ever default across its observations?
  firm_profile <- dt[,
    .(y_ever = as.integer(any(get(TARGET_COL) == 1L))),
    by = id
  ]

  ## Assign folds round-robin within each stratum (y_ever = 0 or 1)
  set.seed(SEED)
  firm_profile[, fold := {
    n     <- .N
    folds <- sample(rep(seq_len(num_folds), length.out = n))
    folds
  }, by = y_ever]

  firm_profile[, .(id, fold)]
}


## ── 01J: Map firm-level fold IDs to row indices ───────────────────────────────
MVstratifiedsampling_CV_Split <- function(data, firm_fold_indices) {
  ## Returns a list with element fold_vector: integer vector of length nrow(data)
  dt <- data.table::as.data.table(data)
  if (!"id" %in% names(dt))
    stop("MVstratifiedsampling_CV_Split: 'id' column required in data.")
  merged    <- merge(dt[, .(row_id = .I, id)], firm_fold_indices, by = "id", all.x = TRUE)
  data.table::setorder(merged, row_id)
  list(fold_vector = merged$fold)
}


#==============================================================================#
#==== 02 - Data Preparation ===================================================#
#==============================================================================#

message("\n── 10_GLM Stage 1/5: Data Preparation ──────────────────────")

tryCatch({

  ## Already loaded and converted to data.frame in Section 02
  Train_df <- Train_Final_df
  Test_df  <- Test_Final_df

  ## ── Attach id for fold stratification (stripped again after) ─────────────
  Train_df_with_id      <- Train_df
  Train_df_with_id$id   <- train_id_vec

  ## ── Train datasets per strategy ───────────────────────────────────────────
  Train_Data_Base_Model  <- Train_df
  Train_Data_Strategy_A  <- Strategy_A_train
  Train_Data_Strategy_B  <- Strategy_B_train
  Train_Data_Strategy_C  <- Strategy_C_train

  ## ── CV fold assignment (firm-level stratified, based on Base Model) ───────
  ## id column needed only for fold construction — Base Model used as reference
  base_for_folds <- Train_df_with_id

  Data_Train_CV_Split_IDs  <- MVstratifiedsampling_CV_ID(
    data      = base_for_folds,
    num_folds = N_FOLDS_GLM
  )
  Data_Train_CV_Base_Model <- MVstratifiedsampling_CV_Split(
    data              = base_for_folds,
    firm_fold_indices = Data_Train_CV_Split_IDs
  )
  Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]

  ## ── Test datasets per strategy ────────────────────────────────────────────
  Test_Data_Base_Model  <- Test_df
  Test_Data_Strategy_A  <- Strategy_A_test
  Test_Data_Strategy_B  <- Strategy_B_test
  Test_Data_Strategy_C  <- Strategy_C_test

  ## ── Align test to train (factor levels + column order) ───────────────────
  ## [FIX 2026-03-30: calls now guarded — align_test_to_train(NULL, NULL) errors]
  Test_Data_Base_Model <- align_test_to_train(Train_Data_Base_Model, Test_Data_Base_Model)
  if (!is.null(Train_Data_Strategy_A))
    Test_Data_Strategy_A <- align_test_to_train(Train_Data_Strategy_A, Test_Data_Strategy_A)
  if (!is.null(Train_Data_Strategy_B))
    Test_Data_Strategy_B <- align_test_to_train(Train_Data_Strategy_B, Test_Data_Strategy_B)
  if (!is.null(Train_Data_Strategy_C))
    Test_Data_Strategy_C <- align_test_to_train(Train_Data_Strategy_C, Test_Data_Strategy_C)

  message(sprintf("  Base  train: %d rows x %d cols", nrow(Train_Data_Base_Model), ncol(Train_Data_Base_Model)))
  message(sprintf("  Strat A    : %d rows x %d cols", nrow(Train_Data_Strategy_A), ncol(Train_Data_Strategy_A)))
  message(sprintf("  Strat B    : %d rows x %d cols", nrow(Train_Data_Strategy_B), ncol(Train_Data_Strategy_B)))
  message(sprintf("  Strat C    : %d rows x %d cols", nrow(Train_Data_Strategy_C), ncol(Train_Data_Strategy_C)))
  message("── Data Preparation complete ────────────────────────────────")

}, error = function(e) stop("10_GLM Stage 1 (Data Prep) failed: ", e$message))


#==============================================================================#
#==== 03 - GLM Training (all strategies) =====================================#
#==============================================================================#
##
##  CHECKPOINTING: each trained model is saved to DIR_GLM_OUT as an .rds file.
##  If the checkpoint already exists, training is skipped and the model is
##  loaded from disk. This means if the run crashes, only the failed strategy
##  needs to be retrained — all completed strategies are reused automatically.
##
##  Checkpoint files:
##    GLM_checkpoint_Base_OoS.rds
##    GLM_checkpoint_StrategyA_OoS.rds
##    GLM_checkpoint_StrategyB_OoS.rds
##    GLM_checkpoint_StrategyC_OoS.rds
##
#==============================================================================#

message("\n── 10_GLM Stage 2/5: GLM Training ──────────────────────────")

## ── Helper: train or load from checkpoint ────────────────────────────────────
train_or_load <- function(label, train_data, cv_vector, checkpoint_name) {

  ## [FIX 2026-03-30: NULL guard — Groups 01-03 have no latent data for A/B/C]
  if (is.null(train_data)) {
    message(sprintf("  Skipping %s — no training data (latent features unavailable for this GROUP).",
                    label))
    return(NULL)
  }

  ckpt_path <- file.path(DIR_GLM_OUT,
                         paste0("GLM_checkpoint_", checkpoint_name,
                                "_", SPLIT_MODE, ".rds"))

  ## If checkpoint exists — load and skip training
  if (file.exists(ckpt_path)) {
    message(sprintf("  Loading checkpoint: %s", basename(ckpt_path)))
    result <- tryCatch(readRDS(ckpt_path),
                       error = function(e) {
                         message("  Checkpoint load failed, retraining: ", e$message)
                         NULL
                       })
    if (!is.null(result)) return(result)
  }

  ## Otherwise — train and save checkpoint
  message(sprintf("  Training: %s", label))
  result <- tryCatch(
    GLM_Training(
      Data_Train_CV_Vector = cv_vector,
      Train_Data           = train_data
    ),
    error = function(e) {
      message(sprintf("  %s ERROR: %s", label, e$message))
      NULL
    }
  )

  if (!is.null(result)) {
    tryCatch(saveRDS(result, ckpt_path),
             error = function(e) message("  Checkpoint save failed: ", e$message))
    message(sprintf("  Checkpoint saved: %s", basename(ckpt_path)))
  }

  result
}

## ── Base Model ───────────────────────────────────────────────────────────────
GLM_Results_BaseModel <- train_or_load(
  label            = "Base Model (raw features)",
  train_data       = Train_Data_Base_Model,
  cv_vector        = Data_Train_CV_Vector,
  checkpoint_name  = "Base"
)

## ── Strategy A ───────────────────────────────────────────────────────────────
GLM_Results_Strategy_A <- train_or_load(
  label            = "Strategy A (latent features only)",
  train_data       = Train_Data_Strategy_A,
  cv_vector        = Data_Train_CV_Vector,
  checkpoint_name  = "StrategyA"
)

## ── Strategy B ───────────────────────────────────────────────────────────────
GLM_Results_Strategy_B <- train_or_load(
  label            = "Strategy B (anomaly scores only)",
  train_data       = Train_Data_Strategy_B,
  cv_vector        = Data_Train_CV_Vector,
  checkpoint_name  = "StrategyB"
)

## ── Strategy C ───────────────────────────────────────────────────────────────
GLM_Results_Strategy_C <- train_or_load(
  label            = "Strategy C (raw + latent + anomaly)",
  train_data       = Train_Data_Strategy_C,
  cv_vector        = Data_Train_CV_Vector,
  checkpoint_name  = "StrategyC"
)

message("── GLM Training complete ─────────────────────────────────────")


#==============================================================================#
#==== 04 - Test-set Scoring ===================================================#
#==============================================================================#

message("\n── 10_GLM Stage 3/5: Test-set Scoring ──────────────────────")

tryCatch({
  GLM_Test_Results_BaseModel  <- GLM_Test(GLM_Results_BaseModel$optimal_model,  Test_Data_Base_Model)
}, error = function(e) { GLM_Test_Results_BaseModel  <<- NULL; message("  Test Base Model ERROR: ", e$message) })

tryCatch({
  GLM_Test_Results_Strategy_A <- GLM_Test(GLM_Results_Strategy_A$optimal_model, Test_Data_Strategy_A)
}, error = function(e) { GLM_Test_Results_Strategy_A <<- NULL; message("  Test Strategy A ERROR: ", e$message) })

tryCatch({
  GLM_Test_Results_Strategy_B <- GLM_Test(GLM_Results_Strategy_B$optimal_model, Test_Data_Strategy_B)
}, error = function(e) { GLM_Test_Results_Strategy_B <<- NULL; message("  Test Strategy B ERROR: ", e$message) })

tryCatch({
  GLM_Test_Results_Strategy_C <- GLM_Test(GLM_Results_Strategy_C$optimal_model, Test_Data_Strategy_C)
}, error = function(e) { GLM_Test_Results_Strategy_C <<- NULL; message("  Test Strategy C ERROR: ", e$message) })

message("── Test-set Scoring complete ─────────────────────────────────")


#==============================================================================#
#==== 05 - Leaderboards =======================================================#
#==============================================================================#

message("\n── 10_GLM Stage 4/5: Leaderboards ──────────────────────────")

## ── CV (Train) Leaderboard ───────────────────────────────────────────────────
tryCatch({

  comparison_table_raw <- dplyr::bind_rows(
    extract_metrics(GLM_Results_BaseModel,  "Base Model"),
    extract_metrics(GLM_Results_Strategy_A, "Strategy A"),
    extract_metrics(GLM_Results_Strategy_B, "Strategy B"),
    extract_metrics(GLM_Results_Strategy_C, "Strategy C")
  )

  base_row <- dplyr::filter(comparison_table_raw, Model == "Base Model") %>%
    dplyr::slice(1L)
  if (nrow(base_row) == 0L) stop("Base Model row not found in train leaderboard.")

  GLM_Leaderboard_Train <- comparison_table_raw %>%
    dplyr::filter(!is.na(AUC)) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(
      Uplift_AUC_pct   = safe_uplift_pct(AUC,             base_row$AUC,   higher_is_better = TRUE),
      Uplift_Brier_pct = safe_uplift_pct(Brier_Score,     base_row$Brier_Score, higher_is_better = FALSE),
      Uplift_PBS_pct   = safe_uplift_pct(Penalized_Brier, base_row$Penalized_Brier, higher_is_better = FALSE)
    ) %>%
    dplyr::select(Model, Type, AUC, Uplift_AUC_pct,
                  Brier_Score, Uplift_Brier_pct,
                  Penalized_Brier, Uplift_PBS_pct,
                  Alpha, Lambda)

  message("  GLM CV (Train) Leaderboard:")
  print(GLM_Leaderboard_Train)

}, error = function(e) message("  Train Leaderboard ERROR: ", e$message))


## ── Test Leaderboard ─────────────────────────────────────────────────────────
tryCatch({

  comparison_table_test_raw <- dplyr::bind_rows(
    extract_test_metrics(GLM_Test_Results_BaseModel,  Test_Data_Base_Model,  GLM_Results_BaseModel,  "Base Model"),
    extract_test_metrics(GLM_Test_Results_Strategy_A, Test_Data_Strategy_A,  GLM_Results_Strategy_A, "Strategy A"),
    extract_test_metrics(GLM_Test_Results_Strategy_B, Test_Data_Strategy_B,  GLM_Results_Strategy_B, "Strategy B"),
    extract_test_metrics(GLM_Test_Results_Strategy_C, Test_Data_Strategy_C,  GLM_Results_Strategy_C, "Strategy C")
  )

  base_row_test <- dplyr::filter(comparison_table_test_raw, Model == "Base Model") %>%
    dplyr::slice(1L)
  if (nrow(base_row_test) == 0L) stop("Base Model row not found in test leaderboard.")

  GLM_Leaderboard_Test <- comparison_table_test_raw %>%
    dplyr::filter(!is.na(AUC)) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(
      Uplift_AUC_pct   = safe_uplift_pct(AUC,             base_row_test$AUC,   higher_is_better = TRUE),
      Uplift_Brier_pct = safe_uplift_pct(Brier_Score,     base_row_test$Brier_Score, higher_is_better = FALSE),
      Uplift_PBS_pct   = safe_uplift_pct(Penalized_Brier, base_row_test$Penalized_Brier, higher_is_better = FALSE)
    ) %>%
    dplyr::select(Model, Type, AUC, Uplift_AUC_pct,
                  Brier_Score, Uplift_Brier_pct,
                  Penalized_Brier, Uplift_PBS_pct,
                  Alpha, Lambda)

  message("  GLM Test Leaderboard:")
  print(GLM_Leaderboard_Test)

}, error = function(e) message("  Test Leaderboard ERROR: ", e$message))

message("── Leaderboards complete ─────────────────────────────────────")


#==============================================================================#
#==== 06 - Variable Importance ================================================#
#==============================================================================#

message("\n── 10_GLM Stage 5/5: Variable Importance ────────────────────")

## ── Train importance (coefficient magnitude) ─────────────────────────────────
tryCatch({

  models_list <- list(
    "Base Model" = GLM_Results_BaseModel$optimal_model,
    "Strategy A" = GLM_Results_Strategy_A$optimal_model,
    "Strategy B" = GLM_Results_Strategy_B$optimal_model,
    "Strategy C" = GLM_Results_Strategy_C$optimal_model
  )

  importance_train_all <- dplyr::bind_rows(lapply(names(models_list), function(nm)
    get_top_coeffs_glmnet(models_list[[nm]], model_name = nm, top_n = TOP_N_COEF)
  ))

  ## Print per strategy
  for (nm in names(models_list)) {
    cat(sprintf("\n── %s — Train Top %d Coefficients ──\n", nm, TOP_N_COEF))
    print(dplyr::filter(importance_train_all, Strategy == nm))
  }

}, error = function(e) message("  Train Variable Importance ERROR: ", e$message))


## ── Test importance (mean |x * beta| contribution) ───────────────────────────
tryCatch({

  test_items <- list(
    list(name = "Base Model", model = GLM_Results_BaseModel$optimal_model,  test_df = Test_Data_Base_Model),
    list(name = "Strategy A", model = GLM_Results_Strategy_A$optimal_model, test_df = Test_Data_Strategy_A),
    list(name = "Strategy B", model = GLM_Results_Strategy_B$optimal_model, test_df = Test_Data_Strategy_B),
    list(name = "Strategy C", model = GLM_Results_Strategy_C$optimal_model, test_df = Test_Data_Strategy_C)
  )

  importance_test_all <- dplyr::bind_rows(lapply(test_items, function(it)
    get_test_contrib(it$model, it$test_df,
                     strategy_name = it$name, top_n = TOP_N_COEF)
  )) %>%
    dplyr::group_by(Strategy) %>%
    dplyr::arrange(dplyr::desc(MeanAbsContrib), .by_group = TRUE) %>%
    dplyr::mutate(Rank = dplyr::row_number()) %>%
    dplyr::ungroup() %>%
    dplyr::select(Strategy, Rank, Feature, Coef, MeanAbsContrib, MeanContrib)

  ## Print per strategy
  for (nm in unique(importance_test_all$Strategy)) {
    cat(sprintf("\n── %s — Test Mean |x*beta| Top %d ──\n", nm, TOP_N_COEF))
    print(dplyr::filter(importance_test_all, Strategy == nm))
  }

}, error = function(e) message("  Test Variable Importance ERROR: ", e$message))

message("── Variable Importance complete ──────────────────────────────")


#==============================================================================#
#==== 07 - Save Outputs =======================================================#
#==============================================================================#

## ── 07A: Leaderboard Excel ────────────────────────────────────────────────────
##   File : 03_Output/GLM/GLM_Leaderboard_OoS.xlsx
##   Sheet "Train_CV" — CV AUC + Brier + Penalized Brier + Alpha/Lambda
##   Sheet "Test"     — same metrics evaluated on held-out test set
tryCatch({

  wb_lb <- openxlsx::createWorkbook()

  if (exists("GLM_Leaderboard_Train") && !is.null(GLM_Leaderboard_Train)) {
    openxlsx::addWorksheet(wb_lb, "Train_CV")
    openxlsx::writeData(wb_lb, "Train_CV", GLM_Leaderboard_Train)
    openxlsx::setColWidths(wb_lb, "Train_CV",
                           cols = seq_len(ncol(GLM_Leaderboard_Train)), widths = "auto")
  }
  if (exists("GLM_Leaderboard_Test") && !is.null(GLM_Leaderboard_Test)) {
    openxlsx::addWorksheet(wb_lb, "Test")
    openxlsx::writeData(wb_lb, "Test", GLM_Leaderboard_Test)
    openxlsx::setColWidths(wb_lb, "Test",
                           cols = seq_len(ncol(GLM_Leaderboard_Test)), widths = "auto")
  }

  lb_path <- file.path(DIR_GLM_OUT, paste0("GLM_Leaderboard_v2_", SPLIT_MODE, ".xlsx"))
  openxlsx::saveWorkbook(wb_lb, lb_path, overwrite = TRUE)
  message(sprintf("  Saved: %s", lb_path))

}, error = function(e) message("  Save Leaderboard ERROR: ", e$message))


## ── 07B: Variable Importance Excel ───────────────────────────────────────────
##   File : 03_Output/GLM/GLM_Variable_Importance_OoS.xlsx
##   Train sheets (4) — Strategy, Feature, Coef, AbsCoef, OddsRatio
##   Test  sheets (4) — Strategy, Rank, Feature, Coef, MeanAbsContrib, MeanContrib
##   Naming: Train_Base_Model | Train_Strategy_A | Train_Strategy_B | Train_Strategy_C
##           Test_Base_Model  | Test_Strategy_A  | Test_Strategy_B  | Test_Strategy_C
tryCatch({

  wb_vi <- openxlsx::createWorkbook()

  if (exists("importance_train_all") && !is.null(importance_train_all)) {
    for (nm in unique(importance_train_all$Strategy)) {
      sheet_nm <- paste0("Train_", gsub(" ", "_", nm))
      openxlsx::addWorksheet(wb_vi, sheet_nm)
      openxlsx::writeData(wb_vi, sheet_nm,
                          dplyr::filter(importance_train_all, Strategy == nm))
      openxlsx::setColWidths(wb_vi, sheet_nm, cols = 1:5, widths = "auto")
    }
  }

  if (exists("importance_test_all") && !is.null(importance_test_all)) {
    for (nm in unique(importance_test_all$Strategy)) {
      sheet_nm <- paste0("Test_", gsub(" ", "_", nm))
      openxlsx::addWorksheet(wb_vi, sheet_nm)
      openxlsx::writeData(wb_vi, sheet_nm,
                          dplyr::filter(importance_test_all, Strategy == nm))
      openxlsx::setColWidths(wb_vi, sheet_nm, cols = 1:6, widths = "auto")
    }
  }

  vi_path <- file.path(DIR_GLM_OUT, paste0("GLM_Variable_Importance_v2_", SPLIT_MODE, ".xlsx"))
  openxlsx::saveWorkbook(wb_vi, vi_path, overwrite = TRUE)
  message(sprintf("  Saved: %s", vi_path))

}, error = function(e) message("  Save Variable Importance ERROR: ", e$message))


## ── 07C: Leaderboard Bar Chart ────────────────────────────────────────────────
##   File : 03_Output/GLM/GLM_Leaderboard_Chart_OoS.png
##   Shows Test AUC per strategy with % uplift labels vs Base Model
tryCatch({

  if (exists("GLM_Leaderboard_Test") && !is.null(GLM_Leaderboard_Test) &&
      nrow(GLM_Leaderboard_Test) > 0L) {

    chart_data <- GLM_Leaderboard_Test %>%
      dplyr::mutate(
        Label = paste0(round(AUC, 4), "\n(", ifelse(Uplift_AUC_pct >= 0, "+", ""),
                       round(Uplift_AUC_pct, 2), "%)"),
        Model = factor(Model, levels = rev(Model))   ## keep sorted order in coord_flip
      )

    p_leaderboard <- ggplot2::ggplot(
      chart_data,
      ggplot2::aes(x = Model, y = AUC, fill = Model)
    ) +
      ggplot2::geom_col(width = 0.6, show.legend = FALSE) +
      ggplot2::geom_text(
        ggplot2::aes(label = Label),
        hjust = -0.05, size = 3.5, fontface = "bold"
      ) +
      ggplot2::coord_flip() +
      ggplot2::scale_fill_manual(values = c(
        "Base Model"  = "#708090",
        "Strategy A"  = "#004890",
        "Strategy B"  = "#F37021",
        "Strategy C"  = "#2E8B57"
      )) +
      ggplot2::scale_y_continuous(
        limits = c(0, max(chart_data$AUC, na.rm = TRUE) * 1.15),
        labels = scales::number_format(accuracy = 0.001)
      ) +
      ggplot2::labs(
        title    = "GLM Strategy Comparison — Test Set AUC",
        subtitle = "Elastic-net regularised logistic regression | OoS split\nUplift % relative to Base Model",
        x        = NULL,
        y        = "Test AUC-ROC"
      ) +
      ggplot2::theme_minimal(base_size = 13) +
      ggplot2::theme(
        plot.title    = ggplot2::element_text(face = "bold"),
        plot.subtitle = ggplot2::element_text(color = "#708090"),
        axis.text.y   = ggplot2::element_text(face = "bold", size = 11),
        panel.grid.major.y = ggplot2::element_blank()
      )

    chart_path <- file.path(DIR_GLM_OUT,
                             paste0("GLM_Leaderboard_Chart_v2_", SPLIT_MODE, ".png"))
    ggplot2::ggsave(chart_path, plot = p_leaderboard,
                    width = 10, height = 5, dpi = 150)
    message(sprintf("  Saved: %s", chart_path))
  }

}, error = function(e) message("  Save Chart ERROR: ", e$message))



#==============================================================================#
#==== 08 - Predictions & Evaluation (matches 06_Evaluation structure) =========#
#==============================================================================#
##
##  OUTPUT FILES (saved to DIR_GLM_OUT):
##
##  predictions_test_GLM_OoS.parquet
##    — columns: id, y, p_default, model, framework
##    — one row per observation per strategy (4 strategies × n_test rows)
##    — directly stackable with AutoGluon and XGBoost prediction files
##
##  GLM_Evaluation_OoS.xlsx
##    — Sheet "Metrics":  AUC, AP, Brier, BSS, R@FPR1/3/5/10,
##                        Youden threshold, Sensitivity, Specificity, F1
##    — Sheet "DeLong":   pairwise AUC comparison vs Base Model
##                        p-value + 95% CI on AUC difference
##
#==============================================================================#

message("\n── 10_GLM Stage 6/6: Predictions & Evaluation ───────────────")

## ── Install arrow if needed (for parquet output) ─────────────────────────────
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
library(arrow)

## ── Model / test-data registry ───────────────────────────────────────────────
## Maps strategy label → (trained model, test dataset, test id vector)
eval_registry <- list(
  list(model    = if (exists("GLM_Results_BaseModel"))  GLM_Results_BaseModel$optimal_model  else NULL,
       test_df  = if (exists("Test_Data_Base_Model"))   Test_Data_Base_Model                 else NULL,
       label    = "GLM_M1_Base"),
  list(model    = if (exists("GLM_Results_Strategy_A")) GLM_Results_Strategy_A$optimal_model else NULL,
       test_df  = if (exists("Test_Data_Strategy_A"))   Test_Data_Strategy_A                 else NULL,
       label    = "GLM_M2_LatentFeatures"),
  list(model    = if (exists("GLM_Results_Strategy_B")) GLM_Results_Strategy_B$optimal_model else NULL,
       test_df  = if (exists("Test_Data_Strategy_B"))   Test_Data_Strategy_B                 else NULL,
       label    = "GLM_M3_AnomalyScore"),
  list(model    = if (exists("GLM_Results_Strategy_C")) GLM_Results_Strategy_C$optimal_model else NULL,
       test_df  = if (exists("Test_Data_Strategy_C"))   Test_Data_Strategy_C                 else NULL,
       label    = "GLM_M4_RawLatentAnomaly")
)


## ── 08A: Build combined predictions data.table ───────────────────────────────
##   Format: id | y | p_default | model | framework
##   Identical structure to AutoGluon and XGBoost prediction files
tryCatch({

  pred_list <- lapply(eval_registry, function(it) {

    if (is.null(it$model) || is.null(it$test_df)) {
      message(sprintf("  Skipping predictions for %s — model or test data is NULL",
                      it$label))
      return(NULL)
    }

    ## ── Predict: handle glm (Strategy B) vs glmnet (all others) ─────────────
    preds <- tryCatch({
      if (inherits(it$model, "glm")) {
        as.numeric(predict(it$model, newdata = it$test_df, type = "response"))
      } else {
        X_test <- stats::model.matrix(y ~ ., data = it$test_df)[, -1L, drop = FALSE]
        as.numeric(predict(it$model, newx = X_test, type = "response"))
      }
    }, error = function(e) {
      message(sprintf("  predict failed for %s: %s", it$label, e$message))
      NULL
    })
    if (is.null(preds)) return(NULL)

    y_test <- as.numeric(as.character(it$test_df[[TARGET_COL]]))

    data.table::data.table(
      id        = test_id_vec,
      y         = y_test,
      p_default = preds,
      model     = it$label,
      framework = "GLM"
    )
  })

  ## Remove NULLs and stack
  pred_list   <- Filter(Negate(is.null), pred_list)
  predictions_combined <- data.table::rbindlist(pred_list)

  message(sprintf("  Combined predictions: %d rows x %d cols  (%d strategies)",
                  nrow(predictions_combined), ncol(predictions_combined),
                  length(pred_list)))

  ## Save as parquet — directly stackable with AG and XGB prediction files
  pred_path <- file.path(DIR_GLM_OUT,
                          paste0("predictions_test_GLM_v2_", SPLIT_MODE, ".parquet"))
  arrow::write_parquet(predictions_combined, pred_path)
  message(sprintf("  Saved: %s", pred_path))

  ## Also keep ROC objects for DeLong tests below
  roc_objects <- lapply(eval_registry, function(it) {
    if (is.null(it$model) || is.null(it$test_df)) return(NULL)
    sub <- predictions_combined[model == it$label]
    if (nrow(sub) == 0L) return(NULL)
    tryCatch(
      pROC::roc(sub$y, sub$p_default, quiet = TRUE),
      error = function(e) NULL
    )
  })
  names(roc_objects) <- sapply(eval_registry, `[[`, "label")

}, error = function(e) message("  Predictions ERROR: ", e$message))


## ── 08B: Compute full metrics per strategy ────────────────────────────────────
##   Matches 06_Evaluation metric set:
##   AUC-ROC, AP, Brier, BSS, R@FPR1/3/5/10,
##   Youden threshold, Sensitivity, Specificity, F1
tryCatch({

  ## Helper: Recall at fixed FPR (= TPR when FPR <= threshold)
  recall_at_fpr <- function(roc_obj, fpr_target) {
    idx <- which(1 - roc_obj$specificities <= fpr_target)
    if (length(idx) == 0L) return(NA_real_)
    max(roc_obj$sensitivities[idx], na.rm = TRUE)
  }

  ## Helper: Average Precision (area under precision-recall curve)
  avg_precision <- function(y, p) {
    ord  <- order(p, decreasing = TRUE)
    y_s  <- y[ord]
    n    <- length(y_s)
    tp   <- cumsum(y_s)
    fp   <- seq_len(n) - tp
    prec <- tp / (tp + fp)
    rec  <- tp / sum(y_s)
    ## trapezoidal integration
    sum(diff(c(0, rec)) * prec, na.rm = TRUE)
  }

  metrics_list <- lapply(eval_registry, function(it) {

    if (!exists("predictions_combined")) return(NULL)
    sub <- predictions_combined[model == it$label]
    if (nrow(sub) == 0L || is.null(roc_objects[[it$label]])) {
      return(data.table::data.table(Model = it$label, Framework = "GLM",
                                    AUC=NA, AP=NA, Brier=NA, BSS=NA,
                                    R_FPR1=NA, R_FPR3=NA, R_FPR5=NA, R_FPR10=NA,
                                    Youden_Threshold=NA,
                                    Sensitivity=NA, Specificity=NA, F1=NA))
    }

    y_t   <- sub$y
    p_t   <- sub$p_default
    roc_o <- roc_objects[[it$label]]

    ## AUC-ROC
    auc_val <- as.numeric(pROC::auc(roc_o))

    ## Average Precision
    ap_val  <- tryCatch(avg_precision(y_t, p_t), error = function(e) NA_real_)

    ## Brier Score
    brier   <- mean((p_t - y_t)^2, na.rm = TRUE)

    ## Brier Skill Score (vs naive forecast = prevalence)
    prev    <- mean(y_t, na.rm = TRUE)
    brier_ref <- mean((prev - y_t)^2, na.rm = TRUE)
    bss     <- if (brier_ref > 0) 1 - brier / brier_ref else NA_real_

    ## Recall at fixed FPR levels
    r1  <- recall_at_fpr(roc_o, 0.01)
    r3  <- recall_at_fpr(roc_o, 0.03)
    r5  <- recall_at_fpr(roc_o, 0.05)
    r10 <- recall_at_fpr(roc_o, 0.10)

    ## Youden threshold (maximises sensitivity + specificity - 1)
    youden_idx <- which.max(roc_o$sensitivities + roc_o$specificities - 1)
    youden_thr <- roc_o$thresholds[youden_idx]
    y_pred_bin <- as.integer(p_t >= youden_thr)

    ## Sensitivity, Specificity, F1 at Youden threshold
    tp  <- sum(y_pred_bin == 1L & y_t == 1L)
    fp  <- sum(y_pred_bin == 1L & y_t == 0L)
    fn  <- sum(y_pred_bin == 0L & y_t == 1L)
    tn  <- sum(y_pred_bin == 0L & y_t == 0L)

    sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    f1   <- if (!is.na(sens) && !is.na(prec) && (sens + prec) > 0)
      2 * sens * prec / (sens + prec) else NA_real_

    data.table::data.table(
      Model            = it$label,
      Framework        = "GLM",
      AUC              = round(auc_val, 6),
      AP               = round(ap_val,  6),
      Brier            = round(brier,   6),
      BSS              = round(bss,     6),
      R_FPR1           = round(r1,      4),
      R_FPR3           = round(r3,      4),
      R_FPR5           = round(r5,      4),
      R_FPR10          = round(r10,     4),
      Youden_Threshold = round(youden_thr, 6),
      Sensitivity      = round(sens,    4),
      Specificity      = round(spec,    4),
      F1               = round(f1,      4)
    )
  })

  GLM_Metrics <- data.table::rbindlist(Filter(Negate(is.null), metrics_list))

  message("  GLM Evaluation Metrics:")
  print(GLM_Metrics)

}, error = function(e) message("  Metrics ERROR: ", e$message))


## ── 08C: DeLong pairwise AUC tests vs Base Model ─────────────────────────────
##   Matches 06_Evaluation DeLong structure:
##   Each strategy vs GLM_M1_Base
##   Output: AUC_model, AUC_base, AUC_diff, CI_lower, CI_upper, p_value
tryCatch({

  base_roc <- roc_objects[["GLM_M1_Base"]]

  if (is.null(base_roc)) {
    message("  DeLong tests skipped — Base Model ROC not available.")
  } else {

    delong_list <- lapply(eval_registry[-1L], function(it) {
      ## Skip Base Model itself (index -1 removes it)
      comp_roc <- roc_objects[[it$label]]
      if (is.null(comp_roc)) {
        return(data.table::data.table(
          Model_A  = it$label, Model_B = "GLM_M1_Base",
          AUC_A = NA, AUC_B = NA, AUC_diff = NA,
          CI_lower = NA, CI_upper = NA, p_value = NA
        ))
      }

      test_res <- tryCatch(
        pROC::roc.test(comp_roc, base_roc, method = "delong"),
        error = function(e) NULL
      )
      if (is.null(test_res)) {
        return(data.table::data.table(
          Model_A  = it$label, Model_B = "GLM_M1_Base",
          AUC_A = as.numeric(pROC::auc(comp_roc)),
          AUC_B = as.numeric(pROC::auc(base_roc)),
          AUC_diff = NA, CI_lower = NA, CI_upper = NA, p_value = NA
        ))
      }

      data.table::data.table(
        Model_A  = it$label,
        Model_B  = "GLM_M1_Base",
        AUC_A    = round(as.numeric(pROC::auc(comp_roc)), 6),
        AUC_B    = round(as.numeric(pROC::auc(base_roc)), 6),
        AUC_diff = round(as.numeric(pROC::auc(comp_roc)) -
                           as.numeric(pROC::auc(base_roc)), 6),
        CI_lower = round(test_res$conf.int[1L], 6),
        CI_upper = round(test_res$conf.int[2L], 6),
        p_value  = round(test_res$p.value,      6)
      )
    })

    GLM_DeLong <- data.table::rbindlist(delong_list)

    message("  GLM DeLong Tests (vs Base Model):")
    print(GLM_DeLong)
  }

}, error = function(e) message("  DeLong ERROR: ", e$message))


## ── 08D: Save Evaluation Excel ───────────────────────────────────────────────
##   File: 03_Output/GLM/GLM_Evaluation_OoS.xlsx
##   Sheet "Metrics" — full metric table
##   Sheet "DeLong"  — pairwise AUC test results
tryCatch({

  wb_eval <- openxlsx::createWorkbook()

  if (exists("GLM_Metrics") && !is.null(GLM_Metrics) && nrow(GLM_Metrics) > 0L) {
    openxlsx::addWorksheet(wb_eval, "Metrics")
    openxlsx::writeData(wb_eval, "Metrics", as.data.frame(GLM_Metrics))
    openxlsx::setColWidths(wb_eval, "Metrics",
                           cols = seq_len(ncol(GLM_Metrics)), widths = "auto")
  }

  if (exists("GLM_DeLong") && !is.null(GLM_DeLong) && nrow(GLM_DeLong) > 0L) {
    openxlsx::addWorksheet(wb_eval, "DeLong")
    openxlsx::writeData(wb_eval, "DeLong", as.data.frame(GLM_DeLong))
    openxlsx::setColWidths(wb_eval, "DeLong",
                           cols = seq_len(ncol(GLM_DeLong)), widths = "auto")
  }

  eval_path <- file.path(DIR_GLM_OUT,
                          paste0("GLM_Evaluation_v2_", SPLIT_MODE, ".xlsx"))
  openxlsx::saveWorkbook(wb_eval, eval_path, overwrite = TRUE)
  message(sprintf("  Saved: %s", eval_path))

}, error = function(e) message("  Save Evaluation ERROR: ", e$message))

message("── Predictions & Evaluation complete ────────────────────────")

message("\n══ 10_GLM complete ══════════════════════════════════════════\n")
