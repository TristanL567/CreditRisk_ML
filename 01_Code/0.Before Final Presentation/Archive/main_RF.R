#==============================================================================#
#==== Random Forest Pipeline — Standalone Script ==============================#
#==============================================================================#
#
# Standalone Random Forest pipeline with Bayesian Optimization (MBO) for
# hyperparameter tuning. Runs across multiple feature-engineering strategies
# (Base, A, B, C, D) and produces strategy-comparison leaderboards.
#
# STRATEGIES:
#   Base — Raw quantile-transformed features          (no prerequisites)
#   A    — VAE latent features (dim. reduction)       (needs artifacts*)
#   B    — Base + VAE anomaly score                   (needs artifacts*)
#   C    — DAE-denoised features                      (needs artifacts*)
#   D    — Base + hand-crafted financial ratios        (no prerequisites)
#
#   * Run generate_vae_dae_artifacts.R FIRST to create the .rds files
#     in data/pipeline_artifacts/. Without them, A/B/C will be skipped.
#
# TYPICAL WORKFLOW:
#   1. source("CreditRisk_ML/01_Code/generate_vae_dae_artifacts.R")  # once
#   2. Set STRATEGIES_TO_RUN below
#   3. source("CreditRisk_ML/01_Code/main_RF.R")
#
# Usage: Source this script or run it from the command line.
#        All configuration is controlled via the knobs below.
#
#==============================================================================#


#==============================================================================#
#==== CONFIGURATION KNOBS =====================================================#
#==============================================================================#
# Set these before running. Modeled after the Python notebook config pattern.

# --- Training Speed ---
TRAIN_MODE <- "medium"  # "fast", "medium", "slow"
#   fast:   Quick test run   — 10 evals, 3 folds  (~5 min/strategy)
#   medium: Standard run     — 30 evals, 5 folds  (~20 min/strategy)
#   slow:   Production run   — 100 evals, 10 folds (~2 hrs/strategy)


# --- Execution Control ---
STRATEGIES_TO_RUN    <- c("Base","A","B","C","D")
# c("A", "B", "C")  # A/B/C need artifacts from generate_vae_dae_artifacts.R



get_training_params <- function(mode = "medium") {
  mode <- tolower(mode)
  if (mode == "fast") {
    return(list(
      n_evals = 10L, stagnation_iters = 10L, stagnation_threshold = 0.002,
      n_folds = 3L
    ))
  }
  if (mode == "medium") {
    return(list(
      n_evals = 30L, stagnation_iters = 50L, stagnation_threshold = 0.001,
      n_folds = 5L
    ))
  }
  if (mode == "slow") {
    return(list(
      n_evals = 100L, stagnation_iters = 80L, stagnation_threshold = 0.0005,
      n_folds = 10L
    ))
  }
  stop(paste("Unknown TRAIN_MODE:", mode, "— choose 'fast', 'medium', or 'slow'"))
}

TRAINING_PARAMS      <- get_training_params(TRAIN_MODE)
N_EVALS              <- TRAINING_PARAMS$n_evals
STAGNATION_ITERS     <- TRAINING_PARAMS$stagnation_iters
STAGNATION_THRESHOLD <- TRAINING_PARAMS$stagnation_threshold
N_FOLDS              <- TRAINING_PARAMS$n_folds

# --- Cross-Validation Settings ---
TRAIN_PROP           <- 0.7       # Proportion of data for training
RANDOM_SEED          <- 123L      # Reproducibility seed

# --- Chart Settings ---
WIDTH_PX             <- 3750      # Chart width in pixels
HEIGHT_PX            <- 1833      # Chart height in pixels
DPI                  <- 300       # Chart resolution
BLUE                 <- "#004890"
GREY                 <- "#708090"
ORANGE               <- "#F37021"
RED                  <- "#B22222"

# --- Parallelization ---
N_CORES              <- "auto"    # "auto" = detectCores(), or set integer
PARALLEL_BACKEND     <- "multisession"  # "multisession", "multicore", "sequential"

# --- Feature Engineering ---
DIVIDE_BY_TOTAL_ASSETS <- TRUE    # Normalize financial features by f1 (total assets)





# --- Data Paths (auto-detected from project root) ---
PROJECT_ROOT         <- file.path(here::here(""))
DATA_DIRECTORY       <- file.path(PROJECT_ROOT, "data", "data.rda")

# --- Derived Constants (do not edit) ---
WIDTH_IN  <- WIDTH_PX / DPI
HEIGHT_IN <- HEIGHT_PX / DPI

if (N_CORES == "auto") {
  N_CORES <- parallel::detectCores()
}

cat(strrep("=", 70), "\n")
cat("RANDOM FOREST PIPELINE — Configuration\n")
cat(strrep("=", 70), "\n")
cat("  Training mode:    ", toupper(TRAIN_MODE), "\n")
cat("  HPO budget:       ", N_EVALS, "evals, stagnation:", STAGNATION_ITERS, "\n")
cat("  CV folds:         ", N_FOLDS, "\n")
cat("  Train proportion: ", TRAIN_PROP, "\n")
cat("  Cores:            ", N_CORES, "\n")
cat("  Strategies:       ", paste(STRATEGIES_TO_RUN, collapse = ", "), "\n")
cat(strrep("=", 70), "\n\n")


#==============================================================================#
#==== 01 — Libraries ==========================================================#
#==============================================================================#

packages <- c(
  # Data manipulation
  "dplyr", "tidyr", "purrr", "data.table",
  # Sampling & preprocessing
  "caret", "lubridate",
  # mlr3 ecosystem (Random Forest)
  "mlr3", "mlr3learners", "mlr3tuning", "mlr3mbo",
  "paradox", "ranger",
  # Metrics
  "pROC",
  # Visualization
  "ggplot2", "scales", "gt",
  # Parallelization
  "future", "future.apply"
)

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
    cat(sprintf("Package '%s' installed and loaded.\n", pkg))
  } else {
    cat(sprintf("Package '%s' loaded.\n", pkg))
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}


#==============================================================================#
#==== 02 — Source Functions ===================================================#
#==============================================================================#

sourceFunctions <- function(functionDirectory) {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R",
                              full.names = TRUE)
  for (path in functionFiles) {
    try(source(path, echo = FALSE, verbose = FALSE, print.eval = TRUE, local = FALSE))
  }
}

Functions_Directory    <- file.path(PROJECT_ROOT, "CreditRisk_ML", "01_Code", "Subfunctions")
RF_Functions_Directory <- file.path(PROJECT_ROOT, "CreditRisk_ML", "01_Code", "RF_Subfunctions")
Charts_RF_Directory    <- file.path(PROJECT_ROOT, "CreditRisk_ML", "03_Charts", "RF")

sourceFunctions(Functions_Directory)
sourceFunctions(RF_Functions_Directory)

# Set up parallelization
future::plan(PARALLEL_BACKEND, workers = N_CORES)
cat("Parallelization:", PARALLEL_BACKEND, "with", N_CORES, "workers\n\n")


#==============================================================================#
#==== 03 — Data Loading & Preprocessing =======================================#
#==============================================================================#

set.seed(RANDOM_SEED)

cat("Loading data from:", DATA_DIRECTORY, "\n")
load(DATA_DIRECTORY)
Data <- d

## Apply preprocessing filters (from Main.R section 02B)
Data <- DataPreprocessing(Data)

## Drop ratio features
Exclude <- paste0("r", seq(1, 18))
Data <- Data[, -which(names(Data) %in% Exclude)]

Exclude_meta <- c("id", "refdate", "size", "sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude_meta)]

## Train/Test split with multivariate stratified sampling
Data_Sampled <- MVstratifiedsampling(Data,
                                     strat_vars = c("sector", "y"),
                                     Train_size = TRAIN_PROP)
Train <- Data_Sampled[["Train"]]
Test  <- Data_Sampled[["Test"]]

## Quick validation
analyse_MVstratifiedsampling(Train, "TRAIN SET",
                             target_col = "y", sector_col = "sector",
                             date_col = "refdate")
analyse_MVstratifiedsampling(Test, "TEST SET",
                             target_col = "y", sector_col = "sector",
                             date_col = "refdate")

## Store IDs before dropping metadata
Train_with_id <- Train
Test_with_id  <- Test

Exclude_id <- c("id", "refdate")
Train <- Train[, -which(names(Train) %in% Exclude_id)]
Test  <- Test[, -which(names(Test) %in% Exclude_id)]

Train_Backup <- Train
Test_Backup  <- Test


#==============================================================================#
#==== 04 — Feature Engineering ================================================#
#==============================================================================#

#==== 04A — Standardization (divide by total assets) ==========================#

if (DIVIDE_BY_TOTAL_ASSETS) {
  asset_col <- "f1"
  cols_to_scale <- c("f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")

  safe_divide <- function(numerator, denominator) {
    if_else(denominator == 0 | is.na(denominator), 0, numerator / denominator)
  }

  Train <- Train %>%
    mutate(across(.cols = all_of(cols_to_scale),
                  .fns = ~ safe_divide(.x, .data[[asset_col]]))) %>%
    as.data.frame()

  Test <- Test %>%
    mutate(across(.cols = all_of(cols_to_scale),
                  .fns = ~ safe_divide(.x, .data[[asset_col]]))) %>%
    as.data.frame()
}

#==== 04B — Quantile Transformation ===========================================#

num_cols <- c("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")

Train_Transformed <- Train
Test_Transformed  <- Test

for (col in num_cols) {
  res <- QuantileTransformation(Train[[col]], Test[[col]])
  Train_Transformed[[col]] <- res$train
  Test_Transformed[[col]]  <- res$test
}

#==== 04C — Stratified CV Folds ===============================================#

Strat_Data <- Train_Transformed %>%
  mutate(id = Train_with_id$id)

cv_result   <- MVstratifiedsampling_CV(Strat_Data, num_folds = N_FOLDS, seed = RANDOM_SEED)
fold_vector <- cv_result$fold_vector

cat("CV folds generated:", N_FOLDS, "\n\n")


#==============================================================================#
#==== 05 — Strategy Definitions ===============================================#
#==============================================================================#
# Each strategy defines how features are prepared before RF training.
# The RF pipeline (HPO → Train → Eval) is identical across strategies.

prepare_strategy_data <- function(strategy, train_df, test_df) {
  
  if (strategy == "Base") {
    # Base Model: use preprocessed features as-is
    return(list(train = train_df, test = test_df, description = "Base Model (raw features)"))
  }
  
  if (strategy == "A") {
    # Strategy A: Dimensionality Reduction — VAE latent features
    # Expects pre-computed latent features from the VAE pipeline
    latent_path_train <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "vae_latent_train.rds")
    latent_path_test  <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "vae_latent_test.rds")
    
    if (!file.exists(latent_path_train) || !file.exists(latent_path_test)) {
      cat("  [SKIP] Strategy A: VAE latent features not found.\n")
      cat("         Run generate_vae_dae_artifacts.R first to create them.\n")
      return(NULL)
    }
    
    latent_train <- readRDS(latent_path_train)
    latent_test  <- readRDS(latent_path_test)
    
    # Replace financial features with latent features, keep y + categorical
    train_out <- cbind(latent_train, train_df[, c("y", "size", "sector"), drop = FALSE])
    test_out  <- cbind(latent_test,  test_df[, c("y", "size", "sector"), drop = FALSE])
    
    return(list(train = train_out, test = test_out,
                description = "Strategy A (Dim. Reduction / VAE Latent)"))
  }
  
  if (strategy == "B") {
    # Strategy B: Anomaly Score — append reconstruction error as feature
    anomaly_path_train <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "vae_anomaly_train.rds")
    anomaly_path_test  <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "vae_anomaly_test.rds")
    
    if (!file.exists(anomaly_path_train) || !file.exists(anomaly_path_test)) {
      cat("  [SKIP] Strategy B: Anomaly score features not found.\n")
      cat("         Run generate_vae_dae_artifacts.R first to create them.\n")
      return(NULL)
    }
    
    anomaly_train <- readRDS(anomaly_path_train)
    anomaly_test  <- readRDS(anomaly_path_test)
    
    train_out <- train_df
    test_out  <- test_df
    train_out$anomaly_score <- anomaly_train
    test_out$anomaly_score  <- anomaly_test
    
    return(list(train = train_out, test = test_out,
                description = "Strategy B (Anomaly Score)"))
  }
  
  if (strategy == "C") {
    # Strategy C: Feature Denoising — DAE reconstructed features
    dae_path_train <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "dae_denoised_train.rds")
    dae_path_test  <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts", "dae_denoised_test.rds")
    
    if (!file.exists(dae_path_train) || !file.exists(dae_path_test)) {
      cat("  [SKIP] Strategy C: DAE denoised features not found.\n")
      cat("         Run generate_vae_dae_artifacts.R first to create them.\n")
      return(NULL)
    }
    
    dae_train <- readRDS(dae_path_train)
    dae_test  <- readRDS(dae_path_test)
    
    train_out <- cbind(dae_train, train_df[, c("y", "size", "sector"), drop = FALSE])
    test_out  <- cbind(dae_test,  test_df[, c("y", "size", "sector"), drop = FALSE])
    
    return(list(train = train_out, test = test_out,
                description = "Strategy C (Feature Denoising / DAE)"))
  }
  
  if (strategy == "D") {
    # Strategy D: Manual Feature Engineering — add ratios & interactions
    train_raw <- Train_Backup
    test_raw  <- Test_Backup
    
    # Compute hand-crafted ratios
    train_out <- train_raw %>%
      mutate(
        leverage = safe_divide(f3, f1),       # debt / assets
        liquidity = safe_divide(f6, f3),      # cash / debt
        profitability = safe_divide(f8, f1),  # profit / assets
        coverage = safe_divide(f9, f7),       # interest coverage
        asset_turnover = safe_divide(f10, f1) # revenue / assets
      ) %>% as.data.frame()

    test_out <- test_raw %>%
      mutate(
        leverage = safe_divide(f3, f1),
        liquidity = safe_divide(f6, f3),
        profitability = safe_divide(f8, f1),
        coverage = safe_divide(f9, f7),
        asset_turnover = safe_divide(f10, f1)
      ) %>% as.data.frame()
    
    # Apply quantile transformation to the new ratio features
    for (col in c("leverage", "liquidity", "profitability", "coverage", "asset_turnover")) {
      res <- QuantileTransformation(train_out[[col]], test_out[[col]])
      train_out[[col]] <- res$train
      test_out[[col]]  <- res$test
    }

    return(list(train = train_out, test = test_out,
                description = "Strategy D (Manual Feature Eng.)"))
  }
  
  stop(paste("Unknown strategy:", strategy))
}


#==============================================================================#
#==== 06 — RF Pipeline Function ===============================================#
#==============================================================================#
# Core pipeline: takes prepared data, runs HPO, trains final model, evaluates.

run_rf_pipeline <- function(train_df, test_df, fold_ids, strategy_name, config) {
  
  cat("\n", strrep("=", 70), "\n")
  cat(sprintf("STRATEGY: %s\n", strategy_name))
  cat(strrep("=", 70), "\n")
  
  # Prepare data for mlr3 — drop non-feature columns
  drop_cols <- c("id", "refdate")
  train_model <- train_df[, !names(train_df) %in% drop_cols]
  test_model  <- test_df[, !names(test_df) %in% drop_cols]
  
  # Ensure y is a factor for classification
  train_model$y <- as.factor(train_model$y)
  test_model$y  <- as.factor(test_model$y)
  
  # Convert sector/size to factor if present
  for (col in c("sector", "size")) {
    if (col %in% names(train_model)) {
      train_model[[col]] <- as.factor(train_model[[col]])
      test_model[[col]]  <- as.factor(test_model[[col]])
    }
  }
  
  # Create mlr3 tasks
  task_train <- TaskClassif$new(
    id = paste0("rf_", strategy_name),
    backend = train_model,
    target = "y",
    positive = "1"
  )
  
  task_test <- TaskClassif$new(
    id = paste0("rf_test_", strategy_name),
    backend = test_model,
    target = "y",
    positive = "1"
  )
  
  n_features <- task_train$n_features
  cat("Features:", n_features, "\n")
  cat("Train obs:", task_train$nrow, "| Test obs:", task_test$nrow, "\n")
  
  # Create custom CV resampling from the fold vector
  custom_cv <- create_custom_cv(task_train, fold_ids)
  
  # Run HPO
  hpo_config <- list(
    n_evals = config$n_evals,
    stagnation_iters = config$stagnation_iters,
    stagnation_threshold = config$stagnation_threshold,
    checkpoint_file = file.path(Charts_RF_Directory,
                                paste0("checkpoint_", strategy_name, ".rds"))
  )
  
  hpo_results <- run_hpo(task_train, custom_cv, n_features, hpo_config)
  
  # Train final model and evaluate
  eval_results <- train_and_eval(hpo_results$best_params, task_train, task_test)
  
  # Also compute Brier score manually for the comparison table
  pred_probs <- eval_results$pred$prob[, "1"]
  actual     <- as.numeric(as.character(eval_results$pred$truth))
  brier      <- BrierScore(pred_probs, actual)
  
  # Compute train AUC (refit prediction on training data)
  train_pred <- eval_results$learner$predict(task_train)
  train_auc  <- train_pred$score(msr("classif.auc"))
  train_probs <- train_pred$prob[, "1"]
  train_actual <- as.numeric(as.character(train_pred$truth))
  train_brier  <- BrierScore(train_probs, train_actual)
  
  cat("\n--- Results ---\n")
  cat(sprintf("  Train AUC:   %.5f\n", train_auc))
  cat(sprintf("  Test AUC:    %.5f\n", eval_results$auc))
  cat(sprintf("  Test Brier:  %.5f\n", brier))
  cat(sprintf("  Test Acc:    %.5f\n", eval_results$acc))
  
  return(list(
    strategy      = strategy_name,
    hpo_results   = hpo_results,
    eval_results  = eval_results,
    train_auc     = train_auc,
    test_auc      = eval_results$auc,
    train_brier   = train_brier,
    test_brier    = brier,
    test_acc      = eval_results$acc,
    n_features    = n_features,
    task_train    = task_train,
    task_test     = task_test
  ))
}


#==============================================================================#
#==== 07 — Execute Pipeline per Strategy ======================================#
#==============================================================================#

hpo_config <- list(
  n_evals = N_EVALS,
  stagnation_iters = STAGNATION_ITERS,
  stagnation_threshold = STAGNATION_THRESHOLD
)

all_results <- list()

for (strategy in STRATEGIES_TO_RUN) {
  
  cat("\n", strrep("#", 70), "\n")
  cat(sprintf("## Processing Strategy: %s\n", strategy))
  cat(strrep("#", 70), "\n")
  
  # Prepare strategy-specific features
  strategy_data <- prepare_strategy_data(strategy, Train_Transformed, Test_Transformed)
  
  if (is.null(strategy_data)) {
    cat(sprintf("  Strategy %s skipped (data not available).\n", strategy))
    next
  }
  
  cat(sprintf("  Description: %s\n", strategy_data$description))
  
  # Run the RF pipeline
  result <- tryCatch({
    run_rf_pipeline(
      train_df = strategy_data$train,
      test_df  = strategy_data$test,
      fold_ids = fold_vector,
      strategy_name = strategy,
      config   = hpo_config
    )
  }, error = function(e) {
    cat(sprintf("  ERROR in strategy %s: %s\n", strategy, e$message))
    NULL
  })
  
  if (!is.null(result)) {
    result$description <- strategy_data$description
    all_results[[strategy]] <- result
    
    # Save per-strategy optimization history chart
    tryCatch({
      p_hist <- plot_optimization_history(result$hpo_results)
      ggsave(
        filename = file.path(Charts_RF_Directory,
                             sprintf("01_OptimHistory_%s.png", strategy)),
        plot = p_hist,
        width = WIDTH_IN, height = HEIGHT_IN, dpi = DPI, limitsize = FALSE
      )
      cat("  Optimization history chart saved.\n")
    }, error = function(e) cat("  Warning: Could not save optimization history chart:", e$message, "\n"))
    
    # Save per-strategy feature importance chart
    tryCatch({
      learner <- result$eval_results$learner
      if (!is.null(learner$model) && inherits(learner$model, "ranger")) {
        # Retrain with importance = "impurity" for feature importance
        imp_learner <- get_learner_config(result$n_features)$learner$clone()
        imp_learner$param_set$values <- c(imp_learner$param_set$values,
                                          result$hpo_results$best_params)
        imp_learner$param_set$values$importance <- "impurity"
        imp_learner$param_set$values$num.threads <- N_CORES
        imp_learner$train(result$task_train)
        
        imp_vals <- imp_learner$model$variable.importance
        imp_df <- data.frame(
          Feature = names(imp_vals),
          Importance = imp_vals / sum(imp_vals)
        ) %>% arrange(desc(Importance)) %>% head(10)
        
        p_imp <- ggplot(imp_df, aes(x = Importance, y = reorder(Feature, Importance))) +
          geom_col(fill = BLUE, width = 0.7) +
          geom_text(aes(label = scales::percent(Importance, accuracy = 0.1)),
                    hjust = -0.1, size = 4.5, fontface = "bold", color = "grey30") +
          scale_x_continuous(labels = scales::percent,
                             expand = expansion(mult = c(0, 0.15))) +
          labs(title = sprintf("RF Feature Importance — %s", strategy_data$description),
               x = "Relative Importance", y = NULL) +
          theme_minimal(base_size = 13) +
          theme(
            plot.title = element_text(face = "bold", size = 16),
            axis.title.x = element_text(size = 13, face = "bold"),
            axis.text.y = element_text(size = 11, face = "bold", color = "black"),
            panel.grid.minor = element_blank(),
            panel.grid.major.y = element_blank(),
            plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
          )
        
        ggsave(
          filename = file.path(Charts_RF_Directory,
                               sprintf("02_FeatureImportance_%s.png", strategy)),
          plot = p_imp,
          width = WIDTH_IN, height = HEIGHT_IN, dpi = DPI, limitsize = FALSE
        )
        cat("  Feature importance chart saved.\n")
      }
    }, error = function(e) cat("  Warning: Could not save feature importance chart:", e$message, "\n"))
    
    # Save per-strategy calibration chart
    tryCatch({
      pred_probs <- result$eval_results$pred$prob[, "1"]
      actual_vals <- as.numeric(as.character(result$eval_results$pred$truth))
      
      calib_data <- data.frame(actual = actual_vals, prob = pred_probs) %>%
        mutate(bin = ntile(prob, 10)) %>%
        group_by(bin) %>%
        summarise(mean_prob = mean(prob), observed_rate = mean(actual), n = n(),
                  .groups = "drop")
      
      calib_long <- calib_data %>%
        select(bin, mean_prob, observed_rate) %>%
        rename(Predicted = mean_prob, Observed = observed_rate) %>%
        pivot_longer(cols = c("Predicted", "Observed"),
                     names_to = "Type", values_to = "Rate") %>%
        mutate(Type = factor(Type, levels = c("Predicted", "Observed")))
      
      p_calib <- ggplot(calib_long, aes(x = factor(bin), y = Rate, fill = Type)) +
        geom_col(position = position_dodge(width = 0.8), width = 0.7) +
        geom_text(aes(label = scales::percent(Rate, accuracy = 0.1)),
                  position = position_dodge(width = 0.8),
                  vjust = -0.5, size = 3.5, fontface = "bold") +
        scale_fill_manual(values = c("Predicted" = BLUE, "Observed" = GREY)) +
        scale_y_continuous(labels = scales::percent,
                           expand = expansion(mult = c(0, 0.15))) +
        labs(title = sprintf("Calibration — %s", strategy_data$description),
             x = "Risk Decile (1 = Lowest Risk, 10 = Highest Risk)",
             y = "Default Rate", fill = "") +
        theme_minimal(base_size = 13) +
        theme(
          plot.title = element_text(face = "bold", size = 16),
          legend.position = "top",
          legend.text = element_text(size = 12, face = "bold"),
          axis.title.x = element_text(size = 13, face = "bold", margin = ggplot2::margin(t = 10)),
          axis.title.y = element_text(size = 13, face = "bold", margin = ggplot2::margin(r = 10)),
          panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
        )
      
      ggsave(
        filename = file.path(Charts_RF_Directory,
                             sprintf("03_Calibration_%s.png", strategy)),
        plot = p_calib,
        width = WIDTH_IN, height = HEIGHT_IN, dpi = DPI, limitsize = FALSE
      )
      cat("  Calibration chart saved.\n")
    }, error = function(e) cat("  Warning: Could not save calibration chart:", e$message, "\n"))
  }
}


#==============================================================================#
#==== 08 — Strategy Comparison Leaderboard ====================================#
#==============================================================================#
# Follows the Plots_final_glm.R pattern: horizontal bar charts + GT tables
# with uplift percentages relative to Base Model.

if (length(all_results) >= 1) {
  
  cat("\n", strrep("=", 70), "\n")
  cat("STRATEGY COMPARISON LEADERBOARD\n")
  cat(strrep("=", 70), "\n")
  
  # ---- Build comparison tables ----
  
  strategy_names <- c(
    "Base" = "Base Model",
    "A"    = "Strategy A (Dim. Reduction)",
    "B"    = "Strategy B (Anomaly Score)",
    "C"    = "Strategy C (Feature Denoising)",
    "D"    = "Strategy D (Manual Feature Eng.)"
  )
  
  comparison_table_train <- do.call(rbind, lapply(names(all_results), function(s) {
    r <- all_results[[s]]
    data.frame(
      Model       = strategy_names[s],
      Type        = "Random Forest",
      AUC         = r$train_auc,
      Brier_Score = r$train_brier,
      stringsAsFactors = FALSE
    )
  }))
  
  comparison_table_test <- do.call(rbind, lapply(names(all_results), function(s) {
    r <- all_results[[s]]
    data.frame(
      Model       = strategy_names[s],
      Type        = "Random Forest",
      AUC         = r$test_auc,
      Brier_Score = r$test_brier,
      stringsAsFactors = FALSE
    )
  }))
  
  # Compute uplifts relative to Base
  compute_uplifts <- function(tbl) {
    base_auc   <- tbl$AUC[tbl$Model == "Base Model"][1]
    base_brier <- tbl$Brier_Score[tbl$Model == "Base Model"][1]
    
    tbl <- tbl %>%
      mutate(
        Base_AUC         = base_auc,
        Base_Brier       = base_brier,
        Uplift_AUC_pct   = (AUC - Base_AUC) / Base_AUC * 100,
        Uplift_Brier_pct = -(Brier_Score - Base_Brier) / Base_Brier * 100
      )
    return(tbl)
  }
  
  if ("Base" %in% names(all_results)) {
    comparison_table_train <- compute_uplifts(comparison_table_train)
    comparison_table_test  <- compute_uplifts(comparison_table_test)
  } else {
    comparison_table_train$Uplift_AUC_pct   <- NA_real_
    comparison_table_train$Uplift_Brier_pct <- NA_real_
    comparison_table_test$Uplift_AUC_pct    <- NA_real_
    comparison_table_test$Uplift_Brier_pct  <- NA_real_
  }
  
  # ---- AUC Leaderboard Charts (Plots_final_glm.R pattern) ----
  
  plot_auc_leaderboard <- function(tbl, set_label) {
    df <- tbl %>%
      mutate(
        Model = factor(Model, levels = rev(Model)),
        auc_lbl = percent(AUC, accuracy = 0.1),
        uplift_lbl = ifelse(
          is.finite(Uplift_AUC_pct),
          paste0(" (", sprintf("%+.1f", Uplift_AUC_pct), "%)"),
          ""
        )
      )
    
    ggplot(df, aes(x = Model, y = AUC)) +
      geom_col(fill = BLUE, width = 0.7) +
      geom_text(
        aes(label = paste0(auc_lbl, uplift_lbl)),
        hjust = -0.05, size = 5, fontface = "bold"
      ) +
      coord_flip() +
      scale_y_continuous(
        labels = percent_format(accuracy = 1),
        expand = expansion(mult = c(0, 0.22))
      ) +
      labs(
        title = sprintf("RF Strategy Performance — %s Set", set_label),
        x = NULL, y = "AUC"
      ) +
      theme_minimal(base_size = 18) +
      theme(
        plot.title = element_text(face = "bold", size = 28, hjust = 0),
        axis.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.margin = margin(20, 40, 20, 20)
      )
  }
  
  # Train leaderboard
  p_auc_train <- plot_auc_leaderboard(comparison_table_train, "Train")
  ggsave(
    filename = file.path(Charts_RF_Directory, "04_AUC_Leaderboard_Train.png"),
    plot = p_auc_train,
    width = WIDTH_IN, height = HEIGHT_IN, dpi = DPI, limitsize = FALSE
  )
  
  # Test leaderboard
  p_auc_test <- plot_auc_leaderboard(comparison_table_test, "Test")
  ggsave(
    filename = file.path(Charts_RF_Directory, "05_AUC_Leaderboard_Test.png"),
    plot = p_auc_test,
    width = WIDTH_IN, height = HEIGHT_IN, dpi = DPI, limitsize = FALSE
  )
  
  cat("AUC leaderboard charts saved.\n")
  
  # ---- GT Leaderboard Tables ----
  
  make_leaderboard_table <- function(tbl, set_label) {
    tbl %>%
      mutate(
        AUC              = round(AUC, 4) * 100,
        Uplift_AUC_pct   = round(Uplift_AUC_pct, 2),
        Brier_Score      = round(Brier_Score, 4) * 100,
        Uplift_Brier_pct = round(Uplift_Brier_pct, 2)
      ) %>%
      select(Model, Type, AUC, Uplift_AUC_pct, Brier_Score, Uplift_Brier_pct) %>%
      gt() %>%
      tab_header(title = sprintf("RF Strategy Leaderboard — %s Set", set_label)) %>%
      cols_label(
        Model            = "Strategy",
        Type             = "Type",
        AUC              = sprintf("%s AUC (%%)", set_label),
        Uplift_AUC_pct   = "AUC Uplift (%)",
        Brier_Score      = "Brier (%)",
        Uplift_Brier_pct = "Brier Uplift (%)"
      ) %>%
      tab_style(
        style = cell_text(weight = "bold"),
        locations = cells_column_labels(everything())
      ) %>%
      tab_style(
        style = cell_text(color = BLUE, weight = "bold"),
        locations = cells_title(groups = "title")
      ) %>%
      tab_options(
        table.font.size = px(16),
        heading.title.font.size = px(24),
        column_labels.font.size = px(16),
        data_row.padding = px(8),
        table.border.top.color = BLUE,
        table.border.bottom.color = BLUE
      )
  }
  
  # Train table
  tbl_train <- make_leaderboard_table(comparison_table_train, "Train")
  gtsave(tbl_train, file.path(Charts_RF_Directory, "06_Leaderboard_Train.png"))
  
  # Test table
  tbl_test <- make_leaderboard_table(comparison_table_test, "Test")
  gtsave(tbl_test, file.path(Charts_RF_Directory, "07_Leaderboard_Test.png"))
  
  cat("GT leaderboard tables saved.\n")
  
  # ---- Print summary ----
  
  cat("\n", strrep("=", 70), "\n")
  cat("FINAL RESULTS SUMMARY\n")
  cat(strrep("=", 70), "\n\n")
  
  cat("--- Train Set ---\n")
  print(comparison_table_train %>%
          select(Model, AUC, Brier_Score, Uplift_AUC_pct) %>%
          arrange(desc(AUC)))
  
  cat("\n--- Test Set ---\n")
  print(comparison_table_test %>%
          select(Model, AUC, Brier_Score, Uplift_AUC_pct) %>%
          arrange(desc(AUC)))
  
  # Save raw results for downstream use
  saveRDS(all_results, file.path(Charts_RF_Directory, "rf_all_results.rds"))
  saveRDS(comparison_table_train, file.path(Charts_RF_Directory, "rf_comparison_train.rds"))
  saveRDS(comparison_table_test, file.path(Charts_RF_Directory, "rf_comparison_test.rds"))
  cat("\nRaw results saved to:", Charts_RF_Directory, "\n")
  
} else {
  cat("\nNo strategies completed successfully. No leaderboard generated.\n")
}


#==============================================================================#
#==== 09 — Cleanup ============================================================#
#==============================================================================#

future::plan("sequential")
cat("\nRandom Forest pipeline complete.\n")
cat(strrep("=", 70), "\n")
