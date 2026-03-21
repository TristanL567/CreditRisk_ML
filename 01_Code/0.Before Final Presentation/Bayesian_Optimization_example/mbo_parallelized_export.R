# ============================================================================
# Simplified HPO Framework with Save/Resume Support
# ============================================================================

# ----------------------------------------------------------------------------
# Setup and Libraries
# ----------------------------------------------------------------------------
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3mbo)
library(mlr3measures)
library(data.table)
library(ggplot2)
library(ranger)
library(paradox)
library(future)
library(future.apply)
library(parallel)
library(lightgbm)
# ============================================================================
# CONFIGURATION
# ============================================================================

HPO_CONFIG <- list(
  learner_name = "ranger",  # Change this to use different learner
  n_evals = 100L,
  stagnation_iters = 30L,
  stagnation_threshold = 0.001,
  n_folds = 5L,
  train_prop = 0.05,
  seed = 42,
  checkpoint_file = "hpo_checkpoint_ranger.rds"
)

# ============================================================================
# PARALLELIZATION SETUP
# ============================================================================
n_cores <- max(1, parallel::detectCores() - 1)
cat("Setting up parallelization with", n_cores, "cores\n")

plan(multisession, workers = n_cores)
cat("Parallel backend:", class(plan())[1], "\n")
cat("Number of workers:", nbrOfWorkers(), "\n")

# ============================================================================
# LEARNER & SEARCH SPACE DEFINITION
# ============================================================================

#' Define learner and search space
#' Modify this function to change learner and parameters
get_learner_config <- function(n_features) {
  
  # ===== RANDOM FOREST =====
  list(
    learner = lrn("classif.ranger",
                  predict_type = "prob",
                  importance = "impurity",
                  num.threads = 1L),
    search_space = ps(
      num.trees = p_int(lower = 50L, upper = 500L),
      mtry = p_int(lower = 1L, upper = max(1L, n_features)),
      min.node.size = p_int(lower = 1L, upper = 20L),
      max.depth = p_int(lower = 5L, upper = 30L),
      sample.fraction = p_dbl(lower = 0.5, upper = 1.0)
    )
  )
  
  #===== XGBOOST ===== #needs not factor variables
  # list(
  #   learner = lrn("classif.xgboost",
  #                 predict_type = "prob",
  #                 nthread = 1L,
  #                 verbose = 0L),
  #   search_space = ps(
  #     nrounds = p_int(lower = 50L, upper = 500L),
  #     max_depth = p_int(lower = 3L, upper = 15L),
  #     eta = p_dbl(lower = 0.01, upper = 0.3),
  #     min_child_weight = p_dbl(lower = 1, upper = 10),
  #     subsample = p_dbl(lower = 0.5, upper = 1.0),
  #     colsample_bytree = p_dbl(lower = 0.5, upper = 1.0)
  #   )
  # )
  
  # ===== LIGHTGBM ===== for this mlr3extralearners package is needed which is only available on github, no cran
#   list(
#     learner = lrn("classif.lightgbm",
#                   predict_type = "prob",
#                   num_threads = 1L,
#                   verbose = -1L),
#     search_space = ps(
#       num_iterations = p_int(lower = 50L, upper = 500L),
#       max_depth = p_int(lower = 3L, upper = 15L),
#       learning_rate = p_dbl(lower = 0.01, upper = 0.3),
#       num_leaves = p_int(lower = 10L, upper = 255L),
#       feature_fraction = p_dbl(lower = 0.5, upper = 1.0)
#     )
#   )
}

# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

#' Prepare data for HPO
prepare_data <- function(data_path, clip_val = 1e4) {
  load(data_path)
  d <- d[, -c(1, 2)]
  d <- as.data.table(d)
  d[, y := as.factor(y)]
  d[, size := as.factor(size)]
  d[, sector := as.factor(sector)]
  
  d <- na.omit(d)
  
  num_cols <- names(which(sapply(d, is.numeric)))
  for (col in num_cols) {
    set(d, i = which(is.infinite(d[[col]]) & d[[col]] > 0), j = col, value = clip_val)
    set(d, i = which(is.infinite(d[[col]]) & d[[col]] < 0), j = col, value = -clip_val)
  }
  
  cat("Dataset dimensions:", nrow(d), "x", ncol(d), "\n")
  cat("Target distribution:\n")
  print(prop.table(table(d$y)))
  
  return(d)
}

#' Stratified train/test split
split_stratified <- function(dt, target_col, train_prop = 0.05, seed = 42) {
  set.seed(seed)
  dt <- copy(dt)
  dt[, row_id := .I]
  train_idxs <- dt[, .(row_id = sample(row_id, size = floor(.N * train_prop))), 
                   by = target_col][, row_id]
  list(
    train = dt[row_id %in% train_idxs][, row_id := NULL],
    test  = dt[!row_id %in% train_idxs][, row_id := NULL]
  )
}

#' Create stratified folds
create_stratified_folds <- function(dt, target_col, n_folds = 5, seed = 42) {
  set.seed(seed)
  dt <- copy(dt)
  dt[, fold_id := NA_integer_]
  for (cls in unique(dt[[target_col]])) {
    idx <- which(dt[[target_col]] == cls)
    shuffled <- sample(idx)
    dt[shuffled, fold_id := rep(1:n_folds, length.out = length(shuffled))]
  }
  dt
}

#' Create custom CV resampling
create_custom_cv <- function(task, fold_ids) {
  n_folds <- length(unique(fold_ids))
  train_sets <- vector("list", n_folds)
  test_sets  <- vector("list", n_folds)
  for (f in seq_len(n_folds)) {
    test_sets[[f]]  <- which(fold_ids == f)
    train_sets[[f]] <- which(fold_ids != f)
  }
  res <- rsmp("custom")
  res$instantiate(task, train_sets, test_sets)
  res
}

# ============================================================================
# CHECKPOINT SAVE/LOAD
# ============================================================================

#' Save HPO checkpoint (very simple)
save_checkpoint <- function(instance, filepath) {
  checkpoint <- list(
    archive = as.data.table(instance$archive$data),
    best_params = instance$result_x_domain,
    best_score = instance$result_y,
    n_evals = nrow(instance$archive$data),
    save_time = Sys.time()
  )
  saveRDS(checkpoint, filepath)
  cat("Checkpoint saved to:", filepath, "\n")
  invisible(checkpoint)
}

#' Load HPO checkpoint
load_checkpoint <- function(filepath) {
  if (!file.exists(filepath)) {
    return(NULL)
  }
  checkpoint <- readRDS(filepath)
  cat("Checkpoint loaded from:", filepath, "\n")
  cat("  Evaluations completed:", checkpoint$n_evals, "\n")
  cat("  Best CV AUC:", round(checkpoint$best_score, 4), "\n")
  return(checkpoint)
}

# ============================================================================
# MAIN HPO FUNCTION
# ============================================================================

#' Run HPO with automatic checkpoint save/resume
run_hpo <- function(task_train, custom_cv, n_features, config) {
  
  cat("\n", strrep("=", 70), "\n")
  cat("HYPERPARAMETER OPTIMIZATION\n")
  cat(strrep("=", 70), "\n")
  
  # Get learner and search space
  learner_config <- get_learner_config(n_features)
  learner <- learner_config$learner
  search_space <- learner_config$search_space
  
  # Calculate initial design points
  n_dims <- search_space$length
  n_initial_design <- n_dims * 4
  
  cat("Search space dimensions:", n_dims, "\n")
  cat("Initial design points:", n_initial_design, "\n")
  cat("\nSearch space:\n")
  print(search_space)
  
  # Check for existing checkpoint
  existing_checkpoint <- load_checkpoint(config$checkpoint_file)
  
  if (!is.null(existing_checkpoint)) {
    cat("\nResuming from checkpoint with", existing_checkpoint$n_evals, 
        "evaluations completed.\n")
    cat("To start fresh, delete:", config$checkpoint_file, "\n")
  }
  
  # Create tuning instance
  measure_auc <- msr("classif.auc")
  
  instance_bo <- TuningInstanceBatchSingleCrit$new(
    task = task_train,
    learner = learner,
    resampling = custom_cv,
    measure = measure_auc,
    search_space = search_space,
    terminator = trm("combo", list(
      trm("evals", n_evals = config$n_evals),
      trm("stagnation", iters = config$stagnation_iters,
          threshold = config$stagnation_threshold)
    ))
  )
  
  # Run optimization
  tuner_bo <- mlr3tuning::tnr("mbo")
  set.seed(config$seed)
  time_start <- Sys.time()
  
  tuner_bo$optimize(instance_bo)
  
  time_end <- Sys.time()
  time_elapsed <- as.numeric(difftime(time_end, time_start, units = "secs"))
  
  # Save checkpoint
  save_checkpoint(instance_bo, config$checkpoint_file)
  
  # Prepare results
  results <- list(
    best_params = instance_bo$result_x_domain,
    best_cv_auc = instance_bo$result_y,
    n_evals = nrow(instance_bo$archive$data),
    time = time_elapsed,
    archive = as.data.table(instance_bo$archive$data),
    n_initial_design = n_initial_design
  )
  
  cat("\nOptimization completed:\n")
  cat("  Total evaluations:", results$n_evals, "\n")
  cat("  Best CV AUC:", round(results$best_cv_auc, 4), "\n")
  cat("  Time (s):", round(results$time, 1), "\n")
  
  return(results)
}

# ============================================================================
# EVALUATION
# ============================================================================

#' Train final model and evaluate on test set
train_and_eval <- function(best_params, train_task, test_task) {
  
  learner_config <- get_learner_config(train_task$n_features)
  learner <- learner_config$learner$clone()
  
  # Set optimal parameters
  learner$param_set$values <- best_params
  
  # Update thread count for final training (use all cores)
  if ("num.threads" %in% learner$param_set$ids()) {
    learner$param_set$values$num.threads <- n_cores
  } else if ("nthread" %in% learner$param_set$ids()) {
    learner$param_set$values$nthread <- n_cores
  } else if ("num_threads" %in% learner$param_set$ids()) {
    learner$param_set$values$num_threads <- n_cores
  }
  
  learner$train(train_task)
  pred <- learner$predict(test_task)
  
  list(
    learner = learner,
    pred = pred,
    auc = pred$score(msr("classif.auc")),
    acc = pred$score(msr("classif.acc")),
    brier = pred$score(msr("classif.bbrier"))
  )
}

# ============================================================================
# PLOTTING
# ============================================================================

#' Create optimization history plot
plot_optimization_history <- function(results) {
  
  hist_data <- results$archive[, .(iter = seq_len(.N), auc = classif.auc)]
  n_initial_design <- results$n_initial_design
  
  # Label phases
  hist_data[, phase := fifelse(iter <= n_initial_design,
                               "Warmup (Random)", "Bayesian Optimization")]
  hist_data[, phase := factor(phase, levels = c("Warmup (Random)",
                                                "Bayesian Optimization"))]
  
  # Calculate cumulative best
  hist_data[, cummax_auc := cummax(auc)]
  
  # Mark best point
  best_iter <- hist_data[which.max(auc), iter]
  hist_data[, is_best := iter == best_iter]
  
  # Create plot
  p <- ggplot(hist_data, aes(x = iter, y = auc)) +
    geom_point(aes(color = phase, shape = phase), size = 3, alpha = 0.7) +
    geom_line(aes(y = cummax_auc), color = "darkgreen", linetype = "dashed",
              linewidth = 1) +
    geom_point(data = hist_data[is_best == TRUE],
               aes(x = iter, y = auc),
               color = "red", size = 5, shape = 8, stroke = 2) +
    geom_vline(xintercept = n_initial_design + 0.5, linetype = "dotted",
               color = "gray40", linewidth = 1) +
    annotate("text", x = n_initial_design / 2, y = min(hist_data$auc),
             label = "Warmup", color = "gray40", size = 4, vjust = -0.5) +
    annotate("text", x = n_initial_design + (nrow(hist_data) - n_initial_design) / 2,
             y = min(hist_data$auc),
             label = "BO", color = "gray40", size = 4, vjust = -0.5) +
    scale_color_manual(values = c("Warmup (Random)" = "#E69F00",
                                  "Bayesian Optimization" = "#0072B2")) +
    scale_shape_manual(values = c("Warmup (Random)" = 16,
                                  "Bayesian Optimization" = 17)) +
    labs(title = "MBO Optimization History: Warmup vs BO Phase",
         subtitle = paste0("Initial design: ", n_initial_design, " points | ",
                           "Best AUC: ", round(max(hist_data$auc), 4),
                           " at iteration ", best_iter),
         x = "Iteration",
         y = "CV AUC",
         color = "Phase",
         shape = "Phase") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  
  return(p)
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Load and prepare data
set.seed(HPO_CONFIG$seed)
d <- prepare_data("path_to_data")

# Split data
split_res <- split_stratified(d, "y", train_prop = HPO_CONFIG$train_prop,
                              seed = HPO_CONFIG$seed)
train_data <- split_res$train
test_data <- split_res$test

cat("\nTrain size:", nrow(train_data), "Test size:", nrow(test_data), "\n")
cat("Train distribution:\n")
print(prop.table(table(train_data$y)))
cat("Test distribution:\n")
print(prop.table(table(test_data$y)))

# Create stratified folds
train_data <- create_stratified_folds(train_data, "y",
                                      n_folds = HPO_CONFIG$n_folds,
                                      seed = HPO_CONFIG$seed)
cat("\nFold counts:\n")
print(table(train_data$fold_id))

# Prepare training task
fold_ids <- train_data$fold_id
train_data[, y := droplevels(as.factor(y))]
train_backend <- copy(train_data)[, "fold_id" := NULL]
train_backend[, y := factor(y)]

task_train <- TaskClassif$new(
  id = "hpo_train",
  backend = train_backend,
  target = "y",
  positive = levels(train_backend$y)[2]
)

custom_cv <- create_custom_cv(task_train, fold_ids)
cat("Custom CV folds:", custom_cv$iters, "\n")

# Get number of features
non_feature_cols <- c("y", "fold_id")
feature_names <- setdiff(colnames(train_data), non_feature_cols)
n_features <- length(feature_names)
cat("Number of features:", n_features, "\n")

# ============================================================================
# RUN HPO
# ============================================================================

hpo_results <- run_hpo(task_train, custom_cv, n_features, HPO_CONFIG)

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

# Prepare test task
test_data[, y := factor(y)]
task_test <- TaskClassif$new(
  id = "hpo_test",
  backend = test_data,
  target = "y",
  positive = levels(test_data$y)[2]
)

cat("\n", strrep("=", 70), "\n")
cat("FINAL EVALUATION ON TEST SET\n")
cat(strrep("=", 70), "\n")

eval_results <- train_and_eval(hpo_results$best_params, task_train, task_test)

cat("\nTest Metrics:\n")
cat("  AUC:", round(eval_results$auc, 4), "\n")
cat("  Accuracy:", round(eval_results$acc, 4), "\n")
cat("  Brier Score:", round(eval_results$brier, 4), "\n")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

summary_table <- data.table(
  Metric = c("Total Evaluations", "Warmup Points", "BO Evaluations",
             "Optimization Time (s)", "Best CV AUC", "Test AUC", "Test Accuracy"),
  Value = c(hpo_results$n_evals,
            min(hpo_results$n_initial_design, hpo_results$n_evals),
            max(0, hpo_results$n_evals - hpo_results$n_initial_design),
            round(hpo_results$time, 2),
            round(hpo_results$best_cv_auc, 4),
            round(eval_results$auc, 4),
            round(eval_results$acc, 4))
)

cat("\n")
print(summary_table)

cat("\nBest hyperparameters:\n")
print(hpo_results$best_params)

# ============================================================================
# PLOT
# ============================================================================

p1 <- plot_optimization_history(hpo_results)
print(p1)
ggsave("mbo_optimization_history.png", p1, width = 10, height = 6, dpi = 150)

# ============================================================================
# CLEANUP
# ============================================================================

plan(sequential)
cat("\nParallel backend reset to sequential.\n")
