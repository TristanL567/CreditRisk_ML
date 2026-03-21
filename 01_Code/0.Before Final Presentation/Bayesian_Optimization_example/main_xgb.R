#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("dplyr", "caret", "lubridate", "purrr",
              "Matrix", "pROC",           ## Sparse Matrices and efficient AUC computation.
              "xgboost",                  ## XGBoost library.
              "rBayesianOptimization",    ## Bayesian Optimization.
              "ggplot2", "Ckmeans.1d.dp",  ## Plotting & Charts | XG-Charts / Feature Importance.
              "mlr3", "mlr3learners", "mlr3tuning",
              "mlr3mbo", "mlr3measures", "data.table",
              "ranger", "paradox", "future", 
              "future.apply", "parallel"
)

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

#==== 1B - Functions ==========================================================#

sourceFunctions <- function (functionDirectory)  {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R", 
                              full.names = T)
  ssource <- function(path) {
    try(source(path, echo = F, verbose = F, print.eval = T, 
               local = F))
  }
  sapply(functionFiles, ssource)
}


#==== 1C - Parameters =========================================================#

## Directories.
Data_Path <- "M:/WS2025" ## Needs to be set manually.
Data_Directory <- file.path(Data_Path, "data.rda")
Charts_Directory <- file.path(Path, "03_Charts")

## Charts Directories.
Charts_XGBoost_Directory <- file.path(Charts_Directory, "XGBoost")
temp_name <- "eta_100rounds_NOPIT"

Functions_Directory <- file.path(Path, "01_Code/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Data Sampling.
set.seed(123)

## Charts.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"


width <- 3750
heigth <- 1833

## Other.


#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02A - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

#==== 02B - Pre-process checks ================================================#

## Apply the simple methodology to exclude some observations (outlined by the Leitner Notes).
## Remove all ratios.

Data <- DataPreprocessing(DataPreprocessing)

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

#==== 02C - Data Sampling =====================================================#
set.seed(123)
Data_Sampled <- MVstratifiedsampling(Data,
                                     strat_vars = c("sector", "y"),
                                     Train_size = 0.7)

Train <- Data_Sampled[["Train"]]
Test <- Data_Sampled[["Test"]]

## Quick validation.
# Analyze the Training Set
analyse_MVstratifiedsampling(Train, "TRAIN SET", 
                             target_col = "y", 
                             sector_col = "sector", 
                             date_col = "refdate")

# Analyze the Test Set
analyse_MVstratifiedsampling(Test, "TEST SET", 
                             target_col = "y", 
                             sector_col = "sector", 
                             date_col = "refdate")



## Get fold_id for CV folds
MVstratifiedCV <- function(Data,
                           strat_vars = c("sector", "y"),
                           num_folds = 5) {
  
  # Create firm-level profile (one row per firm)
  firm_profile <- Data %>%
    group_by(id) %>%
    summarise(
      y = max(y), 
      sector = first(sector),
      size = first(size),
      .groups = 'drop'
    ) %>%
    mutate(
      Strat_Key = interaction(select(., all_of(strat_vars)), drop = TRUE)
    )
  
  # Create stratified folds at the firm level
  fold_assignments <- createFolds(
    y = firm_profile$Strat_Key,
    k = num_folds,
    list = FALSE,
    returnTrain = FALSE
  )
  
  # Map firm-level folds to firm IDs
  firm_folds <- firm_profile %>%
    mutate(fold = fold_assignments) %>%
    select(id, fold)
  
  # Join back to original data to get row-level fold assignments
  fold_ids <- Data %>%
    left_join(firm_folds, by = "id") %>%
    pull(fold)
  
  return(fold_ids)
}

# Get fold assignments (vector of length nrow(Data))
set.seed(123)
fold_ids <- MVstratifiedCV(Train, 
                           strat_vars = c("sector", "y"), 
                           num_folds = 5)

## Exclude id and refdate.
Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
Train <- Train[, -which(names(Train) %in% Exclude)]
Train_Backup <- Train
Test <- Test[, -which(names(Test) %in% Exclude)]
Test_Backup <- Test

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

DivideByTotalAssets <- FALSE

#==== 03A - Standardization ===================================================#

if(DivideByTotalAssets){
  
  asset_col <- "f1" 
  cols_to_scale <- c("f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")
  
  safe_divide <- function(numerator, denominator) {
    if_else(denominator == 0 | is.na(denominator), 
            0,                       
            numerator / denominator
    )
  }
  
  Train <- Train %>%
    mutate(
      across(
        .cols = all_of(cols_to_scale),
        .fns = ~ safe_divide(.x, .data[[asset_col]])
      )
    ) %>%
    as.data.frame()
  
  Test <- Test %>%
    mutate(
      across(
        .cols = all_of(cols_to_scale),
        .fns = ~ safe_divide(.x, .data[[asset_col]])
      )
    ) %>%
    as.data.frame()
  
}

#==== 03B - Quantile Transformation ===========================================#
ApplyQuantileTransformation <- FALSE

Train_Transformed <- Train
Test_Transformed <- Test

if(ApplyQuantileTransformation){
  num_cols <- c("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")
  
  for (col in num_cols) {
    res <- QuantileTransformation(Train[[col]], Test[[col]])
    Train_Transformed[[col]] <- res$train
    Test_Transformed[[col]] <- res$test
  }
  
  summary(Train_Transformed$f1)
}
## Now work with:
## Train_Transformed
## Test_Transformed

#==============================================================================#
#==== 04 - GLMs ===============================================================#
#==============================================================================#

#==== 04A - GLMs ==============================================================#


#==== 04B - Regularized GLMs ==================================================#


#==============================================================================#
#==== 05 - Decision Trees =====================================================#
#==============================================================================#

#==== 05A - Random Forest =====================================================#


#==== 05B - AdaBoost ==========================================================#


#==== 05C - XGBoost ===========================================================#

## To-Do: Fix overall parameters at the beginning of the code.
## Comparison of hyperparameter tuning methods.
## Visualisations and Outputs.

############=======================================This needs to be rewritten to be imported functions and then just call them ==========##################
##==============================##
## Data preparation.
##==============================##
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


##==============================##
## HPO preparation.
##==============================##

HPO_CONFIG <- list(
  learner_name = "xgboost",  # Change this to use different learner
  n_evals = 500L,
  stagnation_iters = 50L,
  stagnation_threshold = 0.001,
  n_folds = 5L,
  train_prop = 0.7,
  checkpoint_file = paste0("hpo_checkpoint_xgboost_", temp_name, ".rds") ######naming
)

# ============================================================================
# PARALLELIZATION SETUP
# ============================================================================
n_cores <- max(1, parallel::detectCores() - 1)
cat("Setting up parallelization with", n_cores, "cores\n")

set.seed(123)
plan(multisession, workers = n_cores)
cat("Parallel backend:", class(plan())[1], "\n")
cat("Number of workers:", nbrOfWorkers(), "\n")

# ============================================================================
# LEARNER & SEARCH SPACE DEFINITION
# ============================================================================

#' Define learner and search space
#' Modify this function to change learner and parameters
get_learner_config <- function(n_features) {
  #===== XGBOOST ===== #needs not factor variables
  list(
    learner = lrn("classif.xgboost",
                  predict_type = "prob",
                  nthread = 1L,
                  verbose = 0L,
                  nrounds = 100L),
    search_space = ps(
      max_depth = p_int(lower = 3L, upper = 15L),
      eta = p_dbl(lower = 0.001, upper = 0.3),
      
      lambda = p_dbl(lower = 0.0, upper = 2.0),
      gamma = p_dbl(lower = 0.0, upper = 2.0),

      subsample = p_dbl(lower = 0.5, upper = 1.0),
      colsample_bytree = p_dbl(lower = 0.5, upper = 1.0),
      
      scale_pos_weight = p_dbl(lower = 0.5, upper = 200.0)
    )
  )
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
  n_initial_design <- n_dims * 5
  
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
  set.seed(123)
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
  set.seed(123)
  learner_config <- get_learner_config(train_task$n_features)
  learner <- learner_config$learner$clone()


  # Set optimal parameters
  learner$param_set$values <- c(learner$param_set$values, best_params)
  
  learner$param_set$values$nrounds <- 100L
  
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

############======================================= End of these functions, now call them ==========##################

##==============================##
## Execution.
##==============================##
#it works with data.table, needs y as factor, everything else should not be a factor for xgboost
Train_Transformed <- as.data.table(Train_Transformed)
Test_Transformed <- as.data.table(Test_Transformed)
Test_Transformed[, y := factor(y)]
Train_Transformed[, y := factor(y)]
size_map <- c("Tiny" = 0, "Small" = 1)
Train_Transformed[, size := size_map[size]]
Test_Transformed[, size := size_map[size]]
# OHE for sector
# sector_dummies_train <- model.matrix(~sector - 1, Train_Transformed)
# sector_dummies_test <- model.matrix(~sector - 1, Test_Transformed)
# 

# Convert to factor with SAME levels for both
all_levels <- union(Train_Transformed$sector, Test_Transformed$sector)

Train_Transformed[, sector := factor(sector, levels = all_levels)]
Test_Transformed[, sector := factor(sector, levels = all_levels)]

# Now model.matrix will create identical columns
sector_dummies_train <- model.matrix(~sector - 1, Train_Transformed)
sector_dummies_test <- model.matrix(~sector - 1, Test_Transformed)
Train_Transformed <- cbind(Train_Transformed[, -"sector"], sector_dummies_train)
Test_Transformed <- cbind(Test_Transformed[, -"sector"], sector_dummies_test)

task_train <- TaskClassif$new(
  id = "hpo_train",
  backend = Train_Transformed,
  target = "y",
  positive = levels(Train_Transformed$y)[2]
)

custom_cv <- create_custom_cv(task_train, fold_ids)

non_feature_cols <- c("y")
feature_names <- setdiff(colnames(Train_Transformed), non_feature_cols)
n_features <- length(feature_names)

#optimization
hpo_results <- run_hpo(task_train, custom_cv, n_features, HPO_CONFIG)

#eval on test
task_test <- TaskClassif$new(
  id = "hpo_test",
  backend = Test_Transformed,
  target = "y",
  positive = levels(Test_Transformed$y)[2]
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
ggsave(paste0("mbo_optimization_history_", temp_name, ".png"), p1, width = 10, height = 6, dpi = 150) ####naming

# ============================================================================
# CLEANUP
# ============================================================================

plan(sequential)
cat("\nParallel backend reset to sequential.\n")

#==============================================================================#
#==== 06 - Save Final Model ===================================================#
#==============================================================================#

# Define output directory and filename
Model_Directory <- file.path(Path, "04_Models")
if (!dir.exists(Model_Directory)) {
  dir.create(Model_Directory, recursive = TRUE)
}

# Create a comprehensive model object to save
final_model_object <- list(
  # The trained learner (contains the model)
  learner = eval_results$learner,
  
  # Best hyperparameters from HPO
  best_params = hpo_results$best_params,
  
  # Performance metrics
  metrics = list(
    cv_auc = hpo_results$best_cv_auc,
    test_auc = eval_results$auc,
    test_accuracy = eval_results$acc,
    test_brier = eval_results$brier
  ),
  
  # Metadata
  metadata = list(
    model_type = "xgboost",
    n_features = n_features,
    feature_names = feature_names,
    training_date = Sys.time(),
    n_hpo_evals = hpo_results$n_evals,
    hpo_time_seconds = hpo_results$time
  )
)

# Save the model object
model_filename <- file.path(Model_Directory, 
                            paste0("xgboost_", temp_name, ####naming
                                   format(Sys.Date(), "%Y%m%d"), 
                                   ".rds"))

saveRDS(final_model_object, model_filename)
cat("\nFinal model saved to:", model_filename, "\n")

# To load and use the model later:
# loaded_model <- readRDS(model_filename)
# new_predictions <- loaded_model$learner$predict_newdata(new_data)