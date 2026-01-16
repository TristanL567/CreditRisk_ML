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
