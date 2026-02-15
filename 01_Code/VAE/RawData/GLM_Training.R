GLM_Training <- function(Data_Train_CV_Vector,
                         Train_Data,
                         n_init_points = 10,
                         n_iter_bayes = 20,
                         ...) {
  
  # Start timer
  start_time <- Sys.time()
  
  ##==============================##
  ## 1. Data preparation.
  ##==============================##
  
  sparse_formula <- as.formula("y ~ . - 1")
  
  # Training Data
  train_y <- as.numeric(as.character(Train_Data$y))
  train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Data)
  
  ##==============================##
  ## 2. Define Objective Function
  ##==============================##
  
  current_bayes_iter <- 0
  total_bayes_runs   <- n_init_points + n_iter_bayes
  
  GLM_bayesoptim <- function(alpha) {
    
    current_bayes_iter <<- current_bayes_iter + 1
    message(sprintf("[%02d/%02d] BayesOpt GLM: alpha=%.4f", 
                    current_bayes_iter, total_bayes_runs, alpha))
    
    cv_fit <- cv.glmnet(
      x = train_matrix, 
      y = train_y,
      family = "binomial",        
      type.measure = "auc",      
      alpha = alpha,
      foldid = Data_Train_CV_Vector,   
      standardize = TRUE        
    )
    
    best_auc <- max(cv_fit$cvm) 
    list(Score = best_auc, Pred = 0)
  }
  
  ##==============================##
  ## 3. Run Optimization
  ##==============================##
  
  bounds_glm <- list(alpha = c(0, 1))
  
  print("Starting Bayesian Optimization for GLM...")
  
  bayes_out_glm <- tryCatch({
    BayesianOptimization(
      FUN = GLM_bayesoptim,
      bounds = bounds_glm,
      init_points = n_init_points,
      n_iter = n_iter_bayes,
      acq = "ei",    
      eps = 0.0,      
      verbose = TRUE
    )
  }, error = function(e) {
    message("Optimization failed: ", e)
    return(NULL)
  })
  
  # Correct early return (now exits GLM_Training properly)
  if(is.null(bayes_out_glm)) return(NULL)
  
  ## Format Results
  results_bayes_glm <- bayes_out_glm$History %>%
    rename(AUC = Value) %>%
    mutate(iteration_index = 1:n()) %>%
    arrange(desc(AUC)) %>%
    as_tibble()
  
  print("--- Top 5 Bayesian GLM Models ---")
  print(head(results_bayes_glm, 5))
  
  ##==============================##
  ## 4. Train Final Model
  ##==============================##
  
  best_bayes_glm <- results_bayes_glm[1, ]
  optimal_alpha <- best_bayes_glm$alpha
  
  message(paste("Training Final GLM Model with alpha =", round(optimal_alpha, 4)))
  
  # Train final model WITH keep = TRUE to get out-of-fold predictions
  final_cv_fit <- cv.glmnet(
    x = train_matrix, 
    y = train_y,
    family = "binomial",
    type.measure = "auc",
    alpha = optimal_alpha,
    foldid = Data_Train_CV_Vector,
    standardize = TRUE,
    keep = TRUE # Crucial for honest Brier Scores
  )
  
  optimal_lambda <- final_cv_fit$lambda.min
  model_bayes_glm <- final_cv_fit
  
  ##==============================##
  ## 5. Predictions & Brier Scores
  ##==============================##
  
  # Extract Out-Of-Fold (OOF) predictions at the optimal lambda.
  # fit.preval returns log-odds (link function) for binomial, so we use plogis() to get probabilities.
  lambda_index <- which(final_cv_fit$lambda == optimal_lambda)
  oof_log_odds <- final_cv_fit$fit.preval[, lambda_index]
  oof_probs <- plogis(oof_log_odds) 
  
  # Create Results Dataframe using OOF probabilities
  df_results <- tibble::tibble(
    Actual = train_y, 
    Predicted = as.numeric(oof_probs) 
  )
  
  # --- Standard Brier Score ---
  brier_score <- mean((df_results$Predicted - df_results$Actual)^2)
  
  # --- Penalized Brier Score (PBS) ---
  R <- 2 
  penalty_term <- (R - 1) / R 
  
  df_results <- df_results %>%
    mutate(
      Predicted_Class = round(Predicted),
      Squared_Error = (Predicted - Actual)^2,
      Penalty = ifelse(Predicted_Class != Actual, penalty_term, 0)
    )
  
  penalized_brier_score <- mean(df_results$Squared_Error + df_results$Penalty)
  
  # End timer
  end_time <- Sys.time()
  time_GLM_bo <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  ##==============================##
  ## 6. Return Statement
  ##==============================##
  
  Results_GLM <- list(
    results = results_bayes_glm,
    optimal_model = model_bayes_glm,
    optimal_parameters = list(alpha = optimal_alpha, lambda = optimal_lambda),
    time = time_GLM_bo,
    Predictions = df_results,  
    Brier_Score = brier_score,
    Penalized_Brier_Score = penalized_brier_score
  )
  
  return(Results_GLM)
}