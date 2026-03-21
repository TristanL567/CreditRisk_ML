XGBoost_Training_revised <- function(
    Data_Train_CV_List,
    Train_Data,
    n_init_points      = 10,
    n_iter_bayes       = 20,
    nthread            = parallel::detectCores() - 1,
    early_stop_bo      = 20,
    early_stop_final   = 50,
    nrounds_bo         = 1000,
    nrounds_final      = 2000,
    eval_metric        = "auc"
) {
  
  ##============================================================##
  ## 1. Data Preparation
  ##============================================================##
  
  message("--- XGBoost Training Start ---")
  
  ## Restore na.action on exit regardless of success/failure
  on.exit(options(na.action = getOption("na.action")), add = TRUE)
  options(na.action = "na.pass")
  
  sparse_formula <- as.formula("y ~ . - 1")
  train_y        <- as.integer(as.character(Train_Data$y))
  
  ## Build matrix once — reused for both CV and final predictions
  train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Data)
  dtrain       <- xgb.DMatrix(data = train_matrix, label = train_y)
  
  ## Class imbalance weight — critical for credit default data
  n_neg              <- sum(train_y == 0L)
  n_pos              <- sum(train_y == 1L)
  scale_pos_weight   <- n_neg / n_pos
  
  cat(sprintf("  Class balance — Negatives: %d | Positives: %d | Weight: %.2f\n",
              n_neg, n_pos, scale_pos_weight))
  
  ##============================================================##
  ## 2. Bayesian Optimisation
  ##============================================================##
  
  current_bayes_iter <- 0L
  total_bayes_runs   <- n_init_points + n_iter_bayes
  
  time_XGBoost_bo <- system.time({
    
    XGBoost_bayesoptim <- function(eta, max_depth, subsample, colsample_bytree,
                                   min_child_weight, gamma, lambda, alpha) {
      
      current_bayes_iter <<- current_bayes_iter + 1L
      current_depth       <- as.integer(round(max_depth))
      current_mcw         <- as.integer(round(min_child_weight))
      
      message(sprintf(
        "[%02d/%02d] eta=%.3f | depth=%d | subs=%.2f | col=%.2f | mcw=%d | gamma=%.2f | L2=%.2f | L1=%.2f",
        current_bayes_iter, total_bayes_runs,
        eta, current_depth, subsample, colsample_bytree,
        current_mcw, gamma, lambda, alpha
      ))
      
      params <- list(
        booster          = "gbtree",
        objective        = "binary:logistic",
        eval_metric      = eval_metric,
        eta              = eta,
        max_depth        = current_depth,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = current_mcw,
        gamma            = gamma,
        lambda           = lambda,
        alpha            = alpha,
        scale_pos_weight = scale_pos_weight,
        max_delta_step   = 1,        ## Recommended for imbalanced classification
        nthread          = nthread
      )
      
      cv_res <- tryCatch(
        xgb.cv(
          params               = params,
          data                 = dtrain,
          nrounds              = nrounds_bo,
          folds                = Data_Train_CV_List,
          early_stopping_rounds = early_stop_bo,
          verbose              = 0,
          maximize             = TRUE
        ),
        error = function(e) {
          message("  CV failed for this iteration: ", e$message)
          return(NULL)
        }
      )
      
      if (is.null(cv_res)) return(list(Score = -Inf, Pred = 0))
      
      best_auc <- max(cv_res$evaluation_log[[paste0("test_", eval_metric, "_mean")]])
      list(Score = best_auc, Pred = 0)
    }
    
    ##============================================================##
    ## 3. Run Optimisation
    ##============================================================##
    
    bounds_bayes <- list(
      eta              = c(0.01, 0.30),
      max_depth        = c(3L,   8L),
      subsample        = c(0.50, 1.00),
      colsample_bytree = c(0.50, 1.00),
      min_child_weight = c(1L,   20L),   ## Higher = more conservative splits
      gamma            = c(0.00, 5.00),  ## Min loss reduction to split
      lambda           = c(0.00, 5.00),  ## L2 regularisation
      alpha            = c(0.00, 5.00)   ## L1 regularisation — sparsity
    )
    
    bayes_out <- tryCatch(
      BayesianOptimization(
        FUN          = XGBoost_bayesoptim,
        bounds       = bounds_bayes,
        init_points  = n_init_points,
        n_iter       = n_iter_bayes,
        acq          = "ucb",
        kappa        = 2.576,
        verbose      = FALSE
      ),
      error = function(e) {
        message("Bayesian Optimisation failed: ", e$message)
        return(NULL)
      }
    )
    
    if (is.null(bayes_out)) stop("Bayesian Optimisation returned NULL — cannot continue.")
    
    ##============================================================##
    ## 4. Parse & Rank Results
    ##============================================================##
    
    results_bayes <- bayes_out$History %>%
      rename(AUC = Value) %>%
      mutate(
        max_depth        = as.integer(round(max_depth)),
        min_child_weight = as.integer(round(min_child_weight)),
        iteration_index  = seq_len(n())
      ) %>%
      arrange(desc(AUC)) %>%
      as_tibble()
    
    message("--- Top 3 Optimised Configurations ---")
    print(head(results_bayes, 3))
    
    ##============================================================##
    ## 5. Final Model Training
    ##============================================================##
    
    best_bayes <- results_bayes[1, ]
    
    final_params <- list(
      booster          = "gbtree",
      objective        = "binary:logistic",
      eval_metric      = eval_metric,
      eta              = best_bayes$eta,
      max_depth        = best_bayes$max_depth,
      subsample        = best_bayes$subsample,
      colsample_bytree = best_bayes$colsample_bytree,
      min_child_weight = best_bayes$min_child_weight,
      gamma            = best_bayes$gamma,
      lambda           = best_bayes$lambda,
      alpha            = best_bayes$alpha,
      scale_pos_weight = scale_pos_weight,
      max_delta_step   = 1,
      nthread          = nthread
    )
    
    ## Determine optimal rounds with final params
    message("  Determining optimal nrounds for best configuration...")
    
    check_cv <- xgb.cv(
      params                = final_params,
      data                  = dtrain,
      nrounds               = nrounds_final,
      folds                 = Data_Train_CV_List,
      early_stopping_rounds = early_stop_final,
      verbose               = 0,
      maximize              = TRUE
    )
    
    optimal_rounds <- check_cv$evaluation_log[
      which.max(check_cv$evaluation_log[[paste0("test_", eval_metric, "_mean")]])
    ]$iter
    
    ## CV AUC at optimal rounds — this is the honest performance estimate
    cv_auc_mean <- max(check_cv$evaluation_log[[paste0("test_", eval_metric, "_mean")]])
    cv_auc_sd   <- check_cv$evaluation_log[
      which.max(check_cv$evaluation_log[[paste0("test_", eval_metric, "_mean")]])
    ][[paste0("test_", eval_metric, "_std")]]
    
    message(sprintf("  Optimal rounds: %d | CV AUC: %.4f (+/- %.4f)",
                    optimal_rounds, cv_auc_mean, cv_auc_sd))
    
    ## Train final model on full training set
    model_final <- xgb.train(
      params  = final_params,
      data    = dtrain,
      nrounds = optimal_rounds,
      verbose = 0
    )
    
  }) ## End system.time
  
  ##============================================================##
  ## 6. Evaluation (In-Sample — clearly labelled)
  ##============================================================##
  
  ## In-sample predictions — reuse train_matrix, no rebuild needed
  preds_insample <- predict(model_final, train_matrix)
  
  df_insample <- tibble::tibble(
    Actual    = train_y,
    Predicted = preds_insample
  ) %>%
    mutate(Predicted_Class = as.integer(Predicted >= 0.5))
  
  ## Standard Brier Score
  brier_score <- mean((df_insample$Predicted - df_insample$Actual)^2)
  
  ## Calibration check — mean predicted prob vs actual default rate
  calibration_gap <- mean(df_insample$Predicted) - mean(df_insample$Actual)
  
  ## In-sample AUC (will be optimistic — use cv_auc_mean for honest estimate)
  insample_auc <- as.numeric(pROC::auc(pROC::roc(
    response  = df_insample$Actual,
    predictor = df_insample$Predicted,
    quiet     = TRUE
  )))
  
  ##============================================================##
  ## 7. Return
  ##============================================================##
  
  message(sprintf("--- XGBoost Complete | Time: %.1fs | CV AUC: %.4f | In-sample AUC: %.4f ---",
                  time_XGBoost_bo["elapsed"], cv_auc_mean, insample_auc))
  
  list(
    ## Model
    optimal_model      = model_final,
    optimal_rounds     = optimal_rounds,
    optimal_parameters = final_params,
    
    ## Optimisation history
    bayes_results      = results_bayes,
    
    ## Honest performance estimate (use this for model comparison)
    cv_auc_mean        = cv_auc_mean,
    cv_auc_sd          = cv_auc_sd,
    
    ## In-sample metrics (clearly labelled — not for model comparison)
    insample_auc       = insample_auc,
    insample_brier     = brier_score,
    calibration_gap    = calibration_gap,
    insample_preds     = df_insample,
    
    ## Meta
    scale_pos_weight   = scale_pos_weight,
    time               = time_XGBoost_bo
  )
}