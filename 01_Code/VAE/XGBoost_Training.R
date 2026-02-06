XGBoost_Training <- function(Data_Train_CV_List,
                             Train_Data,
                             n_init_points = 10,
                             n_iter_bayes = 20,
                             ...){

##==============================##
## 1. Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
  
# Training Data
train_y <- as.numeric(as.character(Train_Data$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Data)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)
  
time_XGBoost_bo <- system.time({
    
##==============================##
## 2. Define Objective Function (LOCALLY)
##==============================##
  
  current_bayes_iter <- 0 
  total_bayes_runs   <- n_init_points + n_iter_bayes 
  
XGBoost_bayesoptim <- function(eta, max_depth, subsample, colsample_bytree) {
    
    current_bayes_iter <<- current_bayes_iter + 1
    current_depth <- as.integer(round(max_depth))
    
    message(sprintf("[%02d/%02d] BayesOpt: eta=%.3f | depth=%d | subs=%.2f | col=%.2f", 
                    current_bayes_iter, total_bayes_runs, 
                    eta, current_depth, subsample, colsample_bytree))
    
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = eta,
      max_depth = current_depth,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      nthread = 4
    )
    
    cv_res <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 1000, 
      folds = Data_Train_CV_List, 
      early_stopping_rounds = 20, 
      verbose = 0,
      maximize = TRUE
    )
    
    list(Score = max(cv_res$evaluation_log$test_auc_mean), Pred = 0)
  }
  
##==============================##
## 3. Run Optimization
##==============================##
    
tryCatch({
      
      ### Parameters
      bounds_bayes <- list(
        eta = c(0.01, 0.3),
        max_depth = c(3L, 8L), 
        subsample = c(0.5, 1.0),
        colsample_bytree = c(0.5, 1.0)
      )
      
      # Wrap in tryCatch to prevent losing all progress if one iter fails
      bayes_out <- tryCatch({
        BayesianOptimization(
          FUN = XGBoost_bayesoptim,
          bounds = bounds_bayes,
          init_points = n_init_points,
          n_iter = n_iter_bayes,
          acq = "ucb",
          kappa = 2.576,
          eps = 0.0,
          verbose = TRUE
        )
      }, error = function(e) {
        message("Optimization failed: ", e)
        return(NULL)
      })
      
      if(is.null(bayes_out)) return(NULL)
      
      
      ## Format Results
      results_bayes <- bayes_out$History %>%
        rename(AUC = Value) %>%
        mutate(
          max_depth = as.integer(round(max_depth)),
          iteration_index = 1:n()
        ) %>%
        arrange(desc(AUC)) %>%
        as_tibble()
      
      print("--- Top 3 Optimized Models ---")
      print(head(results_bayes, 3))
      
      
      ##==============================##
      ## 4. Train Final Model
      ##==============================##
      
      best_bayes <- results_bayes[1, ]
      
      final_params_bayes <- list(
        booster = "gbtree", 
        objective = "binary:logistic", 
        eval_metric = "auc",
        eta = best_bayes$eta, 
        max_depth = best_bayes$max_depth, 
        subsample = best_bayes$subsample, 
        colsample_bytree = best_bayes$colsample_bytree
      )
      
      # Final CV to get exact optimal rounds (more precise than Bayes run)
      message("Determining optimal rounds for Best Model...")
      check_cv_bayes <- xgb.cv(
        params = final_params_bayes,
        data = dtrain,
        nrounds = 2000,
        folds = Data_Train_CV_List,
        early_stopping_rounds = 50,
        verbose = 0,
        maximize = TRUE
      )
      
      optimal_rounds <- check_cv_bayes$evaluation_log[which.max(check_cv_bayes$evaluation_log$test_auc_mean)]$iter
      
      message(paste("Training Final Model with", optimal_rounds, "rounds..."))
      model_bayes <- xgb.train(
        params = final_params_bayes, 
        data = dtrain,
        nrounds = optimal_rounds, 
        verbose = 0
      )
      
}, error = function(e) message(e))
      
}) # End system.time
  
### Results
  Results_XGBoost <- list(
    results = results_bayes,
    optimal_model = model_bayes,
    optimal_rounds = optimal_rounds,
    optimal_parameters = final_params_bayes,
    time = time_XGBoost_bo
)
  
##==============================##
## 4. Return Statement
##==============================##
  
  return(Results_XGBoost)
  
}
  