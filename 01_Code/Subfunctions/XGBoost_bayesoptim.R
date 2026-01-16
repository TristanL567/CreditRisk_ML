XGBoost_bayesoptim <- function(eta, max_depth, subsample, colsample_bytree) {
    
    current_bayes_iter <<- current_bayes_iter + 1
    current_depth <- as.integer(round(max_depth))
    
    ## Prints the progress.
    message(sprintf("[%02d/%02d] BayesOpt: eta=%.3f | depth=%d | subsample=%.2f | cols=%.2f", 
                    current_bayes_iter, total_bayes_runs, 
                    eta, current_depth, subsample, colsample_bytree))
    
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = eta,
      max_depth = current_depth,
      subsample = subsample,
      colsample_bytree = colsample_bytree
    )
    
    ## Run the CV.
    cv_res <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = 2000, 
      folds = cv_folds_list,
      # nfold = 5,
      early_stopping_rounds = 50,
      verbose = 0,
      maximize = TRUE
    )
    
    ## Return statement.
    list(Score = max(cv_res$evaluation_log$test_auc_mean), Pred = 0)
  }