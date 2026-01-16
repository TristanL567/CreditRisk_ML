XGBoost_gridsearch <- function(eta, max_depth, subsample, colsample_bytree, 
                               nrounds, early_stopping_rounds, 
                               current_iter, total_iters,
                               folds_custom) {
  
  ## Prints the progress.
  message(sprintf("[%02d/%02d] CV: eta=%.2f | depth=%d | subsample=%.1f | cols=%.1f", 
                  current_iter, total_iters, eta, max_depth, subsample, colsample_bytree))
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree
  )
  
  ## Run the CV.
  cv_res <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = nrounds, 
    folds = folds_custom,
    # nfold = 5,                                 ## No random splitting. We implemented stratified sampling approach.
    early_stopping_rounds = early_stopping_rounds, 
    verbose = 0,
    maximize = TRUE
  )
  
  ## Return statement.
  tibble(
    AUC = max(cv_res$evaluation_log$test_auc_mean),
    Best_Rounds = cv_res$best_iteration,
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    nrounds = nrounds,
    early_stopping_rounds = early_stopping_rounds
  )
}