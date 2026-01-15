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
    foldid = foldid_vector,   
    standardize = TRUE        
  )
  
  best_auc <- max(cv_fit$cvm)
  list(Score = best_auc, Pred = 0)
}