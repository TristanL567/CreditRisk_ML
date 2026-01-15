GLM_bayesoptim <- function(alpha) {
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