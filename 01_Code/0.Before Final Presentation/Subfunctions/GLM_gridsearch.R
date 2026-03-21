GLM_gridsearch <- function(alpha, current_iter, total_iters, 
                           verbose = TRUE) {
  
  if(verbose) {
    message(sprintf("[%02d/%02d] GLM Elastic Net: alpha=%.4f", 
                    current_iter, total_iters, alpha))
  }
  
  cv_fit <- cv.glmnet(
    x = train_matrix, 
    y = train_y,
    family = "binomial",       
    type.measure = "auc",     
    alpha = alpha,
    foldid = Data_Train_CV_Vector,   
    standardize = TRUE        
  )
  
  best_lambda_idx <- which(cv_fit$lambda == cv_fit$lambda.min)
  best_cv_auc <- cv_fit$cvm[best_lambda_idx]
  train_probs <- predict(cv_fit, 
                         newx = train_matrix, 
                         s = "lambda.min", 
                         type = "response")
  
  train_roc <- roc(train_y, as.vector(train_probs), quiet = TRUE)
  train_auc_val <- pROC::auc(train_roc)
  
  tibble(
    alpha = alpha,
    lambda_best = cv_fit$lambda.min,
    CV_AUC = best_cv_auc,
    Train_AUC = as.numeric(train_auc_val),
    n_nonzero_coeffs = cv_fit$nzero[best_lambda_idx] 
  )
}