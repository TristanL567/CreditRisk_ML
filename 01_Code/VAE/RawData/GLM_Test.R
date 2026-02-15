GLM_Test <- function(Model, Test_Data, target_var = "y") {
  
  tryCatch({
    
    ##==============================##
    ## 1. Data preparation.
    ##==============================##
    
    # GLMnet requires a matrix, just like XGBoost
    sparse_formula <- as.formula(paste(target_var, "~ . - 1"))
    test_y <- as.numeric(as.character(Test_Data[[target_var]]))
    
    # Create the sparse model matrix
    test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Data)
    
    ##==============================##
    ## 2. Get the predictions for the test set data.
    ##==============================##
    
    # For glmnet (cv.glmnet object), we need to specify 's' (lambda).
    # We use "lambda.min" which is the optimal lambda found during training.
    # If your model object is just the 'final_cv_fit' from previous steps, this works automatically.
    
    timer <- system.time({
      # type = "response" ensures we get probabilities (0-1)
      pred_probs <- predict(Model, newx = test_matrix, type = "response", s = "lambda.min")
    })
    
    # Flatten matrix to vector to ensure compatibility with calculation functions
    pred_probs <- as.numeric(pred_probs)
    
    ##==============================##
    ## 3. Compute the model scores.
    ##==============================##
    
    # --- ROC & AUC ---
    roc_obj  <- pROC::roc(test_y, pred_probs, quiet = TRUE)
    auc_val  <- as.numeric(pROC::auc(roc_obj))
    
    # --- Standard Brier Score ---
    brier_val <- mean((pred_probs - test_y)^2)
    
    # --- Log Loss ---
    epsilon <- 1e-15
    preds_clipped <- pmax(pmin(pred_probs, 1 - epsilon), epsilon)
    log_loss <- -mean(test_y * log(preds_clipped) + (1 - test_y) * log(1 - preds_clipped))
    
    # --- Penalized Brier Score (PBS) ---
    # Reference: Ahmadian et al. (2024). 
    # Formula: (1/N) * Sum( (prob - actual)^2 + Penalty )
    # Penalty applies if round(prob) != actual.
    
    R <- 2 # Binary classification
    penalty_term <- (R - 1) / R # 0.5
    
    # Create a temporary dataframe for vectorized calculation
    df_calc <- data.frame(
      Actual = test_y,
      Predicted = pred_probs
    )
    
    df_calc$Predicted_Class <- round(df_calc$Predicted)
    df_calc$Squared_Error   <- (df_calc$Predicted - df_calc$Actual)^2
    df_calc$Penalty         <- ifelse(df_calc$Predicted_Class != df_calc$Actual, penalty_term, 0)
    
    penalized_brier_val <- mean(df_calc$Squared_Error + df_calc$Penalty)
    
    ##==============================##
    ## 4. Output & Return.
    ##==============================##
    
    message(paste("Test Set Evaluation Complete (GLM)."))
    message(paste("AUC:             ", round(auc_val, 5)))
    message(paste("Brier:           ", round(brier_val, 5)))
    message(paste("Penalized Brier: ", round(penalized_brier_val, 5)))
    
    return(list(
      Predictions = pred_probs,
      Actuals = test_y,
      Metrics = data.frame(
        AUC = auc_val,
        Brier_Score = brier_val,
        Penalized_Brier_Score = penalized_brier_val,
        Log_Loss = log_loss,
        Inference_Time_Sec = timer["elapsed"]
      )
    ))
    
  }, error = function(e) {
    message("Error during GLM evaluation: ", e)
    return(NULL)
  })
}