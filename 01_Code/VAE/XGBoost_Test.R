XGBoost_Test <- function(Model, Test_Data, target_var = "y") {
  
  tryCatch({
    
##==============================##
## 1. Data preparation.
##==============================##
    
sparse_formula <- as.formula(paste(target_var, "~ . - 1"))
test_y <- as.numeric(as.character(Test_Data[[target_var]]))
    
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Data)
dtest <- xgb.DMatrix(data = test_matrix, label = test_y)
    
##==============================##
## 2. Get the predictions for the test set data.
##==============================##
  
timer <- system.time({
      pred_probs <- predict(Model, dtest)
    })
    
##==============================##
## 3. Compute the model scores.
##==============================##

roc_obj  <- pROC::roc(test_y, pred_probs, quiet = TRUE)
auc_val  <- as.numeric(pROC::auc(roc_obj))
    
brier_val <- mean((pred_probs - test_y)^2)

epsilon <- 1e-15
preds_clipped <- pmax(pmin(pred_probs, 1 - epsilon), epsilon)
log_loss <- -mean(test_y * log(preds_clipped) + (1 - test_y) * log(1 - preds_clipped))
    
message(paste("Test Set Evaluation Complete."))
message(paste("AUC:       ", round(auc_val, 5)))
message(paste("Brier:     ", round(brier_val, 5)))
 
##==============================##
## 4. Return statement.
##==============================##

    return(list(
      Predictions = pred_probs,
      Actuals = test_y,
      Metrics = data.frame(
        AUC = auc_val,
        Brier_Score = brier_val,
        Log_Loss = log_loss,
        Inference_Time_Sec = timer["elapsed"]
      )
    ))
    
  }, error = function(e) {
    message("Error during evaluation: ", e)
    return(NULL)
  })
}