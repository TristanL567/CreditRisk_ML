XGBoost_Test_revised <- function(Model, Test_Data, target_var = "y") {
  
  tryCatch({
    
    ##==============================##
    ## 1. Data Preparation
    ##==============================##
    
    sparse_formula <- as.formula(paste(target_var, "~ . - 1"))
    test_y         <- as.numeric(as.character(Test_Data[[target_var]]))
    
    test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Data)
    
    ## ── Column Alignment Safety ─────────────────────────────────────────────
    ## sparse.model.matrix builds columns from factor levels in Test_Data.
    ## If Test has different levels than Train (e.g., missing sector), the
    ## column set will differ from what the model expects → silent misalignment.
    ## Fix: enforce the exact column set and order from training.
    
    train_features <- Model$feature_names
    
    if (!is.null(train_features) && !identical(colnames(test_matrix), train_features)) {
      
      ## Add missing columns as zeros
      missing_cols <- setdiff(train_features, colnames(test_matrix))
      if (length(missing_cols) > 0) {
        message(paste("  Adding", length(missing_cols),
                      "missing columns to Test matrix (zero-filled):",
                      paste(head(missing_cols, 5), collapse = ", "),
                      if (length(missing_cols) > 5) "..." else ""))
        zero_mat <- Matrix::sparseMatrix(
          i    = integer(0),
          j    = integer(0),
          dims = c(nrow(test_matrix), length(missing_cols)),
          dimnames = list(NULL, missing_cols)
        )
        test_matrix <- cbind(test_matrix, zero_mat)
      }
      
      ## Drop extra columns not seen during training
      extra_cols <- setdiff(colnames(test_matrix), train_features)
      if (length(extra_cols) > 0) {
        message(paste("  Dropping", length(extra_cols),
                      "extra columns from Test matrix not in Train:",
                      paste(head(extra_cols, 5), collapse = ", "),
                      if (length(extra_cols) > 5) "..." else ""))
        test_matrix <- test_matrix[, !colnames(test_matrix) %in% extra_cols, drop = FALSE]
      }
      
      ## Reorder to match training column order
      test_matrix <- test_matrix[, train_features, drop = FALSE]
    }
    
    dtest <- xgb.DMatrix(data = test_matrix, label = test_y)
    
    ##==============================##
    ## 2. Predictions
    ##==============================##
    
    timer <- system.time({
      pred_probs <- predict(Model, dtest)
    })
    
    ##==============================##
    ## 3. Compute Model Scores
    ##==============================##
    
    ## ── AUC ─────────────────────────────────────────────────────────────────
    roc_obj <- pROC::roc(test_y, pred_probs, quiet = TRUE)
    auc_val <- as.numeric(pROC::auc(roc_obj))
    
    ## ── Brier Score ─────────────────────────────────────────────────────────
    brier_val <- mean((pred_probs - test_y)^2)
    
    ## ── Log Loss ────────────────────────────────────────────────────────────
    epsilon      <- 1e-15
    preds_clipped <- pmax(pmin(pred_probs, 1 - epsilon), epsilon)
    log_loss     <- -mean(test_y * log(preds_clipped) +
                            (1 - test_y) * log(1 - preds_clipped))
    
    ## ── Youden-Optimal Threshold ────────────────────────────────────────────
    ## With ~0.8% default rate, a fixed 0.5 threshold classifies nearly
    ## everything as non-default. The Youden index (max sensitivity + specificity - 1)
    ## finds the threshold that best separates the two classes.
    
    optimal_coords <- pROC::coords(roc_obj, "best",
                                   ret       = c("threshold", "sensitivity", "specificity"),
                                   best.method = "youden")
    optimal_threshold <- optimal_coords$threshold
    
    message(sprintf("  Youden-optimal threshold: %.4f (Sens: %.3f | Spec: %.3f)",
                    optimal_threshold,
                    optimal_coords$sensitivity,
                    optimal_coords$specificity))
    
    ## ── Classification Metrics at Optimal Threshold ─────────────────────────
    predicted_class <- as.integer(pred_probs >= optimal_threshold)
    
    confusion <- table(
      Actual    = factor(test_y, levels = c(0, 1)),
      Predicted = factor(predicted_class, levels = c(0, 1))
    )
    
    tp <- confusion["1", "1"]
    tn <- confusion["0", "0"]
    fp <- confusion["0", "1"]
    fn <- confusion["1", "0"]
    
    precision_val <- tp / max(tp + fp, 1)
    recall_val    <- tp / max(tp + fn, 1)
    f1_val        <- ifelse(precision_val + recall_val > 0,
                            2 * precision_val * recall_val / (precision_val + recall_val),
                            0)
    
    ## ── Penalized Brier Score (at Optimal Threshold) ────────────────────────
    R <- 2  ## Binary classification
    penalty_term <- (R - 1) / R  ## 0.5
    
    squared_error <- (pred_probs - test_y)^2
    penalty       <- ifelse(predicted_class != test_y, penalty_term, 0)
    penalized_brier_val <- mean(squared_error + penalty)
    
    ##==============================##
    ## 4. Summary Output
    ##==============================##
    
    message("--- Test Set Evaluation Complete ---")
    message(sprintf("  AUC          : %.5f", auc_val))
    message(sprintf("  Brier Score  : %.5f", brier_val))
    message(sprintf("  Log Loss     : %.5f", log_loss))
    message(sprintf("  Pen. Brier   : %.5f (threshold: %.4f)", penalized_brier_val, optimal_threshold))
    message(sprintf("  Precision    : %.4f", precision_val))
    message(sprintf("  Recall       : %.4f", recall_val))
    message(sprintf("  F1           : %.4f", f1_val))
    
    ##==============================##
    ## 5. Return
    ##==============================##
    
    return(list(
      ## Predictions
      Predictions     = pred_probs,
      Actuals         = test_y,
      Predicted_Class = predicted_class,
      
      ## Core metrics
      Metrics = data.frame(
        AUC                 = auc_val,
        Brier_Score         = brier_val,
        Penalized_Brier     = penalized_brier_val,
        Log_Loss            = log_loss,
        Inference_Time_Sec  = timer["elapsed"]
      ),
      
      ## Classification metrics at optimal threshold
      Classification = data.frame(
        Threshold  = optimal_threshold,
        Sensitivity = optimal_coords$sensitivity,
        Specificity = optimal_coords$specificity,
        Precision  = precision_val,
        Recall     = recall_val,
        F1_Score   = f1_val,
        TP = tp, TN = tn, FP = fp, FN = fn
      ),
      
      ## ROC object for downstream plotting
      ROC_Object = roc_obj,
      
      ## Confusion matrix
      Confusion_Matrix = confusion
    ))
    
  }, error = function(e) {
    message("Error during evaluation: ", e$message)
    return(NULL)
  })
}