library(pROC)

extract_test_metrics <- function(test_obj, test_df, train_obj, model_name) {
  

  if (is.null(test_obj) || is.null(train_obj) || is.null(train_obj$results)) {
    warning(paste("Model", model_name, "test output missing. Returning NA row."))
    return(data.frame(
      Model = model_name,
      Type  = "Error",
      AUC   = NA,
      Brier_Score = NA,
      Penalized_Brier = NA,
      Alpha = NA,
      Lambda = NA
    ))
  }
  

  params <- train_obj$optimal_parameters
  alpha <- NA; lambda <- NA; model_type <- "Unknown"
  if (!is.null(params) && "eta" %in% names(params)) {
    model_type <- "XGBoost"
  } else if (!is.null(params) && "alpha" %in% names(params)) {
    model_type <- "GLM"
    alpha  <- params$alpha
    lambda <- params$lambda
  }
  

  auc_val <- if (!is.null(test_obj$AUC)) as.numeric(test_obj$AUC)[1] else NA
  brier_val <- if (!is.null(test_obj$Brier_Score)) as.numeric(test_obj$Brier_Score)[1] else NA
  penalized_brier <- if (!is.null(test_obj$Penalized_Brier_Score)) as.numeric(test_obj$Penalized_Brier_Score)[1] else NA
  

  if (is.na(auc_val) || is.na(brier_val) || is.na(penalized_brier)) {
    

    p_hat <- NULL
    if (!is.null(test_obj$pred))  p_hat <- test_obj$pred
    if (!is.null(test_obj$prob))  p_hat <- test_obj$prob
    if (!is.null(test_obj$probs)) p_hat <- test_obj$probs
    if (!is.null(test_obj$p_hat)) p_hat <- test_obj$p_hat
    if (!is.null(test_obj$Predictions)) p_hat <- test_obj$Predictions
    

    if (is.matrix(p_hat)) {
     
      if (ncol(p_hat) >= 2) p_hat <- p_hat[, 2]
      else p_hat <- p_hat[, 1]
    }
    if (is.data.frame(p_hat)) {
      pcol <- intersect(c("p", "prob", "proba", "p_hat", "phat", "pred_prob",
                          "Prediction", "Predicted_Prob", "yhat", "p1", "class1"),
                        names(p_hat))
      if (length(pcol) > 0) {
        p_hat <- p_hat[[pcol[1]]]
      } else {
        # fallback: take last numeric column
        num_cols <- names(p_hat)[vapply(p_hat, is.numeric, logical(1))]
        if (length(num_cols) > 0) p_hat <- p_hat[[num_cols[length(num_cols)]]]
      }
    }
    
    if (is.null(p_hat)) stop(paste("No predicted probabilities found for", model_name))
    

    if (!("y" %in% names(test_df))) stop("test_df must contain column 'y'")
    y_true <- test_df$y
    

    if (is.factor(y_true)) y_true <- as.integer(y_true == levels(y_true)[2])
    y_true <- as.numeric(y_true)
    
    if (is.factor(p_hat)) p_hat <- as.character(p_hat)
    p_hat <- as.numeric(p_hat)
    
  
    ok <- is.finite(y_true) & is.finite(p_hat)
    y_true <- y_true[ok]
    p_hat  <- p_hat[ok]
    

    n <- min(length(y_true), length(p_hat))
    y_true <- y_true[seq_len(n)]
    p_hat  <- p_hat[seq_len(n)]
    
    # compute missing metrics
    if (is.na(auc_val)) {
      auc_val <- as.numeric(pROC::auc(pROC::roc(y_true, p_hat, quiet = TRUE)))
    }
    if (is.na(brier_val)) {
      brier_val <- mean((p_hat - y_true)^2)
    }
    if (is.na(penalized_brier)) {
      penalized_brier <- brier_val
    }
  }
  
  # return EXACT same structure as train leaderboard selection expects
  data.frame(
    Model = model_name,
    Type  = model_type,
    AUC   = auc_val,
    Brier_Score = brier_val,
    Penalized_Brier = penalized_brier,
    Alpha  = alpha,
    Lambda = lambda
  )
}
