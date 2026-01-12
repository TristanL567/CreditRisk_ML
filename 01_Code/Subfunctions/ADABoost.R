packages <- c(
  "dplyr", "tidyr", "tibble",
  "pROC", "rpart",
  "adabag"  
)

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    cat("Installed:", pkg, "\n")
  }
  library(pkg, character.only = TRUE)
}


MVAdaBoost <- function(Train,
                       Test,
                       target         = "y",
                       id_var         = "id",
                       positive_class = "1",     
                       mfinal         = 100,     # number of boosting iterations
                       maxdepth       = 1,       # 1 = decision stumps
                       minsplit       = 20,
                       minbucket      = 7,
                       use_class_weights = TRUE, # helpful for imbalance
                       weight_cap        = 50,   # prevents extreme weights
                       seed           = 123) {
  
  set.seed(seed)
  
  #Check data leakage
  if (id_var %in% names(Train) && id_var %in% names(Test)) {
    overlap_ids <- intersect(unique(Train[[id_var]]), unique(Test[[id_var]]))
    if (length(overlap_ids) > 0) {
      stop("Data leakage detected: some firm IDs appear in both Train and Test.")
    }
  }
  
  # Prepare data 
  predictors <- setdiff(names(Train), c(target, id_var))
  if (length(predictors) == 0) stop("No predictors found after removing target and id_var.")
  
  Train_ab <- Train %>%
    dplyr::select(dplyr::all_of(c(target, predictors))) %>%
    tidyr::drop_na() %>%
    dplyr::mutate(!!target := factor(.data[[target]]))
  
  Test_ab <- Test %>%
    dplyr::select(dplyr::all_of(c(target, predictors))) %>%
    tidyr::drop_na() %>%
    dplyr::mutate(!!target := factor(.data[[target]], levels = levels(Train_ab[[target]])))
  
  if (!positive_class %in% levels(Train_ab[[target]])) {
    stop(paste0("positive_class = '", positive_class, "' not found in levels: ",
                paste(levels(Train_ab[[target]]), collapse = ", ")))
  }
  
  # Define negative class
  negative_class <- setdiff(levels(Train_ab[[target]]), positive_class)
  if (length(negative_class) != 1) stop("This function currently expects a binary target (two classes).")
  negative_class <- negative_class[1]
  
  # Observation weights to mitigate imbalance
  obs_w <- rep(1, nrow(Train_ab))
  
  if (use_class_weights) {
    tab <- table(Train_ab[[target]])
    # inverse-frequency weights: rarer class gets higher weight
    w_pos <- as.numeric(sum(tab) / (2 * tab[positive_class]))
    w_neg <- as.numeric(sum(tab) / (2 * tab[negative_class]))
    
    obs_w[Train_ab[[target]] == positive_class] <- w_pos
    obs_w[Train_ab[[target]] == negative_class] <- w_neg
    
    # cap to avoid extremely large weights (useful if defaults are extremely rare)
    obs_w <- pmin(obs_w, weight_cap)
  }
  
  # Fit AdaBoost
  # adabag::boosting uses rpart trees; control depth via rpart.control(maxdepth=...)
  ctrl <- rpart::rpart.control(maxdepth = maxdepth,
                               minsplit = minsplit,
                               minbucket = minbucket,
                               cp = 0)
  
  ab_fit <- adabag::boosting(
    formula = stats::as.formula(paste(target, "~ .")),
    data    = Train_ab,
    mfinal  = mfinal,
    boos    = TRUE,
    control = ctrl,
    weights = obs_w
  )
  
  # Predict probabilities on test_sample
  pred_obj <- predict(ab_fit, newdata = Test_ab)
  
  prob_mat <- pred_obj$prob
  if (!positive_class %in% colnames(prob_mat)) {
    stop("Positive class not found in predicted probability matrix columns.")
  }
  prob_pos <- prob_mat[, positive_class]
  
  # Threshold classification at 0.5 (maybe, we can later change the threshold?)
  pred_class <- ifelse(prob_pos >= 0.5, positive_class, negative_class) %>%
    factor(levels = levels(Train_ab[[target]]))
  
  # Confusion matrix
  conf_mat <- table(Actual = Test_ab[[target]], Predicted = pred_class)
  
  # AUC 
  auc_val <- NA_real_
  if (length(unique(Test_ab[[target]])) == 2) {
    roc_obj <- pROC::roc(
      response  = Test_ab[[target]],
      predictor = prob_pos,
      levels    = levels(Train_ab[[target]]),
      direction = "<"
    )
    auc_val <- as.numeric(pROC::auc(roc_obj))
  }
  
  # Variable importance
  
  var_imp <- NULL
  if (!is.null(ab_fit$importance)) {
    var_imp <- ab_fit$importance %>%
      as.data.frame() %>%
      tibble::rownames_to_column("variable") %>%
      dplyr::rename(Importance = 2) %>%
      dplyr::arrange(dplyr::desc(Importance))
  }
  
#Output
  list(
    Model      = ab_fit,
    Params     = list(
      mfinal = mfinal, maxdepth = maxdepth, minsplit = minsplit, minbucket = minbucket,
      positive_class = positive_class, use_class_weights = use_class_weights, weight_cap = weight_cap
    ),
    Test_AUC   = auc_val,
    Confusion  = conf_mat,
    Prob_Test  = prob_pos,
    Pred_Test  = pred_class,
    VarImp     = var_imp
  )
}

# ============================================================
# Example usage (fits with MVstratifiedsampling)
# ============================================================

# split_out <- MVstratifiedsampling(Data = your_data, strat_vars = c("sector","y"), Train_size = 0.7)
# Train <- split_out$Train
# Test  <- split_out$Test

# ada_res <- MVAdaBoost(
#   Train = Train,
#   Test  = Test,
#   target = "y",
#   id_var = "id",
#   positive_class = "1",
#   mfinal = 200,     # try 100–400
#   maxdepth = 1,     # stumps; try 2 if needed
#   use_class_weights = TRUE
# )

# ada_res$Test_AUC
# ada_res$Confusion
# head(ada_res$VarImp, 15)
