AdaBoost_cv_auc <- function(params, folds_custom) {
  
  dat <- Train_Transformed
  dat[[target]] <- factor(dat[[target]])
  
  auc_vec <- numeric(length(folds_custom))
  
  for (k in seq_along(folds_custom)) {
    
    test_idx  <- folds_custom[[k]]
    train_idx <- setdiff(seq_len(nrow(dat)), test_idx)
    
    tr <- dat[train_idx, , drop = FALSE]
    te <- dat[test_idx,  , drop = FALSE]
    te[[target]] <- factor(te[[target]], levels = levels(tr[[target]]))
    
    # inverse-frequency weights (stable + fast)
    tab <- table(tr[[target]])
    if (length(tab) < 2 || !positive_class %in% names(tab)) {
      auc_vec[k] <- NA_real_
      next
    }
    neg_class <- setdiff(names(tab), positive_class)[1]
    w_pos <- as.numeric(sum(tab) / (2 * tab[positive_class]))
    w_neg <- as.numeric(sum(tab) / (2 * tab[neg_class]))
    w <- ifelse(tr[[target]] == positive_class, w_pos, w_neg)
    w <- pmin(w, 50)  # weight cap (like your methodology idea)
    
    ctrl <- rpart::rpart.control(
      maxdepth  = as.integer(params$maxdepth),
      minsplit  = as.integer(params$minsplit),
      minbucket = as.integer(params$minbucket),
      cp = 0
    )
    
    fit <- adabag::boosting(
      y ~ .,
      data    = tr,
      mfinal  = as.integer(params$mfinal),
      boos    = TRUE,
      control = ctrl,
      weights = w
    )
    
    pr <- predict(fit, newdata = te)
    prob_pos <- pr$prob[, positive_class]
    
    roc_obj <- pROC::roc(
      response  = te[[target]],
      predictor = prob_pos,
      levels    = levels(tr[[target]]),
      direction = "<",
      quiet     = TRUE
    )
    auc_vec[k] <- as.numeric(pROC::auc(roc_obj))
  }
  
  mean(auc_vec, na.rm = TRUE)
}
