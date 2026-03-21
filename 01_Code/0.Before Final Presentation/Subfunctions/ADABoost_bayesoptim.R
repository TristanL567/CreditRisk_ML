ADABoost_bayesoptim <- function(mfinal, maxdepth, minsplit, minbucket) {
  
  current_bayes_iter <<- current_bayes_iter + 1
  
  current_mfinal   <- as.integer(round(mfinal))
  current_depth    <- as.integer(round(maxdepth))
  current_minsplit <- as.integer(round(minsplit))
  current_minb     <- as.integer(round(minbucket))
  
  message(sprintf("[%02d/%02d] BayesOpt: mfinal=%d | depth=%d | minsplit=%d | minbucket=%d",
                  current_bayes_iter, total_bayes_runs,
                  current_mfinal, current_depth, current_minsplit, current_minb))
  
  params <- list(
    mfinal    = current_mfinal,
    maxdepth  = current_depth,
    minsplit  = current_minsplit,
    minbucket = current_minb
  )
  
  auc_mean <- AdaBoost_cv_auc(params = params, folds_custom = cv_folds_list)
  
  list(Score = auc_mean, Pred = 0)
}
