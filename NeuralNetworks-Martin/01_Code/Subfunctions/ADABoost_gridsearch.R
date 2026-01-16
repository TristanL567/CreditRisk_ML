ADABoost_gridsearch <- function(mfinal, maxdepth, minsplit, minbucket,
                                nrounds, early_stopping_rounds,
                                current_iter, total_iters,
                                folds_custom) {
  
  message(sprintf("[%02d/%02d] CV: mfinal=%d | depth=%d | minsplit=%d | minbucket=%d",
                  current_iter, total_iters,
                  mfinal, maxdepth, minsplit, minbucket))
  
  params <- list(
    mfinal    = mfinal,
    maxdepth  = maxdepth,
    minsplit  = minsplit,
    minbucket = minbucket
  )
  
  auc_mean <- AdaBoost_cv_auc(params = params, folds_custom = folds_custom)
  
  tibble::tibble(
    AUC = auc_mean,
    Best_Rounds = NA_integer_,  
    mfinal = mfinal,
    maxdepth = maxdepth,
    minsplit = minsplit,
    minbucket = minbucket,
    nrounds = nrounds,
    early_stopping_rounds = early_stopping_rounds
  )
}

