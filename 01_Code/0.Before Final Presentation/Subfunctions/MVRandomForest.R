
packages <- c(
  "dplyr", "tidyr",
  "randomForest", "pROC", "tibble"
)

for(i in seq_along(packages)){
  pkg <- packages[i]
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    cat("Installed:", pkg, "\n")
  }
  library(pkg, character.only = TRUE)
}



MVRandomForest <- function(Train,
                           Test,
                           target         = "y",
                           id_var         = "id",
                           positive_class = "1",     # default class = "1" (default)
                           ntree          = 500,
                           mtry           = NULL,    # default: floor(sqrt(p))
                           nodesize       = 1,
                           seed           = 123){
  
set.seed(seed)

#Leakage check
if (id_var %in% names(Train) && id_var %in% names(Test)) {
  overlap_ids <- intersect(unique(Train[[id_var]]), unique(Test[[id_var]]))
  if (length(overlap_ids) > 0) {
    stop("Data leakage detected: some firm IDs appear in both Train and Test.")
  }
}

  
#Prepare data  
predictors <- setdiff(names(Train), c(target, id_var))
p <- length(predictors)
  
Train_rf <- Train %>%
    dplyr::select(dplyr::all_of(c(target, predictors))) %>%
    tidyr::drop_na() %>%
    dplyr::mutate(!!target := factor(.data[[target]]))
  
Test_rf <- Test %>%
    dplyr::select(dplyr::all_of(c(target, predictors))) %>%
    tidyr::drop_na() %>%
    dplyr::mutate(!!target := factor(.data[[target]], levels = levels(Train_rf[[target]])))
  
if (!positive_class %in% levels(Train_rf[[target]])) {
    stop(paste0("positive_class = '", positive_class, "' not found in levels: ",
                paste(levels(Train_rf[[target]]), collapse = ", ")))
  }
  
if (is.null(mtry)) mtry <- floor(sqrt(p))
  
#Fit random forest
rf_fit <- randomForest::randomForest(
    formula    = stats::as.formula(paste(target, "~ .")),
    data       = Train_rf,
    ntree      = ntree,
    mtry       = mtry,
    nodesize   = nodesize,
    importance = TRUE
  )
  

#Prediction, threshold 0.5 is used
prob_mat  <- predict(rf_fit, newdata = Test_rf, type = "prob")
prob_pos  <- prob_mat[, positive_class]
  
negative_class <- setdiff(levels(Train_rf[[target]]), positive_class)[1]
pred_class <- ifelse(prob_pos >= 0.5, positive_class, negative_class) %>%
factor(levels = levels(Train_rf[[target]]))
  
#AUC and confusion matrix
conf_mat <- table(Actual = Test_rf[[target]], Predicted = pred_class)
  
auc_val <- NA_real_
if (length(unique(Test_rf[[target]])) == 2) {
    roc_obj <- pROC::roc(
      response  = Test_rf[[target]],
      predictor = prob_pos,
      levels    = levels(Train_rf[[target]]),
      direction = "<"
    )
auc_val <- as.numeric(pROC::auc(roc_obj))
  }
  
#Estimate variable importance 
imp_mat <- randomForest::importance(rf_fit)

var_imp <- imp_mat %>%
  as.data.frame() %>%
  tibble::rownames_to_column("variable")


if ("MeanDecreaseAccuracy" %in% names(var_imp)) {
  var_imp <- var_imp %>% dplyr::arrange(dplyr::desc(MeanDecreaseAccuracy))
} else if ("MeanDecreaseGini" %in% names(var_imp)) {
  var_imp <- var_imp %>% dplyr::arrange(dplyr::desc(MeanDecreaseGini))
}


gini_imp <- NULL
if ("MeanDecreaseGini" %in% colnames(imp_mat)) {
  gini_imp <- imp_mat[, "MeanDecreaseGini"]
}

#Output
list(
  Model      = rf_fit,
  Params     = list(ntree = ntree, mtry = mtry, nodesize = nodesize, p = p,
                    positive_class = positive_class),
  Test_AUC   = auc_val,
  Confusion  = conf_mat,
  Prob_Test  = prob_pos,
  Pred_Test  = pred_class,
  VarImp     = var_imp,     
  GiniImp    = gini_imp     
)
}
