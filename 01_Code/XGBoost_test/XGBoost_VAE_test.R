#==============================================================================#
#==== 05 - XGBoost Trees ======================================================#
#==============================================================================#

tryCatch({
  
##==============================##
## General Parameters.
##==============================##
  
n_init_points <- 2
n_iter_bayes  <- 4
total_bayes_runs <- n_init_points + n_iter_bayes
  
#==== 05A - Base model ========================================================#
    
tryCatch({
  
##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
    
# Training Data
train_y <- as.numeric(as.character(Train_Transformed$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Transformed)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)
    
# Test Data
# Test_Clean <- Test_Transformed %>% select(-id)
test_y <- as.numeric(as.character(Test_Transformed$y))
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Transformed)
dtest <- xgb.DMatrix(data = test_matrix, label = test_y)
    
##==============================##
## Bayesian Optimization for Hyperparameter Tuning - Dataset 1.
##==============================##
    
time_XGBoost_bo <- system.time({
      
## specific data
Data_Train_CV_List <- Data_Train_CV_List_1
print(paste("CV List loaded with", length(Data_Train_CV_List), "folds."))
      
## Prepare the parameters.
current_bayes_iter <- 0 # Initialize counter
      
bounds_bayes <- list(
                    eta = c(0.01, 0.3),
                    max_depth = c(3L, 8L), 
                    subsample = c(0.5, 1.0),
                    colsample_bytree = c(0.5, 1.0)
                  )
      
## Run Optimization & Format Results.

tryCatch({
current_bayes_iter <- 0 
        
bayes_out <- BayesianOptimization(
          FUN = XGBoost_bayesoptim,
          bounds = bounds_bayes,
          init_points = n_init_points,
          n_iter = n_iter_bayes,
          acq = "ucb",
          kappa = 2.576,                 
          eps = 0.0,                     
          verbose = TRUE
        )
        
## Convert History to Tibble.
results_bayes <- bayes_out$History %>%
          rename(AUC = Value) %>%
          mutate(
            max_depth = as.integer(round(max_depth)),
            nrounds = 2000,
            early_stopping_rounds = 50,
            iteration_index = 1:n()
          ) %>%
          arrange(desc(AUC)) %>%
          as_tibble()
        
print("--- Top 5 Bayesian Models ---")
print(head(results_bayes, 5))
        
}, error = function(e) message(e))
      
## Train the final model.
tryCatch({
        
best_bayes <- results_bayes[1, ]
        
final_params_bayes <- list(
          booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
          eta = best_bayes$eta, 
          max_depth = best_bayes$max_depth, 
          subsample = best_bayes$subsample, 
          colsample_bytree = best_bayes$colsample_bytree)
        
# Determine optimal rounds for the Bayesian optim.
check_cv_bayes <- xgb.cv(
          params = final_params_bayes,
          data = dtrain,
          nrounds = 2000,
          folds = Data_Train_CV_List,
          # nfold = 5,
          early_stopping_rounds = 50,
          verbose = 0,
          maximize = TRUE
        )
        
optimal_rounds_bayes <- check_cv_bayes$evaluation_log[which.max(check_cv_bayes$evaluation_log$test_auc_mean)]$iter
        
## Train the model with the optimal hyperparameters.
model_bayes <- xgb.train(
          params = final_params_bayes, 
          data = dtrain,
          nrounds = optimal_rounds_bayes, 
          verbose = 0
        )
        
}, error = function(e) message(e))
      
## Compute the training AUC.
XGBoost_bayes_probs <- predict(model_bayes, dtrain)
XGBoost_bayes_Train_ROC <- roc(train_y, XGBoost_bayes_probs, quiet = TRUE)
XGBoost_bayes_Train_AUC <- pROC::auc(XGBoost_bayes_Train_ROC)
print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))

### Results:
Results_XGBoost_Base_model <- list(results = results_bayes,
                                   optimal_model = model_bayes,
                                   optimal_rounds = optimal_rounds_bayes,
                                   optimal_parameters = final_params_bayes)
      
print(head(results_bayes, 5))

}) ## time counter.
    
}, error = function(e) message(e))
    
#==== 05B - Dataset with the untuned VAE ======================================#

tryCatch({
  
##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
Strat_Data_1 <- Strat_Data_1 %>%
  select(-id)

# Training Data
train_y <- as.numeric(as.character(Strat_Data_1$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Strat_Data_1)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)

# Test Data
test_y <- as.numeric(as.character(Test_Transformed$y))
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Transformed)
dtest <- xgb.DMatrix(data = test_matrix, label = test_y)

##==============================##
## Bayesian Optimization for Hyperparameter Tuning - Dataset 1.
##==============================##

time_XGBoost_bo <- system.time({
  
## specific data
Data_Train_CV_List <- Data_Train_CV_List_1
print(paste("CV List loaded with", length(Data_Train_CV_List), "folds."))
  
## Prepare the parameters.
current_bayes_iter <- 0 # Initialize counter
  
bounds_bayes <- list(
    eta = c(0.01, 0.3),
    max_depth = c(3L, 8L), 
    subsample = c(0.5, 1.0),
    colsample_bytree = c(0.5, 1.0)
  )
  
  ## Run Optimization & Format Results.
  
  tryCatch({
    current_bayes_iter <- 0 
    
    bayes_out <- BayesianOptimization(
      FUN = XGBoost_bayesoptim,
      bounds = bounds_bayes,
      init_points = n_init_points,
      n_iter = n_iter_bayes,
      acq = "ucb",
      kappa = 2.576,                 
      eps = 0.0,                     
      verbose = TRUE
    )
    
    ## Convert History to Tibble.
    results_bayes <- bayes_out$History %>%
      rename(AUC = Value) %>%
      mutate(
        max_depth = as.integer(round(max_depth)),
        nrounds = 2000,
        early_stopping_rounds = 50,
        iteration_index = 1:n()
      ) %>%
      arrange(desc(AUC)) %>%
      as_tibble()
    
    print("--- Top 5 Bayesian Models ---")
    print(head(results_bayes, 5))
    
  }, error = function(e) message(e))
  
  ## Train the final model.
  tryCatch({
    
    best_bayes <- results_bayes[1, ]
    
    final_params_bayes <- list(
      booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
      eta = best_bayes$eta, 
      max_depth = best_bayes$max_depth, 
      subsample = best_bayes$subsample, 
      colsample_bytree = best_bayes$colsample_bytree)
    
    # Determine optimal rounds for the Bayesian optim.
    check_cv_bayes <- xgb.cv(
      params = final_params_bayes,
      data = dtrain,
      nrounds = 2000,
      folds = Data_Train_CV_List,
      # nfold = 5,
      early_stopping_rounds = 50,
      verbose = 0,
      maximize = TRUE
    )
    
    optimal_rounds_bayes <- check_cv_bayes$evaluation_log[which.max(check_cv_bayes$evaluation_log$test_auc_mean)]$iter
    
    ## Train the model with the optimal hyperparameters.
    model_bayes <- xgb.train(
      params = final_params_bayes, 
      data = dtrain,
      nrounds = optimal_rounds_bayes, 
      verbose = 0
    )
    
  }, error = function(e) message(e))
  
  ## Compute the training AUC.
  XGBoost_bayes_probs <- predict(model_bayes, dtrain)
  XGBoost_bayes_Train_ROC <- roc(train_y, XGBoost_bayes_probs, quiet = TRUE)
  XGBoost_bayes_Train_AUC <- pROC::auc(XGBoost_bayes_Train_ROC)
  print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))
  
### Results:
Results_XGBoost_untuned_VAE <- list(results = results_bayes,
                                    optimal_model = model_bayes,
                                    optimal_rounds = optimal_rounds_bayes,
                                    optimal_parameters = final_params_bayes)
  
print(head(results_bayes, 5))
  
}) ## time counter.

}, error = function(e) message(e))
    
#==== 05C - Dataset with the revised VAE ======================================#

tryCatch({
  
##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
Strat_Data_2 <- Strat_Data_2 %>%
  select(-id)

# Training Data
train_y <- as.numeric(as.character(Strat_Data_2$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Strat_Data_2)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)

# Test Data
test_y <- as.numeric(as.character(Test_Transformed$y))
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Transformed)
dtest <- xgb.DMatrix(data = test_matrix, label = test_y)

##==============================##
## Bayesian Optimization for Hyperparameter Tuning - Dataset 1.
##==============================##

time_XGBoost_bo <- system.time({
  
## specific data
Data_Train_CV_List <- Data_Train_CV_List_1
print(paste("CV List loaded with", length(Data_Train_CV_List), "folds."))
  
## Prepare the parameters.
current_bayes_iter <- 0 # Initialize counter
  
bounds_bayes <- list(
    eta = c(0.01, 0.3),
    max_depth = c(3L, 8L), 
    subsample = c(0.5, 1.0),
    colsample_bytree = c(0.5, 1.0))
  
## Run Optimization & Format Results.
  
tryCatch({
    current_bayes_iter <- 0 
    
    bayes_out <- BayesianOptimization(
      FUN = XGBoost_bayesoptim,
      bounds = bounds_bayes,
      init_points = n_init_points,
      n_iter = n_iter_bayes,
      acq = "ucb",
      kappa = 2.576,                 
      eps = 0.0,                     
      verbose = TRUE
    )
    
## Convert History to Tibble.
results_bayes <- bayes_out$History %>%
      rename(AUC = Value) %>%
      mutate(
        max_depth = as.integer(round(max_depth)),
        nrounds = 2000,
        early_stopping_rounds = 50,
        iteration_index = 1:n()
      ) %>%
      arrange(desc(AUC)) %>%
      as_tibble()
    
    print("--- Top 5 Bayesian Models ---")
    print(head(results_bayes, 5))
    
  }, error = function(e) message(e))
  
## Train the final model.
tryCatch({
    
    best_bayes <- results_bayes[1, ]
    
    final_params_bayes <- list(
      booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
      eta = best_bayes$eta, 
      max_depth = best_bayes$max_depth, 
      subsample = best_bayes$subsample, 
      colsample_bytree = best_bayes$colsample_bytree)
    
    # Determine optimal rounds for the Bayesian optim.
    check_cv_bayes <- xgb.cv(
      params = final_params_bayes,
      data = dtrain,
      nrounds = 2000,
      folds = Data_Train_CV_List,
      # nfold = 5,
      early_stopping_rounds = 50,
      verbose = 0,
      maximize = TRUE
    )
    
    optimal_rounds_bayes <- check_cv_bayes$evaluation_log[which.max(check_cv_bayes$evaluation_log$test_auc_mean)]$iter
    
    ## Train the model with the optimal hyperparameters.
    model_bayes <- xgb.train(
      params = final_params_bayes, 
      data = dtrain,
      nrounds = optimal_rounds_bayes, 
      verbose = 0
    )
    
  }, error = function(e) message(e))
  
  ## Compute the training AUC.
  XGBoost_bayes_probs <- predict(model_bayes, dtrain)
  XGBoost_bayes_Train_ROC <- roc(train_y, XGBoost_bayes_probs, quiet = TRUE)
  XGBoost_bayes_Train_AUC <- pROC::auc(XGBoost_bayes_Train_ROC)
  print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))
  
### Results:
Results_XGBoost_tuned_VAE <- list(results = results_bayes,
                                  optimal_model = model_bayes,
                                  optimal_rounds = optimal_rounds_bayes,
                                  optimal_parameters = final_params_bayes)
  
print(head(results_bayes, 5))

}) ## time counter.

}, error = function(e) message(e))
  
#==== 05D - Continue with the best model ======================================#

Results_XGBoost_Base_model
Results_XGBoost_untuned_VAE
Results_XGBoost_tuned_VAE

tryCatch({
  
##==============================##
## Comparison of the models with each dataset.
##==============================##
  
  
  
  
  
  
##==============================##
## Performance of the model with the highest training AUC in the test set.
##==============================##
    
## Check for the right tree size.
cv_results_train <- xgb.cv(
      params = final_params_bayes,
      data = dtrain,
      folds = Data_Train_CV_List,
      nrounds = 3000,
      early_stopping_rounds = 50,
      verbose = 0,
      maximize = TRUE)
    
## Extract the data. We compare the performance of 1-SE with the full fit.
optimal_rounds_final <- cv_results_train$evaluation_log[which.max(cv_results_train$evaluation_log$test_auc_mean)]$iter
eval_log <- cv_results_train$evaluation_log
    
## Calculate 1-SE Threshold
best_iter_index <- which.max(eval_log$test_auc_mean)
best_auc_mean   <- eval_log$test_auc_mean[best_iter_index]
best_auc_std    <- eval_log$test_auc_std[best_iter_index]
threshold_auc <- best_auc_mean - best_auc_std
    
candidates <- eval_log %>% filter(test_auc_mean >= threshold_auc)
optimal_rounds_1se <- min(candidates$iter)
    
message(sprintf("Absolute Best Rounds: %d (AUC: %.5f)", best_iter_index, best_auc_mean))
message(sprintf("1-SE Rule Rounds:     %d (Threshold: %.5f)", optimal_rounds_1se, threshold_auc))
    
## Train the 1-SE Model
XGBoost_finalmodel_1SE <- xgb.train(
      params = final_params_bayes,
      data = dtrain,
      nrounds = optimal_rounds_1se,
      verbose = 0)
    
## Train the full model.
XGBoost_finalmodel <- xgb.train(
      params = final_params_bayes,
      data = dtrain,
      nrounds = optimal_rounds_final,
      verbose = 0)
    
## Compare the AUC score of the test set and compare 1-SE with the full iteration model.
## Full:
XGBoost_test_probs <- predict(XGBoost_finalmodel, dtest)
XGBoost_test_ROC <- roc(test_y, XGBoost_test_probs, quiet = TRUE)
XGBoost_test_AUC <- pROC::auc(XGBoost_test_ROC)
print(paste("Final Test Set AUC:", round(XGBoost_test_AUC, 5)))
    
## 1-SE:
XGBoost_test_probs_1SE <- predict(XGBoost_finalmodel_1SE, dtest)
XGBoost_test_ROC_1SE   <- roc(test_y, XGBoost_test_probs_1SE, quiet = TRUE)
XGBoost_test_AUC_1SE   <- pROC::auc(XGBoost_test_ROC_1SE)
print(paste("Final Test AUC (1-SE Rule):", round(XGBoost_test_AUC_1SE, 5)))
    
## Brier Score of the full model.
actuals_num <- as.numeric(as.character(test_y))
XGBoost_BrierScore <- BrierScore(XGBoost_test_probs, actuals_num)
XGBoost_BrierScore_1SE <- BrierScore(XGBoost_test_probs_1SE, actuals_num)
    
##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##
    
## CV-AUC:
print(head(results_ds, 5))
print(head(results_rs, 5))
print(head(results_bayes, 5))
    
    # Extract the best Cross-Validation AUC from each results table
    best_cv_ds    <- max(results_ds$AUC)     
    best_cv_rs    <- max(results_rs$AUC)    
    best_cv_bayes <- max(results_bayes$AUC)  
    
    XGBoost_CV_performance <- data.frame(
      Method = c("Grid Search", "Random Search", "Bayesian Opt"),
      CV_AUC = c(best_cv_ds, best_cv_rs, best_cv_bayes)
    )
    
    ## Data summary and consolidation.
    results_ds <- results_ds %>% mutate(Method = "Grid Search", Iteration = 1:n())
    results_rs <- results_rs %>% mutate(Method = "Random Search", Iteration = 1:n())
    results_bayes <- results_bayes %>% mutate(Method = "Bayesian Opt", Iteration = 1:n())
    
    common_cols <- c("Method", "Iteration", "eta", "max_depth", "subsample", "colsample_bytree", "AUC")
    all_search_results <- bind_rows(
      results_ds %>% select(all_of(common_cols)),
      results_rs %>% select(all_of(common_cols)),
      results_bayes %>% select(all_of(common_cols))
    )
    
    XGBoost_method_performance <- data.frame(
      Method = c("Grid Search", "Random Search", "Bayesian Opt"),
      Train_AUC = c(XGBoost_ds_Train_AUC, XGBoost_rs_Train_AUC, XGBoost_bayes_Train_AUC)
    )
  
}, error = function(e) message(e))
  
#==============================================================================#

}, error = function(e) message(e))
