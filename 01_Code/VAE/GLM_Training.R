GLM_Training <- function(Data_Train_CV_Vector,
                         Train_Data,
                         n_init_points = 10,
                         n_iter_bayes = 20,
                         ...) {
  
##==============================##
## 1. Data preparation.
##==============================##
  
sparse_formula <- as.formula("y ~ . - 1")
  
# Training Data
train_y <- as.numeric(as.character(Train_Data$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Data)
  
time_GLM_bo <- system.time({
    
##==============================##
## 2. Define Objective Function (LOCALLY)
##==============================##
    
current_bayes_iter <- 0
total_bayes_runs   <- n_init_points + n_iter_bayes
    
GLM_bayesoptim <- function(alpha) {
      
      current_bayes_iter <<- current_bayes_iter + 1
      message(sprintf("[%02d/%02d] BayesOpt GLM: alpha=%.4f", 
                      current_bayes_iter, total_bayes_runs, alpha))
      
      # cv.glmnet handles cross-validation internally
      cv_fit <- cv.glmnet(
        x = train_matrix, 
        y = train_y,
        family = "binomial",        
        type.measure = "auc",      
        alpha = alpha,
        foldid = Data_Train_CV_Vector,   
        standardize = TRUE        
      )
      
      best_auc <- max(cv_fit$cvm) 
      list(Score = best_auc, Pred = 0)
    }
    
##==============================##
## 3. Run Optimization
##==============================##
    
# Define bounds for alpha (0=Ridge, 1=Lasso)
bounds_glm <- list(alpha = c(0, 1))
    
print("Starting Bayesian Optimization for GLM...")
    
bayes_out_glm <- tryCatch({
      BayesianOptimization(
        FUN = GLM_bayesoptim,
        bounds = bounds_glm,
        init_points = n_init_points,
        n_iter = n_iter_bayes,
        acq = "ei",    
        eps = 0.0,     
        verbose = TRUE
      )
    }, error = function(e) {
      message("Optimization failed: ", e)
      return(NULL)
    })
    
if(is.null(bayes_out_glm)) return(NULL)
    
## Format Results
results_bayes_glm <- bayes_out_glm$History %>%
      rename(AUC = Value) %>%
      mutate(iteration_index = 1:n()) %>%
      arrange(desc(AUC)) %>%
      as_tibble()
    
    print("--- Top 5 Bayesian GLM Models ---")
    print(head(results_bayes_glm, 5))
    
##==============================##
## 4. Train Final Model
##==============================##
    
best_bayes_glm <- results_bayes_glm[1, ]
optimal_alpha <- best_bayes_glm$alpha
    
message(paste("Training Final GLM Model with alpha =", round(optimal_alpha, 4)))
    
# Train the final model on the full dataset
    final_cv_fit <- cv.glmnet(
      x = train_matrix, 
      y = train_y,
      family = "binomial",
      type.measure = "auc",
      alpha = optimal_alpha,
      foldid = Data_Train_CV_Vector,
      standardize = TRUE
    )
    
    optimal_lambda <- final_cv_fit$lambda.min
    model_bayes_glm <- final_cv_fit
    
##==============================##
## 5. Predictions & Brier Scores
##==============================##
    
# Generate in-sample predictions on the Training Data
# type = "response" gives probabilities for binomial family
preds_prob <- predict(model_bayes_glm, newx = train_matrix, type = "response", s = optimal_lambda)
    
# Create Results Dataframe
df_results <- tibble::tibble(
      Actual = train_y, 
      Predicted = as.numeric(preds_prob) # Flatten matrix to vector
    )
    
# --- Standard Brier Score ---
# Formula: Mean Squared Error of Probabilities
brier_score <- mean((df_results$Predicted - df_results$Actual)^2)
    
# --- Penalized Brier Score (PBS) ---
# Reference: Ahmadian et al. (2024). Penalized Brier Score.
# For binary classification (R=2), the penalty is (R-1)/R = 1/2 = 0.5.
# The penalty is applied when the predicted class (argmax p) does not match the true class.
# In binary terms, this is when round(Predicted) != Actual.
    
R <- 2 # Number of classes (Binary: 0 or 1)
penalty_term <- (R - 1) / R # Penalty is 0.5
    
df_results <- df_results %>%
      mutate(
        # Get the predicted class (0 or 1) using a 0.5 threshold
        Predicted_Class = round(Predicted),
        # Calculate the standard squared error component
        Squared_Error = (Predicted - Actual)^2,
        # Apply the penalty if the prediction is incorrect
        # The penalty term is added when (argmax p != arg max y)
        Penalty = ifelse(Predicted_Class != Actual, penalty_term, 0)
      )
    
    # The PBS is the mean of the squared error plus the penalty term
    # PBS = (1/N) * Sum(Squared_Error + Penalty)
    penalized_brier_score <- mean(df_results$Squared_Error + df_results$Penalty)
    
  }) # End system.time
  
  ### Results
  Results_GLM <- list(
    results = results_bayes_glm,
    optimal_model = model_bayes_glm,
    optimal_parameters = list(alpha = optimal_alpha, lambda = optimal_lambda),
    time = time_GLM_bo,
    Predictions = df_results,  
    Brier_Score = brier_score,
    Penalized_Brier_Score = penalized_brier_score
  )
  
##==============================##
## 6. Return Statement
##==============================##
  
  return(Results_GLM)
}