#==============================================================================#
#==== 05 - Ensemble Construction (Corrected & Complete) =======================#
#==============================================================================#

library(dplyr)
library(xgboost)
library(Matrix)
library(pROC)
library(corrplot)
library(tibble)
library(RColorBrewer)

# 1. Setup & Load Data --------------------------------------------------------
# Ensure these objects exist in your environment from previous steps.
# If starting fresh, uncomment and adjust the path below:
# xgb_objects <- readRDS("path/to/xgb_objects.RData")

# Define the models
Model_List <- list(
  Base      = XGBoost_Results_BaseModel$optimal_model,
  StrategyA = XGBoost_Results_Strategy_A$optimal_model,
  StrategyB = XGBoost_Results_Strategy_B$optimal_model,
  StrategyC = XGBoost_Results_Strategy_C$optimal_model
)

# Define the corresponding Test Dataframes
# CRITICAL: These must match the order of Model_List
Test_Data_List <- list(
  Base      = Test_Data_Base_Model,
  StrategyA = Test_Data_Strategy_A,
  StrategyB = Test_Data_Strategy_B,
  StrategyC = Test_Data_Strategy_C
)

# Extract Actuals (y) from the Base test set
# We assume all test sets share the same target 'y'
test_y_raw <- Test_Data_Base_Model$y
actuals <- as.numeric(as.character(test_y_raw))


# 2. Validation Checks --------------------------------------------------------
# Verify all test sets have the exact same number of rows as the target
expected_rows <- length(actuals)
dims_check <- sapply(Test_Data_List, nrow)

if(any(dims_check != expected_rows)) {
  mismatched <- names(dims_check)[dims_check != expected_rows]
  stop(paste("CRITICAL ERROR: Row count mismatch in:", paste(mismatched, collapse=", "), 
             ". All test sets must align exactly with 'test_y_raw'." ))
}


# 3. Robust Prediction Function -----------------------------------------------
get_xgb_pred <- function(model, test_data) {
  
  require(Matrix)
  require(xgboost)
  
  # A. Clean Metadata
  cols_to_drop <- c("id", "refdate", "time_index", "year", "row_id", "company_id")
  test_clean <- test_data[, !names(test_data) %in% cols_to_drop]
  
  # B. Safety: Ensure 'y' exists for the formula
  # If 'y' is missing (unlabeled test data), use dummy 0 to satisfy syntax
  if(!"y" %in% colnames(test_clean)) {
    test_clean$y <- 0 
  }
  
  # C. Create Sparse Matrix
  # This handles Factor -> Dummy conversion automatically
  sparse_formula <- as.formula("y ~ . - 1")
  test_matrix <- sparse.model.matrix(sparse_formula, data = test_clean, na.action = "na.pass")
  
  # D. Alignment Logic (Match Columns to Training Data)
  train_cols <- model$feature_names
  
  if(!is.null(train_cols)) {
    # 1. Add Missing Columns (Fill with 0)
    missing_cols <- setdiff(train_cols, colnames(test_matrix))
    if(length(missing_cols) > 0) {
      new_cols <- Matrix(0, nrow = nrow(test_matrix), ncol = length(missing_cols), sparse = TRUE)
      colnames(new_cols) <- missing_cols
      test_matrix <- cbind(test_matrix, new_cols)
    }
    
    # 2. Reorder & Filter Columns (Remove Extras)
    # 'drop = FALSE' prevents vector conversion if only 1 feature remains
    final_matrix <- test_matrix[, train_cols, drop = FALSE]
  } else {
    warning("Model has no feature names stored. Assuming strictly correct column order.")
    final_matrix <- test_matrix
  }
  
  # E. Predict
  dtest <- xgb.DMatrix(data = final_matrix)
  return(predict(model, dtest))
}


# 4. Generate Predictions (The "Generation Step") -----------------------------
message("Generating predictions for individual models...")

preds_list <- list()
model_names <- names(Model_List)

for(i in seq_along(Model_List)) {
  name <- model_names[i]
  message(paste("Predicting:", name))
  
  # Call the robust helper function
  preds_list[[name]] <- get_xgb_pred(Model_List[[i]], Test_Data_List[[i]])
}

# Verify structure
print(str(preds_list))


# 5. Construct Ensembles ------------------------------------------------------

# Ensemble A: Base + Strategy A
ens_A_preds <- (preds_list$Base + preds_list$StrategyA) / 2

# Ensemble B: Strategy A + Strategy C
ens_B_preds <- (preds_list$StrategyA + preds_list$StrategyC) / 2

# Ensemble C: Average of All Models (Base + A + B + C)
ens_C_preds <- (preds_list$Base + preds_list$StrategyA + preds_list$StrategyB + preds_list$StrategyC) / 4


# 6. Evaluation Function ------------------------------------------------------
evaluate_ensemble <- function(preds, actuals, name) {
  # AUC
  roc_obj <- pROC::roc(actuals, preds, quiet = TRUE)
  auc_score <- pROC::auc(roc_obj)
  
  # Brier Score
  brier <- mean((preds - actuals)^2)
  
  # Penalized Brier Score
  # Custom metric: Penalize errors on the wrong side of threshold 0.5
  R <- 2  
  penalty_term <- (R - 1) / R 
  pred_class <- round(preds) # Threshold at 0.5
  penalty_vec <- ifelse(pred_class != actuals, penalty_term, 0)
  pen_brier <- mean((preds - actuals)^2 + penalty_vec)
  
  return(data.frame(
    Model = name, 
    AUC = round(as.numeric(auc_score), 5), 
    Brier = round(brier, 5), 
    Pen_Brier = round(pen_brier, 5)
  ))
}


# 7. Run Evaluation & Leaderboard ---------------------------------------------

# Individual Models
res_Base   <- evaluate_ensemble(preds_list$Base, actuals, "Single: Base Model")
res_StratA <- evaluate_ensemble(preds_list$StrategyA, actuals, "Single: Strategy A (VAE)")
res_StratB <- evaluate_ensemble(preds_list$StrategyB, actuals, "Single: Strategy B (Anomaly)")
res_StratC <- evaluate_ensemble(preds_list$StrategyC, actuals, "Single: Strategy C (Denoise)")

# Ensembles
res_EnsA <- evaluate_ensemble(ens_A_preds, actuals, "Ensemble A (Base + A)")
res_EnsB <- evaluate_ensemble(ens_B_preds, actuals, "Ensemble B (A + C)")
res_EnsC <- evaluate_ensemble(ens_C_preds, actuals, "Ensemble C (All Models)")

# Combine into Leaderboard
leaderboard <- rbind(res_EnsA, res_EnsB, res_EnsC, 
                     res_StratA, res_StratB, res_StratC, res_Base) %>%
  arrange(desc(AUC))

print("=========================================")
print("          ENSEMBLE LEADERBOARD           ")
print("=========================================")
print(leaderboard)


# 8. Correlation Visualization (Orthogonality) --------------------------------

# Create Dataframe of Predictions
pred_df <- as.data.frame(preds_list)
colnames(pred_df) <- c("Base", "Strat A (VAE)", "Strat B (Anomaly)", "Strat C (Denoise)")

# Calculate Correlation Matrix
M <- cor(pred_df)

# Plot
corrplot(M, 
         method = "color", 
         col = colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))(200),
         type = "upper", 
         order = "hclust", 
         addCoef.col = "black", 
         tl.col = "black", 
         tl.srt = 45, 
         diag = FALSE,
         title = "Prediction Correlation Matrix",
         mar = c(0,0,2,0)
)
