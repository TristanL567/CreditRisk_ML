#==============================================================================#
#==== 05 - Ensemble Construction ==============================================#
#==============================================================================#

library(dplyr)
library(xgboost)
library(Matrix)
library(pROC)
library(corrplot)
library(tibble)
library(RColorBrewer)

# 1. Setup & Load Data --------------------------------------------------------
# Assuming xgb_objects are already loaded or exist in environment
# If not, uncomment the load line:
# xgb_objects <- readRDS(file = file.path(Data_Directory_Ensemble, "xgb_objects.RData"))

# Define the models and their corresponding Test Datasets
# Ensure these objects exist in your environment from previous steps
Model_List <- list(
  Base      = XGBoost_Results_BaseModel$optimal_model,
  StrategyA = XGBoost_Results_Strategy_A$optimal_model,
  StrategyB = XGBoost_Results_Strategy_B$optimal_model,
  StrategyC = XGBoost_Results_Strategy_C$optimal_model
)

# Important: List corresponding Test Dataframes (must match Model_List order)
Test_Data_List <- list(
  Base      = Test_Data_Base_Model,
  StrategyA = Test_Data_Strategy_A,       # From Step 04C (VAE Latent)
  StrategyB = Test_Data_Strategy_B,    # From Step 04C (Anomaly Score)
  StrategyC = Test_Data_Strategy_C        # From Step 04D (Denoising)
)

# Extract Actuals (y) from one of the test sets
test_y_raw <- Test_Data_Base_Model$y
actuals <- as.numeric(as.character(test_y_raw))

# 2. Robust Prediction Helper -------------------------------------------------
# This function handles the different feature sets (Alignment) automatically

get_xgb_pred <- function(model, test_data) {
  
  require(Matrix)
  require(xgboost)
  
  # 1. Clean Metadata
  cols_to_drop <- c("id", "refdate", "time_index", "year", "row_id", "company_id")
  test_clean <- test_data[, !names(test_data) %in% cols_to_drop]
  
  # 2. Create Sparse Matrix
  sparse_formula <- as.formula("y ~ . - 1")
  test_matrix <- sparse.model.matrix(sparse_formula, data = test_clean, na.action = "na.pass")
  
  # 3. ALIGNMENT LOGIC (With Safety Fallback)
  train_cols <- model$feature_names
  
  if(is.null(train_cols)) {
    # --- FALLBACK MODE ---
    # The model lost its column names. We assume Test Data is correct.
    # We warn the user but proceed instead of stopping.
    warning(paste("Model", "has no feature names stored. Skipping column alignment and assuming structure is correct."))
    
    # We accept the matrix as-is
    final_matrix <- test_matrix
    
  } else {
    # --- ROBUST ALIGNMENT MODE ---
    # The model knows what it wants. We force the Test Matrix to match.
    
    # A. Add Missing Columns (Fill with 0)
    missing_cols <- setdiff(train_cols, colnames(test_matrix))
    if(length(missing_cols) > 0) {
      new_cols <- Matrix(0, nrow = nrow(test_matrix), ncol = length(missing_cols), sparse = TRUE)
      colnames(new_cols) <- missing_cols
      test_matrix <- cbind(test_matrix, new_cols)
    }
    
    # B. Remove Extra Columns
    extra_cols <- setdiff(colnames(test_matrix), train_cols)
    if(length(extra_cols) > 0) {
      test_matrix <- test_matrix[, !colnames(test_matrix) %in% extra_cols]
    }
    
    # C. Reorder Columns
    final_matrix <- test_matrix[, train_cols, drop = FALSE]
  }
  
  # 4. EXPLICIT CONVERSION
  # Convert to xgb.DMatrix to prevent "double" errors
  dtest <- xgb.DMatrix(data = final_matrix)
  
  # 5. Predict
  return(predict(model, dtest))
}

# 3. Generate Individual Predictions ------------------------------------------
message("Generating predictions for individual models...")

preds_list <- list()
model_names <- names(Model_List)

for(i in seq_along(Model_List)) {
  name <- model_names[i]
  message(paste("Predicting:", name))
  preds_list[[name]] <- get_xgb_pred(Model_List[[i]], Test_Data_List[[i]])
}

# Verify results
print("Predictions generated:")
print(str(preds_list))

# 4. Construct Ensembles ------------------------------------------------------

# Ensemble Approach A: Base Model + Strategy A
# Logic: Combines raw financials with VAE Latent Features
ens_A_preds <- (preds_list$Base + preds_list$StrategyA) / 2

# Ensemble Approach B: Strategy A + Strategy C
# Logic: Combines Dimensionality Reduction (VAE) with Feature Denoising (DAE)
ens_B_preds <- (preds_list$StrategyA + preds_list$StrategyC) / 2

# Ensemble Approach C: All Models Together (Base + A + B + C)
# Logic: Maximum diversity (Raw + Latent + Anomaly + Denoised)
ens_C_preds <- (preds_list$Base + preds_list$StrategyA + preds_list$StrategyB + preds_list$StrategyC) / 4

# 5. Evaluation Function ------------------------------------------------------

evaluate_ensemble <- function(preds, actuals, name) {
  # AUC
  roc_obj <- pROC::roc(actuals, preds, quiet = TRUE)
  auc_score <- pROC::auc(roc_obj)
  
  # Brier Score
  brier <- mean((preds - actuals)^2)
  
  # Penalized Brier Score (As defined in your setup)
  # Penalty applied if prediction is on the wrong side of 0.5
  R <- 2 
  penalty_term <- (R - 1) / R # 0.5
  
  pred_class <- round(preds)
  penalty_vec <- ifelse(pred_class != actuals, penalty_term, 0)
  pen_brier <- mean((preds - actuals)^2 + penalty_vec)
  
  return(data.frame(
    Model = name, 
    AUC = round(auc_score, 5), 
    Brier = round(brier, 5), 
    Pen_Brier = round(pen_brier, 5)
  ))
}

# 6. Run Evaluation & Leaderboard ---------------------------------------------

# Individual Models (for comparison)
res_Base   <- evaluate_ensemble(preds_list$Base, actuals, "Single: Base Model")
res_StratA <- evaluate_ensemble(preds_list$StrategyA, actuals, "Single: Strategy A (VAE)")

# Ensembles
res_EnsA <- evaluate_ensemble(ens_A_preds, actuals, "Ensemble A (Base + A)")
res_EnsB <- evaluate_ensemble(ens_B_preds, actuals, "Ensemble B (A + C)")
res_EnsC <- evaluate_ensemble(ens_C_preds, actuals, "Ensemble C (All Models)")

# Combine into Leaderboard
leaderboard <- rbind(res_EnsA, res_EnsB, res_EnsC, res_StratA, res_Base) %>%
  arrange(desc(AUC))

print("=========================================")
print("          ENSEMBLE LEADERBOARD           ")
print("=========================================")
print(leaderboard)

# 7. Correlation Visualization (Orthogonality) --------------------------------

# Create Dataframe of Predictions
pred_df <- as.data.frame(preds_list)
colnames(pred_df) <- c("Base", "Strat A (VAE)", "Strat B (Anomaly)", "Strat C (Denoise)")

# Calculate Correlation
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
