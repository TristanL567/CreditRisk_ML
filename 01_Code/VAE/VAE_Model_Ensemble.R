#==============================================================================#
#==== 00 - Data Preparation ===================================================#
#==============================================================================#

library(dplyr)
library(glmnet)
library(pROC)
library(corrplot)

Path <- dirname(this.path::this.path())
setwd(Path)

Data_Directory_Ensemble <- file.path(Path, "02_Data/VAE/Ensemble")
Data_Directory_Ensemble <- gsub("/01_Code/VAE", "", Data_Directory_Ensemble)

##==============================##
## General Parameters.
##==============================##

#==== 0A - Data Preparation ===================================================#

model_names <- c("BaseModel", "StrategyA", "StrategyB", "StrategyC", "StrategyD")

## GLM_models:

# glm_objects <- list(
#   GLM_Results_BaseModel,
#   GLM_Results_Strategy_A,
#   GLM_Results_Strategy_B,
#   GLM_Results_Strategy_C,
#   GLM_Results_Strategy_D
# )

SavePath <- file.path(Data_Directory_Ensemble, "glm_objects.RData")
# saveRDS(glm_objects, file = SavePath)
glm_objects <- readRDS(file = SavePath)

## XGBoost_models:

# xgb_objects <- list(
#   XGBoost_Results_BaseModel,
#   XGBoost_Results_Strategy_A,
#   XGBoost_Results_Strategy_B,
#   XGBoost_Results_Strategy_C,
#   XGBoost_Results_Strategy_D
# )

SavePath <- file.path(Data_Directory_Ensemble, "xgb_objects.RData")
# saveRDS(xgb_objects, file = SavePath)
xgb_objects <- readRDS(file = SavePath)

#==============================================================================#
#==== 01 - Building the Ensemble model ========================================#
#==============================================================================#

library(pROC)
library(dplyr)
library(tibble)

# =============================================================================#
# 1. SETUP: LOAD DATA & PREDICTIONS
# =============================================================================#

# Load Test Y (Actuals)
test_y <- as.numeric(as.character(Test_Data_Base_Model$y))

# Helper to get Test Predictions (Probabilities)
get_pred <- function(model_obj, data, is_glm=TRUE) {
  sparse_formula <- as.formula("y ~ . - 1")
  X_test <- sparse.model.matrix(sparse_formula, data = data)
  if(is_glm) {
    as.numeric(predict(model_obj, newx = X_test, s = "lambda.min", type = "response"))
  } else {
    as.numeric(predict(model_obj, X_test)) 
  }
}

# --- Extract Predictions for Key Models ---

# Base Models
p_glm_base <- get_pred(GLM_Results_BaseModel$optimal_model, Test_Data_Base_Model, TRUE)
p_xgb_base <- get_pred(XGBoost_Results_BaseModel$optimal_model, Test_Data_Base_Model, FALSE)

# Strategy C (GLM)
p_glm_stratC <- get_pred(GLM_Results_Strategy_C$optimal_model, Test_Data_Strategy_C, TRUE)

# Strategy D (XGBoost)
p_xgb_stratD <- get_pred(XGBoost_Results_Strategy_D$optimal_model, Test_Data_Strategy_D, FALSE)

# All Models (For Ensemble B)
# (Assuming you can loop through them or just grab the ones needed if lists are ready)
# We will use the 'p_glm_base' etc variables for clarity.

# =============================================================================#
# 2. CONSTRUCT ENSEMBLES (Averaging)
# =============================================================================#

# --- Ensemble A: Average GLM Base + XGB Base ---
# Logic: Combine Solvency (GLM) + Liquidity (XGB) baselines
ens_A_preds <- (p_glm_base + p_xgb_base) / 2

# --- Ensemble B: Average ALL Models ---
# Logic: "Wisdom of the Crowd" - suppress noise by averaging everything
# We need to gather all 10 predictions first
all_preds_matrix <- do.call(cbind, lapply(1:5, function(i) {
  cbind(
    get_pred(glm_objects[[i]]$optimal_model, data_input_list[[i]], TRUE), # Make sure data_input_list matches Test Data
    get_pred(xgb_objects[[i]]$optimal_model, data_input_list[[i]], FALSE)
  )
}))
# Note: You need to ensure 'data_input_list' here refers to TEST data lists. 
# If not defined, define test_data_list <- list(Test_Data_Base, Test_Data_A...) first.

# robust way for Ensemble B using the specific lists:
test_data_list <- list(Test_Data_Base_Model, Test_Data_Strategy_A, Test_Data_Strategy_B, Test_Data_Strategy_C, Test_Data_Strategy_D)
all_preds <- matrix(0, nrow = length(test_y), ncol = 10)
idx <- 1
for(i in 1:5) {
  all_preds[, idx]   <- get_pred(glm_objects[[i]]$optimal_model, test_data_list[[i]], TRUE)
  all_preds[, idx+1] <- get_pred(xgb_objects[[i]]$optimal_model, test_data_list[[i]], FALSE)
  idx <- idx + 2
}
ens_B_preds <- rowMeans(all_preds)

# --- Ensemble C: GLM Base + GLM Strat C + XGB Base + XGB Strat D ---
# Logic: Base Solvency + Robust Solvency (C) + Base Liquidity + Zombie Hunter (D)
ens_C_preds <- (p_glm_base + p_glm_stratC + p_xgb_base + p_xgb_stratD) / 4


# =============================================================================#
# 3. EVALUATION FUNCTION
# =============================================================================#

evaluate_ensemble <- function(preds, actuals, name) {
  # AUC
  roc_obj <- pROC::roc(actuals, preds, quiet=TRUE)
  auc_score <- pROC::auc(roc_obj)
  
  # Brier
  brier <- mean((preds - actuals)^2)
  
  # Penalized Brier
  R <- 2; penalty_term <- (R - 1) / R
  pen_vec <- ifelse(round(preds) != actuals, penalty_term, 0)
  pen_brier <- mean((preds - actuals)^2 + pen_vec)
  
  return(data.frame(Model = name, AUC = auc_score, Brier = brier, Pen_Brier = pen_brier))
}

# =============================================================================#
# 4. RUN EVALUATION & PRINT LEADERBOARD
# =============================================================================#

res_A <- evaluate_ensemble(ens_A_preds, test_y, "Ensemble A (Base Avg)")
res_B <- evaluate_ensemble(ens_B_preds, test_y, "Ensemble B (All Avg)")
res_C <- evaluate_ensemble(ens_C_preds, test_y, "Ensemble C (Selected Top)")

# Compare with best single models for reference
res_XGB_B <- evaluate_ensemble(
  get_pred(XGBoost_Results_Strategy_B$optimal_model, Test_Data_Strategy_B, FALSE), 
  test_y, "Single Best (XGB Strat B)"
)

final_leaderboard <- rbind(res_A, res_B, res_C, res_XGB_B) %>%
  arrange(desc(AUC))

print(final_leaderboard)

library(dplyr)
library(corrplot)
library(RColorBrewer)

# =============================================================================#
# VISUALIZE MODEL CORRELATION (ORTHOGONALITY CHECK)
# =============================================================================#

# 1. Prepare the Dataframe of Predictions
# Assuming you have your model objects loaded (GLM_Results_... and XGBoost_Results_...)
# We extract the 'Predicted' probabilities from the test set (or OOF CV predictions)

ensemble_df <- data.frame(
  # GLM Predictions
  GLM_Base   = GLM_Results_BaseModel$Predictions$Predicted,
  GLM_StratA = GLM_Results_Strategy_A$Predictions$Predicted,
  GLM_StratC = GLM_Results_Strategy_C$Predictions$Predicted,
  
  # XGBoost Predictions
  XGB_Base   = XGBoost_Results_BaseModel$Predictions$Predicted,
  XGB_StratB = XGBoost_Results_Strategy_B$Predictions$Predicted,
  XGB_StratD = XGBoost_Results_Strategy_D$Predictions$Predicted
)

# 2. Calculate Correlation Matrix
M <- cor(ensemble_df)

# 3. Plotting
# We use a diverging color palette to highlight the difference between
# High Correlation (Dark Blue) and Low Correlation (White/Red)

col_palette <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(M, 
         method = "color",       # Use colored tiles
         col = col_palette(200), # Apply palette
         type = "upper",         # Show only upper triangle
         order = "hclust",       # Cluster similar models together
         addCoef.col = "black",  # Add numeric correlation coefficient
         tl.col = "black",       # Text label color
         tl.srt = 45,            # Rotate text labels
         diag = FALSE,           # Hide diagonal
         title = "Correlation of Model Predictions (Orthogonality Check)",
         mar = c(0,0,2,0)        # Adjust margins for title
)
