#==============================================================================#
#==== 00 - Data Preparation ===================================================#
#==============================================================================#

##==============================##
## General Parameters.
##==============================##

N_folds <- 5

#==== 0A - Data Preparation ===================================================#

tryCatch({
  
##=========================================##
##==== Dataset preparation.
##=========================================##
  
### Base Model.
Train_Data_Base_Model <- Train_Transformed
Train_Data_Base_Model <- Train_Data_Base_Model %>%
    mutate(
      id = Train_with_id$id) 
  
######## DO NOT NEED TO ADD THE ID AS WE USE THE SAME ROWS LATER.
### Strategy A: latent features.
Final_Train_Set_A <- cbind(Train_Transformed, Strategy_A_LF)
Train_Data_Strategy_A <- Final_Train_Set_A
# Train_Data_Strategy_A <- Train_Data_Strategy_A %>%
#   mutate(
#     id = Train_with_id$id) 
  
### Strategy B: anomaly score.
Train_Data_Strategy_B <- Strategy_B_AS
# Train_Data_Strategy_B <- Train_Data_Strategy_B %>%
#   mutate(
#     id = Train_with_id$id) 
  
## Strategy C: regime switching.
Train_Data_Strategy_C <- Strategy_B_AS_revised
# Train_Data_Strategy_C <- Train_Data_Strategy_C %>%
#   mutate(
#     id = Train_with_id$id) 
  
## Strategy D: fitting on the residuals of the base model.
# Train_Data_Strategy_D <- Strategy_D_Soft
  
### Strategy E: Denoising the features.
# Train_Data_Strategy_E <- Train_Data_Strategy_E
  
##=========================================##
##==== First stratify by IDs.
##=========================================##
  
Data_Train_CV_Split_IDs <- MVstratifiedsampling_CV_ID(data = Train_Data_Base_Model, 
                                                      num_folds = N_folds)
  
##=========================================##
##==== Now use the stratified IDs to get the row in the train set for each fold.
##=========================================##
  
## Rows remain the same.
### Base Model.
Data_Train_CV_Base_Model <- MVstratifiedsampling_CV_Split(data = Train_Data_Base_Model, 
                                                            firm_fold_indices = Data_Train_CV_Split_IDs) 
  
### Strategy B.
# Data_Train_CV_Strategy_B <- MVstratifiedsampling_CV_Split(data = Train_Data_Strategy_B, 
#                                                           firm_fold_indices = Data_Train_CV_Split_IDs)
  
##=========================================##
##==== Remove the ID column once more.
##=========================================##
  
### Base Model.
Train_Data_Base_Model <- Train_Data_Base_Model %>%
    select(-id)
  
  ##=========================================##
  
}, error = function(e) message(e))

#==============================================================================#
#==== 01 - GLM Model Training =================================================#
#==============================================================================#

##==============================##
## General Parameters.
##==============================##

n_init_points <- 2
n_iter_bayes  <- 4

#==== 01A - Base model ========================================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Base_Model

##==============================##
## Code.
##==============================##

GLM_Results_BaseModel <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                      Train_Data = Train_Data,
                                      n_init_points = n_init_points,
                                      n_iter_bayes = n_iter_bayes)

}, error = function(e) message("Training Base Model Error: ", e))

#==== 01B - Strategy A: Latent features =======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Strategy_A
  
##==============================##
## Code.
##==============================##
  
GLM_Results_Strategy_A <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                       Train_Data = Train_Data,
                                       n_init_points = n_init_points,
                                       n_iter_bayes = n_iter_bayes)
  
}, error = function(e) message("Training Strategy A Error: ", e))

#==== 01C - Strategy B: Anomaly Score =========================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Strategy_B
  
##==============================##
## Code.
##==============================##
  
  GLM_Results_Strategy_B <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                         Train_Data = Train_Data,
                                         n_init_points = n_init_points,
                                         n_iter_bayes = n_iter_bayes)
  
}, error = function(e) message("Training Strategy B Error: ", e))

#==== 01D - Strategy C: Regime Switching ======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Strategy_C
  
##==============================##
## Code.
##==============================##
  
GLM_Results_Strategy_C <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                       Train_Data = Train_Data,
                                       n_init_points = n_init_points,
                                       n_iter_bayes = n_iter_bayes)
  
}, error = function(e) message("Training Strategy C Error: ", e))

#==== 01E - Strategy D: Residuals of the base model ===========================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Strategy_D
  
##==============================##
## Code.
##==============================##
  
GLM_Results_Strategy_D <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                       Train_Data = Train_Data,
                                       n_init_points = n_init_points,
                                       n_iter_bayes = n_iter_bayes)
  
}, error = function(e) message("Training Strategy D Error: ", e))

#==== 01F - Strategy E: Denoising =============================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]
Train_Data <- Train_Data_Strategy_E
  
##==============================##
## Code.
##==============================##
  
GLM_Results_Strategy_E <- GLM_Training(Data_Train_CV_Vector = Data_Train_CV_Vector,
                                       Train_Data = Train_Data,
                                       n_init_points = n_init_points,
                                       n_iter_bayes = n_iter_bayes)
  
}, error = function(e) message("Training Strategy E Error: ", e))

#==============================================================================#
#==== 02 - GLM Model Comparison (AUC and Parameters) ==========================#
#==============================================================================#

tryCatch({
  
  extract_metrics <- function(model_obj, model_name) {
    
    # Safety Check: If model didn't train, return NAs
    if (is.null(model_obj) || is.null(model_obj$results)) {
      warning(paste("Model", model_name, "is missing. Returning NA row."))
      return(data.frame(Model = model_name, AUC = NA, Brier_Score = NA, 
                        Penalized_Brier = NA, Model_Type = "Error"))
    }
    
    # 1. Standard Metrics (Available in both)
    best_auc  <- model_obj$results$AUC[1]
    brier_val <- if(!is.null(model_obj$Brier_Score)) model_obj$Brier_Score else NA
    
    # 2. Penalized Brier Score (Specific to GLM usually)
    penalized_brier <- if(!is.null(model_obj$Penalized_Brier_Score)) model_obj$Penalized_Brier_Score else NA
    
    # 3. Parameter Extraction (Conditional)
    params <- model_obj$optimal_parameters
    
    # Initialize defaults
    eta <- NA; depth <- NA; subs <- NA; colsample <- NA; rounds <- NA
    alpha <- NA; lambda <- NA
    model_type <- "Unknown"
    
    # Detect XGBoost (has "eta")
    if ("eta" %in% names(params)) {
      model_type <- "XGBoost"
      eta        <- params$eta
      depth      <- params$max_depth
      subs       <- params$subsample
      colsample  <- params$colsample_bytree
      rounds     <- model_obj$optimal_rounds
    } 
    # Detect GLM (has "alpha")
    else if ("alpha" %in% names(params)) {
      model_type <- "GLM"
      alpha      <- params$alpha
      lambda     <- params$lambda
    }
    
    # 4. Return Dataframe
    data.frame(
      Model = model_name,
      Type  = model_type,
      AUC   = best_auc,
      Brier_Score = brier_val,
      Penalized_Brier = penalized_brier,
      
      # Model Specific Params
      Alpha  = alpha,
      Lambda = lambda,
      Rounds = rounds,
      Eta    = eta,
      Max_Depth = depth
    )
  }
  
}, error = function(e) message("Function Definition Error: ", e))

#==== 02A - Compare the AUC-Score =============================================#

tryCatch({
  
  # Combine results from the GLM models you just trained
  comparison_table <- bind_rows(
    extract_metrics(GLM_Results_BaseModel,  "Base Model (GLM)"),
    extract_metrics(GLM_Results_Strategy_A, "Strategy A (Latent)"),
    extract_metrics(GLM_Results_Strategy_B, "Strategy B (Anomaly)"),
    extract_metrics(GLM_Results_Strategy_C, "Strategy C (Regime)"),
    extract_metrics(GLM_Results_Strategy_D, "Strategy D (Residual)"),
    extract_metrics(GLM_Results_Strategy_E, "Strategy E (Denoising)")
  ) %>%
    # Filter out failed models
    filter(!is.na(AUC)) %>%
    arrange(desc(AUC)) %>% 
    mutate(
      # 1. AUC Uplift
      Base_AUC = AUC[Model == "Base Model (GLM)"],
      Uplift_AUC_pct = ((AUC - Base_AUC) / Base_AUC) * 100,
      # 2. Standard Brier Uplift (Lower is Better)
      Base_Brier = Brier_Score[Model == "Base Model (GLM)"],
      Uplift_Brier_pct = -((Brier_Score - Base_Brier) / Base_Brier) * 100,
      # 3. Penalized Brier Uplift (Lower is Better)
      Base_PBS = Penalized_Brier[Model == "Base Model (GLM)"],
      Uplift_PBS_pct = -((Penalized_Brier - Base_PBS) / Base_PBS) * 100
    ) %>%
    # Organize columns logically
    select(Model, AUC, Uplift_AUC_pct, 
           Brier_Score, Uplift_Brier_pct,
           Penalized_Brier, Uplift_PBS_pct,
           Alpha, Lambda) 
  
  print("--- Final GLM Model Leaderboard ---")
  print(comparison_table)
  
}, error = function(e) message("Comparison Error: ", e))

#==============================================================================#
#==== 03 - GLM Model in the test-set ==========================================#
#==============================================================================#

#==== 03A - Data Preparation ==================================================#

tryCatch({
  
### Base model.
Test_Data_Base_Model <- Test_Transformed

### Strategy A: latent features.
Final_Test_Set_A <- cbind(Test_Transformed, Strategy_A_LF_Test)
Test_Data_Strategy_A <- Final_Test_Set_A
  
### Strategy B: anomaly score.
Test_Data_Strategy_B <- Strategy_B_AS_Test
  
### Strategy C: regime switching.
Test_Data_Strategy_C <- Strategy_C_Test_Soft
  
### Strategy D: residual fit
Test_Data_Strategy_D <- Strategy_D_Test_Soft
  
### Strategy E: Denoising
Test_Data_Strategy_E <- Test_Data_Strategy_E
  
}, error = function(e) message(e))

#==== 03B - Base Model ========================================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_BaseModel$optimal_model
Test_Data <- Test_Data_Base_Model
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_BaseModel <- GLM_Test(Model = Model, 
                                           Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==== 03C - Strategy A: Latent features =======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_Strategy_A$optimal_model
Test_Data <- Test_Data_Strategy_A
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_Strategy_A <- GLM_Test(Model = Model, 
                                         Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==== 03D - Strategy B: Anomaly Score =========================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_Strategy_B$optimal_model
Test_Data <- Test_Data_Strategy_B
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_Strategy_B <- GLM_Test(Model = Model, 
                                        Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==== 03E - Strategy C: Regime Switching ======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_Strategy_C$optimal_model
Test_Data <- Test_Data_Strategy_C
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_Strategy_C <- GLM_Test(Model = Model, 
                                        Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==== 03F - Strategy D: Regime Switching ======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_Strategy_D$optimal_model
Test_Data <- Test_Data_Strategy_D
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_Strategy_D <- GLM_Test(Model = Model, 
                                        Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==== 03G - Strategy E: Denoising =============================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Model <- GLM_Results_Strategy_E$optimal_model
Test_Data <- Test_Data_Strategy_E
  
##==============================##
## Code.
##==============================##
  
GLM_Test_Results_Strategy_E <- GLM_Test(Model = Model, 
                                        Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==============================================================================#
#==== 04 - XGBoost Test Comparison (AUC and Parameters) =======================#
#==============================================================================#

#==== 04A - Compare the AUC-Score =============================================#

tryCatch({
  
  # Bind all result dataframes
  Final_Leaderboard <- bind_rows(
    GLM_Test_Results_BaseModel$Metrics  %>% mutate(Strategy = "Base Model"),
    GLM_Test_Results_Strategy_A$Metrics %>% mutate(Strategy = "Strategy A (Latent)"),
    GLM_Test_Results_Strategy_B$Metrics %>% mutate(Strategy = "Strategy B (Anomaly)"),
    GLM_Test_Results_Strategy_C$Metrics %>% mutate(Strategy = "Strategy C (Regime)"),
    GLM_Test_Results_Strategy_D$Metrics %>% mutate(Strategy = "Strategy D (Residual Fit)"),
    GLM_Test_Results_Strategy_E$Metrics %>% mutate(Strategy = "Strategy E (Denoising)")
  ) %>%
    select(Strategy, AUC, Brier_Score, Penalized_Brier_Score, Log_Loss, Inference_Time_Sec) %>%
    arrange(desc(AUC))
  
  # Calculate Uplift based on Base Model
  base_auc <- Final_Leaderboard$AUC[Final_Leaderboard$Strategy == "Base Model"]
  base_pbs <- Final_Leaderboard$Penalized_Brier_Score[Final_Leaderboard$Strategy == "Base Model"]
  
  Final_Leaderboard <- Final_Leaderboard %>%
    mutate(
      Uplift_AUC_pct = round(((AUC - base_auc) / base_auc) * 100, 3),
      # For PBS, lower is better, so we negate the percentage change
      Uplift_PBS_pct = round(-((Penalized_Brier_Score - base_pbs) / base_pbs) * 100, 3),
      Winner = ifelse(AUC == max(AUC), "**WINNER**", "")
    ) %>%
    select(Strategy, Winner, AUC, Uplift_AUC_pct, Penalized_Brier_Score, Uplift_PBS_pct, everything())
  
  print("=======================================================")
  print("             FINAL GLM MODEL LEADERBOARD               ")
  print("=======================================================")
  print(Final_Leaderboard)
  
  # Visualization
  p_results <- ggplot(Final_Leaderboard, aes(x = reorder(Strategy, AUC), y = AUC, fill = Strategy)) +
    geom_col(width = 0.6) +
    coord_flip() +
    geom_text(aes(label = paste0(round(AUC, 4), " (", Uplift_AUC_pct, "%)")), 
              hjust = -0.1, fontface = "bold") +
    scale_fill_brewer(palette = "Set2") +
    labs(title = "Final Impact Analysis: Did the VAE features help GLM?",
         subtitle = "Comparing Test Set AUC across all strategies",
         y = "Test AUC", x = "") +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p_results)
  
}, error = function(e) message("Leaderboard Error: ", e))

#==============================================================================#
#==============================================================================#
#==============================================================================#