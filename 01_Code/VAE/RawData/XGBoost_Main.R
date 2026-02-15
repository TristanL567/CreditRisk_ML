#==============================================================================#
#==== 00 - Data Preparation ===================================================#
#==============================================================================#

Path <- dirname(this.path::this.path())
setwd(Path)

source("XGBoost_Training.R")
source("XGBoost_Test.R")

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

### Strategy B: anomaly score.
Train_Data_Strategy_B <- Strategy_B_AS

### Strategy C: Manual Feature engineering.
Train_Data_Strategy_C <- Strategy_C

### Strategy D: fitting on the residuals of the base model.
Train_Data_Strategy_D <- Strategy_D

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

##=========================================##
##==== Remove the ID column once more.
##=========================================##

### Base Model.
Train_Data_Base_Model <- Train_Data_Base_Model %>%
  select(-id)

##=========================================##

}, error = function(e) message(e))

#==============================================================================#
#==== 01 - XGBoost Model Training =============================================#
#==============================================================================#

##==============================##
## General Parameters.
##==============================##

n_init_points <- 10
n_iter_bayes  <- 20

#==== 01A - Base model ========================================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Base_Model

##==============================##
## Code.
##==============================##

XGBoost_Results_BaseModel <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                              Train_Data = Train_Data,
                                              n_init_points = n_init_points,
                                              n_iter_bayes = n_iter_bayes)

}, error = function(e) message(e))

#==== 01B - Strategy A: Latent features =======================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Strategy_A

##==============================##
## Code.
##==============================##

XGBoost_Results_Strategy_A <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                               Train_Data = Train_Data,
                                               n_init_points = n_init_points,
                                               n_iter_bayes = n_iter_bayes)

##==============================##

}, error = function(e) message(e))

#==== 01C - Strategy B: Anomaly Score =========================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Strategy_B

##==============================##
## Code.
##==============================##

XGBoost_Results_Strategy_B <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                               Train_Data = Train_Data,
                                               n_init_points = n_init_points,
                                               n_iter_bayes = n_iter_bayes)

##==============================##

}, error = function(e) message(e))

#==== 01D - Strategy C: Feature Denoising =====================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Strategy_C

##==============================##
## Code.
##==============================##

XGBoost_Results_Strategy_C <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                               Train_Data = Train_Data,
                                               n_init_points = n_init_points,
                                               n_iter_bayes = n_iter_bayes)

##==============================##

}, error = function(e) message(e))

#==== 01E - Strategy D: Manual Feature Engineering ============================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Strategy_D

##==============================##
## Code.
##==============================##
  
XGBoost_Results_Strategy_D <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                               Train_Data = Train_Data,
                                               n_init_points = n_init_points,
                                               n_iter_bayes = n_iter_bayes)
    
}, error = function(e) message(".", e))

#==============================================================================#
#==== 02 - XGBoost Model Comparison (AUC and Parameters) ======================#
#==============================================================================#

tryCatch({
    
    extract_metrics <- function(model_obj, model_name) {
      
      # 1. Safely extract metrics (handling potential NULLs)
      best_auc <- if(!is.null(model_obj$results$AUC)) model_obj$results$AUC[1] else NA
      params   <- model_obj$optimal_parameters
      
      # Check for Brier Scores in the new structure
      brier_val     <- if(!is.null(model_obj$Brier_Score)) model_obj$Brier_Score else NA
      pen_brier_val <- if(!is.null(model_obj$Penalized_Brier_Score)) model_obj$Penalized_Brier_Score else NA
      
      # 2. Build Dataframe
      data.frame(
        Model = model_name,
        AUC = best_auc,
        Brier_Score = brier_val,  
        Pen_Brier_Score = pen_brier_val,
        Optimal_Rounds = model_obj$optimal_rounds,
        # Safely extract parameters (params is a list/row)
        Eta = params$eta,
        Max_Depth = params$max_depth,
        Subsample = params$subsample,
        Colsample = params$colsample_bytree
      )
    }

}, error = function(e) message(e))

#==== 02A - Compare the AUC-Score =============================================#

tryCatch({
  
  # 1. Combine Results
  comparison_table <- bind_rows(
    extract_metrics(XGBoost_Results_BaseModel,  "Base Model"),
    extract_metrics(XGBoost_Results_Strategy_A, "Strategy A (Dim. Reduction)"),
    extract_metrics(XGBoost_Results_Strategy_B, "Strategy B (Anomaly Score)"),
    extract_metrics(XGBoost_Results_Strategy_C, "Strategy C (Feature Denoising)"),
    extract_metrics(XGBoost_Results_Strategy_D, "Strategy D (Manual Feature Eng.)")
  ) %>%
    # 2. Calculate Uplifts
    mutate(
      # Get Base Metrics for comparison
      Base_AUC   = AUC[Model == "Base Model"],
      Base_Brier = Brier_Score[Model == "Base Model"],
      
      # AUC Uplift (Positive is Good)
      Uplift_AUC_pct = ((AUC - Base_AUC) / Base_AUC) * 100,
      
      # Brier Uplift (Negative is Good - reduction in error)
      Uplift_Brier_pct = ((Brier_Score - Base_Brier) / Base_Brier) * 100
    ) %>%
    arrange(desc(AUC)) %>% 
    # 3. Final Selection
    select(
      Model, 
      AUC, 
      Uplift_AUC_pct, 
      Brier_Score, 
      Uplift_Brier_pct, 
      Pen_Brier_Score, # Added this as it's useful to see
      Optimal_Rounds, 
      Eta, 
      Max_Depth
    )
  
  print("--- Final Model Leaderboard ---")
  print(comparison_table)

}, error = function(e) message(e))

#==============================================================================#
#==== 02 - XGBoost Model Comparison (Plots & Charts) ==========================#
#==============================================================================#

#==== 02B - Settings ==========================================================#

tryCatch({

# Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.
# Charts_Directory_Model <- file.path(Path, "03_Charts/VAE/XGBoost")
# 
# ###### Input parameters.
# ## BaseModel, StrategyA, StrategyB or StrategyC, StrategyD
# model_used_name <- "StrategyD"
# 
# ## XGBoost_Results_BaseModel, XGBoost_Results_Strategy_A,
# ## XGBoost_Results_Strategy_B, XGBoost_Results_Strategy_C,
# ## XGBoost_Results_Strategy_D
# model_object <- XGBoost_Results_Strategy_D$optimal_model
# 
# ## Train_Data_Base_Model, Train_Data_Strategy_A,
# ## Train_Data_Strategy_B, Train_Data_Strategy_C, Train_Data_Strategy_D
# data_input <- Train_Data_Strategy_D
# 
# # Ensure directory exists
# Directory <- file.path(Charts_Directory_Model, model_used_name)
# if(!dir.exists(Directory)) dir.create(Directory)
# 
# #### Feature Names Mapping.
# names(data_input)

}, error = function(e) message(e))

#==== 02B - Settings (revised set up with a loop) =============================#

tryCatch({
  
Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.
Charts_Directory_Model <- file.path(Path, "03_Charts/VAE/XGBoost")

###### Input parameters.
model_used_name_list <- list("BaseModel", "StrategyA", "StrategyB",
                              "StrategyC","StrategyD")

model_object_list <- list(XGBoost_Results_BaseModel$optimal_model,
                           XGBoost_Results_Strategy_A$optimal_model,
                           XGBoost_Results_Strategy_B$optimal_model,
                           XGBoost_Results_Strategy_C$optimal_model,
                           XGBoost_Results_Strategy_D$optimal_model)

data_input_list <- list(Train_Data_Base_Model,
                        Train_Data_Strategy_A,
                        Train_Data_Strategy_B,
                        Train_Data_Strategy_C,
                        Train_Data_Strategy_D)

## Run the loop:

for(model_rep in 1:length(model_used_name_list)){

  model_used_name <- model_used_name_list[[model_rep]]
  model_object <- model_object_list[[model_rep]]
  data_input <- data_input_list[[model_rep]]
  
# Ensure directory exists
Directory <- file.path(Charts_Directory_Model, model_used_name)
if(!dir.exists(Directory)) dir.create(Directory)
    
#==== 02B - Feature Importance ================================================#

tryCatch({
  
  sparse_formula <- as.formula("y ~ . - 1")
  dummy_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  
  real_feature_names <- colnames(dummy_matrix)
  
  print(paste("Original columns:", ncol(data_input) - 1)) 
  print(paste("Expanded features:", length(real_feature_names)))
  
  imp_matrix <- xgb.importance(feature_names = real_feature_names, model = model_object)
  
  p_imp <- xgb.plot.importance(imp_matrix, top_n = 15, measure = "Gain", plot = FALSE)
  feature_map <- c(
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities"
  )
  
  # Apply the mapping safely
  p_imp$Feature_Label <- ifelse(
    p_imp$Feature %in% names(feature_map), 
    feature_map[p_imp$Feature], 
    p_imp$Feature
  )
  
  p_importance <- ggplot(p_imp, aes(x = reorder(Feature_Label, Gain), y = Gain)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = NULL, 
         subtitle = NULL, 
         x = NULL, 
         y = NULL) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 14, color = "black"),
      axis.text.y = element_text(size = 14, color = "black")
    )
  
  print(p_importance)
  
  # 5. Save Plot
  Path_plot <- file.path(Directory, paste0(model_used_name, "_Feature_Importance.png"))
  ggsave(filename = Path_plot, 
         plot = p_importance, width = 8, height = 6)
  
  # 6. Extract Top Drivers (Using readable names for the text output too)
  # We apply the map to the full matrix for the text printout
  top_feature_1_raw <- imp_matrix$Feature[1]
  top_feature_2_raw <- imp_matrix
  
}, error = function(e) message(e))

#==== 02C - Marginal Response (Univariate Partial Dependence) =================#

tryCatch({
  
  # --- 0. Robust Feature Map (Financials + Latent Features) ---
  feature_map <- c(
    # Standard Financials
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities",
    
    # Strategy D: Interactions
    "Gap_Debt_Equity"   = "Solvency Gap",
    "Ratio_Cash_Profit" = "Cash Burn Ratio",
    
    # Strategy A: Latent Features (l1 - l8)
    "l1" = "Latent Dim 1", "l2" = "Latent Dim 2", 
    "l3" = "Latent Dim 3", "l4" = "Latent Dim 4",
    "l5" = "Latent Dim 5", "l6" = "Latent Dim 6", 
    "l7" = "Latent Dim 7", "l8" = "Latent Dim 8",
    
    # Strategy C: Robust Latent Features (dae_l1 - dae_l8)
    "dae_l1" = "Robust Latent 1", "dae_l2" = "Robust Latent 2",
    "dae_l3" = "Robust Latent 3", "dae_l4" = "Robust Latent 4",
    "dae_l5" = "Robust Latent 5", "dae_l6" = "Robust Latent 6",
    "dae_l7" = "Robust Latent 7", "dae_l8" = "Robust Latent 8"
  )
  
  # --- 1. Prepare Data Matrix ---
  # Re-build matrix to ensure it matches the current data_input structure (which might have l1..l8)
  sparse_formula <- as.formula("y ~ . - 1")
  full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  
  # Sample for speed (PDP is slow on full data)
  set.seed(42)
  n_sample <- min(nrow(full_train_matrix), 1000)
  calc_sample_indices <- sample(nrow(full_train_matrix), n_sample) 
  train_matrix_calc <- full_train_matrix[calc_sample_indices, ]
  
  # --- 2. Define Feature List Dynamically ---
  # If we switched models, the old 'all_features_list' might be stale. Re-calculate.
  if(exists("model_object")) {
    imp <- xgb.importance(model = model_object)
    # Use top 20 features to avoid generating hundreds of plots for latent models
    all_features_list <- head(imp$Feature, 20) 
  } else {
    all_features_list <- colnames(train_matrix_calc)
  }
  
  # --- 3. Wrapper Functions ---
  pred_wrapper <- function(object, newdata) {
    return(predict(object, as.matrix(newdata)))
  }
  
  plot_pdp_bars <- function(feature_name) {
    
    # A. Display Name Lookup (Safe Fallback)
    display_name <- feature_name
    if(feature_name %in% names(feature_map)) {
      display_name <- feature_map[[feature_name]]
    }
    
    # B. Safety Check (Crucial for Strategy A vs Base mismatches)
    if(!feature_name %in% colnames(train_matrix_calc)) {
      return(NULL)
    }
    
    # C. Calculate PDP
    pdp_data <- partial(
      model_object, 
      pred.var = feature_name, 
      train = train_matrix_calc, 
      pred.fun = pred_wrapper,
      prob = TRUE,        
      chull = TRUE,
      grid.resolution = 20, 
      progress = "none" 
    )
    
    # D. Plot
    p <- ggplot(pdp_data, aes(x = .data[[feature_name]], y = yhat)) +
      geom_col(fill = "steelblue", width = 0.8) + 
      labs(title = display_name,
           subtitle = NULL,
           y = "Pred. Default Prob.", # (%) removed to avoid confusion if scale is 0-1
           x = NULL) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(size = 14, color = "black"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 14, color = "black"),
        plot.title = element_text(size = 16, face = "bold")
      )
    
    return(p)
  }
  
  # --- 4. Execution Loop ---
  message("--- Generating PDP Batches ---")
  
  # Intersect ensures we only plot features that actually exist in the matrix
  valid_features <- intersect(all_features_list, colnames(train_matrix_calc))
  
  plot_list <- list()
  
  for(feat in valid_features) {
    # Skip one-hot encoded categorical dummies if desired (e.g., "sectorconstruction") 
    # as they make messy PDPs. Optional check:
    # if(grepl("sector|size", feat)) next 
    
    p <- plot_pdp_bars(feat)
    if(!is.null(p)) {
      plot_list[[length(plot_list) + 1]] <- p
    }
  }
  
  # --- 5. Save in Groups of 4 ---
  if(length(plot_list) > 0) {
    num_plots <- length(plot_list)
    plots_per_page <- 4
    num_pages <- ceiling(num_plots / plots_per_page)
    
    for(i in 1:num_pages) {
      
      start_idx <- (i - 1) * plots_per_page + 1
      end_idx   <- min(i * plots_per_page, num_plots)
      
      current_batch <- plot_list[start_idx:end_idx]
      current_batch <- current_batch[!sapply(current_batch, is.null)]
      
      if(length(current_batch) > 0) {
        # Use arrangeGrob (gridExtra)
        combined_plot <- arrangeGrob(grobs = current_batch, ncol = 2, nrow = 2)
        
        file_name <- paste0(model_used_name, "MarginalResponse_Batch_", i, ".png")
        Path_plot <- file.path(Directory, file_name)
        
        ggsave(filename = Path_plot, plot = combined_plot, width = 12, height = 8)
        message(paste("Saved:", file_name))
      }
    }
  } else {
    message("No valid plots generated. Check if 'all_features_list' matches 'data_input' columns.")
  }
  
}, error = function(e) message("PDP Batch Error: ", e))

#==== 02D - Calibration Charts (Loop for Top Features) ========================#

tryCatch({
  
  # --- 0. Setup: Colors & Map ---
  col_predicted <- "#377EB8" # Steel Blue
  col_observed  <- "#7F7F7F" # Dark Grey
  
  feature_map <- c(
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities"
  )
  
  # --- 1. Preparation: Generate Global Predictions ---
  # We need predictions for the whole dataset to bin them
  if(!exists("full_train_matrix")) {
    sparse_formula <- as.formula("y ~ . - 1")
    full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  }
  
  message("Generating global predictions for calibration...")
  global_preds <- predict(model_object, full_train_matrix)
  
  # Create analysis dataframe
  analysis_data <- data_input %>%
    mutate(
      y_actual = as.numeric(as.character(y)),
      y_pred   = global_preds
    )
  
  # --- 2. Plotting Function (Batched Style) ---
  create_calib_plot <- function(feature_name) {
    
    # A. Name Lookup
    display_name <- feature_name
    if(feature_name %in% names(feature_map)) {
      display_name <- feature_map[[feature_name]]
    }
    
    # Validation
    if(!feature_name %in% colnames(analysis_data)) return(NULL)
    
    # B. Data Aggregation (Bin by Feature Value)
    # We use .data[[String]] to handle spaces or special chars safely
    calib_data <- analysis_data %>%
      select(val = all_of(feature_name), y_actual, y_pred) %>%
      mutate(bin = ntile(val, 10)) %>% # Deciles
      group_by(bin) %>%
      summarise(
        mean_prob     = mean(y_pred),   # Model
        observed_rate = mean(y_actual), # Reality
        .groups = 'drop'
      )
    
    # C. Reshape for Side-by-Side Bars
    plot_data <- calib_data %>%
      pivot_longer(cols = c("mean_prob", "observed_rate"), 
                   names_to = "Type", values_to = "Rate") %>%
      mutate(Type = factor(Type, levels = c("mean_prob", "observed_rate"),
                           labels = c("Predicted", "Observed")))
    
    # D. Visualization
    p <- ggplot(plot_data, aes(x = factor(bin), y = Rate, fill = Type)) +
      
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      
      # Colors
      scale_fill_manual(values = c("Predicted" = col_predicted, 
                                   "Observed"  = col_observed)) + 
      
      # Y-Axis: Percent
      scale_y_continuous(labels = scales::percent, 
                         expand = expansion(mult = c(0, 0.15))) + 
      
      # Labels (Clean, no Subtitle)
      labs(
        title = paste0(display_name), # e.g., "Net Profit"
        x = "Decile (Low -> High)",
        y = "Pred. Default Prob. (%)",
        fill = "" 
      ) +
      
      theme_minimal() +
      theme(
        # The Requested Theme
        axis.text.x = element_text(size = 14, color = "black"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 14, color = "black"),
        plot.title  = element_text(size = 16, face = "bold"),
        
        # Legend styling
        legend.position = "top",
        legend.text = element_text(size = 12),
        
        # Clean up grid
        panel.grid.major.x = element_blank()
      )
    
    return(p)
  }
  
  # --- 3. Loop and Batch Generation ---
  message("--- Generating Calibration Batches ---")
  
  # Use the importance list if available, otherwise all numeric columns
  if(!exists("all_features_list")) {
    all_features_list <- names(select(data_input, where(is.numeric)))
  }
  
  valid_features <- intersect(all_features_list, colnames(analysis_data))
  
  # Generate Plot List
  plot_list <- list()
  
  for(feat in valid_features) {
    p <- create_calib_plot(feat)
    if(!is.null(p)) {
      plot_list[[length(plot_list) + 1]] <- p
    }
  }
  
  # --- 4. Save in Groups of 4 ---
  if(length(plot_list) > 0) {
    num_plots <- length(plot_list)
    plots_per_page <- 4
    num_pages <- ceiling(num_plots / plots_per_page)
    
    for(i in 1:num_pages) {
      
      start_idx <- (i - 1) * plots_per_page + 1
      end_idx   <- min(i * plots_per_page, num_plots)
      
      current_batch <- plot_list[start_idx:end_idx]
      current_batch <- current_batch[!sapply(current_batch, is.null)]
      
      if(length(current_batch) > 0) {
        
        combined_plot <- arrangeGrob(grobs = current_batch, ncol = 2, nrow = 2)
        
        file_name <- paste0(model_used_name, "Calibration_Batch_", i, ".png")
        Path_plot <- file.path(Directory, file_name)
        
        ggsave(filename = Path_plot, plot = combined_plot, width = 12, height = 8)
        message(paste("Saved:", file_name))
      }
    }
  } else {
    message("No valid calibration plots generated.")
  }
  
}, error = function(e) message("Calibration Batch Error: ", e))

#==== 02E - Bivariate Interaction (Hexagonal Binning) =========================#

tryCatch({
  
  # --- 0. Setup ---
  feature_map <- c(
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities"
  )
  
  if(!exists("full_train_matrix")) {
    sparse_formula <- as.formula("y ~ . - 1")
    full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  }
  
  preds <- predict(model_object, full_train_matrix)
  
  # --- 1. Define Beamer-Optimized Plotting Function ---
  plot_beamer_hex <- function(feat_x, feat_y) {
    
    # Name Lookup
    name_x <- ifelse(feat_x %in% names(feature_map), feature_map[[feat_x]], feat_x)
    name_y <- ifelse(feat_y %in% names(feature_map), feature_map[[feat_y]], feat_y)
    
    # Extract Data
    if (!feat_x %in% colnames(full_train_matrix) | !feat_y %in% colnames(full_train_matrix)) {
      return(NULL)
    }
    
    plot_df <- data.frame(
      X_Val = full_train_matrix[, feat_x],
      Y_Val = full_train_matrix[, feat_y],
      Prob  = preds
    )
    
    # --- The Plot ---
    p <- ggplot(plot_df, aes(x = X_Val, y = Y_Val, z = Prob)) +
      
      # 1. Add Borders: 'color = "grey90"' makes the hex shape visible even if the fill is white
      stat_summary_hex(fun = mean, bins = 35, color = "grey92", size = 0.1) + 
      
      # 2. Beamer-Safe Color Scale (White -> Yellow -> Orange -> Red -> Dark Red)
      # This ensures low-risk is still white, but medium risk pops as Yellow/Orange
      scale_fill_gradientn(
        colours = c("#FFFFFF", "#FFEDA0", "#FEB24C", "#F03B20", "#800026"),
        name = "Prob %",
        labels = scales::percent
      ) +
      
      # Clean Layout
      labs(title = NULL, subtitle = NULL, 
           x = name_x, y = name_y) +
      
      theme_minimal() +
      theme(
        # Increase text size slightly more for Beamer readability
        axis.text.x = element_text(size = 14, color = "black", face = "bold"),
        axis.text.y = element_text(size = 14, color = "black", face = "bold"),
        axis.title  = element_text(size = 15, face = "bold"),
        
        # Move legend to top to save width on slides
        legend.position = "right", 
        legend.title = element_text(size = 12, face = "bold"),
        
        # Important: Ensure the plot background is distinct from the white hexes
        # Using a very light grey for the panel helps the white hexes stand out
        panel.background = element_rect(fill = "#F5F5F5", color = NA),
        plot.background = element_rect(fill = "white", color = NA)
      )
    
    return(p)
  }
  
  # --- 2. Execution Loop ---
  
  if(!exists("all_features_list")) {
    imp <- xgb.importance(model = model_object)
    all_features_list <- imp$Feature
  }
  
  # Top 5 Features (10 Plots)
  top_n_features <- head(all_features_list[all_features_list %in% colnames(data_input)], 5)
  
  message(paste("Generating Beamer-Optimized Maps for Top", length(top_n_features), "features..."))
  
  for(i in 1:(length(top_n_features)-1)) {
    for(j in (i+1):length(top_n_features)) {
      
      f1 <- top_n_features[i]
      f2 <- top_n_features[j]
      
      p <- plot_beamer_hex(f1, f2)
      
      if(!is.null(p)) {
        file_name <- paste0(model_used_name, "Beamer_Hex_", f1, "_", f2, ".png")
        Path_plot <- file.path(Directory, file_name)
        
        # Saving with slightly larger dimensions for clarity
        ggsave(filename = Path_plot, plot = p, width = 9, height = 7)
        message(paste("Saved:", file_name))
      }
    }
  }
  
}, error = function(e) message("Beamer Plot Error: ", e))

#==== 02F - Residual diagnostics ==============================================#

tryCatch({
  
  # --- 0. Setup: Colors & Map ---
  col_tn <- "#619CFF" # Blue (Safe/True Negative)
  col_fp <- "#FFA500" # Orange (False Positive/False Alarm)
  col_fn <- "#F8766D" # Red (False Negative/Missed Risk)
  col_tp <- "#7F7F7F" # Grey (True Positive/Caught Risk)
  
  feature_map <- c(
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities",
    "Gap_Debt_Equity"   = "Solvency Gap",
    "Ratio_Cash_Profit" = "Cash Burn Ratio"
  )
  
  # --- 1. Prepare Data & Predictions ---
  sparse_formula <- as.formula("y ~ . - 1")
  X_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  label_vec <- as.numeric(as.character(data_input$y))
  
  pred_probs <- predict(model_object, X_matrix)
  
  # Determine Optimal Threshold
  roc_obj <- roc(label_vec, pred_probs, quiet = TRUE)
  best_threshold_obj <- coords(roc_obj, "best", ret = "threshold", transpose = TRUE)
  best_cutoff <- as.numeric(best_threshold_obj[1])
  
  # Create Diagnosis DataFrame
  Diagnosis_DF <- data_input %>%
    mutate(
      y_num = label_vec,
      xgb_prob = pred_probs,
      Prediction_Class = ifelse(xgb_prob > best_cutoff, 1, 0),
      Error_Type = case_when(
        y_num == 1 & Prediction_Class == 1 ~ "True Positive",
        y_num == 0 & Prediction_Class == 0 ~ "True Negative",
        y_num == 0 & Prediction_Class == 1 ~ "False Positive",
        y_num == 1 & Prediction_Class == 0 ~ "False Negative"
      ),
      # Set Factor Level Order for Plotting
      Error_Type = factor(Error_Type, levels = c("True Negative", "False Positive", "False Negative", "True Positive"))
    )
  
  # --- 2. Plotting Function (Single Feature Boxplot) ---
  plot_error_boxplot <- function(feature_name) {
    
    # Name Lookup
    display_name <- feature_name
    if(feature_name %in% names(feature_map)) {
      display_name <- feature_map[[feature_name]]
    }
    
    # Check existence
    if(!feature_name %in% colnames(Diagnosis_DF)) return(NULL)
    
    # Plot
    p <- ggplot(Diagnosis_DF, aes(x = Error_Type, y = .data[[feature_name]], fill = Error_Type)) +
      
      geom_boxplot(outlier.alpha = 0.2, outlier.size = 0.5, outlier.colour = "grey50") +
      
      # Colors
      scale_fill_manual(values = c(
        "True Negative"  = col_tn,
        "False Positive" = col_fp,
        "False Negative" = col_fn,
        "True Positive"  = col_tp
      )) +
      
      # Zoom to remove extreme outliers (Adjust limits as needed, e.g., -3 to 3 Z-score)
      coord_cartesian(ylim = c(-3, 3)) + 
      
      labs(
        title = display_name,
        x = NULL,
        y = NULL # "Normalized Value" implies Y-axis, removed for cleaner batch view
      ) +
      
      theme_minimal() +
      theme(
        legend.position = "none", # Legend will be shared or implied by color
        plot.title = element_text(face = "bold", size = 14),
        axis.text.x = element_blank(), # Remove X labels to save space (rely on color)
        axis.text.y = element_text(size = 10, color = "black")
      )
    
    return(p)
  }
  
  # --- 3. Loop and Batch Generation ---
  message("--- Generating Error Analysis Boxplots ---")
  
  # Select ALL numeric features
  all_numeric_feats <- names(select(Diagnosis_DF, where(is.numeric)))
  # Exclude the metadata columns we created
  exclude_cols <- c("y_num", "xgb_prob", "Prediction_Class", "y")
  valid_features <- setdiff(all_numeric_feats, exclude_cols)
  
  plot_list <- list()
  
  for(feat in valid_features) {
    p <- plot_error_boxplot(feat)
    if(!is.null(p)) {
      plot_list[[length(plot_list) + 1]] <- p
    }
  }
  
  # --- 4. Save in Groups of 4 ---
  if(length(plot_list) > 0) {
    
    # Create a shared legend for the batch (Dummy Plot)
    dummy_df <- data.frame(Error_Type = factor(c("True Negative", "False Positive", "False Negative", "True Positive"), 
                                               levels = c("True Negative", "False Positive", "False Negative", "True Positive")), 
                           Value = 1)
    
    legend_plot <- ggplot(dummy_df, aes(x=Error_Type, y=Value, fill=Error_Type)) +
      geom_bar(stat="identity") +
      scale_fill_manual(values = c("True Negative" = col_tn, "False Positive" = col_fp, 
                                   "False Negative" = col_fn, "True Positive" = col_tp),
                        name = "") +
      theme_minimal() +
      theme(legend.position = "bottom", legend.text = element_text(size = 12))
    
    # Extract Legend
    get_legend <- function(myggplot){
      tmp <- ggplot_gtable(ggplot_build(myggplot))
      leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
      legend <- tmp$grobs[[leg]]
      return(legend)
    }
    shared_legend <- get_legend(legend_plot)
    
    num_plots <- length(plot_list)
    plots_per_page <- 4
    num_pages <- ceiling(num_plots / plots_per_page)
    
    for(i in 1:num_pages) {
      
      start_idx <- (i - 1) * plots_per_page + 1
      end_idx   <- min(i * plots_per_page, num_plots)
      
      current_batch <- plot_list[start_idx:end_idx]
      current_batch <- current_batch[!sapply(current_batch, is.null)]
      
      if(length(current_batch) > 0) {
        
        # Combine plots into grid
        grid_plots <- arrangeGrob(grobs = current_batch, ncol = 2, nrow = 2)
        
        # Add the shared legend at the bottom
        final_grid <- arrangeGrob(grid_plots, shared_legend, nrow = 2, heights = c(10, 1))
        
        file_name <- paste0(model_used_name, "Error_Boxplot_Batch_", i, ".png")
        Path_plot <- file.path(Directory, file_name)
        
        ggsave(filename = Path_plot, plot = final_grid, width = 10, height = 8)
        message(paste("Saved:", file_name))
      }
    }
  } else {
    message("No numeric features found for boxplots.")
  }
  
}, error = function(e) message("Boxplot Batch Error: ", e))

# --- Forensic Summary Table ---
tryCatch({
  
  # --- 0. Setup: Feature Map for Readability ---
  feature_map <- c(
    "f1"  = "Total Assets",
    "f2"  = "Fixed Assets",
    "f3"  = "Current Assets",
    "f4"  = "Inventories",
    "f5"  = "Cash & Equivalents",
    "f6"  = "Equity",
    "f7"  = "Retained Earnings",
    "f8"  = "Net Profit",
    "f9"  = "Profit Carried Forward",
    "f10" = "Provisions",
    "f11" = "Liabilities",
    "Gap_Debt_Equity"   = "Solvency Gap",
    "Ratio_Cash_Profit" = "Cash Burn Ratio"
  )
  
  # --- 1. Identify Top 5 Features Dynamically ---
  if(exists("model_object")) {
    imp_matrix <- xgb.importance(model = model_object)
    # Filter to ensure features actually exist in the dataframe
    valid_features <- intersect(imp_matrix$Feature, colnames(Diagnosis_DF))
    top_5_features <- head(valid_features, 5)
  } else {
    # Fallback if model object is missing
    top_5_features <- head(names(select(Diagnosis_DF, where(is.numeric))), 5)
  }
  
  message(paste("Generating Forensic Summary for Top 5 Drivers:", paste(top_5_features, collapse=", ")))
  
  # --- 2. Dynamic Summarization ---
  Error_Summary <- Diagnosis_DF %>%
    group_by(Error_Type) %>%
    summarise(
      Count = n(),
      # Dynamically calculate Median for the Top 5 features
      across(all_of(top_5_features), \(x) median(x, na.rm = TRUE), .names = "Median_{.col}")
    ) %>%
    # Sort logically: Safe -> False Alarm -> Missed Risk -> Caught Risk
    arrange(match(Error_Type, c("True Negative", "False Positive", "False Negative", "True Positive")))
  
  # --- 3. Rename Columns to English (Optional but recommended) ---
  # This loop replaces "Median_f8" with "Median_Net Profit"
  current_names <- colnames(Error_Summary)
  new_names     <- current_names
  
  for(i in seq_along(current_names)) {
    # Extract the raw feature code (e.g., remove "Median_" prefix)
    col_name <- current_names[i]
    if(startsWith(col_name, "Median_")) {
      raw_feat <- sub("Median_", "", col_name)
      if(raw_feat %in% names(feature_map)) {
        new_names[i] <- paste0("Median_", feature_map[[raw_feat]])
      }
    }
  }
  colnames(Error_Summary) <- new_names
  
  print("--- Forensic Feature Summary (Top 5 Drivers) ---")
  print(Error_Summary)
  
  write.xlsx(x = Error_Summary,
             file = file.path(Directory, "Error_Summary.xlsx"))
  
}, error = function(e) message("Error Summary Failed: ", e))
  
}

#==============================================================================#
  
}, error = function(e) message(e))

#==== 02F - Density Plot (NOT IN USE CURRENTLY) ===============================#

tryCatch({
  
  # plot_data <- data_input %>%
  #   select(y, Specialist_Risk_Score ) %>%
  #   mutate(Status = ifelse(y == "1", "Default", "Non-Default"))
  # 
  # p_dens <- ggplot(plot_data, aes(x = log(Specialist_Risk_Score), fill = Status)) +
  #   geom_density(alpha = 0.6) +
  #   scale_fill_manual(values = c("Non-Default" = "#00BFC4", "Default" = "#F8766D")) +
  #   labs(title = paste("Global Separation: Anomaly Score Distribution (", model_used_name, ")", sep=""),
  #        subtitle = "If the Red peak is to the right of the Blue peak, the VAE works.",
  #        x = "Log(Specialist_Risk_Score)", y = "Density") +
  #   theme_minimal()
  # 
  # print(p_dens)
  # 
  # # Save Chart
  # Path_plot <- file.path(Directory, "Anomaly_Score_Density.png")
  # ggsave(filename = Path_plot, plot = p_dens, width = 8, height = 6)
  # 
  # 
  
}, error = function(e) message("Density Plot Error: ", e))

#==============================================================================#
#==== 03 - XGBoost Model in the test-set ======================================#
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
Test_Data_Strategy_C <- Strategy_C_Test

### Strategy D: Manual feature engineering.
Test_Data_Strategy_D <- Strategy_D_Test

}, error = function(e) message(e))

#==== 03B - Base Model ========================================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Model <- XGBoost_Results_BaseModel$optimal_model
Test_Data <- Test_Data_Base_Model

##==============================##
## Code.
##==============================##

XGBoost_Test_Results_BaseModel <- XGBoost_Test(Model = Model, 
                                               Test_Data = Test_Data)

}, error = function(e) message(e))

#==== 03C - Strategy A: Dimensionality Reduction ==============================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Model <- XGBoost_Results_Strategy_A$optimal_model
Test_Data <- Test_Data_Strategy_A

##==============================##
## Code.
##==============================##

XGBoost_Test_Results_Strategy_A <- XGBoost_Test(Model = Model, 
                                                Test_Data = Test_Data)

}, error = function(e) message(e))

#==== 03D - Strategy B: Anomaly Score =========================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Model <- XGBoost_Results_Strategy_B$optimal_model
Test_Data <- Test_Data_Strategy_B

##==============================##
## Code.
##==============================##

XGBoost_Test_Results_Strategy_B <- XGBoost_Test(Model = Model, 
                                                Test_Data = Test_Data)

}, error = function(e) message(e))

#==== 03E - Strategy C: Feature Denoising =====================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##

Model <- XGBoost_Results_Strategy_C$optimal_model
Test_Data <- Test_Data_Strategy_C

##==============================##
## Code.
##==============================##

XGBoost_Test_Results_Strategy_C <- XGBoost_Test(Model = Model, 
                                                Test_Data = Test_Data)

}, error = function(e) message(e))

#==== 03F - Strategy D: Manual feature engineering ============================#

tryCatch({
  
  ##==============================##
  ## Parameters.
  ##==============================##
  
  Model <- XGBoost_Results_Strategy_D$optimal_model
  Test_Data <- Test_Data_Strategy_D
  
  ##==============================##
  ## Code.
  ##==============================##
  
  XGBoost_Test_Results_Strategy_D <- XGBoost_Test(Model = Model, 
                                                  Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==============================================================================#
#==== 04 - XGBoost Test Comparison (AUC and Parameters) =======================#
#==============================================================================#

#==== 04A - Compare the AUC-Score =============================================#

tryCatch({
  
Final_Leaderboard <- bind_rows(
  XGBoost_Test_Results_BaseModel$Metrics %>% mutate(Strategy = "Base Model"),
  XGBoost_Test_Results_Strategy_A$Metrics %>% mutate(Strategy = "Strategy A (Dim. Reduction)"),
  XGBoost_Test_Results_Strategy_B$Metrics %>% mutate(Strategy = "Strategy B (Anomaly Score)"),
  XGBoost_Test_Results_Strategy_C$Metrics %>% mutate(Strategy = "Strategy C (Feature Denoising)"),
  XGBoost_Test_Results_Strategy_D$Metrics %>% mutate(Strategy = "Strategy D (Manual Feature Eng.)")
) %>%
  select(Strategy, AUC, Brier_Score, Penalized_Brier_Score ,Log_Loss, Inference_Time_Sec) %>%
  arrange(desc(AUC))

base_auc <- Final_Leaderboard$AUC[Final_Leaderboard$Strategy == "Base Model"]

Final_Leaderboard <- Final_Leaderboard %>%
  mutate(
    Uplift_AUC_pct = round(((AUC - base_auc) / base_auc) * 100, 3),
    Winner = ifelse(AUC == max(AUC), "**WINNER**", "")
  )

print("=======================================================")
print("               FINAL MODEL LEADERBOARD                 ")
print("=======================================================")
print(Final_Leaderboard)

p_results <- ggplot(Final_Leaderboard, aes(x = reorder(Strategy, AUC), y = AUC, fill = Strategy)) +
  geom_col(width = 0.6) +
  coord_flip() +
  geom_text(aes(label = paste0(round(AUC, 4), " (", Uplift_AUC_pct, "%)")), 
            hjust = -0.1, fontface = "bold") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Final Impact Analysis: Did the VAE features help?",
       subtitle = "Comparing Test Set AUC across all strategies",
       y = "Test AUC", x = "") +
  theme_minimal() +
  theme(legend.position = "none")

print(p_results)

}, error = function(e) message(e))

#==== 04B - Residual diagnostics ==============================================#

tryCatch({
  
  data_input <- Test_Data_Base_Model
  data_input_C <- Test_Data_Strategy_C
  
  model_object <- XGBoost_Results_BaseModel$optimal_model
  model_object_C <- XGBoost_Results_Strategy_C$optimal_model
  
  data_input <- data_input_C
  model_object <- model_object_C
  
  # 1. Prepare Data (Strictly separating features)
  # We remove 'y' to ensure the matrix only contains predictors
  label_vec <- as.numeric(as.character(data_input$y))
  data_features <- data_input %>% select(-y)
  
  # 2. Create Matrix
  # Note: We use the exact same formula structure (~ . -1) to generate dummies
  X_matrix <- sparse.model.matrix(~ . - 1, data = data_features)
  
  # 3. Generate Raw Probabilities
  pred_probs <- predict(model_object, X_matrix)
  
  # --- DEBUG: CHECK PROBABILITY DISTRIBUTION ---
  # You will likely see the Max is < 0.5
  print("--- Probability Distribution ---")
  print(summary(pred_probs))
  
  # 4. Find the "Best" Threshold (Youden's J Statistic)
  # This finds the point on the ROC curve that balances Sensitivity and Specificity
  roc_obj <- roc(label_vec, pred_probs, quiet = TRUE)
  best_threshold_obj <- coords(roc_obj, "best", ret = "threshold", transpose = TRUE)
  best_cutoff <- best_threshold_obj[1] # Extract the numeric value
  
  print(paste("Optimal Decision Threshold:", round(best_cutoff, 4)))
  
  # 5. Create Diagnosis DF with Optimal Threshold
  Diagnosis_DF <- data_input %>%
    mutate(
      y_num = label_vec,
      xgb_prob = pred_probs,
      
      # CRITICAL CHANGE: Use the calculated best_cutoff instead of 0.5
      Prediction_Class = ifelse(xgb_prob > best_cutoff, 1, 0),
      
      # Classification Logic
      Error_Type = case_when(
        y_num == 1 & Prediction_Class == 1 ~ "True Positive",
        y_num == 0 & Prediction_Class == 0 ~ "True Negative",
        y_num == 0 & Prediction_Class == 1 ~ "False Positive",
        y_num == 1 & Prediction_Class == 0 ~ "False Negative"
      )
    )
  
  # 6. Verify Result
  print("--- Revised Confusion Matrix ---")
  print(table(Diagnosis_DF$Error_Type))
  
  # 1. Prepare the Diagnostic Data
  # We classify every firm into one of 4 buckets: TP, TN, FP, FN.
  # Threshold = 0.5 (Standard cutoff, adjust if your threshold is different)
  
  Residual_Analysis <- Diagnosis_DF %>%
    mutate(
      Prediction_Class = ifelse(xgb_prob > 0.5, 1, 0),
      Error_Type = case_when(
        y_num == 1 & Prediction_Class == 1 ~ "True Positive (Correct Catch)",
        y_num == 0 & Prediction_Class == 0 ~ "True Negative (Correct Safe)",
        y_num == 0 & Prediction_Class == 1 ~ "False Positive (False Alarm)",
        y_num == 1 & Prediction_Class == 0 ~ "False Negative (Missed Risk)"
      )
    )
  
  features_to_plot <- c("f8", "Gap_Debt_Equity", "Ratio_Cash_Profit")
  
  # 2. Reshape Data for Plotting (Long Format)
  Residual_Long <- Diagnosis_DF %>%
    select(Error_Type, all_of(features_to_plot)) %>%
    pivot_longer(cols = all_of(features_to_plot), names_to = "Feature", values_to = "Value")
  
  #==============================================================================#
  #==== PLOT A: THE "STEALTH DEFAULTER" ANALYSIS (Actual Defaulters) ===========#
  #==============================================================================#
  # Focus: Why did we miss the False Negatives? 
  # Comparing: Missed Risks (FN) vs. Caught Risks (TP)
  
  Plot_Data_Defaulters <- Residual_Long %>%
    filter(Error_Type %in% c("False Negative", "True Positive"))
  
  p1 <- ggplot(Plot_Data_Defaulters, aes(x = Value, fill = Error_Type)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~ Feature, scales = "free", ncol = 1) +
    scale_fill_manual(values = c(
      "False Negative" = "#F8766D",  # Red (The Missed Risk / Stealth)
      "True Positive"  = "#00BA38"   # Green (The Caught Risk / Obvious)
    )) +
    labs(
      title = "Plot A: Analysis of Actual Defaulters",
      subtitle = "Comparing 'Stealth Defaulters' (Missed) vs. 'Obvious Defaulters' (Caught)",
      y = "Density", x = "Feature Value"
    ) +
    theme_minimal() +
    theme(legend.position = "top") + 
    # CRITICAL: Zoom in to remove extreme outliers for better visibility
    coord_cartesian(xlim = c(-3, 3)) 
  
  print(p1)
  
  #==============================================================================#
  #==== PLOT B: THE "HEALTHY LOSER" ANALYSIS (Actual Survivors) ================#
  #==============================================================================#
  # Focus: Why did we flag the False Positives?
  # Comparing: False Alarms (FP) vs. Normal Safe Firms (TN)
  
  Plot_Data_Survivors <- Residual_Long %>%
    filter(Error_Type %in% c("False Positive", "True Negative"))
  
  p2 <- ggplot(Plot_Data_Survivors, aes(x = Value, fill = Error_Type)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~ Feature, scales = "free", ncol = 1) +
    scale_fill_manual(values = c(
      "False Positive" = "#FFA500",  # Orange (False Alarm / Healthy Loser)
      "True Negative"  = "#619CFF"   # Blue (The Normal Safe)
    )) +
    labs(
      title = "Plot B: Analysis of Actual Survivors",
      subtitle = "Comparing 'False Alarms' (Risky Profile) vs. 'Safe Profile' (Normal)",
      y = "Density", x = "Feature Value"
    ) +
    theme_minimal() +
    theme(legend.position = "top") + 
    coord_cartesian(xlim = c(-3, 3))
  
  print(p2) 
  
  # 5. Summary Table (The Quantified "Why")
  # Calculate the median value of each feature for each error group.
  Error_Summary <- Diagnosis_DF %>%
    group_by(Error_Type) %>%
    summarise(
      Count = n(),
      # Use na.rm = TRUE to be safe against any missing values
      Median_Profit_f8    = median(f8, na.rm = TRUE),
      Median_Solvency_Gap = median(Gap_Debt_Equity, na.rm = TRUE),
      Median_Burn_Ratio   = median(Ratio_Cash_Profit, na.rm = TRUE)
    ) %>%
    # Arrange logically: Missed Risk -> Caught Risk -> False Alarm -> Safe
    arrange(factor(Error_Type, levels = c("False Negative", "True Positive", "False Positive", "True Negative")))
  
  print("--- Forensic Summary of Model Errors ---")
  print(Error_Summary)
  
}, error = function(e) message(e))

#==============================================================================#
#==============================================================================#
#==============================================================================#