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
# Train_Data_Strategy_B <- Strategy_B_AS
Train_Data_Strategy_B <- Strategy_B_AS
# Train_Data_Strategy_B <- Train_Data_Strategy_B %>%
#   mutate(
#     id = Train_with_id$id) 

### Strategy C: Manual Feature engineering.
Train_Data_Strategy_C <- Strategy_B_AS_revised
# Train_Data_Strategy_C <- Train_Data_Strategy_C %>%
#   mutate(
#     id = Train_with_id$id) 

### Strategy D: fitting on the residuals of the base model.
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

##==============================##
## Add the predicted prob. and the brier score.
##==============================##

model <- XGBoost_Results_BaseModel$optimal_model

sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)

df_results <- tibble::tibble(
  Actual = as.numeric(as.character(Train_Data$y)), 
  Predicted = preds_prob_D
)

brier_score <- mean((df_results$Predicted - df_results$Actual)^2)

XGBoost_Results_BaseModel$Predictions <- df_results
XGBoost_Results_BaseModel$Brier_Score <- brier_score

##==============================##

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
## Add the predicted prob. and the brier score.
##==============================##

model <- XGBoost_Results_Strategy_A$optimal_model

sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)

df_results <- tibble::tibble(
  Actual = as.numeric(as.character(Train_Data$y)), 
  Predicted = preds_prob_D
)

brier_score <- mean((df_results$Predicted - df_results$Actual)^2)

XGBoost_Results_Strategy_A$Predictions <- df_results
XGBoost_Results_Strategy_A$Brier_Score <- brier_score

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
## Add the predicted prob. and the brier score.
##==============================##

model <- XGBoost_Results_Strategy_B$optimal_model

sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)

df_results <- tibble::tibble(
  Actual = as.numeric(as.character(Train_Data$y)), 
  Predicted = preds_prob_D
)

brier_score <- mean((df_results$Predicted - df_results$Actual)^2)

XGBoost_Results_Strategy_B$Predictions <- df_results
XGBoost_Results_Strategy_B$Brier_Score <- brier_score

##==============================##

}, error = function(e) message(e))

#==== 01D - Strategy C: Manual Feature Engineering ============================#

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
## Add the predicted prob. and the brier score.
##==============================##

model <- XGBoost_Results_Strategy_C$optimal_model

sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)

df_results <- tibble::tibble(
  Actual = as.numeric(as.character(Train_Data$y)), 
  Predicted = preds_prob_D
)

brier_score <- mean((df_results$Predicted - df_results$Actual)^2)

XGBoost_Results_Strategy_C$Predictions <- df_results
XGBoost_Results_Strategy_C$Brier_Score <- brier_score

##==============================##

}, error = function(e) message(e))

#==== 01E - Strategy D: Residuals of the base model ===========================#

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

##==============================##
## Add the predicted prob. and the brier score.
##==============================##

model <- XGBoost_Results_Strategy_D$optimal_model

sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)

df_results <- tibble::tibble(
  Actual = as.numeric(as.character(Train_Data$y)), 
  Predicted = preds_prob_D
)

brier_score <- mean((df_results$Predicted - df_results$Actual)^2)

XGBoost_Results_Strategy_D$Predictions <- df_results
XGBoost_Results_Strategy_D$Brier_Score <- brier_score
    
##==============================##

}, error = function(e) message("Strategy 2 Error: ", e))

#==== 01F - Strategy E: Denoising =============================================#

tryCatch({
  
##==============================##
## Parameters.
##==============================##
  
Data_Train_CV_List <- Data_Train_CV_Base_Model[["fold_list"]]
Train_Data <- Train_Data_Strategy_E
  
##==============================##
## Code.
##==============================##
  
XGBoost_Results_Strategy_E <- XGBoost_Training(Data_Train_CV_List = Data_Train_CV_List,
                                               Train_Data = Train_Data,
                                               n_init_points = n_init_points,
                                               n_iter_bayes = n_iter_bayes)
  
##==============================##
## Add the predicted prob. and the brier score.
##==============================##
  
model <- XGBoost_Results_Strategy_E$optimal_model
  
sparse_formula <- as.formula("y ~ . - 1")
data_matrix  <- sparse.model.matrix(sparse_formula, data = Train_Data)
preds_prob_D   <- predict(model, data_matrix)
  
df_results <- tibble::tibble(
    Actual = as.numeric(as.character(Train_Data$y)), 
    Predicted = preds_prob_D)
  
brier_score <- mean((df_results$Predicted - df_results$Actual)^2)
  
XGBoost_Results_Strategy_E$Predictions <- df_results
XGBoost_Results_Strategy_E$Brier_Score <- brier_score
  
##==============================##
  
}, error = function(e) message("Strategy E Error: ", e))

#==============================================================================#
#==== 02 - XGBoost Model Comparison (AUC and Parameters) ======================#
#==============================================================================#

tryCatch({
  
  extract_metrics <- function(model_obj, model_name) {
    best_auc <- model_obj$results$AUC[1]
    params <- model_obj$optimal_parameters
    brier_val <- if(!is.null(model_obj$Brier_Score)) model_obj$Brier_Score else NA
    data.frame(
      Model = model_name,
      AUC = best_auc,
      Brier_Score = brier_val,  
      Optimal_Rounds = model_obj$optimal_rounds,
      Eta = params$eta,
      Max_Depth = params$max_depth,
      Subsample = params$subsample,
      Colsample = params$colsample_bytree
    )
  }

}, error = function(e) message(e))

#==== 02A - Compare the AUC-Score =============================================#

tryCatch({
  
  comparison_table <- bind_rows(
    extract_metrics(XGBoost_Results_BaseModel, "Base Model"),
    extract_metrics(XGBoost_Results_Strategy_A,  "Strategy A (Latent Only)"),
    extract_metrics(XGBoost_Results_Strategy_B,  "Strategy B (Anomaly Only)"),
    extract_metrics(XGBoost_Results_Strategy_C,  "Strategy C (Regime Features)"),
    extract_metrics(XGBoost_Results_Strategy_D,  "Strategy D (Residual Fit)"),
    extract_metrics(XGBoost_Results_Strategy_E,  "Strategy E (Denoising)")
  ) %>%
    arrange(desc(AUC)) %>% 
    mutate(
      # AUC Uplift (Higher is Better)
      Base_AUC = AUC[Model == "Base Model"],
      Uplift_AUC_pct = ((AUC - Base_AUC) / Base_AUC) * 100,
      
      # Brier Uplift (Lower is Better)
      Base_Brier = Brier_Score[Model == "Base Model"],
      # Formula: (New - Old) / Old
      # A Negative Result means the error decreased (Improvement)
      Uplift_Brier_pct = ((Brier_Score - Base_Brier) / Base_Brier) * 100
    ) %>%
    # Select key columns for the final view
    select(Model, AUC, Uplift_AUC_pct, Brier_Score, Uplift_Brier_pct, Optimal_Rounds, Eta, Max_Depth) 
  
  print("--- Final Model Leaderboard ---")
  print(comparison_table)

}, error = function(e) message(e))

#==== 02B - Settings ==========================================================#

tryCatch({
  
Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.
Charts_Directory_Model <- file.path(Path, "03_Charts/VAE")

###### Input parameters.
## BaseModel, StrategyA, StrategyB or Strategy C, StrategyD, StrategyE
model_used_name <- "Strategy"

## XGBoost_Results_BaseModel, XGBoost_Results_Strategy_A, 
## XGBoost_Results_Strategy_B, XGBoost_Results_Strategy_C, 
## XGBoost_Results_Strategy_D
model_object <- XGBoost_Results_Strategy_C$optimal_model

## Train_Data_Base_Model, Train_Data_Strategy_A, 
## Train_Data_Strategy_B, Train_Data_Strategy_C
data_input <- Train_Data_Strategy_C

# Ensure directory exists
Directory <- file.path(Charts_Directory_Model, model_used_name)
if(!dir.exists(Directory)) dir.create(Directory)

}, error = function(e) message(e))

#==== 02B - Feature Importance ================================================#

tryCatch({
  
  # 1. Re-create the Model Matrix to get the EXACT feature names used
  sparse_formula <- as.formula("y ~ . - 1")
  dummy_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  
  # Get the expanded names (e.g., "sectorWholesale", "sizeSmall")
  real_feature_names <- colnames(dummy_matrix)
  
  print(paste("Original columns:", ncol(data_input) - 1)) 
  print(paste("Expanded features:", length(real_feature_names)))
  
  # 2. Extract Importance
  imp_matrix <- xgb.importance(feature_names = real_feature_names, model = model_object)
  
  # 3. Visualization
  p_imp <- xgb.plot.importance(imp_matrix, top_n = 15, measure = "Gain", plot = FALSE)
  
  p_importance <- ggplot(p_imp, aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = paste("What drives Default Risk? (", model_used_name, ")", sep=""),
         subtitle = "Gain = Contribution to the model's predictive power",
         x = "Feature", y = "Gain") +
    theme_minimal()
  
  print(p_importance)
  
  # Save Chart
  Path_plot <- file.path(Directory, "Feature_Importance.png")
  ggsave(filename = Path_plot, 
         plot = p_importance, width = 8, height = 6)
  
  # 4. Identify Top Features for downstream analysis
  top_feature_1 <- imp_matrix$Feature[1]
  top_feature_2 <- imp_matrix$Feature[2]
  all_features  <- imp_matrix$Feature # List for iteration
  
  print(paste("Top Drivers identified:", top_feature_1, "and", top_feature_2))
  
  # Save to Global Env for next blocks
  assign("top_feature_1", top_feature_1, envir = .GlobalEnv)
  assign("top_feature_2", top_feature_2, envir = .GlobalEnv)
  assign("all_features_list", all_features, envir = .GlobalEnv)
  
}, error = function(e) message(e))

#==== 02C - Marginal Response (Univariate Partial Dependence) =================#

tryCatch({
  
  # 1. Prepare Training Data (Reduced Sample for Speed)
  sparse_formula <- as.formula("y ~ . - 1")
  full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  
  set.seed(42)
  calc_sample_indices <- sample(nrow(full_train_matrix), 1000) 
  train_matrix_calc <- full_train_matrix[calc_sample_indices, ]
  
  # 2. Prediction Wrapper
  pred_wrapper <- function(object, newdata) {
    return(predict(object, as.matrix(newdata)))
  }
  
  # 3. Optimized Plotting Function
  plot_pdp_fast <- function(feature_name) {
    
    message(paste("Calculating PDP for:", feature_name, "..."))
    
    # Check if feature exists in matrix
    if(!feature_name %in% colnames(train_matrix_calc)) {
      message(paste("Skipping", feature_name, "- not in matrix (likely dropped)"))
      return(NULL)
    }
    
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
    
    rug_data <- data.frame(val = train_matrix_calc[, feature_name])
    
    p <- ggplot(pdp_data, aes_string(x = feature_name, y = "yhat")) +
      geom_line(color = "#00BFC4", size = 1.2) +
      geom_rug(data = rug_data, aes(x = val), sides = "b", alpha = 0.1, inherit.aes = FALSE) +
      labs(title = paste("Partial Dependence:", feature_name),
           subtitle = "Marginal effect on Default Risk",
           y = "Predicted Default Prob",
           x = feature_name) +
      theme_minimal()
    
    return(p)
  }
  
  # 4. Generate Plots for Top 2 Features
  p1 <- plot_pdp_fast(top_feature_1)
  p2 <- plot_pdp_fast(top_feature_2)
  
  if(!is.null(p1) & !is.null(p2)) {
    combined_plot <- arrangeGrob(p1, p2, ncol = 2)
    grid.arrange(combined_plot)
    
    # Save Chart
    Path_plot <- file.path(Directory, "PDP_Univariate.png")
    ggsave(filename = Path_plot, 
           plot = combined_plot, width = 10, height = 5)
  }
  
}, error = function(e) message("PDP Error: ", e))

#==== 02D - Calibration Charts (Loop for Top Features) ========================#

tryCatch({
  
  # --- DEFINITIONS ---
  # Define colors LOCALLY to prevent "Object not found" errors
  col_predicted <- "#377EB8" # Steel Blue
  col_observed  <- "#7F7F7F" # Dark Grey
  
  # 1. PREPARATION: Generate Predictions for the whole dataset first
  sparse_formula <- as.formula("y ~ . - 1")
  full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
  
  message("Generating global predictions for the dataset...")
  global_preds <- predict(model_object, full_train_matrix)
  
  # Attach predictions to the dataframe
  analysis_data <- data_input %>%
    mutate(
      y_actual = as.numeric(as.character(y)),
      y_pred   = global_preds
    )
  
  # 2. FUNCTION: Create and Save Feature Bar Chart
  create_feature_bar_chart <- function(feature_name) {
    
    # Validation
    if(!feature_name %in% colnames(analysis_data)) {
      message(paste("Skipping", feature_name, "- Not found in dataframe."))
      return(NULL)
    }
    
    message(paste("Processing Feature Bar Chart:", feature_name))
    
    # A. Data Aggregation (Bin by Feature Value)
    calib_data_xgb <- analysis_data %>%
      select(all_of(feature_name), y_actual, y_pred) %>%
      mutate(
        # Bin 1 = Lowest values of Feature, Bin 10 = Highest values
        bin = ntile(get(feature_name), 10) 
      ) %>%
      group_by(bin) %>%
      summarise(
        mean_prob     = mean(y_pred),   # Model Prediction
        observed_rate = mean(y_actual), # Real Default Rate
        mean_feat_val = mean(get(feature_name)), 
        n             = n()
      )
    
    # B. Reshape for Plotting
    calib_plot_data_xgb <- calib_data_xgb %>%
      select(bin, mean_prob, observed_rate) %>%
      rename(Predicted = mean_prob, Observed = observed_rate) %>%
      pivot_longer(cols = c("Predicted", "Observed"), 
                   names_to = "Type", values_to = "Rate") %>%
      mutate(Type = factor(Type, levels = c("Predicted", "Observed")))
    
    # C. Visualization
    plot_calib_bars <- ggplot(calib_plot_data_xgb, aes(x = factor(bin), y = Rate, fill = Type)) +
      
      geom_col(position = position_dodge(width = 0.8), width = 0.7) +
      
      geom_text(aes(label = scales::percent(Rate, accuracy = 0.1)), 
                position = position_dodge(width = 0.8), 
                vjust = -0.5, 
                size = 3.0, 
                fontface = "bold", 
                color = "black") +
      
      scale_fill_manual(values = c("Predicted" = col_predicted, 
                                   "Observed"  = col_observed)) + 
      
      scale_y_continuous(labels = scales::percent, 
                         expand = expansion(mult = c(0, 0.15))) + 
      
      labs(
        title = paste("Feature Calibration:", feature_name),
        subtitle = paste("Model vs Actuals by Feature Decile (1=Low, 10=High)"),
        x = paste(feature_name, "Decile"),
        y = "Default Rate",
        fill = "" 
      ) +
      
      theme_minimal(base_size = 13) +
      theme(
        plot.title = element_text(face = "bold", size = 16),
        legend.position = "top", 
        axis.title.x = element_text(face = "bold", margin = ggplot2::margin(t = 10)),
        axis.title.y = element_text(face = "bold", margin = ggplot2::margin(r = 10)),
        panel.grid.major.x = element_blank()
      )
    
    print(plot_calib_bars)
    
    # Save
    Path_plot <- file.path(Directory, paste0("BarCalib_", feature_name, ".png"))
    ggsave(filename = Path_plot, 
           plot = plot_calib_bars, width = 10, height = 6)
  }
  
  # 3. Execution Loop
  if(!exists("all_features_list")) {
    imp <- xgb.importance(model = model_object)
    all_features_list <- imp$Feature
  }
  
  valid_features <- all_features_list[all_features_list %in% colnames(data_input)]
  
  # Run for Top 3 Features
  for(feat in head(valid_features, 10)) {
    create_feature_bar_chart(feat)
  }
  
}, error = function(e) message("Feature Bar Chart Error: ", e))

#==== 02E - Bivariate Interaction (Hexagonal Binning) =========================#

# Define features manually here if desired, or use the auto-detected top 2
hex_feature_x <- top_feature_1
hex_feature_y <- top_feature_2

tryCatch({
  
  # 1. Re-create Matrix
  sparse_formula <- as.formula("y ~ . - 1")
  train_matrix_full <- sparse.model.matrix(sparse_formula, data = data_input)
  
  # 2. Generate Predictions (Score every row)
  preds <- predict(model_object, train_matrix_full)
  
  # 3. Extract Vectors
  if (!hex_feature_x %in% colnames(train_matrix_full) | !hex_feature_y %in% colnames(train_matrix_full)) {
    stop("Selected features for Hexbin not found in the matrix.")
  }
  
  x_val <- train_matrix_full[, hex_feature_x]
  y_val <- train_matrix_full[, hex_feature_y]
  
  # 4. Consolidate
  plot_data <- data.frame(
    Feature_X = x_val,
    Feature_Y = y_val,
    Predicted_Prob = preds
  )
  
  avg_risk <- mean(preds)
  
  # 5. Plot
  p_hex <- ggplot(plot_data, aes(x = Feature_X, y = Feature_Y, z = Predicted_Prob)) +
    stat_summary_hex(fun = mean, bins = 35, color = "white", size = 0.1) + 
    scale_fill_gradient2(low = "#00BFC4", mid = "white", high = "#F8766D", 
                         midpoint = avg_risk, 
                         name = "Avg Prob\nof Default") +
    labs(title = paste("Risk Surface:", hex_feature_x, "&", hex_feature_y),
         subtitle = paste("Interaction Map (", model_used_name, ")", sep=""),
         x = hex_feature_x, 
         y = hex_feature_y) +
    theme_minimal() +
    theme(legend.position = "right")
  
  print(p_hex)
  
  # Save
  Path_plot <- file.path(Directory, "Hexbin_Interaction.png")
  ggsave(filename = Path_plot, 
         plot = p_hex, width = 8, height = 6)
  
}, error = function(e) message("Hexbin Plot Error: ", e))

#==== 02F - Density Plot ======================================================#

### DO NOT RUN.

tryCatch({
  
    plot_data <- data_input %>%
      select(y, Specialist_Risk_Score ) %>%
      mutate(Status = ifelse(y == "1", "Default", "Non-Default"))
    
    p_dens <- ggplot(plot_data, aes(x = log(Specialist_Risk_Score), fill = Status)) +
      geom_density(alpha = 0.6) +
      scale_fill_manual(values = c("Non-Default" = "#00BFC4", "Default" = "#F8766D")) +
      labs(title = paste("Global Separation: Anomaly Score Distribution (", model_used_name, ")", sep=""),
           subtitle = "If the Red peak is to the right of the Blue peak, the VAE works.",
           x = "Log(Specialist_Risk_Score)", y = "Density") +
      theme_minimal()
    
    print(p_dens)
    
    # Save Chart
    Path_plot <- file.path(Directory, "Anomaly_Score_Density.png")
    ggsave(filename = Path_plot, plot = p_dens, width = 8, height = 6)
  
  
  
}, error = function(e) message("Density Plot Error: ", e))

#==== 02G - Residual diagnostics ==============================================#

tryCatch({
  
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
Test_Data_Strategy_C <- Strategy_B_AS_Test_revised

### Strategy D: residual fit
# Test_Data_Strategy_D <- Strategy_D_Test_Soft

### Strategy E: Denoising
# Test_Data_Strategy_E <- Test_Data_Strategy_E

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

XGBoost_Test_Results_BaseModel$Metrics

}, error = function(e) message(e))

#==== 03C - Strategy A: Latent features =======================================#

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

#==== 03E - Strategy C: Manual Feature Engineering ============================#

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

#==== 03F - Strategy D: Regime Switching ======================================#

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

#==== 03G - Strategy E: Denoising =============================================#

tryCatch({
  
  ##==============================##
  ## Parameters.
  ##==============================##
  
  Model <- XGBoost_Results_Strategy_E$optimal_model
  Test_Data <- Test_Data_Strategy_E
  
  ##==============================##
  ## Code.
  ##==============================##
  
  XGBoost_Test_Results_Strategy_E <- XGBoost_Test(Model = Model, 
                                                  Test_Data = Test_Data)
  
}, error = function(e) message(e))

#==============================================================================#
#==== 04 - XGBoost Test Comparison (AUC and Parameters) =======================#
#==============================================================================#

#==== 04A - Compare the AUC-Score =============================================#

tryCatch({
  
Final_Leaderboard <- bind_rows(
  XGBoost_Test_Results_BaseModel$Metrics %>% mutate(Strategy = "Base Model"),
  # XGBoost_Test_Results_Strategy_A$Metrics %>% mutate(Strategy = "Strategy A (Latent)"),
  # XGBoost_Test_Results_Strategy_B$Metrics %>% mutate(Strategy = "Strategy B (Anomaly)"),
  XGBoost_Test_Results_Strategy_C$Metrics %>% mutate(Strategy = "Strategy C (Manual.Feature.Eng)"),
  # XGBoost_Test_Results_Strategy_D$Metrics %>% mutate(Strategy = "Strategy D (Residual Fit)"),
  # XGBoost_Test_Results_Strategy_E$Metrics %>% mutate(Strategy = "Strategy E (Denoising)")
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