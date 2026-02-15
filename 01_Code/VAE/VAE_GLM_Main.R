#==============================================================================#
#==== 00 - Data Preparation ===================================================#
#==============================================================================#

Path <- dirname(this.path::this.path())
setwd(Path)

source("GLM_Training.R")
source("GLM_Test.R")

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

#==== 01D - Strategy C: Feature Denoising =====================================#

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

#==== 01E - Strategy D: Manual Feature Engineering ============================#

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

#==============================================================================#
#==== 02 - GLM Model Comparison (AUC and Parameters) ==========================#
#==============================================================================#

tryCatch({
  
  extract_metrics <- function(model_obj, model_name) {
    
    # --- Safety Check: Handle Missing/Failed Models ---
    if (is.null(model_obj) || is.null(model_obj$results)) {
      warning(paste("Model", model_name, "is missing. Returning NA row."))
      return(data.frame(
        Model = model_name, 
        Type = "Error",      # Fixed: Match 'Type' below
        AUC = NA, 
        Brier_Score = NA, 
        Penalized_Brier = NA, 
        Alpha = NA, Lambda = NA, 
        Rounds = NA, Eta = NA, Max_Depth = NA
      ))
    }
    
    # 1. Standard Metrics
    # Note: Ensure 'results' is sorted by AUC descending in your training function
    best_auc      <- model_obj$results$AUC[1]
    brier_val     <- if(!is.null(model_obj$Brier_Score)) model_obj$Brier_Score else NA
    pen_brier_val <- if(!is.null(model_obj$Penalized_Brier_Score)) model_obj$Penalized_Brier_Score else NA
    
    # 2. Parameter Extraction
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
      rounds     <- if(!is.null(model_obj$optimal_rounds)) model_obj$optimal_rounds else NA
      
      # Detect GLM (has "alpha")
    } else if ("alpha" %in% names(params)) {
      model_type <- "GLM"
      alpha      <- params$alpha
      lambda     <- params$lambda
    }
    
    # 3. Return Dataframe
    data.frame(
      Model = model_name,
      Type  = model_type,
      AUC   = best_auc,
      Brier_Score = brier_val,
      Penalized_Brier = pen_brier_val,
      
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
  
  # 1. Combine Results
  comparison_table <- bind_rows(
    extract_metrics(GLM_Results_BaseModel,  "Base Model"),
    extract_metrics(GLM_Results_Strategy_A, "Strategy A (Dim. Reduction)"),
    extract_metrics(GLM_Results_Strategy_B, "Strategy B (Anomaly Score)"),
    extract_metrics(GLM_Results_Strategy_C, "Strategy C (Feature Denoising)"),
    extract_metrics(GLM_Results_Strategy_D, "Strategy D (Manual Feature Eng.)")
  ) %>%
    # Filter out failed models
    filter(!is.na(AUC)) %>%
    arrange(desc(AUC)) %>% 
    mutate(
      # 2. Calculate Uplifts
      Base_AUC = AUC[Model == "Base Model"],
      
      # AUC Uplift (Positive is Good)
      Uplift_AUC_pct = ((AUC - Base_AUC) / Base_AUC) * 100,
      
      # Brier Uplifts (Negative Change is Good, so we invert sign to show 'Improvement')
      Base_Brier = Brier_Score[Model == "Base Model"],
      Base_PBS   = Penalized_Brier[Model == "Base Model"],
      
      # Formula: (Old - New) / Old * 100  -> Positive % means error went DOWN
      Uplift_Brier_pct = (Base_Brier - Brier_Score) / Base_Brier * 100,
      Uplift_PBS_pct   = (Base_PBS - Penalized_Brier) / Base_PBS * 100
    ) %>%
    # 3. Final Selection
    select(
      Model, 
      AUC, Uplift_AUC_pct, 
      Brier_Score, Uplift_Brier_pct,
      Penalized_Brier, Uplift_PBS_pct,
      Alpha, Lambda
    )
  
  print("--- Final GLM Model Leaderboard ---")
  print(comparison_table)
  
}, error = function(e) message("Comparison Error: ", e))

#==============================================================================#
#==== 02 - GLM Model Comparison (Plots & Charts) ==============================#
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
  
  Path <- file.path(here::here("")) 
  Charts_Directory_Model <- file.path(Path, "03_Charts/VAE/GLM")
  
  ###### Input parameters.
  model_used_name_list <- list("BaseModel", "StrategyA", "StrategyB",
                               "StrategyC","StrategyD")
  
  # GLM Results List
  model_object_list <- list(GLM_Results_BaseModel$optimal_model,
                            GLM_Results_Strategy_A$optimal_model,
                            GLM_Results_Strategy_B$optimal_model,
                            GLM_Results_Strategy_C$optimal_model,
                            GLM_Results_Strategy_D$optimal_model)
  
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
    if(!dir.exists(Directory)) dir.create(Directory, recursive = TRUE)
    
    #==== 02B - Feature Importance (GLM Adapted) ==================================#
    
    tryCatch({
      
      message(paste("--- Generating Feature Importance for", model_used_name, "---"))
      
      # GLM Feature Importance = Absolute Coefficients
      # Extract coefficients at lambda.min
      coef_sparse <- coef(model_object, s = "lambda.min")
      coef_df <- data.frame(
        Feature = rownames(coef_sparse),
        Coefficient = as.vector(coef_sparse)
      ) %>%
        filter(Feature != "(Intercept)") %>% # Remove Intercept
        mutate(Gain = abs(Coefficient)) %>%  # Magnitude = Importance
        arrange(desc(Gain)) %>%
        head(15) # Top 15
      
      # Define Map
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
      
      # Apply Map
      coef_df$Feature_Label <- ifelse(
        coef_df$Feature %in% names(feature_map), 
        feature_map[coef_df$Feature], 
        coef_df$Feature
      )
      
      p_importance <- ggplot(coef_df, aes(x = reorder(Feature_Label, Gain), y = Gain)) +
        geom_col(fill = "steelblue") +
        coord_flip() +
        labs(title = NULL, 
             subtitle = NULL, 
             x = NULL, 
             y = "Abs. Coefficient (Importance)") + # Updated label for GLM
        theme_minimal() +
        theme(
          axis.text.x = element_text(size = 14, color = "black"),
          axis.text.y = element_text(size = 14, color = "black")
        )
      
      print(p_importance)
      
      # Save Plot
      Path_plot <- file.path(Directory, paste0(model_used_name, "_Feature_Importance.png"))
      ggsave(filename = Path_plot, plot = p_importance, width = 8, height = 6)
      
      # Identify Top Features for later steps
      top_feature_1_raw <- coef_df$Feature[1]
      top_feature_2_raw <- coef_df$Feature[2]
      all_features_list <- coef_df$Feature # Top 15 drivers
      
    }, error = function(e) message("Feature Importance Error: ", e))
    
    #==== 02C - Marginal Response (GLM Adapted) ===================================#
    
    tryCatch({
      
      # Map Update (Same as before)
      feature_map <- c(
        "f1"  = "Total Assets", "f2"  = "Fixed Assets", "f3"  = "Current Assets",
        "f4"  = "Inventories", "f5"  = "Cash & Equivalents", "f6"  = "Equity",
        "f7"  = "Retained Earnings", "f8"  = "Net Profit", "f9"  = "Profit Carried Forward",
        "f10" = "Provisions", "f11" = "Liabilities",
        "Gap_Debt_Equity" = "Solvency Gap", "Ratio_Cash_Profit" = "Cash Burn Ratio",
        "l1"="Latent Dim 1", "l2"="Latent Dim 2", "l3"="Latent Dim 3", "l4"="Latent Dim 4",
        "l5"="Latent Dim 5", "l6"="Latent Dim 6", "l7"="Latent Dim 7", "l8"="Latent Dim 8",
        "dae_l1"="Robust Latent 1", "dae_l2"="Robust Latent 2", "dae_l3"="Robust Latent 3", "dae_l4"="Robust Latent 4",
        "dae_l5"="Robust Latent 5", "dae_l6"="Robust Latent 6", "dae_l7"="Robust Latent 7", "dae_l8"="Robust Latent 8"
      )
      
      # Prepare Matrix
      sparse_formula <- as.formula("y ~ . - 1")
      full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
      
      set.seed(42)
      n_sample <- min(nrow(full_train_matrix), 1000)
      calc_sample_indices <- sample(nrow(full_train_matrix), n_sample) 
      train_matrix_calc <- full_train_matrix[calc_sample_indices, ]
      
      # GLM Prediction Wrapper
      # MUST specify 'newx', 's', and 'type'
      pred_wrapper <- function(object, newdata) {
        predict(object, newx = as.matrix(newdata), s = "lambda.min", type = "response")
      }
      
      plot_pdp_bars <- function(feature_name) {
        
        display_name <- feature_name
        if(feature_name %in% names(feature_map)) display_name <- feature_map[[feature_name]]
        if(!feature_name %in% colnames(train_matrix_calc)) return(NULL)
        
        # Calculate PDP
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
        
        p <- ggplot(pdp_data, aes(x = .data[[feature_name]], y = yhat)) +
          geom_col(fill = "steelblue", width = 0.8) + 
          labs(title = display_name, subtitle = NULL, y = "Pred. Default Prob.", x = NULL) +
          theme_minimal() +
          theme(
            axis.text.x = element_text(size = 14, color = "black"),
            axis.text.y = element_text(size = 14, color = "black"),
            axis.title.y = element_text(size = 14, color = "black"),
            plot.title = element_text(size = 16, face = "bold")
          )
        return(p)
      }
      
      # Batch Generation
      message("--- Generating PDP Batches ---")
      valid_features <- intersect(all_features_list, colnames(train_matrix_calc))
      plot_list <- list()
      
      for(feat in valid_features) {
        p <- plot_pdp_bars(feat)
        if(!is.null(p)) plot_list[[length(plot_list) + 1]] <- p
      }
      
      # Save Batches
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
            file_name <- paste0(model_used_name, "MarginalResponse_Batch_", i, ".png")
            Path_plot <- file.path(Directory, file_name)
            ggsave(filename = Path_plot, plot = combined_plot, width = 12, height = 8)
            message(paste("Saved:", file_name))
          }
        }
      }
      
    }, error = function(e) message("PDP Batch Error: ", e))
    
    #==== 02D - Calibration Charts (GLM Adapted) ==================================#
    
    tryCatch({
      col_predicted <- "#377EB8"
      col_observed  <- "#7F7F7F"
      
      if(!exists("full_train_matrix")) {
        sparse_formula <- as.formula("y ~ . - 1")
        full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
      }
      
      message("Generating global predictions for calibration...")
      # GLM Prediction
      global_preds <- as.vector(predict(model_object, newx = full_train_matrix, s = "lambda.min", type = "response"))
      
      analysis_data <- data_input %>%
        mutate(
          y_actual = as.numeric(as.character(y)),
          y_pred   = global_preds
        )
      
      create_calib_plot <- function(feature_name) {
        display_name <- feature_name
        if(feature_name %in% names(feature_map)) display_name <- feature_map[[feature_name]]
        if(!feature_name %in% colnames(analysis_data)) return(NULL)
        
        calib_data <- analysis_data %>%
          select(val = all_of(feature_name), y_actual, y_pred) %>%
          mutate(bin = ntile(val, 10)) %>% 
          group_by(bin) %>%
          summarise(mean_prob = mean(y_pred), observed_rate = mean(y_actual), .groups = 'drop')
        
        plot_data <- calib_data %>%
          pivot_longer(cols = c("mean_prob", "observed_rate"), names_to = "Type", values_to = "Rate") %>%
          mutate(Type = factor(Type, levels = c("mean_prob", "observed_rate"), labels = c("Predicted", "Observed")))
        
        p <- ggplot(plot_data, aes(x = factor(bin), y = Rate, fill = Type)) +
          geom_col(position = position_dodge(width = 0.8), width = 0.7) +
          scale_fill_manual(values = c("Predicted" = col_predicted, "Observed" = col_observed)) + 
          scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.15))) + 
          labs(title = paste0(display_name), x = "Decile (Low -> High)", y = "Pred. Default Prob. (%)", fill = "") +
          theme_minimal() +
          theme(
            axis.text.x = element_text(size = 14, color = "black"),
            axis.text.y = element_text(size = 14, color = "black"),
            axis.title.y = element_text(size = 14, color = "black"),
            plot.title  = element_text(size = 16, face = "bold"),
            legend.position = "top",
            panel.grid.major.x = element_blank()
          )
        return(p)
      }
      
      message("--- Generating Calibration Batches ---")
      valid_features <- intersect(all_features_list, colnames(analysis_data))
      plot_list <- list()
      
      for(feat in valid_features) {
        p <- create_calib_plot(feat)
        if(!is.null(p)) plot_list[[length(plot_list) + 1]] <- p
      }
      
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
      }
      
    }, error = function(e) message("Calibration Batch Error: ", e))
    
    #==== 02E - Bivariate Interaction (GLM Adapted) ===============================#
    
    tryCatch({
      if(!exists("full_train_matrix")) {
        sparse_formula <- as.formula("y ~ . - 1")
        full_train_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
      }
      
      # GLM Prediction
      preds <- as.vector(predict(model_object, newx = full_train_matrix, s = "lambda.min", type = "response"))
      
      plot_beamer_hex <- function(feat_x, feat_y) {
        name_x <- ifelse(feat_x %in% names(feature_map), feature_map[[feat_x]], feat_x)
        name_y <- ifelse(feat_y %in% names(feature_map), feature_map[[feat_y]], feat_y)
        
        if (!feat_x %in% colnames(full_train_matrix) | !feat_y %in% colnames(full_train_matrix)) return(NULL)
        
        plot_df <- data.frame(X_Val = full_train_matrix[, feat_x], Y_Val = full_train_matrix[, feat_y], Prob = preds)
        
        p <- ggplot(plot_df, aes(x = X_Val, y = Y_Val, z = Prob)) +
          stat_summary_hex(fun = mean, bins = 35, color = "grey92", size = 0.1) + 
          scale_fill_gradientn(colours = c("#FFFFFF", "#FFEDA0", "#FEB24C", "#F03B20", "#800026"), name = "Prob %", labels = scales::percent) +
          labs(title = NULL, subtitle = NULL, x = name_x, y = name_y) +
          theme_minimal() +
          theme(
            axis.text.x = element_text(size = 14, color = "black", face = "bold"),
            axis.text.y = element_text(size = 14, color = "black", face = "bold"),
            axis.title  = element_text(size = 15, face = "bold"),
            legend.position = "right", 
            legend.title = element_text(size = 12, face = "bold"),
            panel.background = element_rect(fill = "#F5F5F5", color = NA),
            plot.background = element_rect(fill = "white", color = NA)
          )
        return(p)
      }
      
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
            ggsave(filename = Path_plot, plot = p, width = 9, height = 7)
            message(paste("Saved:", file_name))
          }
        }
      }
    }, error = function(e) message("Beamer Plot Error: ", e))
    
    #==== 02F - Residual diagnostics (GLM Adapted) ================================#
    
    tryCatch({
      col_tn <- "#619CFF"; col_fp <- "#FFA500"; col_fn <- "#F8766D"; col_tp <- "#7F7F7F"
      
      sparse_formula <- as.formula("y ~ . - 1")
      X_matrix <- sparse.model.matrix(sparse_formula, data = data_input)
      label_vec <- as.numeric(as.character(data_input$y))
      
      # GLM Prediction
      pred_probs <- as.vector(predict(model_object, newx = X_matrix, s = "lambda.min", type = "response"))
      
      roc_obj <- roc(label_vec, pred_probs, quiet = TRUE)
      best_threshold_obj <- coords(roc_obj, "best", ret = "threshold", transpose = TRUE)
      best_cutoff <- as.numeric(best_threshold_obj[1])
      
      Diagnosis_DF <- data_input %>%
        mutate(
          y_num = label_vec, xgb_prob = pred_probs,
          Prediction_Class = ifelse(xgb_prob > best_cutoff, 1, 0),
          Error_Type = case_when(
            y_num == 1 & Prediction_Class == 1 ~ "True Positive",
            y_num == 0 & Prediction_Class == 0 ~ "True Negative",
            y_num == 0 & Prediction_Class == 1 ~ "False Positive",
            y_num == 1 & Prediction_Class == 0 ~ "False Negative"
          ),
          Error_Type = factor(Error_Type, levels = c("True Negative", "False Positive", "False Negative", "True Positive"))
        )
      
      # Boxplot Function
      plot_error_boxplot <- function(feature_name) {
        display_name <- feature_name
        if(feature_name %in% names(feature_map)) display_name <- feature_map[[feature_name]]
        if(!feature_name %in% colnames(Diagnosis_DF)) return(NULL)
        
        p <- ggplot(Diagnosis_DF, aes(x = Error_Type, y = .data[[feature_name]], fill = Error_Type)) +
          geom_boxplot(outlier.alpha = 0.2, outlier.size = 0.5, outlier.colour = "grey50") +
          scale_fill_manual(values = c("True Negative" = col_tn, "False Positive" = col_fp, "False Negative" = col_fn, "True Positive" = col_tp)) +
          coord_cartesian(ylim = c(-3, 3)) + 
          labs(title = display_name, x = NULL, y = NULL) +
          theme_minimal() +
          theme(
            legend.position = "none", plot.title = element_text(face = "bold", size = 14),
            axis.text.x = element_blank(), axis.text.y = element_text(size = 10, color = "black")
          )
        return(p)
      }
      
      message("--- Generating Error Analysis Boxplots ---")
      all_numeric_feats <- names(select(Diagnosis_DF, where(is.numeric)))
      exclude_cols <- c("y_num", "xgb_prob", "Prediction_Class", "y")
      valid_features <- setdiff(all_numeric_feats, exclude_cols)
      
      plot_list <- list()
      for(feat in valid_features) {
        p <- plot_error_boxplot(feat)
        if(!is.null(p)) plot_list[[length(plot_list) + 1]] <- p
      }
      
      # Save Batches
      if(length(plot_list) > 0) {
        dummy_df <- data.frame(Error_Type = factor(c("True Negative", "False Positive", "False Negative", "True Positive"), levels = c("True Negative", "False Positive", "False Negative", "True Positive")), Value = 1)
        legend_plot <- ggplot(dummy_df, aes(x=Error_Type, y=Value, fill=Error_Type)) + geom_bar(stat="identity") + scale_fill_manual(values = c("True Negative" = col_tn, "False Positive" = col_fp, "False Negative" = col_fn, "True Positive" = col_tp), name = "") + theme_minimal() + theme(legend.position = "bottom", legend.text = element_text(size = 12))
        get_legend <- function(myggplot){ tmp <- ggplot_gtable(ggplot_build(myggplot)); leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box"); legend <- tmp$grobs[[leg]]; return(legend) }
        shared_legend <- get_legend(legend_plot)
        
        num_plots <- length(plot_list); plots_per_page <- 4; num_pages <- ceiling(num_plots / plots_per_page)
        for(i in 1:num_pages) {
          start_idx <- (i - 1) * plots_per_page + 1; end_idx <- min(i * plots_per_page, num_plots)
          current_batch <- plot_list[start_idx:end_idx]; current_batch <- current_batch[!sapply(current_batch, is.null)]
          if(length(current_batch) > 0) {
            grid_plots <- arrangeGrob(grobs = current_batch, ncol = 2, nrow = 2)
            final_grid <- arrangeGrob(grid_plots, shared_legend, nrow = 2, heights = c(10, 1))
            file_name <- paste0(model_used_name, "Error_Boxplot_Batch_", i, ".png")
            Path_plot <- file.path(Directory, file_name)
            ggsave(filename = Path_plot, plot = final_grid, width = 10, height = 8)
            message(paste("Saved:", file_name))
          }
        }
      }
      
      # Forensic Summary Table
      tryCatch({
        top_5_features <- head(all_features_list[all_features_list %in% colnames(data_input)], 5)
        message(paste("Generating Forensic Summary for Top 5 Drivers:", paste(top_5_features, collapse=", ")))
        
        Error_Summary <- Diagnosis_DF %>%
          group_by(Error_Type) %>%
          summarise(
            Count = n(),
            across(all_of(top_5_features), \(x) median(x, na.rm = TRUE), .names = "Median_{.col}")
          ) %>%
          arrange(match(Error_Type, c("True Negative", "False Positive", "False Negative", "True Positive")))
        
        current_names <- colnames(Error_Summary); new_names <- current_names
        for(i in seq_along(current_names)) {
          col_name <- current_names[i]
          if(startsWith(col_name, "Median_")) {
            raw_feat <- sub("Median_", "", col_name)
            if(raw_feat %in% names(feature_map)) new_names[i] <- paste0("Median_", feature_map[[raw_feat]])
          }
        }
        colnames(Error_Summary) <- new_names
        print("--- Forensic Feature Summary (Top 5 Drivers) ---")
        print(Error_Summary)
        write.xlsx(x = Error_Summary, file = file.path(Directory, "Error_Summary.xlsx"))
      }, error = function(e) message("Error Summary Failed: ", e))
      
    }, error = function(e) message("Boxplot Batch Error: ", e))
    
  }
  
}, error = function(e) message(e))

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
Test_Data_Strategy_C <- Strategy_C_Test
  
### Strategy D: Manual feature engineering.
Test_Data_Strategy_D <- Strategy_D_Test
  
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

#==== 03E - Strategy C: Feature Denoising =====================================#

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

#==== 03F - Strategy D: Manual feature engineering ============================#

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
    GLM_Test_Results_Strategy_D$Metrics %>% mutate(Strategy = "Strategy D (Residual Fit)")
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