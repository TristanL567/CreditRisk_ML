library(ggplot2)
library(dplyr)
library(xgboost)
library(Matrix)
library(pROC)

Scatterplot_Interaction_Faceted <- function(data_input, model, save_dir, file_name, x_feature = "f8", y_feature = "f5", analysis_target = "defaults") {
  
  tryCatch({
    
    # --- 0. Setup: Feature Map ---
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
    
    x_label <- if (x_feature %in% names(feature_map)) feature_map[[x_feature]] else x_feature
    y_label <- if (y_feature %in% names(feature_map)) feature_map[[y_feature]] else y_feature
    
    message(paste0("--- Starting Faceted Analysis (", analysis_target, "): ", x_label, " vs. ", y_label, " ---"))
    
    # --- 1. Setup Data & Predict ---
    cols_to_drop <- c("id", "refdate", "time_index", "year", "row_id", "company_id")
    data_clean   <- data_input[, !names(data_input) %in% cols_to_drop]
    
    sparse_matrix <- sparse.model.matrix(y ~ . - 1, data = data_clean)
    preds <- predict(model, sparse_matrix)
    actuals <- as.numeric(as.character(data_clean$y))
    
    # --- 2. Compute Optimal Threshold ---
    roc_obj <- roc(actuals, preds, quiet = TRUE)
    best_coords <- coords(roc_obj, "best", ret = "threshold", transpose = TRUE)
    best_cutoff <- as.numeric(best_coords[1])
    
    # --- 3. Classify Data ---
    All_Data <- data_input %>%
      select(y, all_of(x_feature), all_of(y_feature)) %>%
      mutate(
        X_Value = .data[[x_feature]],
        Y_Value = .data[[y_feature]],
        Actual = actuals,
        Pred_Prob = preds,
        Prediction_Class = ifelse(Pred_Prob > best_cutoff, 1, 0),
        
        Error_Type = case_when(
          Actual == 1 & Prediction_Class == 0 ~ "False Negative (Missed Default)",
          Actual == 1 & Prediction_Class == 1 ~ "True Positive (Caught Default)",
          Actual == 0 & Prediction_Class == 1 ~ "False Positive (False Alarm)",
          Actual == 0 & Prediction_Class == 0 ~ "True Negative (Correct Survivor)"
        )
      )
    
    # --- 4. Filter based on Target ---
    if (analysis_target == "defaults") {
      # Focus on ACTUAL DEFAULTS (y=1)
      Plot_Data <- All_Data %>% 
        filter(Actual == 1)
      
      # Colors: Red vs Blue
      color_map <- c(
        "False Negative (Missed Default)" = "#E41A1C", # Red
        "True Positive (Caught Default)"  = "#377EB8"  # Blue
      )
      
    } else if (analysis_target == "survivors") {
      # Focus on ACTUAL SURVIVORS (y=0)
      Plot_Data <- All_Data %>% 
        filter(Actual == 0)
      
      # Colors: Orange (False Alarm) vs Grey (Correct Safe)
      color_map <- c(
        "False Positive (False Alarm)"     = "#FFA500", # Orange
        "True Negative (Correct Survivor)" = "#7F7F7F"  # Grey
      )
      
    } else {
      stop("Invalid analysis_target. Use 'defaults' or 'survivors'.")
    }
    
    # --- 5. Generate Faceted Plot ---
    p_scatter <- ggplot(Plot_Data, aes(x = X_Value, y = Y_Value, color = Error_Type)) +
      
      # Points
      geom_point(alpha = 0.6, size = 2.5) +
      
      # Facet Split
      facet_wrap(~Error_Type) +
      
      # Quadrant Lines
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey30") +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
      
      # Custom Colors
      scale_color_manual(values = color_map) +
      
      labs(
        x = paste0(x_label, ""),
        y = paste0(y_label, ""),
        color = "Outcome"
      ) +
      
      theme_minimal() +
      theme(
        strip.text = element_text(size = 14, face = "bold"),
        plot.title = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 14, color = "black"),
        axis.title.x = element_text(size = 14, color = "black"),
        axis.text  = element_text(size = 14, color = "black"),
        legend.position = "none"
      )
    
    # --- 6. Save ---
    if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    full_path <- file.path(save_dir, file_name)
    ggsave(full_path, p_scatter, width = 12, height = 6)
    
    message(paste("Faceted plot saved to:", full_path))
    print(p_scatter)
    
  }, error = function(e) message("Plot Error: ", e))
}