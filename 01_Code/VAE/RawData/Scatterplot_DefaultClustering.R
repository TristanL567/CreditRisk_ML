library(ggplot2)
library(dplyr)
library(xgboost)
library(Matrix)
library(pROC)

Scatterplot_DefaultClustering <- function(data_input, model, save_dir, file_name, y_feature = "f8") {
  
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
      "f11" = "Liabilities"
    )
    
    y_label <- if (y_feature %in% names(feature_map)) feature_map[[y_feature]] else y_feature
    
    message(paste0("--- Starting Scatter Analysis for Feature: ", y_label, " ---"))
    
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
    
    message(paste("Optimal Threshold calculated:", round(best_cutoff, 4)))
    
    # --- 3. Classify & Filter Data ---
    Forensic_DF <- data_input %>%
      select(y, all_of(y_feature)) %>%
      mutate(
        Y_Value = .data[[y_feature]],
        Actual = actuals,
        Pred_Prob = preds,
        Prediction_Class = ifelse(Pred_Prob > best_cutoff, 1, 0),
        
        Error_Type = case_when(
          Actual == 1 & Prediction_Class == 0 ~ "False Negative (Missed Default)",
          Actual == 1 & Prediction_Class == 1 ~ "True Positive (Caught Default)",
          TRUE ~ "Other"
        )
      ) %>%
      filter(Error_Type %in% c("False Negative (Missed Default)", "True Positive (Caught Default)"))
    
    # --- 4. Generate Scatter Plot ---
    p_scatter <- ggplot(Forensic_DF, aes(x = Pred_Prob, y = Y_Value, color = Error_Type)) +
      
      geom_point(alpha = 0.6, size = 3) +
      
      scale_color_manual(values = c(
        "False Negative (Missed Default)" = "#E41A1C", # Red
        "True Positive (Caught Default)"  = "#377EB8"  # Blue
      )) +
      
      geom_vline(xintercept = best_cutoff, linetype = "dashed", color = "black") +
      annotate("text", x = best_cutoff, y = max(Forensic_DF$Y_Value), 
               label = paste("Threshold:", round(best_cutoff, 3)), vjust = -1, angle = 90) +
      
      labs(
        # title = paste0("Clustering of Defaults: ", y_label),
        # subtitle = paste0("Red Points = Missed Defaults (Predicted Prob < ", round(best_cutoff, 2), ")\nBlue Points = Caught Defaults.\nAnalyzing distribution across ", y_label, "."),
        x = "Predicted Probability of Default",
        y = paste0(y_label, ""),
        color = "Outcome"
      ) +
      
      theme_minimal() +
      theme(
        plot.title = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 14, color = "black"),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 14, color = "black"),
        axis.title.x = element_text(size = 14, color = "black"),
        axis.text  = element_text(size = 14, color = "black"),
        legend.position = "bottom"
      )
    
    # --- 5. Save ---
    if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    
    full_path <- file.path(save_dir, file_name)
    ggsave(full_path, p_scatter, width = 10, height = 7)
    
    message(paste("Scatter plot saved to:", full_path))
    print(p_scatter)
    
  }, error = function(e) message("Scatter Plot Error: ", e))
}