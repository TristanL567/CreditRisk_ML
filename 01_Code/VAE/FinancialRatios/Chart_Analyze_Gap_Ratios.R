library(ggplot2)
library(dplyr)
library(tidyr)
library(xgboost)
library(Matrix)
library(pROC)

Chart_Analyze_Gap_Ratios <- function(data_subset, model, save_dir, file_suffix, title_suffix, analysis_target = "defaults") {
  
  tryCatch({
    message(paste0("--- Starting Gap Analysis (", analysis_target, "): ", title_suffix, " ---"))
    
    # --- 1. Setup & Predict ---
    # Remove metadata
    cols_to_drop <- c("id", "refdate", "time_index", "year", "row_id", "company_id")
    data_clean   <- data_subset[, !names(data_subset) %in% cols_to_drop]
    
    # Predict
    sparse_matrix <- sparse.model.matrix(y ~ . - 1, data = data_clean)
    preds <- predict(model, sparse_matrix)
    actuals <- as.numeric(as.character(data_clean$y))
    
    # Calculate Threshold
    roc_obj <- roc(actuals, preds, quiet = TRUE)
    best_cutoff <- as.numeric(coords(roc_obj, "best", ret = "threshold", transpose = TRUE)[1])
    
    # --- 2. Classify Groups ---
    All_Data <- data_subset %>%
      mutate(
        Pred_Prob = preds,
        Prediction_Class = ifelse(Pred_Prob > best_cutoff, 1, 0),
        
        Group = case_when(
          y == 1 & Prediction_Class == 1 ~ "Caught (TP)",
          y == 1 & Prediction_Class == 0 ~ "Missed (FN)",
          y == 0 & Prediction_Class == 1 ~ "False Alarm (FP)",
          y == 0 & Prediction_Class == 0 ~ "Correct Survivor (TN)",
          TRUE ~ "Other"
        )
      )
    
    # --- 3. Filter based on Target ---
    if (analysis_target == "defaults") {
      # Focus on ACTUAL DEFAULTS (y=1)
      Gap_Data <- All_Data %>% 
        filter(y == 1) %>%
        filter(Group %in% c("Caught (TP)", "Missed (FN)"))
      
      # Define Comparison Logic for Gap Calculation
      group_1 <- "Caught (TP)"
      group_2 <- "Missed (FN)"
      
      # Colors: Blue vs Red
      fill_colors <- c("TRUE" = "#377EB8", "FALSE" = "#E41A1C")
      fill_labels <- c("TRUE" = "Caught Group Higher", "FALSE" = "Missed Group Higher")
      
    } else if (analysis_target == "survivors") {
      # Focus on ACTUAL SURVIVORS (y=0)
      Gap_Data <- All_Data %>% 
        filter(y == 0) %>%
        filter(Group %in% c("Correct Survivor (TN)", "False Alarm (FP)"))
      
      # Define Comparison Logic for Gap Calculation
      group_1 <- "Correct Survivor (TN)"
      group_2 <- "False Alarm (FP)"
      
      # Colors: Grey vs Orange
      fill_colors <- c("TRUE" = "#7F7F7F", "FALSE" = "#FFA500")
      fill_labels <- c("TRUE" = "Correct Survivor Higher", "FALSE" = "False Alarm Higher")
      
    } else {
      stop("Invalid analysis_target. Use 'defaults' or 'survivors'.")
    }
    
    # Check if we have enough data
    counts <- table(Gap_Data$Group)
    if(length(counts) < 2) {
      message("Skipping plot: Not enough samples in both groups to compare.")
      return(NULL)
    }
    
    message(paste("Comparing", counts[group_1], "vs.", counts[group_2]))
    
    # --- 4. Calculate Feature Means & Gaps ---
    Feature_Means <- Gap_Data %>%
      group_by(Group) %>%
      summarise(across(starts_with("r"), mean, .names = "{.col}")) %>%
      pivot_longer(cols = -Group, names_to = "Feature", values_to = "Mean_Value") %>%
      pivot_wider(names_from = Group, values_from = Mean_Value) %>%
      mutate(
        # Dynamic Column Access using .data[[string]] or standard subsetting
        Val_1 = .[[group_1]],
        Val_2 = .[[group_2]],
        
        Gap = Val_1 - Val_2, # Positive = Group 1 (Caught/Correct) is higher
        Abs_Gap = abs(Gap)
      ) %>%
      arrange(desc(Abs_Gap))
    
    # Map Names
    feature_map <- c(
      "r1"  = "Total Assets",                 # Bilanzsumme
      "r2"  = "Total Equity",                 # Eigenkapital
      "r3"  = "Total Liabilities",            # Verbindlichkeiten
      "r4"  = "Total Debt",                   # Fremdkapital (Calculated)
      "r5"  = "Asset Coverage Ratio (I)",     # Anlagendeckungsgrad I (Equity/Fixed Assets)
      "r6"  = "Equity Ratio",                 # Eigenmittelquote
      "r7"  = "Debt Ratio",                   # Gesamtverschuldungsgrad
      "r8"  = "Net Debt Ratio",               # Netto-Gesamtverschuldungsgrad
      "r9"  = "Self-Financing Ratio",         # Selbstfinanzierungsgrad
      "r10" = "Short-term Asset Structure",   # Kurzfr. Vermögensstruktur
      "r11" = "Inventory Intensity",          # Vorratsquote
      "r12" = "Cash to Current Assets",       # Liquiditätsquote (Umlauf)
      "r13" = "Cash to Total Assets",         # Liquiditätsquote (Gesamt)
      "r14" = "Return on Assets (ROA)",       # Gesamtkapitalrentabilität
      "r15" = "Return on Equity (ROE)",       # Eigenkapitalrentabilität
      "r16" = "Return on Fixed Assets",       # Anlagenrentabilität
      "r17" = "Debt Service Coverage",        # Schuldendeckungsfähigkeit
      "r18" = "Net Debt Service Coverage"     # Netto-Entschuldungsfähigkeit
    )
    
    Feature_Means$Feature_Name <- feature_map[Feature_Means$Feature]
    
    # --- 5. Plot ---
    p_gap <- ggplot(Feature_Means, aes(x = reorder(Feature_Name, Gap), y = Gap, fill = Gap > 0)) +
      geom_col() +
      coord_flip() +
      
      scale_fill_manual(values = fill_colors, labels = fill_labels) +
      
      labs(
        x = "",
        y = "Gap (Standardized Mean Difference)",
        fill = "Direction"
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
    
    # --- 6. Save ---
    if(!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
    file_path <- file.path(save_dir, paste0("Gap_Analysis_", file_suffix, ".png"))
    ggsave(file_path, p_gap, width = 10, height = 6)
    
    message(paste("Saved plot to:", file_path))
    print(p_gap)
    
  }, error = function(e) message("Gap Analysis Error: ", e))
}