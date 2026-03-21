analyse_MVstratifiedsampling <- function(df, dataset_name, 
                                         target_col = "y", 
                                         sector_col = "sector", 
                                         date_col = "refdate",
                                         id_col = "id"){
    
    # 1. Standardize column names for internal processing
    df_metrics <- df %>%
      rename(Target = all_of(target_col),
             Sector = all_of(sector_col),
             ID = all_of(id_col)) %>%
      mutate(
        # Ensure Target is numeric 0/1 for calculation
        TargetNum = as.numeric(as.character(Target)),
        # Extract Year if date_col is present, otherwise look for 'Year'
        Year = if(date_col %in% names(df)) year(get(date_col)) else Year
      )
    
    cat(paste0("\n========================================\n"))
    cat(paste0("   ANALYSIS: ", dataset_name, "\n"))
    cat(paste0("========================================\n"))
    
    # --- 2. Global Rates (Firm vs Observation) ---
    # Firm Level: Did the firm EVER default?
    firm_stats <- df_metrics %>%
      group_by(ID) %>%
      summarize(EverDefault = max(TargetNum), .groups = 'drop')
    
    cat("\n[1] GLOBAL DEFAULT RATES\n")
    cat(sprintf("  • Observation Level (Weighted by duration): %5.2f%%\n", mean(df_metrics$TargetNum) * 100))
    cat(sprintf("  • Firm Level (Unique Entities):             %5.2f%%\n", mean(firm_stats$EverDefault) * 100))
    cat(sprintf("  • Total Firms: %d | Total Obs: %d\n", nrow(firm_stats), nrow(df_metrics)))
    
    # --- 3. Stratification Check (Sector x Default) ---
    # This verifies if your multivariate split worked
    cat("\n[2] FIRM-LEVEL BALANCE BY SECTOR\n")
    sector_summary <- df_metrics %>%
      group_by(ID, Sector) %>%
      summarize(EverDefault = max(TargetNum), .groups = 'drop') %>%
      group_by(Sector) %>%
      summarize(
        Firms = n(),
        Def_Firms = sum(EverDefault),
        Def_Rate = (sum(EverDefault) / n()) * 100
      ) %>%
      mutate(across(where(is.numeric), \(x) round(x, 2)))
    
    print(as.data.frame(sector_summary))
    
    # --- 4. Temporal Stability ---
    cat("\n[3] OBSERVATIONS BY YEAR (%)\n")
    print(round(prop.table(table(df_metrics$Year)) * 100, 2))
}