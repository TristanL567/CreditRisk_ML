#==============================================================================#
#==== Load GLM Results for Documentation ======================================#
#==============================================================================#
# This script loads GLM results from Run_GLM_Analysis.R
# to be included in Methodology_Summary.Rnw

# Check if results file exists
Results_Path <- file.path(here::here("CreditRisk_ML"), "06_Documentation", "Results")
results_file <- file.path(Results_Path, "glm_results.rds")

if(file.exists(results_file)) {
  
  # Load results
  glm_results <- readRDS(results_file)
  
  # Extract for use in .Rnw
  glm_method_comp <- glm_results$method_comparison
  
  # Format for table display
  glm_results_table <- data.frame(
    Method = glm_method_comp$Method,
    "Optimal Alpha" = sprintf("%.4f", glm_method_comp$alpha),
    "CV AUC" = sprintf("%.4f", glm_method_comp$CV_AUC),
    "Train AUC" = sprintf("%.4f", glm_method_comp$Train_AUC),
    "Features" = glm_method_comp$n_nonzero_coeffs,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  # Add test results for champion
  glm_champion_summary <- data.frame(
    Metric = c("Method", "Optimal Alpha", "Optimal Lambda", 
               "CV AUC", "Train AUC", "Test AUC (Champion)", 
               "Test AUC (1-SE)", "Features (Champion)", "Features (1-SE)"),
    Value = c(
      glm_results$champion_method,
      sprintf("%.4f", glm_results$champion_alpha),
      sprintf("%.6f", glm_results$champion_lambda),
      sprintf("%.4f", glm_results$cv_auc),
      sprintf("%.4f", glm_results$train_auc),
      sprintf("%.4f", glm_results$test_auc_champion),
      sprintf("%.4f", glm_results$test_auc_1se),
      as.character(glm_results$n_features_champion),
      as.character(glm_results$n_features_1se)
    ),
    stringsAsFactors = FALSE
  )
  
  cat("GLM results loaded successfully.\n")
  cat("Available objects: glm_results_table, glm_champion_summary\n")
  
} else {
  warning(paste("GLM results file not found:", results_file))
  cat("Please run Run_GLM_Analysis.R first to generate results.\n")
  
  # Create placeholder table
  glm_results_table <- data.frame(
    Method = c("Grid Search", "Random Search", "Bayesian Optimization"),
    "Optimal Alpha" = c("TBD", "TBD", "TBD"),
    "CV AUC" = c("TBD", "TBD", "TBD"),
    "Train AUC" = c("TBD", "TBD", "TBD"),
    "Features" = c("TBD", "TBD", "TBD"),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  glm_champion_summary <- data.frame(
    Metric = "Not Available",
    Value = "Run Run_GLM_Analysis.R first",
    stringsAsFactors = FALSE
  )
}

#==============================================================================#
