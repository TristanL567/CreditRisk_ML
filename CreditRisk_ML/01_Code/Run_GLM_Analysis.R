#==============================================================================#
#==== GLM Analysis Execution Script ===========================================#
#==============================================================================#
# This script runs the GLM section of Main.R and exports results
# for inclusion in the Methodology_Summary.Rnw documentation

#==== Setup ===================================================================#
library(here)

# Set paths
Path <- file.path(here::here("CreditRisk_ML"))
Results_Path <- file.path(Path, "06_Documentation", "Results")
dir.create(Results_Path, showWarnings = FALSE, recursive = TRUE)

#==== Source Main.R Setup =====================================================#
# Execute setup sections from Main.R
source(file.path(Path, "01_Code", "Main.R"), 
       echo = FALSE, 
       local = TRUE,
       encoding = "UTF-8")

# Note: The above will run the entire Main.R including GLM section (lines 195-394)
# If you want to run only GLM section, you can manually execute those lines

#==== Export GLM Results ======================================================#

if(exists("glm_method_performance") && exists("final_cv_glm")) {
  
  # Extract key results
  glm_results_export <- list(
    method_comparison = glm_method_performance,
    champion_method = champion_row$Method,
    champion_alpha = best_alpha,
    champion_lambda = final_cv_glm$lambda.min,
    cv_auc = max(final_cv_glm$cvm),
    train_auc = champion_row$Train_AUC,
    test_auc_champion = auc_champion,
    test_auc_1se = auc_1se,
    n_features_champion = n_vars_champion,
    n_features_1se = n_vars_1se
  )
  
  # Save as RDS for easy loading in .Rnw
  saveRDS(glm_results_export, 
          file = file.path(Results_Path, "glm_results.rds"))
  
  # Also save as CSV for easy inspection
  write.csv(glm_method_performance,
            file = file.path(Results_Path, "glm_method_comparison.csv"),
            row.names = FALSE)
  
  # Print summary
  cat("\n")
  cat("=============================================================================\n")
  cat("GLM ANALYSIS COMPLETE\n")
  cat("=============================================================================\n")
  cat(sprintf("Champion Method: %s\n", glm_results_export$champion_method))
  cat(sprintf("Optimal Alpha: %.4f\n", glm_results_export$champion_alpha))
  cat(sprintf("Optimal Lambda: %.6f\n", glm_results_export$champion_lambda))
  cat(sprintf("Cross-Validation AUC: %.4f\n", glm_results_export$cv_auc))
  cat(sprintf("Training AUC: %.4f\n", glm_results_export$train_auc))
  cat(sprintf("Test AUC (Champion): %.4f\n", glm_results_export$test_auc_champion))
  cat(sprintf("Test AUC (1-SE Rule): %.4f\n", glm_results_export$test_auc_1se))
  cat(sprintf("Features Selected (Champion): %d\n", glm_results_export$n_features_champion))
  cat(sprintf("Features Selected (1-SE): %d\n", glm_results_export$n_features_1se))
  cat("\nResults saved to: ", Results_Path, "\n")
  cat("=============================================================================\n")
  
} else {
  warning("GLM results not found. Make sure Main.R executed successfully.")
}

#==============================================================================#
