#!/usr/bin/env Rscript
#==============================================================================#
# QUICK START: GLM Documentation Workflow
#==============================================================================#
# This script demonstrates the complete workflow for GLM analysis and docs

cat("=============================================================================\n")
cat("GLM Documentation Workflow - Quick Start Guide\n")
cat("=============================================================================\n\n")

# Check if we're in the right directory
if (!dir.exists("01_Code") || !dir.exists("06_Documentation")) {
  cat("ERROR: Please run this script from CreditRisk_ML/ directory\n\n")
  cat("Usage:\n")
  cat("  cd CreditRisk_ML\n")
  cat("  Rscript 06_Documentation/quick_start.R\n\n")
  quit(status = 1)
}

cat("Current directory:", getwd(), "\n\n")

# Step 1: Check dependencies
cat("Step 1: Checking dependencies...\n")
cat("-------------------------------------------\n")

required_packages <- c(
  "here", "dplyr", "caret", "lubridate", "purrr", "tidyr",
  "Matrix", "pROC", "glmnet", "rBayesianOptimization",
  "ggplot2", "scales", "knitr", "xtable"
)

missing <- character()
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing <- c(missing, pkg)
  }
}

if (length(missing) > 0) {
  cat("Missing packages:", paste(missing, collapse = ", "), "\n")
  cat("Installing missing packages...\n")
  install.packages(missing, quiet = TRUE)
  cat("Installation complete!\n\n")
} else {
  cat("All required packages are installed.\n\n")
}

# Step 2: Check data file
cat("Step 2: Checking data file...\n")
cat("-------------------------------------------\n")

# You need to update this path!
data_paths <- c(
  "data/data.rda",
  "../data/data.rda",
  "/home/martin-mal/Documents/oenb_standalone/data/data.rda"
)

data_found <- FALSE
for (path in data_paths) {
  if (file.exists(path)) {
    cat("✓ Data file found:", path, "\n\n")
    data_found <- TRUE
    break
  }
}

if (!data_found) {
  cat("✗ Data file not found in expected locations.\n")
  cat("  Please update Data_Path in 01_Code/Main.R line 56\n\n")
  cat("Expected locations:\n")
  for (path in data_paths) {
    cat("  -", path, "\n")
  }
  cat("\nContinuing anyway (will fail at data loading)...\n\n")
}

# Step 3: Run GLM Analysis
cat("Step 3: Running GLM Analysis...\n")
cat("-------------------------------------------\n")
cat("This will execute Main.R and may take several minutes...\n\n")

tryCatch({
  source("01_Code/Run_GLM_Analysis.R", echo = FALSE, local = TRUE)
  cat("\n✓ GLM Analysis completed successfully!\n\n")
}, error = function(e) {
  cat("\n✗ GLM Analysis failed with error:\n")
  print(e)
  cat("\nPlease check:\n")
  cat("  1. Data path in Main.R line 56\n")
  cat("  2. All required packages are installed\n")
  cat("  3. Data file contains expected variables (f1-f11, y, sector, etc.)\n\n")
  quit(status = 1)
})

# Step 4: Check results
cat("Step 4: Verifying results...\n")
cat("-------------------------------------------\n")

results_file <- "06_Documentation/Results/glm_results.rds"
if (file.exists(results_file)) {
  results <- readRDS(results_file)
  cat("✓ Results file created successfully\n")
  cat("\nSummary:\n")
  cat("  Champion Method:", results$champion_method, "\n")
  cat("  Test AUC:", sprintf("%.4f", results$test_auc_champion), "\n")
  cat("  Features Selected:", results$n_features_champion, "\n\n")
} else {
  cat("✗ Results file not created\n\n")
  quit(status = 1)
}

# Step 5: Check charts
cat("Step 5: Checking generated charts...\n")
cat("-------------------------------------------\n")

chart_file <- "03_Charts/GLM/01_HyperparameterTuningMethods_AUC_Training.png"
if (file.exists(chart_file)) {
  cat("✓ Chart created:", chart_file, "\n\n")
} else {
  cat("⚠ Chart not found:", chart_file, "\n")
  cat("  This may cause issues in PDF compilation\n\n")
}

# Step 6: Compile documentation
cat("Step 6: Compiling PDF documentation...\n")
cat("-------------------------------------------\n")

setwd("06_Documentation")
tryCatch({
  library(knitr)
  knit2pdf("Methodology_Summary.Rnw")
  cat("\n✓ PDF compiled successfully!\n\n")
  
  if (file.exists("Methodology_Summary.pdf")) {
    cat("Output: 06_Documentation/Methodology_Summary.pdf\n\n")
  }
}, error = function(e) {
  cat("\n✗ PDF compilation failed with error:\n")
  print(e)
  cat("\nPlease check:\n")
  cat("  1. LaTeX is installed on your system\n")
  cat("  2. All chart files exist in 03_Charts/GLM/\n")
  cat("  3. Check .log file for LaTeX errors\n\n")
})

setwd("..")

# Summary
cat("=============================================================================\n")
cat("Quick Start Complete!\n")
cat("=============================================================================\n\n")

cat("Generated files:\n")
cat("  ✓ Model results: 06_Documentation/Results/glm_results.rds\n")
cat("  ✓ Charts: 03_Charts/GLM/*.png\n")
if (file.exists("06_Documentation/Methodology_Summary.pdf")) {
  cat("  ✓ PDF: 06_Documentation/Methodology_Summary.pdf\n")
} else {
  cat("  ✗ PDF: Compilation may have failed\n")
}

cat("\nNext steps:\n")
cat("  1. Review the PDF output\n")
cat("  2. Update chart paths in Methodology_Summary.Rnw if needed\n")
cat("  3. Apply the same pattern to XGBoost section\n")
cat("  4. Run from terminal with: Rscript 06_Documentation/quick_start.R\n\n")

cat("For more details, see:\n")
cat("  - 06_Documentation/README.md\n")
cat("  - 06_Documentation/INTEGRATION_SUMMARY.md\n\n")
