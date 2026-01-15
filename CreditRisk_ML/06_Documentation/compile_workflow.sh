#!/bin/bash
#==============================================================================#
# Complete Workflow: Run Analysis and Compile Documentation
#==============================================================================#

set -e  # Exit on error

echo "=========================================================================="
echo "Credit Risk ML - Documentation Workflow"
echo "=========================================================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project Root: $PROJECT_ROOT"
echo ""

# Step 1: Run GLM Analysis
echo "Step 1: Running GLM Analysis..."
echo "--------------------------------------------------------------------------"
cd "$PROJECT_ROOT/01_Code"
Rscript Run_GLM_Analysis.R

if [ $? -eq 0 ]; then
    echo "✓ GLM Analysis completed successfully"
else
    echo "✗ GLM Analysis failed"
    exit 1
fi
echo ""

# Step 2: Compile Documentation
echo "Step 2: Compiling Documentation..."
echo "--------------------------------------------------------------------------"
cd "$PROJECT_ROOT/06_Documentation"

# Create R script to compile
cat > compile_doc.R << 'EOF'
library(knitr)
library(here)

Path <- file.path(here::here("CreditRisk_ML"))
Directory <- file.path(Path, "06_Documentation")
Directory_LaTeX <- file.path(Directory, "Methodology_Summary.Rnw")

setwd(Directory)

# Compile
tryCatch({
  knit2pdf(Directory_LaTeX)
  cat("\n✓ PDF compilation successful\n")
}, error = function(e) {
  cat("\n✗ PDF compilation failed:\n")
  print(e)
  quit(status = 1)
})
EOF

Rscript compile_doc.R
rm compile_doc.R

if [ $? -eq 0 ]; then
    echo "✓ Documentation compiled successfully"
    echo ""
    echo "Output: $PROJECT_ROOT/06_Documentation/Methodology_Summary.pdf"
else
    echo "✗ Documentation compilation failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "Workflow Complete!"
echo "=========================================================================="
echo ""
echo "Generated files:"
echo "  - Model Results: 06_Documentation/Results/glm_results.rds"
echo "  - Charts: 03_Charts/GLM/"
echo "  - PDF: 06_Documentation/Methodology_Summary.pdf"
echo ""
