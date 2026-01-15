# Documentation Workflow for Methodology Summary

This directory contains the methodology documentation for the Credit Risk ML project.

## Structure

```
06_Documentation/
├── Methodology_Summary.Rnw       # Main Sweave document
├── Methodology_Summary.tex       # Generated LaTeX (from knitting .Rnw)
├── Methodology_Summary.pdf       # Final PDF output
├── references.bib                # Bibliography
├── Load_GLM_Results.R            # Helper script to load GLM results
└── Results/                      # Directory for exported model results
    ├── glm_results.rds          # GLM results (generated)
    └── glm_method_comparison.csv # GLM comparison (generated)
```

## Workflow

### 1. Run Model Analysis

First, execute the analysis in `Main.R` to generate model results:

```r
# From CreditRisk_ML/01_Code/
setwd("CreditRisk_ML/01_Code")
source("Run_GLM_Analysis.R")
```

This will:
- Execute the GLM section of `Main.R`
- Generate visualizations in `03_Charts/GLM/`
- Export results to `06_Documentation/Results/`

### 2. Compile Documentation

Once results are generated, compile the Methodology_Summary document:

```r
library(knitr)

# Set working directory
setwd("CreditRisk_ML/06_Documentation")

# Compile Sweave document to PDF
knit2pdf("Methodology_Summary.Rnw")
```

Alternatively, use the original knitr setup in the document header:

```r
library(knitr)
Path <- file.path(here::here("CreditRisk_ML"))
Directory <- file.path(Path, "06_Documentation")
Directory_LaTeX <- file.path(Directory, "Methodology_Summary.Rnw")
setwd(Directory)
knit2pdf(Directory_LaTeX)
```

### 3. View Output

The compilation process will generate:
- `Methodology_Summary.tex` (intermediate LaTeX)
- `Methodology_Summary.pdf` (final document)

## Integration with Existing R Code

The documentation now properly references your existing R implementation:

### Code Location
- **Main Analysis**: `01_Code/Main.R`
- **GLM Functions**: `01_Code/Subfunctions/GLM_gridsearch.R` and `GLM_bayesoptim.R`
- **Data Preprocessing**: `01_Code/Subfunctions/DataPreprocessing.R`
- **Feature Engineering**: `01_Code/Subfunctions/QuantileTransformation.R`
- **Sampling**: `01_Code/Subfunctions/MVstratifiedsampling.R`

### Charts Location
Generated charts are saved to `03_Charts/GLM/`:
- `01_HyperparameterTuningMethods_AUC_Training.png`

These are automatically referenced in the `.Rnw` document.

## Data Requirements

The analysis expects:
- **Input Data**: `data/data.rda` containing the variable `d` with balance sheet data
- **Features**: `f1` through `f11` (financial statement items)
- **Target**: `y` (default indicator)
- **Metadata**: `sector`, `refdate`, `id` for stratification

## Troubleshooting

### Issue: "GLM results file not found"
**Solution**: Run `Run_GLM_Analysis.R` first to generate results

### Issue: "Cannot find data.rda"
**Solution**: Update `Data_Path` in `Main.R` line 56 to point to your data location

### Issue: Charts not appearing in PDF
**Solution**: Verify that charts exist in `03_Charts/GLM/` before compiling

### Issue: LaTeX compilation errors
**Solution**: Ensure all required LaTeX packages are installed (geometry, xtable, graphicx, etc.)

## Notes

1. **Code Execution**: The `.Rnw` file references but does not re-execute the full analysis. Run `Main.R` separately for model training.

2. **Result Caching**: GLM results are cached in `Results/` directory. Delete these files to force re-analysis.

3. **Python Integration**: This R-based documentation is separate from the Python notebooks in the `notebooks/` directory. Consider which implementation represents your final methodology.

4. **Chart Paths**: The `.Rnw` file uses relative paths like `../03_Charts/GLM/`. Verify paths match your directory structure.

## Citation

Results are based on the implementation in `Main.R` (lines 195-394 for GLM section).
