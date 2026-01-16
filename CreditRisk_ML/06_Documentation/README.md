# Credit Risk ML - Documentation

This directory contains documentation for the Credit Risk Machine Learning project.

## Quick Start

**See [../QUICK_START.md](../QUICK_START.md) for running the analysis.**

The simplest approach:
1. Open R in the `CreditRisk_ML` directory
2. Run `source("01_Code/Main.R")` or execute block by block
3. GLM results saved to `03_Charts/GLM/`

## Main Documentation File

### Methodology_Summary.Rnw
- Sweave document combining LaTeX and R code
- Contains methodology referencing code from `01_Code/Main.R`
- Compile with: `R CMD Sweave Methodology_Summary.Rnw` then `pdflatex Methodology_Summary.tex`

## Integration with Main.R

The documentation references code from `01_Code/Main.R` rather than duplicating it:

- **Main Analysis**: [../01_Code/Main.R](../01_Code/Main.R)
  - Lines 252-485: GLM Analysis section
  - Lines 500+: XGBoost section (requires R >= 4.3)
  
- **Helper Functions**: `01_Code/Subfunctions/`
  - `GLM_gridsearch.R`: Grid search for GLM
  - `GLM_bayesoptim.R`: Bayesian optimization for GLM
  - `DataPreprocessing.R`: Data filtering
  - `QuantileTransformation.R`: Feature transformation
  - `MVstratifiedsampling.R`: Stratified sampling

- **Output Charts**: `03_Charts/GLM/`
  - Referenced in documentation

## Requirements

- R version 4.1+ (4.3+ for XGBoost)
- All packages from CRAN (auto-installed by Main.R)
- Data file: `../data/data.rda`
- LaTeX distribution for PDF compilation (optional)

## Directory Structure

```
06_Documentation/
├── README.md                     # This file
├── Methodology_Summary.Rnw       # Main Sweave document
├── Methodology_Summary.tex       # Generated LaTeX
├── Methodology_Summary.pdf       # Final PDF output
├── references.bib                # Bibliography
└── Archive/                      # Historical files
```

## Compiling Documentation

```r
# From R console in CreditRisk_ML/06_Documentation:
setwd("06_Documentation")
library(knitr)
knit2pdf("Methodology_Summary.Rnw")
```

Or using command line:
```bash
cd 06_Documentation
R CMD Sweave Methodology_Summary.Rnw
pdflatex Methodology_Summary.tex
```

## Data Requirements

The analysis expects:
- **Input Data**: `../data/data.rda` containing variable `d`
- **Features**: `f1` through `f11` (financial statement items)
- **Target**: `y` (default indicator)
- **Metadata**: `sector`, `refdate`, `id` for stratification
