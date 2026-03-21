# Quick Start Guide

## Prerequisites

- R version 4.1 or higher (4.3+ required for XGBoost)
- Data file: `../data/data.rda` (one level up from CreditRisk_ML)

## Running the GLM Analysis

1. **Open R** in the `CreditRisk_ML` directory:
   ```bash
   cd /home/martin-mal/Documents/oenb_standalone/CreditRisk_ML
   R
   ```

2. **Run Main.R block by block** in the R console:
   
   ```r
   # Source the entire file or run sections manually
   source("01_Code/Main.R")
   ```

   Or run each section separately:
   
   - **Section 1** (lines 1-83): Package installation & setup
   - **Section 2** (lines 128-181): Data loading & preprocessing
   - **Section 3** (lines 186-231): Feature engineering
   - **Section 4** (lines 252-485): **GLM Analysis** (main focus)
   - **Section 5** (lines 500+): XGBoost (optional, requires R >= 4.3)

3. **View Results**:
   - Charts saved to: `03_Charts/GLM/`
   - Console output shows AUC scores and model performance

## Package Notes

- **Core packages** (required): dplyr, caret, lubridate, purrr, tidyr, Matrix, pROC, glmnet
- **Optional packages**: xgboost, rBayesianOptimization, ggplot2, Ckmeans.1d.dp, scales
- All packages install from CRAN
- Installation errors for optional packages won't stop execution

## Project Structure

```
CreditRisk_ML/
├── 01_Code/
│   ├── Main.R                    # Main analysis script
│   └── Subfunctions/             # Helper functions
├── 03_Charts/
│   ├── GLM/                      # GLM output charts
│   └── XGBoost/                  # XGBoost output charts
├── 06_Documentation/
│   └── Methodology_Summary.Rnw   # LaTeX documentation
└── ../data/
    └── data.rda                  # Input data file
```

## Troubleshooting

- **xgboost won't install**: Normal on R < 4.3, GLM section works fine without it
- **Path errors**: Make sure you're running from `CreditRisk_ML` directory
- **Data not found**: Verify `../data/data.rda` exists relative to CreditRisk_ML

## What Gets Executed

The GLM section performs:
1. Discrete grid search (alpha from 0 to 1)
2. Random grid search (20 iterations)
3. Bayesian optimization (5 init + 15 iterations)
4. Model comparison and visualization
5. Test set evaluation with lambda.min and lambda.1se

Results include AUC scores, variable selection counts, and performance charts.
