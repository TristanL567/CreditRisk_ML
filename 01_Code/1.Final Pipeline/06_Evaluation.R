# Preliminary structure of 06_Evaluation:
#   1. Load all prediction files
# — AutoGluon: predictions_test.parquet (M1–M4, via arrow)
# — XGBoost:   predictions_test.rds (M1–M4)
# — Stack into one combined data.table: id, y, p_default, model, framework
# 
# 2. Compute metrics per model × framework
# — AUC-ROC, AP, Brier, BSS, R@FPR1/3/5/10
# — Youden threshold, Sensitivity, Specificity, F1
# 
# 3. DeLong test (pROC::roc.test)
# — Pairwise AUC comparison: each model vs M1 baseline
# — Both within framework (AG vs AG, XGB vs XGB)
# — And cross-framework (AG_M1 vs XGB_M1)
# — Output: p-values + 95% CI on AUC difference
# 
# 4. Combined leaderboard table
# — Rows: model × framework
# — Columns: all metrics + DeLong p-value vs baseline
# — Sorted by AUC descending
# 
# 5. Save
# — leaderboard_full.rds        (R object for 07_Charts)
# — leaderboard_full.xlsx        (Excel, formatted)
# — leaderboard_full.tex         (LaTeX booktabs table)
# — predictions_combined.rds    (all predictions stacked, for charts)