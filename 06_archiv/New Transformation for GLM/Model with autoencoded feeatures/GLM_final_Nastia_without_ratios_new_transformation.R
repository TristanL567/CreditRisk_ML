#==============================================================================#
#==== 00 - Data Preparation (Base + A + B + C + D) ============================#
#==============================================================================#

##==============================##
## General Parameters.
##==============================##
N_folds <- 5

# Helper: align test factors/columns to train (prevents model.matrix/factor issues)
align_test_to_train <- function(train_df, test_df) {
  miss <- setdiff(names(train_df), names(test_df))
  if (length(miss) > 0) for (cc in miss) test_df[[cc]] <- NA
  
  extra <- setdiff(names(test_df), names(train_df))
  if (length(extra) > 0) test_df <- test_df[, setdiff(names(test_df), extra), drop = FALSE]
  
  fac_cols <- names(train_df)[vapply(train_df, is.factor, logical(1))]
  for (cc in fac_cols) {
    if (cc %in% names(test_df)) test_df[[cc]] <- factor(test_df[[cc]], levels = levels(train_df[[cc]]))
  }
  
  test_df <- test_df[, names(train_df), drop = FALSE]
  test_df
}

#==== 0A - Data Preparation ===================================================#
tryCatch({
  
  ##=========================================##
  ##==== Dataset preparation.
  ##=========================================##
  
  ### Base Model (only add id for fold splitting)
  Train_Data_Base_Model <- Train_Transformed %>%
    dplyr::mutate(id = Train_with_id$id)
  
  ### Strategy A: latent features
  Train_Data_Strategy_A <- cbind(Train_Transformed, Strategy_A_LF)
  
  ### Strategy B: anomaly score
  Train_Data_Strategy_B <- Strategy_B_AS
  
  ### Strategy C: feature denoising (DAE latent features)
  Train_Data_Strategy_C <- Strategy_C
  
  ### Strategy D: cash & profit feature engineering
  Train_Data_Strategy_D <- Strategy_D
  
  
  ##=========================================##
  ##==== First stratify by IDs.
  ##=========================================##
  Data_Train_CV_Split_IDs <- MVstratifiedsampling_CV_ID(
    data      = Train_Data_Base_Model,
    num_folds = N_folds
  )
  
  ##=========================================##
  ##==== Use the stratified IDs to get fold vector
  ##=========================================##
  Data_Train_CV_Base_Model <- MVstratifiedsampling_CV_Split(
    data              = Train_Data_Base_Model,
    firm_fold_indices = Data_Train_CV_Split_IDs
  )
  
  ##=========================================##
  ##==== Remove the ID column once more.
  ##=========================================##
  Train_Data_Base_Model <- Train_Data_Base_Model %>% dplyr::select(-id)
  
  ##=========================================##
  ##==== Build TEST datasets (recommended)
  ##=========================================##
  Test_Data_Base_Model <- Test_Transformed
  Test_Data_Strategy_A <- cbind(Test_Transformed, Strategy_A_LF_Test)
  Test_Data_Strategy_B <- Strategy_B_AS_Test
  Test_Data_Strategy_C <- Strategy_C_Test
  Test_Data_Strategy_D <- Strategy_D_Test
  
  ## Align TEST to TRAIN (very important for factors + column order)
  Test_Data_Base_Model <- align_test_to_train(Train_Data_Base_Model, Test_Data_Base_Model)
  Test_Data_Strategy_A <- align_test_to_train(Train_Data_Strategy_A, Test_Data_Strategy_A)
  Test_Data_Strategy_B <- align_test_to_train(Train_Data_Strategy_B, Test_Data_Strategy_B)
  Test_Data_Strategy_C <- align_test_to_train(Train_Data_Strategy_C, Test_Data_Strategy_C)
  Test_Data_Strategy_D <- align_test_to_train(Train_Data_Strategy_D, Test_Data_Strategy_D)
  
}, error = function(e) message("Data Preparation Error: ", e))


#==============================================================================#
#==== 01 - GLM Model Training (Base + A + B + C + D) ==========================#
#==============================================================================#

##==============================##
## General Parameters.
##==============================##
n_init_points <- 2
n_iter_bayes  <- 4

# shared fold vector
Data_Train_CV_Vector <- Data_Train_CV_Base_Model[["fold_vector"]]

#==== 01A - Base model ========================================================#
tryCatch({
  GLM_Results_BaseModel <- GLM_Training(
    Data_Train_CV_Vector = Data_Train_CV_Vector,
    Train_Data           = Train_Data_Base_Model,
    n_init_points        = n_init_points,
    n_iter_bayes         = n_iter_bayes
  )
}, error = function(e) message("Training Base Model Error: ", e))

#==== 01B - Strategy A ========================================================#
tryCatch({
  GLM_Results_Strategy_A <- GLM_Training(
    Data_Train_CV_Vector = Data_Train_CV_Vector,
    Train_Data           = Train_Data_Strategy_A,
    n_init_points        = n_init_points,
    n_iter_bayes         = n_iter_bayes
  )
}, error = function(e) message("Training Strategy A Error: ", e))

#==== 01C - Strategy B ========================================================#
tryCatch({
  GLM_Results_Strategy_B <- GLM_Training(
    Data_Train_CV_Vector = Data_Train_CV_Vector,
    Train_Data           = Train_Data_Strategy_B,
    n_init_points        = n_init_points,
    n_iter_bayes         = n_iter_bayes
  )
}, error = function(e) message("Training Strategy B Error: ", e))

#==== 01D - Strategy C ========================================================#
tryCatch({
  GLM_Results_Strategy_C <- GLM_Training(
    Data_Train_CV_Vector = Data_Train_CV_Vector,
    Train_Data           = Train_Data_Strategy_C,
    n_init_points        = n_init_points,
    n_iter_bayes         = n_iter_bayes
  )
}, error = function(e) message("Training Strategy C Error: ", e))

#==== 01E - Strategy D ========================================================#
tryCatch({
  GLM_Results_Strategy_D <- GLM_Training(
    Data_Train_CV_Vector = Data_Train_CV_Vector,
    Train_Data           = Train_Data_Strategy_D,
    n_init_points        = n_init_points,
    n_iter_bayes         = n_iter_bayes
  )
}, error = function(e) message("Training Strategy D Error: ", e))



#==============================================================================#
#==== 02 - GLM Model Comparison (AUC and Parameters) ==========================#
#==============================================================================#
tryCatch({
  
  extract_metrics <- function(model_obj, model_name) {
    
    # Safety Check: If model didn't train, return NAs
    if (is.null(model_obj) || is.null(model_obj$results)) {
      warning(paste("Model", model_name, "is missing. Returning NA row."))
      return(data.frame(Model = model_name, AUC = NA, Brier_Score = NA, 
                        Penalized_Brier = NA, Model_Type = "Error"))
    }
    
    # 1. Standard Metrics (Available in both)
    best_auc  <- model_obj$results$AUC[1]
    brier_val <- if(!is.null(model_obj$Brier_Score)) model_obj$Brier_Score else NA
    
    # 2. Penalized Brier Score (Specific to GLM usually)
    penalized_brier <- if(!is.null(model_obj$Penalized_Brier_Score)) model_obj$Penalized_Brier_Score else NA
    
    # 3. Parameter Extraction (Conditional)
    params <- model_obj$optimal_parameters
    
    # Initialize defaults
    eta <- NA; depth <- NA; subs <- NA; colsample <- NA; rounds <- NA
    alpha <- NA; lambda <- NA
    model_type <- "Unknown"
    
    # Detect XGBoost (has "eta")
    if ("eta" %in% names(params)) {
      model_type <- "XGBoost"
      eta        <- params$eta
      depth      <- params$max_depth
      subs       <- params$subsample
      colsample  <- params$colsample_bytree
      rounds     <- model_obj$optimal_rounds
    } 
    # Detect GLM (has "alpha")
    else if ("alpha" %in% names(params)) {
      model_type <- "GLM"
      alpha      <- params$alpha
      lambda     <- params$lambda
    }
    
    # 4. Return Dataframe
    data.frame(
      Model = model_name,
      Type  = model_type,
      AUC   = best_auc,
      Brier_Score = brier_val,
      Penalized_Brier = penalized_brier,
      
      # Model Specific Params
      Alpha  = alpha,
      Lambda = lambda,
      Rounds = rounds,
      Eta    = eta,
      Max_Depth = depth
    )
  }
  
}, error = function(e) message("Function Definition Error: ", e))


tryCatch({
  
  comparison_table_raw <- dplyr::bind_rows(
    extract_metrics(GLM_Results_BaseModel,  "Base Model"),
    extract_metrics(GLM_Results_Strategy_A, "Strategy A"),
    extract_metrics(GLM_Results_Strategy_B, "Strategy B"),
    extract_metrics(GLM_Results_Strategy_C, "Strategy C"),
    extract_metrics(GLM_Results_Strategy_D, "Strategy D")
  )
  
  # --- ensure base exists BEFORE filtering ---
  base_row <- comparison_table_raw %>%
    dplyr::filter(Model == "Base Model") %>%
    dplyr::slice(1)
  
  if (nrow(base_row) == 0) stop("Base Model row not found (Model == 'Base Model'). Check naming.")
  
  
  safe_uplift_pct <- function(x, base, higher_is_better = TRUE) {
    ok <- is.finite(x) & is.finite(base) & base != 0
    out <- rep(0, length(x))
    out[ok] <- (x[ok] - base) / base * 100
    if (!higher_is_better) out[ok] <- -out[ok]
    out
  }
  
  comparison_table <- comparison_table_raw %>%
    # Filter out failed models (but keep base in raw already)
    dplyr::filter(!is.na(AUC)) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(
      Base_AUC   = base_row$AUC,
      Base_Brier = base_row$Brier_Score,
      Base_PBS   = base_row$Penalized_Brier,
      
      Uplift_AUC_pct   = safe_uplift_pct(AUC,            Base_AUC,   higher_is_better = TRUE),
      Uplift_Brier_pct = safe_uplift_pct(Brier_Score,    Base_Brier, higher_is_better = FALSE),
      Uplift_PBS_pct   = safe_uplift_pct(Penalized_Brier,Base_PBS,   higher_is_better = FALSE)
    ) %>%
    dplyr::select(
      Model, Type,
      AUC, Uplift_AUC_pct,
      Brier_Score, Uplift_Brier_pct,
      Penalized_Brier, Uplift_PBS_pct,
      Alpha, Lambda
    )
  
  print("--- Final GLM Strategy Leaderboard ---")
  print(comparison_table)
  
}, error = function(e) message("Comparison Error: ", e))


#==============================================================================#
#==== 03 - GLM Model in the test-set ==========================================#
#==============================================================================#

#==== 03A - Data Preparation ==================================================#

# helper: align test factors/columns to train (prevents model.matrix/factor issues)
align_test_to_train <- function(train_df, test_df) {
  miss <- setdiff(names(train_df), names(test_df))
  if (length(miss) > 0) for (cc in miss) test_df[[cc]] <- NA
  
  extra <- setdiff(names(test_df), names(train_df))
  if (length(extra) > 0) test_df <- test_df[, setdiff(names(test_df), extra), drop = FALSE]
  
  fac_cols <- names(train_df)[vapply(train_df, is.factor, logical(1))]
  for (cc in fac_cols) {
    if (cc %in% names(test_df)) test_df[[cc]] <- factor(test_df[[cc]], levels = levels(train_df[[cc]]))
  }
  
  test_df <- test_df[, names(train_df), drop = FALSE]
  test_df
}

tryCatch({
  
  ### Base model.
  Test_Data_Base_Model <- Test_Transformed
  
  ### Strategy A: latent features.
  Final_Test_Set_A <- cbind(Test_Transformed, Strategy_A_LF_Test)
  Test_Data_Strategy_A <- Final_Test_Set_A
  
  ### Strategy B: anomaly score.
  Test_Data_Strategy_B <- Strategy_B_AS_Test
  
  ### Strategy C: DAE denoising (latent features).
  Test_Data_Strategy_C <- Strategy_C_Test
  
  ### Strategy D: Cash & Profit feature engineering.
  Test_Data_Strategy_D <- Strategy_D_Test
  
  ## Align test sets to the corresponding train sets (recommended)
  Test_Data_Base_Model <- align_test_to_train(Train_Data_Base_Model, Test_Data_Base_Model)
  Test_Data_Strategy_A <- align_test_to_train(Train_Data_Strategy_A, Test_Data_Strategy_A)
  Test_Data_Strategy_B <- align_test_to_train(Train_Data_Strategy_B, Test_Data_Strategy_B)
  Test_Data_Strategy_C <- align_test_to_train(Train_Data_Strategy_C, Test_Data_Strategy_C)
  Test_Data_Strategy_D <- align_test_to_train(Train_Data_Strategy_D, Test_Data_Strategy_D)
  
}, error = function(e) message("Test Data Prep Error: ", e))


#==============================================================================#
#==== 03B - Base Model ========================================================#
#==============================================================================#

tryCatch({
  
  Model     <- GLM_Results_BaseModel$optimal_model
  Test_Data <- Test_Data_Base_Model
  
  GLM_Test_Results_BaseModel <- GLM_Test(Model = Model, Test_Data = Test_Data)
  
}, error = function(e) message("Test Base Model Error: ", e))


#==============================================================================#
#==== 03C - Strategy A ========================================================#
#==============================================================================#

tryCatch({
  
  Model     <- GLM_Results_Strategy_A$optimal_model
  Test_Data <- Test_Data_Strategy_A
  
  GLM_Test_Results_Strategy_A <- GLM_Test(Model = Model, Test_Data = Test_Data)
  
}, error = function(e) message("Test Strategy A Error: ", e))


#==============================================================================#
#==== 03D - Strategy B ========================================================#
#==============================================================================#

tryCatch({
  
  Model     <- GLM_Results_Strategy_B$optimal_model
  Test_Data <- Test_Data_Strategy_B
  
  GLM_Test_Results_Strategy_B <- GLM_Test(Model = Model, Test_Data = Test_Data)
  
}, error = function(e) message("Test Strategy B Error: ", e))


#==============================================================================#
#==== 03E - Strategy C (DAE) ==================================================#
#==============================================================================#

tryCatch({
  
  Model     <- GLM_Results_Strategy_C$optimal_model
  Test_Data <- Test_Data_Strategy_C
  
  GLM_Test_Results_Strategy_C <- GLM_Test(Model = Model, Test_Data = Test_Data)
  
}, error = function(e) message("Test Strategy C Error: ", e))


#==============================================================================#
#==== 03F - Strategy D (Cash & Profit) ========================================#
#==============================================================================#

tryCatch({
  
  Model     <- GLM_Results_Strategy_D$optimal_model
  Test_Data <- Test_Data_Strategy_D
  
  GLM_Test_Results_Strategy_D <- GLM_Test(Model = Model, Test_Data = Test_Data)
  
}, error = function(e) message("Test Strategy D Error: ", e))


#==============================================================================#
#==== 04A - Leaderboard on TEST  =============================#
#==============================================================================#





tryCatch({
  
  comparison_table_test_raw <- dplyr::bind_rows(
    extract_test_metrics(GLM_Test_Results_BaseModel,  Test_Data_Base_Model,  GLM_Results_BaseModel,  "Base Model"),
    extract_test_metrics(GLM_Test_Results_Strategy_A, Test_Data_Strategy_A,  GLM_Results_Strategy_A, "Strategy A"),
    extract_test_metrics(GLM_Test_Results_Strategy_B, Test_Data_Strategy_B,  GLM_Results_Strategy_B, "Strategy B"),
    extract_test_metrics(GLM_Test_Results_Strategy_C, Test_Data_Strategy_C,  GLM_Results_Strategy_C, "Strategy C"),
    extract_test_metrics(GLM_Test_Results_Strategy_D, Test_Data_Strategy_D,  GLM_Results_Strategy_D, "Strategy D")
  )
  
  # --- extract base BEFORE filtering ---
  base_row <- comparison_table_test_raw %>%
    dplyr::filter(Model == "Base Model") %>%
    dplyr::slice(1)
  
  if (nrow(base_row) == 0) stop("Base Model row not found (Model == 'Base Model'). Check naming.")
  
  # helper: safe pct uplift (returns 0 instead of NA)
  safe_uplift_pct <- function(x, base, higher_is_better = TRUE) {
    ok <- is.finite(x) & is.finite(base) & base != 0
    out <- rep(0, length(x))
    out[ok] <- (x[ok] - base) / base * 100
    if (!higher_is_better) out[ok] <- -out[ok]
    out
  }
  
  comparison_table_test <- comparison_table_test_raw %>%
    # If you want to exclude models where AUC can't be computed, keep this:
    dplyr::filter(!is.na(AUC)) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(
      Base_AUC   = base_row$AUC,
      Base_Brier = base_row$Brier_Score,
      Base_PBS   = base_row$Penalized_Brier,
      
      Uplift_AUC_pct   = safe_uplift_pct(AUC,             Base_AUC,   higher_is_better = TRUE),
      Uplift_Brier_pct = safe_uplift_pct(Brier_Score,     Base_Brier, higher_is_better = FALSE),
      Uplift_PBS_pct   = safe_uplift_pct(Penalized_Brier, Base_PBS,   higher_is_better = FALSE)
    ) %>%
    dplyr::select(
      Model, Type,
      AUC, Uplift_AUC_pct,
      Brier_Score, Uplift_Brier_pct,
      Penalized_Brier, Uplift_PBS_pct,
      Alpha, Lambda
    )
  
  print("--- Final TEST-SET Strategy Leaderboard ---")
  print(comparison_table_test)
  
}, error = function(e) message("TEST Comparison Error: ", e))




#==============================================================================#
#==== 05 - Variable importance via GLM coefficients (per strategy) ============#
#==============================================================================#

library(dplyr)

#--- helper: extract & rank coefficients from glmnet-style model --------------
get_top_coeffs_glmnet <- function(model, model_name = "Model", top_n = 20) {
  if (is.null(model)) {
    warning(paste(model_name, "model is NULL"))
    return(tibble(
      Strategy = model_name, Feature = character(),
      Coef = numeric(), AbsCoef = numeric(),
      OddsRatio = numeric()
    ))
  }
  
  # coef(model) should work for glmnet/cv.glmnet objects
  cc <- tryCatch(stats::coef(model), error = function(e) NULL)
  if (is.null(cc)) {
    warning(paste("Could not extract coefficients for", model_name))
    return(tibble(
      Strategy = model_name, Feature = character(),
      Coef = numeric(), AbsCoef = numeric(),
      OddsRatio = numeric()
    ))
  }
  
  # make it a tidy data frame
  if (inherits(cc, "dgCMatrix") || is.matrix(cc)) {
    cc_mat <- as.matrix(cc)
    out <- tibble(
      Feature = rownames(cc_mat),
      Coef    = as.numeric(cc_mat[, 1])
    )
  } else {
    # fallback (rare)
    out <- tibble(Feature = names(cc), Coef = as.numeric(cc))
  }
  
  out %>%
    filter(!is.na(Coef), Feature != "(Intercept)") %>%
    mutate(
      Strategy  = model_name,
      AbsCoef   = abs(Coef),
      OddsRatio = exp(Coef)   # logistic interpretation: multiplicative change in odds
    ) %>%
    arrange(desc(AbsCoef)) %>%
    slice_head(n = top_n) %>%
    select(Strategy, Feature, Coef, AbsCoef, OddsRatio)
}


models_list <- list(
  "Base Model" = GLM_Results_BaseModel$optimal_model,
  "Strategy A" = GLM_Results_Strategy_A$optimal_model,
  "Strategy B" = GLM_Results_Strategy_B$optimal_model,
  "Strategy C" = GLM_Results_Strategy_C$optimal_model,
  "Strategy D" = GLM_Results_Strategy_D$optimal_model
)


top_n <- 20

importance_all <- bind_rows(lapply(names(models_list), function(nm) {
  get_top_coeffs_glmnet(models_list[[nm]], model_name = nm, top_n = top_n)
}))


print(importance_all)

importance_by_strategy <- split(importance_all, importance_all$Strategy)
for (nm in names(importance_by_strategy)) {
  cat("\n====================\n", nm, "— Top", top_n, "features\n====================\n")
  print(importance_by_strategy[[nm]])
}

# Optional: save to csv
# write.csv(importance_all, "glm_top_coeffs_by_strategy.csv", row.names = FALSE)




# install.packages("openxlsx")  # run once if not installed
library(openxlsx)

# Define path
file_path <- "/Users/admin/Desktop/Industry Lab/01_Code/GLM_with_ratios_Variable_Importance_By_Strategy_new.xlsx"

# Create workbook
wb <- createWorkbook()

# Split by strategy (clean professional layout)
importance_split <- split(importance_all, importance_all$Strategy)

for (nm in names(importance_split)) {
  addWorksheet(wb, nm)
  writeData(wb, nm, importance_split[[nm]])
  setColWidths(wb, nm, cols = 1:5, widths = "auto")
}

# Save file
saveWorkbook(wb, file_path, overwrite = TRUE)

cat("File saved to:\n", file_path)



#==============================================================================#
#==== 05B - TEST-set variable contributions per strategy ====================#
#==============================================================================#

library(dplyr)

#--- helper: safe coefficient vector from glmnet/cv.glmnet --------------------
get_coef_vec <- function(model) {
  cc <- as.matrix(stats::coef(model))
  # named numeric vector
  setNames(as.numeric(cc[, 1]), rownames(cc))
}


build_x_matrix <- function(df) {
  stats::model.matrix(y ~ ., data = df)  # includes (Intercept)
}


get_test_contrib <- function(model, test_df, strategy_name = "Model", top_n = 20) {
  if (is.null(model) || is.null(test_df)) {
    warning(paste("Missing model or test_df for", strategy_name))
    return(tibble(
      Strategy = strategy_name, Feature = character(),
      Coef = numeric(), MeanAbsContrib = numeric(),
      MeanContrib = numeric()
    ))
  }
  
  beta <- tryCatch(get_coef_vec(model), error = function(e) NULL)
  if (is.null(beta)) {
    warning(paste("Could not extract coefficients for", strategy_name))
    return(tibble(
      Strategy = strategy_name, Feature = character(),
      Coef = numeric(), MeanAbsContrib = numeric(),
      MeanContrib = numeric()
    ))
  }
  
  X <- tryCatch(build_x_matrix(test_df), error = function(e) NULL)
  if (is.null(X)) {
    warning(paste("Could not build model.matrix for", strategy_name))
    return(tibble(
      Strategy = strategy_name, Feature = character(),
      Coef = numeric(), MeanAbsContrib = numeric(),
      MeanContrib = numeric()
    ))
  }
  
  # align columns between X and beta (important!)
  common <- intersect(colnames(X), names(beta))
  if (!"(Intercept)" %in% common) {
    warning(paste(strategy_name, ": intercept missing in common set (ok, continuing)."))
  }
  
  Xc <- X[, common, drop = FALSE]
  bc <- beta[common]
  
  contrib <- sweep(Xc, 2, bc, `*`)
  
  if ("(Intercept)" %in% colnames(contrib)) {
    contrib <- contrib[, setdiff(colnames(contrib), "(Intercept)"), drop = FALSE]
    bc <- bc[setdiff(names(bc), "(Intercept)")]
  }
  
  tibble(
    Strategy = strategy_name,
    Feature  = colnames(contrib),
    Coef     = as.numeric(bc[colnames(contrib)]),
    MeanAbsContrib = colMeans(abs(contrib), na.rm = TRUE),
    MeanContrib    = colMeans(contrib, na.rm = TRUE)
  ) %>%
    arrange(desc(MeanAbsContrib)) %>%
    slice_head(n = top_n)
}


test_items <- list(
  list(name = "Base Model", model = GLM_Results_BaseModel$optimal_model,  test_df = Test_Data_Base_Model),
  list(name = "Strategy A", model = GLM_Results_Strategy_A$optimal_model, test_df = Test_Data_Strategy_A),
  list(name = "Strategy B", model = GLM_Results_Strategy_B$optimal_model, test_df = Test_Data_Strategy_B),
  list(name = "Strategy C", model = GLM_Results_Strategy_C$optimal_model, test_df = Test_Data_Strategy_C),
  list(name = "Strategy D", model = GLM_Results_Strategy_D$optimal_model, test_df = Test_Data_Strategy_D)
)

top_n <- 20

test_contrib_all <- bind_rows(lapply(test_items, function(it) {
  get_test_contrib(it$model, it$test_df, strategy_name = it$name, top_n = top_n)
}))

# 1) combined table
print(test_contrib_all)

# 2) print per strategy
by_strat <- split(test_contrib_all, test_contrib_all$Strategy)
for (nm in names(by_strat)) {
  cat("\n====================\n", nm, "— TEST mean |x*beta| top", top_n, "\n====================\n")
  print(by_strat[[nm]])
}


library(openxlsx)
library(dplyr)

# ---- Add ranking column ----
test_contrib_all_ranked <- test_contrib_all %>%
  group_by(Strategy) %>%
  arrange(desc(MeanAbsContrib), .by_group = TRUE) %>%
  mutate(Rank = row_number()) %>%
  ungroup() %>%
  select(Strategy, Rank, Feature, Coef, MeanAbsContrib, MeanContrib)

# ---- Define file path ----
file_path_test <- "/Users/admin/Desktop/Industry Lab/01_Code/GLM_TEST_with_ratios_Variable_Importance_By_Strategy_new.xlsx"


wb <- createWorkbook()


test_split <- split(test_contrib_all_ranked, test_contrib_all_ranked$Strategy)

for (nm in names(test_split)) {
  addWorksheet(wb, nm)
  writeData(wb, nm, test_split[[nm]])
  
  # Auto column width
  setColWidths(wb, nm, cols = 1:6, widths = "auto")
}

saveWorkbook(wb, file_path_test, overwrite = TRUE)

cat("TEST Variable Importance file saved to:\n", file_path_test)







