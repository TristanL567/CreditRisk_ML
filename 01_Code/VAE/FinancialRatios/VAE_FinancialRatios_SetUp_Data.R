#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

## Last changed: 2026-02-04 | Tristan Leiter
## Data leakage might be an issue since the VAE is not trained separately for all folds
## but on all the training data.

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("dplyr", "caret", "lubridate", "purrr", "tidyr",
              "Matrix", "pROC",            ## Sparse Matrices and efficient AUC computation.
              "glmnet",                    ## GLM library.
              "xgboost",                   ## XGBoost library.
              "rBayesianOptimization",     ## Bayesian Optimization.
              "ggplot2", "Ckmeans.1d.dp",  ## Plotting & Charts | XG-Charts / Feature Importance.
              "scales",                    ## ggplot2 extension for nice charts.
              "ggrepel",                   ## Non-overlapping ggplot2 text labels.
              "ranger",                    ## Random Forest library.
              "mlr3", "mlr3learners", "mlr3tuning",
              "mlr3mbo", "mlr3measures", "data.table",
              "paradox", "future", 
              "future.apply", "parallel",
              "adabag",
              "purrr", "tibble",
              "autotab", "keras",           ## VAE.
              "reticulate", "tensorflow",
              "mclust",                     ## Finding soft interaction features.
              "pdp", "gridExtra", "hexbin", "openxlsx",
              "tensorflow"
)

for(i in 1:length(packages)){
  package_name <- packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

#==== 1A - Python & Tensorflow dependencies ===================================#

# reticulate::install_python(version = "3.10")
# install_tensorflow(version = "2.10.0", envname = "r-reticulate")
# install_miniconda()

# Check if R can find the "real" python version
# py_versions <- py_discover_config()
# print(py_versions)

# use_python("C:/Users/TristanLeiter/OneDrive - Hinder Asset Management AG/Dokumente/.virtualenvs/r-reticulate/Scripts/python.exe", required = TRUE)

#==== 1B - Functions ==========================================================#

sourceFunctions <- function (functionDirectory)  {
  functionFiles <- list.files(path = functionDirectory, pattern = "*.R", 
                              full.names = T)
  ssource <- function(path) {
    try(source(path, echo = F, verbose = F, print.eval = T, 
               local = F))
  }
  sapply(functionFiles, ssource)
}

#==== 1C - Parameters =========================================================#

## Directories.
Data_Path <- "C:/Users/Tristan Leiter/Documents/Privat/ILAB/Data/WS2025" ## Needs to be set manually.
Data_Directory <- file.path(Data_Path, "data.rda")
Data_Directory_write <- file.path(Path, "02_Data")
Charts_Directory <- file.path(Path, "03_Charts")
Data_RF <- Path

## Charts Directories.
Charts_GLM_Directory <- file.path(Charts_Directory, "GLM")
Charts_RF_Directory <- file.path(Charts_Directory, "RF")
Charts_XGBoost_Directory <- file.path(Charts_Directory, "XGBoost")
Charts_TestSet_Directory <- file.path(Charts_Directory, "TestSet")

Functions_Directory <- file.path(Path, "01_Code/Subfunctions")
Functions_RF_Directory <- file.path(Path, "01_Code/RF_Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)
sourceFunctions(Functions_RF_Directory)

## Data Sampling.
set.seed(123)              ## Check seed.

## Charts.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"


width <- 3750
heigth <- 1833

## General Parameters for modeling.
N_folds <- 5

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02A - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

#==== 02B - Pre-process checks ================================================#

## Apply the simple methodology to exclude some observations (outlined by the Leitner Notes).
## Remove all ratios.

Data <- DataPreprocessing(Data, Tolerance = 2)

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .))) %>%
  drop_na()

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

DivideByTotalAssets <- FALSE
Quantile_Transform <- TRUE

#==== 03A - Incorporate deviations from sector means and time dependency ======#

tryCatch({
  
  message("--- Starting Feature Engineering ---")
  
  # A. Setup & Ratio Calculation
  Data_Eng <- Data %>%
    group_by(id) %>%
    arrange(refdate) %>% 
    mutate(
      time_index = row_number(),
      year = year(refdate),
      is_new_company = ifelse(time_index == 1, 1, 0),
      
      # --- 1. Corrected Definitions ---
      r7_Corrected = (f10 + f11) / (f1 + 1e-6),
      r8_Corrected = (f10 + f11 - f5) / (f1 + 1e-6),
      r10_Structure = (f3 - f5) / (f1 + 1e-6),
      
      # Growth Calculation
      # Added 1e-6 to denominator to prevent division by zero
      Asset_Growth_Pct = (r1 - lag(r1, 1)) / (lag(r1, 1) + 1e-6)
    ) %>%
    ungroup()
  
  # B. Sector Benchmarks (Z-Scores)
  target_ratios <- c("r9", "r14", "r8_Corrected", "r16", "r12", 
                     "r10_Structure", "r7_Corrected", "r11", "r5", "r17")
  
  message(paste("Benchmarking", length(target_ratios), "features..."))
  
  Data_Eng <- Data_Eng %>%
    group_by(sector, year) %>%
    mutate(
      across(all_of(target_ratios), 
             list(
               sec_med = ~ median(., na.rm = TRUE),
               sec_mad = ~ mad(., constant = 1.4826, na.rm = TRUE)
             ),
             .names = "{.col}_{.fn}")
    ) %>%
    ungroup()
  
  # C. Calculate Z-Scores Loop
  for(ratio in target_ratios) {
    z_name   <- paste0(ratio, "_Sector_Zscore")
    med_name <- paste0(ratio, "_sec_med")
    mad_name <- paste0(ratio, "_sec_mad")
    
    Data_Eng[[z_name]] <- (Data_Eng[[ratio]] - Data_Eng[[med_name]]) / 
      ifelse(Data_Eng[[mad_name]] == 0, 1, Data_Eng[[mad_name]])
  }
  
  # Remove temp columns
  Data_Eng <- Data_Eng %>% select(-ends_with("_sec_med"), -ends_with("_sec_mad"))
  
  # D. Time Dependency
  Data_Eng <- Data_Eng %>%
    group_by(id) %>%
    mutate(
      r14_Delta = r14 - lag(r14, 1),
      r9_Delta  = r9 - lag(r9, 1),
      r14_Volatility_Exp = sqrt(cummean((r14 - cummean(r14))^2)),
      Profit_Trend_Consistent = ifelse(sign(r14_Delta) == sign(lag(r14_Delta)), 1, 0)
    ) %>%
    ungroup() %>%
    mutate(
      across(ends_with("_Delta"), ~ replace_na(., 0)),
      across(ends_with("_Volatility_Exp"), ~ replace_na(., 0)),
      Asset_Growth_Pct = replace_na(Asset_Growth_Pct, 0),
      Profit_Trend_Consistent = replace_na(Profit_Trend_Consistent, 0),
      risk_New_HighDebt = is_new_company * r7_Corrected
    )
  
  message("Feature Engineering Complete.")
  
  # E. Drop Raw Financial Positions (f1..f18)
  # We use 'intersect' to avoid errors if some columns are missing
  cols_to_remove <- paste0("f", 1:18)
  cols_present   <- intersect(names(Data_Eng), cols_to_remove)
  
  if(length(cols_present) > 0){
    Data_Eng <- Data_Eng %>% select(-all_of(cols_present))
    message(paste("Dropped", length(cols_present), "raw financial columns."))
  }
  
}, error = function(e) message("Error in Feature Engineering: ", e))  

#==== 03B - Data Sampling =====================================================#

tryCatch({
  
  message("--- Starting Data Split ---")
  set.seed(123)
  
  if(exists("MVstratifiedsampling")) {
    Data_Sampled <- MVstratifiedsampling(Data_Eng, strat_vars = c("sector", "y"), Train_size = 0.7)
    Train <- Data_Sampled[["Train"]]
    Test  <- Data_Sampled[["Test"]]
  } else {
    warning("MVstratifiedsampling not found. Using simple random split.")
    train_idx <- sample(1:nrow(Data_Eng), 0.7 * nrow(Data_Eng))
    Train <- Data_Eng[train_idx, ]
    Test  <- Data_Eng[-train_idx, ]
  }
  
  message(paste("Split Complete. Train:", nrow(Train), "Test:", nrow(Test)))
  
}, error = function(e) message("Error in Splitting: ", e))

#==== 03C - Quantile Transformation ===========================================#

tryCatch({
  
  message("--- Starting Quantile Transformation ---")
  
  if(exists("Quantile_Transform") && Quantile_Transform){
    
    # 1. Identify Columns to Transform
    # Exclude metadata, targets, and binary flags
    exclude_cols <- c("y", "id", "company_id", "row_id", "time_index", 
                      "refdate", "sector", "size", "groupmember", "public",
                      "is_new_company", "Profit_Trend_Consistent", "year")
    
    numeric_cols <- names(Train)[sapply(Train, is.numeric)]
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    message(paste("Transforming", length(cols_to_transform), "features..."))
    
    Train_Transformed <- Train
    Test_Transformed  <- Test 
    
    # 2. Transform Loop
    for (col in cols_to_transform) {
      if(col %in% names(Train) && col %in% names(Test)) {
        if(exists("QuantileTransformation")) {
          res <- QuantileTransformation(Train[[col]], Test[[col]])
          Train_Transformed[[col]] <- res$train
          Test_Transformed[[col]]  <- res$test
        }
      }
    }
    
    message("Transformation Complete.")
    
  } else {
    Train_Transformed <- Train
    Test_Transformed  <- Test
    message("Skipping Transformation.")
  }
  
}, error = function(e) message("Error in Transformation: ", e))

#==== 03D - Cleanup and factors ===============================================#

tryCatch({
  
  message("--- Starting Final Cleanup ---")
  
  # 1. define Metadata to Drop from Modeling Set
  cols_to_drop <- c("id", "company_id", "row_id", "time_index", "refdate", "year")
  
  # 2. Align Factors
  # Identify categorical predictors (exclude y and metadata)
  cat_cols <- names(Train_Transformed)[sapply(Train_Transformed, function(x) is.character(x) || is.factor(x))]
  cat_cols <- setdiff(cat_cols, c("y", cols_to_drop))
  
  for (col in cat_cols) {
    # Train
    Train_Transformed[[col]] <- as.factor(Train_Transformed[[col]])
    train_levels <- levels(Train_Transformed[[col]])
    
    # Test (Enforce Train levels)
    Test_Transformed[[col]] <- factor(Test_Transformed[[col]], levels = train_levels)
  }
  
  # 3. Handle Target 'y'
  if("y" %in% names(Train_Transformed)) {
    Train_Transformed$y <- factor(as.character(Train_Transformed$y), levels = c("0", "1"))
    Test_Transformed$y  <- factor(as.character(Test_Transformed$y), levels = c("0", "1"))
  }
  
  # 4. Create Final Sets
  Train_Final <- Train_Transformed[, !names(Train_Transformed) %in% cols_to_drop]
  Test_Final  <- Test_Transformed[, !names(Test_Transformed) %in% cols_to_drop]
  
  message("Processing Complete. Ready for Modelling.")
  glimpse(Train_Final)
  
}, error = function(e) message("Error in Cleanup: ", e))

#==============================================================================#
#==== 04 - VAE (Strategy A: latent features; Strategy B: anomaly score) =======#
#==============================================================================#

#==== 04A - Data preparation ==================================================#

tryCatch({
  
  # --- 1. Filter out Metadata & Target first ---
  # We only want predictors.
  # "y" is the target. "id", "refdate", "year", "time_index", "company_id" are metadata.
  meta_cols <- c("y", "id", "refdate", "year", "time_index", "company_id", "row_id")
  
  # Predictors only
  features_only <- Train_Final[, !names(Train_Final) %in% meta_cols]
  
  # --- 2. Identify Binary Features ---
  # explicitly list your binary 0/1 columns
  bin_cols <- c("groupmember", "public", "is_new_company", "Profit_Trend_Consistent")
  
  # --- 3. Identify Categorical Features (for One-Hot) ---
  # These are your factors like 'sector' and 'size'
  cat_cols <- names(features_only)[sapply(features_only, is.factor)]
  
  # --- 4. Identify Continuous Features ---
  # Everything else that is Numeric is a continuous feature (Ratios, Z-Scores, Deltas)
  # We take all numeric columns and subtract the known binary ones
  num_cols <- names(features_only)[sapply(features_only, is.numeric)]
  cont_cols <- setdiff(num_cols, bin_cols)
  
  message(paste("Continuous Features:", length(cont_cols)))
  message(paste("Binary Features:", length(bin_cols)))
  message(paste("Categorical Features:", length(cat_cols)))
  
  # --- 5. Construct the Input Matrix ---
  
  # A. Continuous Data (Normalized Ratios, Z-Scores)
  data_cont <- features_only[, cont_cols]
  
  # B. Binary Data (Flags)
  data_bin  <- features_only[, bin_cols]
  
  # C. One-Hot Encoded Categoricals
  # We use fullRank = FALSE to keep all levels (usually better for VAEs/Neural Nets)
  dummies_model <- dummyVars(~ ., data = features_only[, cat_cols])
  data_cat_onehot <- predict(dummies_model, newdata = features_only[, cat_cols]) %>% as.data.frame()
  
  # --- 6. Combine ---
  vae_input_data <- cbind(data_cont, data_bin, data_cat_onehot)
  
  # Final Check
  if(any(is.na(vae_input_data))) stop("Input data contains NAs!")
  
  # --- 7. Distribution Setup (Assuming this function exists in your env) ---
  feat_dist <- extracting_distribution(vae_input_data)
  set_feat_dist(feat_dist)
  
  ### Configuration:
  # Note: 64/32 neurons is small if you have many One-Hot columns. 
  # Check ncol(vae_input_data). If it's >100, consider 128->64.
  encoder_config <- list(
    list("dense", 128, "relu"),
    list("dense", 64, "relu")
  )
  
  decoder_config <- list(
    list("dense", 64, "relu"),
    list("dense", 128, "relu")
  )
  
}, error = function(e) message("VAE Prep Error: ", e))

#==== 04B - Train the Baseline VAE ============================================#

tryCatch({
  
vae_fit <- VAE_train(
  data = vae_input_data,
  encoder_info = encoder_config,
  decoder_info = decoder_config,
  latent_dim = 8,
  epoch = 100,
  batchsize = 256,
  beta = 0.5,
  lr = 0.001,
  temperature = 0.5,
  wait = 10,
  kl_warm = TRUE,
  beta_epoch = 10,
  Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0
)

}, error = function(e) message(e))

#==== 04C - Strategy A: Latent features (Dimensional reduction) ===============#

## Training Set.
tryCatch({
  
enc_weights <- Encoder_weights(
  encoder_layers = 2, 
  trained_model = vae_fit$trained_model,
  lip_enc = 0, pi_enc = 0, BNenc_layers = 0, learn_BN = 0
)

enc_model <- encoder_latent(
  encoder_input = vae_input_data, 
  encoder_info = encoder_config, 
  latent_dim = 8,
  Lip_en = 0, power_iterations = 0
) %>% keras::set_weights(enc_weights)

latent_output <- predict(enc_model, as.matrix(vae_input_data))
Strategy_A_LF <- as.data.frame(latent_output[[1]])
colnames(Strategy_A_LF) <- paste0("l", 1:8)

}, error = function(e) message(e))

## Test set.
tryCatch({
  
  message("--- Starting Strategy A Test Preparation (Robust) ---")
  
  # ===========================================================================
  # 1. SETUP & ALIGNMENT (Follow Training Methodology)
  # ===========================================================================
  
  # A. Separate Features (Drop Y)
  test_features_all <- Test_Final %>% select(-y)
  
  # B. Re-Verify Column Definitions (Use Training definitions)
  # (These should exist from the VAE Training step)
  if(!exists("cont_cols") || !exists("bin_cols") || !exists("cat_cols")) {
    stop("Critical: Feature lists (cont_cols, bin_cols, cat_cols) missing from environment.")
  }
  
  # C. Extract Subsets (Exactly as done in Training)
  test_cont_data <- test_features_all[, cont_cols]
  test_bin_data  <- test_features_all[, bin_cols]
  test_cat_data  <- test_features_all[, cat_cols]
  
  # ===========================================================================
  # 2. FEATURE PROCESSING
  # ===========================================================================
  
  # A. One-Hot Encoding
  # strictly apply the 'dummies_model' trained on training set
  test_cat_onehot <- predict(dummies_model, newdata = test_cat_data) %>% as.data.frame()
  
  # B. Combine to Raw VAE Input
  test_vae_input_raw <- cbind(test_cont_data, test_bin_data, test_cat_onehot)
  
  # ===========================================================================
  # 3. MATRIX ALIGNMENT (Crucial for Keras)
  # ===========================================================================
  
  # The test set might have missing or extra columns compared to training 
  # (e.g., missing a specific sector dummy). We must align it to 'vae_input_data'.
  
  # Target columns (from Training VAE input)
  target_cols <- colnames(vae_input_data)
  
  # Add Missing Columns (Fill with 0)
  missing_cols <- setdiff(target_cols, colnames(test_vae_input_raw))
  if(length(missing_cols) > 0) {
    missing_mat <- matrix(0, nrow = nrow(test_vae_input_raw), ncol = length(missing_cols))
    colnames(missing_mat) <- missing_cols
    test_vae_input_raw <- cbind(test_vae_input_raw, as.data.frame(missing_mat))
  }
  
  # Remove Extra Columns
  extra_cols <- setdiff(colnames(test_vae_input_raw), target_cols)
  if(length(extra_cols) > 0) {
    test_vae_input_raw <- test_vae_input_raw[, !colnames(test_vae_input_raw) %in% extra_cols]
  }
  
  # Reorder to match Training EXACTLY
  test_vae_input_final <- test_vae_input_raw[, target_cols]
  
  # Convert to standard Matrix (Keras requires this)
  x_test_matrix <- as.matrix(test_vae_input_final)
  
  # ===========================================================================
  # 4. PREDICT LATENT FEATURES
  # ===========================================================================
  
  message(paste("Generating Latent Features for", nrow(x_test_matrix), "rows..."))
  
  # Use Keras predict
  latent_output_raw <- predict(enc_model, x_test_matrix)
  
  # If output is a list (common in some Keras models), take the first element
  if(is.list(latent_output_raw)) {
    latent_values <- latent_output_raw[[1]]
  } else {
    latent_values <- latent_output_raw
  }
  
  # Create DataFrame
  Strategy_A_LF_Test <- as.data.frame(latent_values)
  colnames(Strategy_A_LF_Test) <- paste0("l", 1:8)
  
  # ===========================================================================
  # 5. CREATE FINAL DATASET FOR XGBOOST
  # ===========================================================================
  
  # Identify Metadata to drop
  cols_to_drop <- c("id", "refdate", "time_index", "year", "row_id", "company_id")
  
  # Clean Base Features (Keep predictors + y, drop metadata)
  # We use Test_Final (which includes y) so we have targets for evaluation later if needed
  Test_Clean_Base <- Test_Final[, !names(Test_Final) %in% cols_to_drop]
  
  # Combine Base Predictors + Latent Features
  # Note: Test_Clean_Base includes 'y'. Latent features are predictors.
  Strategy_A_Test <- cbind(Test_Clean_Base, Strategy_A_LF_Test)
  
  message("Success: Strategy_A_Test created.")
  print(dim(Strategy_A_Test))
  
}, error = function(e) message("Strategy A Prep Error: ", e))

#==== 04C - Strategy B: Anomaly Score =========================================#

## Training set.
tryCatch({
  
  message("--- Calculating Training Anomaly Scores ---")
  
  # 1. Get Reconstruction
  reconstructed_list <- predict(vae_fit$trained_model, as.matrix(vae_input_data))
  
  # Handle Keras output (list vs tensor)
  if(is.list(reconstructed_list)) {
    reconstructed_data <- as.matrix(reconstructed_list[[1]])
  } else {
    reconstructed_data <- as.matrix(reconstructed_list)
  }
  
  # 2. Define Dimensions (Using Sources of Truth)
  # We rely on 'cont_cols' from the VAE Prep block if available
  if(exists("cont_cols")) {
    n_cont_cols <- length(cont_cols)
  } else if (exists("data_cont")) {
    n_cont_cols <- ncol(data_cont)
  } else {
    stop("Critical: Cannot determine number of continuous columns. 'cont_cols' missing.")
  }
  
  n_total_cols <- ncol(vae_input_data)
  n_cat_cols   <- n_total_cols - n_cont_cols # Includes Binary + OneHot
  
  # 3. Extract Matrices
  # Continuous Part (First N columns)
  if (n_cont_cols > 0) {
    input_cont <- as.matrix(vae_input_data[, 1:n_cont_cols, drop = FALSE])
    recon_cont <- as.matrix(reconstructed_data[, 1:n_cont_cols, drop = FALSE])
  }
  
  # Categorical/Binary Part (Remaining columns)
  if (n_cat_cols > 0) {
    cat_indices <- (n_cont_cols + 1):n_total_cols
    input_cat   <- as.matrix(vae_input_data[, cat_indices, drop = FALSE])
    recon_cat   <- as.matrix(reconstructed_data[, cat_indices, drop = FALSE])
  }
  
  # 4. Calculate Scores
  
  # A. Continuous Score (MSE)
  if (n_cont_cols > 0) {
    mse_raw <- (input_cont - recon_cont)^2
    # Clip extreme overflows
    mse_raw[is.infinite(mse_raw)] <- 1e9 
    
    # Row-wise Sum / Count
    mse_normalized <- rowSums(mse_raw, na.rm = TRUE) / n_cont_cols
  } else {
    mse_normalized <- numeric(nrow(vae_input_data))
  }
  
  # B. Categorical Score (BCE for 0/1 data)
  if (n_cat_cols > 0) {
    epsilon   <- 1e-15
    recon_cat <- pmax(pmin(recon_cat, 1 - epsilon), epsilon)
    
    bce_raw <- -(input_cat * log(recon_cat) + (1 - input_cat) * log(1 - recon_cat))
    
    # Row-wise Sum / Count
    bce_normalized <- rowSums(bce_raw, na.rm = TRUE) / n_cat_cols
  } else {
    bce_normalized <- numeric(nrow(vae_input_data))
  }
  
  # 5. Weighted Combination (Balanced)
  final_score <- mse_normalized + bce_normalized
  
  # 6. Apply to Dataframe
  Strategy_B_AS <- Train_Final %>%
    mutate(
      anomaly_score_cont_avg = mse_normalized,
      anomaly_score_cat_avg  = bce_normalized,
      anomaly_score_balanced = final_score 
    )
  
  print("Strategy B (Train) calculated successfully. Summary:")
  print(summary(Strategy_B_AS$anomaly_score_balanced))
  
}, error = function(e) message("Error in Strategy B Train: ", e))

## Test set.
tryCatch({
  
  message("--- Calculating Test Anomaly Scores ---")
  
  # 1. Get Reconstruction
  reconstructed_list <- predict(vae_fit$trained_model, as.matrix(test_matrix))
  
  if(is.list(reconstructed_list)) {
    test_recon_matrix <- as.matrix(reconstructed_list[[1]])
  } else {
    test_recon_matrix <- as.matrix(reconstructed_list)
  }
  
  # Clip Infinite Model Outputs
  test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix > 0] <- 1e9
  test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix < 0] <- -1e9
  test_recon_matrix[is.na(test_recon_matrix)] <- 0 
  
  # 2. Define Dimensions (Consistency Check)
  # USE THE SAME COUNT AS TRAINING
  if(exists("n_cont_cols")) {
    n_cont <- n_cont_cols
  } else if(exists("cont_cols")) {
    n_cont <- length(cont_cols)
  } else {
    # Fallback to test_cont if training vars are lost
    n_cont <- ncol(test_cont)
  }
  
  n_total <- ncol(test_matrix)
  n_cat   <- n_total - n_cont
  
  # 3. Calculate Scores
  
  # --- Continuous Scores (MSE) ---
  if (n_cont > 0) {
    # Slice matrices
    input_cont_test <- as.matrix(test_matrix[, 1:n_cont, drop = FALSE])
    recon_cont_test <- as.matrix(test_recon_matrix[, 1:n_cont, drop = FALSE])
    
    input_cont_test[is.infinite(input_cont_test)] <- 1e9 
    
    sq_error <- (input_cont_test - recon_cont_test)^2
    sq_error[is.infinite(sq_error)] <- 1e9 
    
    mse_norm_test <- rowSums(sq_error, na.rm = TRUE) / n_cont
  } else {
    mse_norm_test <- numeric(nrow(test_matrix)) 
  }
  
  # --- Categorical Scores (BCE) ---
  if (n_cat > 0) {
    # Slice matrices
    cat_indices      <- (n_cont + 1):n_total
    input_cat_test   <- as.matrix(test_matrix[, cat_indices, drop = FALSE])
    recon_cat_test   <- as.matrix(test_recon_matrix[, cat_indices, drop = FALSE])
    
    epsilon <- 1e-15
    recon_cat_test <- pmax(pmin(recon_cat_test, 1 - epsilon), epsilon)
    
    bce_raw <- -(input_cat_test * log(recon_cat_test) + (1 - input_cat_test) * log(1 - recon_cat_test))
    
    bce_norm_test <- rowSums(bce_raw, na.rm = TRUE) / n_cat
  } else {
    bce_norm_test <- numeric(nrow(test_matrix))
  }
  
  # 4. Weighted Combination
  final_score_test <- mse_norm_test + bce_norm_test
  
  # 5. Append to Test Data
  Strategy_B_AS_Test <- Test_Final %>%
    mutate(
      anomaly_score_cont_avg = mse_norm_test,
      anomaly_score_cat_avg  = bce_norm_test,
      anomaly_score_balanced = final_score_test
    )
  
  print("Success: Strategy B Test Set calculated.")
  print(summary(Strategy_B_AS_Test$anomaly_score_balanced))
  
}, error = function(e) message("Error in Test Strategy B: ", e))

#==== 04D - Strategy C: Feature Denoising =====================================#

## Training set.
tryCatch({
  
  message("--- Strategy C: Training DAE ---")
  
  # 1. Prepare Data
  # We use the correctly engineered 'vae_input_data' from the previous step
  x_train_clean <- as.matrix(vae_input_data)
  input_dim <- ncol(x_train_clean)
  
  message(paste("Training DAE on", input_dim, "features..."))
  
  # 2. Define the DAE Architecture (Functional API)
  input_layer <- layer_input(shape = c(input_dim))
  
  encoded <- input_layer %>%
    # Noise Injection: 0.1 stddev is standard for financial data
    layer_gaussian_noise(stddev = 0.1) %>% 
    layer_dense(units = 128, activation = "relu") %>% # Increased to match complex features
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu", name = "bottleneck") # Latent Space
  
  decoded <- encoded %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = input_dim, activation = "linear") 
  
  # 3. Compile
  dae_autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
  
  dae_autoencoder %>% compile(
    optimizer = "adam",
    loss = "mse"
  )
  
  # 4. Train
  # x = Clean Data, y = Clean Data. 
  # The Gaussian Noise layer corrupts 'x' internally during training.
  history <- dae_autoencoder %>% fit(
    x = x_train_clean, 
    y = x_train_clean, 
    epochs = 50, 
    batch_size = 256,
    shuffle = TRUE,
    validation_split = 0.1,
    verbose = 0
  )
  
  # 5. Extract Latent Features
  encoder_only <- keras_model(inputs = dae_autoencoder$input, 
                              outputs = get_layer(dae_autoencoder, "bottleneck")$output)
  
  latent_train <- predict(encoder_only, x_train_clean)
  
  # Create DataFrame
  Strategy_C_LF <- as.data.frame(latent_train)
  colnames(Strategy_C_LF) <- paste0("dae_l", 1:8)
  
  # Combine with original data
  Strategy_C <- cbind(Train_Final, Strategy_C_LF)
  
  message("DAE Training Complete. Latent features extracted.")
  
}, error = function(e) message("Strategy C Training Error: ", e))

## Test set.
tryCatch({
  
  message("--- Preparing Test Set for Strategy C ---")
  
  # 1. Separate Features from Metadata
  test_features <- Test_Final %>% select(-y)
  
  # 2. Select Columns using the SAME lists from Training (Robust)
  # (Assumes 'cont_cols' and 'bin_cols' exist from VAE Prep step)
  
  # Continuous (Ratios, Z-Scores, Deltas)
  test_cont <- test_features[, cont_cols]
  
  # Binary (Flags)
  test_bin  <- test_features[, bin_cols]
  
  # 3. One-Hot Encoding
  test_cat_onehot <- predict(dummies_model, newdata = test_features) %>% as.data.frame()
  
  # 4. Combine
  test_dae_input_data <- cbind(test_cont, test_bin, test_cat_onehot)
  
  # 5. ALIGNMENT (Crucial Step)
  # Ensure columns match Training exactly
  
  # A. Fill Missing
  missing_cols <- setdiff(colnames(vae_input_data), colnames(test_dae_input_data))
  if(length(missing_cols) > 0) {
    missing_df <- as.data.frame(matrix(0, nrow = nrow(test_dae_input_data), ncol = length(missing_cols)))
    colnames(missing_df) <- missing_cols
    test_dae_input_data <- cbind(test_dae_input_data, missing_df)
  }
  
  # B. Drop Extra
  extra_cols <- setdiff(colnames(test_dae_input_data), colnames(vae_input_data))
  if(length(extra_cols) > 0) {
    test_dae_input_data <- test_dae_input_data %>% select(-all_of(extra_cols))
  }
  
  # C. Reorder
  test_dae_input_data <- test_dae_input_data[, colnames(vae_input_data)]
  x_test_matrix <- as.matrix(test_dae_input_data)
  
  # 6. Predict Latent Features
  test_latent_dae <- predict(encoder_only, x_test_matrix)
  
  Strategy_C_LF_Test <- as.data.frame(test_latent_dae)
  colnames(Strategy_C_LF_Test) <- paste0("dae_l", 1:8)
  
  Strategy_C_Test <- cbind(Test_Final, Strategy_C_LF_Test)
  
  print("Success: Strategy C Test Set Prepared and Features Extracted.")
  
}, error = function(e) message("DAE Test Set Error: ", e))

#==== 04E - Strategy D: Feature engineering Cash & Profit =====================#

## Training set.
tryCatch({
  
  # Strategy_D <- Train_Transformed %>%
  #   mutate(
  #     ##======================================================================##
  #     ## Strategy 2: The "Zombie" Interactions (Diagonal Decision Boundaries)
  #     ##======================================================================##
  #     
  #     # 1. Support Structure Ratio
  #     # Logic: How much Debt (f11) is piled on top of the Equity position?
  #     # Interpretation: High Value = High Debt relative to Equity size.
  #     # Note: Added +0.1 to denominator to prevent division by zero near the median.
  #     # Ratio_Support_Structure = f11 / (abs(f6) + 0.0001),
  #     
  #     # 2. The "Distress Gap"
  #     # Logic: Difference between Liabilities and Equity. 
  #     # Zombies have High f11 (+0.36) and Low f6 (-1.2). Result: 0.36 - (-1.2) = 1.56 (High Score)
  #     # Healthy firms have Low f11 (-0.2) and High f6 (+0.5). Result: -0.7 (Low Score)
  #     Gap_Debt_Equity = f11 - f6,
  #     
  #     # 3. Cash Burn Ratio
  #     # Logic: Relates Cash (f5) to Profit (f8). 
  #     # Captures "Profitable but Illiquid" scenarios (The False Negatives).
  #     # If f8 is high (positive) but f5 is low (negative), this ratio drops.
  #     Ratio_Cash_Profit = f5 / (abs(f8) + 0.0001),
  #     Interaction_Cash_Profit = f5 * f8,
  #     Feature_Stabilizer = ifelse(f8 < 0, f5, f8)
  #     ##======================================================================##
  #     ## Strategy 3: The "Red Flag" Counter (Aggregating Risk)
  #     ##======================================================================##
  #     
  #     # Thresholds derived from your Density Cluster Analysis (Median values)
  #     
  #     # Flag 1: Liquidity Crisis (The "Silent Killer")
  #     # Analysis: Defaulters had median f5 = -0.175.
  #     # Flag_Liquidity = ifelse(f5 < -0.2, 1, 0),
  #     
  #     # Flag 2: Solvency Crisis
  #     # Analysis: Cluster 2 Defaulters had f6 = -0.488.
  #     # Flag_Solvency = ifelse(f6 < -0.5, 1, 0),
  #     
  #     # Flag 3: Profitability Crisis
  #     # Analysis: Cluster 3 Defaulters had f8 = -1.09.
  #     # Flag_Profit = ifelse(f8 < -0.5, 1, 0),
  #     
  #     # Analysis: Cluster 4 (Zombies) had f4 = -0.923.
  #     # Flag_Inventory = ifelse(f4 < -0.8, 1, 0),
  #     
  #     # Flag 5: The "Zombie" Profile
  #     # Analysis: The False Positive group had Low Equity (<-1.2) but Positive Liabilities (>0.3).
  #     # This captures the "Protected" status.
  #     # Flag_Zombie_Type = ifelse(f6 < -0.5 & f11 > 0.2, 1, 0),
  #     
  #     ##======================================================================##
  #     ## Aggregation
  #     ##======================================================================##
  #     
  #     # Sum of Flags (0 to 5)
  #     # XGBoost can split on this integer: "If Flags > 3, then High Risk"
  #     # Red_Flag_Counter = Flag_Liquidity + Flag_Solvency + Flag_Profit + Flag_Inventory + Flag_Zombie_Type
  #   )
  # 
  # # Check the new features
  # # print("--- New Engineering Summary ---")
  # # print(glimpse(Strategy_B_AS_revised %>% select(starts_with("Ratio"), starts_with("Flag"), Red_Flag_Counter)))

}, error = function(e) message(e))

## Test set.
tryCatch({
  
  # Strategy_D_Test <- Test_Transformed %>%
  #   mutate(
  #     ##======================================================================##
  #     ## Strategy 2: The "Zombie" Interactions (Exact Match to Train)
  #     ##======================================================================##
  #     
  #     # 1. Support Structure Ratio (Commented out in Train, so commented out here)
  #     # Ratio_Support_Structure = f11 / (abs(f6) + 0.0001),
  #     
  #     # 2. The "Distress Gap"
  #     # Logic: Difference between Liabilities (f11) and Equity (f6). 
  #     Gap_Debt_Equity = f11 - f6,
  #     
  #     # 3. Cash Burn Ratio
  #     # Logic: Relates Cash (f5) to Profit (f8). 
  #     # Added +0.0001 to denominator as per training set to handle zeros.
  #     Ratio_Cash_Profit = f5 / (abs(f8) + 0.0001),
  #     
  #     # 4. Interaction Term
  #     Interaction_Cash_Profit = f5 * f8,
  #     
  #     # 5. Feature Stabilizer (The Strategy D Breakthrough)
  #     # Logic: If Profit is negative, look at Cash.
  #     Feature_Stabilizer = ifelse(f8 < 0, f5, f8)
  #     
  #     ##======================================================================##
  #     ## Strategy 3: Flags (Commented out to match Train)
  #     ##======================================================================##
  #     # Flag_Liquidity = ...
  #     # Red_Flag_Counter = ...
    )

}, error = function(e) message(e))

#==============================================================================#
#==== 00 - END ================================================================#
#==============================================================================#

RunExtension <- FALSE

if(RunExtension){
  
#==============================================================================#
#==== 05 - VAE Evaluation =====================================================#
#==============================================================================#

# Train_Transformed
# Strategy_A_LF
# Strategy_B_AS

#==== 05A - Global Validation: AUC of the Anomaly Score =======================#

tryCatch({
  
  y_true <- as.numeric(as.factor(Strategy_B_AS_revised$y)) - 1 
  roc_obj <- roc(y_true, Strategy_B_AS_revised$Gap_Debt_Equity)
  
  auc_score <- pROC::auc(roc_obj)
  
  print(paste("Global AUC of Anomaly Score alone:", round(auc_score, 4)))
  
  plot(roc_obj, 
       main = paste("Does Weirdness = Risk? (AUC:", round(auc_score, 3), ")"),
       col = "#00BFC4", lwd = 3, legacy.axes = TRUE)
  
}, error = function(e) message(e))

#==== 05B - Global Distribution: Density Plot =================================#

tryCatch({
  
  plot_data <- Strategy_B_AS_revised %>%
    select(y, Ratio_Cash_Profit) %>%
    mutate(Status = ifelse(y == "1", "Default", "Non-Default"))
  
  p_dens <- ggplot(plot_data, aes(x = log(Ratio_Cash_Profit), fill = Status)) +
    geom_density(alpha = 0.6) +
    scale_fill_manual(values = c("Non-Default" = "#00BFC4", "Default" = "#F8766D")) +
    labs(title = "Global Separation: Anomaly Score Distribution",
         subtitle = "If the Red peak is to the right of the Blue peak, the VAE works.",
         x = "Log(Anomaly Score)", y = "Density") +
    theme_minimal()
  
  print(p_dens)
  # p_dens_2 <- p_dens
  
}, error = function(e) message(e))

#==== 05C - Boundaries of the density plot ====================================#

tryCatch({
  
  library(dplyr)
  library(ggplot2)
  library(tidyr)
  
  message("--- Starting Automated Density Segmentation ---")
  
  # 1. Filter for Non-Defaults & Log-Transform
  # We focus only on Non-Defaults (y == 0 or "0") as requested.
  df_safe <- Strategy_B_AS %>%
    filter(y == "0" | y == 0) %>%
    mutate(log_score = log(anomaly_score_total))
  
  # 2. Estimate Density
  # adjust=1.5 smoothes the curve to avoid finding tiny, insignificant noise peaks.
  dens <- density(df_safe$log_score, adjust = 1.5) 
  
  # 3. Find the "Valleys" (Local Minima)
  # A valley is a point lower than its neighbors.
  # We find indices where the derivative changes from negative to positive.
  
  y_vals <- dens$y
  x_vals <- dens$x
  
  # Identify local minima indices
  valley_indices <- which(diff(sign(diff(y_vals))) == 2) + 1
  
  # Get the actual Log-Score values for these valleys (the cut-offs)
  cut_offs <- x_vals[valley_indices]
  
  # Filter boundaries to ensure they are within the data range (sanity check)
  cut_offs <- cut_offs[cut_offs > min(df_safe$log_score) & cut_offs < max(df_safe$log_score)]
  
  message(paste("Detected Boundaries (Valleys) at Log-Scores:", paste(round(cut_offs, 2), collapse = ", ")))
  
  # 4. Segment the Data based on these Cut-Offs
  # We create a function to assign clusters dynamically based on N cut-offs
  assign_cluster <- function(score, cuts) {
    if(length(cuts) == 0) return("Cluster 1 (Single Peak)")
    
    # Sort cuts just in case
    cuts <- sort(cuts)
    
    for(i in 1:length(cuts)) {
      if(score <= cuts[i]) return(paste0("Cluster ", i))
    }
    return(paste0("Cluster ", length(cuts) + 1))
  }
  
  df_safe$Cluster <- sapply(df_safe$log_score, assign_cluster, cuts = cut_offs)
  
  # 5. Feature Profiling: What makes them different?
  # We calculate the Mean of every feature per Cluster
  cluster_profile <- df_safe %>%
    group_by(Cluster) %>%
    summarise(
      Count = n(),
      Avg_Anomaly_Score = mean(anomaly_score_total),
      # Summarize all features (f1...fN)
      across(starts_with("f"), mean, .names = "{.col}") 
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  print("--- Cluster Feature Summary (Non-Defaults) ---")
  print(cluster_profile)
  
  cluster_profile_med <- df_safe %>%
    group_by(Cluster) %>%
    summarise(
      Count = n(),
      Avg_Anomaly_Score = median(anomaly_score_total),
      # Summarize all features (f1...fN)
      across(starts_with("f"), median, .names = "{.col}") 
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  print("--- Cluster Feature Summary (Non-Defaults) ---")
  print(cluster_profile_med)
  
  # 6. Visualization: The "Cut" Plot
  # This visually confirms where the code decided to split the groups.
  p_splits <- ggplot(df_safe, aes(x = log_score)) +
    geom_density(fill = "#00BFC4", alpha = 0.6) +
    geom_vline(xintercept = cut_offs, linetype = "dashed", color = "red", size = 1) +
    annotate("text", x = cut_offs, y = 0, label = "Split", vjust = 1.5, color = "red") +
    labs(title = "Automated Peak Detection (Non-Defaults)",
         subtitle = paste("Red dashed lines indicate detected valleys separating the", length(cut_offs)+1, "clusters."),
         x = "Log(Anomaly Score)", y = "Density") +
    theme_minimal()
  
  print(p_splits)
  
  # 7. (Optional) Z-Score Difference
  # To see purely "what is different" compared to the global average
  global_means <- df_safe %>% 
    summarise(across(starts_with("f"), mean, na.rm = TRUE))
  
  # Calculate Difference from Global Average
  diff_profile <- cluster_profile %>%
    # 1. Remove non-feature columns
    select(-Count, -Avg_Anomaly_Score) %>%
    # 2. Compute differences
    mutate(across(starts_with("f"), ~ . - global_means[[cur_column()]])) %>%
    # 3. Add Cluster label back for readability (optional)
    mutate(Cluster = cluster_profile$Cluster) %>%
    select(Cluster, everything()) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  print("--- Difference from Global Average (What makes this cluster unique?) ---")
  print(diff_profile)
  
}, error = function(e) message(e))

#==== 05D - Boundaries of the density plot (all firms, density safe firms) ====#

##======================================##
## Density via the safe-firms.
##======================================##

tryCatch({
  
  df_safe <- Strategy_B_AS %>%
    filter(y == "0" | y == 0) %>%
    mutate(log_score = log(anomaly_score_total))
  df_safe$Cluster <- sapply(df_safe$log_score, assign_cluster, cuts = cut_offs)
  
  df_default <- Strategy_B_AS %>%
    filter(y == "1" | y == 1) %>%
    mutate(log_score = log(anomaly_score_total))
  df_default$Cluster <- sapply(df_default$log_score, assign_cluster, cuts = cut_offs)
  
############
  cluster_profile_safe <- df_safe %>%
    group_by(Cluster) %>%
    summarise(
      Count = n(),
      Avg_Anomaly_Score = median(anomaly_score_total),
      # Summarize all features (f1...fN)
      across(starts_with("f"), median, .names = "{.col}") 
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  cluster_profile_default <- df_default %>%
    group_by(Cluster) %>%
    summarise(
      Count = n(),
      Avg_Anomaly_Score = median(anomaly_score_total),
      # Summarize all features (f1...fN)
      across(starts_with("f"), median, .names = "{.col}") 
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
########
  
  print(cluster_profile_safe)
  print(cluster_profile_default)
  
  # 6. Visualization: The "Cut" Plot
  # This visually confirms where the code decided to split the groups.
  p_splits <- ggplot(df_safe, aes(x = log_score)) +
    geom_density(fill = "#00BFC4", alpha = 0.6) +
    geom_vline(xintercept = cut_offs, linetype = "dashed", color = "red", size = 1) +
    annotate("text", x = cut_offs, y = 0, label = "Split", vjust = 1.5, color = "red") +
    labs(title = "Automated Peak Detection (Non-Defaults)",
         subtitle = paste("Red dashed lines indicate detected valleys separating the", length(cut_offs)+1, "clusters."),
         x = "Log(Anomaly Score)", y = "Density") +
    theme_minimal()
  
  print(p_splits)
  
  # 7. (Optional) Z-Score Difference
  # To see purely "what is different" compared to the global average
  global_means <- df_safe %>% 
    summarise(across(starts_with("f"), mean, na.rm = TRUE))
  
  # Calculate Difference from Global Average
  diff_profile <- cluster_profile %>%
    # 1. Remove non-feature columns
    select(-Count, -Avg_Anomaly_Score) %>%
    # 2. Compute differences
    mutate(across(starts_with("f"), ~ . - global_means[[cur_column()]])) %>%
    # 3. Add Cluster label back for readability (optional)
    mutate(Cluster = cluster_profile$Cluster) %>%
    select(Cluster, everything()) %>%
    mutate(across(where(is.numeric), \(x) round(x, 4)))
  
  print("--- Difference from Global Average (What makes this cluster unique?) ---")
  print(diff_profile)
  
}, error = function(e) message(e))

#==============================================================================#
#==== 06 - VAE Improvements ===================================================#
#==============================================================================#

### First we identify why the VAE-anomaly score fails to seperate the dataset more accurately. 

#==== 06A - Looking into parts of the dataset (distribution plot) =============#

tryCatch({
  
  # 1. Create the Zones based on your visual cut-offs
  segmented_data <- Strategy_B_AS %>%
    mutate(
      Log_Score = log(anomaly_score_total),
      Anomaly_Zone = case_when(
        Log_Score <= 3.6 ~ "1. Low Anomaly (Normal)",
        Log_Score > 3.6 & Log_Score <= 4.4 ~ "2. Mid Anomaly (Transition)",
        Log_Score > 4.4 ~ "3. High Anomaly (Weird)",
        TRUE ~ "Error"
      ),
      y_num = as.numeric(as.character(y)) # Ensure target is 0/1 numeric
    )
  
  # Check the counts per zone to ensure statistical significance
  print("--- Counts per Anomaly Zone ---")
  print(table(segmented_data$Anomaly_Zone, segmented_data$y))
  
  # 2. Calculate Correlation with Default (y) for EACH Zone
  # We want to know: "How strongly does Feature X predict default in Zone 1 vs Zone 3?"
  
  cor_analysis <- segmented_data %>%
    group_by(Anomaly_Zone) %>%
    summarise(across(starts_with("f"), ~ cor(.x, y_num, use = "complete.obs"))) %>%
    pivot_longer(cols = starts_with("f"), names_to = "Feature", values_to = "Correlation")
  
  # 3. Pivot for Comparison (The "Shift" Table)
  cor_matrix <- cor_analysis %>%
    pivot_wider(names_from = Anomaly_Zone, values_from = Correlation) %>%
    mutate(
      # Calculate the "Hidden Dependency": change in importance from Normal to Weird
      Shift = abs(`3. High Anomaly (Weird)`) - abs(`1. Low Anomaly (Normal)`)
    ) %>%
    arrange(desc(Shift))
  
  print("--- Feature Importance Shift (Normal -> Weird) ---")
  print(head(cor_matrix, 10))
  
}, error = function(e) message(e))

#==== 06B - Heatmap of feature correlations ===================================#

tryCatch({
  
  p_heatmap <- ggplot(cor_analysis, aes(x = Anomaly_Zone, y = reorder(Feature, abs(Correlation)), fill = Correlation)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "#00BFC4", mid = "white", high = "#F8766D", midpoint = 0) +
    labs(title = "Regime Switching: How Risk Drivers Change",
         subtitle = "Features that change color across columns reveal 'Hidden Dependencies'.",
         x = "Anomaly Zone (Weirdness)",
         y = "Feature",
         fill = "Corr with Default") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p_heatmap)
  
}, error = function(e) message(e))

#==============================================================================#
#==============================================================================#
#==============================================================================#
  
}