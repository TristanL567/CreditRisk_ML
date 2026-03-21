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

## Incorporate 4 more ratios.
# Data <- Data %>%
#   mutate(
#     ratio_True_Equity = f6 / (f1 + 1e-6),
#     ratio_Receivables = (f3 - f4 - f5) / (f1 + 1e-6),
#     ratio_Cash_Coverage = f5 / (f11 + 1e-6),
#     ratio_Provisions = f10 / (f1 + 1e-6)
#   )

## Drop all raw data for now.
Exclude <- c(paste("f", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .))) %>%
  drop_na()

#==== 02C - Data Sampling =====================================================#

tryCatch({
  
set.seed(123)
Data_Sampled <- MVstratifiedsampling(Data,
                                     strat_vars = c("sector", "y"),
                                     Train_size = 0.7)

Train <- Data_Sampled[["Train"]]
Test <- Data_Sampled[["Test"]]

## Quick validation.
# Analyze the Training Set
analyse_MVstratifiedsampling(Train, "TRAIN SET", 
                             target_col = "y", 
                             sector_col = "sector", 
                             date_col = "refdate")

# Analyze the Test Set
analyse_MVstratifiedsampling(Test, "TEST SET", 
                             target_col = "y", 
                             sector_col = "sector", 
                             date_col = "refdate")

## Exclude id and refdate.
# Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.$
Exclude <- c("id") ## Drop the id and ref_date (year) for now.

Train_with_id <- Train
Test_with_id <- Test

Train <- Train[, -which(names(Train) %in% Exclude)]
Train_Backup <- Train
Test <- Test[, -which(names(Test) %in% Exclude)]
Test_Backup <- Test

}, error = function(e) message(e))

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

DivideByTotalAssets <- FALSE
Quantile_Transform <- TRUE

#==== 03A - Standardization ===================================================#

tryCatch({
  
if(DivideByTotalAssets){
  
  asset_col <- "f1" 
  cols_to_scale <- c("f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")
  
  safe_divide <- function(numerator, denominator) {
    if_else(denominator == 0 | is.na(denominator), 
            0,                       
            numerator / denominator
    )
  }
  
  Train <- Train %>%
    mutate(
      across(
        .cols = all_of(cols_to_scale),
        .fns = ~ safe_divide(.x, .data[[asset_col]])
      )
    ) %>%
    as.data.frame()
  
  Test <- Test %>%
    mutate(
      across(
        .cols = all_of(cols_to_scale),
        .fns = ~ safe_divide(.x, .data[[asset_col]])
      )
    ) %>%
    as.data.frame()
  
}

}, error = function(e) message(e))

#==== 03B - Quantile Transformation ===========================================#

tryCatch({
  
  if(Quantile_Transform){
    
    # 1. Identify all Numeric Columns to Transform
    # We exclude:
    # - ID/Metadata: "id", "company_id", "row_id", "time_index", "refdate", "sector", "size"
    # - The Target: "y"
    # - Binary Flags: "is_new_company", "Profit_Trend_Consistent" (Transforming 0/1 makes no sense)
    
    exclude_cols <- c("y", "id", "company_id", "row_id", "time_index", 
                      "refdate", "sector", "size", "groupmember", "public",
                      "is_new_company", "Profit_Trend_Consistent", "new_company_flag")
    
    numeric_cols <- names(Train)[sapply(Train, is.numeric)]
    cols_to_transform <- setdiff(numeric_cols, exclude_cols)
    
    message(paste("Transforming", length(cols_to_transform), "features..."))
    
    Train_Transformed <- Train
    Test_Transformed  <- Test 
    
    for (col in cols_to_transform) {
      if(col %in% names(Train) && col %in% names(Test)) {
        res <- QuantileTransformation(Train[[col]], Test[[col]])
        Train_Transformed[[col]] <- res$train
        Test_Transformed[[col]]  <- res$test
      }
    }
    
    message("Transformation Complete. Summary of r14 (ROA) after transform:")
    print(summary(Train_Transformed$r14))
    
  } else {
    Train_Transformed <- Train
    Test_Transformed  <- Test
  }
  
}, error = function(e) message(e))
#==== 03C - Ensure the categorical variables are factors ======================#

tryCatch({
  
## Ensure categorical variables are set up as factors.
cat_cols <- names(Train_Transformed)[sapply(Train_Transformed, is.character)]
cat_cols <- setdiff(cat_cols, "y")

print(paste("Categorical columns found:", paste(cat_cols, collapse = ", ")))

for (col in cat_cols) {
  
  Train_Transformed[[col]] <- as.factor(Train_Transformed[[col]])
  
  train_levels <- levels(Train_Transformed[[col]])
  Test_Transformed[[col]] <- factor(Test_Transformed[[col]], 
                                    levels = train_levels)
  
  na_count <- sum(is.na(Test_Transformed[[col]]))
  if(na_count > 0){
    warning(paste("Variable", col, "in Test set has", na_count, 
                  "new levels not seen in Train. They are now NAs."))
  }
}

Train_Transformed$y <- as.factor(Train_Transformed$y)
Test_Transformed$y  <- factor(Test_Transformed$y, levels = levels(Train_Transformed$y))

str(Train_Transformed[, cat_cols, drop=FALSE])

}, error = function(e) message(e))

#==============================================================================#
#==== 04 - VAE (Strategy A: latent features; Strategy B: anomaly score) =======#
#==============================================================================#

#==== 04A - Data preparation ==================================================#

tryCatch({
  
train_features <- Train_Transformed %>% select(-y)
data_cont <- train_features %>% select(starts_with("f"))
data_bin  <- train_features %>% select(groupmember, public)

dummies_model   <- dummyVars(~ size + sector, data = train_features)
data_cat_onehot <- predict(dummies_model, newdata = train_features) %>% as.data.frame()

vae_input_data <- cbind(data_cont, data_bin, data_cat_onehot)
feat_dist <- extracting_distribution(vae_input_data)
set_feat_dist(feat_dist)

### Configuration:
encoder_config <- list(
  list("dense", 64, "relu"),
  list("dense", 32, "relu")
)

decoder_config <- list(
  list("dense", 32, "relu"),
  list("dense", 64, "relu")
)

}, error = function(e) message(e))

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
  
  test_features <- Test_Transformed %>% select(-y)
  test_cont     <- test_features %>% select(starts_with("f"))
  test_bin      <- test_features %>% select(groupmember, public)
  
  test_cat_onehot <- predict(dummies_model, newdata = test_features) %>% as.data.frame()
  test_vae_input_data <- cbind(test_cont, test_bin, test_cat_onehot)
  
  
  missing_cols <- setdiff(names(vae_input_data), names(test_vae_input_data))
  
  if(length(missing_cols) > 0) {
    for(col in missing_cols) {
      test_vae_input_data[[col]] <- 0
    }
  }
  
  extra_cols <- setdiff(names(test_vae_input_data), names(vae_input_data))
  if(length(extra_cols) > 0) {
    test_vae_input_data <- test_vae_input_data %>% select(-all_of(extra_cols))
  }
  
  test_vae_input_data <- test_vae_input_data %>% select(all_of(names(vae_input_data)))
  test_matrix <- as.matrix(test_vae_input_data)
  
  # Use the standalone encoder we built earlier
  test_latent_raw <- predict(enc_model, test_matrix)
  
  ### Strategy A:
  Strategy_A_LF_Test <- as.data.frame(test_latent_raw[[1]])
  colnames(Strategy_A_LF_Test) <- paste0("l", 1:8)
  
}, error = function(e) message(e))

#==== 04C - Strategy B: Anomaly Score =========================================#

## Training set.
tryCatch({
  
  # 1. Get Reconstruction
  reconstructed_list <- predict(vae_fit$trained_model, as.matrix(vae_input_data))
  
  if(is.list(reconstructed_list)) {
    reconstructed_data <- as.matrix(reconstructed_list[[1]])
  } else {
    reconstructed_data <- as.matrix(reconstructed_list)
  }
  
  # --- SENIOR FIX 1: Fix Dimensions & Indexing ---
  # Check if data_cont exists; if not, assume all numeric columns are continuous
  if (exists("data_cont") && !is.null(data_cont)) {
    n_cont_cols <- ncol(data_cont)
  } else {
    # Fallback: Count numeric columns in input
    n_cont_cols <- sum(sapply(vae_input_data, is.numeric))
    message(paste("Note: 'data_cont' object not found. Detected", n_cont_cols, "continuous columns."))
  }
  
  # Handle edge case where NO continuous columns exist to prevent index errors
  if (n_cont_cols == 0) {
    n_cat_cols <- ncol(vae_input_data)
    cont_indices <- integer(0) # Empty vector
  } else {
    n_cat_cols <- ncol(vae_input_data) - n_cont_cols 
    cont_indices <- 1:n_cont_cols
  }
  
  # 2. Extract Matrices Safely
  if (length(cont_indices) > 0) {
    input_cont <- as.matrix(vae_input_data[, cont_indices, drop = FALSE])
    recon_cont <- as.matrix(reconstructed_data[, cont_indices, drop = FALSE])
  } else {
    input_cont <- matrix(0, nrow = nrow(vae_input_data), ncol = 0)
    recon_cont <- matrix(0, nrow = nrow(vae_input_data), ncol = 0)
  }
  
  if (n_cat_cols > 0) {
    cat_indices <- (n_cont_cols + 1):ncol(vae_input_data)
    input_cat   <- as.matrix(vae_input_data[, cat_indices, drop = FALSE])
    recon_cat   <- as.matrix(reconstructed_data[, cat_indices, drop = FALSE])
  } else {
    input_cat <- matrix(0, nrow = nrow(vae_input_data), ncol = 0)
    recon_cat <- matrix(0, nrow = nrow(vae_input_data), ncol = 0)
  }
  
  # 3. Calculate Scores (MSE & BCE)
  
  # Continuous Score
  if (n_cont_cols > 0) {
    mse_raw <- (input_cont - recon_cont)^2
    # Clip extreme overflows just in case
    mse_raw[is.infinite(mse_raw)] <- 1e9 
    mse_scores <- rowSums(mse_raw, na.rm = TRUE)
  } else {
    mse_scores <- numeric(nrow(vae_input_data))
  }
  
  # Categorical Score
  if (n_cat_cols > 0) {
    epsilon   <- 1e-15
    recon_cat <- pmax(pmin(recon_cat, 1 - epsilon), epsilon)
    bce_scores <- -rowSums(input_cat * log(recon_cat) + (1 - input_cat) * log(1 - recon_cat), na.rm = TRUE)
  } else {
    bce_scores <- numeric(nrow(vae_input_data))
  }
  
  # --- SENIOR FIX 2: Safe Normalization (Prevent Div/0) ---
  
  if (n_cont_cols > 0) {
    mse_normalized <- mse_scores / n_cont_cols
  } else {
    mse_normalized <- 0 
  }
  
  if (n_cat_cols > 0) {
    bce_normalized <- bce_scores / n_cat_cols
  } else {
    bce_normalized <- 0
  }
  
  # 5. Weighted Combination
  final_score <- mse_normalized + bce_normalized
  
  # 6. Apply to Dataframe
  Strategy_B_AS <- Train_Transformed %>%
    mutate(
      anomaly_score_cont_avg = mse_normalized,
      anomaly_score_cat_avg  = bce_normalized,
      anomaly_score_balanced = final_score 
    )
  
  print("Strategy B calculated successfully. Summary:")
  print(summary(Strategy_B_AS$anomaly_score_balanced))
  
}, error = function(e) message("Error in Strategy B: ", e))

## Test set.
tryCatch({
  
  ### Strategy B: Test Set Anomaly Detection
  
  # 1. Get Reconstruction & Handle Output Format
  reconstructed_list <- predict(vae_fit$trained_model, as.matrix(test_matrix))
  
  if(is.list(reconstructed_list)) {
    test_recon_matrix <- as.matrix(reconstructed_list[[1]])
  } else {
    test_recon_matrix <- as.matrix(reconstructed_list)
  }
  
  # SAFETY CHECK: Clip Infinite Model Outputs
  test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix > 0] <- 1e9
  test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix < 0] <- -1e9
  test_recon_matrix[is.na(test_recon_matrix)] <- 0 
  
  # 2. Define Dimensions Safely
  # Detect n_cont based on actual data if possible, or fall back to column count
  if (exists("test_cont") && !is.null(test_cont)) {
    n_cont <- ncol(test_cont)
  } else {
    # Fallback: assume first N columns are numeric if test_cont isn't explicitly defined
    # You might need to hardcode this if you know the exact number, e.g., n_cont <- 12
    n_cont <- sum(sapply(test_matrix, is.numeric))
    message(paste("Note: 'test_cont' not found. Using detected count:", n_cont))
  }
  
  # Define Categorical count
  n_cat <- ncol(test_matrix) - n_cont
  
  # 3. Calculate Scores (with conditional logic for 0 columns)
  
  # --- Continuous Scores (MSE) ---
  if (n_cont > 0) {
    # Slice matrices
    input_cont_test <- as.matrix(test_matrix[, 1:n_cont, drop = FALSE])
    recon_cont_test <- as.matrix(test_recon_matrix[, 1:n_cont, drop = FALSE])
    
    # Sanitize Input (Handle potential Infs in input data)
    input_cont_test[is.infinite(input_cont_test)] <- 1e9 
    
    # Calculate Squared Error
    sq_error <- (input_cont_test - recon_cont_test)^2
    sq_error[is.infinite(sq_error)] <- 1e9 # Clip overflows
    
    # Normalize immediately (Sum / Count)
    mse_norm_test <- rowSums(sq_error, na.rm = TRUE) / n_cont
    
  } else {
    mse_norm_test <- numeric(nrow(test_matrix)) # Zeros if no cont columns
  }
  
  # --- Categorical Scores (BCE) ---
  if (n_cat > 0) {
    # Slice matrices
    cat_indices      <- (n_cont + 1):ncol(test_matrix)
    input_cat_test   <- as.matrix(test_matrix[, cat_indices, drop = FALSE])
    recon_cat_test   <- as.matrix(test_recon_matrix[, cat_indices, drop = FALSE])
    
    # Epsilon Clipping for log stability
    epsilon <- 1e-15
    recon_cat_test <- pmax(pmin(recon_cat_test, 1 - epsilon), epsilon)
    
    # Calculate BCE
    bce_raw <- -(input_cat_test * log(recon_cat_test) + (1 - input_cat_test) * log(1 - recon_cat_test))
    
    # Normalize immediately
    bce_norm_test <- rowSums(bce_raw, na.rm = TRUE) / n_cat
    
  } else {
    bce_norm_test <- numeric(nrow(test_matrix)) # Zeros if no cat columns
  }
  
  # 4. Weighted Combination
  w_cont <- 1.0 
  w_cat  <- 1.0 
  
  final_score_test <- (mse_norm_test * w_cont) + (bce_norm_test * w_cat)
  
  # 5. Append to Test Data
  Strategy_B_AS_Test <- Test_Transformed %>%
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
  
  x_train_clean <- as.matrix(vae_input_data)
  input_dim <- ncol(x_train_clean)
  
  # 2. Define the DAE Architecture
  # We use the Functional API to insert the Noise Layer
  input_layer <- layer_input(shape = c(input_dim))
  
  encoded <- input_layer %>%
    # ADD NOISE HERE: 0.1 is standard deviation (tune between 0.05 - 0.2)
    layer_gaussian_noise(stddev = 0.1) %>% 
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu", name = "bottleneck") # Latent Space
  
  decoded <- encoded %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    # Output layer must match input dimensions
    layer_dense(units = input_dim, activation = "linear") 
  
  # 3. Compile
  dae_autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
  
  dae_autoencoder %>% compile(
    optimizer = "adam",
    loss = "mse" # MSE is standard for DAE reconstruction
  )
  
  # 4. Train (The Critical Step)
  # Notice: x = x_train_clean, y = x_train_clean
  # The layer_gaussian_noise handles the corruption internally during training only.
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
  # Create a separate model that stops at the bottleneck
  encoder_only <- keras_model(inputs = dae_autoencoder$input, 
                              outputs = get_layer(dae_autoencoder, "bottleneck")$output)
  
  # Predict Latent Features
  latent_train <- predict(encoder_only, x_train_clean)
  colnames(latent_train) <- paste0("dae_l", 1:8)
  
  # Combine
  Strategy_C <- cbind(Train_Transformed, as.data.frame(latent_train))
  
  message("DAE Training Complete. Latent features extracted.")
  
}, error = function(e) message("Strategy C: Denoising Training.", e))

## Test set.
tryCatch({
  
  message("--- Preparing Test Set for Strategy C ---")
  
  test_features <- Test_Transformed %>% select(-y)
  test_cont     <- test_features %>% select(starts_with("f"))
  test_bin      <- test_features %>% select(groupmember, public)
  
  test_cat_onehot <- predict(dummies_model, newdata = test_features) %>% as.data.frame()
  test_vae_input_data <- cbind(test_cont, test_bin, test_cat_onehot)
  
  x_test_matrix <- as.matrix(test_vae_input_data)
  
  test_latent_dae <- predict(encoder_only, x_test_matrix)
  
  Strategy_C_LF_Test <- as.data.frame(test_latent_dae)
  colnames(Strategy_C_LF_Test) <- paste0("dae_l", 1:8)
  
  Strategy_C_Test <- cbind(Test_Transformed, Strategy_C_LF_Test)
  
  print("Success: Strategy C Test Set Prepared.")

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