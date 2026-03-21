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

Quantile_Transform <- TRUE

#==== 03A - Incorporate time-series dynamics ==================================#

tryCatch({

## A. Setup:
Data_Eng <- as.data.table(Data)
setorder(Data_Eng, id, refdate)

# Base tracking
Data_Eng[, time_index := seq_len(.N), by = id]
Data_Eng[, is_mature := fifelse(time_index >= 3, 1, 0)]

# Expanding Features
Data_Eng[, `:=`(
  cummean_r14 = cummean(r14),
  max_drop_r6 = r6 - cummax(r6),
  cummean_r9  = cummean(r9),
  cummean_r10 = cummean(r10),
  # Calculate expanding SD safely, yielding NA for time_index == 1
  exp_sd_r13  = sapply(1:.N, function(i) if(i < 2) NA_real_ else sd(r13[1:i], na.rm = TRUE))
), by = id]

# Explicit Cold-Start Handling (Choose XGBoost NA strategy for safety)
Data_Eng[time_index < 2, max_drop_r6 := NA_real_] 
# (exp_sd_r13 is already NA for time_index == 1 due to the if() statement above)
Data_Eng <- copy(Data_Eng)

}, error = function(e) message("3A Error: Time-Series Dynamics.", e))  

#==== 03B - Sector Specific Deviation Features ================================#

tryCatch({
  
### A. Setup:
Data_Eng[, year := year(refdate)]
target_ratios <- c("r11", "r10", "r5", "r14", "r13") 
  
### B. Implementation:
for (col in target_ratios) {
    new_col <- paste0(col, "_sec_Z")
    Data_Eng[, (new_col) := (get(col) - mean(get(col), na.rm = TRUE)) / 
               sd(get(col), na.rm = TRUE), 
             by = .(sector, year)]
    
# Safety Net: If a sector-year combination has all identical values (sd = 0) 
# or only 1 observation (sd = NA), it generates NAs or Infs. 
# We center these edge cases exactly on the sector mean (Z = 0).
    Data_Eng[is.na(get(new_col)) | is.infinite(get(new_col)), (new_col) := 0]
  } 
  
}, error = function(e) message("3B Error: Sector Deviation Dynamics.", e))  

### Remove the balance sheet positions.
cols_to_remove <- paste0("f", 1:18)
cols_present   <- intersect(names(Data_Eng), cols_to_remove)
Data_Eng <- Data_Eng %>% select(-all_of(cols_present))

#==== 03C - Data Sampling Out-of-Sample =======================================#

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

#==== 03D - Quantile Transformation ===========================================#

tryCatch({
  
  message("--- Starting Quantile Transformation ---")
  
  if(exists("Quantile_Transform") && Quantile_Transform){
    
    # 1. Identify Columns to Transform
    # Exclude metadata, targets, and binary flags
    exclude_cols <- c("y", "id", "refdate", "sector", "size", 
                      "groupmember", "public",
                      "time_index", "is_mature", 
                      "cummean_r14", "max_drop_r6", "cummean_r9",  "cummean_r10", "exp_sd_r13",
                      "year", "r11_sec_Z",  "r10_sec_Z", "r5_sec_Z", "r14_sec_Z", "r13_sec_Z")
    
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

#==== 03E - Cleanup and factors ===============================================#

tryCatch({
  
  message("--- Starting Final Cleanup & Imputation ---")
  
  # Ensure we are working with data.tables
  if(!is.data.table(Train_Transformed)) setDT(Train_Transformed)
  if(!is.data.table(Test_Transformed)) setDT(Test_Transformed)
  
  # 1. Impute "Cold Start" Time-Series NAs 
  # (Crucial to prevent sparse.model.matrix from dropping first-year firms)
  if("max_drop_r6" %in% names(Train_Transformed)) {
    Train_Transformed[is.na(max_drop_r6), max_drop_r6 := 0]
    Test_Transformed[is.na(max_drop_r6), max_drop_r6 := 0]
  }
  if("exp_sd_r13" %in% names(Train_Transformed)) {
    Train_Transformed[is.na(exp_sd_r13), exp_sd_r13 := 0]
    Test_Transformed[is.na(exp_sd_r13), exp_sd_r13 := 0]
  }
  
  # 2. Define Metadata to Drop from Modeling Set
  cols_to_drop <- c("id", "company_id", "row_id", "time_index", "refdate", "year", "history_length")
  
  # 3. Align Factors
  # Identify categorical predictors automatically
  cat_cols <- names(Train_Transformed)[sapply(Train_Transformed, function(x) is.character(x) || is.factor(x))]
  cat_cols <- setdiff(cat_cols, c("y", cols_to_drop))
  
  for (col in cat_cols) {
    # Convert Train to factor
    Train_Transformed[, (col) := as.factor(get(col))]
    train_levels <- levels(Train_Transformed[[col]])
    
    # Enforce Train levels on Test strictly
    Test_Transformed[, (col) := factor(get(col), levels = train_levels)]
  }
  
  # 4. Handle Target 'y'
  # Since you are using XGBoost, 'y' MUST remain a numeric 0/1 vector. 
  # Factors will crash the binary:logistic objective in xgb.DMatrix.
  if("y" %in% names(Train_Transformed)) {
    Train_Transformed[, y := as.numeric(as.character(y))]
    Test_Transformed[, y := as.numeric(as.character(y))]
  }
  
  # 5. Create Final Sets (Proper data.table column subsetting)
  cols_to_keep <- setdiff(names(Train_Transformed), cols_to_drop)
  
  Train_Final <- copy(Train_Transformed[, ..cols_to_keep])
  Test_Final  <- copy(Test_Transformed[, ..cols_to_keep])
  
  message("Processing Complete. Ready for Modelling.")
  
  # Print sanity checks
  cat(sprintf("Train_Final Rows: %d | NAs: %d\n", nrow(Train_Final), sum(is.na(Train_Final))))
  cat(sprintf("Test_Final Rows: %d | NAs: %d\n", nrow(Test_Final), sum(is.na(Test_Final))))
  
  glimpse(Train_Final)
  
}, error = function(e) message("Error in Cleanup: ", e))

#==============================================================================#
#==== 04 - VAE Setup & Data Preparation =======================================#
#==============================================================================#

#==== 04A - Data preparation ==================================================#

tryCatch({
  message("--- Starting VAE Data Prep (with Label Encoding) ---")
  
  if(!is.data.table(Train_Final)) setDT(Train_Final)
  
  # 1. Safely identify metadata
  meta_cols <- c("y", "id", "refdate", "year", "time_index", "company_id", "row_id")
  actual_meta <- intersect(meta_cols, names(Train_Final))
  
  cols_to_keep <- setdiff(names(Train_Final), actual_meta)
  features_only <- Train_Final[, ..cols_to_keep]
  
  # 2. Define column types
  expected_bin_cols <- c("groupmember", "public", "is_new_company", "Profit_Trend_Consistent", "is_mature")
  bin_cols <- intersect(expected_bin_cols, names(features_only))
  
  cat_cols <- names(features_only)[sapply(features_only, is.factor)]
  num_cols <- names(features_only)[sapply(features_only, is.numeric)]
  cont_cols <- setdiff(num_cols, bin_cols)
  
  # 3. Extract Continuous and Binary natively
  data_cont <- features_only[, ..cont_cols]
  data_bin  <- features_only[, ..bin_cols]
  
  # 4. LABEL ENCODE Categorical Features for Keras
  data_cat <- features_only[, ..cat_cols]
  for (col in cat_cols) {
    # as.numeric() on a factor returns 1, 2, 3... 
    # We subtract 1 to make it 0-indexed, which Keras strongly prefers
    data_cat[, (col) := as.numeric(get(col)) - 1]
  }
  
  # 5. Combine into a standard data.frame (all numeric now)
  vae_input_data <- as.data.frame(cbind(data_cont, data_bin, data_cat))
  
  if(any(is.na(vae_input_data))) stop("Input data contains NAs! The VAE will crash.")
  
  # 6. Extract and set distribution metadata for autotab
  if(exists("extracting_distribution")) {
    # Because they are integers now, autotab correctly flags them as categorical
    feat_dist <- extracting_distribution(vae_input_data)
    set_feat_dist(feat_dist)
  }
  
  message("VAE Prep Complete. Ready for VAE_train().")
  
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

#==============================================================================#
#==== 05 - VAE Modeling =======================================================#
#==============================================================================#

Use_VAE_Only <- TRUE

tryCatch({

#==== 05A - Strategy A: Latent features (Dimensional reduction) ===============#

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
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_A_Train <- cbind(Train_Final[, .(y)], Strategy_A_LF)
    } else {
      Strategy_A_Train <- cbind(Train_Final, Strategy_A_LF)
    }
    
  }, error = function(e) message("Strategy A Train Error: ", e))
  
## Test set.
tryCatch({
    message("--- Starting Strategy A Test Preparation ---")
    
    # 1. Select exact columns used in training
    target_cols <- colnames(vae_input_data)
    
    # Keep as data.table temporarily so we can apply the encoding loop
    test_vae_input_dt <- as.data.table(Test_Final[, ..target_cols])
    
    # 2. THE FIX: Apply Integer Encoding to Test Set Categoricals
    # (Relies on 'cat_cols' from the VAE Prep block)
    if(exists("cat_cols") && length(cat_cols) > 0) {
      for (col in cat_cols) {
        test_vae_input_dt[, (col) := as.numeric(get(col)) - 1]
      }
    }
    
    # 3. Convert to matrix format required by Keras predict
    # Because all columns are now numeric, as.matrix() will create a numeric matrix
    x_test_matrix <- as.matrix(as.data.frame(test_vae_input_dt))
    
    # 4. Predict Latent Features
    latent_output_raw <- predict(enc_model, x_test_matrix)
    latent_values <- if(is.list(latent_output_raw)) latent_output_raw[[1]] else latent_output_raw
    
    Strategy_A_LF_Test <- as.data.frame(latent_values)
    colnames(Strategy_A_LF_Test) <- paste0("l", 1:8)
    
    # 5. Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_A_Test <- cbind(Test_Final[, .(y)], Strategy_A_LF_Test)
    } else {
      Strategy_A_Test <- cbind(Test_Final, Strategy_A_LF_Test)
    }
    
    message("Success: Strategy_A_Test created.")
    
  }, error = function(e) message("Strategy A Test Error: ", e))
  
#==== 05B - Strategy B: Anomaly Score =========================================#

## Training set.
tryCatch({
    message("--- Calculating Training Anomaly Scores ---")
    
    reconstructed_list <- predict(vae_fit$trained_model, as.matrix(vae_input_data))
    reconstructed_data <- as.matrix(if(is.list(reconstructed_list)) reconstructed_list[[1]] else reconstructed_list)
    
    # Safely extract continuous dimensions for MSE
    n_cont_cols <- length(cont_cols)
    
    if (n_cont_cols > 0) {
      input_cont <- as.matrix(vae_input_data[, 1:n_cont_cols, drop = FALSE])
      recon_cont <- as.matrix(reconstructed_data[, 1:n_cont_cols, drop = FALSE])
      
      mse_raw <- (input_cont - recon_cont)^2
      mse_raw[is.infinite(mse_raw)] <- 1e9 
      mse_normalized <- rowSums(mse_raw, na.rm = TRUE) / n_cont_cols
    } else {
      mse_normalized <- numeric(nrow(vae_input_data))
    }
    
    # Note: Skipping BCE for categorical here because autotab expands factors 
    # into multiple output nodes (softmax), causing dimension mismatch with raw input.
    # MSE on continuous features is the most stable anomaly score.
    final_score <- mse_normalized 
    
    Strategy_B_AS <- data.table(anomaly_score = final_score)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_B_Train <- cbind(Train_Final[, .(y)], Strategy_B_AS)
    } else {
      Strategy_B_Train <- cbind(Train_Final, Strategy_B_AS)
    }
    
  }, error = function(e) message("Error in Strategy B Train: ", e))
  
## Test set.
tryCatch({
    message("--- Calculating Test Anomaly Scores ---")
    
    # We already built x_test_matrix in Strategy A
    reconstructed_list <- predict(vae_fit$trained_model, x_test_matrix)
    test_recon_matrix <- as.matrix(if(is.list(reconstructed_list)) reconstructed_list[[1]] else reconstructed_list)
    
    test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix > 0] <- 1e9
    test_recon_matrix[is.infinite(test_recon_matrix) & test_recon_matrix < 0] <- -1e9
    test_recon_matrix[is.na(test_recon_matrix)] <- 0 
    
    if (n_cont_cols > 0) {
      input_cont_test <- as.matrix(x_test_matrix[, 1:n_cont_cols, drop = FALSE])
      recon_cont_test <- as.matrix(test_recon_matrix[, 1:n_cont_cols, drop = FALSE])
      
      input_cont_test[is.infinite(input_cont_test)] <- 1e9 
      sq_error <- (input_cont_test - recon_cont_test)^2
      sq_error[is.infinite(sq_error)] <- 1e9 
      
      mse_norm_test <- rowSums(sq_error, na.rm = TRUE) / n_cont_cols
    } else {
      mse_norm_test <- numeric(nrow(x_test_matrix)) 
    }
    
    Strategy_B_AS_Test <- data.table(anomaly_score = mse_norm_test)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_B_Test <- cbind(Test_Final[, .(y)], Strategy_B_AS_Test)
    } else {
      Strategy_B_Test <- cbind(Test_Final, Strategy_B_AS_Test)
    }
    
  }, error = function(e) message("Error in Test Strategy B: ", e))
  
#==== 05C - Strategy C: Feature Denoising =====================================#

## Training set.
tryCatch({
    message("--- Strategy C: Training DAE ---")
    
    # 1. Keras Dense requires a strictly numeric matrix. We MUST one-hot encode here.
    # Create a temporary dataframe without 'y'
    dae_train_df <- as.data.frame(Train_Final[, .SD, .SDcols = !c("y")])
    x_train_clean <- model.matrix(~ . - 1, data = dae_train_df)
    input_dim <- ncol(x_train_clean)
    
    # 2. Build DAE
    input_layer <- layer_input(shape = c(input_dim))
    
    encoded <- input_layer %>%
      layer_gaussian_noise(stddev = 0.1) %>% 
      layer_dense(units = 128, activation = "relu") %>% 
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 8, activation = "relu", name = "bottleneck") 
    
    decoded <- encoded %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 128, activation = "relu") %>%
      layer_dense(units = input_dim, activation = "linear") 
    
    dae_autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
    dae_autoencoder %>% compile(optimizer = "adam", loss = "mse")
    
    # 3. Train
    history <- dae_autoencoder %>% fit(
      x = x_train_clean, y = x_train_clean, 
      epochs = 50, batch_size = 256, shuffle = TRUE, validation_split = 0.1, verbose = 0
    )
    
    # 4. Extract Latent Features
    encoder_only <- keras_model(inputs = dae_autoencoder$input, 
                                outputs = get_layer(dae_autoencoder, "bottleneck")$output)
    
    latent_train <- predict(encoder_only, x_train_clean)
    Strategy_C_LF <- as.data.frame(latent_train)
    colnames(Strategy_C_LF) <- paste0("dae_l", 1:8)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_C_Train <- cbind(Train_Final[, .(y)], Strategy_C_LF)
    } else {
      Strategy_C_Train <- cbind(Train_Final, Strategy_C_LF)
    }
    
  }, error = function(e) message("Strategy C Training Error: ", e))
  
## Test set.
tryCatch({
    message("--- Preparing Test Set for Strategy C ---")
    
    # 1. Prepare numeric matrix via model.matrix
    dae_test_df <- as.data.frame(Test_Final[, .SD, .SDcols = !c("y")])
    test_dae_input_raw <- model.matrix(~ . - 1, data = dae_test_df)
    
    # 2. MATRIX ALIGNMENT (Crucial for DAE Keras predict)
    target_cols_dae <- colnames(x_train_clean)
    
    missing_cols <- setdiff(target_cols_dae, colnames(test_dae_input_raw))
    if(length(missing_cols) > 0) {
      missing_mat <- matrix(0, nrow = nrow(test_dae_input_raw), ncol = length(missing_cols))
      colnames(missing_mat) <- missing_cols
      test_dae_input_raw <- cbind(test_dae_input_raw, missing_mat)
    }
    
    extra_cols <- setdiff(colnames(test_dae_input_raw), target_cols_dae)
    if(length(extra_cols) > 0) {
      test_dae_input_raw <- test_dae_input_raw[, !colnames(test_dae_input_raw) %in% extra_cols, drop = FALSE]
    }
    
    x_test_dae_matrix <- test_dae_input_raw[, target_cols_dae]
    
    # 3. Predict Latent Features
    test_latent_dae <- predict(encoder_only, x_test_dae_matrix)
    Strategy_C_LF_Test <- as.data.frame(test_latent_dae)
    colnames(Strategy_C_LF_Test) <- paste0("dae_l", 1:8)
    
    # Apply Use_VAE_Only Logic
    if(Use_VAE_Only) {
      Strategy_C_Test <- cbind(Test_Final[, .(y)], Strategy_C_LF_Test)
    } else {
      Strategy_C_Test <- cbind(Test_Final, Strategy_C_LF_Test)
    }
    
  }, error = function(e) message("DAE Test Set Error: ", e))
  
#==============================================================================#

}, error = function(e) message(e))

#==============================================================================#
#==== 00 - END ================================================================#
#==============================================================================#