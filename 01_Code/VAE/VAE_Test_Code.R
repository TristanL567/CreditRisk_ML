#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

## To-Do: 
##     Outputs for each model:
##     Train-AUC for all hyperparameter-tuning methods.
##     Final model: Test-AUC (and 1-SE rule). Hyperparameters used. Time for the code.
##     No. of iterations. Charts.

##     Model selection:
##     Comparison of each model by Test-AUC. Time-complexity of each model.
##     

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
              "reticulate", "tensorflow"
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
Data_Path <- "C:/Users/TristanLeiter/Documents/Privat/ILAB/Data/WS2025" ## Needs to be set manually.
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

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

##===========================================##
## Check for "cured"-defaults.
##===========================================##

# audit_data <- CuredDefaultsCheck(Data, mode = "flag")
# zombies <- audit_data %>% 
#   filter(Is_Post_Default == TRUE & y == 0)
# 
# print(head(zombies))

## Remove the "cured" defaults.
# Data_CD <- CuredDefaultsCheck(
#   data = Data, 
#   id_col = "id", 
#   date_col = "refdate", 
#   target_col = "y", 
#   mode = "remove"
# )

## check the no. of obs.

# nrow(Data)
# nrow(Data_CD)

# Data <- Data_CD

##===========================================##
## Check for "cured"-defaults.
##===========================================##

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

#==== 02C - Data Sampling =====================================================#

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
Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
Train_with_id <- Train
Test_with_id <- Test

Train <- Train[, -which(names(Train) %in% Exclude)]
Train_Backup <- Train
Test <- Test[, -which(names(Test) %in% Exclude)]
Test_Backup <- Test

#==============================================================================#
#==== 03 - Feature Engineering ================================================#
#==============================================================================#

DivideByTotalAssets <- FALSE

#==== 03A - Standardization ===================================================#

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

#==== 03B - Quantile Transformation ===========================================#

num_cols <- c("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11")

Train_Transformed <- Train
Test_Transformed <- Test

for (col in num_cols) {
  res <- QuantileTransformation(Train[[col]], Test[[col]])
  Train_Transformed[[col]] <- res$train
  Test_Transformed[[col]] <- res$test
}

summary(Train_Transformed$f1)

#==== 03C - Stratified Folds (for CV) =========================================#

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



## Add the firm id and sector back.
# Strat_Data <- Train_Transformed %>%
#   mutate(
#     id = Train_with_id$id,
#     sector = Train_with_id$sector 
#   )
# 
# Data_Train_CV_stratified_sampling <- MVstratifiedsampling_CV(Strat_Data, num_folds = N_folds)
# Data_Train_CV_List <- Data_Train_CV_stratified_sampling[["fold_list"]]
# Data_Train_CV_Vector <- Data_Train_CV_stratified_sampling[["fold_vector"]]
# 
# print(paste("Folds generated:", length(Data_Train_CV_List)))
# length(Data_Train_CV_Vector)

#==============================================================================#
#==== PART 1: BASELINE VAE (Training & Anomaly Scoring) =======================#
#==============================================================================#

library(autotab)
library(dplyr)
library(caret)
library(keras)
library(ggplot2)
library(tidyr)
library(pROC)

#--- 1.1 Data Preparation -----------------------------------------------------#

train_features <- Train_Transformed %>% select(-y)

data_cont <- train_features %>% select(starts_with("f"))
data_bin  <- train_features %>% select(groupmember, public)

dummies_model   <- dummyVars(~ size + sector, data = train_features)
data_cat_onehot <- predict(dummies_model, newdata = train_features) %>% as.data.frame()

vae_input_data <- cbind(data_cont, data_bin, data_cat_onehot)

feat_dist <- extracting_distribution(train_features) %>% 
  feat_reorder(vae_input_data)

set_feat_dist(feat_dist)

#--- 1.2 Architecture Definition ----------------------------------------------#

encoder_config <- list(
  list("dense", 64, "relu"),
  list("dense", 32, "relu")
)

decoder_config <- list(
  list("dense", 32, "relu"),
  list("dense", 64, "relu")
)

#--- 1.3 Train Baseline VAE ---------------------------------------------------#

vae_fit <- VAE_train(
  data = vae_input_data,
  encoder_info = encoder_config,
  decoder_info = decoder_config,
  latent_dim = 8,
  epoch = 50,
  batchsize = 256,
  beta = 0.5,
  lr = 0.001,
  temperature = 0.5,
  wait = 10,
  kl_warm = TRUE,
  beta_epoch = 10,
  Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0
)

#--- 1.4 Strategy A: Extract Latent Features ----------------------------------#

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
latent_features <- as.data.frame(latent_output[[1]])
colnames(latent_features) <- paste0("l", 1:8)

#--- 1.5 Strategy B: Anomaly Detection ----------------------------------------#

dec_weights <- Decoder_weights(
  encoder_layers = 2, 
  trained_model = vae_fit$trained_model,
  lip_enc = 0, pi_enc = 0, prior_learn = "fixed", BNenc_layers = 0, learn_BN = 0
)

dec_model <- decoder_model(
  decoder_input = NULL,
  decoder_info = decoder_config,
  latent_dim = 8,
  feat_dist = feat_dist,
  lip_dec = 0, pi_dec = 0
) %>% keras::set_weights(dec_weights)

reconstructed_data <- predict(dec_model, as.matrix(latent_features))

n_cont <- ncol(data_cont)
input_matrix <- as.matrix(data_cont)
recon_matrix <- as.matrix(reconstructed_data[, 1:n_cont])

anomaly_scores <- rowMeans((input_matrix - recon_matrix)^2)

Train_Result <- Train_Transformed %>%
  mutate(anomaly_score = anomaly_scores)

#==============================================================================#
#==== PART 2: RESEARCH DIAGNOSTICS (Forensic Analysis) ========================#
#==============================================================================#

#--- 2.1 Define the "Confusion Zone" ------------------------------------------#
# We focus on the high-risk zone where the VAE generates False Positives.
# Firms with High Anomaly Scores (Top 10%) -> VAE thinks they look "Dead".

distress_threshold <- quantile(Train_Result$anomaly_score, 0.90)
high_risk_indices  <- which(Train_Result$anomaly_score > distress_threshold)

# Extract Inputs and Reconstructions for this group
inputs_risk <- input_matrix[high_risk_indices, ]
recon_risk  <- recon_matrix[high_risk_indices, ]

# Create metadata
meta_risk <- Train_Result[high_risk_indices, ] %>%
  mutate(Status = ifelse(y == "1", "Casualty (Default)", "Survivor (Healthy)"))

print(paste("Analyzing", nrow(meta_risk), "high-anomaly firms."))
print(table(meta_risk$Status))


#--- 2.2 The Optimism Gap (Why the score is high) -----------------------------#
# HYPOTHESIS: The VAE predicts these distressed firms *should* be healthier
# (Negative Bias), causing high MSE.

# Calculate Bias (Actual - Predicted)
bias_df <- as.data.frame(inputs_risk - recon_risk) %>%
  bind_cols(Status = meta_risk$Status) %>%
  pivot_longer(cols = -Status, names_to = "Feature", values_to = "Bias")

# Summarize
bias_summary <- bias_df %>%
  group_by(Feature, Status) %>%
  summarise(Mean_Bias = mean(Bias), .groups = 'drop')

# PLOT: The Optimism Gap
p_bias <- ggplot(bias_summary, aes(x = reorder(Feature, abs(Mean_Bias)), y = Mean_Bias, fill = Status)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = c("Survivor (Healthy)" = "#00BFC4", "Casualty (Default)" = "#F8766D")) +
  labs(title = "The Optimism Gap: VAE Bias Analysis",
       subtitle = "Negative Bars = VAE predicted values were HIGHER than reality.",
       y = "Mean Bias", x = "Feature") +
  theme_minimal()

print(p_bias)


#--- 2.3 Separability Analysis (Signal vs. Noise) -----------------------------#
# HYPOTHESIS: The VAE penalizes features that don't actually predict default.
# We calculate Fisher's Separability Score to identify specific "Noise" features.

separability_analysis <- meta_risk %>%
  bind_cols(as.data.frame(inputs_risk)) %>%
  select(Status, starts_with("f")) %>%
  pivot_longer(cols = starts_with("f"), names_to = "Feature", values_to = "Value") %>%
  group_by(Feature) %>%
  summarise(
    Mean_S = mean(Value[Status == "Survivor (Healthy)"]),
    Mean_C = mean(Value[Status == "Casualty (Default)"]),
    SD_S   = sd(Value[Status == "Survivor (Healthy)"]),
    SD_C   = sd(Value[Status == "Casualty (Default)"]),
    # Fisher Score: Higher = Better Signal
    Separability_Score = abs(Mean_S - Mean_C) / (SD_S + SD_C)
  ) %>%
  arrange(desc(Separability_Score))

print("--- DIAGNOSIS: SIGNAL VS NOISE ---")
print(separability_analysis)


#--- 2.4 The Solvency Deep Dive ("Healthy Losers") ----------------------------#
# HYPOTHESIS: Survivors are losing money (Negative Profit) but have Low Debt.
# We isolate "Loss Makers" within the High Risk group to find the tie-breaker.

loss_makers <- meta_risk %>%
  filter(f8 < 0) # Only those with Negative Profit

# Compare Liabilities (f11) and Invested Capital (f2)
p_debt <- ggplot(loss_makers, aes(x = Status, y = f11, fill = Status)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#F8766D", "#00BFC4")) +
  labs(title = "Why 'Healthy Losers' Survive", subtitle = "Survivors have significantly lower Liabilities (f11)", y = "Liabilities (f11)") +
  theme_minimal() + theme(legend.position="none")

p_solvency <- ggplot(loss_makers, aes(x = Status, y = (f5 - f11), fill = Status)) +
  geom_boxplot() +
  scale_fill_manual(values = c("#F8766D", "#00BFC4")) +
  labs(title = "The Solvency Shield", subtitle = "Metric: Cash (f5) - Liabilities (f11)", y = "Solvency Score") +
  theme_minimal()

grid.arrange(p_debt, p_solvency, ncol = 2)

#==============================================================================#
#==== PART 3: ENHANCED VAE (Quantitative Solvency) ============================#
#==============================================================================#

#--- 3.1 Feature Engineering (Continuous, No Cutoffs) -------------------------#

Train_Enhanced <- Train_Transformed %>%
  mutate(
    # 1. Solvency Shield: "Can I pay my debts?"
    # High Cash (f5) - High Liabilities (f11)
    # Higher = Better Coverage.
    solvency_shield = f5 - f11,
    
    # 2. Quality Spread: "Is my wealth real or just paper?"
    # High Cash (f5) - High Current Assets (f3)
    # Higher = Leaner, more liquid asset structure.
    quality_spread = f5 - f3
  )

#--- 3.2 Update AutoTab Data Structures ---------------------------------------#

train_features_v2 <- Train_Enhanced %>% select(-y)

# UPDATE: Add new CONTINUOUS variables
# Note: We removed 'zombie_shield' from the binary list.
data_cont_v2 <- train_features_v2 %>% 
  select(starts_with("f"), solvency_shield, quality_spread)

# Binary variables (Metadata only)
data_bin_v2  <- train_features_v2 %>% 
  select(groupmember, public)

# One-Hot Encoding (Standard)
dummies_model   <- dummyVars(~ size + sector, data = train_features_v2)
data_cat_onehot <- predict(dummies_model, newdata = train_features_v2) %>% as.data.frame()

# Bind Inputs
vae_input_v2 <- cbind(data_cont_v2, data_bin_v2, data_cat_onehot)

# Create New Distribution Map
feat_dist_v2 <- extracting_distribution(train_features_v2) %>% 
  feat_reorder(vae_input_v2)

set_feat_dist(feat_dist_v2)

#--- 3.3 Train Enhanced VAE ---------------------------------------------------#

encoder_config <- list(list("dense", 64, "relu"), list("dense", 32, "relu"))
decoder_config <- list(list("dense", 32, "relu"), list("dense", 64, "relu"))

print("Training VAE with Quantitative Solvency Features...")
vae_fit_v2 <- VAE_train(
  data = vae_input_v2,
  encoder_info = encoder_config,
  decoder_info = decoder_config,
  latent_dim = 8,
  epoch = 50,
  batchsize = 256,
  beta = 0.5,
  lr = 0.001,
  wait = 10,
  temperature = 0.5,
  kl_warm = TRUE,
  beta_epoch = 10,
  Lip_en = 0, pi_enc = 0, lip_dec = 0, pi_dec = 0
)

#--- 3.4 Extract & Score ------------------------------------------------------#

# 1. Encoder (Latent Space)
enc_weights_v2 <- Encoder_weights(
  encoder_layers = 2, trained_model = vae_fit_v2$trained_model,
  lip_enc = 0, pi_enc = 0, BNenc_layers = 0, learn_BN = 0
)
enc_model_v2 <- encoder_latent(
  encoder_input = vae_input_v2, encoder_info = encoder_config, latent_dim = 8,
  Lip_en = 0, power_iterations = 0
) %>% keras::set_weights(enc_weights_v2)

latent_v2 <- predict(enc_model_v2, as.matrix(vae_input_v2))[[1]]

# 2. Decoder (Reconstruction)
dec_weights_v2 <- Decoder_weights(
  encoder_layers = 2, trained_model = vae_fit_v2$trained_model,
  lip_enc = 0, pi_enc = 0, prior_learn = "fixed", BNenc_layers = 0, learn_BN = 0
)
dec_model_v2 <- decoder_model(
  decoder_input = NULL, decoder_info = decoder_config, latent_dim = 8,
  feat_dist = feat_dist_v2, lip_dec = 0, pi_dec = 0
) %>% keras::set_weights(dec_weights_v2)

recon_v2 <- predict(dec_model_v2, latent_v2)

# 3. Calculate Anomaly Score (MSE on Continuous Vars Only)
# Note: Columns 1 to 13 correspond to f1-f11 + solvency_shield + quality_spread
n_cont_v2 <- ncol(data_cont_v2)
input_mat_v2 <- as.matrix(data_cont_v2)
recon_mat_v2 <- as.matrix(recon_v2[, 1:n_cont_v2])

# Standard MSE calculation
scores_v2 <- rowMeans((input_mat_v2 - recon_mat_v2)^2)

Result_V2 <- Train_Enhanced %>% mutate(anomaly_score_v2 = scores_v2)

#==============================================================================#
#==== PART 4: COMPARISON ------------------------------------------------------#
#==============================================================================#

# Visual Check: Boxplot
ggplot(Result_V2, aes(x = y, y = anomaly_score_v2, fill = y)) +
  geom_boxplot() +
  labs(title = "Round 2: Enhanced VAE (Quantitative)",
       subtitle = "Using Continuous Solvency Shield (f5-f11) instead of Cutoffs",
       y = "Anomaly Score")

# Quantitative Check: AUC
roc_v1 <- roc(Train_Result$y, Train_Result$anomaly_score)
roc_v2 <- roc(Result_V2$y, Result_V2$anomaly_score_v2)

print(paste("Round 1 AUC (Baseline):", round(pROC::auc(roc_v1), 4)))
print(paste("Round 2 AUC (Quantitative Enhanced):", round(pROC::auc(roc_v2), 4)))

#==============================================================================#
#==== PART 5: WEIGHTED SCORING & FINAL ASSEMBLY ===============================#
#==============================================================================#

#--- 5.1 Define Weights -------------------------------------------------------#
# High weights for 'Signal' (Cash, Solvency), Low weights for 'Noise' (Assets, Provisions)
weights_v2 <- c(
  f1 = 0.02,  f2 = 0.10,  f3 = 0.15,  f4 = 0.05,
  f5 = 1.00,  f6 = 0.80,  f7 = 0.40,  f8 = 1.00,
  f9 = 0.80,  f10 = 0.01, f11 = 0.10,
  solvency_shield = 1.50, # Critical Signal
  quality_spread = 1.00   # Strong Signal
)

#--- 5.2 Calculate Weighted Score (Risk Index) --------------------------------#

# 1. Get Squared Errors from the Enhanced VAE (Part 3)
# (Assumes input_mat_v2 and recon_mat_v2 are from the VAE trained in Part 3)
sq_errors_v2 <- (input_mat_v2 - recon_mat_v2)^2

# 2. Calculate Weighted Sum
# We initialize a zero vector and add weighted columns one by one
weighted_error_sum <- numeric(nrow(sq_errors_v2))
total_weight_used  <- 0

for(col in names(weights_v2)) {
  # Only process if the column exists in our VAE input
  if(col %in% colnames(sq_errors_v2)) {
    # Add weighted error
    weighted_error_sum <- weighted_error_sum + (sq_errors_v2[, col] * weights_v2[col])
    # Track total weight for normalization
    total_weight_used <- total_weight_used + weights_v2[col]
  }
}

# 3. Normalize to get the Final Score
risk_index_score <- weighted_error_sum / total_weight_used

#--- 5.3 Assemble The Comparison Dataset --------------------------------------#

# We combine all our work into one master dataframe.
Train_Comparison <- Train_Enhanced %>%
  mutate(
    # Score 1: Baseline (From Part 1 - ensure Train_Result is loaded or re-joined)
    score_baseline = Train_Result$anomaly_score,
    
    # Score 2: Enhanced Unweighted (From Part 3)
    score_enhanced = Result_V2$anomaly_score_v2,
    
    # Score 3: Enhanced Weighted (calculated above)
    score_weighted = risk_index_score
  ) %>%
  # Add Latent Features (Optional, good for XGBoost later)
  bind_cols(as.data.frame(latent_v2) %>% setNames(paste0("l", 1:8))) %>%
  select(
    y,
    score_baseline, 
    score_enhanced, 
    score_weighted,
    starts_with("f"),         # Original Financials
    solvency_shield,          # Engineered Features
    quality_spread,
    starts_with("l"),         # Latent Features
    sector, size, groupmember
  )

glimpse(Train_Comparison)

#--- 5.4 Quick Validation: The AUC Ladder -------------------------------------#

# Calculate ROCs for all three
roc_1 <- roc(Train_Comparison$y, Train_Comparison$score_baseline)
roc_2 <- roc(Train_Comparison$y, Train_Comparison$score_enhanced)
roc_3 <- roc(Train_Comparison$y, Train_Comparison$score_weighted)

print("--- AUC PERFORMANCE COMPARISON ---")
print(paste("1. Baseline VAE:        ", round(pROC::auc(roc_1), 4)))
print(paste("2. Enhanced VAE (Raw):  ", round(pROC::auc(roc_2), 4)))
print(paste("3. Weighted Risk Index: ", round(pROC::auc(roc_3), 4)))

# Visual Check of the Champion Score
ggplot(Train_Comparison, aes(x = y, y = score_weighted, fill = y)) +
  geom_boxplot() +
  labs(title = "Final Weighted Risk Index",
       subtitle = "Suppressed Noise (Assets/Provisions) + Amplified Signal (Solvency)",
       y = "Weighted Anomaly Score")

#==============================================================================#
#==== PART 6: Data Preparation =-----------------------------------------------#
#==============================================================================#

# Experiment 1: Baseline (Standard Financials + Raw VAE Score)
# Goal: Test if the raw VAE adds value over accounting data.
Train_Exp1 <- Train_Comparison %>%
  select(
    y,                      # Response Variable
    starts_with("f"),       # Original Financials (f1..f11)
    score_baseline,         # Raw Anomaly Score (V1)
    size, sector            # Metadata
  )

# Experiment 2: Enhanced (Solvency Features + Enhanced VAE Score)
# Goal: Test if adding "Solvency Shield" helps.
Train_Exp2 <- Train_Comparison %>%
  select(
    y,                      # Response Variable
    starts_with("f"),
    solvency_shield,        # Engineered Feature (Signal)
    quality_spread,         # Engineered Feature (Signal)
    score_enhanced,         # Enhanced VAE Score (V2)
    size, sector
  )

# Experiment 3: Champion (Full Hybrid + Weighted Score + Latent Features)
# Goal: The "Super Model" with noise suppression and cluster features.
Train_Exp3 <- Train_Comparison %>%
  select(
    y,                      # Response Variable
    starts_with("f"),
    starts_with("l"),       # Latent Features (l1..l8) - CRITICAL for this step
    solvency_shield,
    quality_spread,
    score_weighted,         # The Champion Risk Index
    size, sector
  )

#==============================================================================#
#==== TEST SET PIPELINE (Inference Only) ======================================#
#==============================================================================#

#--- Step 1: Feature Engineering (Apply same formulas) ------------------------#

Test_Enhanced <- Test_Transformed %>% # Assuming this exists and is scaled like Train
  mutate(
    # Create the exact same features
    solvency_shield = f5 - f11,
    quality_spread  = f5 - f3
  )

#--- Step 2: Prepare VAE Inputs (Use Training Encoders) -----------------------#

test_features <- Test_Enhanced %>% select(-y)

# 1. Continuous Vars (Match Train_V2 structure)
data_cont_test <- test_features %>% 
  select(starts_with("f"), solvency_shield, quality_spread)

# 2. Binary Vars
data_bin_test <- test_features %>% 
  select(groupmember, public)

# 3. Categorical One-Hot (CRITICAL: Use existing 'dummies_model' from Train)
# This ensures Test has the exact same columns as Train, even if some sectors are missing.
data_cat_test <- predict(dummies_model, newdata = test_features) %>% as.data.frame()

# 4. Bind into Matrix
vae_input_test <- cbind(data_cont_test, data_bin_test, data_cat_test) %>%
  as.matrix() # VAE expects a matrix

#--- Step 3: Run VAE Prediction (Pass through Trained Models) -----------------#

# 1. Generate Latent Features (l1..l8) for Test
# Use 'enc_model_v2' which is already trained
latent_test <- predict(enc_model_v2, vae_input_test)[[1]]
colnames(latent_test) <- paste0("l", 1:8)

# 2. Generate Reconstructions
# Use 'dec_model_v2' which is already trained
recon_test <- predict(dec_model_v2, latent_test)

#--- Step 4: Calculate Scores (Apply Training Weights) ------------------------#

# Extract Continuous parts for scoring
input_mat_test <- vae_input_test[, 1:ncol(data_cont_test)]
recon_mat_test <- recon_test[, 1:ncol(data_cont_test)]

# A. Enhanced Score (Unweighted MSE)
sq_error_test <- (input_mat_test - recon_mat_test)^2
score_enhanced_test <- rowMeans(sq_error_test)

# B. Weighted Score (Champion Index)
# Apply the exact same 'weights_v2' vector from Part 5
weighted_sum_test <- numeric(nrow(sq_error_test))
total_weight <- 0

for(col in names(weights_v2)) {
  if(col %in% colnames(sq_error_test)) {
    weighted_sum_test <- weighted_sum_test + (sq_error_test[, col] * weights_v2[col])
    total_weight <- total_weight + weights_v2[col]
  }
}

score_weighted_test <- weighted_sum_test / total_weight

#--- Step 5: Final Test Assembly ----------------------------------------------#

Test_Comparison <- Test_Enhanced %>%
  mutate(
    # Add the generated scores
    score_enhanced = score_enhanced_test,
    score_weighted = score_weighted_test
  ) %>%
  # Add Latent Features
  bind_cols(as.data.frame(latent_test)) %>%
  # Ensure 'y' is kept for validation
  select(y, everything())

#--- Step 6: Create the 3 Test Sets for Benchmarking --------------------------#

Test_Exp2 <- Test_Comparison %>%
  select(y, starts_with("f"), solvency_shield, quality_spread, score_enhanced, size, sector)

Test_Exp3 <- Test_Comparison %>%
  select(y, starts_with("f"), starts_with("l"), solvency_shield, quality_spread, score_weighted, size, sector)

print("Test Set Processing Complete.")
glimpse(Test_Exp3)

#==============================================================================#
#==============================================================================#
#==============================================================================#