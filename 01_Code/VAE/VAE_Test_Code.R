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

library(autoTab)
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
#==== PART 2: RESEARCH DIAGNOSTICS (Identifying "Healthy Zombies") ============#
#==============================================================================#

#--- 2.1 Isolate Healthy Outliers ---------------------------------------------#

cutoff <- quantile(Train_Result$anomaly_score, 0.95)

healthy_outliers <- Train_Result %>%
  filter(y == "0", anomaly_score > cutoff)

healthy_normal <- Train_Result %>%
  filter(y == "0", anomaly_score <= cutoff)

comparison <- bind_rows(
  healthy_outliers %>% summarise(across(starts_with("f"), mean)) %>% mutate(Group = "Outliers"),
  healthy_normal   %>% summarise(across(starts_with("f"), mean)) %>% mutate(Group = "Normal")
)

print(comparison)

#--- 2.2 Error Profiling (Forensic Analysis) ----------------------------------#

outlier_indices <- which(Train_Result$y == "0" & Train_Result$anomaly_score > cutoff)

input_subset <- input_matrix[outlier_indices, ]
recon_subset <- recon_matrix[outlier_indices, ]

error_per_column <- colMeans((input_subset - recon_subset)^2)

error_df <- data.frame(
  Feature = names(error_per_column),
  Error = as.numeric(error_per_column)
) %>%
  arrange(desc(Error))

print(error_df)

#--- 2.3 Hypothesis Testing: Survivors vs Casualties --------------------------#

distress_threshold <- quantile(Train_Result$anomaly_score, 0.90)

distressed_firms <- Train_Result %>%
  filter(anomaly_score > distress_threshold) %>%
  mutate(Status = ifelse(y == "1", "Casualty (Defaulted)", "Survivor (Healthy)"))

# Feature Difference Analysis
feature_diff <- distressed_firms %>%
  select(Status, starts_with("f"), groupmember, public) %>%
  group_by(Status) %>%
  summarise(across(everything(), median)) %>% 
  pivot_longer(cols = -Status, names_to = "Feature", values_to = "Value") %>%
  pivot_wider(names_from = Status, values_from = Value) %>%
  mutate(
    Difference = `Survivor (Healthy)` - `Casualty (Defaulted)`,
    Abs_Diff = abs(Difference)
  ) %>%
  arrange(desc(Abs_Diff))

print(feature_diff)


#==============================================================================#
#==== PART 3: ENHANCED VAE (With Zombie Shield) ===============================#
#==============================================================================#

#--- 3.1 Feature Engineering --------------------------------------------------#

equity_cutoff <- -1.0  
cash_cutoff   <- -0.9  

Train_Enhanced <- Train_Transformed %>%
  mutate(
    zombie_shield = ifelse(f6 < equity_cutoff & f5 > cash_cutoff, 1, 0),
    quality_spread = f5 - f3
  )

#--- 3.2 Update AutoTab Data Structures ---------------------------------------#

train_features_v2 <- Train_Enhanced %>% select(-y)

data_cont_v2 <- train_features_v2 %>% select(starts_with("f"), quality_spread)
data_bin_v2  <- train_features_v2 %>% select(groupmember, public, zombie_shield)

dummies_model   <- dummyVars(~ size + sector, data = train_features_v2)
data_cat_onehot <- predict(dummies_model, newdata = train_features_v2) %>% as.data.frame()

vae_input_v2 <- cbind(data_cont_v2, data_bin_v2, data_cat_onehot)

feat_dist_v2 <- extracting_distribution(train_features_v2) %>% 
  feat_reorder(vae_input_v2)

set_feat_dist(feat_dist_v2)

#--- 3.3 Train Enhanced VAE ---------------------------------------------------#

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

#--- 3.4 Extract New Anomaly Scores -------------------------------------------#

# 1. Latent Space V2
enc_weights_v2 <- Encoder_weights(
  encoder_layers = 2, trained_model = vae_fit_v2$trained_model,
  lip_enc = 0, pi_enc = 0, BNenc_layers = 0, learn_BN = 0
)
enc_model_v2 <- encoder_latent(
  encoder_input = vae_input_v2, encoder_info = encoder_config, latent_dim = 8,
  Lip_en = 0, power_iterations = 0
) %>% keras::set_weights(enc_weights_v2)

latent_v2 <- predict(enc_model_v2, as.matrix(vae_input_v2))[[1]]

# 2. Reconstruction V2
dec_weights_v2 <- Decoder_weights(
  encoder_layers = 2, trained_model = vae_fit_v2$trained_model,
  lip_enc = 0, pi_enc = 0, prior_learn = "fixed", BNenc_layers = 0, learn_BN = 0
)
dec_model_v2 <- decoder_model(
  decoder_input = NULL, decoder_info = decoder_config, latent_dim = 8,
  feat_dist = feat_dist_v2, lip_dec = 0, pi_dec = 0
) %>% keras::set_weights(dec_weights_v2)

recon_v2 <- predict(dec_model_v2, latent_v2)

# 3. Calculate Scores V2
n_cont_v2 <- ncol(data_cont_v2)
input_mat_v2 <- as.matrix(data_cont_v2)
recon_mat_v2 <- as.matrix(recon_v2[, 1:n_cont_v2])

scores_v2 <- rowMeans((input_mat_v2 - recon_mat_v2)^2)

Result_V2 <- Train_Enhanced %>% mutate(anomaly_score_v2 = scores_v2)


#==============================================================================#
#==== PART 4: COMPARISON & VALIDATION =========================================#
#==============================================================================#

#--- 4.1 Visual Comparison ----------------------------------------------------#

ggplot(Result_V2, aes(x = y, y = anomaly_score_v2, fill = y)) +
  geom_boxplot() +
  labs(title = "Round 2: Did the Zombie Flag improve separation?",
       subtitle = "Anomaly Scores after teaching VAE about 'Zombie Shield'",
       y = "New Anomaly Score")

#--- 4.2 AUC Comparison -------------------------------------------------------#

roc_v1 <- roc(Train_Result$y, Train_Result$anomaly_score)
roc_v2 <- roc(Result_V2$y, Result_V2$anomaly_score_v2)

print(paste("Round 1 AUC (Original):", round(pROC::auc(roc_v1), 4)))
print(paste("Round 2 AUC (With Zombie Shield):", round(pROC::auc(roc_v2), 4)))

#==============================================================================#
#==============================================================================#
#==============================================================================#