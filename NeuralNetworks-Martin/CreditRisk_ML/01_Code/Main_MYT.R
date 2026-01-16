#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

## Core packages needed for GLM
core_packages <- c("dplyr", "caret", "lubridate", "purrr", "tidyr",
                   "Matrix", "pROC",            ## Sparse Matrices and efficient AUC computation.
                   "glmnet")                    ## GLM library.

## Additional packages for tree-based models (optional for GLM only)
optional_packages <- c("xgboost",                   ## XGBoost library.
                       "rBayesianOptimization",     ## Bayesian Optimization.
                       "ggplot2", "Ckmeans.1d.dp",  ## Plotting & Charts | XG-Charts / Feature Importance.
                       "scales",                    ## ggplot2 extension for nice charts.
                       "future", "furrr")           ## Parallel processing.

## Install and load core packages
for(i in 1:length(core_packages)){
  package_name <- core_packages[i]
  if (!requireNamespace(package_name, quietly = TRUE)) {
    install.packages(package_name, character.only = TRUE)
    cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
  } else {
    cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
  }
  library(package_name, character.only = TRUE)
}

## Try to install optional packages (suppress errors for incompatible versions)
for(i in 1:length(optional_packages)){
  package_name <- optional_packages[i]
  tryCatch({
    if (!requireNamespace(package_name, quietly = TRUE)) {
      install.packages(package_name, character.only = TRUE)
      cat(paste("Package '", package_name, "' was not installed. It has now been installed and loaded.\n", sep = ""))
    } else {
      cat(paste("Package '", package_name, "' is already installed and has been loaded.\n", sep = ""))
    }
    library(package_name, character.only = TRUE)
  }, error = function(e) {
    cat(paste("Warning: Could not install/load '", package_name, "' (may require newer R version)\n", sep = ""))
  })
}

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
Data_Path <- file.path(dirname(Path), "data")  ## One level up from CreditRisk_ML
Data_Directory <- file.path(Data_Path, "data.rda")
Charts_Directory <- file.path(Path, "03_Charts")

## Charts Directories.
Charts_GLM_Directory <- file.path(Charts_Directory, "GLM")
Charts_XGBoost_Directory <- file.path(Charts_Directory, "XGBoost")

Functions_Directory <- file.path(Path, "01_Code/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Data Sampling.
set.seed(123)

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

Data <- DataPreprocessing(DataPreprocessing)

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

#==== 02C - Data Sampling =====================================================#

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

DivideByTotalAssets <- TRUE

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

## Now work with:
## Train_Transformed
## Test_Transformed

#==== 03C - Stratified Folds (for CV) =========================================#

Strat_Data <- Train_Transformed %>%
  mutate(id = Train_with_id$id)

cv_folds_list <- MVstratifiedsampling_CV(Strat_Data, k = N_folds)
print(paste("Folds generated:", length(cv_folds_list)))

#==============================================================================#
#==== 04 - GLMs ===============================================================#
#==============================================================================#

#==== 04A - GLMs ==============================================================#

## Setup parallel processing if available
if(requireNamespace("future", quietly = TRUE) && requireNamespace("furrr", quietly = TRUE)) {
  library(future)
  library(furrr)
  plan(multisession, workers = parallel::detectCores() - 1)  ## Leave 1 core free
  cat(paste("Parallel processing enabled with", future::nbrOfWorkers(), "workers\n"))
  use_parallel <- TRUE
} else {
  cat("Parallel processing packages not available. Running sequentially.\n")
  use_parallel <- FALSE
}

tryCatch({
  
##==============================##
## General Parameters.
##==============================##
  
##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
  
# Training Data
train_y <- as.numeric(as.character(Train_Transformed$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Transformed)

# Test Data
test_y <- as.numeric(as.character(Test_Transformed$y))
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Transformed)

## Setup the stratified cv data.
foldid_vector <- rep(NA, length(train_y))
for (k in seq_along(cv_folds_list)) {
  foldid_vector[cv_folds_list[[k]]] <- k
}

##==============================##
## Discrete grid search (hyperparameter tuning of alpha and lambda).
##==============================##

## Prepare the grid.
grid_ds_glm <- tibble(
  alpha = seq(0, 1, by = 0.1) # 0 = Ridge, 1 = Lasso
) %>%
  mutate(
    current_iter = 1:n(),
    total_iters = n()
  )
## Implement the vectorized grid search function.
if(use_parallel) {
  results_ds_glm <- future_pmap_dfr(grid_ds_glm, GLM_gridsearch, .options = furrr_options(seed = 123)) %>%
    arrange(desc(CV_AUC))
} else {
  results_ds_glm <- pmap_dfr(grid_ds_glm, GLM_gridsearch) %>%
    arrange(desc(CV_AUC))
}

print("--- Top 5 Discrete GLM Models ---")
print(head(results_ds_glm, 5))

##==============================##
## Random grid search (hyperparameter tuning of alpha and lambda).
##==============================##

n_iter_rs <- 10  ## Reduced from 20 for faster execution

## Prepare the grid.
grid_rs_glm <- tibble(
  alpha = runif(n_iter_rs, min = 0, max = 1)
) %>%
  mutate(
    current_iter = 1:n(),
    total_iters = n()
  )

print(paste("Total random combinations:", nrow(grid_rs_glm)))

## Implement the vectorized grid search function.
if(use_parallel) {
  results_rs_glm <- future_pmap_dfr(grid_rs_glm, GLM_gridsearch, .options = furrr_options(seed = 123)) %>%
    arrange(desc(CV_AUC))
} else {
  results_rs_glm <- pmap_dfr(grid_rs_glm, GLM_gridsearch) %>%
    arrange(desc(CV_AUC))
}

print("--- Top 5 Random GLM Models ---")
print(head(results_rs_glm, 5))

##==============================##
## Bayesian Optimization (hyperparameter tuning of alpha and lambda).
##==============================##

## Prepare the grid.
bounds_glm <- list(
  alpha = c(0, 1)
)

# 3. Run Optimization
n_init_glm <- 2   ## Reduced from 5 for faster execution (random exploration)
n_iter_glm <- 5   ## Reduced from 15 for faster execution (smart search)

cat("\n=== Starting Bayesian Optimization ===")
cat("\nInitial random points:", n_init_glm)
cat("\nSmart iterations:", n_iter_glm)
cat("\nTotal evaluations:", n_init_glm + n_iter_glm, "\n\n")
cat("Progress: Each dot = 1 evaluation\n")

## Implement the vectorized grid search function.
bayes_out_glm <- BayesianOptimization(
  FUN = GLM_bayesoptim,
  bounds = bounds_glm,
  init_points = n_init_glm,
  n_iter = n_iter_glm,
  acq = "ei",   
  eps = 0.0,    
  verbose = TRUE
)

cat("\n[Bayesian Optimization finished, processing results...]\n")

## Compute the train AUC.
results_bayes_glm <- bayes_out_glm$History %>%
  rename(alpha = alpha) %>%
  mutate(
    current_iter = 1:n(),
    total_iters = n()
  ) %>%
  pmap_dfr(function(alpha, current_iter, total_iters, ...) {
    GLM_gridsearch(alpha, current_iter, total_iters, verbose = FALSE)
  }) %>%
  arrange(desc(CV_AUC))

print("--- Top 5 Bayesian GLM Models ---")
print(head(results_bayes_glm, 5))

cat("\n=== BAYESIAN OPTIMIZATION COMPLETE ===")
cat("\nBest alpha found:", results_bayes_glm$alpha[1])
cat("\nBest CV AUC:", round(results_bayes_glm$CV_AUC[1], 4), "\n\n")

##==============================##
## Performance of the model with the highest training AUC in the test set.
##==============================##

## Data summary and consolidation.
best_ds <- results_ds_glm[1, ] %>% mutate(Method = "Grid Search")
best_rs <- results_rs_glm[1, ] %>% mutate(Method = "Random Search")
best_bayes <- results_bayes_glm[1, ] %>% mutate(Method = "Bayesian Opt")

glm_method_performance <- bind_rows(best_ds, best_rs, best_bayes) %>%
  select(Method, alpha, CV_AUC, Train_AUC) %>%
  arrange(desc(CV_AUC))

print("--- GLM Method Comparison ---")
print(glm_method_performance)

# We pick the row with the highest CV_AUC across all methods
champion_row <- glm_method_performance[1, ]
best_alpha <- champion_row$alpha

message(paste("Winning Strategy:", champion_row$Method))
message(paste("Optimal Alpha:", round(best_alpha, 4)))

## Refit the Final Model on Full Training Data
final_cv_glm <- cv.glmnet(
  x = train_matrix, 
  y = train_y,
  family = "binomial",       
  type.measure = "auc",     
  alpha = best_alpha,
  foldid = foldid_vector,   
  standardize = TRUE        
)

## Compute Test AUC: Champion (lambda.min)
probs_champion <- predict(final_cv_glm, 
                          newx = test_matrix, 
                          s = "lambda.min", 
                          type = "response")

roc_champion <- roc(test_y, as.vector(probs_champion), quiet = TRUE)
auc_champion <- auc(roc_champion)

## Compute Test AUC: 1-SE Rule (lambda.1se)
probs_1se <- predict(final_cv_glm, 
                     newx = test_matrix, 
                     s = "lambda.1se", 
                     type = "response")

roc_1se <- roc(test_y, as.vector(probs_1se), quiet = TRUE)
auc_1se <- auc(roc_1se)

message("------------------------------------------------")
message(sprintf("Final GLM Test AUC (Champion/Min):  %.5f", auc_champion))
message(sprintf("Final GLM Test AUC (1-SE Rule):     %.5f", auc_1se))
message("------------------------------------------------")

n_vars_champion <- sum(coef(final_cv_glm, s = "lambda.min") != 0)
n_vars_1se      <- sum(coef(final_cv_glm, s = "lambda.1se") != 0)

message(sprintf("Variables Selected (Champion):      %d", n_vars_champion))
message(sprintf("Variables Selected (1-SE Rule):     %d", n_vars_1se))

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

## Visualisation: Method AUC comparison.
colors <- c(
  "Grid Search"   = blue,
  "Random Search" = orange,  
  "Bayesian Opt"  = red   
)

Plot_Train_AUC <- ggplot(XGBoost_method_performance, aes(x = reorder(Method, Train_AUC), y = Train_AUC, 
                                                         fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Train_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "Hyperparameter Tuning Method",
    y = "AUC-Score (Training Set)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"), # Adjusted to "grey30" for safety
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_GLM_Directory, "01_HyperparameterTuningMethods_AUC_Training.png")
ggsave(
  filename = Path,
  plot = Plot_Train_AUC,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

cat("\n\n========================================")
cat("\n=== GLM ANALYSIS COMPLETE ===")
cat("\n========================================")
cat("\n\nChampion Method:", champion_row$Method)
cat("\nOptimal Alpha:", round(best_alpha, 4))
cat("\nTest AUC (Champion):", round(auc_champion, 4))
cat("\nTest AUC (1-SE):", round(auc_1se, 4))
cat("\nFeatures Selected (Champion):", n_vars_champion)
cat("\nFeatures Selected (1-SE):", n_vars_1se)
cat("\n\nChart saved to:", Path)

##==============================##
## Export Results with Interpretation
##==============================##

Results_Dir <- file.path(Path, "01_Code", "GLM_Results")
dir.create(Results_Dir, showWarnings = FALSE, recursive = TRUE)

## Save all method results
write.csv(results_ds_glm, file.path(Results_Dir, "01_grid_search_results.csv"), row.names = FALSE)
write.csv(results_rs_glm, file.path(Results_Dir, "02_random_search_results.csv"), row.names = FALSE)
write.csv(results_bayes_glm, file.path(Results_Dir, "03_bayesian_optim_results.csv"), row.names = FALSE)
write.csv(glm_method_performance, file.path(Results_Dir, "04_method_comparison.csv"), row.names = FALSE)

## Save model objects
saveRDS(final_cv_glm, file.path(Results_Dir, "final_glm_model.rds"))

## Create interpretation summary
summary_text <- paste0(
  "GLM ELASTIC NET RESULTS SUMMARY\n",
  "================================\n\n",
  "Analysis Date: ", Sys.time(), "\n\n",
  
  "1. HYPERPARAMETER TUNING COMPARISON\n",
  "-----------------------------------\n",
  "Grid Search (11 alphas):     CV AUC = ", round(best_ds$CV_AUC, 4), ", alpha = ", round(best_ds$alpha, 4), "\n",
  "Random Search (10 alphas):   CV AUC = ", round(best_rs$CV_AUC, 4), ", alpha = ", round(best_rs$alpha, 4), "\n",
  "Bayesian Optim (7 alphas):   CV AUC = ", round(best_bayes$CV_AUC, 4), ", alpha = ", round(best_bayes$alpha, 4), "\n\n",
  
  "WINNER: ", champion_row$Method, "\n\n",
  
  "2. CHAMPION MODEL CONFIGURATION\n",
  "-------------------------------\n",
  "Optimal Alpha: ", round(best_alpha, 4), "\n",
  "  - Interpretation: ",
  if(best_alpha < 0.3) "Ridge-heavy (L2 penalty dominant) - keeps all features, shrinks coefficients"
  else if(best_alpha > 0.7) "LASSO-heavy (L1 penalty dominant) - aggressive feature selection"
  else "Balanced Elastic Net - mix of feature selection and coefficient shrinkage",
  "\n\n",
  "Optimal Lambda (min): ", round(final_cv_glm$lambda.min, 6), "\n",
  "Optimal Lambda (1se): ", round(final_cv_glm$lambda.1se, 6), "\n\n",
  
  "3. MODEL PERFORMANCE\n",
  "-------------------\n",
  "Cross-Validation AUC: ", round(champion_row$CV_AUC, 4), " (5-fold stratified)\n",
  "Training AUC:          ", round(champion_row$Train_AUC, 4), "\n",
  "Test AUC (Champion):   ", round(auc_champion, 4), "\n",
  "Test AUC (1-SE Rule):  ", round(auc_1se, 4), "\n\n",
  
  "Performance Gap: ", round((champion_row$Train_AUC - auc_champion) * 100, 2), "% drop from train to test\n",
  "  - Interpretation: ",
  if((champion_row$Train_AUC - auc_champion) < 0.03) "Excellent generalization - minimal overfitting"
  else if((champion_row$Train_AUC - auc_champion) < 0.06) "Good generalization - acceptable overfitting"
  else "Significant overfitting - consider simpler model (1-SE rule)",
  "\n\n",
  
  "4. FEATURE SELECTION\n",
  "-------------------\n",
  "Features in Champion Model (lambda.min): ", n_vars_champion, " out of ", ncol(train_matrix), "\n",
  "Features in Parsimonious Model (lambda.1se): ", n_vars_1se, " out of ", ncol(train_matrix), "\n",
  "Features Eliminated: ", ncol(train_matrix) - n_vars_champion, "\n\n",
  
  "5. INTERPRETATION\n",
  "-----------------\n",
  "AUC Score Meaning:\n",
  "  - 0.50 = Random guessing (coin flip)\n",
  "  - 0.70-0.80 = Acceptable discrimination\n",
  "  - 0.80-0.90 = Excellent discrimination\n",
  "  - >0.90 = Outstanding discrimination\n\n",
  
  "Your Test AUC of ", round(auc_champion, 2), " means: ",
  if(auc_champion < 0.7) "Poor model - needs improvement"
  else if(auc_champion < 0.8) "Acceptable model - decent predictive power"
  else if(auc_champion < 0.9) "Excellent model - strong predictive power"
  else "Outstanding model - very strong predictive power",
  "\n\n",
  
  "6. FILES GENERATED\n",
  "-----------------\n",
  "01_grid_search_results.csv - All grid search alpha values tested\n",
  "02_random_search_results.csv - All random search alpha values tested\n",
  "03_bayesian_optim_results.csv - Bayesian optimization path\n",
  "04_method_comparison.csv - Summary comparison of three methods\n",
  "final_glm_model.rds - Trained model object (can be loaded with readRDS)\n",
  "\n",
  "Chart: ", Path, "\n\n",
  
  "7. NEXT STEPS\n",
  "-------------\n",
  "1. Review feature coefficients: coef(final_cv_glm, s='lambda.min')\n",
  "2. Check which features were eliminated (coefficient = 0)\n",
  "3. Compare against tree-based models (XGBoost, Random Forest)\n",
  "4. Consider calibration analysis if using for probability estimates\n"
)

writeLines(summary_text, file.path(Results_Dir, "00_RESULTS_SUMMARY.txt"))

cat("\n\nResults exported to:", Results_Dir)
cat("\nSee 00_RESULTS_SUMMARY.txt for interpretation")
cat("\n========================================\n\n")



}, error = function(e) message(e))

#==============================================================================#
#==== 05 - Decision Trees =====================================================#
#==============================================================================#

#==== 05A - Random Forest =====================================================#


#==== 05B - AdaBoost ==========================================================#


#==== 05C - XGBoost ===========================================================#

## To-Do: Fix overall parameters at the beginning of the code.
## Comparison of hyperparameter tuning methods.
## Visualisations and Outputs.

## Only run XGBoost if package is available
if(requireNamespace("xgboost", quietly = TRUE)) {
  cat("\n=== Running XGBoost Analysis ===\n")

tryCatch({

##==============================##
## General Parameters.
##==============================##

##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")

# Training Data
train_y <- as.numeric(as.character(Train_Transformed$y))
train_matrix <- sparse.model.matrix(sparse_formula, data = Train_Transformed)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)

# Test Data
test_y <- as.numeric(as.character(Test_Transformed$y))
test_matrix <- sparse.model.matrix(sparse_formula, data = Test_Transformed)
dtest <- xgb.DMatrix(data = test_matrix, label = test_y)

##==============================##
## Discrete Grid Search for Hyperparameter Tuning.
##==============================##

## Prepare the grid.
grid_ds <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 5, 6),
  subsample = c(0.7, 0.9),
  colsample_bytree = c(0.7, 0.9)
)

## Add the tuning and progress params.
grid_ds <- grid_ds %>%
  mutate(
    nrounds = 2000,
    early_stopping_rounds = 50,
    current_iter = 1:n(),
    total_iters = n()
  )

print(paste("Total combinations to test:", nrow(grid_ds)))

## Implement the vectorized grid search function.
tryCatch({
results_ds <- pmap_dfr(grid_ds, XGBoost_gridsearch,
                       folds_custom = cv_folds_list) %>%
    arrange(desc(AUC))
  
print("--- Top 5 Discrete Models ---")
print(head(results_ds, 5))
  
}, error = function(e) print(e))

## Train the final model (optimal combination of hyperparameters).
tryCatch({
best_ds <- results_ds[1, ]
  
final_params_ds <- list(
    booster = "gbtree", 
    objective = "binary:logistic", 
    eval_metric = "auc",
    eta = best_ds$eta, 
    max_depth = best_ds$max_depth, 
    subsample = best_ds$subsample, 
    colsample_bytree = best_ds$colsample_bytree
)

## Find the best iteration.
check_cv <- xgb.cv(
  params = final_params_ds,
  data = dtrain,
  nrounds = 2000,
  folds = cv_folds_list,
  # nfold = 5,
  early_stopping_rounds = 50,
  verbose = 0,
  maximize = TRUE
)

optimal_rounds <- check_cv$evaluation_log[which.max(check_cv$evaluation_log$test_auc_mean)]$iter

## Train the model with the optimal hyperparameters.
model_ds <- xgb.train(params = final_params_ds, 
                      data = dtrain, 
                      nrounds = optimal_rounds, 
                      verbose = 0)

}, error = function(e) print(e))

## Compute the training AUC.
XGBoost_ds_probs <- predict(model_ds, dtrain)
XGBoost_ds_Train_ROC <- roc(train_y, XGBoost_ds_probs, quiet = TRUE)
XGBoost_ds_Train_AUC <- auc(XGBoost_ds_Train_ROC)
print(paste("Final Train AUC:", round(XGBoost_ds_Train_AUC, 5)))

##==============================##
## Random Grid Search for Hyperparameter Tuning.
##==============================##

## Prepare the grid.
n_iter <- 20

grid_rs <- data.frame(
  eta = runif(n_iter, 0.01, 0.3),
  max_depth = sample(3:8, n_iter, replace = TRUE),
  subsample = runif(n_iter, 0.5, 1.0),
  colsample_bytree = runif(n_iter, 0.5, 1.0)
)

## Add the tuning and progress params.
grid_rs <- grid_rs %>%
  mutate(
    nrounds = 2000,
    early_stopping_rounds = 50,
    current_iter = 1:n(),
    total_iters = n()
  )

print(paste("Total random combinations:", nrow(grid_rs)))

## Implement the vectorized grid search function.
tryCatch({
  results_rs <- pmap_dfr(grid_rs, XGBoost_gridsearch,
                         folds_custom = cv_folds_list) %>%
    arrange(desc(AUC))
  
  print("--- Top 5 Random Models ---")
  print(head(results_rs, 5))
  
}, error = function(e) message(e))

## Train the final model (optimal combination of hyperparameters).
tryCatch({
best_rs <- results_rs[1, ]

final_params_rs <- list(
  booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
  eta = best_rs$eta, max_depth = best_rs$max_depth, 
  subsample = best_rs$subsample, colsample_bytree = best_rs$colsample_bytree
)

# Determine optimal rounds for the Random Search winner
check_cv_rs <- xgb.cv(
  params = final_params_rs,
  data = dtrain,
  nrounds = 2000,
  folds = cv_folds_list,
  # nfold = 5,
  early_stopping_rounds = 50,
  verbose = 0,
  maximize = TRUE
)
optimal_rounds_rs <- check_cv_rs$evaluation_log[which.max(check_cv_rs$evaluation_log$test_auc_mean)]$iter

## Train the model with the optimal hyperparameters.
model_rs <- xgb.train(params = final_params_rs, 
                      data = dtrain,
                      nrounds = optimal_rounds_rs, 
                      verbose = 0)

}, error = function(e) message(e))

## Compute the training AUC.
XGBoost_rs_probs <- predict(model_rs, dtrain)
XGBoost_rs_Train_ROC <- roc(train_y, XGBoost_rs_probs, quiet = TRUE)
XGBoost_rs_Train_AUC <- auc(XGBoost_rs_Train_ROC)
print(paste("Final Train AUC:", round(XGBoost_rs_Train_AUC, 5)))

##==============================##
## Bayesian Optimization for Hyperparameter Tuning.
##==============================##

## Prepare the parameters.
n_init_points <- 10
n_iter_bayes  <- 20
total_bayes_runs <- n_init_points + n_iter_bayes
current_bayes_iter <- 0 # Initialize counter

bounds_bayes <- list(
  eta = c(0.01, 0.3),
  max_depth = c(3L, 8L), 
  subsample = c(0.5, 1.0),
  colsample_bytree = c(0.5, 1.0)
)

## Run Optimization & Format Results.
tryCatch({
current_bayes_iter <- 0 
  
bayes_out <- BayesianOptimization(
  FUN = XGBoost_bayesoptim,
  bounds = bounds_bayes,
  init_points = n_init_points,
  n_iter = n_iter_bayes,
  # REFINEMENT 1: Use Expected Improvement (Frazier (2018)).
  acq = "ei", 
  # REFINEMENT 2: Use Matern 5/2 Kernel (Better for realistic landscapes )
  kernel = list(type = "matern", nu = 5/2),
  # REFINEMENT 3: Slight epsilon increase to handle CV noise/prevent over-exploitation
  eps = 0.01, 
  verbose = TRUE
)
  
## Convert History to Tibble.
results_bayes <- bayes_out$History %>%
    rename(AUC = Value) %>%
    mutate(
      max_depth = as.integer(round(max_depth)),
      nrounds = 2000,
      early_stopping_rounds = 50,
      iteration_index = 1:n()
    ) %>%
    arrange(desc(AUC)) %>%
    as_tibble()
  
  print("--- Top 5 Bayesian Models ---")
  print(head(results_bayes, 5))
  
}, error = function(e) message(e))

## Train the final model.
tryCatch({
  
best_bayes <- results_bayes[1, ]
  
final_params_bayes <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
    eta = best_bayes$eta, 
    max_depth = best_bayes$max_depth, 
    subsample = best_bayes$subsample, 
    colsample_bytree = best_bayes$colsample_bytree
  )
  
# Determine optimal rounds for the Bayesian optim.
check_cv_bayes <- xgb.cv(
  params = final_params_bayes,
  data = dtrain,
  nrounds = 2000,
  folds = cv_folds_list,
  # nfold = 5,
  early_stopping_rounds = 50,
  verbose = 0,
  maximize = TRUE
)

optimal_rounds_bayes <- check_cv_bayes$evaluation_log[which.max(check_cv_bayes$evaluation_log$test_auc_mean)]$iter
  
## Train the model with the optimal hyperparameters.
model_bayes <- xgb.train(
                        params = final_params_bayes, 
                        data = dtrain,
                        nrounds = optimal_rounds_bayes, 
                        verbose = 0
  )
  
}, error = function(e) message(e))

## Compute the training AUC.
XGBoost_bayes_probs <- predict(model_bayes, dtrain)
XGBoost_bayes_Train_ROC <- roc(train_y, XGBoost_bayes_probs, quiet = TRUE)
XGBoost_bayes_Train_AUC <- auc(XGBoost_bayes_Train_ROC)
print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))

##==============================##
## Performance of the model with the highest training AUC in the test set.
##==============================##

## Check for the right tree size.
cv_results_train <- xgb.cv(
  params = final_params_bayes,
  data = dtrain,
  folds = cv_folds_list,
  nrounds = 3000,
  early_stopping_rounds = 50,
  verbose = 0,
  maximize = TRUE
)

## Extract the data. We compare the performance of 1-SE with the full fit.
optimal_rounds_final <- cv_results_train$evaluation_log[which.max(cv_results_train$evaluation_log$test_auc_mean)]$iter
eval_log <- cv_results_train$evaluation_log

## Calculate 1-SE Threshold
best_iter_index <- which.max(eval_log$test_auc_mean)
best_auc_mean   <- eval_log$test_auc_mean[best_iter_index]
best_auc_std    <- eval_log$test_auc_std[best_iter_index]
threshold_auc <- best_auc_mean - best_auc_std

candidates <- eval_log %>% filter(test_auc_mean >= threshold_auc)
optimal_rounds_1se <- min(candidates$iter)

message(sprintf("Absolute Best Rounds: %d (AUC: %.5f)", best_iter_index, best_auc_mean))
message(sprintf("1-SE Rule Rounds:     %d (Threshold: %.5f)", optimal_rounds_1se, threshold_auc))

## Train the 1-SE Model
XGBoost_finalmodel_1SE <- xgb.train(
  params = final_params_bayes,
  data = dtrain,
  nrounds = optimal_rounds_1se,
  verbose = 0
)

## Train the full model.
XGBoost_finalmodel <- xgb.train(
  params = final_params_bayes,
  data = dtrain,
  nrounds = optimal_rounds_final,
  verbose = 0
)

## Compare the AUC score of the test set and compare 1-SE with the full iteration model.
## Full:
XGBoost_test_probs <- predict(XGBoost_finalmodel, dtest)
XGBoost_test_ROC <- roc(test_y, XGBoost_test_probs, quiet = TRUE)
XGBoost_test_AUC <- auc(XGBoost_test_ROC)
print(paste("Final Test Set AUC:", round(XGBoost_test_AUC, 5)))

## 1-SE:
XGBoost_test_probs_1SE <- predict(XGBoost_finalmodel_1SE, dtest)
XGBoost_test_ROC_1SE   <- roc(test_y, XGBoost_test_probs_1SE, quiet = TRUE)
XGBoost_test_AUC_1SE   <- auc(XGBoost_test_ROC_1SE)
print(paste("Final Test AUC (1-SE Rule):", round(XGBoost_test_AUC_1SE, 5)))

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

## Data summary and consolidation.
results_ds <- results_ds %>% mutate(Method = "Grid Search", Iteration = 1:n())
results_rs <- results_rs %>% mutate(Method = "Random Search", Iteration = 1:n())
results_bayes <- results_bayes %>% mutate(Method = "Bayesian Opt", Iteration = 1:n())

common_cols <- c("Method", "Iteration", "eta", "max_depth", "subsample", "colsample_bytree", "AUC")
all_search_results <- bind_rows(
  results_ds %>% select(all_of(common_cols)),
  results_rs %>% select(all_of(common_cols)),
  results_bayes %>% select(all_of(common_cols))
)

XGBoost_method_performance <- data.frame(
  Method = c("Grid Search", "Random Search", "Bayesian Opt"),
  Train_AUC = c(XGBoost_ds_Train_AUC, XGBoost_rs_Train_AUC, XGBoost_bayes_Train_AUC)
)

## Visualisation: Method AUC comparison.
colors <- c(
  "Grid Search"   = blue,
  "Random Search" = orange,  
  "Bayesian Opt"  = red   
)

Plot_Train_AUC <- ggplot(XGBoost_method_performance, aes(x = reorder(Method, Train_AUC), y = Train_AUC, 
                                                         fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Train_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "Hyperparameter Tuning Method",
    y = "AUC-Score (Training Set)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"), # Adjusted to "grey30" for safety
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_XGBoost_Directory, "01_HyperparameterTuningMethods_AUC_Training.png")
ggsave(
  filename = Path,
  plot = Plot_Train_AUC,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: Learning curve / boosting iterations.
cv_log <- check_cv_bayes$evaluation_log

plot_LC_bayesOptim <- ggplot(cv_log, aes(x = iter, y = test_auc_mean)) +
  geom_ribbon(aes(ymin = test_auc_mean - test_auc_std, 
                  ymax = test_auc_mean + test_auc_std), 
              alpha = 0.2, fill = "#2c3e50") +
  scale_y_continuous(labels = scales::percent, limits = c(0.75, 0.825)) +
  geom_line(color = "#2c3e50", linewidth = 1) +
    geom_vline(xintercept = optimal_rounds_rs, linetype = "dashed", color = red, linewidth = 0.8) +
    annotate("text", 
           x = optimal_rounds_rs + (max(cv_log$iter) * 0.02), # Offset slightly to the right
           y = min(cv_log$test_auc_mean), 
           label = paste("Optimal:", optimal_rounds_rs), 
           color = "#e41a1c", 
           hjust = 0, fontface = "bold") +
    labs(title = "",
       subtitle = "Cross-Validation AUC +/- 1 Std Dev (Bayesian Optim.)",
       x = "Number of Boosting Iterations",
       y = "AUC-Score (Training Set)") +
    theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_XGBoost_Directory, "02_LearningCurve_BayesianOptimization_Training.png")
ggsave(
  filename = Path,
  plot = plot_LC_bayesOptim,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: Feature Importance.

importance_matrix <- xgb.importance(model = model_bayes)
top_features <- head(importance_matrix, 10)

plot_XGBoost_FeatureImport_Train <- ggplot(top_features, 
                        aes(x = Gain, 
                            y = reorder(Feature, Gain))) +
    geom_col(fill = blue, width = 0.7) + 
    geom_text(aes(label = scales::percent(Gain, accuracy = 0.1)), 
            hjust = -0.1, size = 4.5, fontface = "bold", color = "grey30") +
    scale_x_continuous(labels = scales::percent, 
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "",
    subtitle = "",
    x = "Relative Contribution",
    y = NULL # No label needed for feature names
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 11, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#d9d9d9"),
    
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_XGBoost_Directory, "03_FeatureImportance_BayesianOptimization_Training.png")
ggsave(
  filename = Path,
  plot = plot_XGBoost_FeatureImport_Train,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

##==============================##
## Model performance of the best hyperparameter tuning method within the TEST set.
##==============================##

## Visualisation: Test-set performance. Full model and 1-SE Rule.
XGBoost_method_performance <- data.frame(
  Method = c("Training Set", "Test Set (1-SE)", "Test Set"),
  Test_AUC = c(XGBoost_bayes_Train_AUC, XGBoost_test_AUC_1SE, XGBoost_test_AUC)
)

colors <- c(
  "Training Set"   = grey,
  "Test Set (1-SE)" = blue,  
  "Test Set"  = red   
)

Plot_Test_AUC <- ggplot(XGBoost_method_performance, aes(x = reorder(Method, Test_AUC), y = Test_AUC, 
                                                         fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Test_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "",
    y = "AUC-Score"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"), # Adjusted to "grey30" for safety
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_XGBoost_Directory, "04_AUC_Test.png")
ggsave(
  filename = Path,
  plot = Plot_Test_AUC,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: Calibration (predicted vs observed per bracket).
calib_data <- data.frame(
  actual = test_y,
  prob = XGBoost_test_probs
) %>%
  mutate(bin = ntile(prob, 10)) %>%
  group_by(bin) %>%
  summarise(
    mean_prob = mean(prob),
    observed_rate = mean(actual),
    n = n()
  )

calib_plot_data <- calib_data %>%
  select(bin, mean_prob, observed_rate) %>%
  rename(Predicted = mean_prob, Observed = observed_rate) %>%
  pivot_longer(cols = c("Predicted", "Observed"), 
               names_to = "Type", 
               values_to = "Rate") %>%
  mutate(Type = factor(Type, levels = c("Predicted", "Observed")))

plot_calib_bars <- ggplot(calib_plot_data, aes(x = factor(bin), y = Rate, fill = Type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = scales::percent(Rate, accuracy = 0.1)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold", 
            color = "black") +
  scale_fill_manual(values = c("Predicted" = blue, 
                               "Observed"  = grey)) + 
  scale_y_continuous(labels = scales::percent, 
                     expand = expansion(mult = c(0, 0.15))) + 
  labs(
    title = "",
    subtitle = "",
    x = "Risk Decile (1 = Lowest Risk, 10 = Highest Risk)",
    y = "Default Rate",
    fill = "" 
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    legend.position = "top", 
    legend.text = element_text(size = 12, face = "bold"),
    axis.title.x = element_text(size = 13, face = "bold", margin = ggplot2::margin(t = 10)),
    axis.title.y = element_text(size = 13, face = "bold", margin = ggplot2::margin(r = 10)),
    panel.grid.major.x = element_blank(), 
    panel.grid.minor = element_blank(),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_XGBoost_Directory, "05_CalibrationChart_BayesianOptimization_Test.png")
ggsave(
  filename = Path,
  plot = plot_calib_bars,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)


}, error = function(e) message(e))

} else {
  cat("\n=== Skipping XGBoost (package not available for R < 4.3) ===\n")
}

#==============================================================================#
#==============================================================================#
#==============================================================================#

