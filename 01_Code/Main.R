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

packages <- c("dplyr", "caret", "lubridate", "purrr",
              "Matrix", "pROC",        ## Sparse Matrices and efficient AUC computation.
              "xgboost",               ## XGBoost library.
              "rBayesianOptimization"  ## Bayesian Optimization.
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
Charts_Directory <- file.path(Path, "03_Charts")

## Charts Directories.
Charts_XGBoost_Directory <- file.path(Charts_Directory, "XGBoost")


Functions_Directory <- file.path(Path, "01_Code/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Data Sampling.
set.seed(123)

## Charts.
width <- 3750
heigth <- 1833


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

#==============================================================================#
#==== 04 - GLMs ===============================================================#
#==============================================================================#

#==== 04A - GLMs ==============================================================#



#==== 04B - Regularized GLMs ==================================================#


#==============================================================================#
#==== 05 - Decision Trees =====================================================#
#==============================================================================#

#==== 05A - Random Forest =====================================================#


#==== 05B - AdaBoost ==========================================================#


#==== 05C - XGBoost ===========================================================#

## To-Do: Fix overall parameters at the beginning of the code.
## Comparison of hyperparameter tuning methods.
## Visualisations and Outputs.

set.seed(123)

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
results_ds <- pmap_dfr(grid_ds, XGBoost_gridsearch) %>%
    arrange(desc(AUC))
  
print("--- Top 5 Discrete Models ---")
print(head(results_ds, 5))
  
}, error = function(e) print(e))

## Train the final model (optimal combination of hyperparameters).
tryCatch({
best_ds <- results_ds[1, ]
  
final_params_ds <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
    eta = best_ds$eta, max_depth = best_ds$max_depth, 
    subsample = best_ds$subsample, colsample_bytree = best_ds$colsample_bytree
)

model_ds <- xgb.train(params = final_params_ds, 
                      data = dtrain, 
                      nrounds = best_ds$Best_Rounds, 
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
  results_rs <- pmap_dfr(grid_rs, XGBoost_gridsearch) %>%
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

model_rs <- xgb.train(params = final_params_rs, 
                      data = dtrain,
                      nrounds = best_rs$Best_Rounds, 
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
n_init_points <- 5
n_iter_bayes  <- 15
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
    acq = "ucb", 
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
  
## Determine optimal rounds for the winner
check_cv <- xgb.cv(
    params = final_params_bayes,
    data = dtrain,
    nrounds = 2000,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = 0,
    maximize = TRUE
  )
  
optimal_rounds <- check_cv$best_iteration
  
## Final training
model_bayes <- xgb.train(
                        params = final_params_bayes, 
                        data = dtrain,
                        nrounds = optimal_rounds, 
                        verbose = 0
  )
  
}, error = function(e) message(e))

## Compute the training AUC.
XGBoost_bayes_probs <- predict(model_bayes, dtrain)
XGBoost_bayes_Train_ROC <- roc(train_y, XGBoost_bayes_probs, quiet = TRUE)
XGBoost_bayes_Train_AUC <- auc(XGBoost_bayes_Train_ROC)
print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))


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
plot_final_auc <- ggplot(final_performance, aes(x = reorder(Method, Train_AUC), y = Train_AUC, fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(Train_AUC, 5)), vjust = -0.5, size = 5) +
  coord_cartesian(ylim = c(min(final_performance$Train_AUC) * 0.99, max(final_performance$Train_AUC) * 1.005)) +
  labs(title = "Final Model Performance by Tuning Strategy",
       subtitle = "Comparison of Training AUC on the full dataset",
       x = "Tuning Method",
       y = "Training AUC") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

print(plot_final_auc)

Path <- file.path(Charts_XGBoost_Directory, "01_HyperparameterTuningMethods_AUC_Training.png")
ggsave(
  filename = Path,
  plot = plot_importance_gbm,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: Search Efficiency (Convergence).
convergence_data <- all_search_results %>%
  group_by(Method) %>%
  mutate(Best_AUC_So_Far = cummax(AUC)) %>%
  ungroup()

plot_convergence <- ggplot(convergence_data, aes(x = Iteration, y = Best_AUC_So_Far, color = Method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  labs(title = "Optimization Convergence",
       subtitle = "Best CV AUC score found at each iteration",
       x = "Iteration Number",
       y = "Best CV AUC Found") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

print(plot_convergence)

## Visualization: Search Space Exploration
plot_space <- ggplot(all_search_results, aes(x = eta, y = max_depth, color = AUC)) +
  geom_jitter(width = 0.005, height = 0.2, size = 3, alpha = 0.8) +
  facet_wrap(~Method) +
  scale_color_viridis_c(option = "magma", direction = -1) +
  labs(title = "Hyperparameter Search Space Exploration",
       subtitle = "Comparison of sampled Eta vs Max Depth",
       x = "Learning Rate (eta)",
       y = "Max Depth") +
  theme_bw() +
  theme(legend.position = "bottom")

print(plot_space)

##==============================##
## Model performance of the best hyperparameter tuning method within the TEST set.
##==============================##


#==============================================================================#
#==============================================================================#
#==============================================================================#

