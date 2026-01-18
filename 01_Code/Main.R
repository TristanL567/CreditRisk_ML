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
              "purrr", "tibble"      
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

Data_filtered <- DataPreprocessing(DataPreprocessing, Tolerance = 2)

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

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

## Now work with:
## Train_Transformed
## Test_Transformed

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
Strat_Data <- Train_Transformed %>%
  mutate(
    id = Train_with_id$id,
    sector = Train_with_id$sector 
  )

Data_Train_CV_stratified_sampling <- MVstratifiedsampling_CV(Strat_Data, num_folds = N_folds)
Data_Train_CV_List <- Data_Train_CV_stratified_sampling[["fold_list"]]
Data_Train_CV_Vector <- Data_Train_CV_stratified_sampling[["fold_vector"]]

print(paste("Folds generated:", length(Data_Train_CV_List)))
length(Data_Train_CV_Vector)

#==============================================================================#
#==== 04 - GLMs ===============================================================#
#==============================================================================#

#==== 04A - Regularized GLMs ==================================================#

tryCatch({
  
  time_GLM <- system.time({
    
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

##==============================##
## Discrete grid search (hyperparameter tuning of alpha and lambda).
##==============================##

time_GLM_ds <- system.time({
  
## Prepare the grid.
grid_ds_glm <- tibble(
  alpha = seq(0, 1, by = 0.1) # 0 = Ridge, 1 = Lasso
) %>%
  mutate(
    current_iter = 1:n(),
    total_iters = n()
  )
## Implement the vectorized grid search function.
results_ds_glm <- pmap_dfr(grid_ds_glm, GLM_gridsearch) %>%
  arrange(desc(CV_AUC))

print("--- Top 5 Discrete GLM Models ---")
print(head(results_ds_glm, 5))

})

##==============================##
## Random grid search (hyperparameter tuning of alpha and lambda).
##==============================##

time_GLM_rs <- system.time({
  
n_iter_rs <- 20

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
results_rs_glm <- pmap_dfr(grid_rs_glm, GLM_gridsearch) %>%
  arrange(desc(CV_AUC))

print("--- Top 5 Random GLM Models ---")
print(head(results_rs_glm, 5))

})

##==============================##
## Bayesian Optimization (hyperparameter tuning of alpha and lambda).
##==============================##

time_GLM_bo <- system.time({
  
## Prepare the grid.
bounds_glm <- list(
  alpha = c(0, 1)
)

# 3. Run Optimization
n_init_glm <- 5
n_iter_glm <- 15
current_bayes_iter <- 0
total_bayes_runs <- n_init_glm + n_iter_glm # Define this before running BO

print("Starting Bayesian Optimization for GLM...")

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

current_bayes_iter <- 0

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

}) 

##==============================##
## Performance of the model with the highest training AUC in the test set.
##==============================##

## Train results.
print(head(results_ds_glm, 5))
print(head(results_rs_glm, 5))
print(head(results_bayes_glm, 5))


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
  foldid = Data_Train_CV_Vector,   
  standardize = TRUE        
)

## Compute Test AUC: Champion (lambda.min)
probs_champion <- predict(final_cv_glm, 
                          newx = test_matrix, 
                          s = "lambda.min", 
                          type = "response")

roc_champion <- roc(test_y, as.vector(probs_champion), quiet = TRUE)
auc_champion <- pROC::auc(roc_champion)

## Compute Test AUC: 1-SE Rule (lambda.1se)
probs_1se <- predict(final_cv_glm, 
                     newx = test_matrix, 
                     s = "lambda.1se", 
                     type = "response")
colnames(probs_1se) <- "prob"

roc_1se <- roc(test_y, as.vector(probs_1se), quiet = TRUE)
auc_1se <- pROC::auc(roc_1se)

message("------------------------------------------------")
message(sprintf("Final GLM Test AUC (Champion/Min):  %.5f", auc_champion))
message(sprintf("Final GLM Test AUC (1-SE Rule):     %.5f", auc_1se))
message("------------------------------------------------")

n_vars_champion <- sum(coef(final_cv_glm, s = "lambda.min") != 0)
n_vars_1se      <- sum(coef(final_cv_glm, s = "lambda.1se") != 0)

message(sprintf("Variables Selected (Champion):      %d", n_vars_champion))
message(sprintf("Variables Selected (1-SE Rule):     %d", n_vars_1se))

## Regularized GLM performance in the Test-set.

best_cv_bayes_glm <- max(results_bayes_glm$CV_AUC)
glm_performance_test <- tibble(
  Method    = c("Test Set", "Test Set (1-SE)", "Training Set"),
  Test_AUC = c(auc_champion, auc_1se, best_cv_bayes_glm)
)

print("--- GLM Test-set AUC ---")
print(glm_performance_test)

## Brier Score of the full model.
actuals_num <- as.numeric(as.character(test_y))
GLM_BrierScore <- BrierScore(probs_champion, actuals_num)
GLM_BrierScore_1SE <- BrierScore(probs_1se, actuals_num)

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

# Extract the best Cross-Validation AUC from each GLM results table
best_cv_glm_ds    <- max(results_ds_glm$CV_AUC)    
best_cv_glm_rs    <- max(results_rs_glm$CV_AUC)   
best_cv_glm_bayes <- max(results_bayes_glm$CV_AUC)

# Create the plotting dataframe
glm_cv_performance <- data.frame(
  Method = c("Grid Search", "Random Search", "Bayesian Opt"),
  CV_AUC = c(best_cv_glm_ds, best_cv_glm_rs, best_cv_glm_bayes)
)

## Data.
glm_path_data <- data.frame(
  lambda = final_cv_glm$lambda,
  log_lambda = log(final_cv_glm$lambda),
  cv_mean = final_cv_glm$cvm,  
  cv_std = final_cv_glm$cvsd  
)

optimal_lambda_log <- log(final_cv_glm$lambda.min)
optimal_auc_val    <- max(final_cv_glm$cvm)

## Visualisation: Method AUC comparison.

colors <- c(
  "Grid Search"   = blue,
  "Random Search" = orange,
  "Bayesian Opt" = red
)

Plot_Train_AUC <- ggplot(glm_cv_performance, aes(x = reorder(Method, CV_AUC), y = CV_AUC, 
                                                         fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(CV_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "Hyperparameter Tuning Method",
    y = "AUC-Score (CV)"
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

Path <- file.path(Charts_GLM_Directory, "01_GLM_HyperparameterTuningMethods_AUC_Training.png")
ggsave(
  filename = Path,
  plot = Plot_Train_AUC,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: Penalization strength or Bayesian Optimization.
plot_GLM_LearningCurve <- ggplot(glm_path_data, aes(x = log_lambda, y = cv_mean)) +
  geom_ribbon(aes(ymin = cv_mean - cv_std, 
                  ymax = cv_mean + cv_std), 
              alpha = 0.2, fill = "#2c3e50") +
  geom_line(color = "#2c3e50", linewidth = 1) +
  geom_vline(xintercept = optimal_lambda_log, 
             linetype = "dashed", color = "#B22222", linewidth = 0.8) +
  annotate("text", 
           x = optimal_lambda_log + 0.5, # Offset label slightly to the right
           y = min(glm_path_data$cv_mean), 
           label = paste("Optimal Log(Lambda):", round(optimal_lambda_log, 2)), 
           color = "#B22222", 
           hjust = 0, fontface = "bold") +
  scale_x_reverse() + 
  scale_y_continuous(labels = scales::percent) +
  
  labs(title = "",
       subtitle = "",
       x = "Log(Lambda)",
       y = "AUC-Score (CV)") +
  
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

Path <- file.path(Charts_GLM_Directory, "02_RegularizationPath_GLM.png")
ggsave(
  filename = Path,
  plot = plot_GLM_LearningCurve,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

##==============================##
## Visualisations of the test-set performance.
##==============================##

## Visualisation: AUC in the test set.

colors <- c(
  "Training Set"   = grey,
  "Test Set (1-SE)" = blue,
  "Test Set" = red
)

Plot_Test_AUC <- ggplot(glm_performance_test, aes(x = reorder(Method, Test_AUC), y = Test_AUC, 
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
    y = "AUC-Score (Test Set)"
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

Path <- file.path(Charts_GLM_Directory, "04_GLM_AUC_Test.png")
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
glm_risk_deciles <- ntile(probs_1se, 10)

calib_data_glm <- data.frame(
  actual = test_y,
  prob   = probs_1se,
  bin    = glm_risk_deciles
) %>%
  group_by(bin) %>%
  summarise(
    mean_prob     = mean(prob),
    observed_rate = mean(actual),
    n             = n()
  )

calib_plot_data_glm <- calib_data_glm %>%
  select(bin, mean_prob, observed_rate) %>%
  rename(Predicted = mean_prob, Observed = observed_rate) %>%
  pivot_longer(cols = c("Predicted", "Observed"), 
               names_to = "Type", values_to = "Rate") %>%
  mutate(Type = factor(Type, levels = c("Predicted", "Observed")))

plot_calib_bars <- ggplot(calib_plot_data_glm, aes(x = factor(bin), y = Rate, fill = Type)) +
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

Path <- file.path(Charts_GLM_Directory, "05_GLM_CalibrationChart_BayesianOptimization_Test.png")
ggsave(
  filename = Path,
  plot = plot_calib_bars,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)


}) ## End of the time counter.
  
#### Compare the time.
  
timing_comparison_GLM <- tibble(
    Method = c("Grid Search", "Random Search", "Bayesian Optimization", "GLM Code"),
    Time_Seconds = c(time_GLM_ds["elapsed"], time_GLM_rs["elapsed"], 
                     time_GLM_bo["elapsed"], time_GLM["elapsed"])
  ) %>%
    mutate(
      Time_Minutes = Time_Seconds / 60,
      Iterations = c(nrow(grid_ds_glm), 
                     nrow(grid_rs_glm), 
                     nrow(bayes_out_glm$History),
                     NA)
    ) %>%
    arrange(Time_Seconds)
  
}, error = function(e) message(e))

#==============================================================================#
#==== 05 - Decision Trees =====================================================#
#==============================================================================#

#==== 05A - Random Forest =====================================================#

tryCatch({
  
time_RF <- system.time({
    
##==============================##
## General Parameters.
##==============================##
  
temp_name <- "mtry_rounds_NOPIT"
  
HPO_CONFIG <- list(
    learner_name = "ranger",  # Changed from xgboost to ranger
    n_evals = 30L,
    stagnation_iters = 50L,
    stagnation_threshold = 0.001,
    n_folds = 5L,
    train_prop = 0.7,
    checkpoint_file = paste0("hpo_checkpoint_ranger_", temp_name, ".rds") ######naming
)

## Other.
n_cores <- max(1, parallel::detectCores() - 1)
cat("Setting up parallelization with", n_cores, "cores\n")

set.seed(123)
plan(multisession, workers = n_cores)
cat("Parallel backend:", class(plan())[1], "\n")
cat("Number of workers:", nbrOfWorkers(), "\n")

#### Load the checkpoint data.
# non_feature_cols <- c("y")
# feature_names <- setdiff(colnames(Train_Transformed_RF), non_feature_cols)
# n_features <- length(feature_names)
# 
# checkpoint_path <- file.path(Data_RF, 
#                              HPO_CONFIG$checkpoint_file)
# 
# hpo_results <- readRDS(checkpoint_path)

##==============================##
## Data preparation.
##==============================##  
  
Train_Transformed_RF <- as.data.table(Train_Transformed)
Test_Transformed_RF <- as.data.table(Test_Transformed)
Test_Transformed_RF[, y := factor(y)]
Train_Transformed_RF[, y := factor(y)]
size_map <- c("Tiny" = 0, "Small" = 1)
Train_Transformed_RF[, size := size_map[size]]
Test_Transformed_RF[, size := size_map[size]]
  
# Convert to factor with SAME levels for both
all_levels <- union(Train_Transformed_RF$sector, Test_Transformed_RF$sector)
  
Train_Transformed_RF[, sector := factor(sector, levels = all_levels)]
Test_Transformed_RF[, sector := factor(sector, levels = all_levels)]
  
# Now model.matrix will create identical columns
sector_dummies_train <- model.matrix(~sector - 1, Train_Transformed_RF)
sector_dummies_test <- model.matrix(~sector - 1, Test_Transformed_RF)
Train_Transformed_RF <- cbind(Train_Transformed_RF[, -"sector"], sector_dummies_train)
Test_Transformed_RF <- cbind(Test_Transformed_RF[, -"sector"], sector_dummies_test)
  
task_train <- TaskClassif$new(
    id = "hpo_train",
    backend = Train_Transformed_RF,
    target = "y",
    positive = levels(Train_Transformed_RF$y)[2]
  )  

## Create the custom CV-Folds.
custom_cv <- create_custom_cv(task_train, Data_Train_CV_Vector)

##==============================##
## Optimization.
##==============================##  
  
non_feature_cols <- c("y")
feature_names <- setdiff(colnames(Train_Transformed_RF), non_feature_cols)
n_features <- length(feature_names)

#optimization
hpo_results <- run_hpo(task_train, custom_cv, n_features, HPO_CONFIG)

#eval on test
task_test <- TaskClassif$new(
  id = "hpo_test",
  backend = Test_Transformed_RF,
  target = "y",
  positive = levels(Test_Transformed_RF$y)[2]
)

cat("\n", strrep("=", 70), "\n")
cat("FINAL EVALUATION ON TEST SET\n")
cat(strrep("=", 70), "\n")

eval_results <- train_and_eval(hpo_results$best_params, task_train, task_test)

cat("\nTest Metrics:\n")
cat("  AUC:", round(eval_results$auc, 4), "\n")
cat("  Accuracy:", round(eval_results$acc, 4), "\n")
cat("  Brier Score:", round(eval_results$brier, 4), "\n")

RF_BrierScore <- eval_results$brier
  
##==============================##
## Visualisation.
##==============================##    
  
summary_table <- data.table(
  Metric = c("Total Evaluations", "Warmup Points", "BO Evaluations",
             "Best CV AUC", "Test AUC", "Test Accuracy"),
  Value = c(hpo_results$n_evals,
            min(hpo_results$n_initial_design, hpo_results$n_evals),
            max(0, hpo_results$n_evals - hpo_results$n_initial_design),
            # round(hpo_results$time, 2),
            round(hpo_results$best_cv_auc, 4),
            round(eval_results$auc, 4),
            round(eval_results$acc, 4))
)

cat("\n")
print(summary_table)

cat("\nBest hyperparameters:\n")
print(hpo_results$best_params)

# ============================================================================
# PLOT
# ============================================================================

# p1 <- plot_optimization_history(hpo_results)
# print(p1)
# Path <- file.path(Charts_RF_Directory, paste0("01_RF_mbo_optimization_history_", temp_name, ".png"))
# ggsave(
#   filename = Path,
#   plot = p1,
#   width = width,
#   height = heigth,
#   units = "px",
#   dpi = 300,
#   limitsize = FALSE
# )

# ggsave(paste0("mbo_optimization_history_", temp_name, ".png"), p1, width = 10, height = 6, dpi = 150) ####naming

# ============================================================================
# CLEANUP
# ============================================================================

plan(sequential)
cat("\nParallel backend reset to sequential.\n")

#==============================================================================#
#==== 06 - Save Final Model ===================================================#
#==============================================================================#

# Define output directory and filename
# Model_Directory <- file.path(Path, "04_Models")
# if (!dir.exists(Model_Directory)) {
#   dir.create(Model_Directory, recursive = TRUE)
# }
# 
# # Create a comprehensive model object to save
# final_model_object <- list(
# # The trained learner (contains the model)
#   learner = eval_results$learner,
#   
# # Best hyperparameters from HPO
#   best_params = hpo_results$best_params,
#   
# # Performance metrics
#   metrics = list(
#     cv_auc = hpo_results$best_cv_auc,
#     test_auc = eval_results$auc,
#     test_accuracy = eval_results$acc,
#     test_brier = eval_results$brier
#   ),
#   
# # Metadata
#   metadata = list(
#     model_type = "ranger",
#     n_features = n_features,
#     feature_names = feature_names,
#     training_date = Sys.time(),
#     n_hpo_evals = hpo_results$n_evals,
#     hpo_time_seconds = hpo_results$time
#   )
# )
# 
# # Save the model object
# model_filename <- file.path(Model_Directory, 
#                             paste0("ranger_", temp_name, ####naming
#                                    format(Sys.Date(), "%Y%m%d"), 
#                                    ".rds"))
# 
# saveRDS(final_model_object, model_filename)
# cat("\nFinal model saved to:", model_filename, "\n")

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

#### Data.
archive_dt <- as.data.table(hpo_results$archive)
n_initial  <- hpo_results$n_initial_design

best_cv_rs <- max(archive_dt[1:n_initial]$classif.auc)
best_cv_bayes <- max(archive_dt[(n_initial + 1):nrow(archive_dt)]$classif.auc)

RF_CV_performance <- data.frame(
  Method = c("Random Search", "Bayesian Opt"),
  CV_AUC = c(best_cv_rs, best_cv_bayes)
)

#### Plots.
colors <- c(
  "Random Search" = orange,  
  "Bayesian Opt"  = red    
)

Plot_CV_AUC <- ggplot(RF_CV_performance, aes(x = reorder(Method, CV_AUC), y = CV_AUC, 
                                             fill = Method)) +
  geom_col(width = 0.5, show.legend = FALSE) + # Thinner bars for 2 items
  geom_text(aes(label = paste0(round(CV_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "Hyperparameter Tuning Method",
    y = "AUC-Score (CV)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"), 
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_RF_Directory, "01_RF_HyperparameterTuningMethods_AUC_CV.png")
ggsave(filename = Path, 
       plot = Plot_CV_AUC, 
       width = width, 
       height = heigth, 
       units = "px", 
       dpi = 300, 
       limitsize = FALSE)

## Learning-Curve.

archive_dt[, best_so_far := cummax(classif.auc)]
archive_dt[, iteration := 1:.N]

plot_LC_RF <- ggplot(archive_dt, aes(x = iteration, y = best_so_far)) +
  geom_line(color = "#2c3e50", linewidth = 1) +
  geom_point(aes(y = classif.auc), color = "grey70", alpha = 0.5, size = 1) +
  geom_vline(xintercept = n_initial, linetype = "dashed", color = orange, linewidth = 0.8) +
  annotate("text", x = n_initial + 1, y = min(archive_dt$classif.auc), 
           label = "Start BayesOpt", color = orange, hjust = 0, vjust = -1, fontface = "bold") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "",
       subtitle = "",
       x = "Evaluation Iteration",
       y = "AUC-Score (CV)") +
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

Path <- file.path(Charts_RF_Directory, "02_RF_OptimizationHistory.png")
ggsave(filename = Path,
       plot = plot_LC_RF, 
       width = width, 
       height = heigth, 
       units = "px", 
       dpi = 300, 
       limitsize = FALSE)

##### Feature Importance.

rf_imp <- eval_results$learner$importance() 
importance_df <- data.frame(
  Feature = names(rf_imp),
  Gain = as.numeric(rf_imp)
) %>%
  arrange(desc(Gain)) %>%
  head(10)

importance_df$Gain <- importance_df$Gain / sum(eval_results$learner$importance())

plot_RF_FeatureImport <- ggplot(importance_df, aes(x = Gain, y = reorder(Feature, Gain))) +
  geom_col(fill = blue, width = 0.7) + 
  geom_text(aes(label = scales::percent(Gain, accuracy = 0.1)), 
            hjust = -0.1, size = 4.5, fontface = "bold", color = "grey30") +
  scale_x_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.15))) +
  labs(title = "", subtitle = "", x = "Relative Importance (Impurity)", y = NULL) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_blank(),
    axis.text.y = element_text(size = 11, face = "bold", color = "black"),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_RF_Directory, "03_RF_FeatureImportance_BayesianOptimization.png")
ggsave(filename = Path, 
       plot = plot_RF_FeatureImport, 
       width = width, 
       height = heigth, 
       units = "px", 
       dpi = 300, 
       limitsize = FALSE)

##==============================##
## Visualisations: Test-set.
##==============================##

#### Test-AUC.
RF_method_performance <- data.frame(
  Metric = c("Validation (CV)", "Test Set"),
  AUC_Score = c(best_cv_bayes, eval_results$auc)
)

colors_test <- c("Validation (CV)" = grey, 
                 "Test Set" = red)

Plot_Test_AUC <- ggplot(RF_method_performance, aes(x = reorder(Metric, AUC_Score), y = AUC_Score, fill = Metric)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(AUC_Score * 100, 1), "%")), vjust = -0.5, size = 5, fontface = "bold") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors_test) +
  labs(title = "", subtitle = "", x = "", y = "AUC-Score") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_RF_Directory, "04_RF_AUC_Test.png")
ggsave(filename = Path, 
       plot = Plot_Test_AUC, 
       width = width, 
       height = heigth, 
       units = "px", 
       dpi = 300, 
       limitsize = FALSE)

#### Calibration chart.
rf_pred_obj <- eval_results$pred
rf_probs <- if("1" %in% colnames(rf_pred_obj$prob)) {
  rf_pred_obj$prob[, "1"] 
} else { 
  rf_pred_obj$prob[, 2] 
}

calib_data_rf <- data.frame(
  actual = test_y,
  prob   = rf_probs, # RF Prediction
  bin    = ntile(probs_1se, 10) # GLM Buckets!
) %>%
  group_by(bin) %>%
  summarise(
    mean_prob     = mean(prob),
    observed_rate = mean(actual),
    n             = n()
  )

calib_plot_data_rf <- calib_data_rf %>%
  select(bin, mean_prob, observed_rate) %>%
  rename(Predicted = mean_prob, Observed = observed_rate) %>%
  pivot_longer(cols = c("Predicted", "Observed"), 
               names_to = "Type", values_to = "Rate") %>%
  mutate(Type = factor(Type, levels = c("Predicted", "Observed")))

# Plotting
plot_calib_bars <- ggplot(calib_plot_data_rf, aes(x = factor(bin), y = Rate, fill = Type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = scales::percent(Rate, accuracy = 0.1)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold", 
            color = "black") +
  scale_fill_manual(values = c("Predicted" = blue, "Observed" = grey)) + 
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, 0.15))) + 
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
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_RF_Directory, "05_RF_CalibrationChart_Test.png")
ggsave(filename = Path, 
       plot = plot_calib_bars, 
       width = width, 
       height = heigth, 
       units = "px", 
       dpi = 300, 
       limitsize = FALSE)



}) ## time counter.

}, error = function(e) message(e))

#==== 05B - AdaBoost ==========================================================#

tryCatch({
  

  if (!exists("target")) target <- "y"
  if (!exists("positive_class")) positive_class <- "1"
  if (!exists("N_folds")) N_folds <- 5
  

  if (!exists("cv_folds_list")) {
    if (exists("Data_Train_CV_List")) {
      cv_folds_list <- Data_Train_CV_List
    } else if (exists("Data_Train_CV_stratified_sampling") &&
               is.list(Data_Train_CV_stratified_sampling) &&
               "fold_list" %in% names(Data_Train_CV_stratified_sampling)) {
      cv_folds_list <- Data_Train_CV_stratified_sampling[["fold_list"]]
    } else {
      stop("cv_folds_list not found. Run Step 03C (stratified folds) before AdaBoost.")
    }
  }
  

  if (!is.list(cv_folds_list) || length(cv_folds_list) < 2) {
    stop("cv_folds_list must be a non-empty list of fold indices.")
  }
  

  ## Data preparation.

  if (!exists("Train_Transformed") || !exists("Test_Transformed")) {
    stop("Train_Transformed / Test_Transformed not found. Run your preprocessing first.")
  }
  
  Train_AB <- Train_Transformed
  Test_AB  <- Test_Transformed
  
  # ensure target exists
  if (!target %in% names(Train_AB) || !target %in% names(Test_AB)) {
    stop(paste0("Target column '", target, "' not found in Train/Test."))
  }
  
  # enforce factor target + consistent levels
  Train_AB[[target]] <- as.factor(Train_AB[[target]])
  Test_AB[[target]]  <- factor(Test_AB[[target]], levels = levels(Train_AB[[target]]))
  
  # check positive class exists
  if (!positive_class %in% levels(Train_AB[[target]])) {
    stop(paste0("positive_class='", positive_class, "' not in levels(Train_AB[[target]]). Levels are: ",
                paste(levels(Train_AB[[target]]), collapse = ", ")))
  }
  
  ##==============================##
  ## Compute FULL-TRAIN weights (same logic as AdaBoost_cv_auc)
  ##==============================##
  tab_full <- table(Train_AB[[target]])
  if (length(tab_full) < 2 || !positive_class %in% names(tab_full)) {
    stop("Weights cannot be computed: training data does not contain both classes or positive_class not found.")
  }
  neg_class <- setdiff(names(tab_full), positive_class)[1]
  
  w_pos <- as.numeric(sum(tab_full) / (2 * tab_full[positive_class]))
  w_neg <- as.numeric(sum(tab_full) / (2 * tab_full[neg_class]))
  
  w_full <- ifelse(Train_AB[[target]] == positive_class, w_pos, w_neg)
  w_full <- pmin(w_full, 50)
  
  ##==============================##
  ## Discrete Grid Search for Hyperparameter Tuning.
  ##==============================##
  grid_ds <- expand.grid(
    mfinal    = c(50, 100, 200),
    maxdepth  = c(1, 2, 3),
    minsplit  = c(10, 20, 50),
    minbucket = 5
  ) %>%
    dplyr::mutate(
      nrounds = NA_integer_,
      early_stopping_rounds = NA_integer_,
      current_iter = 1:dplyr::n(),
      total_iters  = dplyr::n()
    )
  
  message(paste("Total combinations to test:", nrow(grid_ds)))
  
  results_ds <- purrr::pmap_dfr(
    grid_ds,
    ADABoost_gridsearch,
    folds_custom = cv_folds_list
  ) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(Method = "Grid Search", Iteration = dplyr::row_number())
  
  message("--- Top 5 Discrete Models (AdaBoost) ---")
  print(head(results_ds, 5))
  
  ##==============================##
  ## Random Search for Hyperparameter Tuning.
  ##==============================##
  n_iter <- 20
  
  grid_rs <- data.frame(
    mfinal    = sample(seq(25, 300, by = 25), n_iter, replace = TRUE),
    maxdepth  = sample(1:4, n_iter, replace = TRUE),
    minsplit  = sample(c(5, 10, 20, 50, 100), n_iter, replace = TRUE),
    minbucket = sample(c(2, 5, 10, 20, 50), n_iter, replace = TRUE)
  ) %>%
    dplyr::mutate(
      nrounds = NA_integer_,
      early_stopping_rounds = NA_integer_,
      current_iter = 1:dplyr::n(),
      total_iters  = dplyr::n()
    )
  
  message(paste("Total random combinations:", nrow(grid_rs)))
  
  results_rs <- purrr::pmap_dfr(
    grid_rs,
    ADABoost_gridsearch,
    folds_custom = cv_folds_list
  ) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    dplyr::mutate(Method = "Random Search", Iteration = dplyr::row_number())
  
  message("--- Top 5 Random Models (AdaBoost) ---")
  print(head(results_rs, 5))
  
  ##==============================##
  ## Bayesian Optimization for Hyperparameter Tuning.
  ##==============================##
  n_init_points <- 10
  n_iter_bayes  <- 20
  total_bayes_runs <- n_init_points + n_iter_bayes
  current_bayes_iter <- 0
  
  bounds_bayes <- list(
    mfinal    = c(25L, 300L),
    maxdepth  = c(1L, 4L),
    minsplit  = c(5L, 100L),
    minbucket = c(2L, 50L)
  )
  
  message("Starting Bayesian Optimization (AdaBoost)...")
  
  bayes_out <- rBayesianOptimization::BayesianOptimization(
    FUN = ADABoost_bayesoptim,
    bounds = bounds_bayes,
    init_points = n_init_points,
    n_iter = n_iter_bayes,
    acq = "ei",
    eps = 0.01,
    verbose = TRUE
  )
  
  results_bayes <- bayes_out$History %>%
    dplyr::rename(AUC = Value) %>%
    dplyr::mutate(
      mfinal    = as.integer(round(mfinal)),
      maxdepth  = as.integer(round(maxdepth)),
      minsplit  = as.integer(round(minsplit)),
      minbucket = as.integer(round(minbucket)),
      nrounds = NA_integer_,
      early_stopping_rounds = NA_integer_,
      Method = "Bayesian Opt",
      Iteration = dplyr::row_number()
    ) %>%
    dplyr::arrange(dplyr::desc(AUC)) %>%
    tibble::as_tibble()
  
  message("--- Top 5 Bayesian Models (AdaBoost) ---")
  print(head(results_bayes, 5))
  
  ##==============================##
  ## Compare tuning methods
  ##==============================##
  common_cols <- c("Method","Iteration","mfinal","maxdepth","minsplit","minbucket","AUC")
  
  all_search_results <- dplyr::bind_rows(
    results_ds %>% dplyr::select(dplyr::all_of(common_cols)),
    results_rs %>% dplyr::select(dplyr::all_of(common_cols)),
    results_bayes %>% dplyr::select(dplyr::all_of(common_cols))
  ) %>%
    dplyr::arrange(dplyr::desc(AUC))
  
  message("--- Best overall (across all tuning methods) ---")
  print(head(all_search_results, 1))
  
  ##==============================##
  ## Train ONE final model with the best CV AUC parameters
  ##==============================##
  best_overall <- all_search_results[1, ]
  
  final_params <- list(
    mfinal    = as.integer(best_overall$mfinal),
    maxdepth  = as.integer(best_overall$maxdepth),
    minsplit  = as.integer(best_overall$minsplit),
    minbucket = as.integer(best_overall$minbucket)
  )
  
  message("--- Final chosen method & params (AdaBoost) ---")
  print(best_overall)
  print(final_params)
  
  ctrl_final <- rpart::rpart.control(
    maxdepth  = final_params$maxdepth,
    minsplit  = final_params$minsplit,
    minbucket = final_params$minbucket,
    cp = 0
  )
  
  model_final <- adabag::boosting(
    y ~ .,
    data    = Train_AB,
    mfinal  = final_params$mfinal,
    boos    = TRUE,
    control = ctrl_final,
    weights = w_full
  )
  
  ##==============================##
  ## Helper to safely extract positive-class probs
  ##==============================##
  get_pos_prob <- function(pred_obj, pos_class) {
    if (!is.list(pred_obj) || is.null(pred_obj$prob)) return(NULL)
    prob_mat <- pred_obj$prob
    if (pos_class %in% colnames(prob_mat)) return(prob_mat[, pos_class])
    # fallback: try matching by level order
    return(prob_mat[, which(colnames(prob_mat) == pos_class)[1]])
  }
  
  ##==============================##
  ## Train AUC (factor-based, consistent with CV)
  ##==============================##
  pred_train <- predict(model_final, newdata = Train_AB)
  prob_train <- get_pos_prob(pred_train, positive_class)
  if (is.null(prob_train)) stop("Could not extract train probabilities from adabag::predict().")
  
  roc_train <- pROC::roc(
    response  = Train_AB[[target]],
    predictor = prob_train,
    levels    = levels(Train_AB[[target]]),
    direction = "<",
    quiet     = TRUE
  )
  auc_train <- as.numeric(pROC::auc(roc_train))
  message(paste("Final Train AUC (AdaBoost):", round(auc_train, 5)))
  
  ##==============================##
  ## Test AUC (factor-based)
  ##==============================##
  pred_test <- predict(model_final, newdata = Test_AB)
  prob_test <- get_pos_prob(pred_test, positive_class)
  if (is.null(prob_test)) stop("Could not extract test probabilities from adabag::predict().")
  
  roc_test <- pROC::roc(
    response  = Test_AB[[target]],
    predictor = prob_test,
    levels    = levels(Train_AB[[target]]),
    direction = "<",
    quiet     = TRUE
  )
  auc_test <- as.numeric(pROC::auc(roc_test))
  message(paste("Final Test AUC (AdaBoost):", round(auc_test, 5)))

  ## Method summary table
  
  best_by_method <- all_search_results %>%
    dplyr::group_by(Method) %>%
    dplyr::slice_max(order_by = AUC, n = 1, with_ties = FALSE) %>%
    dplyr::ungroup()
  
  AdaBoost_method_performance <- best_by_method %>%
    dplyr::transmute(Method, CV_AUC = AUC)
  
  message("--- AdaBoost best CV AUC by tuning method ---")
  print(AdaBoost_method_performance)
  
}, error = function(e) {
  message("AdaBoost block failed with error:")
  message(e$message)
})

#==== 05C - XGBoost ===========================================================#

tryCatch({

  time_XGBoost <- system.time({
    
##==============================##
## General Parameters.
##==============================##

##==============================##
## Data preparation.
##==============================##
sparse_formula <- as.formula("y ~ . - 1")
length(unlist(Data_Train_CV_List))

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

time_XGBoost_ds <- system.time({
  
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
                       folds_custom = Data_Train_CV_List) %>%
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
  folds = Data_Train_CV_List,
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
XGBoost_ds_Train_AUC <- pROC::auc(XGBoost_ds_Train_ROC)
print(paste("Final Train AUC:", round(XGBoost_ds_Train_AUC, 5)))

}) ## time counter.

##==============================##
## Random Grid Search for Hyperparameter Tuning.
##==============================##

time_XGBoost_rs <- system.time({
  
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
                         folds_custom = Data_Train_CV_List) %>%
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
  folds = Data_Train_CV_List,
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
XGBoost_rs_Train_AUC <- pROC::auc(XGBoost_rs_Train_ROC)
print(paste("Final Train AUC:", round(XGBoost_rs_Train_AUC, 5)))

}) ## time counter.

##==============================##
## Bayesian Optimization for Hyperparameter Tuning.
##==============================##

time_XGBoost_bo <- system.time({
  
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
  acq = "ucb",
  kappa = 2.576,                 
  eps = 0.0,                     
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
  folds = Data_Train_CV_List,
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
XGBoost_bayes_Train_AUC <- pROC::auc(XGBoost_bayes_Train_ROC)
print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))

}) ## time counter.

##==============================##
## Performance of the model with the highest training AUC in the test set.
##==============================##

## Check for the right tree size.
cv_results_train <- xgb.cv(
  params = final_params_bayes,
  data = dtrain,
  folds = Data_Train_CV_List,
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
XGBoost_test_AUC <- pROC::auc(XGBoost_test_ROC)
print(paste("Final Test Set AUC:", round(XGBoost_test_AUC, 5)))

## 1-SE:
XGBoost_test_probs_1SE <- predict(XGBoost_finalmodel_1SE, dtest)
XGBoost_test_ROC_1SE   <- roc(test_y, XGBoost_test_probs_1SE, quiet = TRUE)
XGBoost_test_AUC_1SE   <- pROC::auc(XGBoost_test_ROC_1SE)
print(paste("Final Test AUC (1-SE Rule):", round(XGBoost_test_AUC_1SE, 5)))

## Brier Score of the full model.
actuals_num <- as.numeric(as.character(test_y))
XGBoost_BrierScore <- BrierScore(XGBoost_test_probs, actuals_num)
XGBoost_BrierScore_1SE <- BrierScore(XGBoost_test_probs_1SE, actuals_num)

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

## CV-AUC:
print(head(results_ds, 5))
print(head(results_rs, 5))
print(head(results_bayes, 5))

# Extract the best Cross-Validation AUC from each results table
best_cv_ds    <- max(results_ds$AUC)     
best_cv_rs    <- max(results_rs$AUC)    
best_cv_bayes <- max(results_bayes$AUC)  

XGBoost_CV_performance <- data.frame(
  Method = c("Grid Search", "Random Search", "Bayesian Opt"),
  CV_AUC = c(best_cv_ds, best_cv_rs, best_cv_bayes)
)

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

Plot_Train_AUC <- Plot_CV_AUC <- ggplot(XGBoost_CV_performance, aes(x = reorder(Method, CV_AUC), y = CV_AUC, 
                                                                    fill = Method)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(CV_AUC * 100, 1), "%")), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = colors) +
  labs(
    title = "",
    subtitle = "",
    x = "Hyperparameter Tuning Method",
    y = "AUC-Score (CV)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"), 
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
       subtitle = "",
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

Path <- file.path(Charts_XGBoost_Directory, "02_LearningCurve_Bayes_Training.png")
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
  Test_AUC = c(best_cv_bayes, XGBoost_test_AUC_1SE, XGBoost_test_AUC)
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
calib_data_xgb <- data.frame(
  actual = test_y,
  prob   = XGBoost_test_probs, # XGB Prediction
  bin    = ntile(probs_1se, 10) # GLM Buckets!
) %>%
  group_by(bin) %>%
  summarise(
    mean_prob     = mean(prob),
    observed_rate = mean(actual),
    n             = n()
  )

calib_plot_data_xgb <- calib_data_xgb %>%
  select(bin, mean_prob, observed_rate) %>%
  rename(Predicted = mean_prob, Observed = observed_rate) %>%
  pivot_longer(cols = c("Predicted", "Observed"), 
               names_to = "Type", values_to = "Rate") %>%
  mutate(Type = factor(Type, levels = c("Predicted", "Observed")))

plot_calib_bars <- ggplot(calib_plot_data_xgb, aes(x = factor(bin), y = Rate, fill = Type)) +
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

})
  
### Time-Counter.
  #### Compare the time.
  
timing_comparison_XGBoost <- tibble(
    Method = c("Grid Search", "Random Search", "Bayesian Optimization", "XGBoost Code"),
    Time_Seconds = c(
      time_XGBoost_ds["elapsed"], 
      time_XGBoost_rs["elapsed"], 
      time_XGBoost_bo["elapsed"], 
      time_XGBoost["elapsed"] 
    )
  ) %>%
    mutate(
      Time_Minutes = Time_Seconds / 60,
      Iterations = c(
        nrow(grid_ds),  
        nrow(grid_rs),  
        nrow(results_bayes), 
        NA
      )
    ) %>%
    arrange(Time_Seconds)

}, error = function(e) message(e))

#==============================================================================#
#==== 06 - Neural Networks ====================================================#
#==============================================================================#


#==============================================================================#
#==== 07 - Overview of the data ===============================================#
#==============================================================================#

tryCatch({

# Helper function to format hyperparameter lists into strings
# format_params <- function(params_list) {
#   if (is.null(params_list)) return("NA")
#   simple_params <- params_list[sapply(params_list, function(x) length(x) == 1)]
#   paste(names(simple_params), simple_params, sep = "=", collapse = "; ")
# }

##==============================##
## Extract GLM-Results.
##==============================##

# We use the 1-SE rule as the "Conservative" metric for GLM
# glm_row <- tibble(
#   Model = "Regularized GLM",
#   Tuning_Method = champion_row$Method,
#   # Performance
#   Train_AUC = champion_row$Train_AUC,
#   Test_AUC_Best = auc_champion,        
#   Test_AUC_Conservative = auc_1se,     
#   # Complexity & Efficiency
#   N_Features = n_vars_1se,            
#   # Time_Minutes = as.numeric(time_GLM["elapsed"]) / 60,
#   # Config
#   Hyperparameters = paste0("Alpha=", round(best_alpha, 2))
# )
# write.csv(glm_row, file = file.path(Data_Directory_write, "GLM_Details.csv"), row.names = FALSE)

##==============================##
## Extract RF-Results.
##==============================##

# rf_row <- tibble(
#   Model = "Random Forest (Ranger)",
#   Tuning_Method = "MBO (mlr3)",
#   # Performance
#   Train_AUC = hpo_results$best_cv_auc, 
#   Test_AUC_Best = eval_results$auc,    
#   Test_AUC_Conservative = eval_results$auc, 
#   # Complexity & Efficiency
#   N_Features = n_features,             
#   # Time_Minutes = as.numeric(time_RF["elapsed"]) / 60,
#   # Config
#   Hyperparameters = format_params(hpo_results$best_params)
# )
# write.csv(rf_row, file = file.path(Data_Directory_write, "RF_Details.csv"), row.names = FALSE)

##==============================##
## Extract XGBoost-Results.
##==============================##

# xgb_row <- tibble(
#   Model = "XGBoost",
#   Tuning_Method = "Bayesian Opt",
#   # Performance
#   Train_AUC = results_bayes[1,]$AUC,  
#   Test_AUC_Best = XGBoost_test_AUC,       
#   Test_AUC_Conservative = XGBoost_test_AUC_1SE, 
#   # Complexity & Efficiency
#   # We calculate features from the matrix columns to be safe
#   N_Features = ncol(train_matrix), 
#   # We use the total time wrapper 'time_XGBoost' to be consistent with GLM/RF
#   Time_Minutes = as.numeric(time_XGBoost["elapsed"]) / 60,
#   # Config
#   Hyperparameters = paste(format_params(final_params_bayes), 
#                           "Rounds_1SE=", optimal_rounds_1se, sep="; ")
# )
# 
# write.csv(xgb_row, file = file.path(Data_Directory_write, "XGBoost_Details.csv"), row.names = FALSE)


##==============================##
## Overall test-results ("Leaderboard").
##==============================##

# Leaderboard <- bind_rows(glm_row, rf_row, xgb_row) %>%
#   mutate(
#     # The "Overfitting Gap": Positive means overfitting. 
#     # Small gap (e.g., < 0.02) is excellent. Large gap (> 0.05) is dangerous.
#     Overfitting_Gap = Train_AUC - Test_AUC_Best,
#     # "Safety Cost": How much AUC do we lose by being conservative?
#     Safety_Cost = Test_AUC_Best - Test_AUC_Conservative
#   ) %>%
#   select(Model, Test_AUC_Conservative, Test_AUC_Best, Overfitting_Gap, 
#          Train_AUC, Time_Minutes, Hyperparameters) %>%
#   arrange(desc(Test_AUC_Conservative))
# 
# # Display the Final Result
# print("--- MODEL CHAMPIONSHIP LEADERBOARD ---")
# print(Leaderboard)
# 
# write.csv(Leaderboard, file = file.path(Data_Directory_write, "Model_Data_TestSet.csv"), row.names = FALSE)

}, error = function(e) message(e))

#==============================================================================#
#==== 08 - Model selection ====================================================#
#==============================================================================#

## Model performance in the test set.

tryCatch({

##==============================##
## Data.
##==============================##

final_test_AUC <- tibble(
  Model = c("Regularized GLM (1-SE)", "Random Forest", "XGBoost"),
  Test_AUC = c(auc_1se, RF_method_performance$AUC_Score[2], XGBoost_test_AUC),
  Brier_Score = c(GLM_BrierScore_1SE, RF_BrierScore, XGBoost_BrierScore)
  )

##==============================##
## Visualisations.
##==============================##

## Visualisation: Test AUC.

comparison_colors <- c(
  "Regularized GLM (1-SE)"     =  blue, 
  "Random Forest" =  orange,
  "XGBoost" = red
)

Plot_Overall_Test_AUC <- ggplot(final_test_AUC, 
                                aes(x = reorder(Model, Test_AUC), 
                                    y = Test_AUC, 
                                    fill = Model)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(round(Test_AUC * 100, 2), "%")), 
            hjust = -0.1, size = 5, fontface = "bold") +
  scale_fill_manual(values = comparison_colors) +
  scale_y_continuous(labels = scales::percent, 
                     limits = c(0, 1), 
                     expand = expansion(mult = c(0, 0.15))) + 
  coord_flip() + 
  labs(
    title = "",
    subtitle = "",
    x = NULL,
    y = "AUC-Score (Test Set)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = "grey30"),
    axis.text.y = element_text(size = 12, face = "bold", color = "black"),
    axis.title.x = element_text(size = 13, face = "bold", 
                                margin = ggplot2::margin(t = 10)),
    
    panel.grid.major.y = element_blank(), 
    panel.grid.major.x = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

Path <- file.path(Charts_TestSet_Directory, "04_AUC_Test_Overall.png")
ggsave(
  filename = Path,
  plot = Plot_Overall_Test_AUC,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## Visualisation: 

}, error = function(e) message(e))
  
#==============================================================================#
#==============================================================================#
#==============================================================================#

