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
              "future.apply", "parallel"
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

#==== 04A - GLMs ==============================================================#

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
results_ds_glm <- pmap_dfr(grid_ds_glm, GLM_gridsearch) %>%
  arrange(desc(CV_AUC))

print("--- Top 5 Discrete GLM Models ---")
print(head(results_ds_glm, 5))

##==============================##
## Random grid search (hyperparameter tuning of alpha and lambda).
##==============================##

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

##==============================##
## Bayesian Optimization (hyperparameter tuning of alpha and lambda).
##==============================##

## Prepare the grid.
bounds_glm <- list(
  alpha = c(0, 1)
)

# 3. Run Optimization
n_init_glm <- 5
n_iter_glm <- 15
current_bayes_iter <- 0

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
  foldid = Data_Train_CV_Vector,   
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
colnames(probs_1se) <- "prob"

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

## Regularized GLM performance in the Test-set.

glm_performance_test <- tibble(
  Method = c("Regularized GLM", "Regularized GLM (1-SE)"),
  Test_AUC = c(auc_champion, auc_1se)
)
print("--- GLM Test-set AUC ---")
print(glm_performance_test)

##==============================##
## Compare the hyperparameter tuning methods. Visualisations.
##==============================##

## Visualisation: Method AUC comparison.
colors <- c(
  "Grid Search"   = blue,
  "Random Search" = orange,
  "Bayesian Opt" = red
)

Plot_Train_AUC <- ggplot(glm_method_performance, aes(x = reorder(Method, Train_AUC), y = Train_AUC, 
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

##==============================##
## Visualisations of the test-set performance.
##==============================##

## Visualisation: AUC in the test set.

colors <- c(
  "Regularized GLM"   = blue,
  "Regularized GLM (1-SE)" = red
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
calib_data <- data.frame(
  actual = test_y,
  prob = probs_1se
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



}, error = function(e) message(e))

#==============================================================================#
#==== 05 - Decision Trees =====================================================#
#==============================================================================#

#==== 05A - Random Forest =====================================================#

tryCatch({
  
##==============================##
## General Parameters.
##==============================##
  
temp_name <- "mtry_rounds_NOPIT"
  
HPO_CONFIG <- list(
    learner_name = "ranger",  # Changed from xgboost to ranger
    n_evals = 500L,
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
  
##==============================##
## Visualisation.
##==============================##    
  
summary_table <- data.table(
  Metric = c("Total Evaluations", "Warmup Points", "BO Evaluations",
             "Optimization Time (s)", "Best CV AUC", "Test AUC", "Test Accuracy"),
  Value = c(hpo_results$n_evals,
            min(hpo_results$n_initial_design, hpo_results$n_evals),
            max(0, hpo_results$n_evals - hpo_results$n_initial_design),
            round(hpo_results$time, 2),
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

p1 <- plot_optimization_history(hpo_results)
print(p1)
Path <- file.path(Charts_RF_Directory, paste0("01_RF_mbo_optimization_history_", temp_name, ".png"))
ggsave(
  filename = Path,
  plot = p1,
  width = width,
  height = heigth,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

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

}, error = function(e) message(e))

#==== 05B - AdaBoost ==========================================================#


#==== 05C - XGBoost ===========================================================#

## To-Do: Fix overall parameters at the beginning of the code.
## Comparison of hyperparameter tuning methods.
## Visualisations and Outputs.

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
XGBoost_bayes_Train_AUC <- auc(XGBoost_bayes_Train_ROC)
print(paste("Final Bayesian Train AUC:", round(XGBoost_bayes_Train_AUC, 5)))

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

#==============================================================================#
#==== 06 - Neural Networks ====================================================#
#==============================================================================#


#==============================================================================#
#==== 07 - Model selection ====================================================#
#==============================================================================#

## Model performance in the test set.

tryCatch({

##==============================##
## Data.
##==============================##

glm_performance_test          ## Regularized GLM - Test AUC.
XGBoost_method_performance    ## XGBoost - Test AUC.

final_test_AUC <- tibble(
  Model = c("XGBoost (Bayesian)", "Regularized GLM (1-SE)"),
  Test_AUC = c(XGBoost_test_AUC, auc_1se)
  )

##==============================##
## Visualisations.
##==============================##

## Visualisation: Test AUC.

comparison_colors <- c(
  "XGBoost (Bayesian)"     =  blue, 
  "Regularized GLM (1-SE)" =  orange
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

print(Plot_Overall_Test_AUC)

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

