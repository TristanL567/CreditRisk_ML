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

packages <- c("here", "corrplot", "dplyr", "tidyr",
              "reshape2", "ggplot2",
              "rsample", "DataExplorer",  ## Necessary for stratified sampling.
              "pROC",                     ## Area under the Curve (AuC) measure.
              "caret",                    ## GLM.
              "glmnet",                    ## Regularized regression
              "skimr",
              "scorecard"
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

## Skewness.
skew <- function(x) {
  x <- na.omit(x)
  n <- length(x)
  mean_x <- mean(x)
  sd_x <- sd(x)
  skewness_value <- (n / ((n - 1) * (n - 2))) * sum(((x - mean_x) / sd_x)^3)
  return(skewness_value)
}

#==== 1C - Parameters =========================================================#

## Directories.
Data_Path <- "C:/Users/TristanLeiter/Documents/Privat/ILAB/Data/WS2025" ## Needs to be set manually.
Data_Directory <- file.path(Data_Path, "data.rda")
Charts_Directory <- file.path(Path, "03_Charts")

## Plotting.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"

height <- 3750
width <- 1833

## Data Sampling.
set.seed(123)

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

### This should only contain data manipulation that is strictly necessary
### for the further code, so far:
# - Load the dataset.
# - Throw out the ratios.
# - Data sampling

# - To-Do: feature engineering
# -        outliers, etc.

#==== 02a - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

## ======================= ##
## Drop all ratios (perfectly linear dependency of features otherwise.)
## ======================= ##

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]


#==== 02b - Splitting the dataset =============================================#
## New approach with 80% training and 20% test set size.

tryCatch({
  
## Employing stratified sampling since we have a very imbalanced dataset (low occurance of defaults).
## We ensure that the original percentage of the dependent variable is preserved in each split.
  
## ======================= ##
## Stratified Sampling.
## ======================= ##
  
First_Split <- initial_split(Data, prop = 0.8, strata = y) ## Split into train and test.
Train <- training(First_Split)
Test <- testing(First_Split)

## Check their frequencies.
Train %>%
  filter(y == 1) %>%
  summarise(n = n()) %>%
  mutate(total_rows = nrow(Train),
         proportion = n / total_rows * 100
  )

Test %>%
  filter(y == 1) %>%
  summarise(n = n()) %>%
  mutate(total_rows = nrow(Test),
         proportion = n / total_rows * 100
  )

}, silent = TRUE)



#==============================================================================#
#==== 03 - Setting up the Loss-function =======================================#
#==============================================================================#




#==============================================================================#
#==== 04 - Generalized Linear Models ==========================================#
#==============================================================================#

#==== 04a - Binary regression model (log) =====================================#

tryCatch({
  
## ======================= ##
## Parameters.
## ======================= ##
Train$y <- as.numeric(Train$y)

## ======================= ##
## Model training.
## ======================= ##
model_logit <- glm(y ~ ., 
                   data = Train, 
                   family = binomial(link = "logit"))
summary(model_logit)

## ======================= ##
## Optimize for AuC.
## ======================= ##
## Here we could consider plotting the true positive rate vs the false positive rate.
## Easier to mentally process than sensitivity and specificity.

pred_prob_val <- predict(model_logit, newdata = Validation, type = "response")
roc_obj <- roc(Validation$y, pred_prob_val)

auc_val <- auc(roc_obj)
print(paste("Validation AUC:", round(auc_val, 4)))

plot(roc_obj, main = "ROC Curve - Logistic Regression")

## ======================= ##
## Recall.
## ======================= ##
best_coords_topleft <- coords(roc_obj, 
                              x = "best", 
                              best.method = "closest.topleft", 
                              ret = c("threshold", "sensitivity", "specificity", "precision"))

print("--- Optimal Threshold (Closest to Top-Left) ---")
print(best_coords_topleft)

best_coords_youden <- coords(roc_obj, 
                             x = "best", 
                             best.method = "youden", 
                             ret = c("threshold", "sensitivity", "specificity", "precision"))

print("--- Optimal Threshold (Youden Index) ---")
print(best_coords_youden)

## ======================= ##
## Evaluation on the test set.
## ======================= ##
pred_prob_test <- predict(model_logit, newdata = Test, type = "response")
pred_class_test <- as.factor(ifelse(pred_prob_test > best_coords_youden$threshold, "1", "0"))
Test$y <- as.factor(Test$y)

final_report <- confusionMatrix(data = pred_class_test, 
                                reference = Test$y,
                                positive = "1")

print("Final Model Report on Test Set:")
print(final_report)

}, silent = TRUE)

#==== 04b - Regularized regression model (log) ================================#

tryCatch({

## ======================= ##
## Parameters.
## ======================= ##
x_train <- model.matrix(y ~ . -1, data = Train)
y_train <- Train$y

x_val <- model.matrix(y ~ . -1, data = Validation)
y_val <- Validation$y

x_test <- model.matrix(y ~ . -1, data = Test)
y_test <- Test$y

## ======================= ##
## Model training.
## ======================= ##
cv_model_logit <- cv.glmnet(x_train, 
                            y_train, 
                            family = "binomial", 
                            alpha = 1)

## ======================= ##
## Optimize for AuC.
## ======================= ##
pred_prob_val <- predict(cv_model_logit, 
                         newx = x_val, 
                         s = "lambda.min", 
                         type = "response")

roc_obj <- roc(y_val, pred_prob_val)
auc_val <- auc(roc_obj)
print(paste("Validation AUC:", round(auc_val, 4)))

plot(roc_obj, main = "ROC Curve - glmnet (Lasso)")

## ======================= ##
## Recall (threshhold for 95% recall).
## ======================= ##
best_coords_topleft <- coords(roc_obj, 
                              x = "best", 
                              best.method = "closest.topleft", 
                              ret = c("threshold", "sensitivity", "specificity", "precision"))

print("--- Optimal Threshold (Closest to Top-Left) ---")
print(best_coords_topleft)

best_coords_youden <- coords(roc_obj, 
                             x = "best", 
                             best.method = "youden", 
                             ret = c("threshold", "sensitivity", "specificity", "precision"))

print("--- Optimal Threshold (Youden Index) ---")
print(best_coords_youden)

## ======================= ##
## Evaluation on the test set.
## ======================= ##
pred_prob_test <- predict(cv_model_logit, 
                          newx = x_test, 
                          s = "lambda.min", 
                          type = "response")

pred_class_test <- as.factor(ifelse(pred_prob_test > best_coords_youden$threshold, "1", "0"))
y_test_factor <- as.factor(y_test)

final_report <- confusionMatrix(data = pred_class_test, 
                                reference = y_test_factor,
                                positive = "1")

print("Final Model Report on Test Set:")
print(final_report)

}, silent = TRUE)

#==============================================================================#
#==============================================================================#
#==============================================================================#
