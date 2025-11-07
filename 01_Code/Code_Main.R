#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

#==============================================================================#
#==== 1 - Working Directory & Libraries =======================================#
#==============================================================================#

silent=F
.libPaths()

Path <- file.path(here::here("")) ## You need to install the package first incase you do not have it.

## Additional:

# Enable_Catboost <- FALSE

#==== 1A - Libraries ==========================================================#

## Needs to enable checking for install & if not then autoinstall.

packages <- c("here", "corrplot", "dplyr", "tidyr",
              "reshape2", "ggplot2",
              "rsample", "DataExplorer",  ## Necessary for stratified sampling.
              "pROC",                     ## Area under the Curve (AuC) measure.
              "caret"                     ## GLM.
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

## Optional: Catboost package.
# if(Enable_Catboost){
#   remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-windows-x86_64-1.2.8.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
#   library(catboost)
# }

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

#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02a - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

#==== 02b - Exploratory Data Analysis =========================================#

glimpse(Data)

## ======================= ##
## First let us check if we have any missing data.
## ======================= ##
colSums(is.na(Data)) ## We fine some NAs, mostly in the financial ratio's.
sapply(Data, function(x) sum(is.infinite(x)))
sapply(Data, function(x) sum(is.nan(x)))

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(!is.finite(.), NA, .)))

## Apply it in the Train, Validation and Test sets seperately (see below).

## ======================= ##
## Check the scale of the features.
## ======================= ##
summary(Data)

## ======================= ##
## Now check our independent variable: loan default.
## ======================= ##
table(Data$y) ## 0.865% of all loans are defaulting.

## ======================= ##
## Check for data variability and skewed data.
## ======================= ##
hist(Data$f8)

## Check the variation of each feature.
Variance <- apply(as.matrix(Features), MARGIN = 2, FUN = var)
Skewness <- apply(as.matrix(Features), MARGIN = 2, FUN = skew)

## We should standardize as the variance differs considerably between our features.

## ======================= ##
## Standardize the features.
## ======================= ##
# scaled_predictors <- data.frame(scale(Features))
# Data_std <- cbind(scaled_predictors, data$quality)
# colnames(Data_std) <- colnames(Data)

## ======================= ##
## Classify categorical data as factors.
## ======================= ##

## Ordinal scale.
unique(Data$size)
Data$size <- factor(Data$size, 
                    levels = c("Tiny", "Small"),
                    ordered = TRUE)

## Nominal scale.
unique(Data$sector)
Data$sector <- factor(Data$sector)

#==== 02c - Multicollinearity =================================================#

## Parameters.
Path <- file.path(Charts_Directory, "01_Correlation_Plot.png")

## Main Code.
cor_matrix <- cor(Features[,])
cor_matrix_upper_na <- cor_matrix
cor_matrix_upper_na[upper.tri(cor_matrix_upper_na)] <- NA
melted_cor_matrix <- melt(cor_matrix_upper_na, na.rm = TRUE)

## Plot.
plot <- ggplot(data = melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") + 
  scale_fill_gradient2(low = blue, high = orange, mid = "white", 
                       midpoint = 0, limit = c(-1, 1), 
                       name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 1.5) +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title = element_blank()) + 
  coord_fixed() 

ggsave(
  filename = Path,
  plot = plot,
  width = height,
  height = width,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

#==== 02d - Splitting the dataset =============================================#
## Employing stratified sampling since we have a very imbalanced dataset (low occurance of defaults).
## We ensure that the original percentage of the dependent variable is preserved in each split.

## ======================= ##
## Stratified Sampling.
## ======================= ##

set.seed(123)

First_Split <- initial_split(Data, prop = 0.50, strata = y) ## Split into train and validation/test bucket.
Train <- training(First_Split)
Temp <- testing(First_Split)

Second_Split <- initial_split(Temp, prop = 0.50, strata = y) ## Now, split the second bucket in validation and testing.
Validation <- training(Second_Split)
Test <- testing(Second_Split)

## ======================= ##
## Check the propensity tables.
## ======================= ##
prop.table(table(Data$y))

prop.table(table(Train$y))
prop.table(table(Validation$y))
prop.table(table(Test$y))

#==== 02e - Impude the NA values ==============================================#
## To prevent data leakage, we need to process the different sets seperately.
## The idea is that the validation and test sets represent new, unseen data
## and should thus not be influenced by the training set.

## ======================= ##
## Parameters.
## ======================= ##
cols_to_impute <- c("r5", "r12", "r15", "r16", "r17", "r18")

## ======================= ##
## Training data.
## ======================= ##
median_values <- Train %>%
  summarise(across(all_of(cols_to_impute), ~median(., na.rm = TRUE))) %>%
  as.list()

Train <- Train %>%
  mutate(
    across(all_of(cols_to_impute), 
           ~ifelse(is.na(.), 1, 0), 
           .names = "{.col}_is_missing"),
        across(all_of(cols_to_impute), 
           ~coalesce(., median_values[[cur_column()]]))
)

##
colSums(is.na(Train)) ## We fine some NAs, mostly in the financial ratio's.
sapply(Train, function(x) sum(is.infinite(x)))
sapply(Train, function(x) sum(is.nan(x)))

## ======================= ##
## Validation data.
## ======================= ##
median_values <- Validation %>%
  summarise(across(all_of(cols_to_impute), ~median(., na.rm = TRUE))) %>%
  as.list()

# Validation <- Validation %>%
#   mutate(across(all_of(cols_to_impute), 
#                 ~coalesce(., median_values[[cur_column()]])))

## ======================= ##
## Test data.
## ======================= ##
Test <- Test %>%
  mutate(across(all_of(cols_to_impute), 
                ~coalesce(., median_values[[cur_column()]])))

#==============================================================================#
#==== 03 - Setting up the Loss-function =======================================#
#==============================================================================#




#==============================================================================#
#==== 04 - Generalized Linear Models ==========================================#
#==============================================================================#

#==== 04a - Binary regression model (log) =====================================#

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
pred_prob_val <- predict(model_logit, newdata = Validation, type = "response")
roc_obj <- roc(Validation$y, pred_prob_val)

auc_val <- auc(roc_obj)
print(paste("Validation AUC:", round(auc_val, 4)))

plot(roc_obj, main = "ROC Curve - Logistic Regression")

## ======================= ##
## Recall (threshhold for 95% recall).
## ======================= ##
best_threshold_coords <- coords(roc_obj, 
                                x = 0.95,
                                input = "sensitivity", 
                                ret = "threshold")

best_threshold <- best_threshold_coords$threshold

print(paste("Threshold for 95% Recall:", round(best_threshold, 4)))

## ======================= ##
## Evaluation on the test set.
## ======================= ##
pred_prob_test <- predict(model_logit, newdata = test_data, type = "response")
pred_class_test <- as.factor(ifelse(pred_prob_test > best_threshold, "1", "0"))
Test$y <- as.factor(Test$y)

final_report <- confusionMatrix(data = pred_class_test, 
                                reference = Test$y,
                                positive = "1")

print("Final Model Report on Test Set:")
print(final_report)




#==============================================================================#
#==============================================================================#
#==============================================================================#