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

## Data Sampling.
set.seed(123)


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

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

#==== 02b - Exploratory Data Analysis =========================================#

glimpse(Data)
skim(Data)

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
Data %>%
  filter(y == 1) %>%
  summarise(n = n()) %>%
  mutate(total_rows = nrow(Data),
         proportion = n / total_rows * 100
         )



barplot(y_prop_table,
        main = "Class Imbalance", 
        ylab = "Proportion", 
        col = blue)
## ======================= ##
## Check for data variability and skewed data.
## ======================= ##
hist(Data$f8)
       
## ======================= ##     
hist(Data$f1,
     col = blue,
     xlab = "Value",
     ylab = "Count",
     main = "Histogram of f1",
     xlim = range(Data$f1, na.rm = TRUE),
     ylim = c(0, max(hist(Data$f1, plot = FALSE)$counts)))

hist(Data$f2,
    col = blue,
    xlab = "Value",
    ylab = "Count",
    main = "Histogram of f2",
    xlim = range(Data$f2, na.rm = TRUE),
    ylim = c(0, max(hist(Data$f2, plot = FALSE)$counts)))

hist(Data$f3,
     col = blue,
     xlab = "Value",
     ylab = "Count",
     main = "Histogram of f3",
     xlim = range(Data$f3, na.rm = TRUE),
     ylim = c(0, max(hist(Data$f3, plot = FALSE)$counts)))

hist(Data$f8,
     col = blue,
     xlab = "Value",
     ylab = "Count",
     main = "Histogram of f8",
     xlim = range(Data$f8, na.rm = TRUE),
     ylim = c(0, max(hist(Data$f8, plot = FALSE)$counts)))

## ===============================================##
## Plot as one function)
## ==============================================##
df6 <- Data %>% dplyr::select(5:10)            
long6 <- tidyr::pivot_longer(df6, cols = everything(),
                             names_to = "Variable", values_to = "Value")

ggplot(long6, aes(x = Value)) +
  geom_histogram(fill = "#004890", color = "white", bins = 30) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(x = "Value", y = "Count", title = "Histograms: f1 to f6")

df11 <- Data %>% dplyr::select(11:15)            
long11 <- tidyr::pivot_longer(df11, cols = everything(),
                              names_to = "Variable", values_to = "Value")

ggplot(long11, aes(x = Value)) +
  geom_histogram(fill = "#004890", color = "white", bins = 30) +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(x = "Value", y = "Count", title = "Histograms: f7 to f11")
       

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

#==== 02c - Informational Value of each feature & other =======================#

## ======================= ##
## Informational value.
## ======================= ##
## Tells us how good a feature seperates between NoDefault (y=0) and Default (y=1).
iv_summary <- iv(Data, y = "y")
print(iv_summary %>% arrange(desc(info_value)))

### Now let's explore f8 and f9.
ggplot(Data, aes(x = as.factor(y), y = f8, fill = as.factor(y))) +
  geom_boxplot() +
  labs(title = "Distribution of f8 by Default Status",
       x = "Default Status (y)",
       y = "f1 Value",
       fill = "Default Status") +
  theme_minimal()

## Density plot for f8.
x_limits <- quantile(Data$f8, probs = c(0.005, 0.975), na.rm = TRUE)
ggplot(Data, aes(x = f8, fill = as.factor(y))) +
  geom_density(alpha = 0.6) +
    coord_cartesian(xlim = x_limits) +
  
  scale_fill_manual(values = c("0" = "darkgreen", "1" = red),
                    name = "Default Status",
                    labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of f8",
       subtitle = "Excluding extreme outliers to reveal distribution shape",
       x = "f8 Value") +
  theme_minimal()

## Density plot for f11
x_limits <- quantile(Data$f11, probs = c(0.000001, 0.99), na.rm = TRUE)
ggplot(Data, aes(x = f11, fill = as.factor(y))) +
  geom_density(alpha = 0.6) +
  coord_cartesian(xlim = x_limits) +
  
  scale_fill_manual(values = c("0" = "darkgreen", "1" = red),
                    name = "Default Status",
                    labels = c("Non-Default", "Default")) +
  labs(title = "Distribution of f11",
       subtitle = "Excluding extreme outliers to reveal distribution shape",
       x = "f11 Value") +
  theme_minimal()

## Correlation between the top features.
cor(Data$f8, Data$f9)
cor(Data$f8, Data$f6)
cor(Data$f8, Data$f11)

## ======================= ##
## Check the default rate by sector.
## ======================= ##

## Frequeny of business sectors.
ggplot(Data, aes(x = sector)) +
  geom_bar(fill = blue) +
  ggtitle("Frequency of Business Sectors") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## Defaults by business sector.
Data %>%
  group_by(sector) %>%
  summarise(
    count = n(),
    default_rate = mean(y, na.rm = TRUE) # Calculate the mean of y (0s and 1s)
  ) %>%
  ggplot(aes(x = reorder(sector, -default_rate), y = default_rate)) +
  geom_bar(stat = "identity", fill = blue) +
  labs(title = "Default Rate by Sector",
       x = "Sector",
       y = "Default Rate") +
  theme_minimal()

#==== 02d - Multicollinearity =================================================#

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

## Old approach

tryCatch({
  
## Employing stratified sampling since we have a very imbalanced dataset (low occurance of defaults).
## We ensure that the original percentage of the dependent variable is preserved in each split.

## ======================= ##
## Stratified Sampling.
## ======================= ##

First_Split <- initial_split(Data, prop = 0.50, strata = y) ## Split into train and validation/test bucket.
Train <- training(First_Split)
Temp <- testing(First_Split)

Second_Split <- initial_split(Temp, prop = 0.50, strata = y) ## Now, split the second bucket in validation and testing.
Validation <- training(Second_Split)
Test <- testing(Second_Split)

}, silent = TRUE)

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

Validation <- Validation %>%
  mutate(
    across(all_of(cols_to_impute), 
           ~ifelse(is.na(.), 1, 0), 
           .names = "{.col}_is_missing"),
    across(all_of(cols_to_impute), 
           ~coalesce(., median_values[[cur_column()]]))
  )

## ======================= ##
## Test data.
## ======================= ##
median_values <- Test %>%
  summarise(across(all_of(cols_to_impute), ~median(., na.rm = TRUE))) %>%
  as.list()

Test <- Test %>%
  mutate(
    across(all_of(cols_to_impute), 
           ~ifelse(is.na(.), 1, 0), 
           .names = "{.col}_is_missing"),
    across(all_of(cols_to_impute), 
           ~coalesce(., median_values[[cur_column()]]))
  )

## ======================= ##
## Findings.
## ======================= ##

## Multicollinearity: Remove f9 and keep f8. They are extremely similar.
## IV: Remove non-predictive variables. Groupmember likely just adds random noise.
## WoE-transformed logistic regression as the "base" case.
## Handling of NAs within the "r" features.

## For the future: Stability of variables over time (using the population stability index).

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
