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

packages <- c("dplyr", "caret", "lubridate"
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

Functions_Directory <- file.path(Path, "01_Code/Subfunctions")

## Load all code files in the functions directory.
sourceFunctions(Functions_Directory)

## Data Sampling.
set.seed(123)


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


#==============================================================================#
#==============================================================================#
#==============================================================================#

