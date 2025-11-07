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

packages <- c("here")

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


#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02a - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d

#==== 02b - Exploratory Data Analysis =========================================#

##




#==============================================================================#
#==== 03 - Generalized Linear Models ==========================================#
#==============================================================================#





#==============================================================================#
#==============================================================================#
#==============================================================================#