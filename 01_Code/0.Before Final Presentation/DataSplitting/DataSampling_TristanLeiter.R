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



#==== 1C - Parameters =========================================================#

## Directories.
Data_Path <- "C:/Users/TristanLeiter/Documents/Privat/ILAB/Data/WS2025" ## Needs to be set manually.
Data_Directory <- file.path(Data_Path, "data.rda")
Charts_Directory <- file.path(Path, "03_Charts")

## Data Sampling.
set.seed(123)


#==============================================================================#
#==== 02 - Stratified Sampling (multivariate approach) ========================#
#==============================================================================#

## The idea is to fix the default ratio across sectors and overall.
## Also do not split over time-periods, but preserve companies by itself in each set.

## 

#==== 02. - Functions =========================================================#

analyze_distribution <- function(df, dataset_name, 
                                 target_col = "y", 
                                 sector_col = "sector", 
                                 date_col = "refdate",
                                 id_col = "id") {
  
  # 1. Standardize column names for internal processing
  df_metrics <- df %>%
    rename(Target = all_of(target_col),
           Sector = all_of(sector_col),
           ID = all_of(id_col)) %>%
    mutate(
      # Ensure Target is numeric 0/1 for calculation
      TargetNum = as.numeric(as.character(Target)),
      # Extract Year if date_col is present, otherwise look for 'Year'
      Year = if(date_col %in% names(df)) year(get(date_col)) else Year
    )
  
  cat(paste0("\n========================================\n"))
  cat(paste0("   ANALYSIS: ", dataset_name, "\n"))
  cat(paste0("========================================\n"))
  
  # --- 2. Global Rates (Firm vs Observation) ---
  # Firm Level: Did the firm EVER default?
  firm_stats <- df_metrics %>%
    group_by(ID) %>%
    summarize(EverDefault = max(TargetNum), .groups = 'drop')
  
  cat("\n[1] GLOBAL DEFAULT RATES\n")
  cat(sprintf("  • Observation Level (Weighted by duration): %5.2f%%\n", mean(df_metrics$TargetNum) * 100))
  cat(sprintf("  • Firm Level (Unique Entities):             %5.2f%%\n", mean(firm_stats$EverDefault) * 100))
  cat(sprintf("  • Total Firms: %d | Total Obs: %d\n", nrow(firm_stats), nrow(df_metrics)))
  
  # --- 3. Stratification Check (Sector x Default) ---
  # This verifies if your multivariate split worked
  cat("\n[2] FIRM-LEVEL BALANCE BY SECTOR\n")
  sector_summary <- df_metrics %>%
    group_by(ID, Sector) %>%
    summarize(EverDefault = max(TargetNum), .groups = 'drop') %>%
    group_by(Sector) %>%
    summarize(
      Firms = n(),
      Def_Firms = sum(EverDefault),
      Def_Rate = (sum(EverDefault) / n()) * 100
    ) %>%
    mutate(across(where(is.numeric), \(x) round(x, 2)))
  
  print(as.data.frame(sector_summary))
  
  # --- 4. Temporal Stability ---
  cat("\n[3] OBSERVATIONS BY YEAR (%)\n")
  print(round(prop.table(table(df_metrics$Year)) * 100, 2))
}

#==== 02a - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d


# Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
# Data <- Data[, -which(names(Data) %in% Exclude)]

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

Data_split <- Data

#==== 02B - Setup & Data ======================================================#

##==============================##
## Setup.
##==============================##

head(Data_split)

# 1. Define the stratification variables.
strat_vars <- c("sector", "y")

# 2. Create the Firm Profile (One row per ID).
firm_profile <- Data_split %>%
  group_by(id) %>%
  summarise(
    y = max(y), 
    sector = first(sector),
    size = first(size),
    .groups = 'drop'
  ) %>%
  mutate(
    Strat_Key = interaction(select(., all_of(strat_vars)), drop = TRUE)
  )

print("Distribution of Stratification Groups:")
print(table(firm_profile$Strat_Key))

##==============================##
## Data partition.
##==============================##

train_index <- createDataPartition(
  y = firm_profile$Strat_Key, 
  p = 0.70, 
  list = FALSE, 
  times = 1
)

train_ids <- firm_profile$id[train_index]
test_ids  <- firm_profile$id[-train_index]

# 4. Create Final Long-Format Sets
Train <- Data_split %>% filter(id %in% train_ids)
Test <- Data_split %>% filter(id %in% test_ids)

##==============================##
## Validation.
##==============================##

# Check 1: Are IDs mutually exclusive?
intersect_check <- length(intersect(Train$id, Test$id))
cat(paste0("\nOverlapping IDs: ", intersect_check, " (Should be 0)\n"))

# Check 2: Did we preserve the default rate?
cat("\nDefault Rate Comparison:\n")
print(rbind(
  Original = prop.table(table(firm_profile$y)),
  Train    = prop.table(table(firm_profile$y[train_index])),
  Test     = prop.table(table(firm_profile$y[-train_index]))
))

# Check 3: Did we preserve the sector distribution?
cat("\nSector Distribution Comparison:\n")
print(rbind(
  Original = prop.table(table(firm_profile$sector)),
  Train    = prop.table(table(firm_profile$sector[train_index])),
  Test     = prop.table(table(firm_profile$sector[-train_index]))
))

#==== 02C - Analysis of the results ===========================================#

# Analyze the Training Set
analyze_distribution(Train, "TRAIN SET", 
                     target_col = "y", 
                     sector_col = "sector", 
                     date_col = "refdate")

# Analyze the Test Set
analyze_distribution(Test, "TEST SET", 
                     target_col = "y", 
                     sector_col = "sector", 
                     date_col = "refdate")

#==============================================================================#
#==== 03 - SMOTE (+ Extions; see Zhao et al. (2024)) ==========================#
#==============================================================================#

#==== 03. - Functions =========================================================#


#==== 03A - Setup + Data ======================================================#


