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

packages <- c("dplyr", "caret", "smotefamily", "unbalanced"
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


#==============================================================================#
#==== 02 - Stratified Sampling ================================================#
#==============================================================================#

## The idea is to fix the default ratio across sectors and overall.
## Also do not split over time-periods, but preserve companies by itself in each set.

#==== 02. - Functions =========================================================#

analyze_distribution <- function(df, dataset_name) {
  cat(paste0("\n--- ", dataset_name, " Analysis ---\n"))
  overall_default <- mean(as.numeric(as.character(df$Target)))
  cat(sprintf("Overall Default Rate: %.2f%%\n", overall_default * 100))
  cat("\nRelative Year Distribution (%):\n")
  print(prop.table(table(df$Year)) * 100)
  
  cat("\nDefault Rates by Sector (%):\n")
  sector_stats <- df %>%
    group_by(Sector) %>%
    summarize(
      Count = n(),
      Defaults = sum(Target == "1"),
      DefaultRate = mean(as.numeric(as.character(Target))) * 100
    )
  print(sector_stats)
}

#==== 02A - Setup & Data ======================================================#

set.seed(123)
n_companies <- 1000
ids <- 1:n_companies

sectors <- sample(c("Energy", "Retail", "Tech", "Finance"), n_companies, replace = TRUE, prob = c(0.1, 0.4, 0.3, 0.2))
defaults <- sample(c(0, 1), n_companies, replace = TRUE, prob = c(0.95, 0.05))
company_info <- data.frame(CompanyID = ids, Sector = sectors, EverDefault = defaults)

## Create the df.
df_panel <- data.frame()
for(year in 2020:2022) {
  temp <- company_info
  temp$Year <- year
  temp$Rev <- runif(n_companies)
  temp$Debt <- runif(n_companies)
  df_panel <- rbind(df_panel, temp)
}

# Ensure Target is a Factor for classification
df_panel$Target <- as.factor(df_panel$EverDefault)
df_panel <- df_panel %>% select(-EverDefault)

print(table(df_panel$Sector, df_panel$Target))

## Group by ID (long-format).
company_summary <- df_panel %>%
  group_by(CompanyID, Sector) %>%
  summarize(Target = max(as.character(Target)), .groups = 'drop')

#==== 02B - Multi-variate stratified sampling =================================#

## Stratification by sector.
company_summary$StratKey <- paste(company_summary$Sector, company_summary$Target, sep = "_")

key_counts <- table(company_summary$StratKey)
valid_keys <- names(key_counts[key_counts > 1])
company_summary <- company_summary %>% filter(StratKey %in% valid_keys)

## Split based on ID.
train_index <- createDataPartition(company_summary$StratKey, p = 0.8, list = FALSE)
train_ids <- company_summary$CompanyID[train_index]
test_ids  <- company_summary$CompanyID[-train_index]

train_data <- df_panel %>% filter(CompanyID %in% train_ids)
test_data  <- df_panel %>% filter(CompanyID %in% test_ids)

# Verification
print("Train Data Sector Distribution:")
print(table(train_data$Sector, train_data$Target))

#==== 02C - Analysis of the results ===========================================#

# Run analysis
analyze_distribution(train_data, "TRAIN SET")
analyze_distribution(test_data, "TEST SET")

# Optional: Verify that companies are not split across sets
intersect_ids <- length(intersect(train_data$CompanyID, test_data$CompanyID))
cat(paste0("\nNumber of Companies leaking between Train and Test: ", intersect_ids, "\n"))

#==============================================================================#
#==== 03 - SMOTE (+ Extions; see Zhao et al. (2024)) ==========================#
#==============================================================================#

#==== 03. - Functions =========================================================#


#==== 03A - Setup + Data ======================================================#

######### Implementing the Zhao et al. 2024 approach SMOTE + ENN.
library(smotefamily)
library(unbalanced)

# Function to apply SMOTE + ENN per sector
apply_sector_smote_enn <- function(data, target_col, sector_col, k_smote = 5) {
  
  # List to store processed chunks
  processed_sectors <- list()
  
  # Get unique sectors
  unique_sectors <- unique(data[[sector_col]])
  
  for(sec in unique_sectors) {
    # 1. Filter Data for this Sector
    sector_df <- data %>% filter(!!sym(sector_col) == sec)
    
    # 2. Check if SMOTE is possible
    # We need enough defaults to find k neighbors. 
    # If defaults < k+1, we cannot SMOTE. We just keep original.
    n_defaults <- sum(sector_df[[target_col]] == "1")
    
    if(n_defaults > (k_smote + 1)) {
      
      # --- Prepare for SMOTE ---
      # SMOTE requires numeric features. 
      # Remove non-numeric columns (ID, Year, Sector) but keep them for merging later if needed.
      # Here we assume Rev and Debt are the predictors.
      features <- sector_df %>% select(Rev, Debt)
      target   <- sector_df[[target_col]]
      
      # --- A. Apply SMOTE ---
      # dup_size = 0 means it determines ratio automatically to balance 50/50
      # K = neighbors
      smote_result <- SMOTE(X = features, target = target, K = k_smote, dup_size = 0)
      
      # Extract SMOTEd data
      smoted_data <- smote_result$data
      
      # The smotefamily package puts the target in a column named "class"
      # We need to separate X and Y for the ENN step
      X_smoted <- smoted_data %>% select(-class)
      Y_smoted <- as.factor(smoted_data$class)
      
      # --- B. Apply ENN (Cleaning) ---
      # ubENN removes instances that are misclassified by their 3 nearest neighbors
      enn_result <- ubENN(X = X_smoted, Y = Y_smoted, k = 3)
      
      # Combine cleaned X and Y back together
      cleaned_sector_data <- enn_result$X
      cleaned_sector_data$Target <- enn_result$Y
      
      # Add the Sector column back (so we can combine later)
      cleaned_sector_data$Sector <- sec
      
      # Note: We lost ID and Year for synthetic samples. 
      # For real samples, you might want to keep them, but usually, in training, 
      # we drop ID/Year anyway to avoid overfitting.
      
      processed_sectors[[sec]] <- cleaned_sector_data
      
      message(paste("Sector:", sec, "- Processed (SMOTE + ENN). Final Size:", nrow(cleaned_sector_data)))
      
    } else {
      # Not enough defaults to SMOTE -> Keep original data
      message(paste("Sector:", sec, "- Skiped (Not enough defaults). Kept Original."))
      
      # Keep only relevant columns to match the SMOTE output structure
      keep_data <- sector_df %>% select(Rev, Debt, Target, Sector)
      processed_sectors[[sec]] <- keep_data
    }
  }
  
  # Combine all sectors back into one dataframe
  final_train <- bind_rows(processed_sectors)
  return(final_train)
}

# --- Execute the Function ---
# Apply to the Training Data created in Step 2
balanced_train_data <- apply_sector_smote_enn(train_data, "Target", "Sector", k_smote = 3) 
# Note: k_smote=3 used here because mock data is small. In real data, use 5.

# --- Final Check ---
print("Original Train Balance:")
print(table(train_data$Target))

print("New Balanced Train Balance (After Sector-Specific SMOTE+ENN):")
print(table(balanced_train_data$Target))

