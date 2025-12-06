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

packages <- c("dplyr", "caret",
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

## 

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

## Now guarantee that no sector has 0 defaults in the test or train set.
test_forced_ids <- company_summary %>%
  filter(Target == "1") %>%
  group_by(Sector) %>%
  sample_n(1) %>% 
  pull(CompanyID)

remaining_pool <- company_summary %>% filter(!CompanyID %in% test_forced_ids) ## Removes the obs. from the pool.

## Split 70/30.
train_idx <- createDataPartition(remaining_pool$StratKey, p = 0.7, list = FALSE)
train_pool_ids <- remaining_pool$CompanyID[train_idx]
test_pool_ids  <- remaining_pool$CompanyID[-train_idx]

## Now split. 
final_train_ids <- train_pool_ids
final_test_ids  <- c(test_pool_ids, test_forced_ids)
train_data_fixed <- df_panel %>% filter(CompanyID %in% final_train_ids)
test_data_fixed  <- df_panel %>% filter(CompanyID %in% final_test_ids)

## Split based on ID.
cat("\n--- TRAIN SET DEFAULTS (Should be >= 1 for all sectors) ---\n")
print(table(train_data_fixed$Sector, train_data_fixed$Target))

cat("\n--- TEST SET DEFAULTS (Should be >= 1 for all sectors) ---\n")
print(table(test_data_fixed$Sector, test_data_fixed$Target))

#==== 02C - Analysis of the results ===========================================#

# Run analysis
analyze_distribution(train_data_fixed, "TRAIN SET")
analyze_distribution(test_data_fixed, "TEST SET")

# Optional: Verify that companies are not split across sets
intersect_ids <- length(intersect(train_data_fixed$CompanyID, test_data_fixed$CompanyID))
cat(paste0("\nNumber of Companies leaking between Train and Test: ", intersect_ids, "\n"))

#==============================================================================#
#==== 03 - SMOTE (+ Extions; see Zhao et al. (2024)) ==========================#
#==============================================================================#

#==== 03. - Functions =========================================================#


#==== 03A - Setup + Data ======================================================#


