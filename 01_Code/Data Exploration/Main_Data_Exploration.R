#==============================================================================#
#==== 00 - Description ========================================================#
#==============================================================================#

### This code includes everything necessary for data exploration and feature selection.
### Do not put anything else in here (like models, or other stuff).

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

packages <- c("here", "corrplot", "dplyr", "tidyr", "car",
              "reshape2", "ggplot2",
              "rsample", "DataExplorer",  ## Necessary for stratified sampling.
              "scorecard", "gt", "reshape2"
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

Charts_Data_Exploration_Directory <- file.path(Charts_Directory, "Data Exploration")

## Plotting.
blue <- "#004890"
grey <- "#708090"
orange <- "#F37021"
red <- "#B22222"

height <- 1833
width <- 3750

## Data Sampling.
set.seed(123)


#==============================================================================#
#==== 02 - Data ===============================================================#
#==============================================================================#

#==== 02a - Read the data file ================================================#

Data <- load(Data_Directory)
Data <- d


Exclude <- c("id", "refdate") ## Drop the id and ref_date (year) for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

## Drop all ratios for now.
Exclude <- c(paste("r", seq(1:18), sep = "")) ## Drop all ratios for now.
Data <- Data[, -which(names(Data) %in% Exclude)]

Exclude <- c("id", "refdate", "size","sector", "y")
Features <- Data[, -which(names(Data) %in% Exclude)]

#==== 02b - Accounting Formulas & Check =======================================#

## ======================= ##
## Delta: f1 AND f2 + f3
## ======================= ##

Aktiva_TotalAssets_f2f3 <- rowSums(Data[,c("f2", "f3")])
Delta_f1_and_f2f3 <- (Data$f1 - Aktiva_TotalAssets_f2f3) / Data$f1
summary(Delta_f1_and_f2f3)

Min_f1_and_f2f3 <- min(Delta_f1_and_f2f3)

## No. of obs with a percentage deviation of more than 5%.
Delta_f1_and_f2f3_abs <- abs(Delta_f1_and_f2f3)

mean(Delta_f1_and_f2f3_abs > 0.05)
sum(Delta_f1_and_f2f3_abs > 0.05)

## Now check the extreme cases. Which company is it?

Pos <- which(Delta_f1_and_f2f3 == Min_f1_and_f2f3)
Company_Inconsistent1 <- Data[Pos,]

## ======================= ##
## Delta: f3 AND f4 + f5
## ======================= ##

Aktiva_TotalAssets_SumConstrained <- rowSums(Data[,c("f4", "f5")])
Delta_f3_and_f4f5 <- (Data$f3 - Aktiva_TotalAssets_SumConstrained) / Data$f3
summary(Delta_f3_and_f4f5)

## No. of obs with a percentage deviation of more than 5%.
Delta_f3_and_f4f5_abs <- abs(Delta_f3_and_f4f5)

mean(Delta_f3_and_f4f5_abs > 0.05, na.rm = TRUE)
sum(Delta_f3_and_f4f5_abs > 0.05)

## ======================= ##
## Delta: f1 AND f6 + f10 + f11 
##        Total Assets = Equity + Provisions + Liabilities
## ======================= ##

Sum_PassiveSide <- rowSums(Data[,c("f6", "f10", "f11")])
Delta_f1_and_f6f10f11 <- (Data$f1 - Sum_PassiveSide) / Data$f1
summary(Delta_f1_and_f6f10f11)

## No. of obs with a percentage deviation of more than 5%.
Delta_f1_and_f6f10f11_abs <- abs(Delta_f1_and_f6f10f11)

mean(Delta_f1_and_f6f10f11_abs > 0.05, na.rm = TRUE)
sum(Delta_f1_and_f6f10f11_abs > 0.05)

## Now check the extreme cases. Which company is it?

Pos <- which(Delta_f1_and_f6f10f11 == min(Delta_f1_and_f6f10f11))
Company_Inconsistent2 <- Data[Pos,]

## ======================= ##
## Now visualize the extreme cases.
## ======================= ##

Data_Visualisation <- rbind(Company_Inconsistent1, Company_Inconsistent2)



Sum_f8f9 <- rowSums(Data[,c("f8", "f9")])
Delta_f7_and_f8f9 <- (Data$f7 - Sum_f8f9) / Data$f7
summary(Delta_f7_and_f8f9)

sum(Data$f7 == Data$f8 + Data$f9)


#==== 02c - Dependent Variable Analysis =======================================#
## Part Tristan (Done).

tryCatch({
  
## ======================= ##
## Data imbalance.
## ======================= ##

## Get the total number of defaults and the frequency in the dataset.
Data %>%
  filter(y == 1) %>%
  summarise(n = n()) %>%
  mutate(total_rows = nrow(Data),
         proportion = n / total_rows * 100)

## Defaults per sector.
categorical_vars <- Data %>%
  select(where(~ is.factor(.) || is.character(.)))

contingency_tables <- lapply(names(categorical_vars), function(var) {
  tab <- table(Data$y, categorical_vars[[var]], useNA = "ifany")
  print(paste("Contingency table for:", var))
  print(tab)
  cat("\n")
  return(tab)
})
names(contingency_tables) <- names(categorical_vars)

## Plot the defaults per business sector.
Plot_Sector_Default <- Data %>%
  group_by(sector) %>%
  summarise(
    count = n(),
    default_rate = mean(y, na.rm = TRUE) # Calculate the mean of y (0s and 1s)
  ) %>%
  ggplot(aes(x = reorder(sector, -default_rate), y = default_rate)) +
  geom_bar(stat = "identity", fill = blue) +
  labs(title = "Default Rate by Sector",
       x = "Sector",
       y = "") +
  theme_minimal()

## Save the plot.
Path <- file.path(Charts_Data_Exploration_Directory, "01_Defaults_per_Sector.png")
ggsave(
  filename = Path,
  plot = Plot_Sector_Default,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

}, silent = TRUE)

#==== 02c - Distributions & bivariate Data Analysis ===========================#
## ======================= ##
## 1.2 Distributions and bivariate data analysis
## ======================= ##

##Renaming vector for graphs
var_names <- c(
  f1  = "Total assets",
  f2  = "Invested Capital",
  f3  = "Current Assets",
  f4  = "Inventories",
  f5  = "Cash",
  f6  = "Equity",
  f7  = "Retained Earning (GR)",
  f8  = "Net profit",
  f9  = "Retained Earning (GV)",
  f10 = "Provisions",
  f11 = "Liabilities"
)


## ======================= ##
## 1.2.1 Distributions and data dependancy ##
## ======================= ##

f1_f11 <- Data %>% dplyr::select(f1:f11)

# Long format
long11 <- tidyr::pivot_longer(
  f1_f11, 
  cols = everything(),
  names_to = "Variable", 
  values_to = "Value"
)


long11$Variable <- factor(long11$Variable, levels = paste0("f", 1:11))

plot <- ggplot(long11 %>% filter(Value > 0), aes(x = Value)) + 
  geom_histogram(fill = blue, color = "white", bins = 30) +
  scale_x_log10() +
  facet_wrap(~ Variable, scales = "free", ncol = 3,
             labeller = labeller(Variable = var_names)) +
  labs(
    title = "",
    x = "Value (log-scale)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = grey),
    axis.title.x = element_text(size = 13, face = "bold", color = "black"),
    axis.title.y = element_text(size = 13, face = "bold", color = "black"),
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#d9d9d9"),
    plot.margin = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

#Save

Path <- file.path(Charts_Data_Exploration_Directory, "010_distribution_f1_f11.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)


## ======================= ##
## 1.2.2 Class wide density plot
## ======================= ##


long11_y <- Data %>% 
  mutate(y = factor(y, levels = c(0, 1),
                    labels = c("No default", "Default"))) %>%
  dplyr:: select(y, f1:f11) %>%
  tidyr::pivot_longer(
    cols      = f1:f11,
    names_to  = "Variable",
    values_to = "Value"
  )

long11_y$Variable <- factor(long11_y$Variable, levels = paste0("f", 1:11))


plot <- ggplot(long11_y %>% dplyr::filter(Value > 0),
       aes(x = Value, colour = y, fill = y)) +
  geom_density(
    alpha    = 0.4,
    position = "identity",
    aes(y = after_stat(scaled)) 
  ) +
  scale_x_log10() +
  facet_wrap(
    ~ Variable,
    scales  = "free_x",
    ncol    = 3,
    labeller = labeller(Variable = var_names)  
  ) +
  scale_color_manual(
    values = c("No default" = blue, "Default" = red),
    name   = "Class"
  ) +
  scale_fill_manual(
    values = c("No default" = blue, "Default" = red),
    name   = "Class"
  ) +
  labs(
    title = "",
    x     = "Value (log-scale)",
    y     = "Scaled density",
    fill  = "Class",
    color = "Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title   = element_text(face = "bold", size = 16),
    axis.title.x = element_text(size = 13, face = "bold", colour = "black"),
    axis.title.y = element_text(size = 13, face = "bold", colour = "black"),
    strip.text   = element_text(size = 12, face = "bold", colour = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = "#d9d9d9"),
    plot.margin  = ggplot2::margin(t = 15, r = 10, b = 10, l = 10)
  )

#Save

Path <- file.path(Charts_Data_Exploration_Directory, "011_distribution.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## 1.2.2  Distribution of binary predictors
## ======================= ##

binary <- Data %>% 
  dplyr::select(groupmember:public)

binary_long <- binary %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "Value"
  )

# Compute proportions for labels
binary_props <- binary_long %>%
  group_by(Variable, Value) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Variable) %>%
  mutate(prop = n / sum(n))

plot <- ggplot(binary_props, 
       aes(x = Variable, y = prop, fill = factor(Value))) +
  geom_col(position = "stack", color = "white", linewidth = 0.3) +
  
  geom_text(
    aes(label = ifelse(prop > 0.05, scales::percent(prop, accuracy = 1), "")),
    position = position_stack(vjust = 0.5),
    size = 4,
    color = "white",
    fontface = "bold"
  ) +
  
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(
    values = c("0" = blue, "1" = red),
    labels = c("0" = "No", "1" = "Yes"),
    name   = "Default"
  ) +
  
  labs(
    title = "Distribution of Binary Predictors",
    x = "Variable",
    y = "Share of observations"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title   = element_text(face = "bold", size = 17),
    axis.text.x  = element_text(angle = 45, hjust = 1, size = 12),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank()
  )

Path <- file.path(Charts_Data_Exploration_Directory, "012_dis_binary.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## 1.2.3 Descriptive statistics
## ======================= ##

# Compute Mean, Variance, Skewness consistently for the same 11 vars
Mean     <- apply(as.matrix(f1_f11 ), MARGIN = 2, FUN = mean)
Variance <- apply(as.matrix(f1_f11 ), MARGIN = 2, FUN = var)
St.dev <- apply(as.matrix(f1_f11 ), MARGIN = 2, FUN = sd)
Skewness <- apply(as.matrix(f1_f11 ), MARGIN = 2, FUN = skew)

# Build table
Summary_f1_f11 <- data.frame(
  Mean     = round(Mean, 1),
  St.dev = round(St.dev, 1),
  Skewness = round(Skewness, 1)
)


# table <- Summary_f1_f11 %>%
#   dplyr::mutate(Variable = var_names) %>%             
#   dplyr::select(Variable, Mean, St.dev, Skewness) %>% 
#   knitr::kable(
#     caption   = "Summary Statistics for Balance Sheet Predictors",
#     col.names = c("Variable", "Mean", "St.dev", "Skewness"),
#     align     = "lccc",
#     booktabs  = TRUE
#   ) %>%
#   kableExtra::kable_classic(full_width = FALSE, html_font = "Cambria") %>%
#   kableExtra::row_spec(0, bold = TRUE, background = "#004890", color = "white") %>%
#   kableExtra::row_spec(1:nrow(Summary_f1_f11), background = "#f7f7f7")

# kableExtra::save_kable(
#   table,
#   file = file.path(Charts_Data_Exploration_Directory, "013_Summary_f1_f11.png")
# )

## Or using the gt-package.

table_gt <- Summary_f1_f11 %>%
  mutate(Variable = var_names) %>%
  select(Variable, Mean, St.dev, Skewness) %>%
    gt() %>%
    tab_header(
    title = ""
  ) %>%
    cols_align(
    align = "left",
    columns = Variable
  ) %>%
  cols_align(
    align = "center",
    columns = c(Mean, St.dev, Skewness)
  ) %>%
  tab_style(
    style = list(
      cell_fill(color = "#004890"),
      cell_text(color = "white", weight = "bold")
    ),
    locations = cells_column_labels()
  ) %>%
  tab_style(
    style = cell_fill(color = "#f7f7f7"),
    locations = cells_body()
  ) %>%
  opt_table_font(
    font = list("Cambria", default_fonts())
  ) %>%
  tab_options(
    table.width = "auto" # Keeps the table compact (full_width = FALSE)
  )

table_gt

Path <- file.path(Charts_Data_Exploration_Directory, "09_SummaryStats_Features.png")
gtsave(table_gt, Path, vwidth = width, vheight = height)

## ======================= ##
## 1.2.4 Descriptive statistics - Visualize
## ======================= ##

{

stats_long <- data.frame(
  Variable = factor(names(f1_f11), levels = names(f1_f11)),
  Mean     = Mean,
  `St.dev` = St.dev,
  Skewness = Skewness
) %>%
  pivot_longer(
    cols = c(Mean, `St.dev`, Skewness),
    names_to = "Statistic",
    values_to = "Value"
  )


stats_large <- stats_long %>% 
  filter(Statistic %in% c("Mean", "St.dev"))

plot_stats <- ggplot(stats_large, 
                     aes(x = Variable, y = Value, 
                         color = Statistic, group = Statistic)) +
  
  geom_line(linewidth = 1.2, alpha = 0.9) +
  geom_point(size = 3, alpha = 0.9) +
    scale_color_manual(values = c("Mean" = blue, "St.dev" = red)) +
  
  scale_x_discrete(labels = var_names) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.1))) +
    labs(
    title = "",
    subtitle = "",
    x = NULL,    #
    y = "Value",
    color = ""
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.margin = margin(t = 40, r = 20, b = 10, l = 10),
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.margin = margin(t = -5),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(margin = margin(r = 10), color = "black"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "gray95"),
    panel.grid.major.y = element_line(color = "gray90")
  ) +
    coord_cartesian(clip = "off")

plot_stats

#Save

Path <- file.path(Charts_Data_Exploration_Directory, "014_mean_stddev.png")
ggsave(
  filename = Path,
  plot = plot_stats,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

stats_skew <- stats_long %>% filter(Statistic == "Skewness")

plot <- ggplot(stats_skew,
       aes(x = Variable, y = Value, group = 1)) +
  geom_line(color = red, linewidth = 1.2) +
  geom_point(color = red, size = 3) +
  scale_x_discrete(
    limits = paste0("f", 1:11),         # enforce correct order
    labels = var_names                  # use your nice names
  ) +
  labs(
    title = "Skewness of Balance Sheet Predictors",
    x = "Variable",
    y = "Skewness"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 12, color = grey),
    axis.text.x   = element_text(angle = 45, hjust = 1)
  )

#Save

Path <- file.path(Charts_Data_Exploration_Directory, "015_skewness.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

}

#==============================================================================#
#==== 1.3 - Feature Selection Engineering =====================================#
#==============================================================================#

tryCatch({
  
## ======================= ##
## 1.3.1. Informational Value.
## ======================= ##

## Tells us how good a feature seperates between NoDefault (y=0) and Default (y=1).
iv_summary <- iv(Data, y = "y")
print(iv_summary %>% arrange(desc(info_value)))

iv_summary <- iv_summary %>%
  mutate(
    variable = coalesce(var_names[variable], variable)
  )

iv_summary$info_value <- round(iv_summary$info_value, 1)
print(iv_summary %>% arrange(desc(info_value)))

## Plot the informational value.

iv_summary <- iv_summary %>%
  mutate(
    power_category = case_when(
      info_value > 0.5   ~ "Very Strong",
      info_value >= 0.3  ~ "Strong",
      info_value >= 0.1  ~ "Medium",
      info_value >= 0.02 ~ "Weak",
      TRUE               ~ "Very Weak"
    ),
    power_category = factor(power_category, 
                            levels = c("Very Strong", "Strong", "Medium", "Weak", "Very Weak"))
  )


plot_IV <- ggplot(iv_summary, aes(x = info_value, y = reorder(variable, info_value))) +
    geom_col(aes(fill = power_category), width = 0.75, alpha = 0.9) + 
    geom_text(aes(label = sprintf("%.1f", info_value)), 
            hjust = -0.2, 
            size = 3.5, 
            fontface = "bold", 
            color = "#333333") +
    geom_vline(xintercept = c(0.02, 0.1, 0.3, 0.5), 
             linetype = "dashed", color = "gray60", linewidth = 0.4) +
  # annotate("text", x = 0.01, y = Inf, label = "Useless", vjust = -1, size = 3, color = "gray40", fontface = "italic") +
  # annotate("text", x = 0.06, y = Inf, label = "Weak",    vjust = -1, size = 3, color = "gray40", fontface = "italic") +
  # annotate("text", x = 0.2,  y = Inf, label = "Medium",  vjust = -1, size = 3, color = "gray40", fontface = "italic") +
  # annotate("text", x = 0.4,  y = Inf, label = "Strong",  vjust = -1, size = 3, color = "gray40", fontface = "italic") +
  # annotate("text", x = 0.55, y = Inf, label = "Very Strong", vjust = -1, size = 3, color = "gray40", fontface = "italic") +
    scale_fill_manual(values = c(
    "Very Strong" = "#d53e4f", 
    "Strong"      = "#f46d43", 
    "Medium"      = "#fdae61", 
    "Weak"        = "#fee08b", 
    "Very Weak"     = "#e6f598"
  )) +
  
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
    title = "",
    subtitle = "",
    x = "Information Value",
    y = NULL, 
    fill = "Predictive Power"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.margin = margin(t = 40, r = 20, b = 10, l = 10), 
    
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.margin = margin(t = -5),
      axis.text.y = element_text(color = "black", size = 10, face = "plain"),
    axis.text.x = element_text(color = "gray40"),
    axis.title.x = element_text(margin = margin(t = 10), color = "gray30"),
    
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "gray90")
  ) +
  coord_cartesian(clip = "off")

plot_IV

Path <- file.path(Charts_Data_Exploration_Directory, "02_IV_per_feature.png")
ggsave(
  filename = Path,
  plot = plot_IV,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## Data separation.
## ======================= ##

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
## 1.3.2 Significance tests.
## ======================= ##

## ======================= ##
## 1.3.2.1 Calculating group means ##
## ======================= ##

df <- cbind(y = Data$y, Features) %>% 
  as.data.frame()

aggregate(. ~ y, data = Data, FUN = mean, na.rm = TRUE)

feature_names <- colnames(Features)

group_means <- df %>%
  dplyr::group_by(y) %>%
  dplyr::summarise(
    across(all_of(feature_names), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

group_means

## ============================================ ##
##  1.3.2.2 Welch two-sample t-tests for all features ##
## ============================================ ##


results <- sapply(feature_names , function(v) {
  test <- try(t.test(df[[v]] ~ df$y), silent = TRUE)
  if (inherits(test, "try-error")) NA_real_ else test$p.value
})

mean_test_results <- data.frame(
  Variable = feature_names ,
  p_value  = as.numeric(results)
)

# Sort by smallest p-value
mean_test_results <- mean_test_results[order(mean_test_results$p_value), ]

print(mean_test_results)


## ======================= ##
## Visualize the tests
## ======================= ##

plot_df <- mean_test_results %>%
  mutate(
    Variable = factor(Variable, levels = Variable),
    log_p = -log10(p_value),
    Sig = p_value < 0.05
  )

plot <- ggplot(plot_df, aes(x = Variable, y = log_p, fill = Sig)) +
  geom_col() +
  scale_fill_manual(
    values = c("TRUE" = "#d62728",   # red = significant
               "FALSE" = "grey70"),  # grey = not significant
    labels = c("TRUE" = "Significant (p < 0.05)",
               "FALSE" = "Not significant"),
    name = ""
  ) +
  geom_hline(yintercept = -log10(0.05), 
             color = "#004890", linetype = "dashed", size = 1) +
  labs(
    title = "",
    subtitle = "",
    x = "Variable",
    y = "-log10(p-value)"
  ) + scale_x_discrete(labels = var_names) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1),
    axis.text.y = element_text(color = "black", size = 10, face = "plain"),
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "top"
  )

##

plot <- ggplot(plot_df, aes(x = Variable, y = log_p)) +
    geom_col(aes(fill = Sig), width = 0.75, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.1f", log_p)), 
            vjust = -0.5, 
            size = 3.5, 
            fontface = "bold", 
            color = "#333333") +
    geom_hline(yintercept = -log10(0.05), 
             color = blue, linetype = "dashed", linewidth = 0.6) +
  scale_fill_manual(
    values = c("TRUE" = red,   # red
               "FALSE" = grey),  # grey
    labels = c("TRUE" = "Significant",
               "FALSE" = "Not significant"),
    name = "Significance (p < 0.05)"
  ) +
  
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    scale_x_discrete(labels = var_names) +
  labs(
    title = "",
    subtitle = "",
    x = NULL, 
    y = "-log10(p-value)"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.margin = margin(t = 40, r = 20, b = 10, l = 10), 
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.margin = margin(t = -5),
    axis.text.y = element_text(color = "black", size = 10),
    axis.text.x = element_text(color = "black", angle = 60, hjust = 1), 
    axis.title.y = element_text(margin = margin(r = 10), color = "black"),
    
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray90")
  ) +
  coord_cartesian(clip = "off")

#Save
Path <- file.path(Charts_Data_Exploration_Directory, "016_Welch_twosample_t_test.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)


########

categorical_vars <- Data %>%
  dplyr::select(where(~ is.factor(.) || is.character(.)))

if (ncol(categorical_vars) == 0) stop("No categorical predictors in Data.")

chi_results <- lapply(names(categorical_vars), function(v) {
  tab  <- table(Data$y, Data[[v]], useNA = "ifany")
  test <- suppressWarnings(chisq.test(tab))
  data.frame(
    Variable = v,
    Chi2     = as.numeric(test$statistic),
    df       = as.numeric(test$parameter),
    p_value  = test$p.value
  )
}) |> bind_rows() |>
  arrange(p_value) |>
  mutate(
    log_p = -log10(p_value),
    Sig   = p_value < 0.05
  )


plot <- ggplot(chi_results,
       aes(x = factor(Variable, levels = Variable),
           y = log_p, fill = Sig)) +
  geom_col() +
  geom_hline(yintercept = -log10(0.05),
             linetype = "dashed") +
  scale_fill_manual(values = c("TRUE" = "#d62728", "FALSE" = "grey70"),
                    labels = c("TRUE" = "Significant (p < 0.05)",
                               "FALSE" = "Not significant"),
                    name = "") +
  labs(
    title = "P-values for Welch's Two-Sample t-tests",
    subtitle = "Higher bars = more significant difference between groups",
    x = "Categorical variable",
    y = "-log10(p-value)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 60, hjust = 1),
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "top"
  )

#Save
Path <- file.path(Charts_Data_Exploration_Directory, "017_chi_test.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## Multicollinearity.
## ======================= ##

cor_matrix <- cor(Features[,])

new_names <- var_names[rownames(cor_matrix)]
new_names[is.na(new_names)] <- rownames(cor_matrix)[is.na(new_names)]
rownames(cor_matrix) <- new_names
colnames(cor_matrix) <- new_names
cor_matrix[upper.tri(cor_matrix)] <- NA
melted_cor_matrix <- melt(cor_matrix, na.rm = TRUE)
var_order <- colnames(cor_matrix)

plot_data <- melted_cor_matrix %>%
  mutate(
    Var1 = factor(Var1, levels = var_order),
    Var2 = factor(Var2, levels = var_order)
  )

## Plot.

plot_MC <- ggplot(plot_data, aes(x = Var2, y = Var1, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  
  geom_text(aes(label = sprintf("%.2f", value)), 
            color = "#333333", 
            size = 2, 
            fontface = "plain") +
  
  scale_fill_gradient2(
    low = blue, 
    high = red, 
    mid = "white", 
    midpoint = 0, 
    limit = c(-1, 1), 
    name = "Correlation"
  ) +
  
  scale_y_discrete(limits = rev(var_order)) +
    scale_x_discrete(position = "bottom") +
  
  labs(
    title = "",
    subtitle = "",
    x = NULL, 
    y = NULL
  ) +
    theme_minimal(base_size = 12) +
  theme(
    plot.margin = margin(t = 40, r = 20, b = 10, l = 10),
    
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10),
    legend.margin = margin(t = -5),
    legend.key.width = unit(1.5, "cm"),
    
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    
    panel.grid = element_blank()
  ) +
  coord_fixed()

plot_MC


Path <- file.path(Charts_Data_Exploration_Directory, "04_Multicollinearity.png")
ggsave(
  filename = Path,
  plot = plot_MC,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

## ======================= ##
## Implications from IV and multicollinearity.
## ======================= ##

summary_table <- iv_summary %>%
  mutate(
    Concern = case_when(
      variable %in% c("f1", "f2", "f6", "f11") ~ "High (Group 1)",
      variable %in% c("f8", "f9") ~ "High (Group 2)",
      TRUE ~ "Low"
    ),
    Recommendation = case_when(
      variable == "f6" ~ "Keep (Highest IV in Group 1)",
      variable %in% c("f1", "f2", "f11") ~ "Remove",
      variable == "f8" ~ "Keep (Highest IV in Group 2)",
      variable == "f9" ~ "Remove",
      info_value < 0.02 ~ "Remove (Useless IV)",
      TRUE ~ "Keep"
    )
  ) %>%
  select(
    "variable",
    "Information Value (IV)" = info_value,
    "Multicollinearity Concern" = Concern,
    "Action" = Recommendation
  )

}, silent = TRUE)

## ======================= ##
## VIF
## ======================= ##

dummy_y <- rnorm(nrow(Features))

# Fit the model with all predictors
vif_model <- lm(dummy_y ~ ., data = Features)

# Compute VIF
vif_values <- car::vif(vif_model)


vif_table <- data.frame(
  Variable = names(vif_values),
  VIF = as.numeric(vif_values)
) %>%
  mutate(Classification = case_when(
    VIF < 5 ~ "OK",
    VIF >= 5 & VIF < 10 ~ "Moderate",
    VIF >= 10 ~ "High"
  ))

plot <- plot_VIF <- ggplot(vif_table, aes(x = reorder(Variable, VIF), y = VIF, fill = VIF)) +
  geom_col(color = "white") +
  coord_flip() +
  scale_fill_gradient(low = blue, high = orange) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "darkred") +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
  labs(
    title = "Variance Inflation Factors (VIF)",
    x = "Predictor",
    y = "VIF"
  ) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank())

Path <- file.path(Charts_Data_Exploration_Directory, "018_VIF.png")
ggsave(
  filename = Path,
  plot = plot,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

#==== 02e - Data splitting ====================================================#
## Part Tristan (DONE)
## See documentation, done 80/20.




#==== 02f - Handling outliers =================================================#
var_order <- paste0("f", 1:11)

# Signed log transformation
long11 <- long11 %>%
  mutate(
    Variable = factor(Variable, levels = var_order),
    Value_sl = sign(Value) * log10(1 + abs(Value)),
    Label    = var_names[Variable],                          # 
    Label    = factor(Label, levels = var_names[var_order])  
  )


outliers<-ggplot(long11, aes(x = Label, y = Value_sl)) +
  geom_boxplot(
    fill           = grey,
    color          = blue,
    outlier.colour = red,
    outlier.alpha  = 0.8,
    outlier.size   = 1.8
  ) +
  scale_y_continuous(
    breaks = -6:6,   # adjust depending on scale
    labels = function(x) scales::comma(10^abs(x) * sign(x))
  ) +
  labs(
    title = "",
    x     = "Variable",
    y     = "Value (signed log10)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title  = element_text(face = "bold", color = blue, size = 16),
    axis.title  = element_text(color = "black", face = "bold"),
    axis.text.x = element_text(angle = 60, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black")
  )

Path <- file.path(Charts_Data_Exploration_Directory, "019_Outliers.png")
ggsave(
  filename = Path,
  plot = outliers,
  width = width,
  height = height,
  units = "px",
  dpi = 300,
  limitsize = FALSE
)

#==============================================================================#
#==== 03 - Data Preparation & Feature selection ===============================#
#==============================================================================#

#==== 03a - Splitting the dataset =============================================#
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

## ======================= ##
## Findings.
## ======================= ##

## Multicollinearity: Remove f9 and keep f8. They are extremely similar.
## IV: Remove non-predictive variables. Groupmember likely just adds random noise.
## WoE-transformed logistic regression as the "base" case.
## Handling of NAs within the "r" features.

## For the future: Stability of variables over time (using the population stability index).

#==============================================================================#
#==== 04 - Old Code (Data exploration) ========================================#
#==============================================================================#

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

## ======================= ##
## Check the group means
## ======================= ##

#Calculate with ratios
Data<-d
colSums(is.na(Data)) ## We fine some NAs, mostly in the financial ratio's.
sapply(Data, function(x) sum(is.infinite(x)))
sapply(Data, function(x) sum(is.nan(x)))

Data <- Data %>%
  mutate(across(where(is.numeric), ~ifelse(!is.finite(.), NA, .)))

aggregate(. ~ y, data = cbind(y = Data$y, Features), FUN = mean, na.rm = TRUE)


# 2. Compute group means by y
group_means <- Data %>%
  select(y, all_of(names(Features))) %>%   # keep only y + feature columns
  group_by(y) %>%
  summarise(across(all_of(names(Features)), mean, na.rm = TRUE))

##Use two sample t-test (Welch's test) to see if means differ

# Exclude group means which produce Nans and Inf
vars <- setdiff(names(Features), c("r5","r15", "r16", "r17", "r18"))

# Run default Welch t-tests for all remaining features
results <- sapply(vars, function(var) {
  test <- try(t.test(Data[[var]] ~ Data$y), silent = TRUE)
  if (inherits(test, "try-error")) return(NA_real_) else return(test$p.value)
})

# Create summary table
mean_test_results <- data.frame(
  Variable = vars,
  p_value = results
)

# Sort by smallest p-value
mean_test_results <- mean_test_results[order(mean_test_results$p_value), ]

print(mean_test_results)


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

## ======================= ##
## Create contingency tables vs. outcome y
## ======================= ##
categorical_vars <- Data %>%
  select(where(~ is.factor(.) || is.character(.)))

contingency_tables <- lapply(names(categorical_vars), function(var) {
  tab <- table(Data$y, categorical_vars[[var]], useNA = "ifany")
  print(paste("Contingency table for:", var))
  print(tab)
  cat("\n")
  return(tab)
})
names(contingency_tables) <- names(categorical_vars)

## ======================= ##
## Conduct a chi-squared test
## ======================= ##

chi_results <- lapply(names(categorical_vars), function(var) {
  tab <- table(Data$y, Data[[var]], useNA = "ifany")
  test <- suppressWarnings(chisq.test(tab))
  data.frame(
    Variable = var,
    Chi2 = test$statistic,
    df = test$parameter,
    p_value = test$p.value
  )
})

chi_results <- do.call(rbind, chi_results)
chi_results <- chi_results[order(chi_results$p_value), ]

print(chi_results)


## ======================= ##
## Plot the results
## ======================= ##

cat_long <- Data %>%
  select(y, all_of(names(categorical_vars))) %>%
  pivot_longer(-y,
               names_to = "Variable",
               values_to = "Category",
               values_ptypes = list(Category = character()))

# Compute percentages manually for labels
cat_plot_data <- cat_long %>%
  group_by(Variable, Category, y) %>%
  summarise(n = n(), .groups = "drop_last") %>%
  mutate(pct = n / sum(n))

ggplot(cat_plot_data, aes(x = Category, y = pct, fill = factor(y))) +
  geom_col(position = "fill", color = "white") +
  geom_text(
    aes(label = scales::percent(pct, accuracy = 1)),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 3.5,
    fontface = "bold"
  ) +
  facet_wrap(~ Variable, scales = "free_x") +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(
    name = "Outcome (y)",
    values = c("0" = grey, "1" = orange)
  ) +
  labs(
    title = "Distribution of Outcome y Across Categorical Variables",
    x = "Category", y = "Percentage", fill = "y"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.text = element_text(face = "bold", color = blue),
    plot.title = element_text(face = "bold", color = red, hjust = 0.5)
  )

ggsave(
  filename = Charts_Directory,
  plot = last_plot(),   
  width = 10,           
  height = 6,           
  dpi = 300,          
  bg = "white"         
)
