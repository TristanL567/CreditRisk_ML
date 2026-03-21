# install.packages(c("readxl","dplyr","ggplot2","stringr"))  # if needed
library(readxl)
library(dplyr)
library(ggplot2)
library(stringr)

# -------- paths --------
excel_path <- "/Users/admin/Desktop/Industry Lab/01_Code/GLM_Variable_Importance_By_Strategy.xlsx"
out_dir    <- "/Users/admin/Desktop/Industry Lab/01_Code/plots_beta_importance"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# -------- list sheets --------
sheets <- excel_sheets(excel_path)

# -------- helper: plot one strategy --------
plot_strategy_beta <- function(sheet_name, top_n = 10) {
  
  df <- read_excel(excel_path, sheet = sheet_name) %>%
    mutate(
      AbsCoef = abs(Coef),
      Feature = as.character(Feature)
    ) %>%
    arrange(desc(AbsCoef)) %>%
    slice_head(n = top_n) %>%
    # order for plotting (largest abs at top)
    arrange(AbsCoef) %>%
    mutate(
      Feature_wrapped = str_wrap(Feature, width = 30),
      Feature_wrapped = factor(Feature_wrapped, levels = Feature_wrapped)
    )
  
  p <- ggplot(df, aes(x = Coef, y = Feature_wrapped)) +
    geom_col() +
    geom_vline(xintercept = 0, linewidth = 0.8) +
    labs(
      title = paste0(sheet_name, ": Top ", top_n, " features by β"),
      x = "Beta coefficient (β)",
      y = NULL
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.y = element_text(size = 11)
    )
  
  # save
  file_out <- file.path(out_dir, paste0(gsub(" ", "_", sheet_name), "_Top10_Beta.png"))
  ggsave(filename = file_out, plot = p, width = 10, height = 6, dpi = 300)
  
  return(invisible(file_out))
}

# -------- run for all strategies --------
saved_files <- sapply(sheets, plot_strategy_beta, top_n = 10)
print(saved_files)
cat("\nSaved plots to:\n", out_dir, "\n")