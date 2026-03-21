library(readxl)
library(dplyr)
library(ggplot2)
library(stringr)

# ---- ILAB colors + export size ----
blue  <- "#004890"
grey  <- "#708090"

width_px  <- 3750
height_px <- 1833
dpi_out   <- 300

theme_ilab_clean <- function(base_size = 22) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(size = base_size + 6, hjust = 0),
      legend.position = "none",
      
      axis.title.x = element_text(size = base_size + 2),
      axis.title.y = element_blank(),
      
      axis.text.x = element_text(size = base_size),
      axis.text.y = element_text(size = base_size),
      
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey85", linewidth = 1.2)
    )
}

# ---- paths ----
excel_path <- "/Users/admin/Desktop/Industry Lab/01_Code/GLM_Variable_Importance_By_Strategy.xlsx"
out_dir    <- "/Users/admin/Desktop/Industry Lab/01_Code/plots_beta_importance_clean"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

plot_strategy_beta_clean <- function(sheet_name, top_n = 10) {
  
  df <- read_excel(excel_path, sheet = sheet_name) %>%
    mutate(
      Feature = as.character(Feature),
      AbsCoef = abs(Coef),
      Sign    = ifelse(Coef >= 0, "Positive", "Negative")
    ) %>%
    arrange(desc(AbsCoef)) %>%
    slice_head(n = top_n) %>%
    arrange(AbsCoef) %>%
    mutate(
      Feature_wrapped = str_wrap(Feature, width = 28),
      Feature_wrapped = factor(Feature_wrapped, levels = Feature_wrapped),
      label = sprintf("%.3f", Coef),
      hjust_pos = ifelse(Coef >= 0, 1.05, -0.05)
    )
  
  p <- ggplot(df, aes(x = Coef, y = Feature_wrapped, fill = Sign)) +
    geom_col(width = 0.78) +
    geom_vline(xintercept = 0, color = "grey60", linewidth = 1.5) +
    geom_text(
      aes(label = label, hjust = hjust_pos),
      color = "white",
      size = 7
    ) +
    scale_fill_manual(values = c("Negative" = grey, "Positive" = blue)) +
    labs(
      title = paste0(sheet_name, ": Top ", top_n, " Features by |β|"),
      x = "Beta coefficient (β)"
    ) +
    coord_cartesian(clip = "off") +
    theme_ilab_clean(base_size = 26)
  
  file_out <- file.path(out_dir, paste0(gsub(" ", "_", sheet_name), "_Top10_Beta_CLEAN.png"))
  
  ggsave(
    filename = file_out,
    plot     = p,
    width    = width_px,
    height   = height_px,
    units    = "px",
    dpi      = dpi_out
  )
  
  message("Saved: ", file_out)
  invisible(file_out)
}

# ---- run for all strategies ----
sheets <- excel_sheets(excel_path)
sapply(sheets, plot_strategy_beta_clean, top_n = 10)

cat("\nSaved to:\n", out_dir, "\n")