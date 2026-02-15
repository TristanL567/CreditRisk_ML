library(readxl)
library(dplyr)
library(ggplot2)
library(stringr)
library(scales)

# ---- style params (your ILAB style) ----
blue  <- "#004890"
grey  <- "#708090"

width_px  <- 3750
height_px <- 1833
dpi_out   <- 300

theme_ilab <- function(base_size = 18) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title    = element_text(face = "bold", size = base_size + 8, hjust = 0),
      plot.subtitle = element_text(size = base_size + 2, hjust = 0),
      legend.position = "top",
      legend.title    = element_blank(),
      legend.text     = element_text(size = base_size),
      axis.title.x  = element_text(face = "bold", size = base_size + 2),
      axis.title.y  = element_text(face = "bold", size = base_size + 2),
      axis.text.x   = element_text(size = base_size),
      axis.text.y   = element_text(size = base_size),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey85", linewidth = 0.8),
      plot.margin = margin(12, 12, 12, 12)
    )
}

# ---- paths ----
excel_path <- "/Users/admin/Desktop/Industry Lab/01_Code/GLM_Variable_Importance_By_Strategy.xlsx"
out_dir    <- "/Users/admin/Desktop/Industry Lab/01_Code/plots_beta_importance"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- plot function ----
plot_strategy_beta_ilab <- function(sheet_name, top_n = 10) {
  
  df <- read_excel(excel_path, sheet = sheet_name) %>%
    mutate(
      AbsCoef = abs(Coef),
      Feature = as.character(Feature)
    ) %>%
    arrange(desc(AbsCoef)) %>%
    slice_head(n = top_n) %>%
    arrange(AbsCoef) %>%   # so biggest is on top in horizontal chart
    mutate(
      Feature_wrapped = str_wrap(Feature, width = 32),
      Feature_wrapped = factor(Feature_wrapped, levels = Feature_wrapped),
      Sign = ifelse(Coef >= 0, "Positive β", "Negative β")
    )
  
  p <- ggplot(df, aes(x = Coef, y = Feature_wrapped, fill = Sign)) +
    geom_col(width = 0.75) +
    geom_vline(xintercept = 0, linewidth = 0.9, color = "grey55") +
    geom_text(
      aes(label = sprintf("%.3f", Coef)),
      hjust = ifelse(df$Coef >= 0, -0.15, 1.15),
      size = 5,
      fontface = "bold",
      color = "black"
    ) +
    scale_fill_manual(values = c("Positive β" = blue, "Negative β" = grey)) +
    labs(
      title = paste0(sheet_name, ": Top ", top_n, " Features by β"),
      x = "Beta coefficient (β)",
      y = NULL
    ) +
    coord_cartesian(clip = "off") +
    theme_ilab(base_size = 18)
  
  file_out <- file.path(out_dir, paste0(gsub(" ", "_", sheet_name), "_Top10_Beta_ILAB.png"))
  
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

# ---- run for all strategies (all sheets) ----
sheets <- excel_sheets(excel_path)
saved_files <- sapply(sheets, plot_strategy_beta_ilab, top_n = 10)

print(saved_files)
cat("\nSaved plots to:\n", out_dir, "\n")