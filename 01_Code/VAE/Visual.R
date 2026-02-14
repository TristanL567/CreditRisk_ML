library(dplyr)
library(ggplot2)
library(scales)
library(gt)
library(stringr)


blue   <- "#004890"
grey   <- "#708090"
orange <- "#F37021"
red    <- "#B22222"

width_px  <- 3750
height_px <- 1833
dpi_out   <- 300
width_in  <- width_px / dpi_out
height_in <- height_px / dpi_out

# Clean, slide-friendly theme (white background, crisp text)
theme_ilab_clean <- function(base_size = 18) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title      = element_text(face = "bold", size = base_size + 10, hjust = 0.5),
      plot.subtitle   = element_text(size = base_size + 1, hjust = 0.5),
      axis.title.x    = element_text(face = "bold", size = base_size + 2),
      axis.title.y    = element_text(face = "bold", size = base_size + 2),
      axis.text.x     = element_text(color = "grey20", size = base_size),
      axis.text.y     = element_text(color = "grey20", size = base_size),
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_line(color = "grey85", linewidth = 0.6),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position  = "top",
      legend.title     = element_text(face = "bold", size = base_size),
      legend.text      = element_text(size = base_size),
      legend.key.size  = unit(1.0, "cm"),
      plot.margin      = margin(20, 30, 20, 30)
    )
}


plot_glm_auc <- function(tbl, title = "GLM Strategy Comparison by AUC") {
  
  tbl2 <- tbl %>%
    filter(!grepl("Strategy D", Model)) %>%
    mutate(
      Model_wrapped = str_wrap(Model, width = 22),
      Model_wrapped = factor(Model_wrapped, levels = Model_wrapped[order(AUC)]),
      Group = case_when(
        grepl("^Base Model", Model) ~ "Base Model",
        grepl("Strategy E", Model)  ~ "Denoising",
        TRUE                        ~ "Other Strategy"
      )
    )
  
  ymax <- max(tbl2$AUC, na.rm = TRUE)
  
  ggplot(tbl2, aes(x = Model_wrapped, y = AUC, fill = Group)) +
    geom_col(width = 0.65) +
    geom_text(aes(label = percent(AUC, accuracy = 0.1)),
              hjust = -0.15, size = 6, fontface = "bold") +
    coord_flip(clip = "off") +
    scale_fill_manual(
      name = "Model Type",
      values = c(
        "Base Model"     = grey,
        "Denoising"      = blue,
        "Other Strategy" = orange
      )
    ) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      limits = c(0, ymax + 0.03),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(title = title, x = NULL, y = "AUC-Score") +
    theme_ilab_clean()
}


plot_glm_uplift <- function(tbl, title = "AUC Uplift vs Base Model") {
  
  tbl2 <- tbl %>%
    filter(!grepl("Strategy D", Model)) %>%
    mutate(
      Uplift_AUC_pct = replace_na(Uplift_AUC_pct, 0),
      Model_wrapped  = str_wrap(Model, width = 22),
      Model_wrapped  = factor(Model_wrapped, levels = Model_wrapped[order(Uplift_AUC_pct)]),
      Group = case_when(
        grepl("^Base Model", Model) ~ "Base Model",
        Uplift_AUC_pct < 0          ~ "Negative Uplift",
        TRUE                        ~ "Positive Uplift"
      )
    )
  
  yabs <- max(abs(tbl2$Uplift_AUC_pct), na.rm = TRUE)
  
  ggplot(tbl2, aes(x = Model_wrapped, y = Uplift_AUC_pct, fill = Group)) +
    geom_col(width = 0.65) +
    geom_vline(xintercept = NULL) +
    geom_hline(yintercept = 0, linewidth = 0.7, color = "grey50") +
    geom_text(aes(label = sprintf("%+.2f%%", Uplift_AUC_pct)),
              hjust = ifelse(tbl2$Uplift_AUC_pct >= 0, -0.15, 1.15),
              size = 6, fontface = "bold") +
    coord_flip(clip = "off") +
    scale_fill_manual(
      name = "Performance Change",
      values = c(
        "Base Model"      = grey,
        "Positive Uplift" = blue,
        "Negative Uplift" = red
      )
    ) +
    scale_y_continuous(
      limits = c(-yabs - 0.1, yabs + 0.1),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(title = title, x = NULL, y = "Uplift (pp vs Base)") +
    theme_ilab_clean()
}

# ----------------------------
# 3) Table (NO Strategy D)
# ----------------------------
make_glm_table <- function(tbl, title = "GLM Training Model Leaderboard") {
  
  tbl2 <- tbl %>%
    filter(!grepl("Strategy D", Model)) %>%
    mutate(
      AUC = round(AUC, 4),
      Uplift_AUC_pct = round(Uplift_AUC_pct, 2),
      Brier_Score = round(Brier_Score, 4),
      Uplift_Brier_pct = round(Uplift_Brier_pct, 2),
      Penalized_Brier = round(Penalized_Brier, 4),
      Uplift_PBS_pct = round(Uplift_PBS_pct, 2),
      Alpha = signif(Alpha, 3),
      Lambda = signif(Lambda, 3)
    )
  
  gt_tbl <- tbl2 %>%
    gt() %>%
    tab_header(title = md(paste0("**", title, "**"))) %>%
    cols_label(
      Model = "Model",
      AUC = "AUC",
      Uplift_AUC_pct = "AUC Uplift (%)",
      Brier_Score = "Brier",
      Uplift_Brier_pct = "Brier Uplift (%)",
      Penalized_Brier = "Penalized Brier",
      Uplift_PBS_pct = "PBS Uplift (%)",
      Alpha = "Alpha",
      Lambda = "Lambda"
    ) %>%
    fmt_number(columns = c(AUC, Brier_Score, Penalized_Brier), decimals = 4) %>%
    fmt_number(columns = c(Uplift_AUC_pct, Uplift_Brier_pct, Uplift_PBS_pct), decimals = 2) %>%
    tab_style(
      style = cell_text(color = blue, weight = "bold"),
      locations = cells_title(groups = "title")
    ) %>%
    tab_style(
      style = cell_fill(color = "#F3F5F7"),
      locations = cells_body(rows = seq(1, nrow(tbl2), by = 2))
    ) %>%
    tab_style(
      style = list(cell_fill(color = "#E8EEF7"), cell_text(weight = "bold")),
      locations = cells_body(rows = grepl("^Base Model", tbl2$Model))
    ) %>%
    tab_style(
      style = cell_text(color = blue, weight = "bold"),
      locations = cells_body(columns = "Uplift_AUC_pct", rows = tbl2$Uplift_AUC_pct > 0)
    ) %>%
    tab_style(
      style = cell_text(color = red, weight = "bold"),
      locations = cells_body(columns = "Uplift_AUC_pct", rows = tbl2$Uplift_AUC_pct < 0)
    ) %>%
    tab_options(
      table.width = pct(100),
      heading.align = "left",
      data_row.padding = px(10),
      table.border.top.color = blue,
      table.border.bottom.color = blue
    )
  
  gt_tbl
}


p_auc    <- plot_glm_auc(comparison_table)
p_uplift <- plot_glm_uplift(comparison_table)
gt_tbl   <- make_glm_table(comparison_table)

# Show
gt_tbl
p_auc
p_uplift

path <- "/Users/admin/Desktop/Industry Lab/01_Code/Functions"

ggsave(
  filename = file.path(path, "GLM_AUC_Comparison.png"),
  plot     = p_auc,
  width    = width_in,
  height   = height_in,
  dpi      = dpi_out
)

ggsave(
  filename = file.path(path, "GLM_AUC_Uplift.png"),
  plot     = p_uplift,
  width    = width_in,
  height   = height_in,
  dpi      = dpi_out
)

gtsave(
  gt_tbl,
  filename = file.path(path, "GLM_Model_Leaderboard.png")
)
