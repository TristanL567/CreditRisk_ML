library(dplyr)
library(ggplot2)
library(scales)
library(gt)


blue <- "#004890"
grey <- "#708090"

width_px  <- 3750
height_px <- 1833
dpi_out   <- 300
width_in  <- width_px  / dpi_out
height_in <- height_px / dpi_out



plot_auc_leaderboard_train <- function(comparison_table) {
  
  df <- comparison_table %>%
    mutate(
      Model = factor(Model, levels = rev(Model)),
      auc_lbl = percent(AUC, accuracy = 0.1),
      uplift_lbl = ifelse(is.finite(Uplift_AUC_pct),
                          paste0(" (", sprintf("%+.1f", Uplift_AUC_pct), "%)"),
                          "")
    )
  
  ggplot(df, aes(x = Model, y = AUC)) +
    geom_col(fill = blue, width = 0.7) +
    geom_text(
      aes(label = paste0(auc_lbl, uplift_lbl)),
      hjust = -0.05,
      size = 5,
      fontface = "bold"
    ) +
    coord_flip() +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.15))
    ) +
    labs(
      title = "GLM Strategy Performance – Train Set",
      x = NULL,
      y = "AUC"
    ) +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(face = "bold", size = 28, hjust = 0),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank()
    )
}


p_auc_train <- plot_auc_leaderboard_train(comparison_table)
path <- "/Users/admin/Desktop/Industry Lab/01_Code/Functions"

ggsave(
  filename = file.path(path, "p_auc_train n.png"),
  plot     = p_auc_train ,
  width    = width_in,
  height   = height_in,
  dpi      = dpi_out
)


make_leaderboard_table_train <- function(comparison_table) {
  
  comparison_table %>%
    mutate(
      AUC = round(AUC, 4) * 100,
      Uplift_AUC_pct = round(Uplift_AUC_pct, 2),
      Brier_Score = round(Brier_Score, 4) * 100,
      Uplift_Brier_pct = round(Uplift_Brier_pct, 2),
      Penalized_Brier = round(Penalized_Brier, 4),
      Uplift_PBS_pct = round(Uplift_PBS_pct, 2),
      Alpha = round(Alpha, 2)
    ) %>%
    select(
      Model,
      Type,
      AUC,
      Uplift_AUC_pct,
      Brier_Score,
      Uplift_Brier_pct,
      Penalized_Brier,
      Uplift_PBS_pct,
      Alpha
    ) %>%
    gt() %>%
    tab_header(title = "GLM Strategy Leaderboard – Train Set") %>%
    cols_label(
      Model = "Strategy",
      Type  = "Type",
      AUC = "Train AUC (%)",
      Uplift_AUC_pct = "AUC Uplift (%)",
      Brier_Score = "Brier (%)",
      Uplift_Brier_pct = "Brier Uplift (%)",
      Penalized_Brier = "Penalized Brier",
      Uplift_PBS_pct = "PBS Uplift (%)",
      Alpha = "Alpha"
    ) %>%
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_column_labels(everything())
    ) %>%
    tab_style(
      style = cell_text(color = blue, weight = "bold"),
      locations = cells_title(groups = "title")
    ) %>%
    tab_options(
      table.font.size = px(16),
      heading.title.font.size = px(24),
      column_labels.font.size = px(16),
      data_row.padding = px(8),
      table.border.top.color = blue,
      table.border.bottom.color = blue
    )
}


path <- "/Users/admin/Desktop/Industry Lab/01_Code/Functions"
tbl_train <- make_leaderboard_table_train(comparison_table)
file_name <- file.path(path, "GLM_Leaderboard_Train.png")
gtsave(tbl_train, file_name)





#==============================================================================#
#==== 04B - TEST SET Visualisation (same style as Train) =======================#
#==============================================================================#

library(dplyr)
library(ggplot2)
library(scales)
library(gt)

blue <- "#004890"
grey <- "#708090"

width_px  <- 3750
height_px <- 1833
dpi_out   <- 300
width_in  <- width_px  / dpi_out
height_in <- height_px / dpi_out


#------------------------------------------------------------------------------#
# 0) FIX BASELINE + UPLIFTS (TEST)
#------------------------------------------------------------------------------#
comparison_table_test <- comparison_table_test %>%
  mutate(
    Base_AUC   = AUC[Model == "Base Model"][1],
    Base_Brier = Brier_Score[Model == "Base Model"][1],
    Base_PBS   = Penalized_Brier[Model == "Base Model"][1],
    
    Uplift_AUC_pct   = (AUC - Base_AUC) / Base_AUC * 100,
    Uplift_Brier_pct = - (Brier_Score - Base_Brier) / Base_Brier * 100,
    Uplift_PBS_pct   = - (Penalized_Brier - Base_PBS) / Base_PBS * 100
  )


#------------------------------------------------------------------------------#
# 1) AUC plot (TEST) – same as Train
#------------------------------------------------------------------------------#
plot_auc_leaderboard_test <- function(comparison_table_test) {
  
  df <- comparison_table_test %>%
    mutate(
      Model = factor(Model, levels = rev(Model)),
      auc_lbl = percent(AUC, accuracy = 0.1),
      uplift_lbl = ifelse(is.finite(Uplift_AUC_pct),
                          paste0(" (", sprintf("%+.1f", Uplift_AUC_pct), "%)"),
                          "")
    )
  
  ggplot(df, aes(x = Model, y = AUC)) +
    geom_col(fill = blue, width = 0.7) +
    geom_text(
      aes(label = paste0(auc_lbl, uplift_lbl)),
      hjust = -0.05,
      size = 5,
      fontface = "bold"
    ) +
    coord_flip() +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.22))  # extra space so labels never cut off
    ) +
    labs(
      title = "GLM Strategy Performance – Test Set",
      x = NULL,
      y = "AUC"
    ) +
    theme_minimal(base_size = 18) +
    theme(
      plot.title = element_text(face = "bold", size = 28, hjust = 0),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      plot.margin = margin(20, 40, 20, 20)
    )
}

p_auc_test <- plot_auc_leaderboard_test(comparison_table_test)

path <- "/Users/admin/Desktop/Industry Lab/01_Code/Functions"

ggsave(
  filename = file.path(path, "p_auc_test.png"),
  plot     = p_auc_test,
  width    = width_in,
  height   = height_in,
  dpi      = dpi_out
)


#------------------------------------------------------------------------------#
# 2) GT table (TEST) – same as Train table (no Lambda)
#------------------------------------------------------------------------------#
make_leaderboard_table_test <- function(comparison_table_test) {
  
  comparison_table_test %>%
    mutate(
      AUC = round(AUC, 4) * 100,
      Uplift_AUC_pct = round(Uplift_AUC_pct, 2),
      Brier_Score = round(Brier_Score, 4) * 100,
      Uplift_Brier_pct = round(Uplift_Brier_pct, 2),
      Penalized_Brier = round(Penalized_Brier, 4),
      Uplift_PBS_pct = round(Uplift_PBS_pct, 2),
      Alpha = round(Alpha, 2)
    ) %>%
    select(
      Model,
      Type,
      AUC,
      Uplift_AUC_pct,
      Brier_Score,
      Uplift_Brier_pct,
      Penalized_Brier,
      Uplift_PBS_pct,
      Alpha
    ) %>%
    gt() %>%
    tab_header(title = "GLM Strategy Leaderboard – Test Set") %>%
    cols_label(
      Model = "Strategy",
      Type  = "Type",
      AUC = "Test AUC (%)",
      Uplift_AUC_pct = "AUC Uplift (%)",
      Brier_Score = "Brier (%)",
      Uplift_Brier_pct = "Brier Uplift (%)",
      Penalized_Brier = "Penalized Brier",
      Uplift_PBS_pct = "PBS Uplift (%)",
      Alpha = "Alpha"
    ) %>%
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_column_labels(everything())
    ) %>%
    tab_style(
      style = cell_text(color = blue, weight = "bold"),
      locations = cells_title(groups = "title")
    ) %>%
    tab_options(
      table.font.size = px(16),
      heading.title.font.size = px(24),
      column_labels.font.size = px(16),
      data_row.padding = px(8),
      table.border.top.color = blue,
      table.border.bottom.color = blue
    )
}


tbl_test <- make_leaderboard_table_test(comparison_table_test)
file_name <- file.path(path, "GLM_Leaderboard_Test.png")
gtsave(tbl_test, file_name)
