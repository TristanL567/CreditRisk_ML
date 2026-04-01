#==============================================================================#
#==== Validate_All_OoS_Models.R ===============================================#
#==============================================================================#
#
# PURPOSE:  Validate ALL 15 OoS model predictions with diagnostic analyses
# INPUT:    Prediction files from all model folders
# OUTPUT:   Comprehensive validation plots and summary Excel
#
# MODELS:   01a, 02a, 03a, 04a, 05a × AutoGluon, GLM, XGBoost_Manual
#
#==============================================================================#


#==============================================================================#
#==== 0 - CONFIGURATION ======================================================#
#==============================================================================#

## Base directory containing all model folders
BASE_DIR <- "/Users/admin/Desktop/Final"

## Output directory for validation plots
OUTPUT_DIR <- "//Users/admin/Desktop/CreditRisk_ML/03_Charts/Validation_Plots_All"

## Create output directory
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

## Define OoS models (a = OoS, b = OoT - we skip b)
MODEL_IDS <- c("01a", "02a", "03a", "04a", "05a")
ALGORITHMS <- c("AutoGluon", "GLM", "XGBoost_Manual")

## Feature set names
FEATURE_SETS <- c(
  "01a" = "Raw Balance Sheet",
  "02a" = "Financial Ratios", 
  "03a" = "Ratios + Time Dynamics",
  "04a" = "Ratios + TD + VAE",
  "05a" = "VAE Latent Only"
)


#==============================================================================#
#==== 1 - LOAD PACKAGES ======================================================#
#==============================================================================#

message("\n", strrep("=", 70))
message("  COMPREHENSIVE OoS MODEL VALIDATION")
message(strrep("=", 70))
message("\nLoading packages...")

packages <- c("data.table", "ggplot2", "pROC", "scales", "gridExtra", 
              "arrow", "openxlsx", "patchwork")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

## OeNB colors
oenb_blue <- "#004890"
oenb_blue_light <- "#7FAFD4"
oenb_teal <- "#1D9E75"
oenb_teal_light <- "#9FE1CB"
oenb_coral <- "#D85A30"
oenb_coral_light <- "#F0997B"
oenb_purple <- "#7F77DD"
oenb_gray <- "#888780"

## Algorithm colors
algo_colors <- c(
  "AutoGluon" = oenb_teal,
  "GLM" = oenb_coral,
  "XGBoost_Manual" = oenb_blue
)


#==============================================================================#
#==== 2 - LOAD ALL PREDICTIONS ===============================================#
#==============================================================================#

message("\nLoading predictions from all OoS models...")

load_predictions <- function(base_dir, model_id, algorithm) {
  
  folder <- file.path(base_dir, paste0(model_id, "_", algorithm))
  
  if (!dir.exists(folder)) {
    message(sprintf("  [SKIP] Folder not found: %s", folder))
    return(NULL)
  }
  
  ## Determine file path based on algorithm
  if (algorithm == "AutoGluon") {
    pred_path <- file.path(folder, "predictions_test.parquet")
    if (!file.exists(pred_path)) {
      message(sprintf("  [SKIP] File not found: %s", pred_path))
      return(NULL)
    }
    dt <- tryCatch(as.data.table(arrow::read_parquet(pred_path)), 
                   error = function(e) NULL)
    
  } else if (algorithm == "GLM") {
    pred_path <- file.path(folder, "predictions_test_GLM_v2_OoS.parquet")
    if (!file.exists(pred_path)) {
      message(sprintf("  [SKIP] File not found: %s", pred_path))
      return(NULL)
    }
    dt <- tryCatch(as.data.table(arrow::read_parquet(pred_path)),
                   error = function(e) NULL)
    
  } else if (algorithm == "XGBoost_Manual") {
    pred_path <- file.path(folder, "predictions_test.rds")
    if (!file.exists(pred_path)) {
      message(sprintf("  [SKIP] File not found: %s", pred_path))
      return(NULL)
    }
    dt <- tryCatch(as.data.table(readRDS(pred_path)),
                   error = function(e) NULL)
  }
  
  if (is.null(dt)) {
    message(sprintf("  [ERROR] Failed to load: %s_%s", model_id, algorithm))
    return(NULL)
  }
  
  ## Standardize column names
  if ("p_csi" %in% names(dt) && !"p_default" %in% names(dt)) {
    setnames(dt, "p_csi", "p_default")
  }
  
  ## Add metadata
  dt[, `:=`(
    model_id = model_id,
    algorithm = algorithm,
    model_name = paste0(model_id, "_", algorithm),
    feature_set = FEATURE_SETS[model_id]
  )]
  
  message(sprintf("  [OK] %s_%s: %d rows, %d firms, %.2f%% default rate",
                  model_id, algorithm, nrow(dt), uniqueN(dt$id), 
                  100 * mean(dt$y)))
  
  return(dt)
}

## Load all models
all_preds <- list()

for (mid in MODEL_IDS) {
  for (algo in ALGORITHMS) {
    key <- paste0(mid, "_", algo)
    pred <- load_predictions(BASE_DIR, mid, algo)
    if (!is.null(pred)) {
      all_preds[[key]] <- pred
    }
  }
}

message(sprintf("\n  ✓ Loaded %d of 15 OoS models", length(all_preds)))

## Combine all predictions
combined_dt <- rbindlist(all_preds, fill = TRUE, ignore.attr = TRUE)


#==============================================================================#
#==== 3 - COMPUTE METRICS FOR ALL MODELS =====================================#
#==============================================================================#

message("\nComputing validation metrics...")

compute_model_metrics <- function(dt) {
  
  y <- dt$y
  p <- dt$p_default
  
  ## ROC and AUC
  roc_obj <- tryCatch(roc(y, p, quiet = TRUE, direction = "<"),
                      error = function(e) NULL)
  
  if (is.null(roc_obj)) return(NULL)
  
  auc_val <- as.numeric(auc(roc_obj))
  
  ## Optimal threshold (Youden)
  opt <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"),
                best.method = "youden")
  
  ## K-S statistic
  ks_test <- ks.test(p[y == 1], p[y == 0])
  
  ## Calibration metrics
  brier <- mean((p - y)^2)
  brier_ref <- mean(y) * (1 - mean(y))
  bss <- 1 - brier / brier_ref
  
  ## Score statistics
  score_mean_default <- mean(p[y == 1])
  score_mean_nondefault <- mean(p[y == 0])
  score_separation <- score_mean_default - score_mean_nondefault
  
  list(
    n_obs = nrow(dt),
    n_firms = uniqueN(dt$id),
    n_defaults = sum(y),
    default_rate = mean(y),
    auc = auc_val,
    gini = 2 * auc_val - 1,
    ks_stat = as.numeric(ks_test$statistic),
    brier = brier,
    bss = bss,
    threshold_opt = opt$threshold,
    sensitivity = opt$sensitivity,
    specificity = opt$specificity,
    score_mean_default = score_mean_default,
    score_mean_nondefault = score_mean_nondefault,
    score_separation = score_separation
  )
}

## Compute metrics for each model
metrics_list <- list()

for (model_name in names(all_preds)) {
  dt <- all_preds[[model_name]]
  metrics <- compute_model_metrics(dt)
  
  if (!is.null(metrics)) {
    metrics$model_name <- model_name
    metrics$model_id <- dt$model_id[1]
    metrics$algorithm <- dt$algorithm[1]
    metrics$feature_set <- dt$feature_set[1]
    metrics_list[[model_name]] <- metrics
  }
}

metrics_dt <- rbindlist(lapply(metrics_list, as.data.table))
setorder(metrics_dt, -auc)

message("\n  Model Performance Summary:")
message(sprintf("  %-25s %-25s %8s %8s %8s", 
                "Model", "Feature Set", "AUC", "K-S", "BSS"))
message("  ", strrep("-", 80))
for (i in 1:nrow(metrics_dt)) {
  r <- metrics_dt[i]
  message(sprintf("  %-25s %-25s %8.4f %8.4f %8.4f",
                  r$model_name, r$feature_set, r$auc, r$ks_stat, r$bss))
}


#==============================================================================#
#==== 4 - PLOT 1: AUC COMPARISON ALL MODELS ==================================#
#==============================================================================#

message("\nCreating validation plots...")

p1 <- ggplot(metrics_dt, aes(x = reorder(model_name, auc), y = auc, color = algorithm)) +
  # Create the "stick" of the lollipop
  geom_segment(aes(xend = reorder(model_name, auc), y = 0.75, yend = auc), 
               color = "gray80", linewidth = 1) +
  # Add the data point (the "pop")
  geom_point(size = 4) + 
  # Bold text labels with a slight offset
  geom_text(aes(label = sprintf("%.3f", auc)), hjust = -0.4, size = 3.5, fontface = "bold") +
  scale_color_manual(values = algo_colors) +
  # Expanded limits to 0.96 so labels are fully visible
  scale_y_continuous(limits = c(0.75, 0.96), labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title = "AUC-ROC Comparison — All OoS Models",
    subtitle = "Points indicate precise AUC value",
    x = NULL,
    y = "AUC-ROC",
    color = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(OUTPUT_DIR, "01_AUC_All_Models.png"), p1, 
       width = 12, height = 8, dpi = 300, bg = "white")


#==============================================================================#
#==== 5 - PLOT 2: K-S STATISTIC COMPARISON ===================================#
#==============================================================================#

p2 <- ggplot(metrics_dt, aes(x = reorder(model_name, ks_stat), y = ks_stat, color = algorithm)) +
  geom_segment(aes(xend = reorder(model_name, ks_stat), y = 0, yend = ks_stat), 
               color = "gray80", linewidth = 1) +
  geom_point(size = 4) +
  geom_text(aes(label = sprintf("%.3f", ks_stat)), hjust = -0.4, size = 3.5, fontface = "bold") +
  scale_color_manual(values = algo_colors) +
  # Expanded limit to 0.78 to prevent label clipping
  scale_y_continuous(limits = c(0, 0.78)) +
  coord_flip() +
  labs(
    title = "K-S Statistic — Class Separation",
    subtitle = "Higher K-S indicates better separation; points show exact statistic",
    x = NULL,
    y = "Kolmogorov-Smirnov Statistic",
    color = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(OUTPUT_DIR, "02_KS_All_Models.png"), p2, 
       width = 12, height = 8, dpi = 300, bg = "white")

#==============================================================================#
#==== 6 - PLOT 3: CALIBRATION (BSS) COMPARISON ===============================#
#==============================================================================#

p3 <- ggplot(metrics_dt, aes(x = reorder(model_name, bss), y = bss, fill = algorithm)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_text(aes(label = sprintf("%.2f", bss), 
                hjust = ifelse(bss >= 0, -0.1, 1.1)), size = 3) +
  scale_fill_manual(values = algo_colors) +
  coord_flip() +
  labs(
    title = "Brier Skill Score — Calibration Quality",
    subtitle = "Positive = better than naive; Negative = worse than naive (miscalibrated)",
    x = NULL,
    y = "Brier Skill Score",
    fill = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom"
  )

ggsave(file.path(OUTPUT_DIR, "03_BSS_All_Models.png"), p3, 
       width = 12, height = 8, dpi = 300, bg = "white")


#==============================================================================#
#==== 7 - PLOT 4: SCORE SEPARATION BY MODEL ==================================#
#==============================================================================#

p4 <- ggplot(metrics_dt, aes(x = reorder(model_name, score_separation))) +
  geom_segment(aes(xend = model_name, y = score_mean_nondefault, yend = score_mean_default),
               color = "gray70", linewidth = 1) +
  geom_point(aes(y = score_mean_nondefault), color = oenb_blue, size = 4) +
  geom_point(aes(y = score_mean_default), color = oenb_coral, size = 4) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title = "Score Separation — Mean Score by Class",
    subtitle = "Blue = Non-Default mean, Coral = Default mean. Larger gap = better discrimination",
    x = NULL,
    y = "Mean Predicted Probability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14)
  )

ggsave(file.path(OUTPUT_DIR, "04_Score_Separation.png"), p4, 
       width = 12, height = 8, dpi = 300, bg = "white")


#==============================================================================#
#==== 8 - PLOT 5: FACETED CALIBRATION PLOTS ==================================#
#==============================================================================#

## Compute calibration data for all models
calib_data <- rbindlist(lapply(names(all_preds), function(model_name) {
  dt <- copy(all_preds[[model_name]])
  
  dt[, decile := cut(p_default, 
                     breaks = quantile(p_default, probs = seq(0, 1, 0.1), na.rm = TRUE),
                     include.lowest = TRUE, labels = 1:10)]
  
  calib <- dt[, .(
    n = .N,
    pred_mean = mean(p_default),
    obs_rate = mean(y)
  ), by = decile]
  
  calib[, model_name := model_name]
  calib[, algorithm := dt$algorithm[1]]
  calib[, feature_set := dt$feature_set[1]]
  
  return(calib)
}))

p5 <- ggplot(calib_data, aes(x = pred_mean, y = obs_rate)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  geom_line(aes(color = algorithm), linewidth = 0.8) +
  geom_point(aes(color = algorithm, size = n)) +
  scale_color_manual(values = algo_colors) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
  scale_size_continuous(range = c(1, 4), guide = "none") +
  facet_wrap(~ feature_set, scales = "free", ncol = 3) +
  labs(
    title = "Calibration by Feature Set",
    subtitle = "Points on diagonal = well-calibrated. XGBoost tends to overpredict.",
    x = "Mean Predicted Probability",
    y = "Observed Default Rate",
    color = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(OUTPUT_DIR, "05_Calibration_Faceted.png"), p5, 
       width = 14, height = 10, dpi = 300, bg = "white")


#==============================================================================#
#==== 9 - PLOT 6: FACETED SCORE DISTRIBUTIONS ================================#
#==============================================================================#

## Sample for plotting (too many points otherwise)
set.seed(42)
sample_dt <- combined_dt[, .SD[sample(.N, min(.N, 5000))], by = model_name]
sample_dt[, class := factor(y, levels = c(0, 1), labels = c("Non-Default", "Default"))]

p6 <- ggplot(sample_dt, aes(x = p_default, fill = class)) +
  geom_density(alpha = 0.6, color = NA) +
  scale_fill_manual(values = c("Non-Default" = oenb_blue, "Default" = oenb_coral)) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  facet_wrap(~ model_name, scales = "free_y", ncol = 5) +
  labs(
    title = "Score Distribution by Model",
    subtitle = "Clear separation between blue (non-default) and coral (default) indicates good discrimination",
    x = "Predicted Default Probability",
    y = "Density",
    fill = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    strip.text = element_text(size = 8),
    axis.text = element_text(size = 7)
  )

ggsave(file.path(OUTPUT_DIR, "06_Score_Distributions_All.png"), p6, 
       width = 16, height = 10, dpi = 300, bg = "white")


#==============================================================================#
#==== 10 - PLOT 7: ROC CURVES OVERLAY ========================================#
#==============================================================================#

## Compute ROC curves for all models
roc_data <- rbindlist(lapply(names(all_preds), function(model_name) {
  dt <- all_preds[[model_name]]
  roc_obj <- roc(dt$y, dt$p_default, quiet = TRUE)
  
  coords_dt <- as.data.table(coords(roc_obj, "all", ret = c("threshold", "sensitivity", "specificity")))
  coords_dt[, fpr := 1 - specificity]
  coords_dt[, model_name := model_name]
  coords_dt[, algorithm := dt$algorithm[1]]
  coords_dt[, feature_set := dt$feature_set[1]]
  
  return(coords_dt)
}))

## Plot by feature set
p7 <- ggplot(roc_data, aes(x = fpr, y = sensitivity, color = algorithm)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray60") +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = algo_colors) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  facet_wrap(~ feature_set, ncol = 3) +
  labs(
    title = "ROC Curves by Feature Set",
    subtitle = "Curves further from diagonal indicate better discrimination",
    x = "False Positive Rate",
    y = "True Positive Rate (Sensitivity)",
    color = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

ggsave(file.path(OUTPUT_DIR, "07_ROC_Curves_Faceted.png"), p7, 
       width = 14, height = 10, dpi = 300, bg = "white")


#==============================================================================#
#==== 11 - PLOT 8: ALGORITHM COMPARISON BY FEATURE SET =======================#
#==============================================================================#

p8 <- ggplot(metrics_dt, aes(x = feature_set, y = auc, fill = algorithm)) +
  # Keep bars but make them slightly transparent
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.7) +
  # Add explicit data points on top of bars
  geom_point(aes(color = algorithm), 
             position = position_dodge(width = 0.8), size = 3, show.legend = FALSE) +
  # Add text labels
  geom_text(aes(label = sprintf("%.3f", auc)), 
            position = position_dodge(width = 0.8), vjust = -1.2, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = algo_colors) +
  scale_color_manual(values = algo_colors) +
  # Expanded limit to 0.96 for label headroom
  scale_y_continuous(limits = c(0.75, 0.96), labels = percent_format(accuracy = 1)) +
  labs(
    title = "AUC by Feature Set and Algorithm",
    subtitle = "Visible points confirm exact performance values across groups",
    x = NULL,
    y = "AUC-ROC",
    fill = "Algorithm"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
    legend.position = "bottom",
    axis.text.x = element_text(angle = 20, hjust = 1, face = "bold")
  )

ggsave(file.path(OUTPUT_DIR, "08_AUC_by_Feature_Algorithm.png"), p8, 
       width = 12, height = 7, dpi = 300, bg = "white")


#==============================================================================#
#==== 12 - PLOT 9: TRAJECTORY ANALYSIS (BEST MODEL) ==========================#
#==============================================================================#

## Use best model for trajectory analysis
best_model <- metrics_dt$model_name[1]
best_dt <- all_preds[[best_model]]

## Trajectory for defaulting firms
default_firms <- unique(best_dt[y == 1, id])
traj_dt <- best_dt[id %in% default_firms]
traj_dt[, obs_num := seq_len(.N), by = id]
traj_dt[, total_obs := .N, by = id]
traj_dt[, obs_from_end := total_obs - obs_num + 1]

avg_traj <- traj_dt[, .(
  mean_p = mean(p_default),
  se = sd(p_default) / sqrt(.N),
  n_firms = uniqueN(id)
), by = obs_from_end][order(-obs_from_end)]

if (nrow(avg_traj) > 1) {
  p9 <- ggplot(avg_traj[obs_from_end <= 10], aes(x = -obs_from_end, y = mean_p)) +
    geom_ribbon(aes(ymin = mean_p - 1.96*se, ymax = mean_p + 1.96*se), 
                alpha = 0.2, fill = oenb_coral) +
    geom_line(color = oenb_coral, linewidth = 1.2) +
    geom_point(aes(size = n_firms), color = oenb_coral) +
    scale_x_continuous(breaks = seq(-10, 0, 2),
                       labels = c("T-10", "T-8", "T-6", "T-4", "T-2", "Default")) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_size_continuous(range = c(2, 8)) +
    labs(
      title = sprintf("Score Trajectory Before Default — %s", best_model),
      subtitle = "Rising scores before default indicate early warning capability",
      x = "Time Relative to Default",
      y = "Mean Predicted Probability",
      size = "N Firms"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", color = oenb_blue, size = 14),
      legend.position = "right"
    )
  
  ggsave(file.path(OUTPUT_DIR, "09_Trajectory_Best_Model.png"), p9, 
         width = 10, height = 6, dpi = 300, bg = "white")
}


#==============================================================================#
#==== 13 - EXPORT METRICS TO EXCEL ===========================================#
#==============================================================================#

message("\nExporting metrics to Excel...")

wb <- createWorkbook()

## Sheet 1: Summary metrics
addWorksheet(wb, "Validation_Summary")
writeData(wb, "Validation_Summary", metrics_dt)

header_style <- createStyle(fontColour = "#FFFFFF", fgFill = oenb_blue,
                            halign = "CENTER", textDecoration = "Bold")
addStyle(wb, "Validation_Summary", header_style, rows = 1, cols = 1:ncol(metrics_dt))
setColWidths(wb, "Validation_Summary", cols = 1:ncol(metrics_dt), widths = "auto")
freezePane(wb, "Validation_Summary", firstRow = TRUE)

## Sheet 2: Calibration data
addWorksheet(wb, "Calibration_Data")
writeData(wb, "Calibration_Data", calib_data)
addStyle(wb, "Calibration_Data", header_style, rows = 1, cols = 1:ncol(calib_data))

## Save
saveWorkbook(wb, file.path(OUTPUT_DIR, "Validation_Metrics_All_OoS.xlsx"), overwrite = TRUE)


#==============================================================================#
#==== 14 - SUMMARY ===========================================================#
#==============================================================================#

message("\n")
message(strrep("=", 70))
message("  VALIDATION COMPLETE")
message(strrep("=", 70))
message(sprintf("\n  Models validated: %d", length(all_preds)))
message(sprintf("  Output directory: %s", OUTPUT_DIR))
message("\n  Plots generated:")
message("    • 01_AUC_All_Models.png")
message("    • 02_KS_All_Models.png")
message("    • 03_BSS_All_Models.png")
message("    • 04_Score_Separation.png")
message("    • 05_Calibration_Faceted.png")
message("    • 06_Score_Distributions_All.png")
message("    • 07_ROC_Curves_Faceted.png")
message("    • 08_AUC_by_Feature_Algorithm.png")
message("    • 09_Trajectory_Best_Model.png")
message("    • Validation_Metrics_All_OoS.xlsx")
message("\n")

## Print key findings
message("  KEY FINDINGS:")
message(sprintf("    • Best model: %s (AUC = %.4f)", 
                metrics_dt$model_name[1], metrics_dt$auc[1]))
message(sprintf("    • Best calibrated: %s (BSS = %.4f)", 
                metrics_dt[which.max(bss), model_name], max(metrics_dt$bss)))
message(sprintf("    • Best separation: %s (K-S = %.4f)",
                metrics_dt[which.max(ks_stat), model_name], max(metrics_dt$ks_stat)))

## Check which algorithms are missing
missing_models <- setdiff(
  paste0(rep(MODEL_IDS, each = 3), "_", rep(ALGORITHMS, 5)),
  names(all_preds)
)
if (length(missing_models) > 0) {
  message("\n  MISSING MODELS:")
  for (m in missing_models) {
    message(sprintf("    • %s", m))
  }
}

message("\n")
