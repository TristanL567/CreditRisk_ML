#==============================================================================#
#==== 06_Evaluate.R ===========================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Load all available prediction files, compute metrics, run DeLong pairwise
#   AUC tests, and save a combined leaderboard as .rds / .xlsx / .tex.
#
#   Gracefully skips missing files — only models that have been run are
#   included. OoS and OoT are evaluated separately (no cross-split comparison).
#
# INPUTS (auto-detected from disk):
#   AutoGluon : 03_Output/AutoGluon/{M}_{split}/predictions_test.parquet
#   XGBoost   : 03_Output/Models/XGBoost/{M}_{split}/predictions_test.rds
#
# OUTPUTS (→ 03_Output/Evaluation/{split}/):
#   leaderboard_{split}.rds          metrics table (R object for 07_Charts)
#   leaderboard_{split}.xlsx         formatted Excel
#   leaderboard_{split}.tex          LaTeX booktabs table
#   delong_pairwise_{split}.rds      full pairwise DeLong matrix
#   delong_vs_m1_{split}.rds         each model vs M1 baseline (same framework)
#   predictions_combined_{split}.rds all predictions stacked (for charts)
#
#==============================================================================#


#==============================================================================#
#==== 0 - Configuration  ← CHANGE HERE =======================================#
#==============================================================================#

## Must match the split used when running 02_, 03_, 04B_, 05_
## SPLIT_MODE is already set in config.R — override here only if needed.
## SPLIT_MODE <- "OoS"

message(sprintf("\n══ 06_Evaluate [SPLIT_MODE = %s] ══", SPLIT_MODE))


#==============================================================================#
#==== 1 - Paths & Directories =================================================#
#==============================================================================#

DIR_AG    <- file.path(PATH_ROOT, "03_Output", "AutoGluon")
DIR_XGB   <- file.path(PATH_ROOT, "03_Output", "Models", "XGBoost")
DIR_EVAL  <- file.path(PATH_ROOT, "03_Output", "Evaluation", SPLIT_MODE)

dir.create(DIR_EVAL, recursive = TRUE, showWarnings = FALSE)

MODELS     <- c("M1", "M2", "M3", "M4")
FRAMEWORKS <- c("AutoGluon", "XGBoost", "GLM")


#==============================================================================#
#==== 2 - Load Predictions ====================================================#
#==============================================================================#

message("  Loading prediction files...")

load_predictions <- function(model, framework, split_mode) {
  
  if (framework == "AutoGluon") {
    path <- file.path(DIR_AG,
                      sprintf("%s_%s", model, split_mode),
                      "predictions_test.parquet")
    if (!file.exists(path)) return(NULL)
    dt <- as.data.table(arrow::read_parquet(path))
    
  } else {
    path <- file.path(DIR_XGB,
                      sprintf("%s_%s", model, split_mode),
                      "predictions_test.rds")
    if (!file.exists(path)) return(NULL)
    dt <- as.data.table(readRDS(path))
  }
  
  ## Standardise column names
  if (!"p_default" %in% names(dt) && "p_csi" %in% names(dt))
    setnames(dt, "p_csi", "p_default")
  
  dt[, `:=`(
    model     = model,
    framework = framework,
    run_id    = sprintf("%s_%s_%s", framework, model, split_mode)
  )]
  
  message(sprintf("  %-12s %-10s : %d rows | loaded from %s",
                  framework, model, nrow(dt), basename(dirname(path))))
  dt
}

pred_list <- list()
for (fw in FRAMEWORKS) {
  for (m in MODELS) {
    dt <- load_predictions(m, fw, SPLIT_MODE)
    if (!is.null(dt)) pred_list[[sprintf("%s_%s", fw, m)]] <- dt
  }
}

if (length(pred_list) == 0L)
  stop(sprintf(
    "No prediction files found for SPLIT_MODE = '%s'.\n",
    "Run 04B_Train_XGBoost.R and/or 05_AutoGluon.py first.", SPLIT_MODE
  ))

predictions_combined <- rbindlist(pred_list, fill = TRUE)
message(sprintf("\n  Loaded %d model × framework combinations",
                length(pred_list)))
message(sprintf("  Combined predictions: %d rows", nrow(predictions_combined)))


#==============================================================================#
#==== 3 - Metric Functions ====================================================#
#==============================================================================#

recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj  <- pROC::roc(y_true, y_pred, quiet = TRUE)
  co       <- pROC::coords(roc_obj, "all",
                           ret = c("fpr", "tpr"), transpose = FALSE)
  eligible <- co$fpr <= fpr_target
  if (!any(eligible)) return(0)
  max(co$tpr[eligible])
}

avg_precision <- function(y_true, y_pred) {
  ## Trapezoidal average precision via PRROC
  if (!requireNamespace("PRROC", quietly = TRUE))
    return(NA_real_)
  pos <- y_pred[y_true == 1L]
  neg <- y_pred[y_true == 0L]
  if (length(pos) == 0L || length(neg) == 0L) return(NA_real_)
  round(PRROC::pr.curve(pos, neg)$auc.integral, 4)
}

brier_skill_score <- function(y_true, y_pred) {
  bs      <- mean((y_pred - y_true)^2)
  bs_clim <- mean(y_true) * (1 - mean(y_true))
  if (bs_clim == 0) return(NA_real_)
  round(1 - bs / bs_clim, 4)
}

compute_metrics <- function(y_true, y_pred, model, framework) {
  valid <- !is.na(y_true) & !is.na(y_pred)
  yt <- y_true[valid]; yp <- y_pred[valid]
  if (length(unique(yt)) < 2L) return(NULL)
  
  roc_obj   <- pROC::roc(yt, yp, quiet = TRUE)
  auc_val   <- as.numeric(pROC::auc(roc_obj))
  brier_val <- mean((yp - yt)^2)
  
  opt <- pROC::coords(roc_obj, "best",
                      ret = c("threshold", "sensitivity", "specificity"),
                      best.method = "youden")
  threshold <- opt$threshold[1L]
  pc        <- as.integer(yp >= threshold)
  tp <- sum(yt == 1L & pc == 1L); fp <- sum(yt == 0L & pc == 1L)
  fn <- sum(yt == 1L & pc == 0L)
  prec <- tp / max(tp + fp, 1L)
  rec  <- tp / max(tp + fn, 1L)
  f1   <- if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0
  
  data.table(
    framework      = framework,
    model          = model,
    split_mode     = SPLIT_MODE,
    run_id         = sprintf("%s_%s_%s", framework, model, SPLIT_MODE),
    n_obs          = length(yt),
    n_defaults     = as.integer(sum(yt)),
    prevalence     = round(mean(yt), 5),
    auc_roc        = round(auc_val, 4),
    avg_precision  = avg_precision(yt, yp),
    brier          = round(brier_val, 5),
    bss            = brier_skill_score(yt, yp),
    recall_fpr1    = round(recall_at_fpr(yt, yp, 0.01), 4),
    recall_fpr3    = round(recall_at_fpr(yt, yp, 0.03), 4),
    recall_fpr5    = round(recall_at_fpr(yt, yp, 0.05), 4),
    recall_fpr10   = round(recall_at_fpr(yt, yp, 0.10), 4),
    threshold_opt  = round(threshold, 4),
    sensitivity    = round(opt$sensitivity[1L], 4),
    specificity    = round(opt$specificity[1L], 4),
    precision      = round(prec, 4),
    f1             = round(f1, 4)
  )
}


#==============================================================================#
#==== 4 - Compute Metrics =====================================================#
#==============================================================================#

message("\n  Computing metrics...")

metrics_list <- lapply(pred_list, function(dt) {
  compute_metrics(
    y_true    = dt$y,
    y_pred    = dt$p_default,
    model     = dt$model[1L],
    framework = dt$framework[1L]
  )
})

leaderboard <- rbindlist(Filter(Negate(is.null), metrics_list))
setorder(leaderboard, -auc_roc)

## AUC uplift vs M1 within each framework
for (fw in unique(leaderboard$framework)) {
  m1_auc <- leaderboard[framework == fw & model == "M1", auc_roc]
  if (length(m1_auc) == 1L)
    leaderboard[framework == fw,
                uplift_auc_vs_m1_pct := round(
                  (auc_roc - m1_auc) / m1_auc * 100, 3)]
}

message("\n══ Leaderboard ════════════════════════════════════════════════")
print(leaderboard[, .(framework, model, auc_roc, avg_precision,
                      brier, bss, recall_fpr3, recall_fpr5,
                      uplift_auc_vs_m1_pct)],
      row.names = FALSE)


#==============================================================================#
#==== 5 - DeLong Tests ========================================================#
#==============================================================================#

message("\n  Running DeLong pairwise AUC tests...")

## Build named ROC object list for all available model × framework combos
roc_list <- lapply(pred_list, function(dt) {
  pROC::roc(dt$y, dt$p_default, quiet = TRUE)
})
run_ids <- names(roc_list)

## ── Full pairwise matrix ───────────────────────────────────────────────────
n_runs <- length(run_ids)
delong_matrix <- data.table(
  run_a    = character(),
  run_b    = character(),
  auc_a    = numeric(),
  auc_b    = numeric(),
  auc_diff = numeric(),
  ci_lo    = numeric(),
  ci_hi    = numeric(),
  p_value  = numeric()
)

for (i in seq_len(n_runs - 1L)) {
  for (j in (i + 1L):n_runs) {
    tryCatch({
      test_res <- pROC::roc.test(
        roc_list[[i]], roc_list[[j]],
        method = "delong", paired = FALSE
      )
      ci <- as.numeric(test_res$conf.int)
      delong_matrix <- rbind(delong_matrix, data.table(
        run_a    = run_ids[i],
        run_b    = run_ids[j],
        auc_a    = as.numeric(pROC::auc(roc_list[[i]])),
        auc_b    = as.numeric(pROC::auc(roc_list[[j]])),
        auc_diff = as.numeric(pROC::auc(roc_list[[i]])) -
          as.numeric(pROC::auc(roc_list[[j]])),
        ci_lo    = ci[1L],
        ci_hi    = ci[2L],
        p_value  = round(test_res$p.value, 5)
      ))
    }, error = function(e)
      message(sprintf("  DeLong failed: %s vs %s — %s",
                      run_ids[i], run_ids[j], e$message))
    )
  }
}

delong_matrix[, significant := p_value < 0.05]
message(sprintf("  Pairwise tests computed: %d pairs", nrow(delong_matrix)))

## ── Each model vs M1 within same framework ────────────────────────────────
delong_vs_m1 <- rbindlist(lapply(FRAMEWORKS, function(fw) {
  baseline_key <- sprintf("%s_M1", fw)
  if (!baseline_key %in% run_ids) return(NULL)
  
  rbindlist(lapply(MODELS[-1L], function(m) {
    target_key <- sprintf("%s_%s", fw, m)
    if (!target_key %in% run_ids) return(NULL)
    
    tryCatch({
      test_res <- pROC::roc.test(
        roc_list[[baseline_key]], roc_list[[target_key]],
        method = "delong", paired = FALSE
      )
      ci <- as.numeric(test_res$conf.int)
      data.table(
        framework    = fw,
        baseline     = "M1",
        comparison   = m,
        auc_baseline = as.numeric(pROC::auc(roc_list[[baseline_key]])),
        auc_comp     = as.numeric(pROC::auc(roc_list[[target_key]])),
        auc_diff     = as.numeric(pROC::auc(roc_list[[baseline_key]])) -
          as.numeric(pROC::auc(roc_list[[target_key]])),
        ci_lo        = ci[1L],
        ci_hi        = ci[2L],
        p_value      = round(test_res$p.value, 5),
        significant  = test_res$p.value < 0.05
      )
    }, error = function(e) {
      message(sprintf("  DeLong vs M1 failed: %s %s — %s", fw, m, e$message))
      NULL
    })
  }), fill = TRUE)
}), fill = TRUE)

if (!is.null(delong_vs_m1) && nrow(delong_vs_m1) > 0L) {
  message("\n  DeLong tests vs M1 baseline:")
  print(delong_vs_m1[, .(framework, comparison, auc_baseline, auc_comp,
                         auc_diff, p_value, significant)],
        row.names = FALSE)
}


#==============================================================================#
#==== 6 - Save Outputs ========================================================#
#==============================================================================#

message(sprintf("\n  Saving outputs → %s", DIR_EVAL))

## ── .rds ──────────────────────────────────────────────────────────────────
saveRDS(leaderboard,
        file.path(DIR_EVAL, sprintf("leaderboard_%s.rds", SPLIT_MODE)))
saveRDS(delong_matrix,
        file.path(DIR_EVAL, sprintf("delong_pairwise_%s.rds", SPLIT_MODE)))
saveRDS(delong_vs_m1,
        file.path(DIR_EVAL, sprintf("delong_vs_m1_%s.rds", SPLIT_MODE)))
saveRDS(predictions_combined,
        file.path(DIR_EVAL, sprintf("predictions_combined_%s.rds", SPLIT_MODE)))

message("  leaderboard .rds saved")
message("  delong_pairwise .rds saved")
message("  delong_vs_m1 .rds saved")
message("  predictions_combined .rds saved")

## ── Excel ─────────────────────────────────────────────────────────────────
wb <- openxlsx::createWorkbook()

## Sheet 1: Leaderboard
openxlsx::addWorksheet(wb, "Leaderboard")
openxlsx::writeData(wb, "Leaderboard", leaderboard)

## Header style
hdr_style <- openxlsx::createStyle(
  fontColour = "#FFFFFF", fgFill = "#004890",
  halign = "CENTER", textDecoration = "Bold"
)
openxlsx::addStyle(wb, "Leaderboard", hdr_style,
                   rows = 1L, cols = seq_len(ncol(leaderboard)),
                   gridExpand = TRUE)

## Highlight best AUC per framework in blue
for (fw in unique(leaderboard$framework)) {
  best_row <- which(leaderboard$framework == fw &
                      leaderboard$auc_roc == max(leaderboard[framework == fw, auc_roc]))
  if (length(best_row) > 0L)
    openxlsx::addStyle(
      wb, "Leaderboard",
      openxlsx::createStyle(fgFill = "#DDEEFF"),
      rows = best_row + 1L,
      cols = seq_len(ncol(leaderboard)),
      gridExpand = TRUE
    )
}
openxlsx::setColWidths(wb, "Leaderboard", cols = seq_len(ncol(leaderboard)),
                       widths = "auto")

## Sheet 2: DeLong vs M1
if (!is.null(delong_vs_m1) && nrow(delong_vs_m1) > 0L) {
  openxlsx::addWorksheet(wb, "DeLong_vs_M1")
  openxlsx::writeData(wb, "DeLong_vs_M1", delong_vs_m1)
  openxlsx::addStyle(wb, "DeLong_vs_M1", hdr_style,
                     rows = 1L, cols = seq_len(ncol(delong_vs_m1)),
                     gridExpand = TRUE)
  openxlsx::setColWidths(wb, "DeLong_vs_M1",
                         cols = seq_len(ncol(delong_vs_m1)), widths = "auto")
}

## Sheet 3: Full pairwise DeLong
openxlsx::addWorksheet(wb, "DeLong_Pairwise")
openxlsx::writeData(wb, "DeLong_Pairwise", delong_matrix)
openxlsx::addStyle(wb, "DeLong_Pairwise", hdr_style,
                   rows = 1L, cols = seq_len(ncol(delong_matrix)),
                   gridExpand = TRUE)
openxlsx::setColWidths(wb, "DeLong_Pairwise",
                       cols = seq_len(ncol(delong_matrix)), widths = "auto")

xlsx_path <- file.path(DIR_EVAL, sprintf("leaderboard_%s.xlsx", SPLIT_MODE))
openxlsx::saveWorkbook(wb, xlsx_path, overwrite = TRUE)
message(sprintf("  leaderboard_%s.xlsx saved (3 sheets)", SPLIT_MODE))

## ── LaTeX ─────────────────────────────────────────────────────────────────
## Subset to key columns for the paper table
tex_cols <- c("framework", "model", "auc_roc", "avg_precision",
              "brier", "bss", "recall_fpr3", "recall_fpr5",
              "uplift_auc_vs_m1_pct")
tex_dt   <- leaderboard[, .SD, .SDcols = intersect(tex_cols, names(leaderboard))]

## Column headers for the LaTeX table
col_labels <- c(
  framework           = "Framework",
  model               = "Model",
  auc_roc             = "AUC-ROC",
  avg_precision       = "Avg. Prec.",
  brier               = "Brier",
  bss                 = "BSS",
  recall_fpr3         = "R@FPR3\\%",
  recall_fpr5         = "R@FPR5\\%",
  uplift_auc_vs_m1_pct = "$\\Delta$AUC\\%"
)

n_cols <- ncol(tex_dt)
col_fmt <- paste(c("l", "l", rep("r", n_cols - 2L)), collapse = "")

tex_lines <- c(
  "% Auto-generated by 06_Evaluate.R — do not edit manually",
  sprintf("%% Split mode: %s", SPLIT_MODE),
  "\\begin{table}[htbp]",
  "\\centering",
  sprintf("\\caption{Model Comparison — %s}", SPLIT_MODE),
  sprintf("\\label{tab:leaderboard_%s}", tolower(SPLIT_MODE)),
  sprintf("\\begin{tabular}{%s}", col_fmt),
  "\\toprule",
  paste(col_labels[names(tex_dt)], collapse = " & ") %>%
    paste0(" \\\\"),
  "\\midrule"
)

for (i in seq_len(nrow(tex_dt))) {
  row_vals <- vapply(names(tex_dt), function(nm) {
    v <- tex_dt[[nm]][i]
    if (is.na(v)) return("—")
    if (is.numeric(v)) {
      if (nm %in% c("auc_roc", "avg_precision", "recall_fpr3",
                    "recall_fpr5", "bss"))
        return(sprintf("%.4f", v))
      if (nm == "brier")        return(sprintf("%.5f", v))
      if (nm == "uplift_auc_vs_m1_pct") return(sprintf("%+.3f", v))
    }
    as.character(v)
  }, character(1L))
  
  ## Bold the best AUC row per framework
  is_best <- !is.na(tex_dt$auc_roc[i]) &&
    tex_dt$auc_roc[i] == max(
      leaderboard[framework == tex_dt$framework[i], auc_roc], na.rm = TRUE)
  row_str <- paste(row_vals, collapse = " & ")
  if (is_best) row_str <- paste0("\\textbf{", row_str, "}")
  tex_lines <- c(tex_lines, paste0(row_str, " \\\\"))
  
  ## Add midrule between frameworks
  if (i < nrow(tex_dt) &&
      tex_dt$framework[i] != tex_dt$framework[i + 1L])
    tex_lines <- c(tex_lines, "\\midrule")
}

tex_lines <- c(tex_lines,
               "\\bottomrule",
               "\\end{tabular}",
               "\\end{table}"
)

tex_path <- file.path(DIR_EVAL, sprintf("leaderboard_%s.tex", SPLIT_MODE))
writeLines(tex_lines, tex_path)
message(sprintf("  leaderboard_%s.tex saved", SPLIT_MODE))


#==============================================================================#
#==== 7 - Console Summary =====================================================#
#==============================================================================#

message(sprintf("\n══ 06_Evaluate complete [%s] ══", SPLIT_MODE))
message(sprintf("  Models evaluated : %d", nrow(leaderboard)))
message(sprintf("  DeLong pairs     : %d", nrow(delong_matrix)))
message(sprintf("  Output dir       : %s", DIR_EVAL))
message("\n  Key results:")
message(sprintf("  %-12s %-6s | %-8s | %-8s | %-7s | %-7s | %s",
                "Framework", "Model", "AUC-ROC", "Avg.Prec", "Brier",
                "BSS", "R@FPR5%"))
message(sprintf("  %s", strrep("-", 72)))
for (i in seq_len(nrow(leaderboard))) {
  r <- leaderboard[i]
  message(sprintf("  %-12s %-6s | %-8.4f | %-8.4f | %-7.5f | %-7.4f | %.4f",
                  r$framework, r$model, r$auc_roc, r$avg_precision,
                  r$brier, r$bss, r$recall_fpr5))
}

