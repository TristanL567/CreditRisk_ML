#==============================================================================#
#==== 07_Charts.R =============================================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# AUTHOR:   Tristan Leiter
# UPDATED:  2026
#
# PURPOSE:
#   Generate all evaluation and diagnostic charts for one SPLIT_MODE.
#   Two layers:
#     A) Overall comparison charts — all available models × frameworks
#        ROC, PR, leaderboard bar, calibration
#     B) Per-model detailed charts — best XGBoost model only
#        Feature importance, PDP, bivariate hexbin, SHAP beeswarm
#
# INPUTS (from 06_Evaluate.R outputs):
#   03_Output/Evaluation/{split}/leaderboard_{split}.rds
#   03_Output/Evaluation/{split}/predictions_combined_{split}.rds
#   03_Output/Models/XGBoost/{model}_{split}/xgb_model.rds
#   02_Data/02_test_final_{split}.rds   + 02_test_id_vec_{split}.rds
#
# OUTPUTS (→ 03_Charts/{split}/):
#   A_ROC/              ROC curves (individual + combined PDF)
#   A_PR/               PR curves
#   A_Leaderboard/      bar + dot plots
#   A_Calibration/      calibration facets
#   B_Importance/       XGBoost feature importance
#   B_PDP/              partial dependence plots
#   B_Hexbin/           bivariate hexbin
#   B_SHAP/             SHAP beeswarm
#
#==============================================================================#


#==============================================================================#
#==== 0 - Configuration  ← CHANGE HERE =======================================#
#==============================================================================#

## Must match 06_Evaluate.R run
## SPLIT_MODE is already set in config.R — override here only if needed.
## SPLIT_MODE <- "OoS"

## Chart parameters
TOP_N_FEATURES  <- 10L    ## Features shown in importance, PDP, hexbin, SHAP
N_HEX_PAIRS     <- 5L     ## Top N features used for bivariate hexbin pairs
PDP_SAMPLE_N    <- 2000L  ## Rows sampled for PDP computation (speed)
SHAP_SAMPLE_N   <- 3000L  ## Rows sampled for SHAP computation

message(sprintf("\n══ 07_Charts [SPLIT_MODE = %s] ══", SPLIT_MODE))


#==============================================================================#
#==== 1 - Paths & Directories =================================================#
#==============================================================================#

DIR_EVAL   <- file.path(PATH_ROOT, "03_Output", "Evaluation", SPLIT_MODE)
DIR_XGB    <- file.path(PATH_ROOT, "03_Output", "Models", "XGBoost")
DIR_CHARTS <- file.path(PATH_ROOT, "03_Charts", SPLIT_MODE)

## Chart subdirectories
chart_dirs <- c("A_ROC", "A_PR", "A_Leaderboard", "A_Calibration",
                "B_Importance", "B_PDP", "B_Hexbin", "B_SHAP")
for (d in chart_dirs)
  dir.create(file.path(DIR_CHARTS, d), recursive = TRUE, showWarnings = FALSE)

## Helper: save chart as PNG + accumulate for PDF
chart_registry <- list()   ## list(section -> list of ggplot objects)

save_chart <- function(p, section, filename,
                       width = CHART_WIDTH / 300,
                       height = CHART_HEIGHT / 300) {
  png_path <- file.path(DIR_CHARTS, section,
                        sprintf("%s.png", filename))
  ggplot2::ggsave(png_path, plot = p,
                  width = width, height = height,
                  dpi = 300, bg = "white")
  chart_registry[[section]] <<- c(chart_registry[[section]], list(p))
  message(sprintf("  Saved: %s/%s.png", section, filename))
  invisible(p)
}

save_pdf_section <- function(section, width = 12, height = 8) {
  plots <- chart_registry[[section]]
  if (length(plots) == 0L) return(invisible(NULL))
  pdf_path <- file.path(DIR_CHARTS, section,
                        sprintf("%s_%s.pdf", section, SPLIT_MODE))
  grDevices::pdf(pdf_path, width = width, height = height)
  for (p in plots) print(p)
  grDevices::dev.off()
  message(sprintf("  PDF saved: %s/%s_%s.pdf", section, section, SPLIT_MODE))
}


#==============================================================================#
#==== 2 - Load Evaluation Inputs ==============================================#
#==============================================================================#

message("  Loading evaluation inputs...")

path_lb   <- file.path(DIR_EVAL, sprintf("leaderboard_%s.rds",            SPLIT_MODE))
path_pred <- file.path(DIR_EVAL, sprintf("predictions_combined_%s.rds",   SPLIT_MODE))

stopifnot(
  "leaderboard not found — run 06_Evaluate.R first"            = file.exists(path_lb),
  "predictions_combined not found — run 06_Evaluate.R first"   = file.exists(path_pred)
)

leaderboard          <- readRDS(path_lb)
predictions_combined <- readRDS(path_pred)

## Load test feature matrix + id vector (for detailed charts)
Test_Final   <- readRDS(get_split_path(SPLIT_OUT_TEST_FINAL))
test_id_vec  <- readRDS(get_split_path(SPLIT_OUT_TEST_IDS))
if (!is.data.table(Test_Final)) setDT(Test_Final)
Test_Final[, .id := test_id_vec]

message(sprintf("  Leaderboard       : %d models", nrow(leaderboard)))
message(sprintf("  Predictions       : %d rows", nrow(predictions_combined)))
message(sprintf("  Test_Final        : %d rows x %d cols",
                nrow(Test_Final), ncol(Test_Final)))

## Colour palette — one colour per run_id
run_ids    <- unique(predictions_combined$run_id)
n_runs     <- length(run_ids)
base_cols  <- c(COL_BLUE, COL_ORANGE, COL_RED, COL_GREY,
                "#2CA02C", "#9467BD", "#8C564B", "#E377C2")
run_colours <- setNames(
  rep_len(base_cols, n_runs),
  run_ids
)

## Linetype by framework
run_lty <- setNames(
  ifelse(grepl("AutoGluon", run_ids), "solid", "dashed"),
  run_ids
)

## Common theme
theme_credit <- function() {
  ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(face = "bold", size = 14),
      plot.subtitle    = ggplot2::element_text(size = 11, colour = "grey40"),
      axis.text        = ggplot2::element_text(colour = "black"),
      legend.position  = "bottom",
      legend.title     = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank()
    )
}


#==============================================================================#
#==== A1 - ROC Curves =========================================================#
#==============================================================================#

message("\n  A1: ROC curves...")

tryCatch({
  roc_data <- rbindlist(lapply(run_ids, function(rid) {
    dt  <- predictions_combined[run_id == rid]
    roc <- pROC::roc(dt$y, dt$p_default, quiet = TRUE)
    data.table(
      fpr    = 1 - roc$specificities,
      tpr    = roc$sensitivities,
      run_id = rid,
      auc    = round(as.numeric(pROC::auc(roc)), 4)
    )
  }))
  
  ## Label with AUC in legend
  roc_data[, label := sprintf("%s (AUC=%.4f)", run_id, auc)]
  label_map <- unique(roc_data[, .(run_id, label)])
  
  p_roc <- ggplot2::ggplot(roc_data,
                           ggplot2::aes(x = fpr, y = tpr,
                                        colour    = run_id,
                                        linetype  = run_id)) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::geom_abline(slope = 1, intercept = 0,
                         linetype = "dotted", colour = "grey60") +
    ggplot2::scale_colour_manual(
      values = run_colours,
      labels = setNames(label_map$label, label_map$run_id)) +
    ggplot2::scale_linetype_manual(
      values = run_lty,
      labels = setNames(label_map$label, label_map$run_id)) +
    ggplot2::labs(
      title    = sprintf("ROC Curves — %s", SPLIT_MODE),
      subtitle = "Dashed = XGBoost | Solid = AutoGluon",
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    ggplot2::coord_equal() +
    theme_credit()
  
  save_chart(p_roc, "A_ROC", sprintf("roc_all_%s", SPLIT_MODE),
             width = 9, height = 7)
  save_pdf_section("A_ROC")
  
}, error = function(e) message("  A1 failed: ", e$message))


#==============================================================================#
#==== A2 - PR Curves ==========================================================#
#==============================================================================#

message("  A2: PR curves...")

tryCatch({
  pr_data <- rbindlist(lapply(run_ids, function(rid) {
    dt  <- predictions_combined[run_id == rid]
    pos <- dt$p_default[dt$y == 1L]
    neg <- dt$p_default[dt$y == 0L]
    if (length(pos) == 0L || length(neg) == 0L) return(NULL)
    pr  <- PRROC::pr.curve(pos, neg, curve = TRUE)
    data.table(
      recall    = pr$curve[, 1L],
      precision = pr$curve[, 2L],
      run_id    = rid,
      ap        = round(pr$auc.integral, 4)
    )
  }), fill = TRUE)
  
  pr_data[, label := sprintf("%s (AP=%.4f)", run_id, ap)]
  label_map_pr <- unique(pr_data[, .(run_id, label)])
  
  ## Baseline: prevalence line
  prevalence <- mean(predictions_combined[run_id == run_ids[1L], y])
  
  p_pr <- ggplot2::ggplot(pr_data,
                          ggplot2::aes(x = recall, y = precision,
                                       colour   = run_id,
                                       linetype = run_id)) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::geom_hline(yintercept = prevalence,
                        linetype = "dotted", colour = "grey60") +
    ggplot2::annotate("text", x = 0.8, y = prevalence + 0.002,
                      label = sprintf("Baseline (%.3f%%)", prevalence * 100),
                      size = 3.5, colour = "grey50") +
    ggplot2::scale_colour_manual(
      values = run_colours,
      labels = setNames(label_map_pr$label, label_map_pr$run_id)) +
    ggplot2::scale_linetype_manual(
      values = run_lty,
      labels = setNames(label_map_pr$label, label_map_pr$run_id)) +
    ggplot2::labs(
      title    = sprintf("Precision-Recall Curves — %s", SPLIT_MODE),
      subtitle = "Dotted line = naive baseline (prevalence)",
      x = "Recall", y = "Precision"
    ) +
    theme_credit()
  
  save_chart(p_pr, "A_PR", sprintf("pr_all_%s", SPLIT_MODE),
             width = 9, height = 7)
  save_pdf_section("A_PR")
  
}, error = function(e) message("  A2 failed: ", e$message))


#==============================================================================#
#==== A3 - Leaderboard Charts =================================================#
#==============================================================================#

message("  A3: Leaderboard charts...")

tryCatch({
  metrics_to_plot <- c("auc_roc", "avg_precision", "bss")
  metric_labels   <- c(auc_roc = "AUC-ROC",
                       avg_precision = "Avg. Precision",
                       bss = "Brier Skill Score")
  
  lb_long <- melt(
    leaderboard[, c("run_id", "framework", "model", metrics_to_plot),
                with = FALSE],
    id.vars = c("run_id", "framework", "model"),
    variable.name = "metric", value.name = "value"
  )
  lb_long[, metric_label := metric_labels[as.character(metric)]]
  lb_long[, model_fw := sprintf("%s\n%s", model, framework)]
  
  ## Dot + segment plot (cleaner than bars for metric comparison)
  p_lb <- ggplot2::ggplot(lb_long,
                          ggplot2::aes(x = value, y = stats::reorder(model_fw, value),
                                       colour = framework, shape = model)) +
    ggplot2::geom_point(size = 3.5) +
    ggplot2::geom_segment(
      ggplot2::aes(x = 0, xend = value,
                   y = stats::reorder(model_fw, value),
                   yend = stats::reorder(model_fw, value)),
      linewidth = 0.4, alpha = 0.4) +
    ggplot2::facet_wrap(~ metric_label, scales = "free_x", nrow = 1L) +
    ggplot2::scale_colour_manual(
      values = c(AutoGluon = COL_BLUE, XGBoost = COL_ORANGE)) +
    ggplot2::labs(
      title    = sprintf("Model Leaderboard — %s", SPLIT_MODE),
      subtitle = "Higher is better for all metrics shown",
      x = NULL, y = NULL
    ) +
    theme_credit() +
    ggplot2::theme(legend.position = "right")
  
  save_chart(p_lb, "A_Leaderboard",
             sprintf("leaderboard_%s", SPLIT_MODE),
             width = 14, height = 6)
  save_pdf_section("A_Leaderboard")
  
}, error = function(e) message("  A3 failed: ", e$message))


#==============================================================================#
#==== A3b - AUC-ROC with Bayesian Bootstrap Credibility Intervals =============#
#==============================================================================#
#
# Method: non-parametric Bayesian bootstrap (Dirichlet(1,...,1) prior),
# equivalent to resampling with replacement. Recomputes AUC on each bootstrap
# sample to produce a posterior distribution. 95% CI = 2.5th/97.5th percentile.
#
# More informative than DeLong at n_defaults = 578 because it shows the
# full posterior uncertainty visually rather than a p-value alone.
#
#==============================================================================#

tryCatch({
  
  message("  A3b: AUC credibility intervals (Bayesian bootstrap)...")
  
  N_BOOT <- 1000L   ## bootstrap resamples — increase for tighter CI
  
  set.seed(SEED)
  auc_ci <- rbindlist(lapply(run_ids, function(rid) {
    dt  <- predictions_combined[run_id == rid]
    y   <- dt$y
    yp  <- dt$p_default
    n   <- length(y)
    
    boot_aucs <- vapply(seq_len(N_BOOT), function(b) {
      idx <- sample(n, n, replace = TRUE)
      yt  <- y[idx]; yh <- yp[idx]
      if (length(unique(yt)) < 2L) return(NA_real_)
      as.numeric(pROC::auc(pROC::roc(yt, yh, quiet = TRUE)))
    }, numeric(1L))
    boot_aucs <- boot_aucs[!is.na(boot_aucs)]
    
    data.table(
      run_id    = rid,
      framework = predictions_combined[run_id == rid, framework[1L]],
      model     = predictions_combined[run_id == rid, model[1L]],
      auc_med   = median(boot_aucs),
      ci_lo     = quantile(boot_aucs, 0.025),
      ci_hi     = quantile(boot_aucs, 0.975),
      ci_width  = quantile(boot_aucs, 0.975) - quantile(boot_aucs, 0.025)
    )
  }))
  
  auc_ci[, run_id := factor(run_id,
                            levels = auc_ci[order(auc_med), run_id])]
  
  p_auc_ci <- ggplot2::ggplot(auc_ci,
                              ggplot2::aes(x = auc_med, y = run_id,
                                           colour = framework, shape = model)) +
    ggplot2::geom_errorbarh(
      ggplot2::aes(xmin = ci_lo, xmax = ci_hi),
      height = 0.25, linewidth = 0.8, alpha = 0.7) +
    ggplot2::geom_point(size = 3.5) +
    ggplot2::geom_vline(
      xintercept = max(auc_ci$auc_med),
      linetype = "dashed", colour = "grey50", linewidth = 0.5) +
    ggplot2::geom_text(
      ggplot2::aes(x = ci_hi,
                   label = sprintf("[%.4f, %.4f]", ci_lo, ci_hi)),
      hjust = -0.05, size = 3.0, colour = "grey30") +
    ggplot2::scale_colour_manual(
      values = c(AutoGluon = COL_BLUE, XGBoost = COL_ORANGE)) +
    ggplot2::scale_x_continuous(
      labels = scales::number_format(accuracy = 0.001),
      expand = ggplot2::expansion(mult = c(0.02, 0.12))) +
    ggplot2::labs(
      title    = sprintf("AUC-ROC with 95%% Bayesian Credibility Intervals — %s",
                         SPLIT_MODE),
      subtitle = sprintf(
        "Non-parametric Bayesian bootstrap | %d resamples | n_defaults = %d",
        N_BOOT,
        predictions_combined[run_id == run_ids[1L], sum(y)]),
      x = "AUC-ROC (posterior median + 95% CI)",
      y = NULL
    ) +
    theme_credit() +
    ggplot2::theme(
      legend.position = "right",
      axis.text.y     = ggplot2::element_text(size = 10)
    )
  
  save_chart(p_auc_ci, "A_Leaderboard",
             sprintf("auc_credibility_%s", SPLIT_MODE),
             width = 12, height = max(5, nrow(auc_ci) * 0.6 + 2))
  
  ## Console summary
  message(sprintf("\n  AUC Credibility Intervals [%s]:", SPLIT_MODE))
  message(sprintf("  %-30s | %-8s | %-8s | %-8s | %s",
                  "Model", "Median", "CI Lo", "CI Hi", "Width"))
  message(sprintf("  %s", strrep("-", 68)))
  for (i in seq_len(nrow(auc_ci[order(-auc_med)]))) {
    r <- auc_ci[order(-auc_med)][i]
    message(sprintf("  %-30s | %.4f   | %.4f  | %.4f  | %.4f",
                    as.character(r$run_id), r$auc_med,
                    r$ci_lo, r$ci_hi, r$ci_width))
  }
  
  ## Save CI table for reference in 06_Evaluate
  saveRDS(auc_ci,
          file.path(DIR_EVAL,
                    sprintf("auc_credibility_%s.rds", SPLIT_MODE)))
  message(sprintf("  Saved: auc_credibility_%s.rds", SPLIT_MODE))
  
}, error = function(e) message("  A3b failed: ", e$message))


#==============================================================================#
#==== A4 - Calibration ========================================================#
#==============================================================================#

message("  A4: Calibration plots...")

tryCatch({
  calib_data <- rbindlist(lapply(run_ids, function(rid) {
    dt <- predictions_combined[run_id == rid]
    dt[, decile := dplyr::ntile(p_default, 10L)]
    dt[, .(
      predicted = mean(p_default),
      observed  = mean(y),
      run_id    = rid,
      framework = framework[1L],
      model     = model[1L]
    ), by = decile]
  }))
  
  calib_long <- melt(
    calib_data,
    id.vars     = c("decile", "run_id", "framework", "model"),
    measure.vars = c("predicted", "observed"),
    variable.name = "type", value.name = "rate"
  )
  
  p_calib <- ggplot2::ggplot(calib_long,
                             ggplot2::aes(x = factor(decile), y = rate * 100,
                                          fill = type)) +
    ggplot2::geom_col(position = ggplot2::position_dodge(0.8),
                      width = 0.7) +
    ggplot2::facet_wrap(~ run_id, ncol = 4L) +
    ggplot2::scale_fill_manual(
      values = c(predicted = COL_BLUE, observed = COL_GREY),
      labels = c(predicted = "Predicted", observed = "Observed")) +
    ggplot2::scale_y_continuous(
      labels = function(x) sprintf("%.2f%%", x)) +
    ggplot2::labs(
      title    = sprintf("Calibration — %s", SPLIT_MODE),
      subtitle = "Predicted vs observed default rate by risk decile",
      x = "Risk Decile (1=Low, 10=High)",
      y = "Default Rate (%)"
    ) +
    theme_credit() +
    ggplot2::theme(
      panel.grid.major.x = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(size = 9)
    )
  
  save_chart(p_calib, "A_Calibration",
             sprintf("calibration_all_%s", SPLIT_MODE),
             width = 18, height = 10)
  save_pdf_section("A_Calibration")
  
}, error = function(e) message("  A4 failed: ", e$message))


#==============================================================================#
#==== B - Select Best XGBoost Model ===========================================#
#==============================================================================#

message("\n  Selecting best XGBoost model for detailed charts...")

xgb_lb <- leaderboard[framework == "XGBoost"]

if (nrow(xgb_lb) == 0L) {
  message("  No XGBoost results found — skipping B sections.")
} else {
  
  best_xgb_model <- xgb_lb[which.max(auc_roc), model]
  message(sprintf("  Best XGBoost model: %s (AUC=%.4f)",
                  best_xgb_model,
                  xgb_lb[model == best_xgb_model, auc_roc]))
  
  ## Load XGBoost result object
  xgb_path <- file.path(DIR_XGB,
                        sprintf("%s_%s", best_xgb_model, SPLIT_MODE),
                        "xgb_model.rds")
  
  if (!file.exists(xgb_path)) {
    message(sprintf("  XGBoost model file not found: %s — skipping B sections.",
                    xgb_path))
  } else {
    
    xgb_result  <- readRDS(xgb_path)
    xgb_model   <- xgb_result$model
    importance  <- xgb_result$importance
    
    ## ── Build test feature matrix ────────────────────────────────────────────
    ## Join Test_Final (features) with test predictions on id
    xgb_preds <- predictions_combined[
      run_id == sprintf("XGBoost_%s_%s", best_xgb_model, SPLIT_MODE)]
    
    ## Merge features onto predictions by id
    test_with_feats <- merge(
      xgb_preds[, .(id, y, p_default)],
      Test_Final,
      by.x = "id", by.y = ".id",
      all.x = TRUE
    )
    
    feature_cols <- setdiff(names(test_with_feats),
                            c("id", "y", "p_default", TARGET_COL))
    
    ## Top N features by Gain
    top_features <- head(importance[order(-Gain), Feature], TOP_N_FEATURES)
    
    ## Apply human-readable labels
    label_feat <- function(f) {
      if (f %in% names(FEATURE_MAP)) FEATURE_MAP[[f]] else f
    }
    
    message(sprintf("  Top %d features: %s ...",
                    TOP_N_FEATURES,
                    paste(head(top_features, 5L), collapse = ", ")))
    
    
    #===========================================================================#
    #==== B1 - Feature Importance =============================================#
    #===========================================================================#
    
    message("  B1: Feature importance...")
    
    tryCatch({
      imp_dt <- as.data.table(importance)[
        Feature %in% top_features][order(-Gain)]
      imp_dt[, Label := sapply(Feature, label_feat)]
      
      p_imp <- ggplot2::ggplot(imp_dt,
                               ggplot2::aes(x = Gain,
                                            y = stats::reorder(Label, Gain))) +
        ggplot2::geom_col(fill = COL_BLUE, width = 0.7) +
        ggplot2::geom_text(
          ggplot2::aes(label = sprintf("%.4f", Gain)),
          hjust = -0.1, size = 3.5, colour = "grey30") +
        ggplot2::scale_x_continuous(
          expand = ggplot2::expansion(mult = c(0, 0.15))) +
        ggplot2::labs(
          title    = sprintf("Feature Importance (Gain) — XGBoost %s [%s]",
                             best_xgb_model, SPLIT_MODE),
          subtitle = sprintf("Top %d features", TOP_N_FEATURES),
          x = "Gain", y = NULL
        ) +
        theme_credit()
      
      save_chart(p_imp, "B_Importance",
                 sprintf("importance_%s_%s", best_xgb_model, SPLIT_MODE),
                 width = 10, height = 7)
      save_pdf_section("B_Importance")
      
    }, error = function(e) message("  B1 failed: ", e$message))
    
    
    #===========================================================================#
    #==== B2 - PDP (Partial Dependence) =======================================#
    #===========================================================================#
    
    message("  B2: PDP / marginal response...")
    
    tryCatch({
      if (!requireNamespace("pdp", quietly = TRUE))
        stop("Package 'pdp' required for PDP charts.")
      
      ## Sample for speed
      set.seed(SEED)
      pdp_idx <- sample(nrow(test_with_feats),
                        min(PDP_SAMPLE_N, nrow(test_with_feats)))
      
      ## Build sparse matrix for prediction
      options(na.action = "na.pass")
      pdp_mat <- sparse.model.matrix(
        as.formula(paste(TARGET_COL, "~ . - 1")),
        data = test_with_feats[pdp_idx,
                               c(feature_cols, TARGET_COL), with = FALSE]
      )
      options(na.action = "na.omit")
      
      ## Column alignment
      train_feats <- xgb_model$feature_names
      miss <- setdiff(train_feats, colnames(pdp_mat))
      if (length(miss) > 0L) {
        zm <- Matrix::sparseMatrix(
          i = integer(0), j = integer(0),
          dims = c(nrow(pdp_mat), length(miss)),
          dimnames = list(NULL, miss))
        pdp_mat <- cbind(pdp_mat, zm)
      }
      pdp_mat <- pdp_mat[, train_feats, drop = FALSE]
      
      pred_wrapper <- function(object, newdata) {
        predict(object, xgboost::xgb.DMatrix(data = as.matrix(newdata)))
      }
      
      pdp_plots <- list()
      for (feat in intersect(top_features, colnames(pdp_mat))) {
        lbl <- label_feat(feat)
        pd  <- tryCatch(
          pdp::partial(xgb_model, pred.var = feat,
                       train      = as.matrix(pdp_mat),
                       pred.fun   = pred_wrapper,
                       prob       = FALSE,
                       grid.resolution = 20L,
                       progress   = "none"),
          error = function(e) NULL
        )
        if (is.null(pd)) next
        
        p <- ggplot2::ggplot(pd,
                             ggplot2::aes(x = .data[[feat]], y = yhat)) +
          ggplot2::geom_line(colour = COL_BLUE, linewidth = 1.1) +
          ggplot2::geom_ribbon(
            ggplot2::aes(ymin = 0, ymax = yhat),
            fill = COL_BLUE, alpha = 0.12) +
          ggplot2::labs(title = lbl, x = NULL,
                        y = "Pred. Default Prob.") +
          theme_credit() +
          ggplot2::theme(
            plot.title = ggplot2::element_text(size = 11, face = "bold"))
        pdp_plots[[length(pdp_plots) + 1L]] <- p
      }
      
      ## Save in batches of 6 (2 rows × 3 cols)
      batch_size <- 6L
      n_batches  <- ceiling(length(pdp_plots) / batch_size)
      for (b in seq_len(n_batches)) {
        idx   <- ((b - 1L) * batch_size + 1L):min(b * batch_size,
                                                  length(pdp_plots))
        batch <- pdp_plots[idx]
        p_pdp <- ggpubr::ggarrange(
          plotlist = batch,
          ncol = 3L, nrow = 2L,
          common.legend = FALSE
        )
        p_pdp <- ggpubr::annotate_figure(
          p_pdp,
          top = ggpubr::text_grob(
            sprintf("Marginal Response — XGBoost %s [%s] — Batch %d",
                    best_xgb_model, SPLIT_MODE, b),
            face = "bold", size = 13)
        )
        save_chart(p_pdp, "B_PDP",
                   sprintf("pdp_%s_%s_batch%d", best_xgb_model, SPLIT_MODE, b),
                   width = 14, height = 9)
      }
      save_pdf_section("B_PDP", width = 14, height = 9)
      
    }, error = function(e) message("  B2 failed: ", e$message))
    
    
    #===========================================================================#
    #==== B3 - Bivariate Hexbin ===============================================#
    #===========================================================================#
    
    message("  B3: Bivariate hexbin...")
    
    tryCatch({
      hex_features <- intersect(
        head(top_features, N_HEX_PAIRS),
        names(test_with_feats)
      )
      
      ## All unique pairs
      pairs <- combn(hex_features, 2L, simplify = FALSE)
      
      for (pair in pairs) {
        fx <- pair[1L]; fy <- pair[2L]
        lx <- label_feat(fx); ly <- label_feat(fy)
        
        plot_df <- test_with_feats[
          !is.na(get(fx)) & !is.na(get(fy)),
          .(x = get(fx), y_feat = get(fy), prob = p_default)
        ]
        
        p_hex <- ggplot2::ggplot(plot_df,
                                 ggplot2::aes(x = x, y = y_feat, z = prob)) +
          ggplot2::stat_summary_hex(
            fun = mean, bins = 40L,
            colour = "grey92", linewidth = 0.1) +
          ggplot2::scale_fill_gradientn(
            colours = c("#FFFFFF", "#DDEEFF", COL_BLUE,
                        COL_ORANGE, COL_RED),
            name    = "Def. Prob.",
            labels  = scales::percent_format(accuracy = 0.01)
          ) +
          ggplot2::labs(
            title    = sprintf("%s × %s", lx, ly),
            subtitle = sprintf("XGBoost %s [%s] — fill = mean predicted default prob.",
                               best_xgb_model, SPLIT_MODE),
            x = lx, y = ly
          ) +
          theme_credit() +
          ggplot2::theme(legend.position = "right")
        
        fname <- sprintf("hexbin_%s_%s_%s_%s",
                         best_xgb_model, SPLIT_MODE,
                         gsub("[^a-zA-Z0-9]", "", fx),
                         gsub("[^a-zA-Z0-9]", "", fy))
        save_chart(p_hex, "B_Hexbin", fname, width = 9, height = 7)
      }
      save_pdf_section("B_Hexbin", width = 9, height = 7)
      
    }, error = function(e) message("  B3 failed: ", e$message))
    
    
    #===========================================================================#
    #==== B4 - SHAP Beeswarm ==================================================#
    #===========================================================================#
    
    message("  B4: SHAP beeswarm...")
    
    tryCatch({
      ## Sample for speed
      set.seed(SEED)
      shap_idx <- sample(nrow(test_with_feats),
                         min(SHAP_SAMPLE_N, nrow(test_with_feats)))
      
      options(na.action = "na.pass")
      shap_mat_full <- sparse.model.matrix(
        as.formula(paste(TARGET_COL, "~ . - 1")),
        data = test_with_feats[shap_idx,
                               c(feature_cols, TARGET_COL), with = FALSE]
      )
      options(na.action = "na.omit")
      
      miss2 <- setdiff(train_feats, colnames(shap_mat_full))
      if (length(miss2) > 0L) {
        zm2 <- Matrix::sparseMatrix(
          i = integer(0), j = integer(0),
          dims = c(nrow(shap_mat_full), length(miss2)),
          dimnames = list(NULL, miss2))
        shap_mat_full <- cbind(shap_mat_full, zm2)
      }
      shap_mat_full <- shap_mat_full[, train_feats, drop = FALSE]
      
      ## Compute SHAP contributions
      shap_contrib <- predict(
        xgb_model,
        xgboost::xgb.DMatrix(data = as.matrix(shap_mat_full)),
        predcontrib = TRUE
      )
      ## Remove BIAS column
      shap_contrib <- shap_contrib[,
                                   colnames(shap_contrib) != "BIAS", drop = FALSE]
      
      ## Subset to top features
      top_shap_feats <- intersect(top_features, colnames(shap_contrib))
      
      ## Build long data.table for plotting
      feat_vals <- as.matrix(shap_mat_full)[, top_shap_feats, drop = FALSE]
      shap_vals <- shap_contrib[, top_shap_feats, drop = FALSE]
      
      shap_long <- rbindlist(lapply(top_shap_feats, function(f) {
        data.table(
          feature     = f,
          label       = label_feat(f),
          shap_value  = shap_vals[, f],
          feat_value  = feat_vals[, f]
        )
      }))
      
      ## Normalise feature value to [0,1] for colour scale
      shap_long[, feat_norm := (feat_value - min(feat_value, na.rm = TRUE)) /
                  (max(feat_value, na.rm = TRUE) -
                     min(feat_value, na.rm = TRUE) + 1e-9),
                by = feature]
      
      ## Order features by mean |SHAP|
      feat_order <- shap_long[,
                              .(mean_abs_shap = mean(abs(shap_value))), by = label][
                                order(-mean_abs_shap), label]
      shap_long[, label := factor(label, levels = rev(feat_order))]
      
      p_shap <- ggplot2::ggplot(shap_long,
                                ggplot2::aes(x = shap_value, y = label,
                                             colour = feat_norm)) +
        ggplot2::geom_jitter(height = 0.25, size = 0.8, alpha = 0.6) +
        ggplot2::geom_vline(xintercept = 0,
                            colour = "grey40", linewidth = 0.5) +
        ggplot2::scale_colour_gradientn(
          colours = c(COL_BLUE, "white", COL_RED),
          name    = "Feature\nValue",
          labels  = c("Low", "", "High"),
          breaks  = c(0, 0.5, 1)
        ) +
        ggplot2::labs(
          title    = sprintf("SHAP Beeswarm — XGBoost %s [%s]",
                             best_xgb_model, SPLIT_MODE),
          subtitle = sprintf("Top %d features | n=%d observations | "
                             "colour = feature value (low=blue, high=red)",
                             length(top_shap_feats), nrow(shap_mat_full)),
          x = "SHAP Value (impact on log-odds of default)",
          y = NULL
        ) +
        theme_credit() +
        ggplot2::theme(legend.position = "right")
      
      save_chart(p_shap, "B_SHAP",
                 sprintf("shap_%s_%s", best_xgb_model, SPLIT_MODE),
                 width = 11, height = 8)
      save_pdf_section("B_SHAP", width = 11, height = 8)
      
    }, error = function(e) message("  B4 failed: ", e$message))
    
  } ## end: xgb_path exists
} ## end: xgb results found


#==============================================================================#
#==== 9 - Summary =============================================================#
#==============================================================================#

total_charts <- sum(lengths(chart_registry))
message(sprintf("\n══ 07_Charts complete [%s] ══", SPLIT_MODE))
message(sprintf("  Charts saved : %d", total_charts))
message(sprintf("  Output dir   : %s", DIR_CHARTS))
message("  Sections:")
for (s in chart_dirs)
  message(sprintf("    %-20s : %d charts",
                  s, length(chart_registry[[s]])))