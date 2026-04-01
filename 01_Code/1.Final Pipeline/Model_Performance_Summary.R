#==============================================================================#
#==== Generate_Model_Performance_Summary.R ====================================#
#==============================================================================#
#
# PROJECT:  Credit Risk Modelling — VAE-Augmented Default Prediction
# PURPOSE:  Loop through all 30 model folders and generate Excel summary
#
# CORRECTED VERSION - Properly extracts feature set and split from model_id
#
# KEY FIX: The feature set and split are derived from the model_id:
#   - 01a/01b = Raw Balance Sheet (a=OoS, b=OoT)
#   - 02a/02b = Financial Ratios
#   - 03a/03b = Ratios + Time Dynamics
#   - 04a/04b = Ratios + TD + VAE
#   - 05a/05b = VAE Latent Only
#
#==============================================================================#


#==============================================================================#
#==== 0 - CONFIGURATION — EDIT THESE PATHS ===================================#
#==============================================================================#

## Path to your 03_Output/Final folder containing all model subfolders
BASE_OUTPUT <- "/Users/admin/Desktop/Final"

## Output Excel file path
OUTPUT_FILE <- "/Users/admin/Desktop/Model_Performance_Summary.xlsx"


#==============================================================================#
#==== 1 - LOAD REQUIRED PACKAGES =============================================#
#==============================================================================#

message("\n", strrep("=", 80))
message("  CREDIT RISK ML — MODEL PERFORMANCE SUMMARY GENERATOR")
message(strrep("=", 80))
message("\n  Loading required packages...")

required_packages <- c("data.table", "openxlsx", "arrow", "jsonlite", "pROC", "PRROC")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("    Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

message("    ✓ All packages loaded\n")


#==============================================================================#
#==== 2 - HELPER FUNCTION: Get config from model_id ==========================#
#==============================================================================#

#' Extract feature set, split, and other info from model_id (e.g., "03a", "05b")
#' This is the KEY FUNCTION that ensures correct mapping!
get_model_config <- function(mid) {
  
  ## Extract group number (01, 02, 03, 04, 05)
  group_num <- substr(mid, 1, 2)
  
  ## Extract split letter (a = OoS, b = OoT)
  split_letter <- substr(mid, 3, 3)
  split <- if (split_letter == "a") "OoS" else "OoT"
  
  ## Map group number to feature set
  feature_set <- switch(group_num,
    "01" = "01_Raw_BS",
    "02" = "02_Ratios",
    "03" = "03_Ratios+TD",
    "04" = "04_Ratios+TD+VAE",
    "05" = "05_VAE_Only",
    "Unknown"
  )
  
  feature_set_short <- switch(group_num,
    "01" = "Raw Balance Sheet",
    "02" = "Financial Ratios",
    "03" = "Ratios + Time Dynamics",
    "04" = "Ratios + TD + VAE",
    "05" = "VAE Latent Only",
    "Unknown"
  )
  
  description <- switch(group_num,
    "01" = paste0("Raw Balance Sheet Data (", if(split=="OoS") "Out-of-Sample" else "Out-of-Time", ")"),
    "02" = paste0("Financial Ratios (", if(split=="OoS") "Out-of-Sample" else "Out-of-Time", ")"),
    "03" = paste0("Ratios + Time Dynamics (", if(split=="OoS") "Out-of-Sample" else "Out-of-Time", ")"),
    "04" = paste0("Ratios + TD + VAE Latent (", if(split=="OoS") "Out-of-Sample" else "Out-of-Time", ")"),
    "05" = paste0("VAE Latent Features Only (", if(split=="OoS") "Out-of-Sample" else "Out-of-Time", ")"),
    "Unknown"
  )
  
  has_time_dynamics <- group_num %in% c("03", "04", "05")
  has_vae <- group_num %in% c("04", "05")
  
  list(
    model_id = mid,
    group = group_num,
    split = split,
    feature_set = feature_set,
    feature_set_short = feature_set_short,
    description = description,
    has_time_dynamics = has_time_dynamics,
    has_vae = has_vae
  )
}


#==============================================================================#
#==== 3 - METRIC COMPUTATION FUNCTIONS =======================================#
#==============================================================================#

recall_at_fpr <- function(y_true, y_pred, fpr_target) {
  roc_obj <- pROC::roc(y_true, y_pred, quiet = TRUE, direction = "<")
  coords <- pROC::coords(roc_obj, "all", ret = c("fpr", "tpr"), transpose = FALSE)
  eligible <- coords$fpr <= fpr_target
  if (!any(eligible)) return(0)
  max(coords$tpr[eligible])
}

compute_avg_precision <- function(y_true, y_pred) {
  pos <- y_pred[y_true == 1L]
  neg <- y_pred[y_true == 0L]
  if (length(pos) == 0L || length(neg) == 0L) return(NA_real_)
  tryCatch({
    round(PRROC::pr.curve(pos, neg)$auc.integral, 5)
  }, error = function(e) NA_real_)
}

compute_bss <- function(y_true, y_pred) {
  brier <- mean((y_pred - y_true)^2)
  brier_climatology <- mean(y_true) * (1 - mean(y_true))
  if (brier_climatology == 0) return(NA_real_)
  round(1 - brier / brier_climatology, 4)
}

compute_ece <- function(y_true, y_pred, n_bins = 10) {
  bins <- cut(y_pred, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  ece <- 0
  n_total <- length(y_true)
  for (b in levels(bins)) {
    idx <- which(bins == b)
    if (length(idx) > 0) {
      avg_pred <- mean(y_pred[idx])
      avg_true <- mean(y_true[idx])
      ece <- ece + (length(idx) / n_total) * abs(avg_pred - avg_true)
    }
  }
  round(ece, 5)
}

compute_all_metrics <- function(y_true, y_pred) {
  valid <- !is.na(y_true) & !is.na(y_pred)
  yt <- as.integer(y_true[valid])
  yp <- as.numeric(y_pred[valid])
  
  if (length(unique(yt)) < 2L) {
    warning("Only one class present in y_true")
    return(NULL)
  }
  
  roc_obj <- pROC::roc(yt, yp, quiet = TRUE, direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj))
  brier <- mean((yp - yt)^2)
  
  opt_coords <- pROC::coords(roc_obj, "best", 
                              ret = c("threshold", "sensitivity", "specificity"),
                              best.method = "youden")
  threshold_opt <- opt_coords$threshold[1L]
  pred_class <- as.integer(yp >= threshold_opt)
  
  tp <- sum(yt == 1L & pred_class == 1L)
  fp <- sum(yt == 0L & pred_class == 1L)
  fn <- sum(yt == 1L & pred_class == 0L)
  tn <- sum(yt == 0L & pred_class == 0L)
  
  precision_val <- tp / max(tp + fp, 1L)
  recall_val <- tp / max(tp + fn, 1L)
  f1_val <- if (precision_val + recall_val > 0) {
    2 * precision_val * recall_val / (precision_val + recall_val)
  } else 0
  
  specificity_val <- tn / max(tn + fp, 1L)
  sensitivity_val <- tp / max(tp + fn, 1L)
  
  list(
    n_obs = length(yt),
    n_defaults = sum(yt),
    n_non_defaults = sum(yt == 0),
    prevalence = round(mean(yt), 5),
    prevalence_pct = round(mean(yt) * 100, 3),
    auc_roc = round(auc_val, 4),
    gini = round(2 * auc_val - 1, 4),
    avg_precision = compute_avg_precision(yt, yp),
    brier = round(brier, 5),
    bss = compute_bss(yt, yp),
    ece = compute_ece(yt, yp),
    recall_fpr01 = round(recall_at_fpr(yt, yp, 0.01), 4),
    recall_fpr03 = round(recall_at_fpr(yt, yp, 0.03), 4),
    recall_fpr05 = round(recall_at_fpr(yt, yp, 0.05), 4),
    recall_fpr10 = round(recall_at_fpr(yt, yp, 0.10), 4),
    recall_fpr20 = round(recall_at_fpr(yt, yp, 0.20), 4),
    threshold_opt = round(threshold_opt, 5),
    sensitivity = round(sensitivity_val, 4),
    specificity = round(specificity_val, 4),
    precision = round(precision_val, 4),
    recall = round(recall_val, 4),
    f1_score = round(f1_val, 4),
    tp = tp, fp = fp, fn = fn, tn = tn,
    score_mean = round(mean(yp), 5),
    score_median = round(median(yp), 5),
    score_sd = round(sd(yp), 5),
    score_min = round(min(yp), 5),
    score_max = round(max(yp), 5),
    score_q25 = round(quantile(yp, 0.25), 5),
    score_q75 = round(quantile(yp, 0.75), 5),
    score_q95 = round(quantile(yp, 0.95), 5),
    score_mean_default = round(mean(yp[yt == 1]), 5),
    score_mean_nondefault = round(mean(yp[yt == 0]), 5),
    score_median_default = round(median(yp[yt == 1]), 5),
    score_median_nondefault = round(median(yp[yt == 0]), 5)
  )
}


#==============================================================================#
#==== 4 - DEFINE ALL MODEL FOLDERS TO PROCESS ================================#
#==============================================================================#

## All model IDs: 01a, 01b, 02a, 02b, 03a, 03b, 04a, 04b, 05a, 05b
MODEL_IDS <- c("01a", "01b", "02a", "02b", "03a", "03b", "04a", "04b", "05a", "05b")

## All algorithms
ALGORITHMS <- c("AutoGluon", "GLM", "XGBoost_Manual")

message(sprintf("  Base folder: %s\n", BASE_OUTPUT))

if (!dir.exists(BASE_OUTPUT)) {
  stop(sprintf("ERROR: Base folder not found: %s", BASE_OUTPUT))
}


#==============================================================================#
#==== 5 - LOAD PREDICTIONS FROM ALL FOLDERS ==================================#
#==============================================================================#

results_list <- list()
importance_list <- list()
folder_count <- 0
success_count <- 0

for (mid in MODEL_IDS) {
  for (algo in ALGORITHMS) {
    
    folder_name <- sprintf("%s_%s", mid, algo)
    folder_path <- file.path(BASE_OUTPUT, folder_name)
    folder_count <- folder_count + 1
    
    if (!dir.exists(folder_path)) {
      message(sprintf("  [%02d] SKIP  %-25s — folder not found", folder_count, folder_name))
      next
    }
    
    message(sprintf("  [%02d] LOAD  %-25s", folder_count, folder_name))
    
    ## GET CONFIG FROM MODEL ID - THIS IS THE KEY FIX
    cfg <- get_model_config(mid)
    
    pred_df <- NULL
    cv_auc <- NA_real_
    train_auc <- NA_real_
    params_str <- ""
    
    ## ═══════════════════════════════════════════════════════════════════════
    ## AUTOGLUON
    ## ═══════════════════════════════════════════════════════════════════════
    if (algo == "AutoGluon") {
      
      pred_path <- file.path(folder_path, "predictions_test.parquet")
      json_path <- file.path(folder_path, "eval_summary.json")
      fi_path <- file.path(folder_path, "feature_importance.csv")
      
      if (file.exists(pred_path)) {
        pred_df <- tryCatch({
          as.data.table(arrow::read_parquet(pred_path))
        }, error = function(e) NULL)
      }
      
      if (file.exists(json_path)) {
        tryCatch({
          meta <- jsonlite::fromJSON(json_path)
          params_str <- sprintf("Preset: %s", meta$preset %||% "unknown")
        }, error = function(e) NULL)
      }
      
      if (file.exists(fi_path)) {
        tryCatch({
          fi <- fread(fi_path)
          if ("importance" %in% names(fi)) setnames(fi, "importance", "Importance", skip_absent = TRUE)
          if ("feature" %in% names(fi)) setnames(fi, "feature", "Feature", skip_absent = TRUE)
          if (all(c("Feature", "Importance") %in% names(fi))) {
            top_fi <- head(fi[order(-Importance)], 15)
            for (i in seq_len(nrow(top_fi))) {
              importance_list[[length(importance_list) + 1]] <- data.table(
                model_id = mid,
                algorithm = algo,
                folder = folder_name,
                feature_set = cfg$feature_set_short,
                split = cfg$split,
                rank = i,
                feature = top_fi$Feature[i],
                importance = round(top_fi$Importance[i], 6)
              )
            }
          }
        }, error = function(e) NULL)
      }
    }
    
    ## ═══════════════════════════════════════════════════════════════════════
    ## GLM
    ## ═══════════════════════════════════════════════════════════════════════
    else if (algo == "GLM") {
      
      ## Use the split from config (derived from model_id letter a/b)
      split_suffix <- cfg$split
      
      pred_path <- file.path(folder_path, sprintf("predictions_test_GLM_v2_%s.parquet", split_suffix))
      lb_path <- file.path(folder_path, sprintf("GLM_Leaderboard_v2_%s.xlsx", split_suffix))
      vi_path <- file.path(folder_path, sprintf("GLM_Variable_Importance_v2_%s.xlsx", split_suffix))
      
      if (file.exists(pred_path)) {
        pred_df <- tryCatch({
          as.data.table(arrow::read_parquet(pred_path))
        }, error = function(e) NULL)
      }
      
      if (file.exists(lb_path)) {
        tryCatch({
          lb <- as.data.table(openxlsx::read.xlsx(lb_path))
          if (nrow(lb) > 0) {
            params_str <- sprintf("Alpha: %s, Lambda: %s", lb$Alpha[1], lb$Lambda[1])
            if ("AUC" %in% names(lb)) cv_auc <- round(as.numeric(lb$AUC[1]), 4)
          }
        }, error = function(e) NULL)
      }
      
      if (file.exists(vi_path)) {
        tryCatch({
          vi <- as.data.table(openxlsx::read.xlsx(vi_path))
          if (all(c("Feature", "Overall") %in% names(vi))) {
            top_fi <- head(vi[order(-Overall)], 15)
            for (i in seq_len(nrow(top_fi))) {
              importance_list[[length(importance_list) + 1]] <- data.table(
                model_id = mid,
                algorithm = algo,
                folder = folder_name,
                feature_set = cfg$feature_set_short,
                split = cfg$split,
                rank = i,
                feature = top_fi$Feature[i],
                importance = round(top_fi$Overall[i], 6)
              )
            }
          }
        }, error = function(e) NULL)
      }
    }
    
    ## ═══════════════════════════════════════════════════════════════════════
    ## XGBOOST
    ## ═══════════════════════════════════════════════════════════════════════
    else if (algo == "XGBoost_Manual") {
      
      pred_path <- file.path(folder_path, "predictions_test.rds")
      model_path <- file.path(folder_path, "xgb_model.rds")
      
      if (file.exists(pred_path)) {
        pred_df <- tryCatch({
          as.data.table(readRDS(pred_path))
        }, error = function(e) NULL)
      }
      
      if (file.exists(model_path)) {
        tryCatch({
          xgb_obj <- readRDS(model_path)
          if (!is.null(xgb_obj$cv_score_mean)) {
            cv_auc <- round(as.numeric(xgb_obj$cv_score_mean), 4)
          }
          if (!is.null(xgb_obj$eval_table)) {
            eval_tbl <- as.data.table(xgb_obj$eval_table)
            train_row <- eval_tbl[set == "train_insample"]
            if (nrow(train_row) > 0 && "auc_roc" %in% names(train_row)) {
              train_auc <- round(train_row$auc_roc[1], 4)
            }
          }
          if (!is.null(xgb_obj$params)) {
            params_str <- paste(names(xgb_obj$params), unlist(xgb_obj$params), sep = "=", collapse = "; ")
          }
          if (!is.null(xgb_obj$importance)) {
            imp_dt <- as.data.table(xgb_obj$importance)
            if ("Gain" %in% names(imp_dt) && "Feature" %in% names(imp_dt)) {
              top_fi <- head(imp_dt[order(-Gain)], 15)
              for (i in seq_len(nrow(top_fi))) {
                importance_list[[length(importance_list) + 1]] <- data.table(
                  model_id = mid,
                  algorithm = algo,
                  folder = folder_name,
                  feature_set = cfg$feature_set_short,
                  split = cfg$split,
                  rank = i,
                  feature = top_fi$Feature[i],
                  importance = round(top_fi$Gain[i], 6)
                )
              }
            }
          }
        }, error = function(e) NULL)
      }
    }
    
    ## ═══════════════════════════════════════════════════════════════════════
    ## COMPUTE METRICS
    ## ═══════════════════════════════════════════════════════════════════════
    if (!is.null(pred_df) && nrow(pred_df) > 0) {
      
      if ("p_csi" %in% names(pred_df) && !"p_default" %in% names(pred_df)) {
        setnames(pred_df, "p_csi", "p_default")
      }
      
      if (!all(c("y", "p_default") %in% names(pred_df))) {
        message(sprintf("         WARNING: Missing y or p_default column"))
        next
      }
      
      metrics <- compute_all_metrics(pred_df$y, pred_df$p_default)
      
      if (!is.null(metrics)) {
        
        ## STORE RESULTS WITH CORRECT CONFIG FROM MODEL ID
        results_list[[length(results_list) + 1]] <- data.table(
          model_id = mid,
          algorithm = algo,
          folder = folder_name,
          feature_set = cfg$feature_set,
          feature_set_short = cfg$feature_set_short,
          split = cfg$split,
          description = cfg$description,
          has_time_dynamics = cfg$has_time_dynamics,
          has_vae = cfg$has_vae,
          train_auc = train_auc,
          cv_auc = cv_auc,
          n_obs = metrics$n_obs,
          n_defaults = metrics$n_defaults,
          n_non_defaults = metrics$n_non_defaults,
          prevalence = metrics$prevalence,
          prevalence_pct = metrics$prevalence_pct,
          auc_roc = metrics$auc_roc,
          gini = metrics$gini,
          avg_precision = metrics$avg_precision,
          brier = metrics$brier,
          bss = metrics$bss,
          ece = metrics$ece,
          recall_fpr01 = metrics$recall_fpr01,
          recall_fpr03 = metrics$recall_fpr03,
          recall_fpr05 = metrics$recall_fpr05,
          recall_fpr10 = metrics$recall_fpr10,
          recall_fpr20 = metrics$recall_fpr20,
          threshold_opt = metrics$threshold_opt,
          sensitivity = metrics$sensitivity,
          specificity = metrics$specificity,
          precision = metrics$precision,
          recall = metrics$recall,
          f1_score = metrics$f1_score,
          tp = metrics$tp,
          fp = metrics$fp,
          fn = metrics$fn,
          tn = metrics$tn,
          score_mean = metrics$score_mean,
          score_median = metrics$score_median,
          score_sd = metrics$score_sd,
          score_min = metrics$score_min,
          score_max = metrics$score_max,
          score_q25 = metrics$score_q25,
          score_q75 = metrics$score_q75,
          score_q95 = metrics$score_q95,
          score_mean_default = metrics$score_mean_default,
          score_mean_nondefault = metrics$score_mean_nondefault,
          score_median_default = metrics$score_median_default,
          score_median_nondefault = metrics$score_median_nondefault,
          params = params_str
        )
        
        success_count <- success_count + 1
        message(sprintf("         ✓ AUC: %.4f | Brier: %.5f | Recall@5%%: %.4f | Split: %s | FS: %s",
                        metrics$auc_roc, metrics$brier, metrics$recall_fpr05,
                        cfg$split, cfg$feature_set_short))
      }
    } else {
      message(sprintf("         WARNING: No predictions loaded"))
    }
  }
}

## Combine results
if (length(results_list) == 0) {
  stop("ERROR: No model results found.")
}

results_dt <- rbindlist(results_list, fill = TRUE)
importance_dt <- if (length(importance_list) > 0) rbindlist(importance_list) else data.table()

message(sprintf("\n  ═══════════════════════════════════════════════════════════════"))
message(sprintf("  Successfully loaded %d of %d model folders", success_count, folder_count))
message(sprintf("  ═══════════════════════════════════════════════════════════════\n"))


#==============================================================================#
#==== 6 - CREATE SUMMARY TABLES ==============================================#
#==============================================================================#

message("  Creating summary tables...")

## 1. OVERVIEW
overview_dt <- results_dt[order(-auc_roc)]
overview_dt[, Rank := .I]

overview_display <- overview_dt[, .(
  Rank,
  Model = folder,
  `Feature Set` = feature_set_short,
  Split = split,
  Algorithm = algorithm,
  `N Test` = format(n_obs, big.mark = ","),
  `N Defaults` = format(n_defaults, big.mark = ","),
  `Prevalence %` = sprintf("%.3f%%", prevalence_pct),
  `Test AUC` = auc_roc,
  `Gini` = gini,
  `Avg Precision` = avg_precision,
  `Brier Score` = brier,
  `BSS` = bss,
  `Recall@1%` = recall_fpr01,
  `Recall@5%` = recall_fpr05,
  `Recall@10%` = recall_fpr10
)]

## 2. OoS vs OoT
oos_dt <- results_dt[split == "OoS", .(
  model_id, algorithm, feature_set_short,
  oos_auc = auc_roc, oos_gini = gini, oos_brier = brier, 
  oos_bss = bss, oos_recall05 = recall_fpr05, oos_ap = avg_precision
)]
oos_dt[, group := gsub("[ab]$", "", model_id)]

oot_dt <- results_dt[split == "OoT", .(
  model_id, algorithm, feature_set_short,
  oot_auc = auc_roc, oot_gini = gini, oot_brier = brier,
  oot_bss = bss, oot_recall05 = recall_fpr05, oot_ap = avg_precision
)]
oot_dt[, group := gsub("[ab]$", "", model_id)]

comparison_dt <- merge(oos_dt, oot_dt, by = c("group", "algorithm", "feature_set_short"), all = TRUE)
comparison_dt[, `:=`(
  `AUC Gap` = round(oos_auc - oot_auc, 4),
  `Gini Gap` = round(oos_gini - oot_gini, 4),
  `Brier Gap` = round(oos_brier - oot_brier, 5),
  `Recall@5% Gap` = round(oos_recall05 - oot_recall05, 4)
)]

comparison_display <- comparison_dt[order(-oos_auc), .(
  `Feature Set` = feature_set_short,
  Algorithm = algorithm,
  `OoS AUC` = oos_auc,
  `OoT AUC` = oot_auc,
  `AUC Gap (OoS-OoT)` = `AUC Gap`,
  `OoS Gini` = oos_gini,
  `OoT Gini` = oot_gini,
  `OoS Brier` = oos_brier,
  `OoT Brier` = oot_brier,
  `OoS BSS` = oos_bss,
  `OoT BSS` = oot_bss,
  `OoS Recall@5%` = oos_recall05,
  `OoT Recall@5%` = oot_recall05
)]

## 3. BY FEATURE SET
by_fs_dt <- results_dt[, .(
  `N Models` = .N,
  `Mean AUC` = round(mean(auc_roc, na.rm = TRUE), 4),
  `Max AUC` = round(max(auc_roc, na.rm = TRUE), 4),
  `Min AUC` = round(min(auc_roc, na.rm = TRUE), 4),
  `SD AUC` = round(sd(auc_roc, na.rm = TRUE), 4),
  `Mean Gini` = round(mean(gini, na.rm = TRUE), 4),
  `Mean Brier` = round(mean(brier, na.rm = TRUE), 5),
  `Mean BSS` = round(mean(bss, na.rm = TRUE), 4),
  `Mean Recall@5%` = round(mean(recall_fpr05, na.rm = TRUE), 4)
), by = .(feature_set_short, split)][order(feature_set_short, split)]

## 4-6. BY ALGORITHM
algo_tables <- list()
for (a in unique(results_dt$algorithm)) {
  algo_dt <- results_dt[algorithm == a][order(-auc_roc)]
  algo_dt[, Rank := .I]
  algo_tables[[a]] <- algo_dt[, .(
    Rank, Model = model_id, `Feature Set` = feature_set_short, Split = split,
    `N Test` = format(n_obs, big.mark = ","), `N Defaults` = n_defaults,
    `CV AUC` = cv_auc, `Test AUC` = auc_roc, Gini = gini,
    `Avg Precision` = avg_precision, Brier = brier, BSS = bss, ECE = ece,
    `Recall@1%` = recall_fpr01, `Recall@5%` = recall_fpr05, `Recall@10%` = recall_fpr10,
    F1 = f1_score, Precision = precision, Specificity = specificity
  )]
}

## 7. FEATURE IMPORTANCE
if (nrow(importance_dt) > 0) {
  fi_summary <- importance_dt[, .(
    `Avg Importance` = round(mean(importance, na.rm = TRUE), 6),
    `Max Importance` = round(max(importance, na.rm = TRUE), 6),
    `N Models` = .N,
    `Avg Rank` = round(mean(rank, na.rm = TRUE), 1)
  ), by = feature][order(-`Avg Importance`)]
  
  fi_by_model <- importance_dt[order(folder, rank)]
}

## 8. CONFUSION MATRIX
cm_dt <- results_dt[order(-auc_roc), .(
  Model = folder, `Feature Set` = feature_set_short, Split = split, Algorithm = algorithm,
  `True Positives` = tp, `False Positives` = fp, `False Negatives` = fn, `True Negatives` = tn,
  Precision = precision, `Recall/Sensitivity` = recall, Specificity = specificity,
  `F1 Score` = f1_score, `Optimal Threshold` = threshold_opt
)]

## 9. CALIBRATION
calib_dt <- results_dt[order(-auc_roc), .(
  Model = folder, `Feature Set` = feature_set_short, Split = split, Algorithm = algorithm,
  `Brier Score` = brier, BSS = bss, ECE = ece,
  `Score Mean` = score_mean, `Score Median` = score_median, `Score SD` = score_sd,
  `Score Mean (Default)` = score_mean_default, `Score Mean (Non-Default)` = score_mean_nondefault,
  `Score Median (Default)` = score_median_default, `Score Median (Non-Default)` = score_median_nondefault,
  `Prevalence %` = prevalence_pct
)]

## 10. FULL DETAILS
full_details <- results_dt[order(-auc_roc)]

message("    ✓ Summary tables created")


#==============================================================================#
#==== 7 - CREATE EXCEL WORKBOOK ==============================================#
#==============================================================================#

message(sprintf("\n  Creating Excel workbook: %s", OUTPUT_FILE))

wb <- createWorkbook()

header_style <- createStyle(
  fontColour = "#FFFFFF", fgFill = "#004890",
  halign = "CENTER", valign = "CENTER",
  textDecoration = "Bold", wrapText = TRUE,
  border = "TopBottomLeftRight", borderColour = "#FFFFFF"
)
best_style <- createStyle(fgFill = "#C6EFCE", fontColour = "#006100")
alt_row_style <- createStyle(fgFill = "#F2F2F2")

add_sheet <- function(wb, sheet_name, data, highlight_col = NULL, highlight_max = TRUE) {
  addWorksheet(wb, sheet_name)
  writeData(wb, sheet_name, data, headerStyle = header_style)
  setColWidths(wb, sheet_name, cols = seq_len(ncol(data)), widths = "auto")
  freezePane(wb, sheet_name, firstRow = TRUE)
  
  if (!is.null(highlight_col) && highlight_col %in% names(data) && nrow(data) > 0) {
    col_vals <- data[[highlight_col]]
    if (is.numeric(col_vals)) {
      best_val <- if (highlight_max) max(col_vals, na.rm = TRUE) else min(col_vals, na.rm = TRUE)
      best_rows <- which(col_vals == best_val) + 1
      if (length(best_rows) > 0) {
        addStyle(wb, sheet_name, best_style, rows = best_rows, 
                 cols = seq_len(ncol(data)), gridExpand = TRUE, stack = TRUE)
      }
    }
  }
  
  if (nrow(data) > 1) {
    for (i in seq(3, nrow(data) + 1, by = 2)) {
      addStyle(wb, sheet_name, alt_row_style, rows = i, 
               cols = seq_len(ncol(data)), gridExpand = TRUE, stack = TRUE)
    }
  }
}

add_sheet(wb, "1_Overview", overview_display, highlight_col = "Test AUC")
add_sheet(wb, "2_OoS_vs_OoT", comparison_display, highlight_col = "OoS AUC")
add_sheet(wb, "3_By_Feature_Set", by_fs_dt, highlight_col = "Mean AUC")

for (a in names(algo_tables)) {
  sheet_name <- sprintf("4_%s", gsub("_", "", a))
  add_sheet(wb, sheet_name, algo_tables[[a]], highlight_col = "Test AUC")
}

if (nrow(importance_dt) > 0) {
  add_sheet(wb, "5_Feature_Importance", fi_summary, highlight_col = "Avg Importance")
  add_sheet(wb, "5b_FI_by_Model", fi_by_model)
}

add_sheet(wb, "6_Confusion_Matrix", cm_dt)
add_sheet(wb, "7_Calibration", calib_dt, highlight_col = "Brier Score", highlight_max = FALSE)
add_sheet(wb, "8_Full_Details", full_details, highlight_col = "auc_roc")

saveWorkbook(wb, OUTPUT_FILE, overwrite = TRUE)

message(sprintf("\n  ═══════════════════════════════════════════════════════════════"))
message(sprintf("  ✓ Excel workbook saved: %s", OUTPUT_FILE))
message(sprintf("    • %d models processed", nrow(results_dt)))
message(sprintf("    • %d sheets created", length(names(wb))))
message(sprintf("  ═══════════════════════════════════════════════════════════════"))


#==============================================================================#
#==== 8 - CONSOLE SUMMARY ====================================================#
#==============================================================================#

message("\n")
message(strrep("═", 80))
message("  VERIFICATION: MODELS BY FEATURE SET AND SPLIT")
message(strrep("═", 80))

verify_table <- results_dt[, .N, by = .(feature_set_short, split)][order(feature_set_short, split)]
message("\n  Feature Set                  Split   N Models")
message("  ", strrep("-", 55))
for (i in seq_len(nrow(verify_table))) {
  message(sprintf("  %-28s %-6s  %d", 
                  verify_table$feature_set_short[i],
                  verify_table$split[i],
                  verify_table$N[i]))
}

message("\n")
message(strrep("═", 80))
message("  TOP 5 MODELS BY TEST AUC")
message(strrep("═", 80))

top5 <- head(results_dt[order(-auc_roc)], 5)
message(sprintf("\n  %-25s %-25s %-6s %-8s", "Model", "Feature Set", "Split", "AUC"))
message("  ", strrep("-", 70))
for (i in seq_len(nrow(top5))) {
  r <- top5[i]
  message(sprintf("  %-25s %-25s %-6s %-8.4f", 
                  r$folder, r$feature_set_short, r$split, r$auc_roc))
}

message("\n")
message(strrep("═", 80))
message("  SUMMARY COMPLETE")
message(strrep("═", 80))
message("\n")
