#==============================================================================#
#==== Hardware Benchmark — Recommends TRAIN_MODE for main_RF.R ================#
#==============================================================================#
# Run this ONCE before main_RF.R to find out what your machine can handle.
# Takes ~30 seconds. Outputs a recommended TRAIN_MODE setting.
#==============================================================================#

cat("\n", strrep("=", 60), "\n")
cat("  HARDWARE BENCHMARK\n")
cat(strrep("=", 60), "\n\n")

# ---- 1. System info ----
n_cores <- parallel::detectCores()
ram_gb  <- as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / 1e9
os_name <- Sys.info()["sysname"]

cat(sprintf("  OS:          %s\n", os_name))
cat(sprintf("  CPU cores:   %d\n", n_cores))
cat(sprintf("  RAM:         %.1f GB\n", ram_gb))

# ---- 2. Quick RF benchmark (synthetic data, mimics real workload) ----
cat("\n  Running benchmark (5-fold CV on synthetic data)...\n")

suppressPackageStartupMessages({
  library(mlr3)
  library(mlr3learners)
  library(ranger)
})

set.seed(42)
n_obs  <- 5000L
n_feat <- 13L

bench_data <- as.data.frame(matrix(rnorm(n_obs * n_feat), ncol = n_feat))
colnames(bench_data) <- paste0("x", seq_len(n_feat))
bench_data$y <- factor(sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.97, 0.03)))

task <- TaskClassif$new("bench", backend = bench_data, target = "y", positive = "1")
learner <- lrn("classif.ranger", predict_type = "prob", num.trees = 500L,
               num.threads = n_cores)
resampling <- rsmp("cv", folds = 5L)

t0 <- Sys.time()
rr <- resample(task, learner, resampling, store_models = FALSE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

auc_score <- rr$aggregate(msr("classif.auc"))
time_per_fold <- elapsed / 5

cat(sprintf("  Benchmark AUC:       %.4f\n", auc_score))
cat(sprintf("  Total time:          %.1f s\n", elapsed))
cat(sprintf("  Time per CV fold:    %.2f s\n", time_per_fold))

# ---- 3. Estimate total runtimes ----
# Scale factor: real data is ~3-5x larger, MBO overhead ~1.3x
scale_factor <- 4.0 * 1.3

est_per_eval <- time_per_fold * 5 * scale_factor  # 5 folds per eval

est_fast   <- est_per_eval * 10  / 60   # 10 evals, 3 folds → scale by 3/5
est_medium <- est_per_eval * 30  / 60   # 30 evals, 5 folds
est_slow   <- est_per_eval * 100 / 60   # 100 evals, 10 folds → scale by 10/5

# Adjust for different fold counts
est_fast <- est_fast * (3/5)
est_slow <- est_slow * (10/5)

cat("\n  Estimated runtimes PER STRATEGY:\n")
cat(sprintf("    fast:   %5.1f min  (10 evals, 3 folds)\n", est_fast))
cat(sprintf("    medium: %5.1f min  (30 evals, 5 folds)\n", est_medium))
cat(sprintf("    slow:   %5.1f min  (100 evals, 10 folds)\n", est_slow))

cat("\n  Estimated runtimes ALL 5 STRATEGIES:\n")
cat(sprintf("    fast:   %5.1f min  (%.1f hrs)\n", est_fast * 5, est_fast * 5 / 60))
cat(sprintf("    medium: %5.1f min  (%.1f hrs)\n", est_medium * 5, est_medium * 5 / 60))
cat(sprintf("    slow:   %5.1f min  (%.1f hrs)\n", est_slow * 5, est_slow * 5 / 60))

# ---- 4. Recommend mode ----
# Heuristic: recommend the most intensive mode that finishes 1 strategy in <30 min
cat("\n", strrep("-", 60), "\n")

if (est_slow <= 30) {
  recommended <- "slow"
  reason <- "Your machine is powerful enough for production-quality runs."
} else if (est_medium <= 30) {
  recommended <- "medium"
  reason <- "Good balance of quality and speed for your hardware."
} else {
  recommended <- "fast"
  reason <- "Limited resources — use fast mode for development, slow mode overnight."
}

cat(sprintf("  RECOMMENDED:  TRAIN_MODE <- \"%s\"\n", recommended))
cat(sprintf("  Reason:       %s\n", reason))
cat(strrep("-", 60), "\n")

cat(sprintf("\n  Copy this into main_RF.R:\n"))
cat(sprintf("  ───────────────────────────────────\n"))
cat(sprintf("  TRAIN_MODE <- \"%s\"\n", recommended))
cat(sprintf("  ───────────────────────────────────\n\n"))
