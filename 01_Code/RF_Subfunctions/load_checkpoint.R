load_checkpoint <- function(filepath) {
  if (!file.exists(filepath)) {
    return(NULL)
  }
  checkpoint <- readRDS(filepath)
  cat("Checkpoint loaded from:", filepath, "\n")
  cat("  Evaluations completed:", checkpoint$n_evals, "\n")
  cat("  Best CV AUC:", round(checkpoint$best_score, 4), "\n")
  return(checkpoint)
}
