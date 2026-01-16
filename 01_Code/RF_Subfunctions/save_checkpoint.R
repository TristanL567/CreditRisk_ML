save_checkpoint <- function(instance, filepath) {
  checkpoint <- list(
    archive = as.data.table(instance$archive$data),
    best_params = instance$result_x_domain,
    best_score = instance$result_y,
    n_evals = nrow(instance$archive$data),
    save_time = Sys.time()
  )
  saveRDS(checkpoint, filepath)
  cat("Checkpoint saved to:", filepath, "\n")
  invisible(checkpoint)
}
