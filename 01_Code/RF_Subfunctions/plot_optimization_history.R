plot_optimization_history <- function(results) {
  
  hist_data <- results$archive[, .(iter = seq_len(.N), auc = classif.auc)]
  n_initial_design <- results$n_initial_design
  
  # Label phases
  hist_data[, phase := fifelse(iter <= n_initial_design,
                               "Warmup (Random)", "Bayesian Optimization")]
  hist_data[, phase := factor(phase, levels = c("Warmup (Random)",
                                                "Bayesian Optimization"))]
  
  # Calculate cumulative best
  hist_data[, cummax_auc := cummax(auc)]
  
  # Mark best point
  best_iter <- hist_data[which.max(auc), iter]
  hist_data[, is_best := iter == best_iter]
  
  # Create plot
  p <- ggplot(hist_data, aes(x = iter, y = auc)) +
    geom_point(aes(color = phase, shape = phase), size = 3, alpha = 0.7) +
    geom_line(aes(y = cummax_auc), color = "darkgreen", linetype = "dashed",
              linewidth = 1) +
    geom_point(data = hist_data[is_best == TRUE],
               aes(x = iter, y = auc),
               color = "red", size = 5, shape = 8, stroke = 2) +
    geom_vline(xintercept = n_initial_design + 0.5, linetype = "dotted",
               color = "gray40", linewidth = 1) +
    annotate("text", x = n_initial_design / 2, y = min(hist_data$auc),
             label = "Warmup", color = "gray40", size = 4, vjust = -0.5) +
    annotate("text", x = n_initial_design + (nrow(hist_data) - n_initial_design) / 2,
             y = min(hist_data$auc),
             label = "BO", color = "gray40", size = 4, vjust = -0.5) +
    scale_color_manual(values = c("Warmup (Random)" = "#E69F00",
                                  "Bayesian Optimization" = "#0072B2")) +
    scale_shape_manual(values = c("Warmup (Random)" = 16,
                                  "Bayesian Optimization" = 17)) +
    labs(title = "MBO Optimization History: Warmup vs BO Phase",
         subtitle = paste0("Initial design: ", n_initial_design, " points | ",
                           "Best AUC: ", round(max(hist_data$auc), 4),
                           " at iteration ", best_iter),
         x = "Iteration",
         y = "CV AUC",
         color = "Phase",
         shape = "Phase") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(face = "bold"),
          panel.grid.minor = element_blank())
  
  return(p)
}
