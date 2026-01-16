get_learner_config <- function(n_features) {
  #===== RANDOM FOREST (RANGER) ===== 
  list(
    learner = lrn("classif.ranger",
                  predict_type = "prob",
                  verbose = FALSE),
    search_space = ps(
      mtry = p_int(lower = 1L, upper = max(1L, 2*as.integer(sqrt(n_features)))),
      min.node.size = p_int(lower = 1L, upper = 50L),
      sample.fraction = p_dbl(lower = 0.2, upper = 1.0),
      splitrule = p_fct(levels = c("gini", "extratrees")),
      num.trees = p_int(lower = 300L, upper = 1000L),
      replace = p_lgl()
    )
  )
}
