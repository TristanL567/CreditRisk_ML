train_and_eval <- function(best_params, train_task, test_task) {
  set.seed(123)
  learner_config <- get_learner_config(train_task$n_features)
  learner <- learner_config$learner$clone()
  
  # Set optimal parameters
  learner$param_set$values <- c(learner$param_set$values, best_params)
  
  #learner$param_set$values$num.trees <- 500L
  
  # Update thread count for final training (use all cores)
  if ("num.threads" %in% learner$param_set$ids()) {
    learner$param_set$values$num.threads <- n_cores
  } else if ("num_threads" %in% learner$param_set$ids()) {
    learner$param_set$values$num_threads <- n_cores
  }
  
  learner$train(train_task)
  pred <- learner$predict(test_task)
  
  list(
    learner = learner,
    pred = pred,
    auc = pred$score(msr("classif.auc")),
    acc = pred$score(msr("classif.acc")),
    brier = pred$score(msr("classif.bbrier"))
  )
}
