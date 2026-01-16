create_custom_cv <- function(task, fold_ids) {
  
  if (length(fold_ids) != task$nrow) {
    stop(sprintf("Length mismatch: fold_ids (%d) != task rows (%d)", 
                 length(fold_ids), task$nrow))
  }
  
  unique_folds <- sort(unique(fold_ids))
  n_folds <- length(unique_folds)
  
  train_sets <- vector("list", n_folds)
  test_sets  <- vector("list", n_folds)
  
  task_ids <- task$row_ids
  
  for (i in seq_along(unique_folds)) {
    f_id <- unique_folds[i]
    
    test_indices  <- which(fold_ids == f_id)
    train_indices <- which(fold_ids != f_id)
    
    test_sets[[i]]  <- task_ids[test_indices]
    train_sets[[i]] <- task_ids[train_indices]
  }
  
  res <- rsmp("custom")
  res$instantiate(task, train_sets, test_sets)
  
  return(res)
}