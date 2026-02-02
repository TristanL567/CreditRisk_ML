MVstratifiedsampling_CV_Split <- function(data, 
                                          seed = 123,
                                          firm_fold_indices
) {
## Determines the split on each row.
  
  set.seed(seed)
  unique_firm_ids <- unique(data$id)
  folds_list_indices <- lapply(firm_fold_indices, function(indices) {
    test_firm_ids <- unique_firm_ids[indices]
    which(data$id %in% test_firm_ids)
  })
  
  fold_vector <- integer(nrow(data))
  
  for (f in seq_along(folds_list_indices)) {
    fold_vector[folds_list_indices[[f]]] <- f
  }
  
  return(list(
    fold_vector = fold_vector,   
    fold_list = folds_list_indices
  ))
}
