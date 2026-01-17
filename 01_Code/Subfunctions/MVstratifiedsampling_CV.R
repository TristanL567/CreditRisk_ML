MVstratifiedsampling_CV <- function(data, 
                                    num_folds = 5, 
                                    strat_vars = c("sector", "y"),
                                    seed = 123
                                    ) {
  set.seed(seed)
  
  firm_profile <- data %>%
    group_by(id) %>%
    summarise(
      across(all_of(strat_vars), ~ { if(is.numeric(.)) max(.) else first(.) }), 
      .groups = 'drop'
    ) %>%
    mutate(
      Strat_Key = do.call(interaction, c(select(., all_of(strat_vars)), list(drop = TRUE)))
    )
  
  firm_fold_indices <- createFolds(
    y = firm_profile$Strat_Key,
    k = num_folds,
    list = TRUE,
    returnTrain = FALSE
  )
  
  folds_list_indices <- lapply(firm_fold_indices, function(indices) {
    test_firm_ids <- firm_profile$id[indices]
    which(data$id %in% test_firm_ids)
  })
  
  firm_fold_map <- data.frame(
    id = integer(),
    fold_id = integer()
  )
  
  for (f in seq_along(firm_fold_indices)) {
    current_ids <- firm_profile$id[firm_fold_indices[[f]]]
    
    firm_fold_map <- rbind(firm_fold_map, 
                           data.frame(id = current_ids, fold_id = f))
  }
  
  fold_vector <- data %>%
    select(id) %>%
    left_join(firm_fold_map, by = "id") %>%
    pull(fold_id)
  
  return(list(
    fold_vector = fold_vector,   # Vector of length N (1, 2, 1, 3...)
    fold_list = folds_list_indices # List of length k (indices for each fold)
  ))
  
}