MVstratifiedsampling_CV <- function(data, k = 5, strat_vars = c("sector", "y")) {
  
  firm_profile <- data %>%
    group_by(id) %>%
    summarise(
      y_strat = max(y),         
      sector = first(sector),   
      .groups = 'drop'
    ) %>%
    mutate(
      Strat_Key = interaction(select(., all_of(c("sector", "y_strat"))), drop = TRUE)
    )
  
  firm_folds <- createFolds(y = firm_profile$Strat_Key, k = k, list = TRUE, returnTrain = FALSE)
  row_folds <- lapply(firm_folds, function(fold_indices) {
    
    test_ids <- firm_profile$id[fold_indices]
    which(data$id %in% test_ids)
  })
  
  return(row_folds)
}