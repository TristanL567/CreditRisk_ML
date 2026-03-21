MVstratifiedsampling_CV_ID <- function(data, 
                                       num_folds = 5, 
                                       strat_vars = c("sector", "y"),
                                       seed = 123
) {
  
## Determines the split at the ID level.
  
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
  
  return(firm_fold_indices)
  
}
