MVstratifiedsampling <- function(Data,
                                 strat_vars = c("sector", "y"),
                                 Train_size = 0.7){
  
  firm_profile <- Data %>%
    group_by(id) %>%
    summarise(
      y = max(y), 
      sector = first(sector),
      size = first(size),
      .groups = 'drop'
    ) %>%
    mutate(
      Strat_Key = interaction(select(., all_of(strat_vars)), drop = TRUE)
    )
  
  train_index <- createDataPartition(
    y = firm_profile$Strat_Key, 
    p = Train_size, 
    list = FALSE, 
    times = 1
  )
  
  train_ids <- firm_profile$id[train_index]
  test_ids  <- firm_profile$id[-train_index]
  
  Train <- Data %>% filter(id %in% train_ids)
  Test <- Data %>% filter(id %in% test_ids)
  
  Output <- list(Train = Train,
                 Test = Test)
  
  return(Output)
}