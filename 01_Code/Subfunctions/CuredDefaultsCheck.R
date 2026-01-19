## Currently not in use in the "Main" Code.

CuredDefaultsCheck <- function(data, 
                               id_col = "id", 
                               date_col = "refdate", 
                               target_col = "y", 
                               mode = "remove") {
  
  # 1. Input Validation
  required_cols <- c(id_col, date_col, target_col)
  missing_cols <- setdiff(required_cols, names(data))
  if(length(missing_cols) > 0) stop(paste("Missing columns:", paste(missing_cols, collapse = ", ")))
  
  # 2. Processing
  processed_data <- data %>%
    # Dynamic column selection
    mutate(!!date_col := as.Date(.data[[date_col]])) %>%
    group_by(.data[[id_col]]) %>%
    mutate(
      # FIX: Use standard 'if' instead of 'if_else'. 
      First_Default_Date = if (any(.data[[target_col]] == 1)) {
        min(.data[[date_col]][.data[[target_col]] == 1], na.rm = TRUE)
      } else {
        as.Date(NA)
      },
      
      # Flag: TRUE if this row is strictly after the first default
      Is_Post_Default = !is.na(First_Default_Date) & .data[[date_col]] > First_Default_Date
    ) %>%
    ungroup()
  
  # 3. Output Mode
  if (mode == "remove") {
    clean_data <- processed_data %>%
      filter(Is_Post_Default == FALSE) %>%
      select(-First_Default_Date, -Is_Post_Default)
    
    n_removed <- nrow(data) - nrow(clean_data)
    message(sprintf("Success: Removed %d 'cured/zombie' observations.", n_removed))
    return(clean_data)
    
  } else {
    return(processed_data)
  }
}