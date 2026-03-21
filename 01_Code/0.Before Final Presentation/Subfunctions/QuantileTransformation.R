QuantileTransformation <- function(train_vec, test_vec) {
  valid_train_idx <- !is.na(train_vec)
  valid_train_data <- train_vec[valid_train_idx]
  
  n <- length(valid_train_data)
  
  if (n == 0) stop("Training vector contains only NAs")
  
  train_transformed <- rep(NA_real_, length(train_vec))
    train_ranks <- (rank(valid_train_data, ties.method = "average") - 0.5) / n
    train_transformed[valid_train_idx] <- qnorm(train_ranks)
  
  train_ecdf <- ecdf(valid_train_data)
  test_transformed <- rep(NA_real_, length(test_vec))
  valid_test_idx <- !is.na(test_vec)
  test_probs <- train_ecdf(test_vec[valid_test_idx])
  epsilon <- 1 / (n + 1)
  test_probs <- pmax(epsilon, pmin(test_probs, 1 - epsilon))
  test_transformed[valid_test_idx] <- qnorm(test_probs)
  
  return(list(train = train_transformed, test = test_transformed))
}
