QuantileTransformation <- function(train_vec, test_vec) {
  n <- length(train_vec)
  train_ranks <- (rank(train_vec, ties.method = "average") - 0.5) / n
  train_transformed <- qnorm(train_ranks)
  train_ecdf <- ecdf(train_vec)
  
  test_probs <- train_ecdf(test_vec)
  epsilon <- 1 / (n + 1)
  test_probs <- pmax(epsilon, pmin(test_probs, 1 - epsilon))
  
  test_transformed <- qnorm(test_probs)
  
  return(list(train = train_transformed, test = test_transformed))
}
