BrierScore <- function(probs, actuals) {
  mean((probs - actuals)^2)
}