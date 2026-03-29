pkgs <- c("xgboost","rBayesianOptimization","pROC","PRROC",
          "dplyr","data.table","arrow","here","lubridate",
          "caret","Matrix","tibble","tidyr","purrr","ggplot2","openxlsx")
status <- sapply(pkgs, requireNamespace, quietly = TRUE)
cat("Package check:\n")
for (p in names(status))
  cat(sprintf("  %-30s %s\n", p, ifelse(status[p], "OK", "MISSING")))
cat("\nxgboost version: ", as.character(packageVersion("xgboost")), "\n")
