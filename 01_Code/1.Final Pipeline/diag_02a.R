library(here)
PROJ_ROOT    <- here::here("")
PIPELINE_DIR <- file.path(PROJ_ROOT, "01_Code", "1.Final Pipeline")
source(file.path(PIPELINE_DIR, "config.R"))
MODEL_GROUP           <- "02"
KEEP_FEATURES         <- "r"
INCLUDE_TIME_DYNAMICS <- FALSE
SPLIT_MODE            <- "OoS"

message("Step 1: config loaded OK")

suppressPackageStartupMessages(library(data.table))
message("Step 2: data.table loaded OK")

source(file.path(PATH_FN_GENERAL, "DataPreprocessing.R"),  echo = FALSE)
message("Step 3: DataPreprocessing.R sourced OK")

source(file.path(PATH_FN_GENERAL, "QuantileTransformation.R"), echo = FALSE)
message("Step 4: QuantileTransformation.R sourced OK")

path_train <- get_split_path(SPLIT_OUT_TRAIN_FINAL)
message("Step 5: train path = ", path_train)
message("        exists = ", file.exists(path_train))
message("        size   = ", file.info(path_train)$size, " bytes")

Train_Final <- readRDS(path_train)
message("Step 6: readRDS train OK — ", nrow(Train_Final), " x ", ncol(Train_Final))

path_cv <- file.path(PATH_DATA_OUT, sprintf("cv_folds_%s.rds", SPLIT_MODE))
message("Step 7: cv path = ", path_cv)
message("        exists = ", file.exists(path_cv))

cv_obj   <- readRDS(path_cv)
cv_folds <- cv_obj$cv_folds
message("Step 8: cv_folds loaded — ", length(cv_folds), " folds")
message("        fold 1 class  = ", class(cv_folds[[1]]))
message("        fold 1 length = ", length(cv_folds[[1]]))
message("        fold 1 range  = ", min(cv_folds[[1]]), " – ", max(cv_folds[[1]]))
message("        max index vs nrow: ", max(unlist(cv_folds)), " vs ", nrow(Train_Final))

message("All steps complete — no crash")
