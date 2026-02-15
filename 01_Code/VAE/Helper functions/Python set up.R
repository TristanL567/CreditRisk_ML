library(reticulate)
py_config()

library(reticulate)

# install into the currently active python (your r-tensorflow venv)
py_install("tensorflow-addons", pip = TRUE)

library(reticulate)

setwd("/Users/admin/Desktop/Industry Lab/01_Code")

py_run_string("import tensorflow_addons as tfa; print('tfa', tfa.__version__)")
library(reticulate)
library(tensorflow)
library(keras)
vae_file <- list.files("/Users/admin/Desktop/Industry Lab/01_Code",
                       pattern = "VAE_Test_Code\\.R$",
                       recursive = TRUE,
                       full.names = TRUE)

stopifnot(length(vae_file) == 1)
source(vae_file)


exists("VAE_train")
exists("Encoder_weights")
exists("encoder_latent")
