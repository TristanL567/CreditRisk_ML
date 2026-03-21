#==============================================================================#
#==== VAE / DAE Artifact Generator ============================================#
#==============================================================================#
#
# PURPOSE:
#   Trains a Variational Autoencoder (VAE) and a Denoising Autoencoder (DAE)
#   on the quantile-transformed financial features (f1–f11) and exports the
#   artifacts that main_RF.R Strategies A, B, and C consume.
#
#   This script is a PREREQUISITE for running Strategies A/B/C in main_RF.R.
#   Without running this first, those strategies will print [SKIP].
#
# WHAT IT DOES:
#   1. Loads data.rda and applies the SAME preprocessing + train/test split
#      as main_RF.R (same seed, same stratified sampling, same quantile
#      transformation). This guarantees row-level alignment.
#   2. Trains a VAE (Encoder → Latent → Decoder) using beta-VAE loss.
#   3. Trains a DAE (corrupts inputs with Gaussian noise, learns to
#      reconstruct the clean version).
#   4. Extracts and saves 6 .rds artifact files.
#
# OUTPUT FILES (saved to data/pipeline_artifacts/):
#   ┌────────────────────────────┬────────────────────────────────────────────┐
#   │ File                       │ Contents                    → Used by     │
#   ├────────────────────────────┼────────────────────────────────────────────┤
#   │ vae_latent_train.rds       │ data.frame (N_train × 6)    → Strategy A  │
#   │ vae_latent_test.rds        │ data.frame (N_test × 6)     → Strategy A  │
#   │ vae_anomaly_train.rds      │ numeric vector (N_train)    → Strategy B  │
#   │ vae_anomaly_test.rds       │ numeric vector (N_test)     → Strategy B  │
#   │ dae_denoised_train.rds     │ data.frame (N_train × 11)   → Strategy C  │
#   │ dae_denoised_test.rds      │ data.frame (N_test × 11)    → Strategy C  │
#   └────────────────────────────┴────────────────────────────────────────────┘
#
# DEPENDENCIES:
#   - R packages: dplyr, tidyr, caret, torch, here
#   - torch backend: auto-installed on first run (~2 GB one-time download)
#   - Shared Subfunctions: DataPreprocessing.R, MVstratifiedsampling.R,
#     QuantileTransformation.R (sourced from 01_Code/Subfunctions/)
#   - Data: data/data.rda
#
# RUNTIME: ~2–5 minutes (CPU only, no GPU required)
#
# IMPORTANT:
#   The config values RANDOM_SEED, TRAIN_PROP, and DIVIDE_BY_TOTAL_ASSETS
#   MUST match what is set in main_RF.R. If you change them in one script,
#   change them here too, otherwise the train/test rows won't align.
#
# USAGE:
#   setwd("/path/to/oenb_standalone")
#   source("CreditRisk_ML/01_Code/generate_vae_dae_artifacts.R")
#
#   Then run main_RF.R with:
#   STRATEGIES_TO_RUN <- c("Base", "A", "B", "C", "D")
#
#==============================================================================#


#==============================================================================#
#==== CONFIG ==================================================================#
#==============================================================================#

# ---- Must match main_RF.R (keep in sync!) ----
RANDOM_SEED            <- 123L     # Same seed → same train/test split
TRAIN_PROP             <- 0.7      # Same 70/30 split
DIVIDE_BY_TOTAL_ASSETS <- TRUE     # Same normalization

# ---- VAE Architecture ----
VAE_LATENT_DIM       <- 6L         # Bottleneck size → number of columns in Strategy A
                                   #   Lower = more compression, higher = more information retained
                                   #   Rule of thumb: ~half the input dimension (11 features → 6)
VAE_HIDDEN_DIM       <- 32L        # Hidden layer width (encoder & decoder)
VAE_EPOCHS           <- 300L       # Training epochs — loss should plateau by ~200
VAE_LR               <- 1e-3       # Adam learning rate
VAE_BATCH_SIZE       <- 256L       # Mini-batch size
VAE_KL_WEIGHT        <- 0.5        # β in β-VAE: balances reconstruction vs. regularization
                                   #   Higher β → more regular latent space, worse reconstruction
                                   #   Lower β  → better reconstruction, less structured latent space

# ---- DAE Architecture ----
DAE_HIDDEN_DIM       <- 32L        # Hidden layer width
DAE_EPOCHS           <- 300L       # Training epochs
DAE_LR               <- 1e-3       # Adam learning rate
DAE_BATCH_SIZE       <- 256L       # Mini-batch size
DAE_NOISE_SD         <- 0.3        # Gaussian noise σ added to inputs during training
                                   #   Higher → more aggressive denoising but may distort signal
                                   #   Lower  → less denoising, outputs closer to inputs


#==============================================================================#
#==== LIBRARIES ===============================================================#
#==============================================================================#

cat("\n", strrep("=", 60), "\n")
cat("  VAE / DAE ARTIFACT GENERATOR\n")
cat(strrep("=", 60), "\n\n")

packages <- c("dplyr", "tidyr", "caret", "torch")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# torch needs a backend; install if missing
if (!torch::torch_is_installed()) {
  cat("Installing torch backend (one-time, ~2 GB download)...\n")
  torch::install_torch()
}


#==============================================================================#
#==== SOURCE SHARED SUBFUNCTIONS ==============================================#
#==============================================================================#

PROJECT_ROOT      <- file.path(here::here(""))
DATA_DIRECTORY    <- file.path(PROJECT_ROOT, "data", "data.rda")
ARTIFACT_DIR      <- file.path(PROJECT_ROOT, "data", "pipeline_artifacts")

sourceFunctions <- function(dir) {
  for (f in list.files(dir, pattern = "*.R", full.names = TRUE)) {
    try(source(f, echo = FALSE, verbose = FALSE, local = FALSE))
  }
}
sourceFunctions(file.path(PROJECT_ROOT, "CreditRisk_ML", "01_Code", "Subfunctions"))


#==============================================================================#
#==== DATA PIPELINE (mirrors main_RF.R sections 03–04 EXACTLY) ================#
#==============================================================================#
# This MUST produce the same Train / Test as main_RF.R.
# Same seed + same preprocessing + same stratified split = same rows.
# If you change anything here, change main_RF.R sections 03–04 too.

set.seed(RANDOM_SEED)

cat("Loading data:", DATA_DIRECTORY, "\n")
load(DATA_DIRECTORY)
Data <- d

Data <- DataPreprocessing(Data)

Exclude <- paste0("r", seq(1, 18))
Data <- Data[, -which(names(Data) %in% Exclude)]

Data_Sampled <- MVstratifiedsampling(Data,
                                     strat_vars = c("sector", "y"),
                                     Train_size = TRAIN_PROP)
Train <- Data_Sampled[["Train"]]
Test  <- Data_Sampled[["Test"]]

Exclude_id <- c("id", "refdate")
Train <- Train[, -which(names(Train) %in% Exclude_id)]
Test  <- Test[, -which(names(Test) %in% Exclude_id)]

# 04A — Divide by total assets
if (DIVIDE_BY_TOTAL_ASSETS) {
  asset_col <- "f1"
  cols_to_scale <- paste0("f", 2:11)
  safe_divide <- function(num, den) ifelse(den == 0 | is.na(den), 0, num / den)

  for (col in cols_to_scale) {
    Train[[col]] <- safe_divide(Train[[col]], Train[[asset_col]])
    Test[[col]]  <- safe_divide(Test[[col]],  Test[[asset_col]])
  }
}

# 04B — Quantile Transformation
num_cols <- paste0("f", 1:11)
Train_Transformed <- Train
Test_Transformed  <- Test

for (col in num_cols) {
  res <- QuantileTransformation(Train[[col]], Test[[col]])
  Train_Transformed[[col]] <- res$train
  Test_Transformed[[col]]  <- res$test
}

# Extract numeric feature matrices for the autoencoders
# Only the 11 financial features go into VAE/DAE — not sector, size, or y
feature_cols <- num_cols
X_train <- as.matrix(Train_Transformed[, feature_cols])
X_test  <- as.matrix(Test_Transformed[, feature_cols])

# Remove rows with NAs (autoencoders can't handle them)
# We track which rows were complete so we can re-expand later
train_complete <- complete.cases(X_train)
test_complete  <- complete.cases(X_test)
X_train <- X_train[train_complete, ]
X_test  <- X_test[test_complete, ]

n_features <- ncol(X_train)
cat(sprintf("Features: %d | Train: %d obs | Test: %d obs\n\n",
            n_features, nrow(X_train), nrow(X_test)))


#==============================================================================#
#==== VAE MODEL DEFINITION ====================================================#
#==============================================================================#
# Architecture: Input(11) → Hidden(32) → Latent(6) → Hidden(32) → Output(11)
#
# The VAE learns two things simultaneously:
#   1. A compressed representation (latent space) of 11 financial features
#      into 6 dimensions — used by Strategy A
#   2. How to reconstruct the original features from the compressed form —
#      the reconstruction error is the "anomaly score" used by Strategy B
#
# The encoder outputs TWO vectors per observation:
#   - mu (mean):    the "best guess" position in latent space
#   - logvar:       the log-variance (uncertainty) around that position
# During training, we sample from N(mu, exp(logvar)) — the "reparameterization trick".
# At inference time (artifact extraction), we just use mu as the latent feature.

vae_module <- nn_module(
  "VAE",
  initialize = function(input_dim, hidden_dim, latent_dim) {
    # Encoder: maps input → hidden → (mu, logvar)
    self$enc_fc1 <- nn_linear(input_dim, hidden_dim)
    self$enc_mu  <- nn_linear(hidden_dim, latent_dim)
    self$enc_lv  <- nn_linear(hidden_dim, latent_dim)
    # Decoder: maps latent → hidden → reconstructed input
    self$dec_fc1 <- nn_linear(latent_dim, hidden_dim)
    self$dec_out <- nn_linear(hidden_dim, input_dim)
  },

  encode = function(x) {
    h <- torch_relu(self$enc_fc1(x))
    list(mu = self$enc_mu(h), logvar = self$enc_lv(h))
  },

  reparameterize = function(mu, logvar) {
    std <- torch_exp(logvar * 0.5)
    eps <- torch_randn_like(std)
    mu + eps * std
  },

  decode = function(z) {
    h <- torch_relu(self$dec_fc1(z))
    self$dec_out(h)
  },

  forward = function(x) {
    enc   <- self$encode(x)
    z     <- self$reparameterize(enc$mu, enc$logvar)
    x_hat <- self$decode(z)
    list(x_hat = x_hat, mu = enc$mu, logvar = enc$logvar, z = z)
  }
)

# β-VAE loss = Reconstruction Loss + β × KL Divergence
#   Reconstruction: MSE between input and output (how well it reconstructs)
#   KL Divergence:  forces the latent space to stay close to N(0,1)
#   β (kl_weight):  trade-off between the two
vae_loss <- function(x, x_hat, mu, logvar, kl_weight) {
  recon <- nnf_mse_loss(x_hat, x, reduction = "mean")
  kl    <- -0.5 * torch_mean(1 + logvar - mu$pow(2) - logvar$exp())
  recon + kl_weight * kl
}


#==============================================================================#
#==== DAE MODEL DEFINITION ====================================================#
#==============================================================================#
# Architecture: Input(11) → 32 → 16 → 32 → Output(11)
#
# The DAE learns to reconstruct clean features from noisy inputs.
# During training, Gaussian noise (σ = 0.3) is added to each input.
# The network learns to ignore the noise and output the "true" signal.
#
# At inference, we feed CLEAN data through the trained DAE.
# The output is a "denoised" version — smoothing out measurement errors,
# accounting anomalies, and other noise in the financial data.
# Strategy C uses these denoised features instead of the original ones.

dae_module <- nn_module(
  "DAE",
  initialize = function(input_dim, hidden_dim) {
    self$enc1 <- nn_linear(input_dim, hidden_dim)
    self$enc2 <- nn_linear(hidden_dim, hidden_dim %/% 2L)
    self$dec1 <- nn_linear(hidden_dim %/% 2L, hidden_dim)
    self$dec2 <- nn_linear(hidden_dim, input_dim)
  },

  forward = function(x) {
    h <- torch_relu(self$enc1(x))
    h <- torch_relu(self$enc2(h))
    h <- torch_relu(self$dec1(h))
    self$dec2(h)
  }
)


#==============================================================================#
#==== TRAINING HELPERS ========================================================#
#==============================================================================#

make_batches <- function(X_tensor, batch_size) {
  n <- X_tensor$size(1)
  idx <- torch_randperm(n) + 1L  # 1-based
  starts <- seq(1, n, by = batch_size)
  lapply(starts, function(s) {
    e <- min(s + batch_size - 1L, n)
    batch_idx <- idx[s:e]
    X_tensor$index_select(1, batch_idx)
  })
}


#==============================================================================#
#==== TRAIN VAE ===============================================================#
#==============================================================================#

cat(strrep("-", 60), "\n")
cat("  Training VAE  (", VAE_LATENT_DIM, "latent dims,",
    VAE_HIDDEN_DIM, "hidden,", VAE_EPOCHS, "epochs )\n")
cat(strrep("-", 60), "\n")
flush.console()

torch_manual_seed(RANDOM_SEED)

vae <- vae_module(n_features, VAE_HIDDEN_DIM, VAE_LATENT_DIM)
vae_optim <- optim_adam(vae$parameters, lr = VAE_LR)

X_train_t <- torch_tensor(X_train, dtype = torch_float())

t0 <- Sys.time()
for (epoch in seq_len(VAE_EPOCHS)) {
  vae$train()
  epoch_loss <- 0
  batches <- make_batches(X_train_t, VAE_BATCH_SIZE)

  for (batch in batches) {
    vae_optim$zero_grad()
    out  <- vae(batch)
    loss <- vae_loss(batch, out$x_hat, out$mu, out$logvar, VAE_KL_WEIGHT)
    loss$backward()
    vae_optim$step()
    epoch_loss <- epoch_loss + loss$item()
  }

  if (epoch %% 50 == 0 || epoch == 1) {
    cat(sprintf("  Epoch %3d/%d  loss: %.4f\n",
                epoch, VAE_EPOCHS, epoch_loss / length(batches)))
    flush.console()
  }
}
vae_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  VAE trained in %.1f s\n\n", vae_time))


#==============================================================================#
#==== EXTRACT VAE ARTIFACTS ===================================================#
#==============================================================================#
# Switch to eval mode (disables dropout etc.) and disable gradient tracking

vae$eval()

# --- Strategy A: Latent features ---
# Use the encoder's mu (mean) output as deterministic latent features.
# We do NOT sample — mu is the most likely position in latent space.
with_no_grad({
  enc_train <- vae$encode(X_train_t)
  latent_train_mat <- as.matrix(enc_train$mu)

  X_test_t <- torch_tensor(X_test, dtype = torch_float())
  enc_test <- vae$encode(X_test_t)
  latent_test_mat <- as.matrix(enc_test$mu)
})

latent_train_df <- as.data.frame(latent_train_mat)
latent_test_df  <- as.data.frame(latent_test_mat)
colnames(latent_train_df) <- paste0("latent_", seq_len(VAE_LATENT_DIM))
colnames(latent_test_df)  <- paste0("latent_", seq_len(VAE_LATENT_DIM))

# --- Strategy B: Anomaly scores (per-observation reconstruction error) ---
# MSE between original and reconstructed features, averaged across the 11 dimensions.
# High anomaly score = the VAE struggles to reconstruct this observation = "unusual" company.
with_no_grad({
  out_train <- vae(X_train_t)
  out_test  <- vae(X_test_t)
  anomaly_train <- as.numeric(torch_mean((out_train$x_hat - X_train_t)^2, dim = 2))
  anomaly_test  <- as.numeric(torch_mean((out_test$x_hat  - X_test_t)^2,  dim = 2))
})


#==============================================================================#
#==== TRAIN DAE ===============================================================#
#==============================================================================#

cat(strrep("-", 60), "\n")
cat("  Training DAE  (", DAE_HIDDEN_DIM, "hidden,",
    DAE_EPOCHS, "epochs, noise_sd:", DAE_NOISE_SD, ")\n")
cat(strrep("-", 60), "\n")
flush.console()

torch_manual_seed(RANDOM_SEED + 1L)

dae <- dae_module(n_features, DAE_HIDDEN_DIM)
dae_optim <- optim_adam(dae$parameters, lr = DAE_LR)

t0 <- Sys.time()
for (epoch in seq_len(DAE_EPOCHS)) {
  dae$train()
  epoch_loss <- 0
  batches <- make_batches(X_train_t, DAE_BATCH_SIZE)

  for (batch in batches) {
    dae_optim$zero_grad()
    noisy <- batch + torch_randn_like(batch) * DAE_NOISE_SD
    x_hat <- dae(noisy)
    loss  <- nnf_mse_loss(x_hat, batch)
    loss$backward()
    dae_optim$step()
    epoch_loss <- epoch_loss + loss$item()
  }

  if (epoch %% 50 == 0 || epoch == 1) {
    cat(sprintf("  Epoch %3d/%d  loss: %.4f\n",
                epoch, DAE_EPOCHS, epoch_loss / length(batches)))
    flush.console()
  }
}
dae_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("  DAE trained in %.1f s\n\n", dae_time))


#==============================================================================#
#==== EXTRACT DAE ARTIFACTS ===================================================#
#==============================================================================#
# Feed CLEAN data through the trained DAE (no noise added at inference).
# The output has the same shape as the input (11 features) but "denoised".

dae$eval()

with_no_grad({
  dae_train_mat <- as.matrix(dae(X_train_t))
  dae_test_mat  <- as.matrix(dae(X_test_t))
})

dae_train_df <- as.data.frame(dae_train_mat)
dae_test_df  <- as.data.frame(dae_test_mat)
colnames(dae_train_df) <- feature_cols
colnames(dae_test_df)  <- feature_cols


#==============================================================================#
#==== HANDLE NA ROWS ==========================================================#
#==============================================================================#
# Earlier we removed rows with NAs to feed clean data into the autoencoders.
# Now we re-expand the outputs to match the original row count, filling
# the NA-row positions with NA. This ensures that row i in these artifacts
# corresponds to row i in main_RF.R's Train_Transformed / Test_Transformed.

n_train_orig <- nrow(Train_Transformed)
n_test_orig  <- nrow(Test_Transformed)

if (sum(train_complete) < n_train_orig) {
  # Re-expand latent train
  full_latent_train <- as.data.frame(matrix(NA_real_, n_train_orig, VAE_LATENT_DIM))
  colnames(full_latent_train) <- colnames(latent_train_df)
  full_latent_train[train_complete, ] <- latent_train_df
  latent_train_df <- full_latent_train

  # Re-expand anomaly train
  full_anomaly_train <- rep(NA_real_, n_train_orig)
  full_anomaly_train[train_complete] <- anomaly_train
  anomaly_train <- full_anomaly_train

  # Re-expand DAE train
  full_dae_train <- as.data.frame(matrix(NA_real_, n_train_orig, n_features))
  colnames(full_dae_train) <- feature_cols
  full_dae_train[train_complete, ] <- dae_train_df
  dae_train_df <- full_dae_train
}

if (sum(test_complete) < n_test_orig) {
  full_latent_test <- as.data.frame(matrix(NA_real_, n_test_orig, VAE_LATENT_DIM))
  colnames(full_latent_test) <- colnames(latent_test_df)
  full_latent_test[test_complete, ] <- latent_test_df
  latent_test_df <- full_latent_test

  full_anomaly_test <- rep(NA_real_, n_test_orig)
  full_anomaly_test[test_complete] <- anomaly_test
  anomaly_test <- full_anomaly_test

  full_dae_test <- as.data.frame(matrix(NA_real_, n_test_orig, n_features))
  colnames(full_dae_test) <- feature_cols
  full_dae_test[test_complete, ] <- dae_test_df
  dae_test_df <- full_dae_test
}


#==============================================================================#
#==== SAVE ARTIFACTS ==========================================================#
#==============================================================================#

dir.create(ARTIFACT_DIR, showWarnings = FALSE, recursive = TRUE)

saveRDS(latent_train_df,  file.path(ARTIFACT_DIR, "vae_latent_train.rds"))
saveRDS(latent_test_df,   file.path(ARTIFACT_DIR, "vae_latent_test.rds"))
saveRDS(anomaly_train,    file.path(ARTIFACT_DIR, "vae_anomaly_train.rds"))
saveRDS(anomaly_test,     file.path(ARTIFACT_DIR, "vae_anomaly_test.rds"))
saveRDS(dae_train_df,     file.path(ARTIFACT_DIR, "dae_denoised_train.rds"))
saveRDS(dae_test_df,      file.path(ARTIFACT_DIR, "dae_denoised_test.rds"))

cat(strrep("=", 60), "\n")
cat("  ARTIFACTS SAVED to:", ARTIFACT_DIR, "\n")
cat(strrep("=", 60), "\n\n")

cat("  Files:\n")
for (f in c("vae_latent_train.rds", "vae_latent_test.rds",
            "vae_anomaly_train.rds", "vae_anomaly_test.rds",
            "dae_denoised_train.rds", "dae_denoised_test.rds")) {
  fpath <- file.path(ARTIFACT_DIR, f)
  sz <- file.size(fpath)
  cat(sprintf("    %-30s  %s\n", f,
              if (sz > 1e6) sprintf("%.1f MB", sz/1e6) else sprintf("%.0f KB", sz/1e3)))
}

cat("\n  Latent dimensions (Strategy A):", VAE_LATENT_DIM, "\n")
cat("  Anomaly score range (Strategy B):",
    sprintf("[%.4f, %.4f]", min(anomaly_train, na.rm = TRUE),
            max(anomaly_train, na.rm = TRUE)), "\n")
cat("  DAE feature columns (Strategy C):", paste(feature_cols, collapse = ", "), "\n")

cat("\n  Next step: run main_RF.R with STRATEGIES_TO_RUN <- c(\"Base\",\"A\",\"B\",\"C\",\"D\")\n\n")
