"""
03_Autoencoder.py
=================
Beta-VAE with optional supervised classification loss for credit default
prediction. Adapted from 08B_Autoencoder.py (reference project).

SPLIT SELECTION:
    Set SPLIT_MODE at the top of Section 3.
    "OoS" → loads 02_train_final_vae_OoS.rds / 02_test_final_vae_OoS.rds
    "OoT" → loads 02_train_final_vae_OoT.rds / 02_test_final_vae_OoT.rds

INPUT:
    R pipeline saves two versions of the feature matrix:
      - Uniform (0,1)  → 02_train_final_{split}.rds       (XGBoost etc.)
      - Normal N(0,1)  → 02_train_final_vae_{split}.rds   (VAE — this script)
    This script reads the VAE/normal version only. No further PIT is applied.

ARCHITECTURE (scaled for 508 input features):
    Encoder     : n_features → 512 → 256 → 128 → (z_mean, z_log_var)
    Decoder     : z_dim → 128 → 256 → 512 → n_features
    Classifier  : z_mean → 32 → 1  (logits, optional, controlled by gamma)

LOSS (correct ELBO):
    total      = recon_loss + beta * KL + gamma * BCE_clf
    recon_loss = MSE.sum(dim=1).mean()        continuous features
               + BCE.sum(dim=1).mean()        binary features
    KL         = -0.5 * mean(sum(1 + lv - mu^2 - exp(lv)))
    BCE_clf    = BCEWithLogitsLoss            numerically stable

OUTPUTS (all parquet, suffixed by SPLIT_MODE):
    latent_train_{split}.parquet   id, y, z1..zN, vae_recon_error
    latent_test_{split}.parquet    id, y, z1..zN, vae_recon_error
    anomaly_train_{split}.parquet  id, y, vae_recon_error
    anomaly_test_{split}.parquet   id, y, vae_recon_error

    Use latent_* to test whether VAE latent dims add signal over raw features.
    Use anomaly_* to test whether the reconstruction error alone is predictive.
    Both are joined in 04_Train.R for model comparison.
"""

# ==============================================================================
# 0. Imports
# ==============================================================================

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. Configuration  ← CHANGE HERE
# ==============================================================================

SPLIT_MODE = "OoS"    # "OoS" | "OoT"

# Feature config — must match 02_FeatureEngineering.R settings.
# VAE is only relevant for groups 03/04/05 (ratios + time dynamics).
KEEP_FEATURES         = "r"    # "r" | "f" | "both"
INCLUDE_TIME_DYNAMICS = True   # should always be True for VAE runs

# Allow CLI override: python 03_Autoencoder.py <SPLIT_MODE>
if len(sys.argv) >= 2:
    SPLIT_MODE = sys.argv[1]

assert SPLIT_MODE in ("OoS", "OoT"), \
    f"SPLIT_MODE must be 'OoS' or 'OoT', got: '{SPLIT_MODE}'"

_td         = "TD" if INCLUDE_TIME_DYNAMICS else "noTD"
FEAT_SUFFIX = f"_{KEEP_FEATURES}_{_td}"
FILE_SUFFIX = f"{FEAT_SUFFIX}_{SPLIT_MODE}"   # e.g. "_r_TD_OoS"

# ==============================================================================
# 2. Paths
# ==============================================================================

DATA_ROOT  = Path(r"C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML")
DIR_DATA   = DATA_ROOT / "02_Data"

DIR_OUT    = DATA_ROOT / "03_Output"
DIR_LAT    = DIR_OUT / "Latent"
DIR_FIG    = DIR_OUT / "Figures" / "VAE"
DIR_MODELS = DIR_OUT / "Models"  / "VAE"

for d in [DIR_LAT, DIR_FIG, DIR_MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# VAE/normal-scores inputs produced by 02_FeatureEngineering.R (Stage 02E)
PATH_TRAIN = DIR_DATA / f"02_train_final_vae{FILE_SUFFIX}.rds"
PATH_TEST  = DIR_DATA / f"02_test_final_vae{FILE_SUFFIX}.rds"

assert PATH_TRAIN.exists(), (
    f"VAE train file not found: {PATH_TRAIN}\n"
    f"Run 02_FeatureEngineering.R with KEEP_FEATURES='{KEEP_FEATURES}', "
    f"INCLUDE_TIME_DYNAMICS={INCLUDE_TIME_DYNAMICS}, SPLIT_MODE='{SPLIT_MODE}' first."
)
assert PATH_TEST.exists(), (
    f"VAE test file not found: {PATH_TEST}\n"
    f"Run 02_FeatureEngineering.R with KEEP_FEATURES='{KEEP_FEATURES}', "
    f"INCLUDE_TIME_DYNAMICS={INCLUDE_TIME_DYNAMICS}, SPLIT_MODE='{SPLIT_MODE}' first."
)

print(f"[03] ══════════════════════════════════════")
print(f"  SPLIT_MODE : {SPLIT_MODE}")
print(f"  FILE_SUFFIX: {FILE_SUFFIX}")
print(f"  Train      : {PATH_TRAIN.name}")
print(f"  Test       : {PATH_TEST.name}")
print(f"  Output dir : {DIR_LAT}")
print(f"[03] ══════════════════════════════════════\n")

# ==============================================================================
# 3. Hyperparameters
# ==============================================================================

CFG = {
    # Architecture — scaled for ~508 input features
    "z_dim"          : 32,
    "encoder_dims"   : [512, 256, 128],
    "decoder_dims"   : [128, 256, 512],
    "classifier_dims": [32],

    # Loss weights
    "beta"           : 1.0,    # KL weight; increase to force more disentanglement
    "gamma"          : 0.1,    # Supervised clf weight; 0.0 = pure unsupervised VAE

    # Training
    "epochs"         : 150,
    "batch_size"     : 512,
    "lr"             : 1e-3,
    "weight_decay"   : 1e-5,
    "patience"       : 15,     # Early stopping on validation loss
    "kl_warmup"      : 20,     # Epochs to linearly ramp beta from 0 → beta
    "val_fraction"   : 0.15,   # Fraction of train used for validation

    # Reproducibility
    "seed"           : 123,
}

# ==============================================================================
# 4. Device & Seeds
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[03] Device: {DEVICE}\n")

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ==============================================================================
# 5. Load Data
# ==============================================================================

# Metadata columns carried through but NOT fed to the VAE
ID_COLS    = ["id", "y"]
TARGET_COL  = "y"

# Binary feature patterns — these are fed to the VAE but reconstructed with
# BCE rather than MSE (handled via n_binary split in the model)
## consec_decline_ are counts (0,1,2,...) not binary — treated as continuous.
## BCE requires values in [0,1] strictly; counts violate this.
BINARY_PATTERNS = ["^sector_", "^size_", "^is_", "^has_", "^groupmember$"]

def load_rds(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    return result[None]

print("[03] Loading data...")
train_df = load_rds(PATH_TRAIN)
test_df  = load_rds(PATH_TEST)

print(f"  Train : {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
print(f"  Test  : {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")

# Validate ID columns are present
for col in ID_COLS:
    assert col in train_df.columns, f"Missing ID column in train: '{col}'"
    assert col in test_df.columns,  f"Missing ID column in test:  '{col}'"

# ==============================================================================
# 6. Feature Classification
# ==============================================================================

import re

all_cols     = [c for c in train_df.columns if c not in ID_COLS]
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(train_df[c])]

def is_binary_col(name: str) -> bool:
    return any(re.match(p, name) for p in BINARY_PATTERNS)

bin_cols  = [c for c in numeric_cols if is_binary_col(c)]
cont_cols = [c for c in numeric_cols if not is_binary_col(c)]

# Enforce column order: continuous first, binary last
# This matches the decoder output split: out_cont | out_bin
col_order  = cont_cols + bin_cols
n_cont     = len(cont_cols)
n_binary   = len(bin_cols)
n_features = len(col_order)

print(f"\n[03] Feature classification:")
print(f"  Total      : {n_features}")
print(f"  Continuous : {n_cont}  (N(0,1) from R VAE files — no further PIT)")
print(f"  Binary     : {n_binary}  (reconstructed with BCE)")

# ==============================================================================
# 7. Build Arrays
# ==============================================================================

def df_to_array(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Extract columns, replace any residual Inf, return float32."""
    X = df[cols].values.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    # Median imputation per column using train stats (applied uniformly here;
    # imputation was already done in R — this is a safety fallback only)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = col_medians[j]
    return X.astype(np.float32)

X_train = df_to_array(train_df, col_order)
X_test  = df_to_array(test_df,  col_order)

y_train = train_df[TARGET_COL].values.astype(np.float32)
y_test  = test_df[TARGET_COL].values.astype(np.float32)

assert not np.isnan(X_train).any(), "NAs remain in X_train after fallback imputation"
assert not np.isnan(X_test).any(),  "NAs remain in X_test after fallback imputation"

print(f"\n[03] Array shapes:")
print(f"  X_train : {X_train.shape}  |  default rate: {y_train.mean():.4f}")
print(f"  X_test  : {X_test.shape}   |  default rate: {y_test.mean():.4f}")

# Verify N(0,1) distribution on continuous features (sanity check)
cont_mean = X_train[:, :n_cont].mean()
cont_std  = X_train[:, :n_cont].std()
print(f"\n[03] Continuous feature distribution (should be ≈ N(0,1)):")
print(f"  mean : {cont_mean:.4f}  (target ≈ 0)")
print(f"  std  : {cont_std:.4f}  (target ≈ 1)")
if abs(cont_mean) > 0.1 or abs(cont_std - 1.0) > 0.1:
    print("  ⚠ Distribution deviates from N(0,1) — check R pipeline output")

# ==============================================================================
# 8. Validation Split (stratified on target)
# ==============================================================================

y_strat = np.where(np.isnan(y_train), 0.0, y_train)

train_idx, val_idx = train_test_split(
    np.arange(len(X_train)),
    test_size=CFG["val_fraction"],
    random_state=CFG["seed"],
    stratify=y_strat
)

X_vae_train = X_train[train_idx]
X_vae_val   = X_train[val_idx]
y_vae_train = y_train[train_idx]
y_vae_val   = y_train[val_idx]

print(f"\n[03] VAE train/val split:")
print(f"  VAE train : {len(X_vae_train):,} rows")
print(f"  VAE val   : {len(X_vae_val):,} rows")

# PyTorch tensors
X_vae_train_t = torch.tensor(X_vae_train, dtype=torch.float32)
X_vae_val_t   = torch.tensor(X_vae_val,   dtype=torch.float32)
X_train_t     = torch.tensor(X_train,     dtype=torch.float32)
X_test_t      = torch.tensor(X_test,      dtype=torch.float32)
y_vae_train_t = torch.tensor(y_vae_train, dtype=torch.float32)
y_vae_val_t   = torch.tensor(y_vae_val,   dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_vae_train_t, y_vae_train_t),
    batch_size=CFG["batch_size"], shuffle=True, drop_last=False
)
val_loader = DataLoader(
    TensorDataset(X_vae_val_t, y_vae_val_t),
    batch_size=CFG["batch_size"], shuffle=False
)

print(f"  Train batches : {len(train_loader)}")
print(f"  Val batches   : {len(val_loader)}")

# ==============================================================================
# 9. Model Architecture
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, z_dim: int):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.net     = nn.Sequential(*layers)
        self.mu      = nn.Linear(in_d, z_dim)
        self.log_var = nn.Linear(in_d, z_dim)

    def forward(self, x):
        h    = self.net(x)
        z_mu = self.mu(h)
        z_lv = self.log_var(h).clamp(-10, 10)
        return z_mu, z_lv


class Decoder(nn.Module):
    def __init__(self, z_dim: int, hidden_dims: list,
                 n_cont: int, n_binary: int):
        super().__init__()
        layers, in_d = [], z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        self.net      = nn.Sequential(*layers)
        self.out_cont = nn.Linear(in_d, n_cont)   if n_cont   > 0 else None
        self.out_bin  = nn.Linear(in_d, n_binary) if n_binary > 0 else None

    def forward(self, z):
        h = self.net(z)
        parts = []
        if self.out_cont is not None:
            parts.append(self.out_cont(h))           # linear → N(0,1) space
        if self.out_bin is not None:
            parts.append(torch.sigmoid(self.out_bin(h)))  # sigmoid → (0,1)
        return torch.cat(parts, dim=1)


class ClassifierHead(nn.Module):
    def __init__(self, z_dim: int, hidden_dims: list):
        super().__init__()
        layers, in_d = [], z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_mu):
        return self.net(z_mu).squeeze(-1)


class BetaVAE(nn.Module):
    def __init__(self, input_dim, encoder_dims, decoder_dims,
                 classifier_dims, z_dim, n_cont, n_binary):
        super().__init__()
        self.encoder    = Encoder(input_dim, encoder_dims, z_dim)
        self.decoder    = Decoder(z_dim, decoder_dims, n_cont, n_binary)
        self.classifier = ClassifierHead(z_dim, classifier_dims)
        self.n_cont     = n_cont
        self.n_binary   = n_binary
        self.z_dim      = z_dim

    def reparameterise(self, z_mu, z_lv):
        std = torch.exp(0.5 * z_lv)
        return z_mu + torch.randn_like(std) * std

    def forward(self, x):
        z_mu, z_lv = self.encoder(x)
        z          = self.reparameterise(z_mu, z_lv)
        x_recon    = self.decoder(z)
        y_logit    = self.classifier(z_mu)
        return x_recon, z_mu, z_lv, y_logit

    def compute_loss(self, x, x_recon, z_mu, z_lv,
                     y_true, beta, gamma, labelled_mask):
        # Reconstruction: MSE for continuous (N(0,1) inputs + linear decoder)
        if self.n_cont > 0:
            mse = F.mse_loss(
                x_recon[:, :self.n_cont],
                x[:, :self.n_cont],
                reduction="none"
            ).sum(dim=1).mean()
        else:
            mse = torch.tensor(0.0, device=x.device)

        # Reconstruction: BCE for binary features
        if self.n_binary > 0:
            bce_recon = F.binary_cross_entropy(
                x_recon[:, self.n_cont:],
                x[:, self.n_cont:],
                reduction="none"
            ).sum(dim=1).mean()
        else:
            bce_recon = torch.tensor(0.0, device=x.device)

        recon_loss = mse + bce_recon

        # KL divergence vs N(0,1) prior
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_lv - z_mu.pow(2) - z_lv.exp(), dim=1)
        )

        # Supervised classification loss (labelled observations only)
        if gamma > 0 and labelled_mask.sum() > 0:
            logits_lab = self.classifier(z_mu)[labelled_mask]
            y_true_lab = y_true[labelled_mask]
            clf_loss   = F.binary_cross_entropy_with_logits(
                logits_lab, y_true_lab
            )
        else:
            clf_loss = torch.tensor(0.0, device=x.device)

        total = recon_loss + beta * kl_loss + gamma * clf_loss

        return {
            "total"    : total,
            "recon"    : recon_loss,
            "kl"       : kl_loss,
            "clf"      : clf_loss,
            "mse"      : mse,
            "bce_recon": bce_recon,
        }

# ==============================================================================
# 10. Training
# ==============================================================================

def get_beta(epoch: int, max_beta: float, warmup: int) -> float:
    if warmup <= 0:
        return max_beta
    return min(max_beta, max_beta * (epoch + 1) / warmup)


def train_vae(model, train_loader, val_loader, cfg, device):
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    history = {k: [] for k in [
        "train_total", "train_recon", "train_kl", "train_clf",
        "val_total",   "val_recon",   "val_kl",   "val_clf"
    ]}

    best_val_loss = float("inf")
    best_state    = None
    patience_ct   = 0

    for epoch in range(cfg["epochs"]):
        beta_now     = get_beta(epoch, cfg["beta"], cfg["kl_warmup"])
        epoch_losses = {k: 0.0 for k in ["total", "recon", "kl", "clf"]}
        n_batches    = 0

        model.train()
        for x_batch, y_batch in train_loader:
            x_batch  = x_batch.to(device)
            y_batch  = y_batch.to(device)
            lab_mask = ~torch.isnan(y_batch)

            optimiser.zero_grad()
            x_recon, z_mu, z_lv, _ = model(x_batch)
            losses = model.compute_loss(
                x_batch, x_recon, z_mu, z_lv, y_batch,
                beta=beta_now, gamma=cfg["gamma"], labelled_mask=lab_mask
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        for k in epoch_losses:
            history[f"train_{k}"].append(epoch_losses[k] / n_batches)

        # Validation
        val_losses    = {k: 0.0 for k in ["total", "recon", "kl", "clf"]}
        n_val_batches = 0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val    = x_val.to(device)
                y_val    = y_val.to(device)
                lab_mask = ~torch.isnan(y_val)
                x_recon, z_mu, z_lv, _ = model(x_val)
                v_losses = model.compute_loss(
                    x_val, x_recon, z_mu, z_lv, y_val,
                    beta=beta_now, gamma=cfg["gamma"], labelled_mask=lab_mask
                )
                for k in val_losses:
                    val_losses[k] += v_losses[k].item()
                n_val_batches += 1

        for k in val_losses:
            history[f"val_{k}"].append(val_losses[k] / n_val_batches)

        scheduler.step(history["val_total"][-1])

        if epoch % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
                f"β={beta_now:.3f} | "
                f"train={history['train_total'][-1]:.4f} | "
                f"val={history['val_total'][-1]:.4f} | "
                f"kl={history['train_kl'][-1]:.4f} | "
                f"clf={history['train_clf'][-1]:.4f}"
            )

        if history["val_total"][-1] < best_val_loss - 1e-4:
            best_val_loss = history["val_total"][-1]
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_ct   = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best val: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best weights restored (val loss: {best_val_loss:.4f})")

    return history

# ==============================================================================
# 11. Encoding
# ==============================================================================

@torch.no_grad()
def encode(model, X_tensor, device, batch_size=1024):
    """
    Returns:
        z_means  : (n, z_dim)  — deterministic latent means (used as features)
        recon_err: (n,)        — per-observation MSE reconstruction error
                                 computed in the N(0,1) space of continuous
                                 features only (binary reconstruction excluded
                                 to keep the anomaly score interpretable)
    """
    model.eval()
    z_list, err_list = [], []
    for (x_batch,) in DataLoader(TensorDataset(X_tensor), batch_size=batch_size):
        x_batch    = x_batch.to(device)
        z_mu, z_lv = model.encoder(x_batch)
        z_samp     = model.reparameterise(z_mu, z_lv)
        x_recon    = model.decoder(z_samp)

        # Anomaly score: MSE on continuous features only
        if model.n_cont > 0:
            err = F.mse_loss(
                x_recon[:, :model.n_cont],
                x_batch[:, :model.n_cont],
                reduction="none"
            ).mean(dim=1)
        else:
            err = torch.zeros(x_batch.size(0), device=device)

        z_list.append(z_mu.cpu().numpy())
        err_list.append(err.cpu().numpy())

    return np.vstack(z_list), np.concatenate(err_list)

# ==============================================================================
# 12. Diagnostics & Plots
# ==============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history["train_total"], label="Train", linewidth=1.5)
    axes[0].plot(history["val_total"],   label="Val",   linewidth=1.5, ls="--")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_recon"], label="Train", linewidth=1.5)
    axes[1].plot(history["val_recon"],   label="Val",   linewidth=1.5, ls="--")
    axes[1].set_title("Reconstruction Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(history["train_kl"],  label="KL",  linewidth=1.5, color="orange")
    axes[2].plot(history["train_clf"], label="Clf", linewidth=1.5, color="red")
    axes[2].set_title("KL & Classification Loss"); axes[2].legend(); axes[2].grid(alpha=0.3)
    for ax in axes: ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path.name}")


def plot_latent_space(z_means, y_labels, save_path, n_pairs=4):
    valid  = ~np.isnan(y_labels)
    z_v, y_v = z_means[valid], y_labels[valid]
    n_dims    = z_means.shape[1]
    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4))
    for i in range(n_pairs):
        j = min(i + 1, n_dims - 1)
        axes[i].scatter(z_v[y_v == 0, i], z_v[y_v == 0, j],
                        alpha=0.1, s=2, c="steelblue", label="Non-default")
        axes[i].scatter(z_v[y_v == 1, i], z_v[y_v == 1, j],
                        alpha=0.4, s=6, c="crimson",   label="Default")
        axes[i].set_title(f"z{i+1} vs z{j+1}")
        axes[i].set_xlabel(f"z{i+1}"); axes[i].set_ylabel(f"z{j+1}")
    axes[0].legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path.name}")


def print_latent_diagnostics(z_train, y_train, err_train, err_test):
    dim_var     = z_train.var(axis=0)
    n_active    = int((dim_var >= 1e-3).sum())
    n_collapsed = int((dim_var  < 1e-3).sum())
    print(f"\n[03] Latent space diagnostics:")
    print(f"  z_dim              : {z_train.shape[1]}")
    print(f"  Active dims        : {n_active}   (var >= 1e-3)")
    print(f"  Collapsed dims     : {n_collapsed}  (var <  1e-3)  target: ≤ 3")
    print(f"  Dim var range      : [{dim_var.min():.4f}, {dim_var.max():.4f}]")

    valid      = ~np.isnan(y_train)
    def_err    = err_train[valid & (y_train == 1)].mean()
    nondef_err = err_train[valid & (y_train == 0)].mean()
    ratio      = def_err / nondef_err if nondef_err > 0 else float("nan")
    print(f"  Recon err default  : {def_err:.4f}")
    print(f"  Recon err non-def  : {nondef_err:.4f}")
    print(f"  Ratio def/non-def  : {ratio:.3f}  (>1.0 → VAE learned anomaly signal)")
    print(f"  Test recon p95     : {np.percentile(err_test, 95):.4f}")

    if n_collapsed > 5:
        print(f"  ⚠ >5 collapsed dims — consider reducing beta (current: {CFG['beta']})")
    if ratio < 1.0:
        print(f"  ⚠ Ratio < 1.0 — VAE not learning default anomaly signal")

# ==============================================================================
# 13. Main
# ==============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"[03] Beta-VAE — Credit Default  [{SPLIT_MODE}]")
    print(f"{'='*60}")
    print(f"  Input features : {n_features}")
    print(f"  Continuous     : {n_cont}  (N(0,1), MSE reconstruction)")
    print(f"  Binary         : {n_binary}  (BCE reconstruction)")
    print(f"  z_dim          : {CFG['z_dim']}")
    print(f"  beta           : {CFG['beta']}")
    print(f"  gamma          : {CFG['gamma']}  "
          f"({'supervised' if CFG['gamma'] > 0 else 'pure VAE'})")

    # Build model
    model = BetaVAE(
        input_dim       = n_features,
        encoder_dims    = CFG["encoder_dims"],
        decoder_dims    = CFG["decoder_dims"],
        classifier_dims = CFG["classifier_dims"],
        z_dim           = CFG["z_dim"],
        n_cont          = n_cont,
        n_binary        = n_binary,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters : {n_params:,}")
    print(f"  Encoder    : {n_features} → "
          f"{' → '.join(str(d) for d in CFG['encoder_dims'])} → z({CFG['z_dim']})")
    print(f"  Decoder    : z({CFG['z_dim']}) → "
          f"{' → '.join(str(d) for d in CFG['decoder_dims'])} → {n_features}")

    # Train
    print(f"\n[03] Training...")
    history = train_vae(model, train_loader, val_loader, CFG, DEVICE)

    # Save model
    model_dir = DIR_MODELS / SPLIT_MODE
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(),          model_dir / "vae_weights.pt")
    torch.save(model.encoder.state_dict(),  model_dir / "encoder_weights.pt")

    cfg_save = {
        **CFG,
        "split_mode" : SPLIT_MODE,
        "n_features" : n_features,
        "n_cont"     : n_cont,
        "n_binary"   : n_binary,
        "col_order"  : col_order,
        "bin_cols"   : bin_cols,
        "cont_cols"  : cont_cols,
        "input_dist" : "normal",   # R pipeline applied qnorm()
    }
    with open(model_dir / "vae_config.json", "w") as f:
        json.dump(cfg_save, f, indent=2)
    print(f"\n[03] Model saved → {model_dir}")

    # Encode
    print("\n[03] Encoding train / test...")
    z_train, err_train = encode(model, X_train_t, DEVICE)
    z_test,  err_test  = encode(model, X_test_t,  DEVICE)

    # Diagnostics & plots
    print_latent_diagnostics(z_train, y_train, err_train, err_test)
    plot_training_curves(history, DIR_FIG / f"training_curves_{SPLIT_MODE}.png")
    plot_latent_space(z_train, y_train, DIR_FIG / f"latent_space_{SPLIT_MODE}.png")

    # ── Assemble output dataframes ────────────────────────────────────────────
    z_cols = [f"z{i+1}" for i in range(CFG["z_dim"])]

    def build_output(src_df, z_arr, err_arr):
        """Assemble id columns + latent dims + anomaly score."""
        id_part  = src_df[ID_COLS].reset_index(drop=True)
        z_part   = pd.DataFrame(z_arr,  columns=z_cols)
        err_part = pd.Series(err_arr, name="vae_recon_error")
        return pd.concat([id_part, z_part, err_part], axis=1)

    latent_train = build_output(train_df, z_train, err_train)
    latent_test  = build_output(test_df,  z_test,  err_test)

    # Anomaly-score-only versions (drop z cols)
    anomaly_train = latent_train[ID_COLS + ["vae_recon_error"]].copy()
    anomaly_test  = latent_test[ ID_COLS + ["vae_recon_error"]].copy()

    # ── Save ──────────────────────────────────────────────────────────────────
    outputs = {
        f"latent_train{FILE_SUFFIX}.parquet" : latent_train,
        f"latent_test{FILE_SUFFIX}.parquet"  : latent_test,
        f"anomaly_train{FILE_SUFFIX}.parquet": anomaly_train,
        f"anomaly_test{FILE_SUFFIX}.parquet" : anomaly_test,
    }
    print(f"\n[03] Saving outputs → {DIR_LAT}")
    for fname, df_out in outputs.items():
        path = DIR_LAT / fname
        df_out.to_parquet(path, index=False)
        print(f"  {fname:<45} {df_out.shape[0]:>7,} rows × {df_out.shape[1]} cols")

    # ── Final assertions ──────────────────────────────────────────────────────
    for fname, df_out in outputs.items():
        assert df_out[["id", "y"]].isna().sum().sum() == 0, \
            f"NAs in ID cols of {fname}"
        assert df_out["vae_recon_error"].isna().sum() == 0, \
            f"NAs in vae_recon_error of {fname}"

    lat_cols = z_cols + ["vae_recon_error"]
    assert latent_train[lat_cols].isna().sum().sum() == 0, \
        "NAs in latent train features"
    assert latent_test[lat_cols].isna().sum().sum() == 0, \
        "NAs in latent test features"

    n_collapsed = int((z_train.var(axis=0) < 1e-3).sum())
    print(f"\n[03] Collapsed dims: {n_collapsed} / {CFG['z_dim']}")
    if n_collapsed > 5:
        print(f"  Consider reducing beta from {CFG['beta']} → {CFG['beta']*0.5:.2f}")

    print("\n[03] All assertions passed.")
    print(f"[03] DONE [{SPLIT_MODE}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()