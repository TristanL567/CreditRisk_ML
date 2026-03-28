"""
05_AutoGluon.py
===============
AutoGluon AutoML for credit default prediction.

SPLIT SELECTION:
    Set SPLIT_MODE at the top — routes to the correct R pipeline outputs.
    "OoS" → firm-level random split
    "OoT" → out-of-time split (last N years to test)

MODEL SELECTION:
    Set MODEL at the top — controls which feature set is used.
    Each run saves to its own subdirectory; no run overwrites another.

    M1 — "raw"        : Uniform(0,1) features from R pipeline (~508 features)
    M2 — "latent"     : VAE latent dims + reconstruction error (z1..z32 + recon_error)
    M3 — "anomaly"    : Reconstruction error only (vae_recon_error)
    M4 — "augmented"  : Raw uniform features + VAE latent dims + recon error combined
                        AutoGluon selects the best combination internally

INPUTS:
    M1        : 02_train_final_{split}.rds, 02_test_final_{split}.rds
    M2        : latent_train_{split}.parquet, latent_test_{split}.parquet
    M3        : anomaly_train_{split}.parquet, anomaly_test_{split}.parquet
    M4        : all of the above, joined on id

OUTPUTS (saved to 03_Output/AutoGluon/{MODEL}_{SPLIT_MODE}/):
    ag_predictor/              AutoGluon model weights (full predictor)
    predictions_test.parquet   id, y, p_default, split_mode, model_name, year
    eval_summary.json          AUC-ROC, AP, Brier, BSS, Recall@FPR metrics
    feature_importance.csv     permutation feature importance

METRICS:
    AUC-ROC, Average Precision, Brier Score, Brier Skill Score (BSS),
    Recall@FPR 1/3/5/10%

    BSS = 1 - BS / BS_climatology
    BS_climatology = prevalence * (1-prevalence)  [always predicting the mean]
    BSS > 0 means the model beats the naive baseline.

NOTE ON YEAR COLUMN:
    If 'year' is not present in the feature files, the predictions output will
    contain year=NA. To include year, add it to the saved .rds files in 02E
    by keeping it in cols_to_keep (remove "^year$" from drop_patterns).
"""

# ==============================================================================
# 0. Imports
# ==============================================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

try:
    from autogluon.tabular import TabularDataset, TabularPredictor
except ImportError:
    raise ImportError(
        "AutoGluon not installed.\n"
        "Run: pip install autogluon.tabular"
    )

# ==============================================================================
# 1. Configuration  ← CHANGE HERE
# ==============================================================================

SPLIT_MODE  = "OoS"   # "OoS" | "OoT"
MODEL_GROUP = "01"    # "01" | "02" | "03" | "04" | "05"

# Allow CLI override: python 05_AutoGluon.py <MODEL_GROUP> <SPLIT_MODE>
if len(sys.argv) >= 3:
    MODEL_GROUP = sys.argv[1]
    SPLIT_MODE  = sys.argv[2]

# AutoGluon settings
TIME_LIMIT  = 1800       # seconds per run (default: 30 min)
PRESET      = "good_quality"
EVAL_METRIC = "roc_auc"  # used by AutoGluon internally for optimization
TARGET_COL  = "y"

assert SPLIT_MODE in ("OoS", "OoT"), \
    f"SPLIT_MODE must be 'OoS' or 'OoT', got: '{SPLIT_MODE}'"
assert MODEL_GROUP in ("01", "02", "03", "04", "05"), \
    f"MODEL_GROUP must be one of 01–05, got: '{MODEL_GROUP}'"

SPLIT_LETTER = "a" if SPLIT_MODE == "OoS" else "b"
RUN_NAME     = f"{MODEL_GROUP}{SPLIT_LETTER}_AutoGluon"

# Feature config derived from MODEL_GROUP — must stay consistent with config.R
_KF_MAP  = {"01": "f", "02": "r", "03": "r", "04": "r", "05": "r"}
_TD_MAP  = {"01": False, "02": False, "03": True, "04": True, "05": True}
_kf      = _KF_MAP[MODEL_GROUP]
_td      = "TD" if _TD_MAP[MODEL_GROUP] else "noTD"
FEAT_SUFFIX = f"_{_kf}_{_td}"
FILE_SUFFIX = f"{FEAT_SUFFIX}_{SPLIT_MODE}"   # e.g. "_r_TD_OoS"

MODEL_GROUP_DESCRIPTIONS = {
    "01": "Raw Balance Sheet + Sector Information",
    "02": "Financial Ratios + Sector Information",
    "03": "Financial Ratios + Sector Information + Time Dynamics",
    "04": "Financial Ratios + Sector Information + Time Dynamics + Latent Features (VAE)",
    "05": "Latent Features (VAE) Only",
}

# ==============================================================================
# 2. Paths
# ==============================================================================

DATA_ROOT = Path(r"C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML")
DIR_DATA  = DATA_ROOT / "02_Data"
DIR_LAT   = DATA_ROOT / "03_Output" / "Latent"
DIR_OUT   = DATA_ROOT / "03_Output" / "Final"

DIR_RUN   = DIR_OUT / RUN_NAME
DIR_RUN.mkdir(parents=True, exist_ok=True)

# Input file paths
PATH_RAW_TRAIN    = DIR_DATA / f"02_train_final{FILE_SUFFIX}.rds"
PATH_RAW_TEST     = DIR_DATA / f"02_test_final{FILE_SUFFIX}.rds"
PATH_LATENT_TRAIN = DIR_LAT  / f"latent_train{FILE_SUFFIX}.parquet"
PATH_LATENT_TEST  = DIR_LAT  / f"latent_test{FILE_SUFFIX}.parquet"

print(f"[05] ══════════════════════════════════════════")
print(f"  RUN        : {RUN_NAME}")
print(f"  SPLIT_MODE : {SPLIT_MODE}")
print(f"  GROUP      : {MODEL_GROUP}  —  {MODEL_GROUP_DESCRIPTIONS[MODEL_GROUP]}")
print(f"  Preset     : {PRESET}")
print(f"  Time limit : {TIME_LIMIT}s ({TIME_LIMIT//60} min)")
print(f"  Output dir : {DIR_RUN}")
print(f"[05] ══════════════════════════════════════════\n")

# ==============================================================================
# 3. Data Loading Helpers
# ==============================================================================

def load_rds(path: Path) -> pd.DataFrame:
    assert path.exists(), (
        f"File not found: {path}\n"
        f"Run 02_FeatureEngineering.R with SPLIT_MODE='{SPLIT_MODE}' first."
    )
    result = pyreadr.read_r(str(path))
    df = result[None]
    print(f"  Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_parquet(path: Path, source: str) -> pd.DataFrame:
    assert path.exists(), (
        f"File not found: {path}\n"
        f"Run 03_Autoencoder.py with SPLIT_MODE='{SPLIT_MODE}' first."
    )
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def get_year(df: pd.DataFrame) -> pd.Series:
    """Extract year from df if available, else return NA series."""
    if "year" in df.columns:
        return df["year"].copy()
    return pd.Series([pd.NA] * len(df), name="year")

# ==============================================================================
# 4. Load Data by Model
# ==============================================================================

print(f"[05] Loading data for {RUN_NAME}...")

# Columns from latent files that are NOT features
LAT_META_COLS = ["id", "y"]

# Helper: load id vectors for raw feature files (id dropped in 02E)
def load_id_vectors():
    id_path_train = DIR_DATA / f"02_train_id_vec{FILE_SUFFIX}.rds"
    id_path_test  = DIR_DATA / f"02_test_id_vec{FILE_SUFFIX}.rds"
    assert id_path_train.exists(), f"id vector not found: {id_path_train}"
    id_tr = pyreadr.read_r(str(id_path_train))[None].squeeze()
    id_te = pyreadr.read_r(str(id_path_test))[None].squeeze()
    print(f"  Loaded id vectors from .rds")
    return id_tr, id_te

if MODEL_GROUP in ("01", "02", "03"):
    # Base features only (raw uniform features from R pipeline)
    train_raw = load_rds(PATH_RAW_TRAIN)
    test_raw  = load_rds(PATH_RAW_TEST)

    year_train = get_year(train_raw)
    year_test  = get_year(test_raw)

    if "id" in train_raw.columns:
        id_train = train_raw["id"].copy()
        id_test  = test_raw["id"].copy()
    else:
        id_train, id_test = load_id_vectors()

    train_df = train_raw.copy()
    test_df  = test_raw.copy()

elif MODEL_GROUP == "04":
    # Base features + VAE latent features (augmented)
    train_raw = load_rds(PATH_RAW_TRAIN)
    test_raw  = load_rds(PATH_RAW_TEST)
    train_lat = load_parquet(PATH_LATENT_TRAIN, "latent_train")
    test_lat  = load_parquet(PATH_LATENT_TEST,  "latent_test")

    year_train = get_year(train_lat)
    year_test  = get_year(test_lat)
    id_train   = train_lat["id"].copy()
    id_test    = test_lat["id"].copy()

    raw_id_train, raw_id_test = load_id_vectors()
    train_raw = train_raw.copy()
    test_raw  = test_raw.copy()
    train_raw["id"] = raw_id_train.values
    test_raw["id"]  = raw_id_test.values

    lat_feat_cols = [c for c in train_lat.columns if c not in LAT_META_COLS]

    assert len(train_raw) == len(train_lat), (
        f"Group 04 row mismatch: raw={len(train_raw)} vs latent={len(train_lat)}"
    )
    assert len(test_raw) == len(test_lat), (
        f"Group 04 row mismatch: raw={len(test_raw)} vs latent={len(test_lat)}"
    )

    train_raw = train_raw.reset_index(drop=True)
    test_raw  = test_raw.reset_index(drop=True)
    train_lat = train_lat.reset_index(drop=True)
    test_lat  = test_lat.reset_index(drop=True)

    train_df = pd.concat(
        [train_raw.drop(columns=["id"], errors="ignore"),
         train_lat[lat_feat_cols]],
        axis=1
    )
    test_df = pd.concat(
        [test_raw.drop(columns=["id"], errors="ignore"),
         test_lat[lat_feat_cols]],
        axis=1
    )
    print(f"  Group 04 cbind: train {train_df.shape} | test {test_df.shape}")

elif MODEL_GROUP == "05":
    # VAE latent features + categorical variables (sector_*, size_*, groupmember, public)
    train_raw = load_rds(PATH_RAW_TRAIN)
    test_raw  = load_rds(PATH_RAW_TEST)
    train_lat = load_parquet(PATH_LATENT_TRAIN, "latent_train")
    test_lat  = load_parquet(PATH_LATENT_TEST,  "latent_test")

    year_train = get_year(train_lat)
    year_test  = get_year(test_lat)
    id_train   = train_lat["id"].copy()
    id_test    = test_lat["id"].copy()

    lat_feat_cols = [c for c in train_lat.columns if c not in LAT_META_COLS]

    import re
    _cat_re   = re.compile(r"^(sector_|size_|groupmember$|public$)")
    cat_cols  = [c for c in train_raw.columns if _cat_re.match(c)]

    assert len(train_raw) == len(train_lat), (
        f"Group 05 row mismatch: raw={len(train_raw)} vs latent={len(train_lat)}"
    )
    assert len(test_raw) == len(test_lat), (
        f"Group 05 row mismatch: raw={len(test_raw)} vs latent={len(test_lat)}"
    )

    train_df = pd.concat([
        train_lat[lat_feat_cols + [TARGET_COL]].reset_index(drop=True),
        train_raw[cat_cols].reset_index(drop=True),
    ], axis=1)
    test_df = pd.concat([
        test_lat[ lat_feat_cols + [TARGET_COL]].reset_index(drop=True),
        test_raw[ cat_cols].reset_index(drop=True),
    ], axis=1)

    print(f"  Group 05: {len(lat_feat_cols)} latent + {len(cat_cols)} categorical cols")

# Drop id if it survived into the modelling df
for col in ["id"]:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# ==============================================================================
# 5. Validate
# ==============================================================================

assert TARGET_COL in train_df.columns, f"Target '{TARGET_COL}' missing from train"
assert TARGET_COL in test_df.columns,  f"Target '{TARGET_COL}' missing from test"

feature_cols = [c for c in train_df.columns if c != TARGET_COL]

print(f"\n[05] Data ready:")
print(f"  Train : {train_df.shape[0]:,} rows × {len(feature_cols)} features")
print(f"  Test  : {test_df.shape[0]:,} rows × {len(feature_cols)} features")
print(f"  Train default rate : {train_df[TARGET_COL].mean():.4f}")
print(f"  Test  default rate : {test_df[TARGET_COL].mean():.4f}")

na_train = train_df[feature_cols].isna().sum().sum()
na_test  = test_df[feature_cols].isna().sum().sum()
if na_train > 0 or na_test > 0:
    print(f"  ⚠ NAs in features — Train: {na_train}  Test: {na_test}")
    print(f"    AutoGluon will handle these internally.")

# ==============================================================================
# 6. Evaluation Metrics
# ==============================================================================

def recall_at_fpr(y_true: np.ndarray, y_pred: np.ndarray,
                  fpr_target: float) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eligible = np.where(fpr <= fpr_target)[0]
    return 0.0 if len(eligible) == 0 else float(tpr[eligible].max())


def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    BSS = 1 - BS / BS_climatology
    BS_climatology = prevalence * (1 - prevalence)
    A BSS > 0 means the model beats always predicting the base rate.
    BSS = 1 is perfect. BSS = 0 equals the naive baseline.
    """
    bs           = float(np.mean((y_pred - y_true) ** 2))
    prevalence   = float(y_true.mean())
    bs_clim      = prevalence * (1.0 - prevalence)
    if bs_clim == 0:
        return float("nan")
    return round(1.0 - bs / bs_clim, 4)


def compute_metrics(y_true, y_pred, set_name: str) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    valid  = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]

    if len(np.unique(yt)) < 2:
        print(f"  ⚠ Only one class in {set_name} — metrics skipped")
        return {}

    return {
        "set"          : set_name,
        "model"        : RUN_NAME,
        "split_mode"   : SPLIT_MODE,
        "n_obs"        : int(len(yt)),
        "n_defaults"   : int(yt.sum()),
        "prevalence"   : round(float(yt.mean()), 5),
        "auc_roc"      : round(float(roc_auc_score(yt, yp)), 4),
        "avg_precision": round(float(average_precision_score(yt, yp)), 4),
        "brier"        : round(float(np.mean((yp - yt) ** 2)), 5),
        "bss"          : brier_skill_score(yt, yp),
        "recall_fpr1"  : round(recall_at_fpr(yt, yp, 0.01), 4),
        "recall_fpr3"  : round(recall_at_fpr(yt, yp, 0.03), 4),
        "recall_fpr5"  : round(recall_at_fpr(yt, yp, 0.05), 4),
        "recall_fpr10" : round(recall_at_fpr(yt, yp, 0.10), 4),
    }

# ==============================================================================
# 7. Train AutoGluon
# ==============================================================================

print(f"\n[05] Training AutoGluon [{RUN_NAME}]...")
print(f"  Preset     : {PRESET}")
print(f"  Time limit : {TIME_LIMIT}s")
print(f"  Features   : {len(feature_cols)}")

ag_train = TabularDataset(train_df.reset_index(drop=True))
ag_test  = TabularDataset(test_df.reset_index(drop=True))

predictor = TabularPredictor(
    label        = TARGET_COL,
    problem_type = "binary",
    eval_metric  = EVAL_METRIC,
    path         = str(DIR_RUN / "ag_predictor"),
    verbosity    = 2,
).fit(
    train_data       = ag_train,
    presets          = PRESET,
    time_limit       = TIME_LIMIT,
    num_bag_folds    = 5,     # bagging for more robust estimates
    num_stack_levels = 1,     # one level of stacking
)

print(f"\n[05] Training complete.")
print(predictor.leaderboard(ag_test, silent=True).to_string())

# ==============================================================================
# 8. Predict on Test Set
# ==============================================================================

print(f"\n[05] Predicting on test set...")

# predict_proba returns P(class=1) for binary classification
proba_test = predictor.predict_proba(ag_test, as_multiclass=False)

# Assemble prediction dataframe
preds_test = pd.DataFrame({
    "id"          : id_test.values   if hasattr(id_test, "values") else id_test,
    "y"           : test_df[TARGET_COL].values,
    "p_default"   : proba_test.values,
    "split_mode"  : SPLIT_MODE,
    "model_name"  : RUN_NAME,
    "year"        : year_test.values if hasattr(year_test, "values") else year_test,
})

# ==============================================================================
# 9. Evaluate
# ==============================================================================

print(f"\n[05] Evaluation [{RUN_NAME}]:")

metrics_test = compute_metrics(
    preds_test["y"], preds_test["p_default"], f"test_{SPLIT_MODE}"
)

if metrics_test:
    print(f"  AUC-ROC    : {metrics_test['auc_roc']:.4f}")
    print(f"  Avg Prec   : {metrics_test['avg_precision']:.4f}")
    print(f"  Brier      : {metrics_test['brier']:.5f}")
    print(f"  BSS        : {metrics_test['bss']:.4f}  (>0 beats naive baseline)")
    print(f"  R@FPR1%    : {metrics_test['recall_fpr1']:.4f}")
    print(f"  R@FPR3%    : {metrics_test['recall_fpr3']:.4f}")
    print(f"  R@FPR5%    : {metrics_test['recall_fpr5']:.4f}")
    print(f"  R@FPR10%   : {metrics_test['recall_fpr10']:.4f}")

# ==============================================================================
# 10. Feature Importance
# ==============================================================================

print(f"\n[05] Computing feature importance...")
try:
    importance_df = predictor.feature_importance(ag_test, silent=True)
    importance_df = importance_df.reset_index()
    importance_df.columns = ["feature"] + list(importance_df.columns[1:])
    importance_df.to_csv(DIR_RUN / "feature_importance.csv", index=False)
    print(f"  Top 10 features:")
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"  ⚠ Feature importance failed: {e}")
    importance_df = pd.DataFrame()

# ==============================================================================
# 11. Save Outputs
# ==============================================================================

print(f"\n[05] Saving outputs → {DIR_RUN}")

# Predictions
pred_path = DIR_RUN / "predictions_test.parquet"
preds_test.to_parquet(pred_path, index=False)
print(f"  predictions_test.parquet  {preds_test.shape[0]:,} rows × {preds_test.shape[1]} cols")

# Eval summary
eval_summary = {
    "model"           : RUN_NAME,
    "model_description": MODEL_GROUP_DESCRIPTIONS[MODEL_GROUP],
    "split_mode"      : SPLIT_MODE,
    "preset"          : PRESET,
    "time_limit_s"    : TIME_LIMIT,
    "eval_metric_ag"  : EVAL_METRIC,
    "n_features"      : len(feature_cols),
    "n_train"         : len(train_df),
    "n_test"          : len(test_df),
    "metrics"         : metrics_test,
}

summary_path = DIR_RUN / "eval_summary.json"
with open(summary_path, "w") as f:
    json.dump(eval_summary, f, indent=2)
print(f"  eval_summary.json")

if not importance_df.empty:
    print(f"  feature_importance.csv    {len(importance_df)} features")

print(f"\n[05] ══ RESULTS: {RUN_NAME} ══")
print(f"  {'Metric':<20} {'Value'}")
print(f"  {'-'*30}")
if metrics_test:
    for k, v in metrics_test.items():
        if k not in ("set", "model", "split_mode"):
            print(f"  {k:<20} {v}")

print(f"\n[05] DONE [{RUN_NAME}]")
print(f"  Model  : {DIR_RUN / 'ag_predictor'}")
print(f"  Tables : {DIR_RUN}")