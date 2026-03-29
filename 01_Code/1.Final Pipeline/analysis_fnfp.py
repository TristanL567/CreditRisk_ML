"""
analysis_fnfp.py
Deep FN/FP breakdown by sector, size, history length, time index
for best OoS (03a) and best OoT (03b) AutoGluon models.
"""
import pandas as pd
import numpy as np
import pyreadr
import warnings
warnings.filterwarnings("ignore")

BASE_DATA = r"C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML\02_Data"
BASE_OUT  = r"C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML\03_Output\Final"

SECTOR_COLS = [
    "sector_wholesale", "sector_manufacture", "sector_construction",
    "sector_real_estate", "sector_retail", "sector_energy"
]

def decode_sector(df):
    sec = pd.Series("service", index=df.index)
    for col in SECTOR_COLS:
        if col in df.columns:
            sec[df[col] == 1] = col.replace("sector_", "")
    return sec

def classify(df, threshold):
    df = df.copy()
    df["pred_pos"] = (df["p_default"] >= threshold).astype(int)
    df["outcome"] = "TN"
    df.loc[(df["y"] == 1) & (df["pred_pos"] == 1), "outcome"] = "TP"
    df.loc[(df["y"] == 1) & (df["pred_pos"] == 0), "outcome"] = "FN"
    df.loc[(df["y"] == 0) & (df["pred_pos"] == 1), "outcome"] = "FP"
    return df

def pct(n, d):
    return round(100 * n / d, 1) if d > 0 else np.nan

def med(series):
    return round(series.median(), 5) if len(series) > 0 else np.nan

RUNS = [
    ("03a", "OoS", "_r_TD_OoS"),
    ("03b", "OoT", "_r_TD_OoT"),
]

for run, split, feat_suffix in RUNS:
    print(f"\n{'='*72}")
    print(f"  MODEL: {run}  SPLIT: {split}")
    print(f"{'='*72}")

    pred = pd.read_parquet(f"{BASE_OUT}\\{run}_AutoGluon\\predictions_test.parquet")
    feat = pyreadr.read_r(f"{BASE_DATA}\\02_test_final{feat_suffix}.rds")[None]
    ids  = pyreadr.read_r(f"{BASE_DATA}\\02_test_id_vec{feat_suffix}.rds")[None].squeeze()
    feat["id"] = ids.values

    df = pred.copy()
    df["sector"]         = decode_sector(feat).values
    df["is_tiny"]        = (feat["size_Tiny"].values == 1) if "size_Tiny" in feat.columns else False
    df["history_length"] = feat["history_length"].values
    df["time_index"]     = feat["time_index"].values
    df["groupmember"]    = feat["groupmember"].values if "groupmember" in feat.columns else 0
    df["public"]         = feat["public"].values if "public" in feat.columns else 0

    # Key financial ratios for FN/FP profiling
    for r in ["r8", "r9", "r12", "r14", "r6", "r7"]:
        if r in feat.columns:
            df[r] = feat[r].values

    df["hl_q"] = pd.qcut(df["history_length"], q=5, duplicates="drop")
    df["ti_q"] = pd.qcut(df["time_index"],     q=5, duplicates="drop")

    thresh = df["p_default"].quantile(0.95)
    df = classify(df, thresh)

    fn_df = df[df["outcome"] == "FN"].copy()
    fp_df = df[df["outcome"] == "FP"].copy()
    tp_df = df[df["outcome"] == "TP"].copy()
    tn_df = df[df["outcome"] == "TN"].copy()

    n_def = (df["y"] == 1).sum()
    n_non = (df["y"] == 0).sum()

    print(f"\n  Threshold (p95): {thresh:.5f}")
    print(f"  Defaults: {n_def}  |  Non-defaults: {n_non}")
    print(f"  TP: {len(tp_df)}  FN: {len(fn_df)}  FP: {len(fp_df)}  TN: {len(tn_df)}")
    print(f"  Overall recall: {pct(len(tp_df), n_def)}%  |  FP rate: {pct(len(fp_df), n_non)}%")

    # ── 1. Sector breakdown ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  1. SECTOR — recall, FP rate, median scores")
    print(f"{'─'*72}")

    rows = []
    for sec, g in df.groupby("sector"):
        nd = (g["y"] == 1).sum()
        nn = (g["y"] == 0).sum()
        rows.append({
            "sector"          : sec,
            "n_obs"           : len(g),
            "n_defaults"      : nd,
            "prevalence_%"    : round(100 * nd / len(g), 3),
            "recall_%"        : pct((g["outcome"] == "TP").sum(), nd),
            "FN_rate_%"       : pct((g["outcome"] == "FN").sum(), nd),
            "FP_rate_%"       : pct((g["outcome"] == "FP").sum(), nn),
            "med_score_def"   : med(g.loc[g["y"] == 1, "p_default"]),
            "med_score_nondef": med(g.loc[g["y"] == 0, "p_default"]),
        })
    sec_df = pd.DataFrame(rows).sort_values("FN_rate_%", ascending=True)
    print(sec_df.to_string(index=False))

    # ── 2. Size ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  2. SIZE (Tiny vs Non-Tiny)")
    print(f"{'─'*72}")
    rows = []
    for size_val, lbl in [(True, "Tiny"), (False, "Non-Tiny")]:
        g = df[df["is_tiny"] == size_val]
        nd = (g["y"] == 1).sum()
        nn = (g["y"] == 0).sum()
        rows.append({
            "size"         : lbl,
            "n_obs"        : len(g),
            "n_defaults"   : nd,
            "prevalence_%" : round(100 * nd / len(g), 3) if len(g) > 0 else np.nan,
            "recall_%"     : pct((g["outcome"] == "TP").sum(), nd),
            "FN_rate_%"    : pct((g["outcome"] == "FN").sum(), nd),
            "FP_rate_%"    : pct((g["outcome"] == "FP").sum(), nn),
            "med_score_def": med(g.loc[g["y"] == 1, "p_default"]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 3. History length quintiles ──────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  3. HISTORY LENGTH quintiles")
    print(f"{'─'*72}")
    rows = []
    for q, g in df.groupby("hl_q", observed=True):
        nd = (g["y"] == 1).sum()
        nn = (g["y"] == 0).sum()
        rows.append({
            "quintile"     : str(q),
            "med_hl"       : round(g["history_length"].median(), 1),
            "n_obs"        : len(g),
            "n_defaults"   : nd,
            "prevalence_%" : round(100 * nd / len(g), 3) if len(g) > 0 else np.nan,
            "recall_%"     : pct((g["outcome"] == "TP").sum(), nd),
            "FN_rate_%"    : pct((g["outcome"] == "FN").sum(), nd),
            "FP_rate_%"    : pct((g["outcome"] == "FP").sum(), nn),
            "med_score_def": med(g.loc[g["y"] == 1, "p_default"]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 4. Time index quintiles ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  4. TIME INDEX quintiles (position in panel history per firm)")
    print(f"{'─'*72}")
    rows = []
    for q, g in df.groupby("ti_q", observed=True):
        nd = (g["y"] == 1).sum()
        nn = (g["y"] == 0).sum()
        rows.append({
            "quintile"     : str(q),
            "med_ti"       : round(g["time_index"].median(), 1),
            "n_obs"        : len(g),
            "n_defaults"   : nd,
            "prevalence_%" : round(100 * nd / len(g), 3),
            "recall_%"     : pct((g["outcome"] == "TP").sum(), nd),
            "FN_rate_%"    : pct((g["outcome"] == "FN").sum(), nd),
            "FP_rate_%"    : pct((g["outcome"] == "FP").sum(), nn),
            "med_score_def": med(g.loc[g["y"] == 1, "p_default"]),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 5. Group member & Public ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  5. GROUP MEMBER  &  PUBLIC flags")
    print(f"{'─'*72}")
    for flag, lbl in [("groupmember", "Group Member"), ("public", "Public")]:
        rows = []
        for val, label in [(1, "Yes"), (0, "No")]:
            g = df[df[flag] == val]
            nd = (g["y"] == 1).sum()
            nn = (g["y"] == 0).sum()
            rows.append({
                lbl          : label,
                "n_obs"      : len(g),
                "n_defaults" : nd,
                "prev_%"     : round(100 * nd / len(g), 3) if len(g) > 0 else np.nan,
                "recall_%"   : pct((g["outcome"] == "TP").sum(), nd),
                "FN_rate_%"  : pct((g["outcome"] == "FN").sum(), nd),
                "FP_rate_%"  : pct((g["outcome"] == "FP").sum(), nn),
            })
        print(f"  {lbl}:")
        print("  " + pd.DataFrame(rows).to_string(index=False).replace("\n", "\n  "))

    # ── 6. FN vs TP financial ratio comparison ────────────────────────────────
    print(f"\n{'─'*72}")
    print("  6. FINANCIAL RATIOS — FN vs TP vs FP comparison")
    print(f"{'─'*72}")
    ratio_cols = [c for c in ["r6", "r7", "r8", "r9", "r12", "r14"] if c in df.columns]
    ratio_labels = {
        "r6": "Equity Ratio", "r7": "Debt Ratio", "r8": "Net Debt Ratio",
        "r9": "Self-Financing", "r12": "Cash/Current Assets", "r14": "ROA"
    }
    rows = []
    for r in ratio_cols:
        rows.append({
            "ratio"  : ratio_labels.get(r, r),
            "FN_med" : round(fn_df[r].median(), 4) if r in fn_df.columns else np.nan,
            "TP_med" : round(tp_df[r].median(), 4) if r in tp_df.columns else np.nan,
            "FP_med" : round(fp_df[r].median(), 4) if r in fp_df.columns else np.nan,
            "TN_med" : round(tn_df[r].median(), 4) if r in tn_df.columns else np.nan,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 7. Sector × Outcome cross-tab (FN share within sector defaults) ───────
    print(f"\n{'─'*72}")
    print("  7. SECTOR x OUTCOME — FN share of defaults per sector")
    print(f"{'─'*72}")
    def_df = df[df["y"] == 1]
    cross = def_df.groupby(["sector", "outcome"]).size().unstack(fill_value=0)
    cross["total"]   = cross.sum(axis=1)
    cross["FN_%"]    = (cross.get("FN", 0) / cross["total"] * 100).round(1)
    cross["recall_%"] = (cross.get("TP", 0) / cross["total"] * 100).round(1)
    print(cross.sort_values("FN_%", ascending=False).to_string())

    # ── 8. FP chronic repeaters ───────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  8. FP CHRONIC REPEATERS (firms appearing >1 obs as FP)")
    print(f"{'─'*72}")
    fp_counts = fp_df.groupby("id").size().sort_values(ascending=False)
    repeaters = fp_counts[fp_counts > 1]
    print(f"  FP firms with >1 obs flagged: {len(repeaters)} / {fp_df['id'].nunique()}")
    print(f"  Distribution of repeat counts:")
    print("  " + fp_counts.value_counts().sort_index().to_string().replace("\n", "\n  "))
    repeat_ids = repeaters.index
    fp_rep = fp_df[fp_df["id"].isin(repeat_ids)]
    print(f"  Repeat FP sector mix:")
    print("  " + fp_rep["sector"].value_counts().to_string().replace("\n", "\n  "))
    print(f"  Repeat FP size (Tiny): {fp_rep['is_tiny'].mean()*100:.1f}%")
    print(f"  Repeat FP med history_length: {fp_rep['history_length'].median():.1f}")
    print(f"  Repeat FP med score: {fp_rep['p_default'].median():.4f}")
