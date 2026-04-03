import io
import json
import re
import subprocess
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr


ROOT_DIR = Path(r"C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML")
DATA_DIR = ROOT_DIR / "02_Data"
LATENT_DIR = ROOT_DIR / "03_Output" / "Latent"
FINAL_DIR = ROOT_DIR / "03_Output" / "Final"
CHARTS_DIR = ROOT_DIR / "03_Output" / "Charts"

SPLITS = ("OoS", "OoT", "OoT2")
TITLE_COLOR = "#004890"
BG_COLOR = "#FFFFFF"
MODEL_SAMPLE_N = 5000

R_SCRIPT_CANDIDATES = [
    Path(r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"),
    Path(r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe"),
    Path(r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"),
    Path(r"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe"),
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def load_r_object(path: Path):
    if not path.exists():
        return None
    result = pyreadr.read_r(str(path))
    if not result:
        return None
    return next(iter(result.values()))


def get_rscript_path() -> Path | None:
    for candidate in R_SCRIPT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def group_to_feature_spec(group: str):
    feature_map = {
        "01": ("f", "noTD"),
        "02": ("r", "noTD"),
        "03": ("r", "TD"),
        "04": ("r", "TD"),
        "05": ("r", "TD"),
    }
    return feature_map.get(group)


def load_test_features(group: str, split_name: str) -> pd.DataFrame | None:
    spec = group_to_feature_spec(group)
    if spec is None:
        return None

    feature_family, td_flag = spec
    feature_path = DATA_DIR / f"02_test_final_{feature_family}_{td_flag}_{split_name}.rds"
    id_path = DATA_DIR / f"02_test_id_vec_{feature_family}_{td_flag}_{split_name}.rds"

    feature_df = load_r_object(feature_path)
    id_df = load_r_object(id_path)
    if feature_df is None or id_df is None:
        return None

    feature_df = feature_df.copy()
    feature_df.insert(0, "id", id_df.iloc[:, 0].to_numpy())

    if group not in {"04", "05"}:
        return feature_df

    latent_path = LATENT_DIR / f"latent_test_{feature_family}_{td_flag}_{split_name}.parquet"
    if not latent_path.exists():
        return feature_df

    latent_df = pd.read_parquet(latent_path).copy()
    latent_cols = [col for col in latent_df.columns if col not in {"id", "y"}]

    feature_df = feature_df.reset_index(drop=True)
    latent_df = latent_df.reset_index(drop=True)

    if group == "04":
        return pd.concat([feature_df, latent_df[latent_cols]], axis=1)

    category_pattern = re.compile(r"^(sector_|size_|groupmember$|public$)")
    category_cols = [col for col in feature_df.columns if category_pattern.match(col)]
    return pd.concat(
        [
            latent_df[["id", "y"] + latent_cols],
            feature_df[category_cols],
        ],
        axis=1,
    )


def load_prediction_frame(model_dir: Path, split_name: str) -> pd.DataFrame | None:
    model_name = model_dir.name

    if model_name.endswith("_AutoGluon"):
        prediction_path = model_dir / "predictions_test.parquet"
        if not prediction_path.exists():
            return None
        prediction_df = pd.read_parquet(prediction_path).copy()
    elif model_name.endswith("_XGBoost_Manual"):
        prediction_path = model_dir / "predictions_test.rds"
        prediction_df = load_r_object(prediction_path)
        if prediction_df is None:
            return None
        prediction_df = prediction_df.copy()
    elif model_name.endswith("_GLM"):
        prediction_path = model_dir / f"predictions_test_GLM_v2_{split_name}.parquet"
        if not prediction_path.exists():
            return None
        prediction_df = pd.read_parquet(prediction_path).copy()
    else:
        return None

    pred_col = "p_default" if "p_default" in prediction_df.columns else "pred"
    if "id" not in prediction_df.columns or pred_col not in prediction_df.columns:
        return None

    return prediction_df[["id", pred_col]].rename(columns={pred_col: "pred"})


def align_feature_and_prediction_rows(
    feature_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    left = feature_df[["id"] + feature_cols].copy().reset_index(drop=True)
    right = pred_df[["id", "pred"]].copy().reset_index(drop=True)

    left["_row_in_id"] = left.groupby("id").cumcount()
    right["_row_in_id"] = right.groupby("id").cumcount()

    aligned = left.merge(
        right,
        on=["id", "_row_in_id"],
        how="inner",
        validate="1:1",
    )
    return aligned.drop(columns="_row_in_id")


def load_ag_feature_rank(model_dir: Path) -> list[str]:
    feature_path = model_dir / "feature_importance.csv"
    if not feature_path.exists():
        return []
    feature_df = pd.read_csv(feature_path)
    if "feature" not in feature_df.columns:
        return []
    return (
        feature_df.sort_values("importance", ascending=False)["feature"]
        .dropna()
        .astype(str)
        .tolist()
    )


def load_glm_feature_rank(model_dir: Path, split_name: str) -> list[str]:
    feature_path = model_dir / f"GLM_Variable_Importance_v2_{split_name}.xlsx"
    prediction_path = model_dir / f"predictions_test_GLM_v2_{split_name}.parquet"
    if not feature_path.exists():
        return []

    feature_df = pd.read_excel(feature_path)
    if feature_df.empty or "Feature" not in feature_df.columns:
        return []

    strategy_label = None
    if prediction_path.exists():
        pred_df = pd.read_parquet(prediction_path, columns=["model"])
        if not pred_df.empty and "model" in pred_df.columns:
            model_label = str(pred_df["model"].dropna().iloc[0]) if pred_df["model"].dropna().any() else ""
            if "Base" in model_label and "Strategy" in feature_df.columns:
                strategy_label = "Base Model"

    if strategy_label is not None:
        filtered = feature_df.loc[feature_df["Strategy"] == strategy_label].copy()
        if not filtered.empty:
            feature_df = filtered

    sort_col = "AbsCoef" if "AbsCoef" in feature_df.columns else "Overall"
    return (
        feature_df.sort_values(sort_col, ascending=False)["Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )


def load_xgb_feature_rank(model_dir: Path) -> list[str]:
    xgb_path = model_dir / "xgb_model.rds"
    rscript = get_rscript_path()
    if rscript is None or not xgb_path.exists():
        return []

    r_expr = (
        "obj <- readRDS(commandArgs(trailingOnly = TRUE)[1]); "
        "if ('importance' %in% names(obj)) { "
        "write.csv(obj$importance, stdout(), row.names = FALSE) "
        "} else { "
        "write.csv(data.frame(), stdout(), row.names = FALSE) "
        "}"
    )

    result = subprocess.run(
        [str(rscript), "-e", r_expr, str(xgb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    feature_df = pd.read_csv(io.StringIO(result.stdout))
    if feature_df.empty or "Feature" not in feature_df.columns:
        return []

    sort_col = "Gain" if "Gain" in feature_df.columns else feature_df.columns[1]
    return (
        feature_df.sort_values(sort_col, ascending=False)["Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )


def load_feature_rank(model_dir: Path, split_name: str) -> list[str]:
    model_name = model_dir.name
    if model_name.endswith("_AutoGluon"):
        return load_ag_feature_rank(model_dir)
    if model_name.endswith("_GLM"):
        return load_glm_feature_rank(model_dir, split_name)
    if model_name.endswith("_XGBoost_Manual"):
        return load_xgb_feature_rank(model_dir)
    return []


def collect_model_records() -> pd.DataFrame:
    records = []
    expected_suffix = {"OoS": "a", "OoT": "b", "OoT2": "c"}

    for model_dir in sorted(FINAL_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        group = model_name[:2]

        if model_name.endswith("_AutoGluon"):
            summary_path = model_dir / "eval_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            auc = meta.get("metrics", {}).get("auc_roc")
            split_name = meta.get("split_mode")
            algorithm = "AutoGluon"
        elif model_name.endswith("_XGBoost_Manual"):
            summary_path = model_dir / "eval_summary.rds"
            summary_df = load_r_object(summary_path)
            if summary_df is None or summary_df.empty:
                continue
            test_row = summary_df.loc[summary_df["set"] == "test"]
            if test_row.empty:
                continue
            test_row = test_row.iloc[0]
            auc = test_row.get("auc_roc")
            split_name = test_row.get("split_mode")
            algorithm = "XGBoost_Manual"
        elif model_name.endswith("_GLM"):
            split_name = None
            auc = None
            algorithm = "GLM"
            for candidate_split in SPLITS:
                leaderboard_path = model_dir / f"GLM_Leaderboard_v2_{candidate_split}.xlsx"
                if leaderboard_path.exists():
                    leaderboard_df = pd.read_excel(leaderboard_path)
                    if leaderboard_df.empty:
                        continue
                    auc = leaderboard_df.iloc[0].get("AUC")
                    split_name = candidate_split
                    break
            if split_name is None:
                continue
        else:
            continue

        if split_name not in SPLITS or pd.isna(auc):
            continue

        if len(model_name) < 3 or model_name[2] != expected_suffix[split_name]:
            continue

        records.append(
            {
                "model_name": model_name,
                "group": group,
                "split_name": split_name,
                "algorithm": algorithm,
                "auc": float(auc),
            }
        )

    model_df = pd.DataFrame(records)
    if model_df.empty:
        return model_df

    return model_df.sort_values(["split_name", "auc"], ascending=[True, False]).reset_index(drop=True)


def build_bivariate_charts_for_split(split_name: str, model_df: pd.DataFrame):
    split_dir = CHARTS_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    top_models = (
        model_df.loc[model_df["split_name"] == split_name]
        .sort_values("auc", ascending=False)
        .head(3)
        .to_dict("records")
    )

    insights = []
    print(f"\nProcessing split: {split_name}")
    print(f"Top models: {[row['model_name'] for row in top_models]}")

    for model_row in top_models:
        model_name = model_row["model_name"]
        group = model_row["group"]
        model_dir = FINAL_DIR / model_name

        feature_rank = load_feature_rank(model_dir, split_name)
        feature_rank = list(dict.fromkeys(feature_rank))

        feature_df = load_test_features(group, split_name)
        pred_df = load_prediction_frame(model_dir, split_name)
        if feature_df is None or pred_df is None:
            print(f"  Skipping {model_name}: missing test features or predictions.")
            continue

        available_features = [feat for feat in feature_rank if feat in feature_df.columns]
        top_features = available_features[:5]
        if len(top_features) < 2:
            print(f"  Skipping {model_name}: fewer than two plottable top features.")
            continue

        plot_df = align_feature_and_prediction_rows(feature_df, pred_df, top_features)
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred"]).copy()
        if plot_df.empty:
            print(f"  Skipping {model_name}: merged plot frame is empty.")
            continue

        sample_n = min(MODEL_SAMPLE_N, len(plot_df))
        plot_df = plot_df.sample(n=sample_n, random_state=42).copy()

        print(f"  {model_name}: using top features {top_features}")
        for f1, f2 in combinations(top_features[:5], 2):
            pair_df = plot_df[[f1, f2, "pred"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if len(pair_df) < 10:
                continue

            s1 = pair_df[f1].std(ddof=0)
            s2 = pair_df[f2].std(ddof=0)
            if pd.isna(s1) or pd.isna(s2) or s1 == 0 or s2 == 0:
                continue

            pair_df[f"{f1}_z"] = (pair_df[f1] - pair_df[f1].mean()) / s1
            pair_df[f"{f2}_z"] = (pair_df[f2] - pair_df[f2].mean()) / s2

            pair_df = pair_df.loc[
                pair_df[f"{f1}_z"].between(-2.0, 2.0, inclusive="neither")
                & pair_df[f"{f2}_z"].between(-2.0, 2.0, inclusive="neither")
            ].copy()

            if len(pair_df) < 10:
                continue
            if len(np.unique(pair_df[f"{f1}_z"])) < 3 or len(np.unique(pair_df[f"{f2}_z"])) < 3:
                continue

            try:
                fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=BG_COLOR)
                fig.patch.set_facecolor(BG_COLOR)
                ax.set_facecolor(BG_COLOR)

                contour = ax.tricontourf(
                    pair_df[f"{f1}_z"].to_numpy(),
                    pair_df[f"{f2}_z"].to_numpy(),
                    pair_df["pred"].to_numpy(),
                    levels=20,
                    cmap="Blues_r",
                )

                colorbar = fig.colorbar(contour, ax=ax)
                colorbar.set_label("P(Default)")

                ax.set_xlim(-2.0, 2.0)
                ax.set_ylim(-2.0, 2.0)
                ax.set_xlabel(f"{f1} (z-score)")
                ax.set_ylabel(f"{f2} (z-score)")
                ax.set_title(
                    f"2D PDP: {f1} x {f2}\n({model_name}, {split_name})",
                    color=TITLE_COLOR,
                    fontweight="bold",
                    fontsize=16,
                    loc="center",
                    pad=12,
                )

                max_idx = pair_df["pred"].idxmax()
                max_row = pair_df.loc[max_idx]
                insights.append(
                    {
                        "Model": model_name,
                        "Feature 1": f1,
                        "Feature 2": f2,
                        "Max_Prob": float(max_row["pred"]),
                        "Max_Prob_F1_Zscore": float(max_row[f"{f1}_z"]),
                        "Max_Prob_F2_Zscore": float(max_row[f"{f2}_z"]),
                    }
                )

                out_file = split_dir / f"8_{safe_name(model_name)}_{safe_name(f1)}_x_{safe_name(f2)}.png"
                fig.tight_layout()
                fig.savefig(
                    out_file,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=BG_COLOR,
                )
                plt.close(fig)
            except Exception as exc:
                plt.close("all")
                print(f"    Failed on {model_name} / {f1} x {f2}: {exc}")

    insights_df = pd.DataFrame(
        insights,
        columns=[
            "Model",
            "Feature 1",
            "Feature 2",
            "Max_Prob",
            "Max_Prob_F1_Zscore",
            "Max_Prob_F2_Zscore",
        ],
    )
    insights_path = split_dir / f"{split_name}_Bivariate_Insights.csv"
    insights_df.to_csv(insights_path, index=False)
    print(f"Saved insights: {insights_path}")
    print(f"Saved {len(list(split_dir.glob('8_*.png')))} bivariate charts for {split_name}.")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    model_df = collect_model_records()
    if model_df.empty:
        raise RuntimeError("No completed model records were found under 03_Output/Final.")

    print("Top models by split:")
    print(model_df.groupby("split_name").head(3).to_string(index=False))

    for split_name in SPLITS:
        build_bivariate_charts_for_split(split_name, model_df)

    print("\nFinished generating CHART 2 bivariate outputs.")


if __name__ == "__main__":
    main()
