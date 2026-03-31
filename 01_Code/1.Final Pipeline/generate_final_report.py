import json
import os
import pandas as pd
import pyreadr
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score
from autogluon.tabular import TabularPredictor

root_dir = Path(r'C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML')
data_dir = root_dir / '02_Data'
lat_dir  = root_dir / '03_Output' / 'Latent'
out_dir  = root_dir / '03_Output' / 'Final'
dest_dir = Path(r'C:\Users\Tristan Leiter\Documents\ILAB_OeNB\CreditRisk_ML\05_Documentation\For final presentation\ModelPipeline\Results\All')
dest_dir.mkdir(parents=True, exist_ok=True)

def load_rds(path):
    if not Path(path).exists(): return None
    try:
        return pyreadr.read_r(str(path))[None]
    except: return None

def load_parquet(path):
    if not Path(path).exists(): return None
    try:
        return pd.read_parquet(path)
    except: return None

def get_train_data(group, split):
    feat_map = {'01': 'f', '02': 'r', '03': 'r', '04': 'r', '05': 'r'}
    td_map   = {'01': False, '02': False, '03': True, '04': True, '05': True}
    if group not in feat_map: return None
    kf = feat_map[group]
    td = 'TD' if td_map[group] else 'noTD'
    suffix = f'_{kf}_{td}_{split}'
    raw_path = data_dir / f'02_train_final{suffix}.rds'
    raw_df = load_rds(raw_path)
    if raw_df is None: return None
    if group in ('01', '02', '03'): return raw_df
    lat_path = lat_dir / f'latent_train{suffix}.parquet'
    lat_df = load_parquet(lat_path)
    if lat_df is None: return raw_df
    lat_feats = [c for c in lat_df.columns if c not in ('id', 'y')]
    raw_df = raw_df.reset_index(drop=True)
    lat_df = lat_df.reset_index(drop=True)
    if group == '04': return pd.concat([raw_df, lat_df[lat_feats]], axis=1)
    if group == '05':
        cat_re = re.compile(r'^(sector_|size_|groupmember$|public$)')
        cat_cols = [c for c in raw_df.columns if cat_re.match(c)]
        return pd.concat([lat_df[lat_feats + ['y']], raw_df[cat_cols]], axis=1)
    return raw_df

metrics_list = []
importance_list = []

# --- 1. AutoGluon ---
print("Processing AutoGluon...")
for m_dir in out_dir.glob('*_AutoGluon'):
    try:
        json_path = m_dir / 'eval_summary.json'
        if not json_path.exists(): continue
        with open(json_path, 'r') as f:
            meta = json.load(f)
        m_name = meta['model']
        group, split = m_name[:2], meta['split_mode']
        test_m = meta.get('metrics', {})
        train_auc = 'N/A'
        try:
            predictor = TabularPredictor.load(str(m_dir / 'ag_predictor'))
            train_df = get_train_data(group, split)
            if train_df is not None:
                # Subsample for speed - very small for diagnostic
                train_sub = train_df.head(1000) if len(train_df) > 1000 else train_df
                y_prob = predictor.predict_proba(train_sub, as_multiclass=False)
                train_auc = round(roc_auc_score(train_sub['y'], y_prob), 4)
                
                # Use subsample for importance too
                fi = predictor.feature_importance(train_sub)
                if fi is not None:
                    top = fi.nlargest(10, 'importance')
                    for f, val in top.iterrows():
                        importance_list.append({'Model': m_name, 'Feature': f, 'Importance': val['importance']})
        except Exception as e: print(f"  Error in AutoGluon diag for {m_name}: {e}")

        metrics_list.append({
            'Model Group': group, 'Split': split, 'Algorithm': 'AutoGluon',
            'Train AUC': train_auc, 'CV AUC': 'N/A', 'Test AUC': test_m.get('auc_roc', 'N/A'),
            'Brier Score': test_m.get('brier', 'N/A'), 'Params': f"Preset: {meta.get('preset')}"
        })
    except Exception as e: print(f"  Error processing AutoGluon {m_dir.name}: {e}")

# --- 2. XGBoost ---
print("Processing XGBoost...")
for m_dir in out_dir.glob('*_XGBoost_Manual'):
    try:
        res = load_rds(m_dir / 'xgb_model.rds')
        if res is None: continue
        eval_t = res['eval_table']
        test_m = eval_t[eval_t['set'] == 'test'].iloc[0]
        train_m = eval_t[eval_t['set'] == 'train_insample'].iloc[0]
        m_name = m_dir.name
        metrics_list.append({
            'Model Group': m_name[:2], 'Split': 'OoS' if 'a_XGB' in m_name else 'OoT', 'Algorithm': 'XGBoost',
            'Train AUC': train_m.get('auc_roc', 'N/A'), 'CV AUC': round(res.get('cv_score_mean', 0), 4),
            'Test AUC': test_m.get('auc_roc', 'N/A'), 'Brier Score': test_m.get('brier', 'N/A'),
            'Params': str(res.get('params', {}))
        })
        imp = res['importance']
        top = imp.nlargest(10, 'Gain')
        for _, row in top.iterrows():
            importance_list.append({'Model': m_name, 'Feature': row['Feature'], 'Importance': row['Gain']})
    except Exception as e: print(f"  Error processing XGBoost {m_dir.name}: {e}")

# --- 3. GLM ---
print("Processing GLM...")
for m_dir in out_dir.glob('*_GLM'):
    try:
        m_name = m_dir.name
        split = 'OoS' if 'a_GLM' in m_name else 'OoT'
        lb_path = m_dir / f'GLM_Leaderboard_v2_{split}.xlsx'
        if not lb_path.exists(): continue
        lb = pd.read_excel(lb_path).iloc[0]
        metrics_list.append({
            'Model Group': m_name[:2], 'Split': split, 'Algorithm': 'GLM',
            'Train AUC': 'N/A', 'CV AUC': 'N/A', 'Test AUC': round(lb.get('AUC', 0), 4),
            'Brier Score': round(lb.get('Brier_Score', 0), 4),
            'Params': f"Alpha: {lb.get('Alpha')}, Lambda: {lb.get('Lambda')}"
        })
        imp_path = m_dir / f'GLM_Variable_Importance_v2_{split}.xlsx'
        if imp_path.exists():
            imp_df = pd.read_excel(imp_path)
            top = imp_df.nlargest(10, 'Overall')
            for _, row in top.iterrows():
                importance_list.append({'Model': m_name, 'Feature': row['Feature'], 'Importance': row['Overall']})
    except Exception as e: print(f"  Error processing GLM {m_dir.name}: {e}")

# --- Final Export ---
print("Exporting to Excel...")
output_path = dest_dir / 'Model_Results_Summary.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    pd.DataFrame(metrics_list).to_excel(writer, sheet_name='Model Metrics', index=False)
    pd.DataFrame(importance_list).to_excel(writer, sheet_name='Feature Importance', index=False)

print(f"Summary report written to: {output_path}")
