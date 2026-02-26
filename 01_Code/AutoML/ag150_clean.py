import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
import numpy as np

TRAIN_PATH = 'pathtrain'
TEST_PATH = 'pathtest'
TARGET = 'y'
FOLD_COL = 'fold_id' #assumes train is already split and has this column (same for test, not necessary tho)

# Load data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

drop_cols = ['id', 'refdate']
train_data = train_data.drop(columns=drop_cols)
test_data = test_data.drop(columns=drop_cols + [FOLD_COL])
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

import autogluon.tabular as ag
import copy
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
# Step 1: Get FULL default hyperparameters (preserves ALL models/configs)
hyperparameters = get_hyperparameter_config('zeroshot_2025_12_18_cpu') #['default', 'light', 'very_light', 'toy', 'multimodal', 'interpretable', 'zeroshot', 'zeroshot_2023', 'zeroshot_2025_tabfm', 'zeroshot_2025_12_18_gpu', 'zeroshot_2025_12_18_cpu', 'experimental_2024', 'experimental']
print("NN_TORCH configs before:", len(hyperparameters['NN_TORCH']))  
#modifications below ensure CUDA GPU is used, if no such thing, do not do it
# Step 2: Patch EVERY NN_TORCH config → add GPU (non-destructive!)
nn_torch_configs = hyperparameters['NN_TORCH']  # Your printed list
for config in nn_torch_configs:
    # ag_args_fit merges w/ existing (if any)
    if 'ag_args_fit' not in config:
        config['ag_args_fit'] = {}
    config['ag_args_fit']['num_gpus'] = 1  # 🚀 Force GPU!
nn_torch_configs = hyperparameters['FASTAI']  # Your printed list
for config in nn_torch_configs:
    # ag_args_fit merges w/ existing (if any)
    if 'ag_args_fit' not in config:
        config['ag_args_fit'] = {}
    config['ag_args_fit']['num_gpus'] = 1  # 🚀 Force GPU!
print("Sample patched config:", nn_torch_configs[0])  # Now has 'ag_args_fit': {'num_gpus': 1}

#main training and leaderbords + feature improtance
def train():
    predictor = TabularPredictor(
        problem_type= 'binary',
        label=TARGET,
        eval_metric='roc_auc',
        groups=FOLD_COL,
    )
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    predictor.fit(
        presets='best_v150',
        train_data=train_data,
        time_limit=1*30*60,
        hyperparameters=hyperparameters,
        #hyperparameter_tune_kwargs = 'auto', can do hpo for some models, look into documentation
        #excluded_model_types=['CAT', 'GBM', 'GBM_PREP', 'NN_TORCH', 'TABDPT', 'FT_TRANSFORMER'],
        num_gpus=1,
        #save_space = True, #saves disk space, set to true not to save train data! but when True cannot get oof predictions from train
    )
    print("\n--- Per-model ROC-AUC on test ---")
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(leaderboard[['model', 'score_test', 'score_val']].to_string(index=False))
    leaderboard.to_csv('leaderboard.csv')
    print("\n--- Importance on Train ---")
    train_imp = predictor.feature_importance(train_data, subsample_size=5*1000, time_limit=1*60)
    train_imp.to_csv('train_imp.csv')
    print(train_imp)
    print("\n--- Importance on Test ---")
    test_imp = predictor.feature_importance(test_data, subsample_size=5*1000, time_limit=1*60)
    test_imp.to_csv('test_imp.csv')
    print(test_imp)

if __name__ == '__main__':
    train()
#===========================
#possible all models, some do not work properly
#['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'REALMLP', 'NN_TORCH', 'LR', 'FASTAI', 
# 'GBM_PREP', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'AG_AUTOMM', 'FT_TRANSFORMER', 'TABDPT', 
# 'TABICL', 'TABM', 'TABPFNMIX', 'REALTABPFN-V2', 'REALTABPFN-V2.5', 'MITRA', 'FASTTEXT', 
# 'ENS_WEIGHTED', 'SIMPLE_ENS_WEIGHTED', 'IM_RULEFIT', 'IM_GREEDYTREE', 'IM_FIGS', 'IM_HSTREE', 
# 'IM_BOOSTEDRULES', 'DUMMY', 'EBM']

#Valid presets: ['best_quality', 'best_quality_v082', 'best_quality_v150', 
# 'experimental_quality_v120', 'extreme_quality', 'extreme_quality_v140', 
# 'good_quality', 'good_quality_v082', 'high_quality', 'high_quality_v082', 
# 'high_quality_v150', 'ignore_text', 'ignore_text_ngrams', 'interpretable', 
# 'medium_quality', 'optimize_for_deployment', 'tabarena']
