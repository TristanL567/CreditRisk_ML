#!/bin/bash
# Phase 2: Groups 03, 04, 05 (ratios + time dynamics + VAE latent features)
# Sequence per split: data prep → autoencoder → AutoGluon 03/04/05 → XGBoost 03/04/05

RSCRIPT="C:/Program Files/R/R-4.5.2/bin/Rscript.exe"
PYTHON="C:/venvs/autotab_env/Scripts/python.exe"
PIPE="C:/Users/Tristan Leiter/Documents/ILAB_OeNB/CreditRisk_ML/01_Code/1.Final Pipeline"
AE_SCRIPT="$PIPE/03_Autoencoder.py"
AG_SCRIPT="$PIPE/05_AutoGluon.py"
DATA_RUNNER="C:/Users/Tristan Leiter/Documents/ILAB_OeNB/CreditRisk_ML/run_data_only.R"
XGB_RUNNER="C:/Users/Tristan Leiter/Documents/ILAB_OeNB/CreditRisk_ML/run_xgb_single.R"
LOG="C:/Users/Tristan Leiter/Documents/ILAB_OeNB/CreditRisk_ML/03_Output/Final"

export PYTHONIOENCODING=utf-8
cd "C:/Users/Tristan Leiter/Documents/ILAB_OeNB/CreditRisk_ML"

# ── OoS side ──────────────────────────────────────────────────────────────────
echo "=== [OoS] Data prep (r + TD) ===" && \
"$RSCRIPT" "$DATA_RUNNER" 03 r TRUE OoS > "$LOG/run_data_03_OoS.log" 2>&1 && echo "data r_TD_OoS OK" && \

echo "=== [OoS] Autoencoder ===" && \
"$PYTHON" "$AE_SCRIPT" OoS > "$LOG/run_ae_OoS.log" 2>&1 && echo "AE OoS OK" && \

echo "=== [OoS] AutoGluon 03a ===" && \
"$PYTHON" "$AG_SCRIPT" 03 OoS > "$LOG/run_ag_03a.log" 2>&1 && echo "AG 03a OK" && \
echo "=== [OoS] AutoGluon 04a ===" && \
"$PYTHON" "$AG_SCRIPT" 04 OoS > "$LOG/run_ag_04a.log" 2>&1 && echo "AG 04a OK" && \
echo "=== [OoS] AutoGluon 05a ===" && \
"$PYTHON" "$AG_SCRIPT" 05 OoS > "$LOG/run_ag_05a.log" 2>&1 && echo "AG 05a OK" && \

echo "=== [OoS] XGBoost 03a ===" && \
"$RSCRIPT" "$XGB_RUNNER" 03 r TRUE OoS > "$LOG/run_03a.log" 2>&1 && echo "XGB 03a OK" && \
echo "=== [OoS] XGBoost 04a ===" && \
"$RSCRIPT" "$XGB_RUNNER" 04 r TRUE OoS > "$LOG/run_04a.log" 2>&1 && echo "XGB 04a OK" && \
echo "=== [OoS] XGBoost 05a ===" && \
"$RSCRIPT" "$XGB_RUNNER" 05 r TRUE OoS > "$LOG/run_05a.log" 2>&1 && echo "XGB 05a OK" && \

# ── OoT side ──────────────────────────────────────────────────────────────────
echo "=== [OoT] Data prep (r + TD) ===" && \
"$RSCRIPT" "$DATA_RUNNER" 03 r TRUE OoT > "$LOG/run_data_03_OoT.log" 2>&1 && echo "data r_TD_OoT OK" && \

echo "=== [OoT] Autoencoder ===" && \
"$PYTHON" "$AE_SCRIPT" OoT > "$LOG/run_ae_OoT.log" 2>&1 && echo "AE OoT OK" && \

echo "=== [OoT] AutoGluon 03b ===" && \
"$PYTHON" "$AG_SCRIPT" 03 OoT > "$LOG/run_ag_03b.log" 2>&1 && echo "AG 03b OK" && \
echo "=== [OoT] AutoGluon 04b ===" && \
"$PYTHON" "$AG_SCRIPT" 04 OoT > "$LOG/run_ag_04b.log" 2>&1 && echo "AG 04b OK" && \
echo "=== [OoT] AutoGluon 05b ===" && \
"$PYTHON" "$AG_SCRIPT" 05 OoT > "$LOG/run_ag_05b.log" 2>&1 && echo "AG 05b OK" && \

echo "=== [OoT] XGBoost 03b ===" && \
"$RSCRIPT" "$XGB_RUNNER" 03 r TRUE OoT > "$LOG/run_03b.log" 2>&1 && echo "XGB 03b OK" && \
echo "=== [OoT] XGBoost 04b ===" && \
"$RSCRIPT" "$XGB_RUNNER" 04 r TRUE OoT > "$LOG/run_04b.log" 2>&1 && echo "XGB 04b OK" && \
echo "=== [OoT] XGBoost 05b ===" && \
"$RSCRIPT" "$XGB_RUNNER" 05 r TRUE OoT > "$LOG/run_05b.log" 2>&1 && echo "XGB 05b OK" && \

echo "=== PHASE 2 COMPLETE ==="
