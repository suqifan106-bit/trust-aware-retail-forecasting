#!/usr/bin/env bash
# Runs the full supplementary-experiment pipeline. Usage: bash run_all.sh [RUN_ID]
set -euo pipefail
cd "$(dirname "$0")"
export RUN_ID="${1:-${RUN_ID:-main}}"
export N_JOBS="${N_JOBS:-24}"
# keep BLAS / OpenMP threads inside the CPU budget (<= 80 % of 32 threads)
export OMP_NUM_THREADS=$N_JOBS MKL_NUM_THREADS=$N_JOBS OPENBLAS_NUM_THREADS=$N_JOBS
export PYTHONUNBUFFERED=1
mkdir -p "results/$RUN_ID/logs"
for s in e00_build_data e01_baselines e02_benchmark_tuning e03_rolling_origin e04_stats_bootstrap \
         e05_segment_errors e06_shap_global e07_shap_local e08_fidelity e09_stability \
         e10_plausibility e11_decision_review e12_ablation e13_thesis_replication; do
  echo "=== $(date '+%F %T') START $s" | tee -a "results/$RUN_ID/logs/run_all.log"
  python "$s.py" 2>&1 | tee -a "results/$RUN_ID/logs/run_all.log"
  echo "=== $(date '+%F %T') END $s" | tee -a "results/$RUN_ID/logs/run_all.log"
done
echo "ALL DONE -> results/$RUN_ID"
