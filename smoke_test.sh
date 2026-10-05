#!/usr/bin/env bash
# Quick end-to-end check on 5 stores with tiny search budgets (~5-10 min on a laptop).
set -euo pipefail
cd "$(dirname "$0")"
export SMOKE=1 RUN_ID="${1:-smoke}" N_JOBS="${N_JOBS:-2}"
bash run_all.sh "$RUN_ID"
