#!/bin/bash
set -euo pipefail

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
params="${2:-$REPO_ROOT/configs/default_procap.yaml}"
cell_type="${3:-K562}"
data_type="${4:-procap}"
gpu="${5:-}"
stage="${6:-both}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

for fold in $folds; do
  sbatch --job-name "capy_${cell_type}_f${fold}" \
    "$REPO_ROOT/examples/procap/run_train_capy.sh" "$params" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu" "$stage"
  sleep 1  # Sleep for a short time to avoid overwhelming the scheduler
done
