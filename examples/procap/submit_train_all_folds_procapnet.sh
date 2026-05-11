#!/bin/bash
set -euo pipefail

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
cell_type="${2:-K562}"
data_type="${3:-procap}"
gpu="${4:-}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

for fold in $folds; do
  sbatch --job-name "procapnet_${cell_type}_f${fold}" \
    "$REPO_ROOT/examples/procap/run_train_procapnet.sh" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu"
  sleep 1  # Sleep for a short time to avoid overwhelming the scheduler
done
