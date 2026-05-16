#!/bin/bash
set -euo pipefail

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_name> <timestamp> [cell_type] [data_type] [split] [gpu] [reverse_complement]" >&2
  echo "Example: $0 procapnet 260509 K562 procap test 0 1" >&2
  echo "Set FOLDS='1 2 3' to override the default fold list." >&2
  exit 1
fi

model_name="$1"
timestamp="$2"
cell_type="${3:-K562}"
data_type="${4:-procap}"
split="${5:-test}"
gpu="${6:-}"
reverse_complement="${7:-1}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

for fold in $folds; do
  sbatch --job-name "eval_${model_name}_${cell_type}_f${fold}" \
    "$REPO_ROOT/examples/procap/run_evaluate.sh" "$model_name" "$timestamp" "$cell_type" "$data_type" "$fold" "$split" "$gpu" "$reverse_complement"
  # sleep 1
done
