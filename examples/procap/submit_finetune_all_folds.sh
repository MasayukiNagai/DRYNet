#!/bin/bash
set -euo pipefail

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model_name> <timestamp> <mode> [cell_type] [data_type] [gpu] [validation_iter]" >&2
  echo "Example: $0 capy 260511 final_layer K562 procap 0 100" >&2
  echo "Set FOLDS='1 2 3' to override the default fold list." >&2
  exit 1
fi

model_name="$1"
timestamp="$2"
mode="$3"
cell_type="${4:-K562}"
data_type="${5:-procap}"
gpu="${6:-}"
validation_iter="${7:-}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

for fold in $folds; do
  sbatch --job-name "ft_${model_name}_${mode}_${cell_type}_f${fold}" \
    "$REPO_ROOT/examples/procap/run_finetune_count.sh" \
    "$model_name" "$timestamp" "$mode" "$cell_type" "$data_type" "$fold" "$gpu" "$validation_iter"
  sleep 1
done
