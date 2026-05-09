#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

timestamp="${1:-$(date +%y%m%d_%H%M%S)}"
cell_type="${2:-K562}"
data_type="${3:-procap}"
gpu="${4:-}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

mkdir -p "$script_dir/out"

for fold in $folds; do
  if command -v sbatch >/dev/null 2>&1; then
    sbatch --job-name "procapnet_${cell_type}_f${fold}" \
      --output "$script_dir/out/%x_%j.log" \
      --error "$script_dir/out/%x_%j.log" \
      "$script_dir/run_train_procapnet.sh" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu"
  else
    "$script_dir/run_train_procapnet.sh" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu"
  fi
done
