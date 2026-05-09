#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

params="${1:-$repo_root/configs/default_procap.yaml}"
timestamp="${2:-$(date +%y%m%d_%H%M%S)}"
cell_type="${3:-K562}"
data_type="${4:-procap}"
gpu="${5:-}"
folds="${FOLDS:-1 2 3 4 5 6 7}"

mkdir -p "$script_dir/out"

for fold in $folds; do
  if command -v sbatch >/dev/null 2>&1; then
    sbatch --job-name "capy_${cell_type}_f${fold}" \
      --output "$script_dir/out/%x_%j.log" \
      --error "$script_dir/out/%x_%j.log" \
      "$script_dir/run_train_capy.sh" "$params" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu"
  else
    "$script_dir/run_train_capy.sh" "$params" "$timestamp" "$cell_type" "$data_type" "$fold" "$gpu"
  fi
done
