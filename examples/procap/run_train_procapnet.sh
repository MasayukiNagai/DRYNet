#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

proj_dir="${PROCAP_PROJ_DIR:-/grid/koo/home/shared/capybara/procap}"
timestamp="${1:-}"
cell_type="${2:-K562}"
data_type="${3:-procap}"
fold="${4:-1}"
gpu="${5:-0}"

cmd=(python "$script_dir/train_procapnet.py"
  --proj_dir "$proj_dir"
  --cell_type "$cell_type"
  --data_type "$data_type"
  --fold "$fold")

if [[ -n "$timestamp" ]]; then
  cmd+=(--timestamp "$timestamp")
fi
if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
