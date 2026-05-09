#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

proj_dir="${PROCAP_PROJ_DIR:-/grid/koo/home/shared/capybara/procap}"
params="${1:-$repo_root/configs/default_procap.yaml}"
timestamp="${2:-}"
cell_type="${3:-K562}"
data_type="${4:-procap}"
fold="${5:-1}"
gpu="${6:-}"

cmd=(python "$script_dir/train_capy.py"
  --proj_dir "$proj_dir"
  --params "$params"
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
