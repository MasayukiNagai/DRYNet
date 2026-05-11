#!/bin/bash
#SBATCH --job-name=train_procap
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00

if command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

proj_dir="${PROCAP_PROJ_DIR:-/grid/koo/home/shared/capybara/procap}"
timestamp="${1:-}"
cell_type="${2:-K562}"
data_type="${3:-procap}"
fold="${4:-1}"
gpu="${5:-0}"

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"
script="${REPO_ROOT}/examples/procap/train_procapnet.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"

cmd=("$PYTHON"
  "$script"
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
