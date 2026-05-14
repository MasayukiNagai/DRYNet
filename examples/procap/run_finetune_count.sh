#!/bin/bash
#SBATCH --job-name=ft_count
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --time=24:00:00

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

proj_dir="${PROCAP_PROJ_DIR:-/grid/koo/home/shared/capybara/procap}"
model_name="${1:?model_name is required: capy or procapnet}"
timestamp="${2:?timestamp is required}"
mode="${3:?mode is required: count_head or final_layer}"
cell_type="${4:-K562}"
data_type="${5:-procap}"
fold="${6:-1}"
gpu="${7:-0}"
validation_iter="${8:-300}"
LEARNING_RATE="5e-7"

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"
script="${REPO_ROOT}/examples/procap/finetune_count.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"

cmd=("$PYTHON"
  "$script"
  --proj_dir "$proj_dir"
  --model_name "$model_name"
  --timestamp "$timestamp"
  --mode "$mode"
  --cell_type "$cell_type"
  --data_type "$data_type"
  --fold "$fold"
  --learning_rate "$LEARNING_RATE")

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi
if [[ -n "$validation_iter" ]]; then
  cmd+=(--validation_iter "$validation_iter")
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
