#!/bin/bash
#SBATCH --job-name=eval_procap
#SBATCH --output=out/%x_%j.log
#SBATCH --error=out/%x_%j.log
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=128G
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=2:00:00

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v job_notify_slurm >/dev/null 2>&1; then
  source "$(command -v job_notify_slurm)"
  notify_job_start || true
fi

proj_dir="${PROCAP_PROJ_DIR:-/grid/koo/home/shared/capybara/procap}"
model_name="${1:-procapnet}"
timestamp="${2:?timestamp is required}"
cell_type="${3:-K562}"
data_type="${4:-procap}"
fold="${5:-1}"
split="${6:-test}"
gpu="${7:-0}"
reverse_complement="${8:-1}"

REPO_ROOT="/grid/koo/home/nagai/projects/capybara"
script="${REPO_ROOT}/examples/procap/evaluate.py"
PYTHON="${REPO_ROOT}/.venv/bin/python"

cmd=("$PYTHON"
  "$script"
  --proj_dir "$proj_dir"
  --model_name "$model_name"
  --cell_type "$cell_type"
  --data_type "$data_type"
  --fold "$fold"
  --timestamp "$timestamp"
  --split "$split"
  --save_predictions)

case "$reverse_complement" in
  1)
    cmd+=(--reverse_complement)
    ;;
  0)
    ;;
  *)
    echo "Invalid reverse_complement value: $reverse_complement" >&2
    echo "Use 1 to enable RC or 0 to disable it." >&2
    exit 1
    ;;
esac

if [[ -n "$gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$gpu"
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
