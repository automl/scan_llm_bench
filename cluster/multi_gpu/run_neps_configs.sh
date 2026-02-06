#!/bin/bash

#SBATCH --partition mlhiwidlc_gpu-rtx2080-advanced

#SBATCH --job-name Train_Multi_Diff_Budgets_1M_20MT_From_NePS_Configs

# %a = array task ID
#SBATCH --output logs/from_neps_configs/SlimPajama_1M_20MT/%x-%A_%a.out
#SBATCH --error logs/from_neps_configs/SlimPajama_1M_20MT/%x-%A_%a.err

# Job array: default 1-50, max 2 concurrent jobs (%2)
#SBATCH --array=1-50%2

# Resources per job: 2 GPUs, memory for 2-GPU training
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

# Time limit per config
#SBATCH --time=24:00:00

echo "Workingdir: $PWD"
echo "Started at $(date)"

echo "Running job $SLURM_JOB_NAME array task $SLURM_ARRAY_TASK_ID using $SLURM_CPUS_PER_TASK cpus, JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

source ~/anaconda3/bin/activate
conda activate scan_llm

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Default paths (can be overridden via command line arguments)
BASE_CONFIG="${1:-/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/chinchilla_scaling_configs/1M_20MT_scaled.yaml}"
NEPS_RESULTS_DIR="${2:-/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/neps_configs/SlimPajama_1M_20M}"

CONFIGS_DIR="${NEPS_RESULTS_DIR}/configs"
config_number="${SLURM_ARRAY_TASK_ID}"
config_dir="${CONFIGS_DIR}/config_${config_number}"
config_file="${config_dir}/config.yaml"

echo "=============================================="
echo "Base config: $BASE_CONFIG"
echo "NePS results dir: $NEPS_RESULTS_DIR"
echo "Config: config_${config_number} -> $config_file"
echo "=============================================="

if [[ ! -f "$config_file" ]]; then
    echo "WARNING: No config.yaml at $config_file, skipping (config may not exist for this array index)"
    exit 0
fi

start=$(date +%s)

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --config="$BASE_CONFIG" \
    --config_overrides="$config_file" \
    --job_cluster="$SLURM_JOB_ID" \
    --config_number="$config_number"

exit_code=$?
end=$(date +%s)
runtime=$((end - start))

if [[ $exit_code -eq 0 ]]; then
    echo "config_${config_number} completed successfully in ${runtime}s"
else
    echo "ERROR: config_${config_number} failed with exit code $exit_code"
fi

echo "Job execution complete."
echo "Runtime: ${runtime}s"
exit $exit_code
