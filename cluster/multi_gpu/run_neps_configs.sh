#!/bin/bash

# =============================================================================
# Run NePS sampled configurations sequentially on MULTIPLE GPUs.
# Each configuration is run one after another, using torchrun to utilize all GPUs.
#
# Usage:
#   sbatch run_neps_configs_multi_gpu.sh
#   
# Config range (env vars, set before sbatch):
#   START_CONFIG=15 END_CONFIG=50 sbatch run_neps_configs_multi_gpu.sh
# =============================================================================

# Define the partition
#SBATCH --partition mldlc2_gpu-l40s

# Define a name for your job
#SBATCH --job-name Neps_Search_Space_Fixed_HPs_Config_25_to_100

# Define the files to write the outputs of the job to.
#SBATCH --output logs/from_neps_configs/search_space_fixed_hps/%x-%A.out
#SBATCH --error logs/from_neps_configs/search_space_fixed_hps/%x-%A.err

# Request Multiple GPUs (4 in this case)
#SBATCH --gres=gpu:8

# CPUs per task (usually need more CPUs to feed multiple GPUs)
#SBATCH --cpus-per-task=64

# Define the amount of memory required per node
#SBATCH --mem 384GB  

# Time limit 
#SBATCH --time 24:00:00

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
source ~/anaconda3/bin/activate
conda activate scan_llm

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Default paths (can be overridden via command line arguments)
# BASE_CONFIG="${1:-/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/chinchilla_scaling_configs/50M_1BT_scaled.yaml}"
NEPS_RESULTS_DIR="${2:-/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/neps/neps_configs/Neps_Scale_Search_Space}"

# Config range: e.g. START_CONFIG=15 END_CONFIG=50 runs config_15 through config_50
START_CONFIG="${START_CONFIG:-25}"
END_CONFIG="${END_CONFIG:-100}"
CONFIGS_DIR="${NEPS_RESULTS_DIR}/configs"

# Set Number of GPUs for torchrun
GPUS_PER_NODE=8

echo "=============================================="
# echo "Base config: $BASE_CONFIG"
echo "NePS results dir: $NEPS_RESULTS_DIR"
echo "Configs dir: $CONFIGS_DIR"
echo "Config range: ${START_CONFIG} to ${END_CONFIG}"
echo "GPUs per run: $GPUS_PER_NODE"
echo "=============================================="

# Count configs in range
total_configs=0
for config_dir in $(find "$CONFIGS_DIR" -maxdepth 1 -type d -name "config_*" | sort -V); do
    config_num=$(basename "$config_dir" | sed 's/config_//')
    if [[ "$config_num" =~ ^[0-9]+$ ]] && [[ "$config_num" -ge "$START_CONFIG" ]] && [[ "$config_num" -le "$END_CONFIG" ]]; then
        ((total_configs++))
    fi
done
echo "Found $total_configs configurations to run (config_${START_CONFIG} to config_${END_CONFIG})"
echo ""

# Track progress
completed=0
failed=0

start_total=$(date +%s)

for config_dir in $(find "$CONFIGS_DIR" -maxdepth 1 -type d -name "config_*" | sort -V); do
    config_num=$(basename "$config_dir" | sed 's/config_//')
    
    # Validation checks
    if [[ ! "$config_num" =~ ^[0-9]+$ ]]; then continue; fi
    if [[ "$config_num" -lt "$START_CONFIG" ]] || [[ "$config_num" -gt "$END_CONFIG" ]]; then continue; fi
    
    config_file="${config_dir}/config.yaml"
    if [[ ! -f "$config_file" ]]; then
        echo "WARNING: No config.yaml found in $config_dir, skipping..."
        continue
    fi
    
    echo "=============================================="
    echo "Running config_${config_num} ($(($completed + failed + 1))/$total_configs)"
    echo "Config overrides: $config_file"
    echo "Started at $(date)"
    echo "=============================================="
    
    start_config=$(date +%s)
    
    SAVE_NEPS_REPORT="${SAVE_NEPS_REPORT:-1}"
    TRAIN_ARGS=(--config="$config_file" --job_cluster="$SLURM_JOB_ID" --config_number="$config_num" --neps_root_dir="$NEPS_RESULTS_DIR")
    [[ "$SAVE_NEPS_REPORT" == "1" || "$SAVE_NEPS_REPORT" == "true" ]] && TRAIN_ARGS+=(--save_neps_report)
    
    # --- CHANGED: Using torchrun for multi-gpu execution ---
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$GPUS_PER_NODE \
        train.py \
        "${TRAIN_ARGS[@]}"
    
    exit_code=$?
    end_config=$(date +%s)
    runtime_config=$((end_config - start_config))
    
    if [[ $exit_code -eq 0 ]]; then
        echo "config_${config_num} completed successfully in ${runtime_config}s"
        ((completed++))
    else
        echo "ERROR: config_${config_num} failed with exit code $exit_code"
        ((failed++))
    fi
    
    echo ""
done

end_total=$(date +%s)
runtime_total=$((end_total - start_total))

echo "=============================================="
echo "All configurations processed!"
echo "Completed: $completed / $total_configs"
echo "Failed: $failed"
echo "Total runtime: ${runtime_total}s ($(($runtime_total / 60)) minutes)"
echo "Finished at $(date)"
echo "=============================================="
