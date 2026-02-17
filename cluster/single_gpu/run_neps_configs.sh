#!/bin/bash

# =============================================================================
# Run NePS sampled configurations sequentially.
# Each configuration is run one after another on a single GPU.
#
# Usage:
#   sbatch run_neps_configs.sh
#   
# Or with custom paths:
#   sbatch run_neps_configs.sh /path/to/base_config.yaml /path/to/neps_results
#
# Config range (env vars, set before sbatch):
#   START_CONFIG=15 END_CONFIG=50 sbatch run_neps_configs.sh   # run config_15 through config_50
#
# Save NePS report (env var, default 1):
#   SAVE_NEPS_REPORT=1 sbatch run_neps_configs.sh   # save report.yaml to each config dir
# =============================================================================

# Define the partition on which the job shall run.
#SBATCH --partition mldlc2_gpu-l40s    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name New_Neps_Full_Commands             # short: -J <job name>

# Define the files to write the outputs of the job to.
#SBATCH --output logs/from_neps_configs/Neps_Test/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error logs/from_neps_configs/Neps_Test/%x-%A.err    # STDERR

# Define the amount of memory required per node
#SBATCH --mem 48GB  

# Time limit (adjust based on expected total runtime)
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
NEPS_RESULTS_DIR="${2:-/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/neps/neps_configs/Neps_Test}"

# Config range: e.g. START_CONFIG=15 END_CONFIG=50 runs config_15 through config_50
START_CONFIG="${START_CONFIG:-1}"
END_CONFIG="${END_CONFIG:-50}"
CONFIGS_DIR="${NEPS_RESULTS_DIR}/configs"

echo "=============================================="
echo "NePS results dir: $NEPS_RESULTS_DIR"
echo "Configs dir: $CONFIGS_DIR"
echo "Config range: ${START_CONFIG} to ${END_CONFIG}"
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
    if [[ ! "$config_num" =~ ^[0-9]+$ ]]; then
        continue
    fi
    if [[ "$config_num" -lt "$START_CONFIG" ]] || [[ "$config_num" -gt "$END_CONFIG" ]]; then
        continue
    fi
    
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
    python train.py "${TRAIN_ARGS[@]}"
    
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
