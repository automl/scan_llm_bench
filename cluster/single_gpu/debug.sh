#!/bin/bash

# Read environment variables from config.conf
SCRIPT_DIR="./vscode_remote_debugging"
while read var value
do
    export "$var"="$value"
done < $SCRIPT_DIR/config.conf

cd $WORKDIR

# Activate conda environment
source $CONDA_SOURCE
conda activate $CONDA_ENV

echo Waiting for debugger to attach...
onfig=/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/chinchilla_scaling_configs/0_4M_8MT_scaled.yaml

job_idx=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job_idx="${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n "$job_idx" ]]; then
  python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $WORKDIR/train.py  --config="$config" --job_idx="$job_idx" --job_cluster="$SLURM_JOB_ID"
else
  python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $WORKDIR/train.py  --config="$config" --job_cluster="$SLURM_JOB_ID"
fi

conda deactivate