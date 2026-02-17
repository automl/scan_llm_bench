#!/bin/bash

# Read environment variables from config.conf
SCRIPT_DIR="./vscode_remote_debugging"
while read var value
do
    export "$var"="$value"
done < $SCRIPT_DIR/config.conf

cd $WORKDIR

# Activate conda environment
source ~/anaconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate scan_llm

echo Waiting for debugger to attach...

job_idx=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job_idx="${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n "$job_idx" ]]; then
  python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $WORKDIR/scripts/backfill_neps_reports.py --dry-run
else
  python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $WORKDIR/scripts/backfill_neps_reports.py --dry-run
fi

conda deactivate