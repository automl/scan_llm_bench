#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition mlhiwidlc_gpu-rtx2080-advanced    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name DemoTrain-400k_8MT             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error logs/%x-%A.err    # STDERR

# Define the amount of memory required per node
#SBATCH --mem 8GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/anaconda3/bin/activate # Adjust to your path of Miniconda installation
conda activate scan_llm

# Running the job
start=`date +%s`

export PYTHONPATH="${PWD}:${PYTHONPATH}"

config=/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/0_4M_8MT_scaled.yaml

job_idx=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job_idx="${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n "$job_idx" ]]; then
  python train.py --config="$config" --job_idx="$job_idx" --job_cluster="$SLURM_JOB_ID"
else
  python train.py --config="$config" --job_cluster="$SLURM_JOB_ID"
fi


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
