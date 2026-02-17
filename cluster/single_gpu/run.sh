#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition testdlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name Sample_Neps_Configs_Scale_Search_Space        # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/Sample_Configs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error logs/Sample_Configs/%x-%A.err    # STDERR

# Define the amount of memory required per node
#SBATCH --mem 2GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


source ~/anaconda3/bin/activate
conda activate scan_llm


start=`date +%s`

export PYTHONPATH="${PWD}:${PYTHONPATH}"


python neps/hpo_neps.py --config config/neps_configs/hpo_scale_search_space.yaml --root_dir neps/neps_configs/Neps_Scale_Search_Space --max_evaluations 100
# --out-dist plots/1B_Token_flops_distribution.png --out-mfu-distribution plots/1B_Token_mfu_distribution.png --out-flops-vs-mfu  plots/1B_Token_flops_vs_mfu.png


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
