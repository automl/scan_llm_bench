#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition metaadm_cpu-test    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name Generate_Train_Single_Script        # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error logs/%x-%A.err    # STDERR

# Define the amount of memory required per node
#SBATCH --mem 2GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


source ~/anaconda3/bin/activate
conda activate scan_llm


start=`date +%s`

export PYTHONPATH="${PWD}:${PYTHONPATH}"


python cluster/slurm_manager.py --partition testdlc_gpu-rtx2080 --job_name Generate_Train_Single_Script --mem_gb 8 --cpus_per_task 4 --time_hours 1 train_single --config_path neps/neps_test_configs/neps_configs_samples/configs/config_7/config.yaml --config_number 7 --neps_root_dir neps/neps_test_configs/neps_configs_samples


end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
