#!/bin/bash

#SBATCH --partition metaadm_cpu-test
#SBATCH --job-name Generate_LM_Eval_Array_Script
#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err
#SBATCH --mem 2GB

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

source ~/anaconda3/bin/activate
conda activate scan_llm

start=`date +%s`

export PYTHONPATH="${PWD}:${PYTHONPATH}"

python cluster/slurm_manager.py \
  --partition testdlc2_gpu-l40s \
  --cluster test_eval \
  --job_name Generate_LM_Eval_Array_Script \
  --mem_gb 48 \
  --cpus_per_task 8 \
  --time_hours 1 \
  lm_eval_array \
  --hf_exports_dir test_hf_checkpoints \
  --results_root eval_results \
  --tasks hellaswag \
  --max_concurrent 2

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
