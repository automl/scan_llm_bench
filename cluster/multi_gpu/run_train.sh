#!/bin/bash

#SBATCH --job-name=4_GPU_Train-1M_20MT_With_FLOPS_Calculation
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=04:00:00
#SBATCH --partition=mlhiwidlc_gpu-rtx2080-advanced 

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB        

source ~/anaconda3/bin/activate 
conda activate scan_llm

config=/work/dlclarge2/sermaxha-scan_thesis/scan_llm_bench/config/chinchilla_scaling_configs/5M_100MT_scaled.yaml
job_idx=""
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  job_idx="${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n "$job_idx" ]]; then
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py --config=$config --job_idx=$job_idx --job_cluster=$SLURM_JOB_ID
else
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py --config=$config --job_cluster=$SLURM_JOB_ID
fi