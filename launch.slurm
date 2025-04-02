#!/bin/bash

#SBATCH --job-name=mpt-benchmark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=1440:00:00
#SBATCH --output=log/%j-%x/slurm.log
#SBATCH --error=log/%j-%x/error.log

# this bash script is only for testing, do not modify

set -euo pipefail

export TZ='Asia/Singapore'

module load env

source .venv/bin/activate

export BASH_SCRIPT=${BASH_SCRIPT:-$(pwd)/launch.sh}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# must be set

export TMPDIR=${TMPDIR:-/tmp}
export TRITON_CACHE_DIR="/tmp/cache/triton"

gpus_per_node=$SLURM_GPUS_ON_NODE
world_size=$(($gpus_per_node * $SLURM_JOB_NUM_NODES))
master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_port=$((10000 + $RANDOM % 9000))

log_dir="$(pwd)/log/${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
mkdir -p $log_dir
export log_dir

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="${log_dir}/nccl/%h.log"

mkdir -p "${log_dir}/nccl"

srun_args=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "
launch_cmd=" \
    bash $BASH_SCRIPT \
    $gpus_per_node \
    $world_size \
    \$SLURM_PROCID \
    $master_addr \
    $master_port \
    "
echo $launch_cmd

srun $srun_args --jobid $SLURM_JOBID bash -c "${launch_cmd}" 2>&1 | tee -a "${log_dir}/main.log"