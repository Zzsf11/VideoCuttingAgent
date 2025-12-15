#!/bin/bash

#SBATCH --job-name=run_python
#SBATCH --output=/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/ttt/run_python_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

# 1. 加载环境 (根据你的集群环境修改)
echo "========================================================"
echo "Loading environment modules..."
# module load cuda/12.1 anaconda3/latest # 如果你的集群使用module
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VCA # *** 修改为你的Conda环境名 ***
echo "Environment loaded."
echo ""


# Check if a python script was provided
if [ -z "$1" ]; then
    echo "Error: No python script provided."
    echo "Usage: sbatch run_python.sbatch <script.py> [args...]"
    exit 1
fi

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Command: python $@"
echo "========================================================"

# Run the python script with all arguments passed to sbatch
python "$@"

echo "========================================================"
echo "End time: $(date)"
echo "========================================================"
