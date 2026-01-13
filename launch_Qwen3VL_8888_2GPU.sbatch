#!/bin/bash

#===============================================================
# SBATCH Part: Slurm 资源申请指令
#===============================================================
#SBATCH --job-name=Qwen3VL_8888      # 任务名，方便识别
#SBATCH --gres=gpu:2                # 申请2张GPU卡
#SBATCH --nodes=1                   # 申请1个节点
#SBATCH --ntasks-per-node=1         # 每个节点运行1个任务
#SBATCH --cpus-per-task=16          # 为任务申请16个CPU核心
#SBATCH --mem=128G                  # 申请128GB内存
#SBATCH --time=8:00:00              # 任务最长运行时间（4小时）
#SBATCH --output=output/Qwen3VL_vllm-server.out       # 将标准输出重定向到 slurm-JOBID.out 文件

#===============================================================
# Script Part: 在计算节点上执行的命令
#===============================================================

# 1. 加载环境 (根据你的集群环境修改)
echo "========================================================"
echo "Loading environment modules..."
# module load cuda/12.1 anaconda3/latest # 如果你的集群使用module
source ~/miniconda3/etc/profile.d/conda.sh
conda activate VCA # *** 修改为你的Conda环境名 ***
echo "Environment loaded."
echo ""

# 2. 获取网络信息
#    - 获取当前节点的主机名
NODE_HOSTNAME=$(hostname -f)
#    - 选择一个未被占用的端口 (8000-9000之间随机选择一个，减少冲突)
PORT=$(shuf -i 8000-9000 -n 1)

# 3. 打印关键的连接信息到输出文件，这是最重要的部分！
echo "========================================================"
echo "LLM Service Starting on Compute Node"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Compute Node Hostname: ${NODE_HOSTNAME}"
echo "LLM Listening Port: ${PORT}"
echo ""
echo "--- Instructions for connecting from the LOGIN NODE ---"
echo "1. Copy and execute this command in your terminal:"
echo ""
echo "   ssh -fN -L 8888:localhost:${PORT} ${USER}@${NODE_HOSTNAME}"
echo ""
echo "   (This will run in the background. Use -L 8889:... if port 8888 is taken)"
echo ""
echo "2. Once the tunnel is active, interact with the LLM at:"
echo "   API Base URL: http://localhost:8888/v1"
echo ""
echo "3. To kill the background SSH tunnel later:"
echo "   ps aux | grep 'ssh.*8888' | grep -v grep | awk '{print \$2}' | xargs kill"
echo "========================================================"
echo ""

# 4. 启动 vLLM OpenAI API 服务器
#    --host 0.0.0.0 确保服务监听在所有网络接口上，而不是仅监听localhost
#    --port ${PORT} 使用我们随机选择的端口
vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port ${PORT} \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --disable-mm-preprocessor-cache \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \

echo "vLLM server has been stopped."

# curl http://localhost:8888/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
#     "messages": [
#       {"role": "user", "content": "你好，你是谁？"}
#     ]
#   }'

