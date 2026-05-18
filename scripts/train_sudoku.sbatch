#!/bin/bash
#SBATCH --job-name=dtm-sud
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100GB
#SBATCH --time=48:00:00

# Train DTM on 4x4 Sudoku across 4 nodes x 2 GPUs (8 H100 total).
# Usage: sbatch scripts/train_sudoku.sbatch

set -eo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CONDA_ENV=${CONDA_ENV:-dtm}
MODEL_PATH=${MODEL_PATH:-${LLADA_MODEL_PATH:-/path/to/LLaDA-8B-Instruct}}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO/runs"}
# Write checkpoints + logs to netscratch (large, fast) instead of home (small NFS).
OUTPUT_DIR=${OUTPUT_DIR:-"$OUTPUT_ROOT/sudoku_${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_DIR/logs"

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HOME=${HF_HOME:-"$OUTPUT_ROOT/hf_cache"}
mkdir -p "$HF_HOME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_USE_IPV6=0
export GLOO_SOCKET_IFNAME=em4
export NCCL_SOCKET_IFNAME=ib0
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export NNODES=$SLURM_JOB_NUM_NODES

MASTER_FILE="$OUTPUT_DIR/master_addr.txt"
if [ "$SLURM_NODEID" -eq 0 ]; then
  MASTER_ADDR=$(ip -o -4 addr show dev em4 | awk '{print $4}' | cut -d/ -f1)
  echo "$MASTER_ADDR" > "$MASTER_FILE"
fi
while [ ! -s "$MASTER_FILE" ]; do sleep 1; done
export MASTER_ADDR=$(cat "$MASTER_FILE")
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"

cd "$REPO"

srun --ntasks-per-node=1 --gpus-per-task=2 \
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id="$SLURM_JOB_ID" \
    --log_dir="$OUTPUT_DIR/logs" \
    --redirects 3 \
    --tee 3 \
    -m dtm.train \
    --config-path "$REPO/configs" \
    --config-name sudoku.yaml \
    nodes=4 devices=2 world_size=8 \
    "base_model_path=$MODEL_PATH" \
    "checkpoint_dir=$OUTPUT_DIR/ckpts"
