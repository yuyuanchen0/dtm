#!/bin/bash
#SBATCH --job-name=dtm-eval
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=80GB
#SBATCH --time=2:00:00

# Evaluate a DTM checkpoint on Countdown / Sudoku.
# Usage:
#   sbatch scripts/eval.sbatch <dataset> <checkpoint_path> [output_dir]
# e.g.
#   sbatch scripts/eval.sbatch countdown $RUN/ckpts/last.ckpt $RUN/eval_countdown

set -eo pipefail

DATASET=${1:-countdown}
CHECKPOINT_PATH=${2:?usage: sbatch scripts/eval.sbatch <dataset> <ckpt> [out_dir]}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
OUTPUT_DIR=${3:-${OUTPUT_DIR:-"$REPO/runs/eval_${SLURM_JOB_ID}"}}
MODEL_PATH=${MODEL_PATH:-${LLADA_MODEL_PATH:-/path/to/LLaDA-8B-Instruct}}
CONDA_ENV=${CONDA_ENV:-dtm}
GEN_LENGTH=${GEN_LENGTH:-256}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-128}
FEW_SHOT=${FEW_SHOT:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_NTASKS_PER_NODE:-2}}
[ "$DATASET" = "sudoku" ] && FEW_SHOT=${FEW_SHOT_SUDOKU:-3}

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$REPO/eval/eval.py" ]; then
  echo "ERROR: REPO=$REPO does not contain eval/eval.py" >&2
  echo "Submit from the repository root or pass --export=ALL,REPO=/path/to/DTM." >&2
  exit 2
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 2
fi

module load cuda/12.4.1-fasrc01
source /n/sw/Anaconda2-2019.10/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "REPO=$REPO"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MODEL_PATH=$MODEL_PATH"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "BATCH_SIZE=$BATCH_SIZE"

python -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" -m eval.eval \
  --dataset "$DATASET" \
  --model_path "$MODEL_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --gen_length "$GEN_LENGTH" \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --few_shot "$FEW_SHOT" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUTPUT_DIR"

cd "$OUTPUT_DIR"
python -m eval.parser_json
