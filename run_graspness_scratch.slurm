#!/bin/bash
#SBATCH --job-name=graspness
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR=/mnt/lustre/work/zell/zto428/graspness_modern
CONTAINER=/mnt/lustre/work/zell/zto428/containers/graspness_modern_latest-2025-11-06-2aa652688f6c.sif
DATA_DIR=/mnt/lustre/work/zell/zto428/datasets/graspnet

# Prefer node-local $SCRATCH, fall back to $TMPDIR if needed
SCRATCH_DIR="${SCRATCH:-${TMPDIR:-}}"

if [ -z "${SCRATCH_DIR}" ]; then
  echo "ERROR: Neither \$SCRATCH nor \$TMPDIR is set. Are we on a compute node?"
  exit 1
fi

DATA_SCRATCH_DIR="${SCRATCH_DIR}/datasets/graspnet"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using container: $CONTAINER"
echo "Project dir: $PROJECT_DIR"
echo "Lustre data dir: $DATA_DIR"
echo "Scratch base: $SCRATCH_DIR"
echo "Scratch data dir: $DATA_SCRATCH_DIR"

mkdir -p "${PROJECT_DIR}/logs/gsnet_dev"
mkdir -p "${DATA_SCRATCH_DIR}"

echo "Copying dataset to node-local scratch..."
rsync -a "${DATA_DIR}/" "${DATA_SCRATCH_DIR}/"
echo "Dataset copied."

# Optional: clean up scratch on exit
trap "rm -rf \"${SCRATCH_DIR}/datasets/graspnet\"" EXIT

singularity exec --nv \
  -B ${PROJECT_DIR}:/workspace \
  -B ${DATA_SCRATCH_DIR}:/datasets/graspnet \
  "$CONTAINER" \
  bash -lc "
    cd /workspace
    python train.py \
      --dataset_root /datasets/graspnet \
      --camera kinect \
      --model_name gsnet_dev \
      --log_dir logs/gsnet_dev \
      --max_epoch 15 \
      --batch_size 1 \
      --num_point 300000 \
      --num_workers 8 \
      --use_amp \
      --lazy_grasp_labels
  "

# Using A100 (40GB+) - 300k points should fit comfortably
# With batch_size=1 and AMP, expect ~15-20GB VRAM usage

