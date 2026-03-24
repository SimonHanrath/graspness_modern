#!/bin/bash
#SBATCH --job-name=gen_graspness_full
#SBATCH --partition=2080-galvani
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                    # Only need 1 GPU for KNN queries
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00                 # ~4 hours for 100 scenes, buffer for safety
#SBATCH --output=logs/gen_graspness_full-%j.out
#SBATCH --array=0-7                     # 8 jobs: 0-3 training, 4-7 test

# =============================================================================
# Generate graspness_full labels (without workspace masking)
# 
# This runs as an array job with 8 tasks:
#   Tasks 0-3: training scenes (0-99), 25 scenes each
#   Tasks 4-7: test scenes (100-189), ~22 scenes each
#
# Total time: ~30-60 min per task
#
# To run only training scenes (0-99), use: --array=0-3
# To run only test scenes (100-189), use: --array=4-7
# =============================================================================

set -euo pipefail

PROJECT_DIR=/mnt/lustre/work/zell/zto428/graspness_modern
CONTAINER=/mnt/lustre/work/zell/zto428/containers/graspness_modern_latest-2025-11-06-2aa652688f6c.sif
DATA_DIR=/mnt/lustre/work/zell/zto428/datasets/graspnet

# Camera type (change if needed)
CAMERA_TYPE="realsense"

# Define scene ranges for each array task
# Tasks 0-3: training scenes (0-99), 25 scenes each
# Tasks 4-7: test scenes (100-189), ~22 scenes each
case $SLURM_ARRAY_TASK_ID in
    0) SCENE_START=0;   SCENE_END=25  ;;
    1) SCENE_START=25;  SCENE_END=50  ;;
    2) SCENE_START=50;  SCENE_END=75  ;;
    3) SCENE_START=75;  SCENE_END=100 ;;
    4) SCENE_START=100; SCENE_END=122 ;;
    5) SCENE_START=122; SCENE_END=145 ;;
    6) SCENE_START=145; SCENE_END=167 ;;
    7) SCENE_START=167; SCENE_END=190 ;;
    *) echo "Invalid array task ID"; exit 1 ;;
esac

echo "=============================================="
echo "Generate graspness_full labels"
echo "=============================================="
echo "Host: $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Scene range: $SCENE_START to $SCENE_END"
echo "Camera: $CAMERA_TYPE"
echo "Container: $CONTAINER"
echo "Data dir: $DATA_DIR"
echo "=============================================="

mkdir -p "${PROJECT_DIR}/logs"

# Run the generation script
# Note: We write directly to Lustre since graspness_full is lightweight
# and we want it to persist. No need to copy dataset to scratch.
singularity exec --nv \
  -B ${PROJECT_DIR}:/workspace \
  -B ${DATA_DIR}:/datasets/graspnet \
  "$CONTAINER" \
    bash -lc "
    cd /workspace
    
    python dataset/generate_graspness_full.py \
      --dataset_root /datasets/graspnet \
      --camera_type ${CAMERA_TYPE} \
      --scene_start ${SCENE_START} \
      --scene_end ${SCENE_END} \
      --chunk 100000
    "

echo "Done! Task $SLURM_ARRAY_TASK_ID completed scenes $SCENE_START to $SCENE_END"
