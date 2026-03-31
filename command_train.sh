#!/bin/bash
# =============================================================================
# GSNet Training Script - Quick Reference Guide
# =============================================================================
#
# AVAILABLE BACKBONES:
#   - resunet         : Default ResUNet-14D backbone (MinkowskiEngine sparse convs)
#   - resunet18       : Deeper ResUNet-18D
#   - resunet_rgb     : ResUNet with 6-channel RGB input
#   - pointnet2       : PointNet++ backbone (raw point processing)
#   - transformer     : PTv3 transformer (train from scratch)
#   - transformer_pretrained : PTv3 with pretrained weights
#   - sonata          : Self-supervised PTv3 backbone (CVPR 2025)
#
# KEY LEARNING RATE OPTIONS:
#   --learning_rate   : Base LR for heads (default: 0.001)
#   --backbone_lr_scale : Multiplier for backbone LR (default: 0.1 for pretrained, 1.0 otherwise)
#   --layer_decay     : Layer-wise LR decay for pretrained backbones (default: 0.65 for sonata/ptv3)
#                       Each encoder stage gets: lr * layer_decay^(num_stages - stage)
#                       Set to 1.0 to disable LLRD (uniform LR across all layers)
#   --cosine_lr       : Use cosine annealing with warmup instead of exponential decay
#   --warmup_epochs   : Number of warmup epochs for cosine LR (default: 2)
#
# DATA OPTIONS:
#   --camera          : realsense | kinect
#   --train_split     : train | train_all | train_reduced (95 scenes when using --use_val)
#   --use_val         : Enable validation (reserves 5 scenes for validation)
#   --num_point       : Number of points to sample (default: 15000)
#   --graspness_threshold : Threshold for graspness filtering (default: 0.1)
#   --include_floor   : Include floor/table points (requires generate_graspness_full.py)
#
# TRAINING OPTIMIZATION:
#   --batch_size      : Batch size per GPU (default: 4, use 1 for transformers)
#   --accumulation_steps : Gradient accumulation steps (simulate larger batch)
#   --grad_clip       : Gradient clipping max norm (recommended: 1.0 for transformers)
#   --weight_decay    : Weight decay for AdamW (recommended: 0.02-0.05 for transformers)
#   --use_amp         : Enable mixed-precision training
#
# CHECKPOINT & RESUME:
#   --checkpoint_path : Path to checkpoint for resuming or fine-tuning
#   --resume          : Resume training (continue from last epoch, load optimizer state)
#   --finetune        : Fine-tune mode (load weights but reset epoch to 0)
#
# AUGMENTATION:
#   --no_translation_aug : Disable random translation (paper uses no translation)
#
# =============================================================================

# -----------------------------------------------------------------------------
# CURRENT COMMAND (modify this for your experiment)
# -----------------------------------------------------------------------------
python train.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --model_name gsnet_dev \
    --log_dir logs/gsnet_dev \
    --learning_rate 0.001 \
    --max_epoch 20 \
    --batch_size 4 \
    --backbone resunet \
    --num_point 15000 \
    --train_split train_reduced \
    --graspness_threshold 0.01 \
    --no_translation_aug \
    --use_val

# =============================================================================
# EXAMPLE CONFIGURATIONS (from backbone experiments)
# =============================================================================

# -----------------------------------------------------------------------------
# Example 1: Sonata backbone with Layer-wise LR Decay (LLRD)
# Lower LR for early (pretrained) layers, higher LR for later layers & heads
# -----------------------------------------------------------------------------
# python train.py \
#     --dataset_root /datasets/graspnet \
#     --camera realsense \
#     --model_name gsnet_sonata \
#     --log_dir logs/backbone_experiments/gsnet_sonata_t01_n15_llrd065_val \
#     --backbone sonata \
#     --num_point 15000 \
#     --graspness_threshold 0.1 \
#     --max_epoch 20 \
#     --batch_size 4 \
#     --learning_rate 0.001 \
#     --layer_decay 0.65 \
#     --grad_clip 1.0 \
#     --cosine_lr \
#     --warmup_epochs 2 \
#     --no_translation_aug \
#     --use_val \
#     --lazy_grasp_labels \
#     --num_workers 8

# -----------------------------------------------------------------------------
# Example 2: Sonata backbone with Uniform LR (no LLRD)
# -----------------------------------------------------------------------------
# python train.py \
#     --dataset_root /datasets/graspnet \
#     --camera realsense \
#     --model_name gsnet_sonata \
#     --log_dir logs/backbone_experiments/gsnet_sonata_t01_n15_uniform_lr_val \
#     --backbone sonata \
#     --num_point 15000 \
#     --graspness_threshold 0.1 \
#     --max_epoch 20 \
#     --batch_size 4 \
#     --learning_rate 0.001 \
#     --layer_decay 1.0 \
#     --grad_clip 1.0 \
#     --cosine_lr \
#     --warmup_epochs 2 \
#     --no_translation_aug \
#     --use_val \
#     --lazy_grasp_labels \
#     --num_workers 8

# -----------------------------------------------------------------------------
# Example 3: PTv3 Pretrained backbone with LLRD
# Similar to Sonata but uses different pretrained weights
# -----------------------------------------------------------------------------
# python train.py \
#     --dataset_root /datasets/graspnet \
#     --camera realsense \
#     --model_name gsnet_ptv3 \
#     --log_dir logs/backbone_experiments/gsnet_ptv3_t01_n15_llrd065_val \
#     --backbone transformer_pretrained \
#     --num_point 15000 \
#     --graspness_threshold 0.1 \
#     --max_epoch 20 \
#     --batch_size 4 \
#     --learning_rate 0.001 \
#     --layer_decay 0.65 \
#     --grad_clip 1.0 \
#     --cosine_lr \
#     --warmup_epochs 2 \
#     --no_translation_aug \
#     --use_val \
#     --lazy_grasp_labels \
#     --num_workers 8

# -----------------------------------------------------------------------------
# Example 4: PointNet++ baseline (no pretrained weights)
# Traditional point cloud backbone, uses exponential LR decay by default
# -----------------------------------------------------------------------------
# python train.py \
#     --dataset_root /datasets/graspnet \
#     --camera realsense \
#     --model_name gsnet_pointnet2_rawpoints \
#     --log_dir logs/backbone_experiments/gsnet_pointnet2_rawpoints_realsense_t01_n15 \
#     --backbone pointnet2 \
#     --num_point 15000 \
#     --graspness_threshold 0.1 \
#     --max_epoch 15 \
#     --batch_size 4 \
#     --learning_rate 0.001 \
#     --no_translation_aug \
#     --use_val \
#     --lazy_grasp_labels \
#     --num_workers 8

# =============================================================================
# NOTES ON LEARNING RATE STRATEGIES
# =============================================================================
#
# 1. LAYER-WISE LR DECAY (LLRD) - Recommended for pretrained transformers
#    --layer_decay 0.65 means:
#    - embedding layer: lr * 0.65^5 = lr * 0.116  (lowest LR, most pretrained)
#    - stage0:          lr * 0.65^4 = lr * 0.178
#    - stage1:          lr * 0.65^3 = lr * 0.275
#    - stage2:          lr * 0.65^2 = lr * 0.423
#    - stage3:          lr * 0.65^1 = lr * 0.650
#    - stage4:          lr * 0.65^0 = lr * 1.000  (highest LR, closest to task)
#    - heads:           lr * 1.0 (full learning rate, random init)
#
# 2. UNIFORM LR (--layer_decay 1.0)
#    All layers get the same learning rate.
#    Simpler, but may destroy pretrained features in early layers.
#
# 3. BACKBONE LR SCALE (--backbone_lr_scale 0.1)
#    Applies a global multiplier to all backbone parameters.
#    Use when NOT using LLRD (layer_decay=1.0).
#    With LLRD active, backbone_lr_scale defaults to 1.0 (LLRD handles scaling).
#
# 4. LR SCHEDULE:
#    - Default: Exponential decay (0.95^epoch)
#    - Cosine: --cosine_lr with --warmup_epochs 2
#              Starts at 0, warms up to base LR, then cosine decays to 0
#
# =============================================================================
# MEMORY OPTIMIZATION TIPS
# =============================================================================
#
# For large models (transformers), use:
#   --batch_size 1 --accumulation_steps 4   # Simulate batch_size=4
#   --lazy_grasp_labels                      # Load labels on-demand
#   --num_workers 8                          # Parallel data loading
#   --use_amp                                # Mixed precision


