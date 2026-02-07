#!/bin/bash

# Train PTv3 backbone FROM SCRATCH (no pretrained weights)
# 
# ⚠️  WARNING: This is EXPERIMENTAL! Training transformers from scratch 
#     on ~25K samples is challenging. ResUNet achieves 79.5% AP reliably.
#     Only use this for research/comparison purposes.
#
# Architecture: Small PointTransformerV3 (~5M params)
# - Intentionally small to reduce overfitting on limited data
# - 6 encoder + 4 decoder blocks
# - 192-dim bottleneck
#
# Training strategy for from-scratch transformers:
# - Lower LR (0.0005) - transformers are sensitive to LR
# - Strong weight_decay (0.1) - heavy regularization needed
# - Higher drop_path (0.2) already in model config
# - Longer training with patience

python train.py --dataset_root /datasets/graspnet --camera realsense \
  --model_name gsnet_ptv3_scratch --log_dir logs/gsnet_ptv3_scratch \
  --learning_rate 0.0005 \
  --backbone_lr_scale 1.0 \
  --weight_decay 0.1 \
  --max_epoch 30 \
  --batch_size 2 \
  --backbone transformer \
  --grad_clip 1.0 \
  --lazy_grasp_labels \
  --use_amp
  # Optional flags:
  # --enable_stable_score \   # Enable stable score training
  # --single_sample \         # For debugging (1 sample per scene)
  # --resume \                # Resume from checkpoint
