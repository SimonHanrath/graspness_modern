#!/bin/bash

# Train PTv3 backbone with RGB features (6-channel input matching pretrained model)
# Key improvements:
# 1. transformer_pretrained: Automatically uses 6-channel input (XYZ + RGB)
# 2. weight_decay: Adds regularization (important for transformers)  
# 3. grad_clip: Prevents gradient explosion
# 4. Discriminative LRs: backbone gets 0.1x LR (pretrained), heads get full LR (random init)
#    - backbone_lr_scale=0.1 is default for transformer_pretrained
#    - Heads (graspable, rotation, crop, swad) need higher LR to learn from scratch

python train.py --dataset_root /datasets/graspnet --camera realsense \
  --model_name gsnet_ptv3_rgb --log_dir logs/gsnet_ptv3_rgb \
  --learning_rate 0.001 \
  --backbone_lr_scale 0.1 \
  --weight_decay 0.01 \
  --max_epoch 30 \
  --batch_size 1 \
  --backbone transformer_pretrained \
  --grad_clip 1.0 \
  --lazy_grasp_labels \
  --single_sample \
  #--num_workers 8
