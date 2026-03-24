#!/bin/bash
cd "$(dirname "$0")"

python model_analysis/test.py --dataset_root /datasets/graspnet --camera realsense \
    --checkpoint_path ../logs/gsnet_resunet18_vanilla_realsense_t001_n15/gsnet_resunet_epoch11.tar \
    --dump_dir ../dumps/realsense_dev --batch_size 1 --infer --eval --backbone resunet18 --num_point 15000 --split test_seen_mini  --friction 0.8 --graspness_threshold 0.01