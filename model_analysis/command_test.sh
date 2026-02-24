#!/bin/bash
cd "$(dirname "$0")"

python test.py --dataset_root /datasets/graspnet --camera realsense \
    --checkpoint_path ../logs/gsnet_resunet_vanilla_stable_score01_n15k/gsnet_resunet_epoch10.tar \
    --dump_dir ../dumps/realsense_dev --batch_size 1 --infer --eval --backbone resunet --num_point 15000 --split test_seen_mini --enable_stable_score