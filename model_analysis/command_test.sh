#!/bin/bash
cd "$(dirname "$0")"

python test.py --dataset_root /datasets/graspnet --camera realsense \
    --checkpoint_path ../logs/cluster_100scenes_13epochs_realsense/gsnet_dev_epoch10.tar \
    --dump_dir ../dumps/realsense_dev --batch_size 1 --infer --eval --backbone resunet --num_point 15000 --split test_train_mini