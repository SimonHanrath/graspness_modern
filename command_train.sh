# Original ResUNet training (CNN backbone)
# python train.py --dataset_root /datasets/graspnet --camera realsense \
#   --model_name gsnet_dev --log_dir logs/gsnet_dev --max_epoch 1 --batch_size 1

# PTv3 Transformer backbone training
python train.py --dataset_root /datasets/graspnet --camera kinect \
  --model_name gsnet_ptv3 --log_dir logs/gsnet_ptv3 \
  --max_epoch 13 --batch_size 1 \
  --learning_rate 0.005 \
  --use_adamw --weight_decay 0.05 \