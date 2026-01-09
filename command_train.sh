#python train.py --dataset_root /datasets/graspnet --camera realsense \
#  --model_name gsnet_dev --log_dir logs/gsnet_dev --max_epoch 1 --batch_size 1


python train.py --dataset_root /datasets/graspnet --camera realsense \
  --model_name gsnet_dev --log_dir logs/gsnet_dev --learning_rate 0.003 --max_epoch 1 --batch_size 1 --weight_decay 0.05 --backbone pointnet2