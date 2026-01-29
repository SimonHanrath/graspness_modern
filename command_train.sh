#python train.py --dataset_root /datasets/graspnet --camera realsense \
#  --model_name gsnet_dev --log_dir logs/gsnet_dev --max_epoch 1 --batch_size 1


#python train.py --dataset_root /datasets/graspnet --camera realsense \
#  --model_name gsnet_dev --log_dir logs/gsnet_dev --learning_rate 0.001 \
#   --max_epoch 1 --batch_size 1 --backbone pointnet2 --resume \ --max_epoch 20 \
#   --checkpoint_path logs/cluster_100scenes_19epochs_pointnet2_input_fix/gsnet_pointnet2_epoch19.tar


#python train.py --dataset_root /datasets/graspnet --camera realsense \
#  --model_name gsnet_transformer --log_dir logs/gsnet_trafo_overfitting --learning_rate 0.001 \
#  --max_epoch 20 --batch_size 1 --backbone transformer_pretrained --single_sample \



python train.py --dataset_root /datasets/graspnet --camera realsense \
   --model_name gsnet_resunet_stable --log_dir logs/gsnet_stable_score --learning_rate 0.001 \
   --max_epoch 20 --batch_size 1 --backbone resunet --enable_stable_score --single_sample --use_amp