python visualize_grasps.py \
    --dataset_root /datasets/graspnet \
    --checkpoint_path logs/gsnet_pointcept_pretrained_lower_backbone_lr/gsnet_pointcept_epoch27.tar \
    --backbone transformer_pretrained \
    --scene '0010' \
    --index '0250' \
    --num_grasps 30 \
    --num_point 30000 \
    --interactive



# Interactive Plotly visualization with resunet backbone
# python visualize_grasps.py --num_point 30000  \
#     --dataset_root /datasets/graspnet \
#     --checkpoint_path logs/cluster_100scenes_13epochs_realsense/gsnet_dev_epoch13.tar \
#     --backbone resunet \
#     --scene '0010' \
#     --index '0005' \
#     --interactive