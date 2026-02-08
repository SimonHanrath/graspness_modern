# python test.py --dataset_root /datasets/graspnet --camera realsense --checkpoint_path logs/gsnet_pointcept_pretrained_lower_backbone_lr/gsnet_pointcept_epoch27.tar --dump_dir dumps/realsense_backbone_e21 --batch_size 1 --infer --eval --backbone transformer_pretrained 
python test.py --dataset_root /datasets/graspnet --camera realsense \
    --checkpoint_path logs/gsnet_resunet_strong_stable_score_augmented_n20k/gsnet_resunet_epoch13.tar \
    --dump_dir dumps/realsense_dev --batch_size 1 --infer --eval --backbone resunet --enable_stable_score --num_point 20000