# python test.py --dataset_root /datasets/graspnet --camera realsense --checkpoint_path logs/gsnet_pointcept_pretrained_lower_backbone_lr/gsnet_pointcept_epoch27.tar --dump_dir dumps/realsense_backbone_e21 --batch_size 1 --infer --eval --backbone transformer_pretrained 
python test.py --dataset_root /datasets/graspnet --camera realsense \
    --checkpoint_path logs/gsnet_resunet_vanilla_cosine_anealing__n300k/gsnet_resunet_epoch13.tar \
    --dump_dir dumps/realsense_dev --batch_size 1 --infer --eval --backbone resunet --num_point 300000 --split test_seen_single