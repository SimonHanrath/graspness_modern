import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--backbone', type=str, default='transformer', choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet', 'resunet18', 'resunet_rgb', 'resunet18_rgb'],
                    help='Backbone architecture [default: transformer]. resunet=14D, resunet18=18D (more layers). Use _rgb suffix for 6-channel RGB input.')
parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                    help='Path to PTv3 pretrained weights (.pth file). If not specified, uses models/pointcept/model_best.pth')
parser.add_argument('--enable_flash', action='store_true', default=False,
                    help='Enable flash attention in PTv3 backbone (requires flash_attn package)')
parser.add_argument('--enable_stable_score', action='store_true', default=False,
                    help='Enable stable score prediction (model architecture) [default: False]')
parser.add_argument('--no_stable_reweight', action='store_true', default=False,
                    help='Disable stable score reweighting at inference (use raw grasp scores). Use with --enable_stable_score to compare with/without reweighting.')
parser.add_argument('--split', type=str, default='test_seen',
                    choices=['test', 'test_seen', 'test_seen_single', 'test_seen_mini', 'test_similar', 'test_similar_mini', 'test_novel', 'test_novel_single', 'test_novel_mini', 'test_train_mini'],
                    help='Dataset split to evaluate on [default: test_seen]')
parser.add_argument('--graspness_threshold', type=float, default=-0.1,
                    help='Threshold for graspness score filtering during forward pass [default: -0.1]')
parser.add_argument('--nsample', type=int, default=16,
                    help='Number of samples for cloud crop in GraspNet [default: 16]')
parser.add_argument('--friction', type=float, nargs='+', default=None,
                    help='Friction coefficient(s) for AP evaluation. '
                         'Default: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]. '
                         'Example: --friction 0.8 for single value, or --friction 0.2 0.4 0.8 for multiple.')
parser.add_argument('--include_floor', action='store_true', default=False,
                    help='Include floor/table points in inference (for models trained with --include_floor)')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    # Auto-enable RGB for backbones that require 6-channel input (XYZ + RGB)
    use_rgb = (cfgs.backbone in ['transformer_pretrained', 'resunet_rgb'])
    if use_rgb:
        print("Using RGB features for 6-channel input (XYZ + RGB)")
    
    test_dataset = GraspNetDataset(cfgs.dataset_root, split=cfgs.split, camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False,
                                   use_rgb=use_rgb, include_floor=cfgs.include_floor)
    print(f'Test dataset length: {len(test_dataset)} (split: {cfgs.split})')
    scene_list = test_dataset.scene_list()
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=spconv_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))
    # Init the model
    net = GraspNet(
        seed_feat_dim=cfgs.seed_feat_dim, 
        is_training=False,
        backbone=cfgs.backbone,
        ptv3_pretrained_path=cfgs.ptv3_pretrained_path,
        enable_flash=cfgs.enable_flash,
        enable_stable_score=cfgs.enable_stable_score,
        graspness_threshold=cfgs.graspness_threshold,
        nsample=cfgs.nsample,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Determine if we should use stable score reweighting at inference
    use_stable_reweight = cfgs.enable_stable_score and not cfgs.no_stable_reweight
    if cfgs.enable_stable_score:
        if use_stable_reweight:
            print("Stable score enabled: grasps will be reweighted by (1 - stable_score) during ranking")
        else:
            print("Stable score model loaded, but reweighting DISABLED (using raw grasp scores)")
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    
    # Handle old checkpoint format: bundled stable scores in conv_swad (108 outputs)
    # vs new format: separate conv_stable layer (conv_swad=96, conv_stable=12)
    if 'swad.conv_swad.weight' in state_dict:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108 and cfgs.enable_stable_score:
            print("Converting old checkpoint (bundled 108 outputs) to new format (96 + 12 separate)...")
            # Split: first 96 for scores/widths, last 12 for stable
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]
            print("  Converted swad.conv_swad: [108, 256, 1] -> [96, 256, 1]")
            print("  Created swad.conv_stable: [12, 256, 1]")
    
    net.load_state_dict(state_dict, strict=False)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    print("Note: Missing keys (buffers) will use default values from model initialization")

    batch_interval = 100
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(test_dataloader):
        # Transfer to GPU with non_blocking=True for async copy
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device, non_blocking=True)
            else:
                batch_data[key] = batch_data[key].to(device, non_blocking=True)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points, use_stable_score=use_stable_reweight)

        # Debug output for first batch
        if batch_idx == 0:
            print("\n" + "="*80)
            print("DEBUG: Network Output (end_points)")
            print("="*80)
            for key in sorted(end_points.keys()):
                val = end_points[key]
                if isinstance(val, torch.Tensor):
                    # Check if tensor is floating point for mean calculation
                    if val.dtype in [torch.float32, torch.float64, torch.float16]:
                        print(f"{key:40s}: shape={tuple(val.shape)}, dtype={val.dtype}, "
                              f"min={val.min().item():.6f}, max={val.max().item():.6f}, "
                              f"mean={val.mean().item():.6f}")
                    else:
                        # For integer tensors, skip mean
                        print(f"{key:40s}: shape={tuple(val.shape)}, dtype={val.dtype}, "
                              f"min={val.min().item()}, max={val.max().item()}")
                else:
                    print(f"{key:40s}: type={type(val)}")
            
            print("\n" + "="*80)
            print("DEBUG: Decoded Grasp Predictions (grasp_preds)")
            print("="*80)
            for i in range(min(cfgs.batch_size, len(grasp_preds))):
                preds = grasp_preds[i]
                print(f"\nBatch {i}:")
                print(f"  Shape: {tuple(preds.shape)}")
                print(f"  Dtype: {preds.dtype}")
                print(f"  Grasp scores (col 0): min={preds[:, 0].min().item():.6f}, "
                      f"max={preds[:, 0].max().item():.6f}, mean={preds[:, 0].mean().item():.6f}")
                print(f"  Grasp widths (col 1): min={preds[:, 1].min().item():.6f}, "
                      f"max={preds[:, 1].max().item():.6f}, mean={preds[:, 1].mean().item():.6f}")
                print(f"  Grasp depths (col 3): min={preds[:, 3].min().item():.6f}, "
                      f"max={preds[:, 3].max().item():.6f}, mean={preds[:, 3].mean().item():.6f}")
                print(f"  Top 10 grasp scores: {preds[:, 0].topk(10).values.cpu().numpy()}")
                print(f"  Num grasps with score > 0.5: {(preds[:, 0] > 0.5).sum().item()}")
                print(f"  Num grasps with score > 0.3: {(preds[:, 0] > 0.3).sum().item()}")
                print(f"  Num grasps with score > 0.0: {(preds[:, 0] > 0.0).sum().item()}")
            print("="*80 + "\n")

        # devbug end

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            
            # Transform grasp centers back to camera coordinates
            # During data loading, point cloud was shifted by offset to make all coords >= 0
            # Grasps are predicted in this shifted space, so we need to subtract the offset
            # to get back to camera coordinates
            if 'cloud_offset' in batch_data:
                offset = batch_data['cloud_offset'][i].cpu().numpy()  # [3,]
                # Grasp centers are in columns 13:16
                preds[:, 13:16] = preds[:, 13:16] - offset

            gg = GraspGroup(preds)
            # collision detection
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate(dump_dir):
    # Map split variants to their base eval split
    eval_split = cfgs.split
    if cfgs.split in ['test_seen_single', 'test_seen_mini']:
        eval_split = 'test_seen'
    elif cfgs.split in ['test_novel_single', 'test_novel_mini']:
        eval_split = 'test_novel'
    elif cfgs.split in ['test_similar_mini']:
        eval_split = 'test_similar'
    elif cfgs.split in ['test_train_mini']:
        eval_split = 'train'  # Use train split for proper object/grasp ground truth
    
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=eval_split)
    
    fric = cfgs.friction  # None means use default [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    
    # For single scene splits, evaluate only that scene directly
    if cfgs.split == 'test_seen_single':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[181], dump_folder=dump_dir, proc=1, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Seen (scene 181 only)={}'.format(cfgs.camera, ap))
    elif cfgs.split == 'test_seen_mini':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[101, 115, 128], dump_folder=dump_dir, proc=3, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Seen Mini (scenes 101,115,128)={}'.format(cfgs.camera, ap))
    elif cfgs.split == 'test_similar_mini':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[131, 145, 158], dump_folder=dump_dir, proc=3, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Similar Mini (scenes 131,145,158)={}'.format(cfgs.camera, ap))
    elif cfgs.split == 'test_novel_single':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[180], dump_folder=dump_dir, proc=1, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Novel (scene 180 only)={}'.format(cfgs.camera, ap))
    elif cfgs.split == 'test_novel_mini':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[161, 175, 188], dump_folder=dump_dir, proc=3, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Novel Mini (scenes 161,175,188)={}'.format(cfgs.camera, ap))
    elif cfgs.split == 'test_train_mini':
        res = np.array(ge.parallel_eval_scenes(scene_ids=[0, 50, 80], dump_folder=dump_dir, proc=3, list_coe_of_friction=fric))
        ap = np.mean(res)
        print('\nEvaluation Result:\n----------\n{}, AP Train Mini (scenes 0,50,80)={}'.format(cfgs.camera, ap))
    elif eval_split == 'test_seen':
        res, ap = ge.eval_seen(dump_folder=dump_dir, proc=6, list_coe_of_friction=fric)
    elif eval_split == 'test_similar':
        res, ap = ge.eval_similar(dump_folder=dump_dir, proc=6, list_coe_of_friction=fric)
    elif eval_split == 'test_novel':
        res, ap = ge.eval_novel(dump_folder=dump_dir, proc=6, list_coe_of_friction=fric)
    else:
        res, ap = ge.eval_all(dump_folder=dump_dir, proc=6, list_coe_of_friction=fric)
    
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)


if __name__ == '__main__':
    if cfgs.infer:
        inference()
    if cfgs.eval:
        evaluate(cfgs.dump_dir)