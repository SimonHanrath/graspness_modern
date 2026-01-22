import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

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
parser.add_argument('--backbone', type=str, default='transformer', choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet'],
                    help='Backbone architecture [default: transformer]. Use transformer_pretrained for PTv3 with Pointcept pretrained weights.')
parser.add_argument('--ptv3_pretrained_path', type=str, default=None,
                    help='Path to PTv3 pretrained weights (.pth file). If not specified, uses models/pointcept/model_best.pth')
parser.add_argument('--enable_flash', action='store_true', default=False,
                    help='Enable flash attention in PTv3 backbone (requires flash_attn package)')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    # Auto-enable RGB for transformer_pretrained backbone (requires 6-channel input)
    use_rgb = (cfgs.backbone == 'transformer_pretrained')
    if use_rgb:
        print("Using RGB features for 6-channel input (XYZ + RGB) - required for transformer_pretrained")
    
    test_dataset = GraspNetDataset(cfgs.dataset_root, split='test_seen', camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False,
                                   use_rgb=use_rgb)
    print('Test dataset length: ', len(test_dataset))
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
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    print("Note: Missing keys (buffers) will use default values from model initialization")

    batch_interval = 100
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

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
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test_seen')
    res, ap = ge.eval_seen(dump_folder=dump_dir, proc=6)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)


if __name__ == '__main__':
    if cfgs.infer:
        inference()
    if cfgs.eval:
        evaluate(cfgs.dump_dir)