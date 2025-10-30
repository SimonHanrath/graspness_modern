import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', default=None,
                    help='Model checkpoint path (optional). If omitted, runs with random weights.')
parser.add_argument('--dump_dir', default=None, required=True,
                    help='Directory to save outputs (.npy grasp groups).')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point-wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection (set -1 to skip) [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False, help='Run inference / dump predictions')
parser.add_argument('--eval', action='store_true', default=False, help='Evaluate dumped predictions')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='If >0, stop after this many batches (handy for smoke tests).')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.makedirs(cfgs.dump_dir, exist_ok=True)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    # load_label=False -> no need for graspness / labels at inference-time feature extraction
    test_dataset = GraspNetDataset(
        cfgs.dataset_root,
        split='test_seen',
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        voxel_size=cfgs.voxel_size,
        remove_outlier=True,
        augment=False,
        load_label=False
    )
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=my_worker_init_fn,
        collate_fn=minkowski_collate_fn
    )
    print('Test dataloader length: ', len(test_dataloader))

    # Init the model
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint if provided
    if cfgs.checkpoint_path:
        checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"-> loaded checkpoint {cfgs.checkpoint_path} (epoch: {start_epoch})")
        else:
            # allow loading a raw state_dict too
            net.load_state_dict(checkpoint)
            print(f"-> loaded state_dict checkpoint {cfgs.checkpoint_path}")
    else:
        print("-> no checkpoint provided: running with random weights (smoke test).")

    # Inference loop
    batch_interval = 100
    net.eval()
    tic = time.time()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            # Move to device
            for key in batch_data:
                if 'list' in key:
                    for i in range(len(batch_data[key])):
                        for j in range(len(batch_data[key][i])):
                            batch_data[key][i][j] = batch_data[key][i][j].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            # Guard: skip batches that collapse to zero voxels after quantization
            if batch_data['coors'].numel() == 0 or batch_data['feats'].numel() == 0:
                print(">> Skipping empty batch (0 voxels after quantization).")
                continue

            # Forward pass & decode
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

            # Dump results for evaluation
            for i in range(cfgs.batch_size):
                data_idx = batch_idx * cfgs.batch_size + i
                if data_idx >= len(test_dataset):
                    continue
                preds = grasp_preds[i].detach().cpu().numpy()

                gg = GraspGroup(preds)

                # Optional collision detection
                if cfgs.collision_thresh is not None and cfgs.collision_thresh > 0:
                    cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                    collision_mask = mfcdetector.detect(
                        gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh
                    )
                    gg = gg[~collision_mask]

                # Save grasps
                save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
                gg.save_npy(save_path)

            if (batch_idx + 1) % batch_interval == 0:
                toc = time.time()
                print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
                tic = time.time()

            # Early stop for smoke tests
            if cfgs.max_batches > 0 and (batch_idx + 1) >= cfgs.max_batches:
                print(f">> Stopping early after {cfgs.max_batches} batch(es) (as requested).")
                break


def evaluate(dump_dir):
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test_seen')
    res, ap = ge.eval_seen(dump_folder=dump_dir, proc=6)
    save_dir = os.path.join(cfgs.dump_dir, f'ap_{cfgs.camera}.npy')
    np.save(save_dir, res)
    print(f"Saved AP metrics to: {save_dir}")


if __name__ == '__main__':
    if cfgs.infer:
        inference()
    if cfgs.eval:
        evaluate(cfgs.dump_dir)
