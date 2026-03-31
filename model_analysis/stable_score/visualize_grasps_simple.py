#!/usr/bin/env python3
"""
Vibe coded !!!!
Simple grasp visualization script.
Visualizes top-N grasps for a model on a scene, overlaid on the RGB image.
Works with both vanilla models and models with stable score heads.

Usage:
    python experiments/stable_score/visualize_grasps_simple.py \
        --checkpoint_path logs/cluster_100scenes_13epochs_realsense/gsnet_dev_epoch10.tar \
        --scene 0100 --index 0000 --num_grasps 10
"""

import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Setup paths - go up two levels to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, ROOT_DIR)

from graspnetAPI.graspnet_eval import GraspGroup
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize grasps for a model')
    parser.add_argument('--dataset_root', default='/datasets/graspnet',
                        help='Path to GraspNet dataset root')
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dump_dir', default=None,
                        help='Directory to save outputs (default: same folder as script)')
    parser.add_argument('--seed_feat_dim', default=512, type=int,
                        help='Point-wise feature dimension')
    parser.add_argument('--camera', default='realsense', choices=['realsense', 'kinect'],
                        help='Camera type')
    parser.add_argument('--num_point', type=int, default=15000,
                        help='Number of points to sample')
    parser.add_argument('--voxel_size', type=float, default=0.005,
                        help='Voxel size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='Collision threshold (set < 0 to disable)')
    parser.add_argument('--voxel_size_cd', type=float, default=0.01,
                        help='Voxel size for collision detection')
    parser.add_argument('--scene', type=str, default='0100',
                        help='Scene number (e.g., 0100)')
    parser.add_argument('--index', type=str, default='0000',
                        help='Frame index (e.g., 0000)')
    parser.add_argument('--backbone', type=str, default='resunet',
                        choices=['transformer', 'transformer_pretrained', 'pointnet2', 'resunet'],
                        help='Backbone architecture')
    parser.add_argument('--num_grasps', type=int, default=10,
                        help='Number of top grasps to visualize')
    parser.add_argument('--enable_stable_score', action='store_true', default=False,
                        help='Enable stable score head (for models trained with stable score)')
    parser.add_argument('--show_cog', action='store_true', default=False,
                        help='Show object centers of gravity')
    return parser.parse_args()


def load_scene_data(args):
    """Load and preprocess scene data."""
    root = args.dataset_root
    camera_type = args.camera
    scene_id = 'scene_' + args.scene
    index = args.index
    
    use_rgb = (args.backbone == 'transformer_pretrained')
    
    depth_path = os.path.join(root, 'scenes', scene_id, camera_type, 'depth', index + '.png')
    rgb_path = os.path.join(root, 'scenes', scene_id, camera_type, 'rgb', index + '.png')
    seg_path = os.path.join(root, 'scenes', scene_id, camera_type, 'label', index + '.png')
    meta_path = os.path.join(root, 'scenes', scene_id, camera_type, 'meta', index + '.mat')
    
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    
    depth = np.array(Image.open(depth_path))
    rgb = np.array(Image.open(rgb_path))
    seg = np.array(Image.open(seg_path))
    meta = scio.loadmat(meta_path)
    
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1],
                        intrinsic[0][2], intrinsic[1][2], factor_depth)
    
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'scenes', scene_id, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(index)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)
    
    cloud_masked = cloud[mask]
    colors_masked = rgb.reshape(-1, 3)[mask.flatten()]
    
    if len(cloud_masked) >= args.num_point:
        idxs = np.random.choice(len(cloud_masked), args.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), args.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    colors_sampled = colors_masked[idxs]
    
    offset = -cloud_sampled.min(axis=0)
    cloud_sampled = cloud_sampled + offset
    
    if use_rgb:
        feats = colors_sampled.astype(np.float32) / 255.0
    else:
        feats = np.ones_like(cloud_sampled).astype(np.float32)
    
    ret_dict = {
        'point_clouds': cloud_sampled.astype(np.float32),
        'coors': cloud_sampled.astype(np.float32) / args.voxel_size,
        'feats': feats,
        'cloud_offset': offset.astype(np.float32),
        'cloud_colors': colors_sampled.astype(np.float32) / 255.0,
        'rgb_image': rgb,
        'intrinsic': intrinsic.astype(np.float32),
    }
    
    # Load COG data if requested
    if args.show_cog:
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        
        cogs_camera = []
        for i, obj_idx in enumerate(obj_idxs):
            obj_id_str = str(obj_idx - 1).zfill(3)
            stable_file = os.path.join(root, 'stable_labels', f'{obj_id_str}_stable.npz')
            mesh_path = os.path.join(root, 'models', obj_id_str, 'nontextured.ply')
            
            cog_obj = None
            if os.path.exists(stable_file):
                try:
                    data = np.load(stable_file)
                    if 'cog' in data:
                        cog_obj = data['cog']
                except Exception:
                    pass
            
            if cog_obj is None and os.path.exists(mesh_path):
                try:
                    import trimesh
                    mesh = trimesh.load(mesh_path)
                    cog_obj = mesh.centroid
                except Exception:
                    pass
            
            if cog_obj is not None:
                pose = poses[:, :, i]
                cog_cam = pose[:3, :3] @ cog_obj + pose[:3, 3]
                cogs_camera.append({
                    'obj_idx': obj_idx,
                    'cog': cog_cam.astype(np.float32)
                })
        
        ret_dict['cogs'] = cogs_camera
        print(f"   Loaded COGs for {len(cogs_camera)} objects")
    
    return ret_dict


def load_model(args, device):
    """Load the model from checkpoint."""
    print(f"Loading model from: {args.checkpoint_path}")
    
    net = GraspNet(
        seed_feat_dim=args.seed_feat_dim,
        is_training=False,
        backbone=args.backbone,
        enable_stable_score=args.enable_stable_score
    )
    net.to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle module. prefix from DDP
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    
    # Handle old checkpoint format: bundled stable scores in conv_swad (108 outputs)
    if 'swad.conv_swad.weight' in state_dict and args.enable_stable_score:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108:
            print("Converting old checkpoint (bundled 108 outputs) to new format (96 + 12 separate)...")
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]
    
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    
    print(f"-> loaded checkpoint (epoch: {checkpoint.get('epoch', 'N/A')})")
    
    return net


def run_inference(net, data_input, device, args):
    """Run model inference and return end_points."""
    batch_data = spconv_collate_fn([data_input])
    
    for key in batch_data:
        if isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)
        elif hasattr(batch_data[key], 'to'):
            batch_data[key] = batch_data[key].to(device)
    
    with torch.inference_mode():
        end_points = net(batch_data)
    
    return end_points, batch_data


def process_grasps(end_points, batch_data, data_input, args):
    """Process grasps with collision detection and NMS."""
    grasp_preds = pred_decode(end_points, use_stable_score=False)
    
    preds = grasp_preds[0].detach().cpu().numpy()
    
    # Transform back to camera coordinates
    offset = batch_data['cloud_offset'][0].cpu().numpy()
    preds[:, 13:16] = preds[:, 13:16] - offset
    
    gg = GraspGroup(preds)
    
    # Collision detection
    if args.collision_thresh > 0:
        cloud = data_input['point_clouds'] - data_input['cloud_offset']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=args.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
        gg = gg[~collision_mask]
        print(f"   Collision filtering: {collision_mask.sum()} / {len(collision_mask)} grasps removed")
    
    # NMS
    gg = gg.nms()
    
    # Sort by score
    gg = gg.sort_by_score()
    
    # Get top N
    num_grasps = min(args.num_grasps, len(gg))
    gg = gg[:num_grasps]
    
    return gg


def project_to_image(points_3d, intrinsic):
    """Project 3D points to 2D image coordinates."""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z = np.clip(z, 1e-6, None)
    
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return np.stack([u, v], axis=-1)


def get_gripper_outline_points(center, R, width, depth=0.02):
    """Get key points defining the gripper outline for 2D visualization."""
    finger_length = depth + 0.02
    tail_length = 0.04
    half_width = width / 2
    
    tail = np.array([
        [-tail_length - 0.02, 0, 0],
        [-0.02, 0, 0],
    ])
    
    base = np.array([
        [-0.02, -half_width, 0],
        [-0.02, half_width, 0],
    ])
    
    left_finger = np.array([
        [-0.02, -half_width, 0],
        [finger_length - 0.02, -half_width, 0],
    ])
    
    right_finger = np.array([
        [-0.02, half_width, 0],
        [finger_length - 0.02, half_width, 0],
    ])
    
    segments = {
        'tail': (R @ tail.T).T + center,
        'base': (R @ base.T).T + center,
        'left_finger': (R @ left_finger.T).T + center,
        'right_finger': (R @ right_finger.T).T + center,
    }
    
    return segments


def plot_grasps_on_image(ax, rgb_image, intrinsic, gg, title, cogs=None):
    """Plot grasps overlaid on RGB image with legend."""
    ax.imshow(rgb_image)
    
    # Draw COGs first
    if cogs is not None and len(cogs) > 0:
        cog_colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'white', 'red']
        for i, cog_data in enumerate(cogs):
            cog_3d = cog_data['cog'].reshape(1, 3)
            cog_2d = project_to_image(cog_3d, intrinsic)[0]
            
            h, w = rgb_image.shape[:2]
            if 0 <= cog_2d[0] < w and 0 <= cog_2d[1] < h:
                color = cog_colors[i % len(cog_colors)]
                ax.plot(cog_2d[0], cog_2d[1], 'o',
                        color=color, markersize=15, markeredgecolor='black', markeredgewidth=2)
                ax.text(cog_2d[0] + 12, cog_2d[1], f'obj{cog_data["obj_idx"]}',
                        fontsize=7, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))
    
    # Colors for grasps
    grasp_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    legend_handles = []
    
    for idx in range(len(gg)):
        grasp = gg[idx]
        score = grasp.score
        width = grasp.width
        depth = grasp.depth
        center = grasp.translation
        R = grasp.rotation_matrix
        
        segments = get_gripper_outline_points(center, R, width, depth)
        
        color = grasp_colors[idx % 10][:3]
        
        # Draw each segment
        for seg_name, seg_3d in segments.items():
            seg_2d = project_to_image(seg_3d, intrinsic)
            
            h, w = rgb_image.shape[:2]
            valid = (seg_2d[:, 0] >= 0) & (seg_2d[:, 0] < w) & \
                    (seg_2d[:, 1] >= 0) & (seg_2d[:, 1] < h)
            
            if valid.all():
                ax.plot(seg_2d[:, 0], seg_2d[:, 1], 
                        color=color, linewidth=2.5, solid_capstyle='round')
        
        # Draw finger tips
        left_tip_2d = project_to_image(segments['left_finger'][1:2], intrinsic)[0]
        right_tip_2d = project_to_image(segments['right_finger'][1:2], intrinsic)[0]
        ax.plot(left_tip_2d[0], left_tip_2d[1], 'o', 
                color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
        ax.plot(right_tip_2d[0], right_tip_2d[1], 'o', 
                color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        # Draw grasp center
        center_2d = project_to_image(center.reshape(1, 3), intrinsic)[0]
        ax.plot(center_2d[0], center_2d[1], 'x', 
                color=color, markersize=8, markeredgewidth=2)
        
        # Add rank number
        ax.text(center_2d[0] + 10, center_2d[1] - 10, str(idx + 1),
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
        
        # Add to legend
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=2.5,
                          label=f'{idx+1}: Score={score:.3f}')
        )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              framealpha=0.9, title='Grasps', title_fontsize=10)


def main():
    args = parse_args()
    
    if args.dump_dir is None:
        args.dump_dir = os.path.join(SCRIPT_DIR, 'visualization_output')
    os.makedirs(args.dump_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    net = load_model(args, device)
    
    # Load scene data
    print(f"Loading scene {args.scene}, frame {args.index}...")
    data_input = load_scene_data(args)
    
    # Run inference
    print("Running inference...")
    tic = time.time()
    end_points, batch_data = run_inference(net, data_input, device, args)
    print(f"   Inference time: {time.time() - tic:.2f}s")
    
    # Process grasps
    print("Processing grasps...")
    gg = process_grasps(end_points, batch_data, data_input, args)
    print(f"   Top {len(gg)} grasps selected")
    
    # Get data for visualization
    rgb_image = data_input['rgb_image']
    intrinsic = data_input['intrinsic']
    cogs = data_input.get('cogs', None)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    model_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    title = f'Top {len(gg)} Grasps\nModel: {model_name}\nScene: {args.scene}, Frame: {args.index}'
    
    plot_grasps_on_image(ax, rgb_image, intrinsic, gg, title, cogs=cogs)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(args.dump_dir, 
                               f'grasps_scene{args.scene}_frame{args.index}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("GRASP SUMMARY")
    print("="*50)
    print(f"{'Rank':<6} {'Score':<12}")
    print("-"*50)
    for i in range(len(gg)):
        print(f"{i+1:<6} {gg[i].score:<12.4f}")


if __name__ == '__main__':
    main()
