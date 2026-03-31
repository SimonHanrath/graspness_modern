#!/usr/bin/env python3
"""
Compare top-N grasps with and without stable score reweighting.
Shows side-by-side visualization with stable scores for each grasp.

Usage:
    python model_analysis/stable_score/compare_reweighting.py \
        --checkpoint_path logs/gsnet_resunet_vanilla_stable_score_fine_tuned1/gsnet_resunet_epoch05.tar \
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

# Configure matplotlib for thesis (A4, 14cm text width, 12pt base font, Computer Modern)
# Text width: 14cm = 5.51 inches
THESIS_TEXTWIDTH_INCHES = 14 / 2.54  # ~5.51 inches

plt.rcParams.update({
    # Figure size based on thesis text width
    'figure.figsize': (THESIS_TEXTWIDTH_INCHES, 2.5),
    'figure.constrained_layout.use': True,
    
    # Font settings - Computer Modern (LaTeX default)
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',  # Computer Modern math
    
    # Font sizes - scaled for 12pt thesis body text
    'font.size': 10,           # Base font
    'axes.labelsize': 10,      # Axis labels
    'axes.titlesize': 10,      # Subplot titles
    'xtick.labelsize': 8,      # Tick labels
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    
    # Clean styling
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    
    # Disable LaTeX rendering (not available in container)
    'text.usetex': False,
})

# Color palette - thesis-friendly muted colors
COLORS = {
    'original': '#E07A5F',      # Terra cotta (yellow-ish/orange)
    'reweight': '#2E86AB',      # Steel blue
}

from graspnetAPI.graspnet_eval import GraspGroup
from models.graspnet import GraspNet
from dataset.graspnet_dataset import spconv_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


# Constants from graspnet.py
NUM_ANGLE = 12
NUM_DEPTH = 4
GRASP_MAX_WIDTH = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description='Compare grasps with/without stable reweighting')
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
    return parser.parse_args()


def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):
    """Convert viewpoint parameters to rotation matrices."""
    axis_x = batch_towards
    ones = torch.ones(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    zeros = torch.zeros(axis_x.shape[0], dtype=axis_x.dtype, device=axis_x.device)
    axis_y = torch.stack([-axis_x[:, 1], axis_x[:, 0], zeros], dim=-1)
    mask_y = (torch.norm(axis_y, dim=-1) == 0)
    axis_y[mask_y, 0] = 1
    axis_x = axis_x / torch.norm(axis_x, dim=-1, keepdim=True)
    axis_y = axis_y / torch.norm(axis_y, dim=-1, keepdim=True)
    axis_z = torch.cross(axis_x, axis_y)
    sin = torch.sin(batch_angle)
    cos = torch.cos(batch_angle)
    R1 = torch.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], dim=-1).view(-1, 3, 3)
    R2 = torch.stack([axis_x, axis_y, axis_z], dim=-1)
    matrix = torch.matmul(R2, R1)
    return matrix


def decode_grasps_with_stable_info(end_points, use_stable_score=False):
    """
    Decode grasp predictions with stable score extraction.
    
    Returns:
        grasp_preds: List of grasp predictions [M, 17]
        stable_scores: List of stable scores for each grasp [M]
        raw_scores: List of raw (non-reweighted) scores [M]
    """
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    stable_scores_list = []
    raw_scores_list = []
    
    has_stable = 'grasp_stable_pred' in end_points
    
    for i in range(batch_size):
        grasp_center = end_points['xyz_graspable'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        M = grasp_score.shape[0]
        grasp_score = grasp_score.view(M, NUM_ANGLE * NUM_DEPTH)
        
        if has_stable:
            stable_score = end_points['grasp_stable_pred'][i].float()
            stable_expanded = stable_score.unsqueeze(-1).expand(-1, -1, NUM_DEPTH)
            stable_expanded = stable_expanded.reshape(M, NUM_ANGLE * NUM_DEPTH)
            
            if use_stable_score:
                # Reweight: grasp_score * (1 - stable_score)
                grasp_score_reweighted = grasp_score * (1.0 - stable_expanded)
                grasp_score_final, grasp_score_inds = torch.max(grasp_score_reweighted, -1)
            else:
                grasp_score_final, grasp_score_inds = torch.max(grasp_score, -1)
            
            # Get the stable score for the selected angle
            angle_idx = grasp_score_inds // NUM_DEPTH
            selected_stable = torch.gather(stable_score, 1, angle_idx.unsqueeze(-1)).squeeze(-1)
            stable_scores_list.append(selected_stable)
            
            # Get raw (non-reweighted) score for the selected grasp
            raw_score = torch.gather(grasp_score, 1, grasp_score_inds.unsqueeze(-1)).squeeze(-1)
            raw_scores_list.append(raw_score)
        else:
            grasp_score_final, grasp_score_inds = torch.max(grasp_score, -1)
            stable_scores_list.append(torch.zeros(M, device=grasp_score.device))
            raw_scores_list.append(grasp_score_final)
        
        grasp_score_final = grasp_score_final.view(-1, 1)
        grasp_angle = (grasp_score_inds // NUM_DEPTH).float() * (torch.pi / 12)
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1).float() * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M, NUM_ANGLE * NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)
        
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M, 9)
        
        grasp_height = 0.02 * torch.ones_like(grasp_score_final)
        obj_ids = -1 * torch.ones_like(grasp_score_final)
        
        grasp_preds.append(torch.cat([
            grasp_score_final, grasp_width, grasp_height, grasp_depth,
            grasp_rot, grasp_center, obj_ids
        ], dim=-1))
    
    return grasp_preds, stable_scores_list, raw_scores_list


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
        'rgb_image': rgb,  # Full RGB image for visualization
        'intrinsic': intrinsic.astype(np.float32),  # Camera intrinsics
    }
    
    # Load COG data for each object in the scene
    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    poses = meta['poses']  # (3, 4, num_objects)
    
    cogs_camera = []  # COGs in camera frame
    for i, obj_idx in enumerate(obj_idxs):
        obj_id_str = str(obj_idx - 1).zfill(3)
        stable_file = os.path.join(root, 'stable_labels', f'{obj_id_str}_stable.npz')
        mesh_path = os.path.join(root, 'models', obj_id_str, 'nontextured.ply')
        
        cog_obj = None
        # Try to load from stable labels first (precomputed)
        if os.path.exists(stable_file):
            try:
                data = np.load(stable_file)
                if 'cog' in data:
                    cog_obj = data['cog']
            except:
                pass
        
        # Fallback: compute from mesh centroid
        if cog_obj is None and os.path.exists(mesh_path):
            try:
                import trimesh
                mesh = trimesh.load(mesh_path)
                cog_obj = mesh.centroid
            except:
                pass
        
        if cog_obj is not None:
            # Transform COG from object frame to camera frame
            pose = poses[:, :, i]  # (3, 4)
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
        enable_stable_score=True  # Always enable to get stable scores
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
    # vs new format: separate conv_stable layer (conv_swad=96, conv_stable=12)
    if 'swad.conv_swad.weight' in state_dict:
        old_weight = state_dict['swad.conv_swad.weight']
        old_bias = state_dict['swad.conv_swad.bias']
        if old_weight.shape[0] == 108:
            print("Converting old checkpoint (bundled 108 outputs) to new format (96 + 12 separate)...")
            # Split: first 96 for scores/widths, last 12 for stable
            state_dict['swad.conv_swad.weight'] = old_weight[:96]
            state_dict['swad.conv_swad.bias'] = old_bias[:96]
            state_dict['swad.conv_stable.weight'] = old_weight[96:]
            state_dict['swad.conv_stable.bias'] = old_bias[96:]
            print("  Converted swad.conv_swad: [108, 256, 1] -> [96, 256, 1]")
            print("  Created swad.conv_stable: [12, 256, 1]")
    
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


def process_grasps(end_points, batch_data, data_input, args, use_reweight):
    """Process grasps with optional collision detection and NMS."""
    grasp_preds, stable_scores_list, raw_scores_list = decode_grasps_with_stable_info(
        end_points, use_stable_score=use_reweight
    )
    
    preds = grasp_preds[0].detach().cpu().numpy()
    stable_scores = stable_scores_list[0].detach().cpu().numpy()
    raw_scores = raw_scores_list[0].detach().cpu().numpy()
    
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
        stable_scores = stable_scores[~collision_mask]
        raw_scores = raw_scores[~collision_mask]
    
    # Store stable scores in object_ids for NMS preservation
    gg.object_ids[:] = (stable_scores * 10000).astype(np.int32)
    
    # NMS
    gg = gg.nms()
    stable_scores = gg.object_ids / 10000.0
    
    # Sort by score
    gg = gg.sort_by_score()
    stable_scores = gg.object_ids / 10000.0
    
    # Get top N
    num_grasps = min(args.num_grasps, len(gg))
    gg = gg[:num_grasps]
    stable_scores = stable_scores[:num_grasps]
    
    return gg, stable_scores


def project_to_image(points_3d, intrinsic):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in camera frame
        intrinsic: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates (u, v)
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    # Avoid division by zero
    z = np.clip(z, 1e-6, None)
    
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return np.stack([u, v], axis=-1)


def get_gripper_outline_points(center, R, width, depth=0.02):
    """
    Get key points defining the gripper outline for 2D visualization.
    Returns dict with separate line segments for gripper parts.
    """
    # Gripper dimensions
    finger_length = depth + 0.02  # depth + base
    tail_length = 0.04
    half_width = width / 2
    
    # Define separate segments in gripper frame
    # Tail: from tail end to base center
    tail = np.array([
        [-tail_length - 0.02, 0, 0],  # Tail end
        [-0.02, 0, 0],                 # Base center
    ])
    
    # Base bar: horizontal bar connecting the two fingers
    base = np.array([
        [-0.02, -half_width, 0],  # Base left
        [-0.02, half_width, 0],   # Base right
    ])
    
    # Left finger: from base to tip
    left_finger = np.array([
        [-0.02, -half_width, 0],               # Finger base
        [finger_length - 0.02, -half_width, 0], # Finger tip
    ])
    
    # Right finger: from base to tip
    right_finger = np.array([
        [-0.02, half_width, 0],                # Finger base
        [finger_length - 0.02, half_width, 0], # Finger tip
    ])
    
    # Transform all segments to world frame
    segments = {
        'tail': (R @ tail.T).T + center,
        'base': (R @ base.T).T + center,
        'left_finger': (R @ left_finger.T).T + center,
        'right_finger': (R @ right_finger.T).T + center,
    }
    
    return segments


def plot_grasps_on_image(ax, rgb_image, intrinsic, gg, stable_scores, title, score_label='Score', cogs=None, grasp_color=None, show_image=True, legend_handles=None):
    """
    Plot grasps overlaid on RGB image with legend.
    
    Args:
        ax: matplotlib axis
        rgb_image: (H, W, 3) RGB image
        intrinsic: (3, 3) camera intrinsic matrix
        gg: GraspGroup with grasps
        stable_scores: array of stable scores
        title: plot title
        score_label: label for score in legend
        cogs: list of {'obj_idx': int, 'cog': np.array} for COG visualization
        grasp_color: fixed color for all grasps (RGB tuple or color name), None uses tab10 colormap
        show_image: whether to display the RGB image (set False when overlaying multiple grasp sets)
        legend_handles: existing legend handles to append to (for combined legends)
    
    Returns:
        legend_handles: list of legend handles for all grasps
    """
    # Display image
    if show_image:
        ax.imshow(rgb_image)
    
    # Draw COGs first (so grasps are drawn on top)
    if cogs is not None and len(cogs) > 0 and show_image:
        cog_colors = ['cyan', 'magenta', 'lime', 'orange', 'pink', 'white', 'red']
        for i, cog_data in enumerate(cogs):
            cog_3d = cog_data['cog'].reshape(1, 3)
            cog_2d = project_to_image(cog_3d, intrinsic)[0]
            
            # Check bounds
            h, w = rgb_image.shape[:2]
            if 0 <= cog_2d[0] < w and 0 <= cog_2d[1] < h:
                color = cog_colors[i % len(cog_colors)]
                ax.plot(cog_2d[0], cog_2d[1], 'o',
                        color=color, markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    # Colors for top grasps
    if grasp_color is None:
        grasp_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if legend_handles is None:
        legend_handles = []
    
    for idx in range(len(gg)):
        grasp = gg[idx]
        score = grasp.score
        width = grasp.width
        depth = grasp.depth
        center = grasp.translation
        R = grasp.rotation_matrix
        stable = stable_scores[idx]
        
        # Get gripper outline segments in 3D
        segments = get_gripper_outline_points(center, R, width, depth)
        
        if grasp_color is not None:
            color = grasp_color
        else:
            color = grasp_colors[idx % 10][:3]
        
        # Draw each segment separately
        for seg_name, seg_3d in segments.items():
            seg_2d = project_to_image(seg_3d, intrinsic)
            
            # Check if points are within image bounds
            h, w = rgb_image.shape[:2]
            valid = (seg_2d[:, 0] >= 0) & (seg_2d[:, 0] < w) & \
                    (seg_2d[:, 1] >= 0) & (seg_2d[:, 1] < h)
            
            if valid.all():
                ax.plot(seg_2d[:, 0], seg_2d[:, 1], 
                        color=color, linewidth=2.5, solid_capstyle='round')
        
        # Draw grasp center
        center_2d = project_to_image(center.reshape(1, 3), intrinsic)[0]
        ax.plot(center_2d[0], center_2d[1], 'x', 
                color=color, markersize=8, markeredgewidth=2)
        
        # Add rank number near the grasp
        ax.text(center_2d[0] + 10, center_2d[1] - 10, str(idx + 1),
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
        
        # Add to legend
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=2.5,
                          label=f'{idx+1}: {score_label}={score:.3f}, Stable={stable:.3f}')
        )
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    return legend_handles


def main():
    args = parse_args()
    
    if args.dump_dir is None:
        args.dump_dir = os.path.join(SCRIPT_DIR, 'comparison_output')
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
    
    # Run inference (once - we'll decode twice)
    print("Running inference...")
    tic = time.time()
    end_points, batch_data = run_inference(net, data_input, device, args)
    print(f"   Inference time: {time.time() - tic:.2f}s")
    
    # Process grasps without reweighting
    print("Processing grasps without stable reweighting...")
    gg_no_reweight, stable_no_reweight = process_grasps(
        end_points, batch_data, data_input, args, use_reweight=False
    )
    print(f"   Top {len(gg_no_reweight)} grasps selected")
    
    # Process grasps with reweighting
    print("Processing grasps with stable reweighting...")
    gg_reweight, stable_reweight = process_grasps(
        end_points, batch_data, data_input, args, use_reweight=True
    )
    print(f"   Top {len(gg_reweight)} grasps selected")
    
    # Get RGB image and intrinsic for visualization
    rgb_image = data_input['rgb_image']
    intrinsic = data_input['intrinsic']
    cogs = data_input.get('cogs', None)
    
    # Create single figure with both grasp sets overlaid
    # Use thesis-width figure with aspect ratio matching the RGB image
    img_aspect = rgb_image.shape[1] / rgb_image.shape[0]  # width / height
    fig_width = THESIS_TEXTWIDTH_INCHES
    fig_height = fig_width / img_aspect
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), constrained_layout=True)
    
    # Use thesis color palette
    original_color = COLORS['original']    # Terra cotta
    reweighted_color = COLORS['reweight']  # Steel blue
    
    # Plot original grasps (yellow) - this also draws the image and COGs
    legend_handles = plot_grasps_on_image(
        ax, rgb_image, intrinsic, gg_no_reweight, stable_no_reweight,
        '',  # Title will be set after
        score_label='Original', cogs=cogs,
        grasp_color=original_color, show_image=True, legend_handles=None
    )
    
    # Plot reweighted grasps (blue) - overlay on same image
    legend_handles = plot_grasps_on_image(
        ax, rgb_image, intrinsic, gg_reweight, stable_reweight,
        '',  # Title will be set after
        score_label='Reweighted', cogs=None,
        grasp_color=reweighted_color, show_image=False, legend_handles=legend_handles
    )
    
    # Add combined legend with category headers and individual grasp details
    # Create section headers
    original_header = mlines.Line2D([], [], color=original_color, linewidth=3, marker='s', markersize=8,
                                    label=f'--- Original ---')
    reweighted_header = mlines.Line2D([], [], color=reweighted_color, linewidth=3, marker='s', markersize=8,
                                      label=f'--- Reweighted ---')
    
    # Build individual grasp handles
    original_handles = []
    for idx in range(len(gg_no_reweight)):
        stable = stable_no_reweight[idx]
        original_handles.append(
            mlines.Line2D([], [], color=original_color, linewidth=2.5,
                          label=f'  {idx+1}: Stable={stable:.3f}')
        )
    
    reweighted_handles = []
    for idx in range(len(gg_reweight)):
        stable = stable_reweight[idx]
        reweighted_handles.append(
            mlines.Line2D([], [], color=reweighted_color, linewidth=2.5,
                          label=f'  {idx+1}: Stable={stable:.3f}')
        )
    
    # Combine: original header + original grasps + reweighted header + reweighted grasps
    all_handles = [original_header] + original_handles + [reweighted_header] + reweighted_handles
    ax.legend(handles=all_handles, loc='upper right', fontsize=8,
              framealpha=0.9, title='Grasps Comparison', title_fontsize=10,
              bbox_to_anchor=(-0.02, 1), borderaxespad=0)
    
    # Save figure in multiple formats for thesis use
    output_base = os.path.join(args.dump_dir, 
                               f'compare_scene{args.scene}_frame{args.index}')
    
    # High-quality PDF for LaTeX inclusion
    fig.savefig(f'{output_base}.pdf', bbox_inches='tight', dpi=300, facecolor='white')
    
    # PNG for quick preview
    fig.savefig(f'{output_base}.png', bbox_inches='tight', dpi=300, facecolor='white')
    
    plt.close()
    
    print(f"\nVisualization saved to: {output_base}.pdf and {output_base}.png")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Rank':<6} {'No Reweight Score':<18} {'Stable':<10} {'Reweight Score':<18} {'Stable':<10}")
    print("-"*60)
    n = max(len(gg_no_reweight), len(gg_reweight))
    for i in range(n):
        row = f"{i+1:<6}"
        if i < len(gg_no_reweight):
            row += f"{gg_no_reweight[i].score:<18.4f}{stable_no_reweight[i]:<10.4f}"
        else:
            row += " "*28
        if i < len(gg_reweight):
            row += f"{gg_reweight[i].score:<18.4f}{stable_reweight[i]:<10.4f}"
        print(row)


if __name__ == '__main__':
    main()
